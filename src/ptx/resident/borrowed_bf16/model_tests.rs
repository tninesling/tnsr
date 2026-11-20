//! Actual decoder state comparison, not archive first-update qualification/timing.
use super::*;
use crate::{
    ptx::mixed::{
        MixedOptions,
        test_decoder::{self, Config, Model},
    },
    tensor::TensorExpr,
    tile::{
        BatchedAttentionBf16TilePolicy, ColumnReductionPolicy, MatMulPrecision,
        MediumBf16TilePolicy, SplitKBf16TilePolicy, VirtualTensor,
    },
};
use serde_json::json;
use std::path::PathBuf;

struct Fixture {
    model: Model,
    graph: TensorGraph<f32, WithGrad>,
    loss: NodeIndex,
}
fn config(large_batch: bool) -> Config {
    Config {
        width: if large_batch { 512 } else { 1024 },
        batch: if large_batch { 8 } else { 1 },
        sequence: 512,
        heads: if large_batch { 8 } else { 16 },
        hidden: if large_batch { 1408 } else { 2752 },
        layers: 4,
        vocab: 4096,
        epsilon: 1e-5,
        learning_rate: 1e-4,
        max_grad_norm: 1.,
    }
}
fn fixture(config: Config) -> Result<Fixture> {
    let model = test_decoder::synthetic(config);
    // External scalar only supplies microbatch weighting and deliberate NaN
    // rejection, as in earlier actual-decoder state fixtures. It is not part of
    // any archived benchmark or performance/numerical qualification claim.
    let expression = model.loss()? * TensorExpr::input("model_loss_scale", vec![]);
    let mut graph = TensorGraph::new();
    let loss = graph.add_expr(&expression, false);
    let graph = graph.with_gradients_with_options(
        loss,
        crate::graph::GradientOptions {
            compact_softmax: true,
        },
    );
    Ok(Fixture { model, graph, loss })
}
fn options(large_batch: bool, direct: bool) -> MixedOptions {
    MixedOptions {
        physical_order_casts: true,
        shared_memory_reuse: true,
        incremental_allocation_reuse: true,
        direct_master_bindings: direct,
        attention_softmax_fusion: true,
        attention_softmax_backward_fusion: true,
        narrow_bf16_tile: true,
        transposed_bf16_outputs: true,
        medium_bf16_tile_policy: MediumBf16TilePolicy::SquareThenTall,
        qkv_grad_pack_fusion: true,
        swiglu_pair_fusion: true,
        column_reduction_policy: ColumnReductionPolicy::Four,
        batched_attention_bf16_tile_policy: BatchedAttentionBf16TilePolicy::TallScoresByGrid,
        layer_norm_pair_fusion: true,
        split_k_bf16_tile_policy: SplitKBf16TilePolicy::Square64,
        indexed_loss_cooperative_ordered_backward: true,
        indexed_loss_cooperative_ordered_forward: true,
        contiguous_bias_reshape_aliases: true,
        plain_bf16_output: large_batch,
        fused_nt128_bias: large_batch,
        online_attention_pair_numerical_screen: large_batch,
        ..MixedOptions::default()
    }
}
fn setup(
    f: &Fixture,
    borrowed: bool,
    small_norm_batches: bool,
    vector: bool,
) -> Result<(ResidentAdam, MixedExecutor)> {
    let parameters = f
        .model
        .parameters
        .iter()
        .map(|(_, e)| (e.parameter.clone(), 0))
        .collect();
    let groups = vec![AdamGroup::new(f.model.config.learning_rate)];
    let mut owner = if borrowed {
        ResidentAdam::new_with_graph_replay(parameters, groups)?
    } else {
        ResidentAdam::new(parameters, groups)?
    };
    owner.set_update_batch_size(8)?;
    owner.set_experimental_first_accumulation_overwrite(true)?;
    owner.set_experimental_gradient_norm_partials(true)?;
    owner.set_experimental_borrowed_bf16_adam_source(borrowed)?;
    owner.set_experimental_vector_bf16_first_adam(vector)?;
    if borrowed && small_norm_batches {
        owner.set_experimental_small_accumulation_norm_batches(true)?;
    }
    owner.begin_step()?;
    let executor = owner.mixed_executor_with_options(
        &f.graph,
        MatMulPrecision::Tf32x3,
        options(f.model.config.batch == 8, borrowed),
    )?;
    Ok((owner, executor))
}
fn exact(a: &[f32], b: &[f32], context: &str) -> Result<()> {
    ensure!(a.len() == b.len(), "{context}: length mismatch");
    for (i, (a, b)) in a.iter().zip(b).enumerate() {
        ensure!(
            a.to_bits() == b.to_bits(),
            "{context}: first mismatch{i}: {a:?} vs {b:?}"
        );
    }
    Ok(())
}
fn state_equal(a: &ResidentAdam, b: &ResidentAdam, fa: &Fixture, fb: &Fixture) -> Result<()> {
    let ga = a.gradients()?;
    let gb = b.gradients()?;
    for ((name, pa), (other, pb)) in fa.model.parameters.iter().zip(fb.model.parameters.iter()) {
        ensure!(name == other, "state parameter order changed");
        exact(
            &ga[&pa.parameter.id()],
            &gb[&pb.parameter.id()],
            &format!("{name} clipped gradient"),
        )?;
        exact(
            &pa.parameter.values()?,
            &pb.parameter.values()?,
            &format!("{name} weight"),
        )?;
    }
    drop(ga);
    drop(gb);
    let ma = a.moments()?;
    let mb = b.moments()?;
    for ((name, pa), (_, pb)) in fa.model.parameters.iter().zip(fb.model.parameters.iter()) {
        exact(
            &ma[&pa.parameter.id()].first,
            &mb[&pb.parameter.id()].first,
            &format!("{name} first moment"),
        )?;
        exact(
            &ma[&pa.parameter.id()].second,
            &mb[&pb.parameter.id()].second,
            &format!("{name} second moment"),
        )?;
    }
    Ok(())
}
struct Published {
    weights: Vec<Vec<f32>>,
    moments: Vec<AdamMoments>,
}
fn published(owner: &ResidentAdam, f: &Fixture) -> Result<Published> {
    let mut moments = owner.moments()?;
    let ordered = f
        .model
        .parameters
        .iter()
        .map(|(_, p)| {
            moments
                .remove(&p.parameter.id())
                .context("published moment absent")
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Published {
        weights: f
            .model
            .parameters
            .iter()
            .map(|(_, p)| p.parameter.values())
            .collect::<Result<_>>()?,
        moments: ordered,
    })
}
fn unchanged(owner: &ResidentAdam, f: &Fixture, old: &Published) -> Result<()> {
    let current = owner.moments()?;
    for (index, ((name, p), weight)) in f.model.parameters.iter().zip(&old.weights).enumerate() {
        exact(
            &p.parameter.values()?,
            weight,
            &format!("{name} rejected weight"),
        )?;
        let id = p.parameter.id();
        exact(
            &current[&id].first,
            &old.moments[index].first,
            &format!("{name} rejected first moment"),
        )?;
        exact(
            &current[&id].second,
            &old.moments[index].second,
            &format!("{name} rejected second moment"),
        )?;
    }
    Ok(())
}
fn accumulate(
    owner: &mut ResidentAdam,
    executor: &mut MixedExecutor,
    f: &Fixture,
    scale: f32,
    revision: usize,
) -> Result<Vec<f32>> {
    let mut feeds = f.model.config.feeds()?;
    let offset = u32::try_from(revision.checked_mul(31).context("feed revision overflow")?)?;
    let vocab = u32::try_from(f.model.config.vocab)?;
    for indices in feeds.values_mut() {
        for token in indices {
            *token = (*token + offset) % vocab;
        }
    }
    owner.accumulate_mixed(
        executor,
        &f.graph,
        HashMap::from([("model_loss_scale".into(), vec![scale])]),
        feeds,
    )?;
    // Deliberately execute the normal loss readback between accumulation and
    // step_mixed; it must neither stale the borrow nor overwrite gradient sinks.
    executor.read(f.loss)
}
fn prefixes(f: &Fixture, e: &MixedExecutor) -> Result<Vec<serde_json::Value>> {
    let mut rows = Vec::new();
    for name in [
        "layer.3.down.weight",
        "layer.3.out.weight",
        "layer.3.qkv.weight",
    ] {
        let record = f
            .model
            .records
            .iter()
            .find(|r| r.name == name)
            .context("prefix record missing")?;
        let parameter = f.model.parameters.get(name, &record.shape)?;
        let node = f.graph.gradient_metadata().param_to_grad[&parameter.id()];
        let (capacity, zero_offset) = e
            .borrowed_gradient_cached_owner_extent(node, record.count)
            .context("prefix was not admitted/cached")?;
        ensure!(
            capacity > record.count && zero_offset,
            "{name}: actual larger zero-offset prefix not observed"
        );
        rows.push(json!({"name":name,"logical_elements":record.count,"owner_elements":capacity,"zero_offset":zero_offset}));
    }
    Ok(rows)
}
/// Capture this fixture's actual post-run metadata before coverage assertions.
/// A square transpose is explicitly not a contiguous prefix.
fn binding_descriptors(
    f: &Fixture,
    e: &MixedExecutor,
    owner: &ResidentAdam,
) -> Result<Vec<serde_json::Value>> {
    let dispatch = owner.actual_adam_dispatch_for_census()?;
    f.model.parameters.iter().map(|(name, entry)| {
        let record = f.model.records.iter().find(|r| r.name == name).context("descriptor record absent")?;
        let node = f.graph.gradient_metadata().param_to_grad[&entry.parameter.id()];
        let (binding, view, source_shape) = e.device_binding(node)?;
        let actual = view.access.clone().normalize_in_domain(&view.shape)?;
        let canonical = VirtualTensor::identity(view.source, source_shape.clone(), binding.dtype).reshape(view.shape.clone())?;
        let canonical = canonical.access.normalize_in_domain(&view.shape)?;
        let square_transpose = if source_shape == [512,512] && view.shape == [512,512] {
            let transpose = VirtualTensor::identity(view.source, source_shape.clone(), binding.dtype).permute(&[1,0])?;
            actual == transpose.access.normalize_in_domain(&view.shape)?
        } else { false };
        let batched = dispatch.iter().find(|d| d.parameter_id == entry.parameter.id()).context("descriptor dispatch absent")?.batched;
        let admitted = e.borrowed_bf16_gradient(node, record.count).is_some();
        Ok(json!({"name":name,"parameter_elements":record.count,"gradient_node":node.index(),
            "binding_dtype":format!("{:?}",binding.dtype),"binding_elements":binding.len,"pointer":binding.pointer,
            "view_dtype":format!("{:?}",view.dtype),"view_shape":view.shape,"source_shape":source_shape,
            "actual_normalized_map":format!("{actual:?}"),"canonical_normalized_map":format!("{canonical:?}"),
            "whole_contiguous_map":actual==canonical,"exact_512_square_transpose_map":square_transpose,
            "typed_prefix_admitted":admitted,"actual_legacy_dispatch_batched":batched,
            "cached_owner_extent":e.borrowed_gradient_cached_owner_extent(node,record.count)}))
    }).collect()
}
fn prove_matrix_coverage(f: &Fixture, rows: &[serde_json::Value], large_batch: bool) -> Result<()> {
    let mut admitted_elements = 0_usize;
    let mut admitted_count = 0_usize;
    for layer in 0..f.model.config.layers {
        for role in ["down", "gate", "out", "qkv", "up"] {
            let name = format!("layer.{layer}.{role}.weight");
            let row = rows
                .iter()
                .find(|r| r["name"] == name)
                .context("matrix descriptor absent")?;
            ensure!(
                row["binding_dtype"] == "BF16" && row["actual_legacy_dispatch_batched"] == false,
                "{name}: unexpected matrix dtype/dispatch"
            );
            if large_batch && role == "out" {
                // Long K4096 for512square deliberately bypasses physical-output
                // transposition. Preserve this strict production fallback.
                ensure!(
                    row["typed_prefix_admitted"] == false
                        && row["whole_contiguous_map"] == false
                        && row["exact_512_square_transpose_map"] == true,
                    "{name}: expected exact nonidentity square-transpose fallback: {row}"
                );
            } else {
                ensure!(
                    row["typed_prefix_admitted"] == true && row["whole_contiguous_map"] == true,
                    "{name}: expected admitted contiguous matrix: {row}"
                );
                admitted_count += 1;
                admitted_elements += usize::try_from(
                    row["parameter_elements"]
                        .as_u64()
                        .context("matrix elements absent")?,
                )?;
            }
        }
    }
    ensure!(
        admitted_count == if large_batch { 16 } else { 20 },
        "wrong matrix coverage"
    );
    ensure!(
        admitted_elements == if large_batch { 11_796_480 } else { 50_593_792 },
        "wrong matrix element coverage"
    );
    Ok(())
}
fn run_model(large_batch: bool, small_norm_batches: bool) -> Result<()> {
    run_model_with_vector(large_batch, small_norm_batches, false)
}
fn run_model_with_vector(large_batch: bool, small_norm_batches: bool, vector: bool) -> Result<()> {
    let available = CudaContext::new(0).is_ok();
    ensure!(
        available || std::env::var_os("TNSR_REQUIRE_CUDA").is_none(),
        "CUDA required"
    );
    if !available {
        return Ok(());
    }
    let root = PathBuf::from(
        std::env::var_os("TNSR_BORROWED_BF16_MODEL_STATE_DIR")
            .context("set TNSR_BORROWED_BF16_MODEL_STATE_DIR")?,
    );
    std::fs::create_dir_all(&root)?;
    let base = if vector {
        "vector-actual-w1024-s128-b2"
    } else if large_batch {
        "actual-b8"
    } else {
        "actual-55m"
    };
    let stem = if small_norm_batches {
        format!("smallnorm-{base}")
    } else {
        base.to_owned()
    };
    let out = root.join(format!("{stem}.json"));
    ensure!(!out.exists(), "fresh actual model state output required");
    let mut cfg = config(large_batch);
    if vector {
        cfg.sequence = 128;
        cfg.batch = 2;
    }
    let fa = fixture(cfg.clone())?;
    let fb = fixture(cfg)?;
    let (mut a, mut ea) = setup(&fa, false, false, false)?;
    let (mut b, mut eb) = setup(&fb, true, small_norm_batches, vector)?;
    let expected_single = if large_batch { 16 } else { 20 };
    let mut counts = None;
    let mut rows = Vec::new();
    let mut prefix_rows = Vec::new();
    for (interval, micros) in [1_usize, 2, 1, 0].into_iter().enumerate() {
        if interval > 0 {
            a.begin_step()?;
            b.begin_step()?;
        }
        for micro in 0..micros {
            let revision = interval * 3 + micro;
            let la = accumulate(&mut a, &mut ea, &fa, 1. / micros as f32, revision)?;
            let lb = accumulate(&mut b, &mut eb, &fb, 1. / micros as f32, revision)?;
            exact(&la, &lb, "actual model loss before step")?;
            ensure!(
                eb.transfers().parameter_copies == 0,
                "direct update copied masters"
            );
        }
        let sa = a.step(fa.model.config.max_grad_norm)?;
        let sb = b.step_mixed(&eb, fb.model.config.max_grad_norm)?;
        ensure!(
            sa.gradient_norm.to_bits() == sb.gradient_norm.to_bits()
                && sa.gradient_scale.to_bits() == sb.gradient_scale.to_bits(),
            "actual model norm/scale differ"
        );
        state_equal(&a, &b, &fa, &fb)?;
        if small_norm_batches {
            check_small_norm_status(&b, micros)?;
            ensure!(
                sa.kernel_launches == sb.kernel_launches + if micros > 0 { 33 } else { 0 },
                "actual total launch accounting does not include exact accumulation savings"
            );
        }
        if interval == 0 {
            let descriptors = binding_descriptors(&fb, &eb, &b)?;
            let path = root.join(format!("{stem}-gradient-bindings.json"));
            ensure!(!path.exists(), "fresh descriptor output required");
            std::fs::write(
                path,
                serde_json::to_vec_pretty(
                    &json!({"scope":"Actual live fixture binding/map/admission metadata after real accumulation and legacy/borrowed state comparison; before coverage assertions.","parameters":&descriptors}),
                )?,
            )?;
            prove_matrix_coverage(&fb, &descriptors, large_batch)?;
        }
        let expected = if micros == 1 { expected_single } else { 0 };
        ensure!(
            b.borrowed_bf16_adam_status().standalone_bf16_parameters == expected,
            "actual borrowed selection: {:?}",
            b.borrowed_bf16_adam_status()
        );
        if vector {
            let status = b.vector_bf16_first_adam_status();
            ensure!(
                status.enabled
                    && status.submitted_parameters == expected
                    && status.committed_parameters == expected
                    && status.submitted_elements == status.committed_elements,
                "actual vector coverage/status: {status:?}"
            );
            if micros == 1 {
                ensure!(
                    status.committed_elements == 50_593_792 && !status.resources.is_empty(),
                    "actual vector matrix element census changed"
                );
            }
        }
        let current = eb.borrowed_gradient_proof_counts();
        if let Some(first) = counts {
            ensure!(
                current == first,
                "proof normalization/scans repeated across master variants"
            );
        } else {
            ensure!(
                current.0 > 0 && current.1 > 0,
                "positive proof warmup absent"
            );
            counts = Some(current);
        }
        if !large_batch && !vector && interval == 0 {
            prefix_rows = prefixes(&fb, &eb)?;
        }
        if interval >= 1 {
            ensure!(
                eb.info().direct_capture_variants == 2 && eb.info().direct_capture_misses == 2,
                "expected exactly two direct master variants"
            );
        }
        let mut row = json!({"interval":interval,"microbatches":micros,"exact_loss_norm_scale_gradients_weights_both_moments":true,"status":b.borrowed_bf16_adam_status(),"proof_counts":current,"mixed_info":eb.info()});
        if small_norm_batches {
            row["small_accumulation_norm"] =
                serde_json::to_value(b.small_accumulation_norm_batches_status())?;
            row["baseline_complete_step_launches"] = sa.kernel_launches.into();
            row["candidate_complete_step_launches"] = sb.kernel_launches.into();
        }
        if vector {
            row["vector_bf16_first_adam"] =
                serde_json::to_value(b.vector_bf16_first_adam_status())?;
        }
        rows.push(row);
    }
    // Save only the candidate's published state. The F32 comparator has already
    // matched exactly; both invalid attempts must reject without publication.
    let before = published(&b, &fb)?;
    a.begin_step()?;
    b.begin_step()?;
    let la = accumulate(&mut a, &mut ea, &fa, f32::NAN, 20)?;
    let lb = accumulate(&mut b, &mut eb, &fb, f32::NAN, 20)?;
    ensure!(
        la.iter().all(|x| x.is_nan()) && lb.iter().all(|x| x.is_nan()),
        "NaN feed did not reach loss"
    );
    ensure!(
        a.step(1.).is_err() && b.step_mixed(&eb, 1.).is_err(),
        "NaN gradients were not rejected"
    );
    ensure!(
        b.borrowed_bf16_adam_status().standalone_bf16_parameters == 0,
        "failed step launched borrowed update"
    );
    if vector {
        ensure!(
            b.vector_bf16_first_adam_status().submitted_parameters == 0
                && b.vector_bf16_first_adam_status().committed_parameters == 0,
            "nonfinite gradients submitted a vector update"
        );
    }
    unchanged(&a, &fa, &before)?; // Snapshot order is by stable names, not owner-specific IDs.
    unchanged(&b, &fb, &before)?;
    drop(before);
    a.begin_step()?;
    b.begin_step()?;
    let la = accumulate(&mut a, &mut ea, &fa, 1., 21)?;
    let lb = accumulate(&mut b, &mut eb, &fb, 1., 21)?;
    exact(&la, &lb, "recovery loss")?;
    let sa = a.step(1.)?;
    let sb = b.step_mixed(&eb, 1.)?;
    ensure!(
        sa.gradient_norm.to_bits() == sb.gradient_norm.to_bits()
            && sa.gradient_scale.to_bits() == sb.gradient_scale.to_bits(),
        "recovery norm/scale differ"
    );
    state_equal(&a, &b, &fa, &fb)?;
    if small_norm_batches {
        check_small_norm_status(&b, 1)?;
        ensure!(
            sa.kernel_launches == sb.kernel_launches + 33,
            "recovery accumulation launch accounting changed"
        );
    }
    ensure!(
        b.borrowed_bf16_adam_status().standalone_bf16_parameters == expected_single,
        "recovery did not borrow"
    );
    ensure!(
        Some(eb.borrowed_gradient_proof_counts()) == counts,
        "proof cache changed after recovery"
    );
    let mut report = json!({"scope":"Actual synthetic decoder evolving-state test; extra scalar weights microbatches/NaN. No archive first-update qualification or performance claim.","config":fb.model.config,"intervals":rows,"NaN_rejection_recovery":true,"normal_loss_read_before_every_step":true,"all_full_finite_state_bits_exact":true,"three_larger_pooled_prefixes":prefix_rows,"expected_single_borrowed_matrices":expected_single,"proof_counts_after_warmup":counts,"capture_variants":eb.info().direct_capture_variants,"final_borrowed_status":b.borrowed_bf16_adam_status()});
    if small_norm_batches {
        report["small_accumulation_norm"] =
            serde_json::to_value(b.small_accumulation_norm_batches_status())?;
        report["reference_small_norm_enabled"] =
            a.small_accumulation_norm_batches_status().enabled.into();
    }
    if vector {
        report["final_vector_status"] = serde_json::to_value(b.vector_bf16_first_adam_status())?;
        // Finite gradients, deliberately overflowing Adam settings: kernels are
        // actually submitted but staged-state health must prevent publication.
        let before = published(&b, &fb)?;
        a.groups[0].learning_rate = f32::MAX;
        b.groups[0].learning_rate = f32::MAX;
        a.groups[0].weight_decay = f32::MAX;
        b.groups[0].weight_decay = f32::MAX;
        a.begin_step()?;
        b.begin_step()?;
        ensure!(
            b.vector_bf16_first_adam_status().submitted_parameters == 0,
            "begin_step retained old vector status"
        );
        let la = accumulate(&mut a, &mut ea, &fa, 1., 22)?;
        let lb = accumulate(&mut b, &mut eb, &fb, 1., 22)?;
        exact(&la, &lb, "staged-overflow loss")?;
        ensure!(
            a.step(1.).is_err() && b.step_mixed(&eb, 1.).is_err(),
            "overflowing staged updates were published"
        );
        let status = b.vector_bf16_first_adam_status();
        ensure!(
            status.submitted_parameters == 20 && status.committed_parameters == 0,
            "vector submitted/committed rollback accounting: {status:?}"
        );
        report["staged_health_rejection_vector_status"] = serde_json::to_value(status)?;
        unchanged(&a, &fa, &before)?;
        unchanged(&b, &fb, &before)?;
        a.groups[0].learning_rate = fa.model.config.learning_rate;
        b.groups[0].learning_rate = fb.model.config.learning_rate;
        a.groups[0].weight_decay = 0.;
        b.groups[0].weight_decay = 0.;
        // A real second executor run makes the saved completed-run stamp stale.
        // Accumulated F32 gradients remain the old, exact fallback source.
        a.begin_step()?;
        b.begin_step()?;
        ensure!(
            b.vector_bf16_first_adam_status().submitted_parameters == 0
                && b.vector_bf16_first_adam_status().committed_parameters == 0,
            "recovery begin retained rejected vector counters"
        );
        let la = accumulate(&mut a, &mut ea, &fa, 1., 23)?;
        let lb = accumulate(&mut b, &mut eb, &fb, 1., 23)?;
        exact(&la, &lb, "stale-source recovery loss")?;
        eb.run()?;
        let sa = a.step(1.)?;
        let sb = b.step_mixed(&eb, 1.)?;
        ensure!(
            sa.gradient_norm.to_bits() == sb.gradient_norm.to_bits()
                && sa.gradient_scale.to_bits() == sb.gradient_scale.to_bits(),
            "stale source fallback norm changed"
        );
        state_equal(&a, &b, &fa, &fb)?;
        ensure!(
            b.vector_bf16_first_adam_status().submitted_parameters == 0
                && b.vector_bf16_first_adam_status().committed_parameters == 0
                && b.borrowed_bf16_adam_status().standalone_bf16_parameters == 0,
            "stale source rescued by vector selector"
        );
        report["stale_source_full_f32_fallback"] = true.into();
        report["staged_health_rejection_recovery_exact_state"] = true.into();
    }
    std::fs::write(out, serde_json::to_vec_pretty(&report)?)?;
    Ok(())
}
#[test]
fn borrowed_bf16_actual_models_preserve_v27_policy_split() {
    let a = options(true, true);
    let b = options(false, true);
    assert!(a.online_attention_pair_numerical_screen && a.fused_nt128_bias && a.plain_bf16_output);
    assert!(
        !b.online_attention_pair_numerical_screen && !b.fused_nt128_bias && !b.plain_bf16_output
    );
    assert!(a.contiguous_bias_reshape_aliases && b.contiguous_bias_reshape_aliases);
}
#[test]
#[ignore = "manual actual B8 borrowed Adam full state/direct replay/NaN recovery"]
fn borrowed_bf16_adam_actual_b8_model_state() -> Result<()> {
    run_model(true, false)
}
#[test]
#[ignore = "manual actual55M borrowed Adam full state/pool prefixes/direct replay/NaN recovery"]
fn borrowed_bf16_adam_actual_55m_model_state() -> Result<()> {
    run_model(false, false)
}

fn check_small_norm_status(owner: &ResidentAdam, micros: usize) -> Result<()> {
    let s = owner.small_accumulation_norm_batches_status();
    ensure!(s.enabled, "small norm option not enabled");
    if micros == 1 {
        ensure!(
            s.last_accumulation_used
                && s.selected_members == 38
                && s.batched_launches == 5
                && s.standalone_launches == 22
                && s.baseline_launches == 60
                && s.actual_launches == 27
                && s.fallback_reason.is_empty(),
            "actual small norm plan: {s:?}"
        );
    } else {
        ensure!(
            !s.last_accumulation_used
                && s.selected_members == 0
                && s.batched_launches == 0
                && s.actual_launches == if micros == 0 { 0 } else { 60 },
            "multi/empty fallback: {s:?}"
        );
    }
    Ok(())
}
#[test]
#[ignore = "manual actual55M small accumulation norm batches plus borrowedBF16 exact state/recovery"]
fn small_accumulation_norm_actual_55m_borrowed_state() -> Result<()> {
    run_model(false, true)
}
#[test]
#[ignore = "manual actualB8 small accumulation norm batches plus borrowedBF16 exact state/recovery"]
fn small_accumulation_norm_actual_b8_borrowed_state() -> Result<()> {
    run_model(true, true)
}

#[test]
#[ignore = "manual actual W1024/S128/B2 vector owner census/state/replay/stale/health rollback"]
fn vector_bf16_first_adam_actual_w1024_s128_b2_state() -> Result<()> {
    run_model_with_vector(false, true, true)
}

#[test]
#[ignore = "manual live closest-case existing weight-cast/owner census, no shadows or timers"]
fn shadow_live_actual_w1024_s128_b2_weight_cast_owner_census() -> Result<()> {
    use crate::ptx::mixed::shadow_cast_census::{self, RegisteredMaster};
    use std::collections::HashSet;
    let root = PathBuf::from(
        std::env::var_os("TNSR_SHADOW_LIVE_CENSUS_DIR")
            .context("set TNSR_SHADOW_LIVE_CENSUS_DIR to fresh census output")?,
    );
    std::fs::create_dir_all(&root)?;
    let out = root.join("live-weight-cast-census.json");
    ensure!(!out.exists(), "fresh census output required");
    let context = CudaContext::new(0);
    if let Err(error) = context {
        ensure!(
            std::env::var_os("TNSR_REQUIRE_CUDA").is_none(),
            "CUDA required: {error}"
        );
        return Ok(());
    }
    drop(context);
    let mut cfg = config(false);
    cfg.sequence = 128;
    cfg.batch = 2;
    let model = test_decoder::synthetic(cfg);
    let (graph, loss) = model.training_graph_with_options(crate::graph::GradientOptions {
        compact_softmax: true,
    })?;
    ensure!(
        model.parameters.iter().count() == 60 && model.parameter_count() == 54_980_096,
        "closest-case registered parameter census changed"
    );
    let mut owner = ResidentAdam::new_with_graph_replay(
        model
            .parameters
            .iter()
            .map(|(_, p)| (p.parameter.clone(), 0))
            .collect(),
        vec![AdamGroup::new(model.config.learning_rate)],
    )?;
    owner.set_update_batch_size(8)?;
    owner.set_experimental_first_accumulation_overwrite(true)?;
    owner.set_experimental_gradient_norm_partials(true)?;
    owner.set_experimental_borrowed_bf16_adam_source(true)?;
    owner.set_experimental_vector_bf16_first_adam(true)?;
    owner.set_experimental_small_accumulation_norm_batches(true)?;
    owner.begin_step()?;
    let selected_options = MixedOptions {
        small_grid_split_k_bf16: true,
        ..options(false, true)
    };
    let mut executor =
        owner.mixed_executor_with_options(&graph, MatMulPrecision::Tf32x3, selected_options)?;
    let names = model
        .parameters
        .iter()
        .map(|(n, p)| (p.parameter.id(), n))
        .collect::<HashMap<_, _>>();
    let mut expected_names = HashSet::from(["token.weight".to_owned()]);
    for layer in 0..4 {
        for role in ["qkv", "out", "gate", "up", "down"] {
            expected_names.insert(format!("layer.{layer}.{role}.weight"));
        }
    }
    let mut observations = Vec::new();
    for round in 0..2 {
        if round != 0 {
            owner.begin_step()?;
        }
        owner.accumulate_mixed(&mut executor, &graph, HashMap::new(), model.config.feeds()?)?;
        let loss = executor.read(loss)?;
        ensure!(
            loss.iter().all(|v| v.is_finite()),
            "live census loss is nonfinite"
        );
        let registered = graph
            .graph
            .node_indices()
            .filter_map(|node| {
                let TensorGraphNode::Parameter { id, data, .. } = &graph[node] else {
                    return None;
                };
                Some((|| -> Result<_> {
                    let parameter = owner
                        .parameters
                        .iter()
                        .find(|p| p.id == *id)
                        .context("unregistered graph master")?;
                    ensure!(
                        Arc::ptr_eq(data, &parameter.storage)
                            && data.revision()? == parameter.revision,
                        "owner/graph storage or revision mismatch"
                    );
                    Ok((
                        node,
                        RegisteredMaster {
                            parameter_id: *id,
                            storage: parameter.storage.clone(),
                            revision: parameter.revision,
                            allocation: parameter.weight.clone(),
                        },
                    ))
                })())
            })
            .collect::<Result<HashMap<_, _>>>()?;
        let mut observation = shadow_cast_census::census(&executor, &graph, &registered)?;
        let rows = observation["rows"]
            .as_array_mut()
            .context("census rows absent")?;
        for row in rows.iter_mut() {
            if let Some(id) = row["parameter_id"]
                .as_u64()
                .and_then(|id| usize::try_from(id).ok())
                && let Some(name) = names.get(&id)
            {
                row["parameter_name"] = (*name).into();
            }
        }
        // Preserve observed metadata before fixed coverage assertions, so a
        // failed census remains inspectable without rerunning a live executor.
        let raw = root.join(format!("raw-round{round}.json"));
        ensure!(!raw.exists(), "fresh raw census output required");
        std::fs::write(&raw, serde_json::to_vec_pretty(&rows)?)?;
        let candidates = rows
            .iter()
            .filter(|r| r["admitted_candidate"] == true)
            .collect::<Vec<_>>();
        let actual_names = candidates
            .iter()
            .map(|r| {
                r["parameter_name"]
                    .as_str()
                    .context("candidate name absent")
                    .map(str::to_owned)
            })
            .collect::<Result<HashSet<_>>>()?;
        let ids = candidates
            .iter()
            .map(|r| r["parameter_id"].as_u64().context("candidate id absent"))
            .collect::<Result<HashSet<_>>>()?;
        let elements = candidates
            .iter()
            .map(|r| {
                r["source_elements"]
                    .as_u64()
                    .context("candidate elements absent")
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .sum::<u64>();
        ensure!(
            candidates.len() == 21
                && ids.len() == 21
                && actual_names == expected_names
                && elements == 54_788_096,
            "live matrix-cast census changed"
        );
        ensure!(
            candidates.iter().all(|r| r["source_dtype"] == "F32"
                && r["target_dtype"] == "BF16"
                && r["current_revision"] == round
                && r["expected_revision"] == round
                && r["f32_master_binding_preserved"] == true
                && r["source_chain"]
                    .as_array()
                    .is_some_and(|c| c.len() == 1 && c[0]["kind"] == "Transpose")),
            "live candidates are not exact current whole transposed matrix casts"
        );
        let head = candidates
            .iter()
            .find(|r| r["parameter_name"] == "token.weight")
            .context("tied head absent")?;
        ensure!(
            head["parameter_f32_consumers"]
                .as_array()
                .is_some_and(|a| a.iter().any(|r| r["kind"] == "Embedding")),
            "tied token F32 embedding consumer not retained"
        );
        ensure!(
            observation["physical_order_cast_materializations"] == 21,
            "existing physical materializer count changed"
        );
        observation["round"] = round.into();
        observation["matrix_owner_count"] = 21.into();
        observation["matrix_owner_elements"] = elements.into();
        observation["activation_casts_admitted"] = 0.into();
        observation["loss_read_before_census"] = true.into();
        observation["mixed_info"] = serde_json::to_value(executor.info())?;
        observations.push(observation);
        if round == 0 {
            owner.step_mixed(&executor, model.config.max_grad_norm)?;
            ensure!(
                owner.vector_bf16_first_adam_status().committed_parameters == 20,
                "actual vector owner census changed"
            );
        }
    }
    ensure!(
        executor.info().direct_capture_variants == 2 && executor.info().direct_capture_misses == 2,
        "expected two real direct master capture variants"
    );
    std::fs::write(
        out,
        serde_json::to_vec_pretty(&json!({
            "scope":"CFG(test) actual accepted W1024/S128/B2+experimental vector-owner existing cast census. No shadow allocations/stores, no performance claim.",
            "config":model.config,"exact_accepted_small_grid_option":true,"registered_parameters":60,
            "registered_elements":model.parameter_count(),"candidate_matrix_owners":21,"candidate_elements":54_788_096u64,
            "tied_head_f32_embedding_retained":true,"zero_activation_casts_admitted":true,
            "same_twenty_one_after_real_master_publication":true,"two_direct_capture_variants":true,
            "observations":observations,
        }))?,
    )?;
    Ok(())
}

#[path = "../bf16_shadow/owner_state_tests.rs"]
mod shadow_state;
