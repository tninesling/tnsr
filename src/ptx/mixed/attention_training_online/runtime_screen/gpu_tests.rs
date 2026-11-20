use super::*;
use crate::ptx::{
    mixed::{
        MixedOptions,
        test_decoder::{Config, synthetic},
    },
    resident::{AdamGroup, ResidentAdam},
};

pub(super) fn options() -> MixedOptions {
    MixedOptions {
        online_attention_pair_numerical_screen: true,
        layer_norm_pair_fusion: true,
        attention_softmax_fusion: true,
        attention_softmax_backward_fusion: true,
        swiglu_pair_fusion: true,
        qkv_grad_pack_fusion: true,
        transposed_bf16_outputs: true,
        split_k_bf16_tile_policy: crate::tile::SplitKBf16TilePolicy::Square64,
        direct_master_bindings: true,
        incremental_allocation_reuse: true,
        physical_order_casts: true,
        shared_memory_reuse: true,
        narrow_bf16_tile: true,
        medium_bf16_tile_policy: crate::tile::MediumBf16TilePolicy::SquareThenTall,
        column_reduction_policy: crate::tile::ColumnReductionPolicy::Four,
        batched_attention_bf16_tile_policy:
            crate::tile::BatchedAttentionBf16TilePolicy::TallScoresByGrid,
        indexed_loss_cooperative_ordered_forward: true,
        indexed_loss_cooperative_ordered_backward: true,
        plain_bf16_output: true,
        ..MixedOptions::default()
    }
}
fn actual_admission(batch: usize) -> Result<()> {
    if !crate::ptx::mixed::tests::cuda_available() {
        return Ok(());
    }
    let model = synthetic(Config {
        batch,
        sequence: 512,
        width: 512,
        heads: 8,
        hidden: 1408,
        layers: 4,
        vocab: 4096,
        epsilon: 1e-5,
        learning_rate: 1e-4,
        max_grad_norm: 1.,
    });
    let (graph, _) = model.training_graph_with_options(crate::graph::GradientOptions {
        compact_softmax: true,
    })?;
    let parameters = model
        .parameters
        .iter()
        .map(|(_, p)| (p.parameter.clone(), 0))
        .collect();
    let mut owner = ResidentAdam::new_with_graph_replay(parameters, vec![AdamGroup::new(1e-4)])?;
    owner.set_update_batch_size(8)?;
    owner.set_experimental_first_accumulation_overwrite(true)?;
    owner.set_experimental_gradient_norm_partials(true)?;
    owner.begin_step()?;
    let executor = owner.mixed_executor_with_options(
        &graph,
        crate::tile::MatMulPrecision::Tf32x3,
        options(),
    )?;
    let installed = executor.kernels.iter().filter(|k| k.label.starts_with("online attention NUMERICAL SCREEN")).map(|k| {
        Ok(serde_json::json!({"owner":k.owner.map(|n|n.index()),"label":k.label,"arguments":k.pointers.len(),"grid":k.config.grid_dim,"block":k.config.block_dim,"registers":k.function.num_regs()?,"shared":k.function.shared_size_bytes()?,"local":k.function.local_size_bytes()?,"capacity":k.function.occupancy_max_active_blocks_per_multiprocessor(256,0,None)?}))
    }).collect::<Result<Vec<_>>>()?;
    if let Some(dir) = std::env::var_os("TNSR_ONLINE_SCREEN_AUDIT") {
        std::fs::create_dir_all(&dir)?;
        std::fs::write(
            std::path::PathBuf::from(dir).join(format!("actual-b{batch}.json")),
            serde_json::to_vec_pretty(
                &serde_json::json!({"info":executor.info(),"installed":installed,"original_or_replacement_boundaries":executor.kernels.iter().map(|k|serde_json::json!({"owner":k.owner.map(|n|n.index()),"label":k.label})).collect::<Vec<_>>(),"scope":"Numerical screen retains old quadratic owners. API capacity is not achieved occupancy."}),
            )?,
        )?;
    }
    assert_eq!(
        executor.info().online_attention_pair_numerical_screen_pairs,
        4,
        "{:?}",
        executor.info()
    );
    assert_eq!(
        executor
            .info()
            .online_attention_pair_numerical_screen_fallback_pairs,
        0
    );
    assert_eq!(
        executor
            .info()
            .online_attention_pair_numerical_screen_async16_pairs,
        4
    );
    assert_eq!(
        executor
            .info()
            .online_attention_pair_numerical_screen_scalar_pairs,
        0
    );
    assert_eq!(
        executor
            .info()
            .online_attention_pair_numerical_screen_workspace_bytes,
        4 * batch * 8 * 512 * 12
    );
    assert_eq!(installed.len(), 12);
    assert_eq!(executor.info().attention_softmax_regions, 0);
    assert_eq!(executor.info().attention_softmax_backward_regions, 0);
    assert_eq!(executor.info().attention_training_forward_regions, 0);
    assert_eq!(executor.info().qkv_grad_pack_regions, 4);
    assert_eq!(executor.info().layer_norm_pair_forward_regions, 9);
    assert_eq!(executor.info().layer_norm_pair_backward_regions, 9);
    assert_eq!(executor.transfers().parameter_copies, 0);
    Ok(())
}

#[test]
fn online_screen_actual_b8_four_pairs_and_retained_allocation_scope() -> Result<()> {
    actual_admission(8)
}
#[test]
fn online_screen_actual_b16_kept_binding_metadata_and_admission() -> Result<()> {
    actual_admission(16)
}

#[test]
fn online_screen_missing_consumer_graph_paths_cpu() -> Result<()> {
    let model = synthetic(Config {
        batch: 8,
        sequence: 512,
        width: 512,
        heads: 8,
        hidden: 1408,
        layers: 4,
        vocab: 4096,
        epsilon: 1e-5,
        learning_rate: 1e-4,
        max_grad_norm: 1.,
    });
    let (graph, _) = model.training_graph_with_options(crate::graph::GradientOptions {
        compact_softmax: true,
    })?;
    let packs = crate::ptx::mixed::qkv_grad_pack::recognize(&graph);
    let selected = packs.iter().map(|p| p.output).collect();
    let swiglu = crate::ptx::mixed::swiglu_pair::recognize(&graph)?;
    let norms = crate::ptx::mixed::layer_norm_pair::recognize(&graph)?;
    let pairs = pair_recognize::recognize(&graph, &selected);
    assert_eq!(pairs.len(), 4);
    for pair in pairs {
        for (output, consumer, _) in pair.surviving_consumers {
            println!(
                "output={} consumer={} op={} shape={:?} inputs={:?} pack={:?} swiglu={:?} ln={:?}",
                output.index(),
                consumer.index(),
                graph[consumer].name(),
                graph[consumer].shape(),
                graph
                    .inputs(consumer)
                    .iter()
                    .map(|n| n.index())
                    .collect::<Vec<_>>(),
                packs
                    .iter()
                    .filter(|p| p.members.contains(&consumer) || p.output == consumer)
                    .map(|p| p.output.index())
                    .collect::<Vec<_>>(),
                swiglu
                    .iter()
                    .filter(|p| p.members.contains(&consumer))
                    .map(|p| [p.forward_emit.index(), p.backward_emit.index()])
                    .collect::<Vec<_>>(),
                norms
                    .iter()
                    .filter(|p| p.members.contains(&consumer))
                    .map(|p| [p.forward.index(), p.backward.index()])
                    .collect::<Vec<_>>()
            );
        }
    }
    Ok(())
}

pub(super) struct StateFixture {
    pub(super) graph: TensorGraph<f32, crate::graph::WithGrad>,
    pub(super) loss: NodeIndex,
    pub(super) parameters: Vec<crate::tensor::Parameter<f32>>,
    pub(super) feeds: crate::tensor::IndexInputs,
    vocab: usize,
}
fn state_fixture(batch: usize) -> Result<StateFixture> {
    state_fixture_with_vocab(batch, 4096)
}
pub(super) fn state_fixture_with_vocab(batch: usize, vocab: usize) -> Result<StateFixture> {
    use crate::tensor::TensorExpr;
    let model = synthetic(Config {
        batch,
        sequence: 512,
        width: 512,
        heads: 8,
        hidden: 1408,
        layers: 4,
        vocab,
        epsilon: 1e-5,
        learning_rate: 1e-4,
        max_grad_norm: 1.,
    });
    let expression = model.loss()? * TensorExpr::input("loss_seed", vec![]);
    let mut graph = TensorGraph::new();
    let loss = graph.add_expr(&expression, false);
    Ok(StateFixture {
        graph: graph.with_gradients_with_options(
            loss,
            crate::graph::GradientOptions {
                compact_softmax: true,
            },
        ),
        loss,
        parameters: model
            .parameters
            .iter()
            .map(|(_, p)| p.parameter.clone())
            .collect(),
        feeds: model.config.feeds()?,
        vocab,
    })
}
pub(super) fn state_owner(f: &StateFixture, replay: bool) -> Result<ResidentAdam> {
    state_owner_with_smallnorm(f, replay, false)
}
pub(super) fn state_owner_with_smallnorm(
    f: &StateFixture,
    replay: bool,
    smallnorm: bool,
) -> Result<ResidentAdam> {
    state_owner_with_policies(f, replay, smallnorm, false)
}
pub(super) fn state_owner_with_policies(
    f: &StateFixture,
    replay: bool,
    smallnorm: bool,
    borrowed: bool,
) -> Result<ResidentAdam> {
    let parameters = f.parameters.iter().cloned().map(|p| (p, 0)).collect();
    let groups = vec![AdamGroup::new(1e-4)];
    let mut owner = if replay {
        ResidentAdam::new_with_graph_replay(parameters, groups)?
    } else {
        ResidentAdam::new(parameters, groups)?
    };
    owner.set_update_batch_size(8)?;
    owner.set_experimental_first_accumulation_overwrite(true)?;
    owner.set_experimental_gradient_norm_partials(true)?;
    if borrowed {
        owner.set_experimental_borrowed_bf16_adam_source(true)?;
    }
    if smallnorm {
        owner.set_experimental_small_accumulation_norm_batches(true)?;
    }
    owner.begin_step()?;
    Ok(owner)
}
pub(super) fn same_bits(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (i, (&a, &b)) in a.iter().zip(b).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "exact state mismatch at {i}: {a:?}/{b:?}"
        );
    }
}
type PublishedState = (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>);
pub(super) fn published(owner: &ResidentAdam, f: &StateFixture) -> Result<PublishedState> {
    let moments = owner.moments()?;
    Ok((
        f.parameters
            .iter()
            .map(|p| p.values())
            .collect::<Result<_>>()?,
        f.parameters
            .iter()
            .map(|p| moments[&p.id()].first.clone())
            .collect(),
        f.parameters
            .iter()
            .map(|p| moments[&p.id()].second.clone())
            .collect(),
    ))
}
pub(super) fn same_published(a: &PublishedState, b: &PublishedState) {
    for (a, b) in [(&a.0, &b.0), (&a.1, &b.1), (&a.2, &b.2)] {
        assert_eq!(a.len(), b.len());
        for (a, b) in a.iter().zip(b) {
            same_bits(a, b);
        }
    }
}
pub(super) fn changed_feeds(f: &StateFixture, revision: usize) -> crate::tensor::IndexInputs {
    let mut feeds = f.feeds.clone();
    for indices in feeds.values_mut() {
        for n in indices {
            *n = (*n + revision as u32) % u32::try_from(f.vocab).unwrap();
        }
    }
    feeds
}
fn exact_state_and_recovery(
    batch: usize,
    fused_nt128_bias: bool,
    contiguous_bias_reshape_aliases: bool,
) -> Result<()> {
    let fixtures = [state_fixture(batch)?, state_fixture(batch)?];
    let mut owners = [
        state_owner(&fixtures[0], false)?,
        state_owner(&fixtures[1], true)?,
    ];
    let mut executors = owners
        .iter()
        .enumerate()
        .map(|(i, o)| {
            let mut selected = options();
            selected.direct_master_bindings = i == 1;
            selected.fused_nt128_bias = fused_nt128_bias;
            selected.contiguous_bias_reshape_aliases = contiguous_bias_reshape_aliases;
            o.mixed_executor_with_options(
                &fixtures[i].graph,
                crate::tile::MatMulPrecision::Tf32x3,
                selected,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    for e in &executors {
        assert_eq!(
            e.info().online_attention_pair_numerical_screen_pairs,
            4,
            "{:?}",
            e.info()
        );
        assert_eq!(
            e.info()
                .online_attention_pair_numerical_screen_fallback_pairs,
            0
        );
        assert_eq!(
            e.info()
                .online_attention_pair_numerical_screen_async16_pairs,
            4
        );
        assert_eq!(
            e.info().online_attention_pair_numerical_screen_scalar_pairs,
            0
        );
        assert_eq!(e.info().layer_norm_pair_forward_regions, 9);
        assert_eq!(e.info().layer_norm_pair_backward_regions, 9);
        if fused_nt128_bias {
            assert!(e.info().fused_nt128_bias_programs > 0, "{:?}", e.info());
        }
        if contiguous_bias_reshape_aliases {
            assert_eq!(
                e.info().contiguous_bias_reshape_alias_steps,
                20,
                "{:?}",
                e.info()
            );
            assert_eq!(e.info().contiguous_bias_reshape_fallback_steps, 0);
            assert_eq!(e.info().fused_nt128_bias_programs, 13);
            assert_eq!(
                e.info().contiguous_bias_reshape_avoided_bytes,
                batch * 512 * (8 * 512 + 8 * 1408 + 4 * 1536) * 2
            );
        }
    }
    if fused_nt128_bias {
        assert_eq!(
            executors[0].info().fused_nt128_bias_programs,
            executors[1].info().fused_nt128_bias_programs
        );
        assert_eq!(
            executors[0].info().fused_nt128_bias_fallback_programs,
            executors[1].info().fused_nt128_bias_fallback_programs
        );
    }
    assert!(!executors[0].info().replay);
    assert!(executors[1].info().replay);
    let mut private_copies = 0;
    for (interval, microbatches) in [1, 2, 1].into_iter().enumerate() {
        let mut steps = Vec::new();
        for i in 0..2 {
            if interval > 0 {
                owners[i].begin_step()?;
            }
            for micro in 0..microbatches {
                owners[i].accumulate_mixed(
                    &mut executors[i],
                    &fixtures[i].graph,
                    HashMap::from([(
                        "loss_seed".into(),
                        vec![1. + (interval + micro) as f32 / 8.],
                    )]),
                    changed_feeds(&fixtures[i], interval + micro),
                )?;
                if i == 0 {
                    private_copies += executors[i].transfers().parameter_copies;
                } else {
                    assert_eq!(executors[i].transfers().parameter_copies, 0);
                }
            }
            steps.push(owners[i].step(1.)?);
        }
        assert_eq!(
            steps[0].gradient_norm.to_bits(),
            steps[1].gradient_norm.to_bits()
        );
        assert_eq!(
            steps[0].gradient_scale.to_bits(),
            steps[1].gradient_scale.to_bits()
        );
        assert_eq!(steps[0].timestep, steps[1].timestep);
        same_bits(
            &executors[0].read(fixtures[0].loss)?,
            &executors[1].read(fixtures[1].loss)?,
        );
        let gradients = [owners[0].gradients()?, owners[1].gradients()?];
        for (a, b) in fixtures[0].parameters.iter().zip(&fixtures[1].parameters) {
            same_bits(&gradients[0][&a.id()], &gradients[1][&b.id()]);
        }
        same_published(
            &published(&owners[0], &fixtures[0])?,
            &published(&owners[1], &fixtures[1])?,
        );
        if interval == 1 {
            for i in 0..2 {
                let before = published(&owners[i], &fixtures[i])?;
                owners[i].begin_step()?;
                let _ = owners[i].accumulate_mixed(
                    &mut executors[i],
                    &fixtures[i].graph,
                    HashMap::from([("loss_seed".into(), vec![f32::NAN])]),
                    changed_feeds(&fixtures[i], interval),
                );
                assert!(
                    owners[i].step(1.).is_err(),
                    "nonfinite interval must reject publication"
                );
                same_published(&before, &published(&owners[i], &fixtures[i])?);
            }
        }
    }
    assert_eq!(executors[1].transfers().parameter_copies, 0);
    assert_eq!(executors[1].info().direct_capture_variants, 2);
    assert_eq!(executors[1].info().direct_capture_misses, 2);
    // Transfers are per upload, not cumulative. After a rejected update the
    // final recovery may correctly reuse the same source identity and copy0.
    assert!(private_copies > 0);
    if let Some(dir) = std::env::var_os("TNSR_ONLINE_SCREEN_AUDIT") {
        std::fs::create_dir_all(&dir)?;
        std::fs::write(
            std::path::PathBuf::from(dir).join(if contiguous_bias_reshape_aliases {
                format!("exact-state-nt128-reshape-alias-b{batch}.json")
            } else if fused_nt128_bias {
                format!("exact-state-nt128-b{batch}.json")
            } else {
                format!("exact-state-b{batch}.json")
            }),
            serde_json::to_vec_pretty(
                &serde_json::json!({"batch":batch,"sequence":512,"width":512,"hidden":1408,"vocab":4096,"intervals":[1,2,1],"nan_rollback_recovery":true,"exact_loss_clipped_gradients_masters_both_moments":true,"uncaptured_private":executors[0].info(),"captured_direct":executors[1].info(),"scope":if contiguous_bias_reshape_aliases { "same screen-on, fused-NT128 and contiguous-bias-reshape alias policy in both modes; nominal avoided allocation bytes are not measured memory savings" } else if fused_nt128_bias { "same screen-on and fused-NT128 policy in both modes, not equality to NT-off or original checkpoint arithmetic" } else { "same screen-on numerical policy in both modes, not equality to original checkpoint arithmetic" }}),
            )?,
        )?;
    }
    Ok(())
}
#[test]
fn online_screen_actual_b8_captured_direct_vs_uncaptured_exact_state_and_recovery() -> Result<()> {
    if !crate::ptx::mixed::tests::cuda_available() {
        return Ok(());
    }
    exact_state_and_recovery(8, false, false)
}
#[test]
fn online_screen_actual_b16_captured_direct_vs_uncaptured_exact_state_and_recovery() -> Result<()> {
    if !crate::ptx::mixed::tests::cuda_available() {
        return Ok(());
    }
    exact_state_and_recovery(16, false, false)
}

#[test]
fn nt128_bias_actual_b8_captured_direct_vs_uncaptured_exact_state_and_recovery() -> Result<()> {
    if !crate::ptx::mixed::tests::cuda_available() {
        return Ok(());
    }
    exact_state_and_recovery(8, true, false)
}
#[test]
fn nt128_bias_actual_b16_captured_direct_vs_uncaptured_exact_state_and_recovery() -> Result<()> {
    if !crate::ptx::mixed::tests::cuda_available() {
        return Ok(());
    }
    exact_state_and_recovery(16, true, false)
}

#[test]
fn contiguous_bias_alias_actual_b8_captured_direct_vs_private_exact_state_and_recovery()
-> Result<()> {
    if !crate::ptx::mixed::tests::cuda_available() {
        return Ok(());
    }
    exact_state_and_recovery(8, true, true)
}
#[test]
fn contiguous_bias_alias_actual_b16_captured_direct_vs_private_exact_state_and_recovery()
-> Result<()> {
    if !crate::ptx::mixed::tests::cuda_available() {
        return Ok(());
    }
    exact_state_and_recovery(16, true, true)
}

/// Sequential GPU owners bound the peak: the complete old-WMMA trajectory is
/// retained as CPU snapshots, then its GPU allocations are dropped before the
/// captured/direct selected-ldmatrix trajectory. Unique token/target permutations
/// avoid token scatter atomics; this is a synthetic V16384 fixture, not archive V4096.
#[test]
#[ignore = "requires real CUDA; selected B32 unique-token sequential full-state proof"]
fn production_ldmatrix_actual_b32_unique_tokens_selected_exact_state_and_recovery() -> Result<()> {
    unique_b32_selected_state_and_recovery(false)
}
#[test]
#[ignore = "requires real CUDA; selected NN128x64 B32 unique-token sequential full-state proof"]
fn production_nn128x64_actual_b32_unique_tokens_selected_exact_state_and_recovery() -> Result<()> {
    unique_b32_selected_state_and_recovery(true)
}
fn unique_b32_selected_state_and_recovery(nn128x64: bool) -> Result<()> {
    if !crate::ptx::mixed::tests::cuda_available() {
        return Ok(());
    }
    struct Interval {
        loss: Vec<f32>,
        norm: u64,
        scale: u32,
        timestep: usize,
        gradients: Vec<Vec<f32>>,
        published: PublishedState,
    }
    let mut expected: Vec<Interval> = Vec::new();
    let mut summary = Vec::new();
    for selected in [false, true] {
        let f = state_fixture_with_vocab(32, 16384)?;
        for feed in f.feeds.values() {
            assert_eq!(feed.len(), 16384);
            assert_eq!(
                feed.iter().collect::<std::collections::HashSet<_>>().len(),
                16384
            );
        }
        let mut owner = state_owner_with_smallnorm(&f, selected, nn128x64)?;
        let mut policy = options();
        policy.fused_nt128_bias = true;
        policy.fused_nt128_bias_ldmatrix = nn128x64 || selected;
        policy.nn128x64_ldmatrix = nn128x64 && selected;
        policy.contiguous_bias_reshape_aliases = true;
        policy.direct_master_bindings = selected;
        let mut e = owner.mixed_executor_with_options(
            &f.graph,
            crate::tile::MatMulPrecision::Tf32x3,
            policy,
        )?;
        assert_eq!(e.info().fused_nt128_bias_programs, 13, "{:?}", e.info());
        assert_eq!(
            e.info().fused_nt128_bias_ldmatrix_programs,
            if nn128x64 || selected { 12 } else { 0 },
            "{:?}",
            e.info()
        );
        assert_eq!(
            e.info().fused_nt128_bias_ldmatrix_fallback_programs,
            0,
            "{:?}",
            e.info()
        );
        assert_eq!(
            e.info().nn128x64_ldmatrix_programs,
            if nn128x64 && selected { 8 } else { 0 },
            "{:?}",
            e.info()
        );
        assert_eq!(
            e.info().nn128x64_ldmatrix_fallback_programs,
            0,
            "{:?}",
            e.info()
        );
        assert_eq!(
            e.info()
                .online_attention_pair_numerical_screen_async16_pairs,
            4
        );
        assert_eq!(e.info().contiguous_bias_reshape_alias_steps, 20);
        let mut rows = Vec::new();
        for (interval, microbatches) in [1, 2, 1].into_iter().enumerate() {
            if interval > 0 {
                owner.begin_step()?;
            }
            for micro in 0..microbatches {
                owner.accumulate_mixed(
                    &mut e,
                    &f.graph,
                    HashMap::from([(
                        "loss_seed".into(),
                        vec![1. + (interval + micro) as f32 / 8.],
                    )]),
                    changed_feeds(&f, interval + micro),
                )?;
                if selected {
                    assert_eq!(e.transfers().parameter_copies, 0);
                }
            }
            let norm_status = owner.small_accumulation_norm_batches_status().clone();
            if nn128x64 {
                assert!(norm_status.enabled, "{norm_status:?}");
                if microbatches == 1 {
                    assert!(
                        norm_status.last_accumulation_used
                            && norm_status.selected_members > 0
                            && norm_status.batched_launches > 0
                            && norm_status.fallback_reason.is_empty(),
                        "{norm_status:?}"
                    );
                } else {
                    assert!(
                        !norm_status.last_accumulation_used
                            && norm_status.selected_members == 0
                            && norm_status.batched_launches == 0
                            && norm_status.actual_launches > 0
                            && norm_status.fallback_reason == "not_first_overwrite_norm_attempt",
                        "{norm_status:?}"
                    );
                }
            }
            let step = owner.step(1.)?;
            let gradients = owner.gradients()?;
            let state = Interval {
                loss: e.read(f.loss)?,
                norm: step.gradient_norm.to_bits(),
                scale: step.gradient_scale.to_bits(),
                timestep: step.timestep,
                gradients: f
                    .parameters
                    .iter()
                    .map(|p| gradients[&p.id()].clone())
                    .collect(),
                published: published(&owner, &f)?,
            };
            if selected {
                let old = &expected[interval];
                same_bits(&old.loss, &state.loss);
                assert_eq!(
                    (old.norm, old.scale, old.timestep),
                    (state.norm, state.scale, state.timestep)
                );
                for (a, b) in old.gradients.iter().zip(&state.gradients) {
                    same_bits(a, b);
                }
                same_published(&old.published, &state.published);
            } else {
                expected.push(state);
            }
            rows.push(serde_json::json!({"interval":interval,"microbatches":microbatches,"small_accumulation_norm":norm_status,"loss_norm_scale_all_gradients_masters_both_moments_exact":true}));
            if interval == 1 {
                let before = published(&owner, &f)?;
                owner.begin_step()?;
                let _ = owner.accumulate_mixed(
                    &mut e,
                    &f.graph,
                    HashMap::from([("loss_seed".into(), vec![f32::NAN])]),
                    changed_feeds(&f, interval),
                );
                assert!(owner.step(1.).is_err());
                same_published(&before, &published(&owner, &f)?);
            }
        }
        if selected {
            assert_eq!(e.info().direct_capture_variants, 2);
            assert_eq!(e.info().direct_capture_misses, 2);
            assert_eq!(e.transfers().parameter_copies, 0);
        }
        summary.push(serde_json::json!({"ldmatrix":selected,"info":e.info(),"intervals":rows}));
        e.synchronize()?;
        drop(e);
        drop(owner);
        drop(f);
    }
    if let Some(dir) = std::env::var_os("TNSR_ONLINE_SCREEN_AUDIT") {
        std::fs::create_dir_all(&dir)?;
        std::fs::write(
            std::path::PathBuf::from(dir).join(if nn128x64 {
                "exact-state-nn128x64-b32-unique-v16384.json"
            } else {
                "exact-state-ldmatrix-b32-unique-v16384.json"
            }),
            serde_json::to_vec_pretty(
                &serde_json::json!({"scope":if nn128x64 { "Synthetic B32 V16384 unique-token full-state proof, distinct from archived V4096 gates. AcceptedNTLD12+smallnorm both owners; sequential oldNN64 private/uncaptured versus selectedNN128x64 captured/direct; no performance claim." } else { "Synthetic B32 V16384 unique-token full-state proof, distinct from archived V4096 fixed gates. Sequential old-WMMA private/uncaptured versus selected ldmatrix captured/direct; no performance claim." },"batch":32,"vocab":16384,"rows":summary,"nan_rollback_recovery":true}),
            )?,
        )?;
    }
    Ok(())
}

// Reuse the same unique-token decoder and state readers without changing them.
#[path = "../../split_k_rectangle_state_tests.rs"]
mod split_k_rectangle_state_tests;

#[path = "register_statistics_gpu.rs"]
mod register_statistics_gpu;
