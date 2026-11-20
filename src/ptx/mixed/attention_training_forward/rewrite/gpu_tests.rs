use super::*;
use crate::{
    graph::WithGrad,
    ptx::resident::{AdamGroup, ResidentAdam},
    tensor::{Parameter, TensorExpr},
};
struct Fixture {
    graph: TensorGraph<f32, WithGrad>,
    loss: NodeIndex,
    parameters: Vec<Parameter<f32>>,
    indices: crate::tensor::IndexInputs,
    vocab: usize,
}
fn fixture(batch: usize) -> Fixture {
    use crate::ptx::mixed::test_decoder::{Config, synthetic};
    // Full decoder graph/AD interfaces, bounded hidden width; unique embedding
    // indices remove unrelated atomic ordering from exact state qualification.
    let vocab = batch * 512;
    let model = synthetic(Config {
        batch,
        sequence: 512,
        width: 512,
        heads: 8,
        hidden: 64,
        layers: 4,
        vocab,
        epsilon: 1e-5,
        learning_rate: 1e-4,
        max_grad_norm: 1.,
    });
    let loss = model.loss().unwrap() * TensorExpr::input("loss_seed", vec![]);
    let mut g = TensorGraph::new();
    let loss = g.add_expr(&loss, false);
    Fixture {
        graph: g.with_gradients_with_options(
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
        indices: model.config.feeds().unwrap(),
        vocab,
    }
}
fn indices(f: &Fixture, revision: usize) -> crate::tensor::IndexInputs {
    let mut feeds = f.indices.clone();
    for values in feeds.values_mut() {
        for value in values {
            *value = (*value + revision as u32) % f.vocab as u32;
        }
    }
    feeds
}
fn options(enabled: bool) -> MixedOptions {
    MixedOptions {
        attention_training_forward_fusion: enabled,
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
        ..MixedOptions::default()
    }
}
fn owner(parameters: &[Parameter<f32>]) -> Result<ResidentAdam> {
    let mut owner = ResidentAdam::new_with_graph_replay(
        parameters.iter().cloned().map(|p| (p, 0)).collect(),
        vec![AdamGroup::new(1e-4)],
    )?;
    owner.set_update_batch_size(8)?;
    owner.set_experimental_first_accumulation_overwrite(true)?;
    owner.set_experimental_gradient_norm_partials(true)?;
    owner.begin_step()?;
    Ok(owner)
}
fn audit(e: &MixedExecutor, name: &str) -> Result<()> {
    if let Some(dir) = std::env::var_os("TNSR_ATTENTION_TRAINING_RUNTIME_AUDIT") {
        let dir = std::path::PathBuf::from(dir);
        std::fs::create_dir_all(&dir)?;
        let kernels=e.kernels.iter().filter(|k|k.label.starts_with("attention training forward four checkpoints")).map(|k|Ok(serde_json::json!({"owner":k.owner.map(|n|n.index()),"registers":k.function.num_regs()?,"shared":k.function.shared_size_bytes()?,"local":k.function.local_size_bytes()?,"max_threads":k.function.max_threads_per_block()?,"capacity":k.function.occupancy_max_active_blocks_per_multiprocessor(256,0,None)?,"arguments":k.pointers.len(),"grid":k.config.grid_dim,"block":k.config.block_dim}))).collect::<Result<Vec<_>>>()?;
        std::fs::write(
            dir.join(format!("{name}.json")),
            serde_json::to_vec_pretty(
                &serde_json::json!({"info":e.info(),"installed":kernels,"scope":"actual installed resources; capacity is not measured occupancy"}),
            )?,
        )?;
    }
    Ok(())
}
fn bits(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (i, (&a, &b)) in a.iter().zip(b).enumerate() {
        assert!(
            (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits(),
            "index{i}: {a:?}/{b:?}"
        );
    }
}
type State = (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>);
fn state(owner: &ResidentAdam, parameters: &[Parameter<f32>]) -> Result<State> {
    let moments = owner.moments()?;
    Ok((
        parameters
            .iter()
            .map(|p| p.values())
            .collect::<Result<_>>()?,
        parameters
            .iter()
            .map(|p| moments[&p.id()].first.clone())
            .collect(),
        parameters
            .iter()
            .map(|p| moments[&p.id()].second.clone())
            .collect(),
    ))
}
fn same_state(a: &State, b: &State) {
    for (aa, bb) in [(&a.0, &b.0), (&a.1, &b.1), (&a.2, &b.2)] {
        for (a, b) in aa.iter().zip(bb) {
            bits(a, b);
        }
    }
}
#[test]
fn attention_training_runtime_bh128_ln_direct_replay_state_and_recovery() -> Result<()> {
    if !crate::ptx::mixed::tests::cuda_available() {
        return Ok(());
    }
    let fixtures = [fixture(16), fixture(16)];
    let mut owners = fixtures
        .iter()
        .map(|f| owner(&f.parameters))
        .collect::<Result<Vec<_>>>()?;
    let mut executors = owners
        .iter()
        .enumerate()
        .map(|(i, o)| {
            o.mixed_executor_with_options(
                &fixtures[i].graph,
                MatMulPrecision::Tf32x3,
                options(i == 1),
            )
        })
        .collect::<Result<Vec<_>>>()?;
    for e in &executors {
        assert_eq!(e.info().layer_norm_pair_forward_regions, 9);
        assert_eq!(e.info().layer_norm_pair_backward_regions, 9);
        assert!(e.info().reused_allocations > 0);
    }
    assert_eq!(executors[0].info().attention_training_forward_regions, 0);
    assert_eq!(executors[1].info().attention_training_forward_regions, 4);
    assert_eq!(
        executors[1]
            .info()
            .attention_training_forward_fallback_regions,
        0
    );
    assert_eq!(
        executors[1]
            .info()
            .attention_training_forward_materializations,
        0
    );
    assert_eq!(executors[0].info().attention_softmax_regions, 4);
    assert_eq!(executors[1].info().attention_softmax_regions, 0);
    audit(&executors[1], "state-bh128-hidden64")?;
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
                    indices(&fixtures[i], interval + micro),
                )?;
            }
            steps.push(owners[i].step(1.)?);
            assert_eq!(executors[i].transfers().parameter_copies, 0);
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
        bits(
            &executors[0].read(fixtures[0].loss)?,
            &executors[1].read(fixtures[1].loss)?,
        );
        let grads = [owners[0].gradients()?, owners[1].gradients()?];
        for (a, b) in fixtures[0].parameters.iter().zip(&fixtures[1].parameters) {
            bits(&grads[0][&a.id()], &grads[1][&b.id()]);
        }
        same_state(
            &state(&owners[0], &fixtures[0].parameters)?,
            &state(&owners[1], &fixtures[1].parameters)?,
        );
        if interval == 1 {
            for i in 0..2 {
                let before = state(&owners[i], &fixtures[i].parameters)?;
                owners[i].begin_step()?;
                let _ = owners[i].accumulate_mixed(
                    &mut executors[i],
                    &fixtures[i].graph,
                    HashMap::from([("loss_seed".into(), vec![f32::NAN])]),
                    indices(&fixtures[i], interval),
                );
                assert!(owners[i].step(1.).is_err());
                same_state(&before, &state(&owners[i], &fixtures[i].parameters)?);
            }
        }
    }
    for e in &executors {
        assert!(e.info().replay);
        assert_eq!(e.info().direct_capture_variants, 2);
        assert_eq!(e.info().direct_capture_misses, 2);
    }
    Ok(())
}
fn full_program_ptx(e: &MixedExecutor) -> Vec<String> {
    e.programs
        .iter()
        .filter_map(|p| {
            p.downcast_ref::<Program<F32>>()
                .map(|p| p.ptx.clone())
                .or_else(|| p.downcast_ref::<Program<BF16>>().map(|p| p.ptx.clone()))
        })
        .collect()
}
#[test]
fn attention_training_runtime_bh64_full_baseline_policy_and_forced_jit_fallback() -> Result<()> {
    if !crate::ptx::mixed::tests::cuda_available() {
        return Ok(());
    }
    for (batch, forced) in [(8, false), (16, true)] {
        let fixtures = [fixture(batch), fixture(batch)];
        let mut owners = fixtures
            .iter()
            .map(|f| owner(&f.parameters))
            .collect::<Result<Vec<_>>>()?;
        let baseline = owners[0].mixed_executor_with_options(
            &fixtures[0].graph,
            MatMulPrecision::Tf32x3,
            options(false),
        )?;
        let candidate = FORCE_FALLBACK.with(|flag| {
            let old = flag.replace(forced);
            let result = owners[1].mixed_executor_with_options(
                &fixtures[1].graph,
                MatMulPrecision::Tf32x3,
                options(true),
            );
            flag.set(old);
            result
        })?;
        assert_eq!(candidate.info().attention_training_forward_regions, 0);
        assert_eq!(
            candidate.info().attention_training_forward_fallback_regions,
            4
        );
        assert_eq!(baseline.kernel_labels(), candidate.kernel_labels());
        assert_eq!(full_program_ptx(&baseline), full_program_ptx(&candidate));
        assert_eq!(
            baseline
                .kernels
                .iter()
                .map(|k| (
                    k.owner,
                    k.config.grid_dim,
                    k.config.block_dim,
                    k.config.shared_mem_bytes,
                    k.pointers.len()
                ))
                .collect::<Vec<_>>(),
            candidate
                .kernels
                .iter()
                .map(|k| (
                    k.owner,
                    k.config.grid_dim,
                    k.config.block_dim,
                    k.config.shared_mem_bytes,
                    k.pointers.len()
                ))
                .collect::<Vec<_>>()
        );
        if batch == 8 {
            assert_eq!(baseline.info().f32_bytes, candidate.info().f32_bytes);
            assert_eq!(baseline.info().bf16_bytes, candidate.info().bf16_bytes);
            assert_eq!(
                baseline.info().reused_allocations,
                candidate.info().reused_allocations
            );
        }
        let mut executors = [baseline, candidate];
        for i in 0..2 {
            owners[i].accumulate_mixed(
                &mut executors[i],
                &fixtures[i].graph,
                HashMap::from([("loss_seed".into(), vec![1.])]),
                indices(&fixtures[i], 0),
            )?;
            owners[i].step(1.)?;
        }
        bits(
            &executors[0].read(fixtures[0].loss)?,
            &executors[1].read(fixtures[1].loss)?,
        );
        same_state(
            &state(&owners[0], &fixtures[0].parameters)?,
            &state(&owners[1], &fixtures[1].parameters)?,
        );
    }
    Ok(())
}
#[test]
fn attention_training_runtime_actual_four_layer_b16_b32_activation() -> Result<()> {
    if !crate::ptx::mixed::tests::cuda_available() {
        return Ok(());
    }
    use crate::ptx::mixed::test_decoder::{Config, synthetic};
    for batch in [16, 32] {
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
            .map(|(_, p)| p.parameter.clone())
            .collect::<Vec<_>>();
        let owner = owner(&parameters)?;
        let e =
            owner.mixed_executor_with_options(&graph, MatMulPrecision::Tf32x3, options(true))?;
        assert_eq!(
            e.info().attention_training_forward_regions,
            4,
            "B{batch}: {:?}",
            e.info()
        );
        assert_eq!(e.info().attention_training_forward_fallback_regions, 0);
        assert_eq!(e.info().layer_norm_pair_forward_regions, 9);
        assert_eq!(e.info().layer_norm_pair_backward_regions, 9);
        assert_eq!(e.info().attention_softmax_regions, 0);
        assert_eq!(e.info().attention_softmax_backward_regions, 4);
        assert_eq!(e.info().attention_training_forward_materializations, 0);
        audit(&e, &format!("actual-b{batch}"))?;
    }
    Ok(())
}
