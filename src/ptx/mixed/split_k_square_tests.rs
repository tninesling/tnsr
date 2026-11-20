//! Actual special-path activation, rounding-boundary and replay tests.
use super::*;
use crate::{
    tensor::{Parameter, TensorExpr},
    tile::SplitKBf16TilePolicy as Policy,
};

#[test]
fn split_square64_actual_mixed_allgrad_replay_and_bf16_boundaries() -> Result<()> {
    if !super::tests::cuda_available() {
        return Ok(());
    }
    let (rows, k, n) = (2048, 512, 512);
    let x = Parameter::new(
        (0..rows * k)
            .map(|i| (i % 17) as f32 / 64. - 0.125)
            .collect(),
        vec![rows, k],
    );
    let weight = Parameter::new(
        (0..k * n).map(|i| (i % 13) as f32 / 32. - 0.0625).collect(),
        vec![n, k],
    );
    let bias = Parameter::new(vec![0.03125; n], vec![n]);
    let output = (TensorExpr::from(x.clone()) + TensorExpr::input("delta", vec![rows, k]))
        .linear_bf16(
            TensorExpr::from(weight.clone()).transpose(),
            bias.clone().into(),
        )?
        .cast(crate::tensor::ValueDType::F32);
    let mut graph = TensorGraph::new();
    let forward = graph.add_expr(&output, false);
    let loss = graph.add_expr(
        &(output * TensorExpr::input("seed", vec![rows, n])).mean_all(),
        false,
    );
    let graph = graph.with_gradients(loss);
    let options = MixedOptions {
        shared_memory_reuse: true,
        physical_order_casts: true,
        ..MixedOptions::default()
    };
    let mut old =
        MixedExecutor::new_with_options(&graph, MatMulPrecision::NativeHalf, true, options)?;
    let mut new = MixedExecutor::new_with_options(
        &graph,
        MatMulPrecision::NativeHalf,
        true,
        MixedOptions {
            split_k_bf16_tile_policy: Policy::Square64,
            ..options
        },
    )?;
    assert!(
        new.info().split_k_square64_programs > 0,
        "actual dW must activate, not merely request policy"
    );
    assert_eq!(old.info().kernels, new.info().kernels);
    assert_eq!(old.info().f32_bytes, new.info().f32_bytes);
    assert_eq!(old.info().bf16_bytes, new.info().bf16_bytes);
    for phase in 0..3 {
        let feeds = HashMap::from([
            (
                "delta".into(),
                (0..rows * k)
                    .map(|i| ((i + phase) % 7) as f32 / 128.)
                    .collect(),
            ),
            (
                "seed".into(),
                (0..rows * n)
                    .map(|i| ((i + phase) % 5) as f32 / 16. - 0.125)
                    .collect(),
            ),
        ]);
        old.upload(feeds.clone())?;
        new.upload(feeds)?;
        old.run()?;
        new.run()?;
        for node in [forward, loss] {
            assert_eq!(
                old.read(node)?
                    .into_iter()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>(),
                new.read(node)?
                    .into_iter()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>()
            );
        }
        let a = old.gradients(&graph)?;
        let b = new.gradients(&graph)?;
        for id in [x.id(), weight.id(), bias.id()] {
            assert_eq!(
                a[&id].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                b[&id].iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
    }
    assert!(old.captured.is_some());
    assert!(new.captured.is_some());
    Ok(())
}

// Still mandatory at integration: old64²case actualcounter0; whole-map/small
// grid/reuse/parts/tails fallbacks; resident incremental lifetime reuse and
// invalid-update recovery; unchanged first-step loss/gradient/weight/update
// gates, actual region/copy/kernel/pool counters, target-derived PTX/SASS and
// JIT register/local/shared evidence. Never expand tolerances to rescue a miss.

#[test]
fn split_square64_small_grid_and_reuse_disabled_keep_baseline() -> Result<()> {
    if !super::tests::cuda_available() {
        return Ok(());
    }
    // These full physical inputs have too few candidate CTAs. The policy
    // must neither change the original packing decision nor claim activation.
    for reuse in [false, true] {
        let (m, k, n) = (128, 2048, 128);
        let a = TensorExpr::input("a", vec![m, k]);
        let b = TensorExpr::input("b", vec![k, n]);
        let mut graph = TensorGraph::new();
        let output = graph.add_expr(
            &a.cast(ValueDType::BF16)
                .matmul(b.cast(ValueDType::BF16))
                .cast(ValueDType::F32),
            false,
        );
        let options = MixedOptions {
            shared_memory_reuse: reuse,
            split_k_bf16_tile_policy: Policy::Square64,
            ..MixedOptions::default()
        };
        let mut candidate =
            MixedExecutor::new_with_options(&graph, MatMulPrecision::NativeHalf, true, options)?;
        assert_eq!(candidate.info().split_k_square64_programs, 0);
        assert_eq!(candidate.info().split_k_square64_fallback_programs, 1);
        let mut baseline = MixedExecutor::new_with_options(
            &graph,
            MatMulPrecision::NativeHalf,
            true,
            MixedOptions {
                split_k_bf16_tile_policy: Policy::Off,
                ..options
            },
        )?;
        assert_eq!(candidate.info().kernels, baseline.info().kernels);
        let feeds = HashMap::from([
            ("a".into(), vec![0.125; m * k]),
            ("b".into(), vec![0.25; k * n]),
        ]);
        candidate.upload(feeds.clone())?;
        baseline.upload(feeds)?;
        candidate.run()?;
        baseline.run()?;
        assert_eq!(
            candidate
                .read(output)?
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            baseline
                .read(output)?
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>()
        );
    }
    Ok(())
}

#[test]
fn split_square64_resident_incremental_replay_state_and_invalid_recovery() -> Result<()> {
    use crate::ptx::resident::{AdamGroup, ResidentAdam};
    if !super::tests::cuda_available() {
        return Ok(());
    }
    let (rows, k, n) = (2048, 512, 512);
    let mut parameters = Vec::new();
    let mut graphs = Vec::new();
    for _ in 0..2 {
        let x = Parameter::new(
            (0..rows * k)
                .map(|i| (i % 17) as f32 / 64. - 0.125)
                .collect(),
            vec![rows, k],
        );
        let w = Parameter::new(
            (0..k * n).map(|i| (i % 13) as f32 / 32. - 0.0625).collect(),
            vec![n, k],
        );
        let bias = Parameter::new(vec![0.03125; n], vec![n]);
        let y = (TensorExpr::from(x.clone()) + TensorExpr::input("delta", vec![rows, k]))
            .linear_bf16(TensorExpr::from(w.clone()).transpose(), bias.clone().into())?
            .cast(ValueDType::F32);
        let mut graph = TensorGraph::new();
        let loss = graph.add_expr(
            &(y * TensorExpr::input("seed", vec![rows, n])).mean_all(),
            false,
        );
        graphs.push((graph.with_gradients(loss), loss));
        parameters.push([x, w, bias]);
    }
    let mut owners = parameters
        .iter()
        .map(|p| {
            ResidentAdam::new_with_graph_replay(
                p.iter().cloned().map(|p| (p, 0)).collect(),
                vec![AdamGroup::new(1e-4)],
            )
        })
        .collect::<Result<Vec<_>>>()?;
    for o in &mut owners {
        o.set_update_batch_size(8)?;
        o.set_experimental_first_accumulation_overwrite(true)?;
        o.set_experimental_gradient_norm_partials(true)?;
        o.begin_step()?;
    }
    let mut executors = owners
        .iter()
        .enumerate()
        .map(|(i, o)| {
            o.mixed_executor_with_options(
                &graphs[i].0,
                MatMulPrecision::Tf32x3,
                MixedOptions {
                    split_k_bf16_tile_policy: if i == 0 {
                        Policy::Off
                    } else {
                        Policy::Square64
                    },
                    shared_memory_reuse: true,
                    physical_order_casts: true,
                    incremental_allocation_reuse: true,
                    direct_master_bindings: true,
                    ..MixedOptions::default()
                },
            )
        })
        .collect::<Result<Vec<_>>>()?;
    assert_eq!(executors[0].info().split_k_square64_programs, 0);
    assert!(executors[1].info().split_k_square64_programs > 0);
    let bits = |v: Vec<f32>| v.into_iter().map(f32::to_bits).collect::<Vec<_>>();
    for (phase, microbatches) in [1, 2, 1].into_iter().enumerate() {
        for i in 0..2 {
            if phase > 0 {
                owners[i].begin_step()?;
            }
            for micro in 0..microbatches {
                let feeds = HashMap::from([
                    (
                        "delta".into(),
                        vec![(phase + micro) as f32 / 128.; rows * k],
                    ),
                    (
                        "seed".into(),
                        (0..rows * n)
                            .map(|j| ((j + phase + micro) % 5) as f32 / 16. - 0.125)
                            .collect(),
                    ),
                ]);
                owners[i].accumulate_mixed(
                    &mut executors[i],
                    &graphs[i].0,
                    feeds,
                    HashMap::new(),
                )?;
            }
        }
        let a = owners[0].step(1.)?;
        let b = owners[1].step(1.)?;
        assert_eq!(a.timestep, b.timestep);
        assert_eq!(a.gradient_norm.to_bits(), b.gradient_norm.to_bits());
        assert_eq!(a.gradient_scale.to_bits(), b.gradient_scale.to_bits());
        assert_eq!(
            bits(executors[0].read(graphs[0].1)?),
            bits(executors[1].read(graphs[1].1)?)
        );
        let grads = [owners[0].gradients()?, owners[1].gradients()?];
        let moments = [owners[0].moments()?, owners[1].moments()?];
        for j in 0..3 {
            let (a, b) = (&parameters[0][j], &parameters[1][j]);
            assert_eq!(
                bits(grads[0][&a.id()].clone()),
                bits(grads[1][&b.id()].clone())
            );
            assert_eq!(bits(a.values()?), bits(b.values()?));
            assert_eq!(
                bits(moments[0][&a.id()].first.clone()),
                bits(moments[1][&b.id()].first.clone())
            );
            assert_eq!(
                bits(moments[0][&a.id()].second.clone()),
                bits(moments[1][&b.id()].second.clone())
            );
        }
        if phase == 1 {
            for i in 0..2 {
                let before = parameters[i]
                    .iter()
                    .map(|p| p.values())
                    .collect::<Result<Vec<_>>>()?;
                owners[i].begin_step()?;
                let _ = owners[i].accumulate_mixed(
                    &mut executors[i],
                    &graphs[i].0,
                    HashMap::from([
                        ("delta".into(), vec![0.; rows * k]),
                        ("seed".into(), vec![f32::NAN; rows * n]),
                    ]),
                    HashMap::new(),
                );
                assert!(
                    owners[i].step(1.).is_err(),
                    "invalid gradients must not publish an update"
                );
                for (p, old) in parameters[i].iter().zip(before) {
                    assert_eq!(bits(p.values()?), bits(old));
                }
            }
        }
    }
    for e in &executors {
        assert!(e.info().replay);
        assert_eq!(e.info().direct_capture_variants, 2);
        assert!(e.info().direct_capture_hits > 0);
    }
    Ok(())
}

#[test]
fn split_square64_actual_b32_decoder_compiled_activation_and_resources() -> Result<()> {
    if !super::tests::cuda_available() {
        return Ok(());
    }
    let model = super::test_decoder::synthetic(super::test_decoder::Config {
        width: 512,
        layers: 4,
        heads: 8,
        hidden: 1408,
        sequence: 512,
        batch: 32,
        vocab: 4096,
        epsilon: 1e-5,
        learning_rate: 1e-4,
        max_grad_norm: 1.,
    });
    let (graph, _) = model.training_graph_with_options(crate::graph::GradientOptions {
        compact_softmax: true,
    })?;
    let mut owner = crate::ptx::resident::ResidentAdam::new_with_graph_replay(
        model
            .parameters
            .iter()
            .map(|(_, p)| (p.parameter.clone(), 0))
            .collect(),
        vec![crate::ptx::resident::AdamGroup::new(1e-4)],
    )?;
    owner.begin_step()?;
    let executor = owner.mixed_executor_with_options(
        &graph,
        MatMulPrecision::Tf32x3,
        MixedOptions {
            split_k_bf16_tile_policy: Policy::Square64,
            shared_memory_reuse: true,
            physical_order_casts: true,
            incremental_allocation_reuse: true,
            direct_master_bindings: true,
            transposed_bf16_outputs: true,
            narrow_bf16_tile: true,
            medium_bf16_tile_policy: crate::tile::MediumBf16TilePolicy::SquareThenTall,
            column_reduction_policy: crate::tile::ColumnReductionPolicy::Four,
            batched_attention_bf16_tile_policy:
                crate::tile::BatchedAttentionBf16TilePolicy::TallScores,
            attention_softmax_fusion: true,
            attention_softmax_backward_fusion: true,
            swiglu_pair_fusion: true,
            qkv_grad_pack_fusion: true,
            layer_norm_pair_fusion: true,
            ..MixedOptions::default()
        },
    )?;
    let selected = executor
        .kernels
        .iter()
        .filter(|k| k.label.contains("square64=true"))
        .collect::<Vec<_>>();
    assert!(
        !selected.is_empty(),
        "actual B32 decoder must not be a no-op"
    );
    assert_eq!(selected.len(), executor.info().split_k_square64_programs);
    let resources=selected.iter().map(|k| {
        assert_eq!(k.config.block_dim,(32,8,1));
        assert_eq!(k.function.shared_size_bytes()?,18432);assert_eq!(k.function.local_size_bytes()?,0);assert!(k.function.num_regs()?<=128);
        let capacity=k.function.occupancy_max_active_blocks_per_multiprocessor(256,0,None)?;assert!(capacity>=2);
        Ok(serde_json::json!({"label":k.label,"grid":k.config.grid_dim,"block":k.config.block_dim,"registers":k.function.num_regs()?,"shared":k.function.shared_size_bytes()?,"local":k.function.local_size_bytes()?,"api_max_active_blocks_per_sm":capacity}))
    }).collect::<Result<Vec<_>>>()?;
    eprintln!(
        "SplitSquare actual B32 selected={} fallback={} resources={resources:?}",
        executor.info().split_k_square64_programs,
        executor.info().split_k_square64_fallback_programs
    );
    if let Some(directory) = std::env::var_os("TNSR_SPLIT_K_SQUARE_PTX_DIR") {
        let directory = std::path::PathBuf::from(directory);
        std::fs::create_dir_all(&directory)?;
        std::fs::write(
            directory.join("actual-b32-activation.json"),
            serde_json::to_vec_pretty(
                &serde_json::json!({"info":executor.info(),"selected_partial_resources":resources,"scope":"actual compiled B32 four-layer decoder; no throughput measurement or full-model numerical comparison in this test"}),
            )?,
        )?;
    }
    Ok(())
}
