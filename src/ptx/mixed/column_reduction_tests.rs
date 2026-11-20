use super::*;
use crate::{
    tensor::{Parameter, TensorExpr},
    tile::ColumnReductionPolicy as Policy,
};

fn same(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) {
        if b.is_nan() {
            assert!(a.is_nan());
        } else {
            assert_eq!(a.to_bits(), b.to_bits(), "{a:?} vs {b:?}");
        }
    }
}

#[test]
fn column_policy_mixed_f32_affine_and_bf16_bias_gradients_replay() -> Result<()> {
    if !super::tests::cuda_available() {
        return Ok(());
    }
    for bf16 in [false, true] {
        let columns = 9;
        let rows = 257;
        let weight = Parameter::new(
            vec![0.125f32; if bf16 { 16 * columns } else { columns }],
            if bf16 {
                vec![16, columns]
            } else {
                vec![columns]
            },
        );
        let bias = Parameter::new(vec![0.03125f32; columns], vec![columns]);
        let input_shape = if bf16 {
            vec![rows, 16]
        } else {
            vec![2, rows, columns]
        };
        let input = TensorExpr::input("x", input_shape.clone());
        let output = if bf16 {
            input
                .linear_bf16(weight.clone().into(), bias.clone().into())?
                .cast(crate::tensor::ValueDType::F32)
        } else {
            input
                * TensorExpr::from(weight.clone())
                    .reshape(vec![1, 1, columns])
                    .broadcast(vec![2, rows, columns])
                + TensorExpr::from(bias.clone())
                    .reshape(vec![1, 1, columns])
                    .broadcast(vec![2, rows, columns])
        };
        let output_shape = output.shape().clone();
        let mut graph = TensorGraph::new();
        let forward = graph.add_expr(&output, false);
        let loss = graph.add_expr(
            &(output * TensorExpr::input("seed", output_shape.clone())).mean_all(),
            false,
        );
        let graph = graph.with_gradients(loss);
        for policy in [Policy::Four, Policy::Eight] {
            let mut baseline = MixedExecutor::new_with_options(
                &graph,
                MatMulPrecision::NativeHalf,
                true,
                MixedOptions::default(),
            )?;
            let mut candidate = MixedExecutor::new_with_options(
                &graph,
                MatMulPrecision::NativeHalf,
                true,
                MixedOptions {
                    column_reduction_policy: policy,
                    ..MixedOptions::default()
                },
            )?;
            let counts = (
                candidate.info().column_reduction_four_regions,
                candidate.info().column_reduction_eight_regions,
            );
            assert!(
                if policy == Policy::Four {
                    counts.0 > 0 && counts.1 == 0
                } else {
                    counts.1 > 0 && counts.0 == 0
                },
                "actual activation: {counts:?}"
            );
            assert_eq!(baseline.info().kernels, candidate.info().kernels);
            assert_eq!(baseline.info().f32_bytes, candidate.info().f32_bytes);
            assert_eq!(baseline.info().bf16_bytes, candidate.info().bf16_bytes);
            for phase in 0..3usize {
                let feeds = HashMap::from([
                    (
                        "x".into(),
                        (0..input_shape.iter().product::<usize>())
                            .map(|i| ((i + phase * 3) % 29) as f32 / 64. - 0.125)
                            .collect(),
                    ),
                    (
                        "seed".into(),
                        (0..output_shape.iter().product::<usize>())
                            .map(|i| ((i + phase) % 7) as f32 / 16. - 0.125)
                            .collect(),
                    ),
                ]);
                baseline.upload(feeds.clone())?;
                candidate.upload(feeds)?;
                baseline.run()?;
                candidate.run()?;
                for node in [forward, loss] {
                    let expected = baseline.read(node)?;
                    assert!(expected.iter().all(|v| v.is_finite()));
                    same(&candidate.read(node)?, &expected);
                }
                let a = baseline.gradients(&graph)?;
                let b = candidate.gradients(&graph)?;
                for id in [weight.id(), bias.id()] {
                    same(&b[&id], &a[&id]);
                }
            }
            assert!(
                candidate.captured.is_some(),
                "supported mixed replay must remain captured"
            );
        }
    }
    Ok(())
}

#[test]
fn column_policy_nonfinite_classes_signed_zero_and_fallbacks() -> Result<()> {
    if !super::tests::cuda_available() {
        return Ok(());
    }
    for (shape, axis, eligible) in [
        (vec![2, 257, 9], 1, true),
        (vec![2, 129, 9], 1, false),
        (vec![2, 257, 9], 2, false),
        (vec![32, 1, 9], 0, false),
    ] {
        let graph: TensorGraph<f32> = TensorExpr::input("x", shape.clone())
            .reduce_sum(axis)
            .into();
        for policy in [Policy::Off, Policy::Four, Policy::Eight] {
            let mut candidate = MixedExecutor::new_with_options(
                &graph,
                MatMulPrecision::NativeHalf,
                false,
                MixedOptions {
                    column_reduction_policy: policy,
                    ..MixedOptions::default()
                },
            )?;
            let active = candidate.info().column_reduction_four_regions
                + candidate.info().column_reduction_eight_regions;
            assert_eq!(active > 0, eligible && policy != Policy::Off);
            let mut baseline = MixedExecutor::new_with_options(
                &graph,
                MatMulPrecision::NativeHalf,
                false,
                MixedOptions::default(),
            )?;
            let root = graph.graph.node_indices().next_back().expect("root");
            let count = shape.iter().product::<usize>();
            for special in [-0.0f32, f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
                let mut values = vec![special; count];
                if special.is_infinite() && count > 9 {
                    values[9] = -special;
                }
                let feeds = HashMap::from([("x".into(), values)]);
                baseline.upload(feeds.clone())?;
                candidate.upload(feeds)?;
                baseline.run()?;
                candidate.run()?;
                same(&candidate.read(root)?, &baseline.read(root)?);
            }
        }
    }
    Ok(())
}

#[test]
fn column_same_function_128_thread_resource_fallback_matches_original() -> Result<()> {
    if !super::tests::cuda_available() {
        return Ok(());
    }
    for policy in [Policy::Four, Policy::Eight] {
        let graph: TensorGraph<f32> = (TensorExpr::input("x", vec![2, 257, 9])
            * TensorExpr::input("y", vec![2, 257, 9]))
        .reduce_sum(1)
        .into();
        let mut baseline = MixedExecutor::new_with_options(
            &graph,
            MatMulPrecision::NativeHalf,
            false,
            MixedOptions::default(),
        )?;
        let mut candidate = MixedExecutor::new_with_options(
            &graph,
            MatMulPrecision::NativeHalf,
            false,
            MixedOptions {
                column_reduction_policy: policy,
                ..MixedOptions::default()
            },
        )?;
        if let Some(directory) = std::env::var_os("TNSR_COLUMN_PTX_DIR") {
            let directory = std::path::PathBuf::from(directory);
            std::fs::create_dir_all(&directory)?;
            let mut resources = Vec::new();
            for (name, executor) in [("old", &baseline), ("column", &candidate)] {
                for (index, program) in executor
                    .programs
                    .iter()
                    .filter_map(|p| p.downcast_ref::<Program<F32>>())
                    .enumerate()
                {
                    std::fs::write(
                        directory.join(format!("{policy:?}-{name}-{index}.ptx")),
                        &program.ptx,
                    )?;
                }
                for kernel in &executor.kernels {
                    resources.push(serde_json::json!({"arm":name,"label":kernel.label,"block":kernel.config.block_dim,"grid":kernel.config.grid_dim,"registers":kernel.function.num_regs()?,"shared_bytes":kernel.function.shared_size_bytes()?,"local_bytes":kernel.function.local_size_bytes()?,"max_threads":kernel.function.max_threads_per_block()?}));
                }
            }
            std::fs::write(
                directory.join(format!("{policy:?}-resources.json")),
                serde_json::to_vec_pretty(&resources)?,
            )?;
        }
        let mut forced = 0;
        for kernel in &mut candidate.kernels {
            if kernel.label.contains("column_tile=") {
                kernel.config.block_dim = (128, 1, 1);
                kernel.config.grid_dim = (18, 1, 1);
                forced += 1;
            }
        }
        assert_eq!(
            forced, 1,
            "must exercise the retained original body of a column function"
        );
        let feeds = HashMap::from([
            (
                "x".into(),
                (0..2 * 257 * 9)
                    .map(|i| (i % 31) as f32 / 16. - 0.75)
                    .collect(),
            ),
            (
                "y".into(),
                (0..2 * 257 * 9)
                    .map(|i| (i % 17) as f32 / 32. - 0.25)
                    .collect(),
            ),
        ]);
        baseline.upload(feeds.clone())?;
        candidate.upload(feeds)?;
        baseline.run()?;
        candidate.run()?;
        let root = graph.graph.node_indices().next_back().expect("root");
        same(&candidate.read(root)?, &baseline.read(root)?);
    }
    Ok(())
}

#[test]
fn column_policy_actual_four_layer_decoder_activation_counts() -> Result<()> {
    use super::test_decoder::{Config, synthetic};
    if !super::tests::cuda_available() {
        return Ok(());
    }
    for sequence in [128, 512] {
        let model = synthetic(Config {
            width: 512,
            layers: 4,
            heads: 8,
            hidden: 1408,
            sequence,
            batch: 1,
            vocab: 4096,
            epsilon: 1e-5,
            learning_rate: 1e-4,
            max_grad_norm: 1.,
        });
        let (graph, _) = model.training_graph_with_options(crate::graph::GradientOptions {
            compact_softmax: true,
        })?;
        for policy in [Policy::Four, Policy::Eight] {
            // Same accepted graph/codegen policies as the benchmark. Resident
            // allocation/Adam choices do not participate in local eligibility.
            let executor = MixedExecutor::new_with_options(
                &graph,
                MatMulPrecision::NativeHalf,
                false,
                MixedOptions {
                    column_reduction_policy: policy,
                    physical_order_casts: true,
                    shared_memory_reuse: true,
                    narrow_bf16_tile: true,
                    transposed_bf16_outputs: true,
                    medium_bf16_tile_policy: crate::tile::MediumBf16TilePolicy::SquareThenTall,
                    attention_softmax_fusion: true,
                    attention_softmax_backward_fusion: true,
                    qkv_grad_pack_fusion: true,
                    swiglu_pair_fusion: true,
                    ..MixedOptions::default()
                },
            )?;
            let f32_regions: usize = executor
                .programs
                .iter()
                .filter_map(|p| p.downcast_ref::<Program<F32>>())
                .map(|p| p.column_reduction_counts.0 + p.column_reduction_counts.1)
                .sum();
            let bf16_regions: usize = executor
                .programs
                .iter()
                .filter_map(|p| p.downcast_ref::<Program<BF16>>())
                .map(|p| p.column_reduction_counts.0 + p.column_reduction_counts.1)
                .sum();
            eprintln!(
                "actual four-layer W512 B1 S{sequence} {policy:?}: F32={f32_regions}, BF16={bf16_regions}, total={}",
                f32_regions + bf16_regions
            );
            assert_eq!(
                executor.info().column_reduction_four_regions
                    + executor.info().column_reduction_eight_regions,
                f32_regions + bf16_regions
            );
            if sequence == 128 {
                assert_eq!((f32_regions, bf16_regions), (0, 0));
            } else {
                assert_eq!(
                    (f32_regions, bf16_regions),
                    (18, 20),
                    "actual compiled regions must match the decoder affine/bias topology"
                );
            }
        }
    }
    Ok(())
}

/// Structural B32 eligibility, without allocating B32 activation storage. Each
/// real graph anchor is checked through a typed isolated region at the already
/// materialized row-major boundary. Full B32 compiled activation still belongs
/// to the complete-model campaign, rather than this CPU fixture.
#[test]
fn column_actual_b32_four_layer_graph_boundary_shape_counts_without_gpu() -> Result<()> {
    use super::test_decoder::{Config, synthetic};
    use crate::tensor::ValueDType;
    let model = synthetic(Config {
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
    let dtypes = value_dtypes(&graph);
    for policy in [Policy::Four, Policy::Eight] {
        let mut counts = (0, 0);
        for node in graph.graph.node_indices() {
            let TensorGraphNode::ReduceAxis {
                op: crate::tensor::ReduceOp::Sum,
                axis,
                ..
            } = graph[node]
            else {
                continue;
            };
            let input_shape = graph[graph.inputs(node)[0]].shape().clone();
            let dtype = dtypes.get(&node).copied().unwrap_or(ValueDType::F32);
            let mut plan = if dtype == ValueDType::F32 && axis == 1 && input_shape.len() == 3 {
                let single: TensorGraph<f32> = TensorExpr::input("physical", input_shape)
                    .reduce_sum(axis)
                    .into();
                crate::ptx::PtxExecutionPlan::build_with_reduction_mode(
                    &single,
                    crate::ptx::PtxReductionMode::DeterministicTree,
                )?
            } else if dtype == ValueDType::BF16 && axis == 0 && input_shape.len() == 2 {
                let single: TensorGraph<half::bf16> = TensorExpr::input("physical", input_shape)
                    .reduce_sum(axis)
                    .into();
                crate::ptx::PtxExecutionPlan::build_with_reduction_mode(
                    &single,
                    crate::ptx::PtxReductionMode::DeterministicTree,
                )?
            } else {
                continue;
            };
            plan.set_column_reduction_policy(policy, 49152);
            let selected = plan.column_reduction_counts();
            if selected.0 + selected.1 > 0 {
                if dtype == ValueDType::F32 {
                    counts.0 += 1;
                } else {
                    counts.1 += 1;
                }
            }
        }
        eprintln!(
            "actual four-layer W512 B32 S512 {policy:?} boundary shape eligibility F32/BF16={counts:?}; this is not a compiled-resource activation claim"
        );
        assert_eq!(counts, (18, 20));
    }
    Ok(())
}
