use super::*;
use crate::{
    Executor, SimpleExecutor,
    tensor::{Parameter, TensorExpr},
};

pub(super) fn cuda_available() -> bool {
    let available = CudaContext::new(0).is_ok();
    assert!(
        available || std::env::var_os("TNSR_REQUIRE_CUDA").is_none(),
        "CUDA required"
    );
    available
}

fn cast_metadata_source(shape: Vec<usize>) -> Value {
    Value {
        binding: Binding {
            pointer: 0,
            len: shape.iter().product(),
            dtype: DType::F32,
        },
        view: VirtualTensor::identity(NodeIndex::new(0), shape.clone(), DType::F32),
        source_shape: shape,
        chain: Vec::new(),
    }
}

#[test]
fn physical_order_cast_metadata_preserves_bijective_view_rounding() {
    let mut cases = Vec::new();
    for shape in [vec![3, 5], vec![17, 33], vec![32, 64]] {
        let mut source = cast_metadata_source(shape);
        source.view = source.view.permute(&[1, 0]).unwrap();
        source.chain.push(TensorGraphNode::Transpose {
            shape: source.view.shape.clone(),
        });
        cases.push(source);
    }
    let mut composed = cast_metadata_source(vec![2, 3, 5]);
    composed.view = composed.view.permute(&[1, 0, 2]).unwrap();
    composed.chain.push(TensorGraphNode::Permute {
        axes: vec![1, 0, 2],
        shape: composed.view.shape.clone(),
    });
    composed.view = composed.view.reshape(vec![6, 5]).unwrap();
    composed.chain.push(TensorGraphNode::Reshape {
        shape: composed.view.shape.clone(),
    });
    composed.view = composed.view.permute(&[1, 0]).unwrap();
    composed.chain.push(TensorGraphNode::Transpose {
        shape: composed.view.shape.clone(),
    });
    cases.push(composed);
    let pattern = [
        0.,
        -0.,
        1. + 1. / 256.,
        1. + 3. / 256.,
        -1. - 1. / 256.,
        f32::MAX,
        f32::MIN_POSITIVE,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(0x7fc1_2345),
        f32::from_bits(1),
    ];
    for source in cases {
        let physical = physical_order_cast_source(&source, DType::BF16).unwrap();
        assert_eq!(physical.view.shape, source.source_shape);
        assert_eq!(
            physical.view.access,
            crate::tile::IndexMap::identity(source.source_shape.len())
        );
        assert!(physical.chain.is_empty());
        let target = physical_order_cast_result(
            &source,
            NodeIndex::new(7),
            Binding {
                pointer: 2,
                len: source.binding.len,
                dtype: DType::BF16,
            },
        );
        assert_eq!(target.view.source, NodeIndex::new(7));
        assert_eq!(target.view.dtype, DType::BF16);
        assert_eq!(target.view.shape, source.view.shape);
        assert_eq!(target.source_shape, source.source_shape);
        assert_eq!(target.chain.len(), source.chain.len());
        assert!(
            target
                .chain
                .iter()
                .zip(&source.chain)
                .all(|(a, b)| a.name() == b.name() && a.shape() == b.shape())
        );
        let physical_values: Vec<_> = (0..source.binding.len)
            .map(|i| half::bf16::from_f32(pattern[i % pattern.len()]))
            .collect();
        for linear in 0..source.binding.len {
            let mut remaining = linear;
            let mut point = vec![0; source.view.shape.len()];
            for (coordinate, &extent) in point.iter_mut().zip(&source.view.shape).rev() {
                *coordinate = remaining % extent;
                remaining /= extent;
            }
            let original = source.view.source_index(&point).unwrap();
            let index = original
                .iter()
                .zip(&source.source_shape)
                .fold(0, |index, (&i, &extent)| index * extent + i);
            let after = target.view.source_index(&point).unwrap();
            assert_eq!(after, original);
            let expected = half::bf16::from_f32(pattern[index % pattern.len()]);
            assert_eq!(physical_values[index].to_bits(), expected.to_bits());
        }
    }
}

#[test]
fn physical_order_cast_rejects_broadcast_masks_and_unproven_maps() {
    use crate::tile::{IndexExpr, IndexPredicate, access::ScalarValue};
    assert!(!MixedOptions::default().physical_order_casts);
    let mut source = cast_metadata_source(vec![3, 5]);
    source.view = source.view.permute(&[1, 0]).unwrap();
    source.chain.push(TensorGraphNode::Transpose {
        shape: source.view.shape.clone(),
    });
    assert!(physical_order_cast_source(&source, DType::BF16).is_some());
    assert!(physical_order_cast_source(&source, DType::F32).is_none());
    let mut masked = source.clone();
    masked.view.predicate = Some(IndexPredicate::And(Vec::new()));
    assert!(physical_order_cast_source(&masked, DType::BF16).is_none());
    let mut padded = source.clone();
    padded.view.out_of_bounds = Some(ScalarValue::F32(0));
    assert!(physical_order_cast_source(&padded, DType::BF16).is_none());
    let mut wrong_map = source.clone();
    wrong_map.view.access.results[0] = IndexExpr::Const(0);
    assert!(physical_order_cast_source(&wrong_map, DType::BF16).is_none());
    let mut wrong_extent = source.clone();
    wrong_extent.binding.len += 1;
    assert!(physical_order_cast_source(&wrong_extent, DType::BF16).is_none());
    let mut wrong_chain = source.clone();
    wrong_chain.chain.clear();
    assert!(physical_order_cast_source(&wrong_chain, DType::BF16).is_none());
    let mut broadcast = cast_metadata_source(vec![1, 5]);
    broadcast.view = broadcast
        .view
        .broadcast_axis(0, 3)
        .unwrap()
        .permute(&[1, 0])
        .unwrap();
    broadcast.chain.push(TensorGraphNode::BroadcastAxis {
        axis: 0,
        shape: vec![3, 5],
    });
    broadcast
        .chain
        .push(TensorGraphNode::Transpose { shape: vec![5, 3] });
    assert!(physical_order_cast_source(&broadcast, DType::BF16).is_none());
}

#[test]
fn physical_order_casts_match_rectangular_views_with_replay_and_changed_feeds() {
    if !cuda_available() {
        return;
    }
    for (rows, columns) in [(3, 5), (17, 33), (32, 64)] {
        let input = TensorExpr::input("source", vec![rows, columns]);
        for broadcast in [false, true] {
            let view = if broadcast {
                input
                    .clone()
                    .reshape(vec![1, rows * columns])
                    .broadcast_axis(0, 2)
                    .transpose()
            } else {
                input.clone().transpose()
            };
            let mut graph = TensorGraph::new();
            let out = graph.add_expr(&view.cast(ValueDType::BF16).cast(ValueDType::F32), false);
            for replay in [false, true] {
                let mut baseline =
                    MixedExecutor::new(&graph, MatMulPrecision::NativeHalf, replay).unwrap();
                let mut candidate = MixedExecutor::new_with_options(
                    &graph,
                    MatMulPrecision::NativeHalf,
                    replay,
                    MixedOptions {
                        physical_order_casts: true,
                        ..MixedOptions::default()
                    },
                )
                .unwrap();
                assert_eq!(baseline.info().physical_order_cast_materializations, 0);
                assert_eq!(
                    candidate.info().physical_order_cast_materializations,
                    usize::from(!broadcast)
                );
                assert_eq!(candidate.info().kernels, baseline.info().kernels);
                for revision in 0..2 {
                    let data = (0..rows * columns)
                        .map(|i| ((i + revision) % 31) as f32 * 0.03125 + 1. / 256.)
                        .collect();
                    let inputs = HashMap::from([("source".into(), data)]);
                    baseline.upload(inputs.clone()).unwrap();
                    candidate.upload(inputs).unwrap();
                    baseline.run().unwrap();
                    candidate.run().unwrap();
                    assert_eq!(
                        baseline
                            .read(out)
                            .unwrap()
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>(),
                        candidate
                            .read(out)
                            .unwrap()
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>()
                    );
                }
            }
        }
    }
}

#[test]
fn physical_order_weight_casts_preserve_linear_backward() {
    if !cuda_available() {
        return;
    }
    for (rows, input, output) in [(3, 7, 5), (32, 64, 96)] {
        let weight = Parameter::new(
            (0..output * input)
                .map(|i| (i % 13) as f32 / 32. - 0.125)
                .collect(),
            vec![output, input],
        );
        let bias = Parameter::new(vec![0.03125; output], vec![output]);
        let loss = TensorExpr::input("x", vec![rows, input])
            .linear_bf16(
                TensorExpr::from(weight.clone()).transpose(),
                bias.clone().into(),
            )
            .unwrap()
            .cast(ValueDType::F32)
            .mean_all();
        let mut graph = TensorGraph::new();
        let loss = graph.add_expr(&loss, false);
        let graph = graph.with_gradients(loss);
        let mut baseline = MixedExecutor::new(&graph, MatMulPrecision::NativeHalf, true).unwrap();
        let mut candidate = MixedExecutor::new_with_options(
            &graph,
            MatMulPrecision::NativeHalf,
            true,
            MixedOptions {
                physical_order_casts: true,
                ..MixedOptions::default()
            },
        )
        .unwrap();
        assert!(candidate.info().physical_order_cast_materializations > 0);
        for revision in 0..2 {
            let data = (0..rows * input)
                .map(|i| ((i + revision) % 17) as f32 / 16. - 0.25)
                .collect();
            let inputs = HashMap::from([("x".into(), data)]);
            baseline.upload(inputs.clone()).unwrap();
            candidate.upload(inputs).unwrap();
            baseline.run().unwrap();
            candidate.run().unwrap();
            assert_eq!(baseline.read(loss).unwrap(), candidate.read(loss).unwrap());
            let expected = baseline.gradients(&graph).unwrap();
            let actual = candidate.gradients(&graph).unwrap();
            for parameter in [weight.id(), bias.id()] {
                assert_eq!(
                    expected[&parameter]
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    actual[&parameter]
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>()
                );
            }
        }
    }
}

#[test]
fn native_half_uses_bf16_mma_without_tf32_for_f32_regions() {
    if !cuda_available() {
        return;
    }
    let x = TensorExpr::input("x", vec![32, 32]);
    let weight: TensorExpr<f32> = Parameter::new(vec![1.0078125; 1024], vec![32, 32]).into();
    let bias: TensorExpr<f32> = Parameter::new(vec![0.125; 32], vec![32]).into();
    let mixed = x.clone().linear_bf16(weight, bias).unwrap();
    let f32_weight: TensorExpr<f32> = Parameter::new(vec![1.0001; 1024], vec![32, 32]).into();
    let plain = x.matmul(f32_weight);
    let mut graph = TensorGraph::new();
    let mixed_output = graph.add_expr(&mixed, false);
    let f32_output = graph.add_expr(&plain, false);
    let mut gpu = MixedExecutor::new(&graph, MatMulPrecision::NativeHalf, true).unwrap();
    assert!(gpu.info().bf16_tensor_core_programs > 0);
    gpu.upload(HashMap::from([("x".into(), vec![1.0078125; 1024])]))
        .unwrap();
    gpu.run().unwrap();
    // F32 accumulation and one final bias/output rounding differs from rounded
    // BF16 products (32.5) and from a separately rounded BF16 matmul (32.5).
    assert_eq!(gpu.read(mixed_output).unwrap(), vec![32.75; 1024]);
    let expected = 32. * 1.0078125_f32 * 1.0001_f32;
    for value in gpu.read(f32_output).unwrap() {
        assert!((value - expected).abs() < 1e-5);
    }
}

#[test]
fn mixed_fused_linear_rounds_only_after_bias() {
    if !cuda_available() {
        return;
    }
    let x = TensorExpr::input("x", vec![1, 1]);
    let w: TensorExpr<f32> = Parameter::new(vec![1.0078125_f32], vec![1, 1]).into();
    let b: TensorExpr<f32> = Parameter::new(vec![-1.015625_f32], vec![1]).into();
    let y = x.linear_bf16(w, b).unwrap().cast(ValueDType::F32);
    let mut graph = TensorGraph::new();
    let output = graph.add_expr(&y, true);
    let graph = graph.with_gradients(output);
    let mut gpu = MixedExecutor::new(&graph, MatMulPrecision::Tf32x3, true).unwrap();
    let mut cpu = SimpleExecutor::new();
    let inputs = HashMap::from([("x".into(), vec![1.0078125])]);
    cpu.execute(&graph, inputs.clone()).unwrap();
    gpu.upload(inputs).unwrap();
    gpu.run().unwrap();
    assert_eq!(gpu.read(output).unwrap(), vec![1. / 16384.]);
    assert_eq!(gpu.read(output).unwrap(), cpu.get_value(output).unwrap());
    assert_eq!(gpu.gradients(&graph).unwrap(), cpu.get_gradients(&graph));
}

#[test]
fn mixed_exact_indices_preserve_365_and_reset_scatter_on_replay() {
    use crate::tensor::IndexTensor;
    if !cuda_available() {
        return;
    }
    let table = Parameter::new(
        (0..1200).map(|i| (i % 13) as f32 / 8.).collect(),
        vec![400, 3],
    );
    let id = table.id();
    let y = TensorExpr::from(table).embedding_indices(IndexTensor::input("ids", vec![3]));
    let y = y.cast(ValueDType::BF16).cast(ValueDType::F32);
    let loss = (y * TensorExpr::input("g", vec![3, 3]))
        .reduce_sum(0)
        .reduce_sum(1);
    let mut graph = TensorGraph::new();
    let output = graph.add_expr(&loss, false);
    let graph = graph.with_gradients(output);
    let mut gpu = MixedExecutor::new(&graph, MatMulPrecision::Tf32x3, true).unwrap();
    let inputs = HashMap::from([("g".into(), vec![0.5; 9])]);
    let indices = HashMap::from([("ids".into(), vec![365, 365, 1])]);
    let mut cpu = SimpleExecutor::new();
    cpu.execute_with_indices(&graph, inputs.clone(), indices.clone())
        .unwrap();
    gpu.upload_with_indices(inputs.clone(), indices).unwrap();
    for _ in 0..3 {
        gpu.run().unwrap();
        assert_eq!(gpu.read(output).unwrap(), cpu.get_value(output).unwrap());
        assert_eq!(
            gpu.gradients(&graph).unwrap()[&id],
            cpu.get_gradients(&graph)[&id]
        );
    }
    assert!(
        gpu.upload_with_indices(inputs, HashMap::from([("ids".into(), vec![365, 400, 1])]))
            .is_err()
    );
    gpu.run().unwrap();
    assert_eq!(
        gpu.gradients(&graph).unwrap()[&id],
        cpu.get_gradients(&graph)[&id]
    );
}

#[test]
fn mixed_autograd_shared_cast_and_replay() {
    if !cuda_available() {
        return;
    }
    for replay in [false, true] {
        for shared in [false, true] {
            let (graph, id) = crate::graph::mixed::tests::fanout(shared);
            let mut executor = MixedExecutor::new(&graph, MatMulPrecision::Tf32x3, replay).unwrap();
            assert!(executor.info().bf16_bytes > 0);
            executor.upload(HashMap::new()).unwrap();
            executor.run().unwrap();
            assert_eq!(
                executor.gradients(&graph).unwrap()[&id],
                vec![if shared { 1. } else { 1. + 1. / 256. }]
            );
            for _ in 0..3 {
                executor.run().unwrap();
            }
            assert_eq!(
                executor.gradients(&graph).unwrap()[&id],
                vec![if shared { 1. } else { 1. + 1. / 256. }]
            );
        }
    }
}

#[test]
fn mixed_graph_matmul_views_and_changed_masters() {
    if !cuda_available() {
        return;
    }
    // Power-of-two fixtures give an exact independent scalar CPU oracle.
    let master = Parameter::new((0..35).map(|i| (i % 7) as f32 / 8.).collect(), vec![5, 7]);
    let id = master.id();
    let weight: TensorExpr<f32> = master.clone().into();
    let x = TensorExpr::input("x", vec![3, 7]).cast(ValueDType::BF16);
    let y = x
        .matmul(weight.cast(ValueDType::BF16).transpose())
        .cast(ValueDType::F32);
    let loss = y.clone().reduce_sum(0).reduce_sum(1);
    let mut graph = TensorGraph::new();
    let output = graph.add_expr(&loss, false);
    let graph = graph.with_gradients(output);
    let mut gpu = MixedExecutor::new(&graph, MatMulPrecision::Tf32x3, true).unwrap();
    for step in 0..2 {
        let inputs = HashMap::from([(
            "x".into(),
            (0..21).map(|i| (i % 5) as f32 / 8. + step as f32).collect(),
        )]);
        let mut cpu = SimpleExecutor::new();
        cpu.execute(&graph, inputs.clone()).unwrap();
        gpu.upload(inputs).unwrap();
        gpu.run().unwrap();
        assert_eq!(gpu.read(output).unwrap(), cpu.get_value(output).unwrap());
        assert_eq!(
            gpu.gradients(&graph).unwrap()[&id],
            cpu.get_gradients(&graph)[&id]
        );
        master.set_values(vec![0.5; 35]).unwrap();
    }
    // Rejected uploads must preserve the previously executable inputs.
    assert!(
        gpu.upload(HashMap::from([("x".into(), vec![0.; 20])]))
            .is_err()
    );
    gpu.run().unwrap();
    assert!(!gpu.read(output).unwrap().iter().any(|x| x.is_nan()));
}

#[test]
fn mixed_cast_materializes_noncontiguous_views_once() {
    if !cuda_available() {
        return;
    }
    let x = TensorExpr::input("x", vec![2, 3])
        .transpose()
        .cast(ValueDType::BF16)
        .cast(ValueDType::F32);
    let mut graph = TensorGraph::new();
    let output = graph.add_expr(&x, false);
    let mut gpu = MixedExecutor::new(&graph, MatMulPrecision::Tf32x3, true).unwrap();
    gpu.upload(HashMap::from([(
        "x".into(),
        vec![1. + 1. / 256., 2., 3., 4., 5., 6.],
    )]))
    .unwrap();
    gpu.run().unwrap();
    assert_eq!(gpu.read(output).unwrap(), vec![1., 4., 2., 5., 3., 6.]);
    // Promotion is a typed read of the narrowed storage; only narrowing needs a kernel.
    assert_eq!(gpu.info().kernels, 1);
}

#[test]
fn mixed_f32_regions_save_silu_primals_across_cast_boundaries() {
    if !cuda_available() {
        return;
    }
    let master = Parameter::new(vec![-2_f32, -0.5, 0., 0.25, 1., 3.], vec![2, 3]);
    let id = master.id();
    let primal: TensorExpr<f32> = master.clone().into();
    let x = primal
        .cast(ValueDType::BF16)
        .transpose()
        .cast(ValueDType::F32);
    let y = crate::nn::silu(x).cast(ValueDType::BF16);
    let loss = (y.clone().cast(ValueDType::F32) * TensorExpr::input("a", vec![3, 2])
        + y.cast(ValueDType::F32) * TensorExpr::input("b", vec![3, 2]))
    .reduce_sum(0)
    .reduce_sum(1);
    let mut graph = TensorGraph::new();
    let output = graph.add_expr(&loss, false);
    let graph = graph.with_gradients(output);
    let mut fused = MixedExecutor::new(&graph, MatMulPrecision::Tf32x3, true).unwrap();
    let mut unfused = MixedExecutor::new_unfused(&graph, MatMulPrecision::Tf32x3, false).unwrap();
    assert!(fused.info().kernels < unfused.info().kernels);
    for step in 0..2 {
        let inputs = HashMap::from([
            ("a".into(), vec![1. + 1. / 256.; 6]),
            ("b".into(), vec![1. / 256.; 6]),
        ]);
        let mut cpu = SimpleExecutor::new();
        cpu.execute(&graph, inputs.clone()).unwrap();
        let expected = cpu.get_gradients(&graph)[&id].clone();
        for gpu in [&mut fused, &mut unfused] {
            gpu.upload(inputs.clone()).unwrap();
            gpu.run().unwrap();
            let actual = gpu.gradients(&graph).unwrap();
            assert_eq!(actual[&id], expected, "step {step}");
            assert_eq!(gpu.read(output).unwrap(), cpu.get_value(output).unwrap());
        }
        master.set_values(vec![0.5; 6]).unwrap();
    }
}

#[test]
fn mixed_identity_graph_needs_no_capture_or_kernel() {
    if !cuda_available() {
        return;
    }
    let x = TensorExpr::input("x", vec![2]).cast(ValueDType::F32);
    let mut graph = TensorGraph::new();
    let output = graph.add_expr(&x, false);
    let mut gpu = MixedExecutor::new(&graph, MatMulPrecision::Tf32x3, true).unwrap();
    assert_eq!(gpu.info().kernels, 0);
    assert!(!gpu.info().replay);
    gpu.upload(HashMap::from([("x".into(), vec![1., 2.])]))
        .unwrap();
    gpu.run().unwrap();
    assert_eq!(gpu.read(output).unwrap(), vec![1., 2.]);
}

#[test]
fn mixed_norm_linear_gradients_match_cpu() {
    if !cuda_available() {
        return;
    }
    let weight = Parameter::new(vec![1f32; 4], vec![4]);
    let bias = Parameter::new(vec![0f32; 4], vec![4]);
    let x = TensorExpr::input("x", vec![2, 4]);
    let normalized = crate::nn::layer_norm(x, 1, weight.clone(), bias.clone(), 1e-5);
    let y = normalized
        .linear_bf16(
            TensorExpr::constant(vec![0.5; 12], vec![4, 3]),
            TensorExpr::constant(vec![0.; 3], vec![3]),
        )
        .unwrap();
    let y = (y.cast(ValueDType::F32)
        * TensorExpr::constant(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]))
    .mean_all();
    let mut graph = TensorGraph::new();
    let output = graph.add_expr(&y, false);
    let graph = graph.with_gradients(output);
    let inputs = HashMap::from([("x".into(), vec![1., 2., 3., 4., 5., 6., 7., 8.])]);
    let mut cpu = SimpleExecutor::new();
    cpu.execute(&graph, inputs.clone()).unwrap();
    let expected = cpu.get_gradients(&graph);
    let mut gpu = MixedExecutor::new(&graph, MatMulPrecision::Tf32x3, true).unwrap();
    gpu.upload(inputs).unwrap();
    gpu.run().unwrap();
    let actual = gpu.gradients(&graph).unwrap();
    for id in [weight.id(), bias.id()] {
        assert!(expected[&id].iter().any(|x| x.abs() > 0.1));
        for (a, b) in actual[&id].iter().zip(&expected[&id]) {
            assert!((a - b).abs() < 1e-4, "{id}: {a} != {b}");
        }
    }
}

#[test]
fn mixed_linear_bias_reduction_matches_cpu() {
    if !cuda_available() {
        return;
    }
    let bias = Parameter::new(vec![0f32; 768], vec![768]);
    let weight = Parameter::new(
        (0..(32 * 768))
            .map(|i| ((i % 17) as f32 - 5.) / 8.)
            .collect(),
        vec![32, 768],
    );
    let x = Parameter::new(
        (0..(192 * 32))
            .map(|i| ((i % 29) as f32 - 10.) / 4.)
            .collect(),
        vec![3, 64, 32],
    );
    let y = TensorExpr::from(x.clone())
        .reshape(vec![192, 32])
        .linear_bf16(weight.clone().into(), bias.clone().into())
        .unwrap()
        .reshape(vec![3, 64, 768])
        .cast(ValueDType::F32);
    let loss = (y * TensorExpr::constant(
        (0..(192 * 768))
            .map(|i| ((i % 13) as f32 - 4.) / 8.)
            .collect(),
        vec![3, 64, 768],
    ))
    .mean_all();
    let mut graph = TensorGraph::new();
    let out = graph.add_expr(&loss, false);
    let graph = graph.with_gradients(out);
    let mut cpu = SimpleExecutor::new();
    cpu.execute(&graph, HashMap::new()).unwrap();
    let mut gpu = MixedExecutor::new(&graph, MatMulPrecision::Tf32x3, true).unwrap();
    gpu.upload(HashMap::new()).unwrap();
    gpu.run().unwrap();
    let actual = gpu.gradients(&graph).unwrap();
    let expected = cpu.get_gradients(&graph);
    for id in [x.id(), weight.id(), bias.id()] {
        for (a, b) in actual[&id].iter().zip(&expected[&id]) {
            assert!((a - b).abs() < 1e-4, "{id}: {a} != {b}");
        }
    }
}

#[test]
fn mixed_register_tiled_bf16_matmul_handles_edges_and_replay() {
    if !cuda_available() {
        return;
    }
    let (m, k, n) = (2048, 65, 257);
    let a = TensorExpr::input("a", vec![m, k]).cast(ValueDType::BF16);
    let b = TensorExpr::input("b", vec![k, n]).cast(ValueDType::BF16);
    let mut graph = TensorGraph::new();
    let out = graph.add_expr(&a.matmul(b), false);
    let a: Vec<f32> = (0..m * k)
        .map(|i| half::bf16::from_f32((i % 23) as f32 / 37. - 0.2).to_f32())
        .collect();
    let b: Vec<f32> = (0..k * n)
        .map(|i| half::bf16::from_f32((i % 17) as f32 / 29. - 0.3).to_f32())
        .collect();
    for replay in [false, true] {
        let mut gpu = MixedExecutor::new(&graph, MatMulPrecision::Tf32x3, replay).unwrap();
        gpu.upload(HashMap::from([
            ("a".into(), a.clone()),
            ("b".into(), b.clone()),
        ]))
        .unwrap();
        for _ in 0..2 {
            gpu.run().unwrap();
            let actual = gpu.read(out).unwrap();
            for row in [0, 63, 64, 2047] {
                for col in [0, 63, 64, 255, 256] {
                    let expected: f64 = (0..k)
                        .map(|inner| f64::from(a[row * k + inner]) * f64::from(b[inner * n + col]))
                        .sum();
                    let expected = half::bf16::from_f32(expected as f32).to_f32();
                    assert!(
                        (actual[row * n + col] - expected).abs() <= 0.0078125,
                        "row {row} col {col}"
                    );
                }
            }
        }
    }
}

#[test]
fn mixed_split_k_keeps_f32_partials_and_rounds_once() {
    if !cuda_available() {
        return;
    }
    let (m, k, n) = (128, 2048, 128);
    let a = TensorExpr::input("a", vec![k, m])
        .cast(ValueDType::BF16)
        .transpose();
    let b = TensorExpr::input("b", vec![k, n]).cast(ValueDType::BF16);
    let expression = a.matmul(b);
    let mut graph = TensorGraph::new();
    let out = graph.add_expr(&expression, false);
    let mut b = vec![0.; k * n];
    for part in 0..4 {
        for col in 0..n {
            b[(part * (k / 4)) * n + col] = 1.;
            b[(part * (k / 4) + 1) * n + col] = if part == 3 { 1. / 128. } else { 1. / 256. };
        }
    }
    for replay in [false, true] {
        let mut gpu = MixedExecutor::new(&graph, MatMulPrecision::Tf32x3, replay).unwrap();
        assert!(
            gpu.kernel_labels()
                .iter()
                .any(|s| s.contains("split-K partial"))
        );
        gpu.upload(HashMap::from([
            ("a".into(), vec![1.; k * m]),
            ("b".into(), b.clone()),
        ]))
        .unwrap();
        for _ in 0..2 {
            gpu.run().unwrap();
            for (i, value) in gpu.read(out).unwrap().iter().enumerate() {
                assert_eq!(*value, 4.03125, "element {i}");
            }
        }
    }
}

#[test]
fn mixed_destruction_waits_for_shared_stream_capture() {
    use std::{sync::mpsc, time::Duration};
    if !cuda_available() {
        return;
    }
    let graph: TensorGraph<f32> = TensorExpr::input("x", vec![4])
        .cast(ValueDType::BF16)
        .into();
    let executor = MixedExecutor::new(&graph, MatMulPrecision::Tf32x3, true).unwrap();
    let gate = executor.gate.clone();
    let stream = executor.stream.clone();
    let kernel = executor.kernels[0].clone();
    let (started_tx, started_rx) = mpsc::channel();
    let (done_tx, done_rx) = mpsc::channel();
    let task = std::thread::spawn(move || {
        let guard = gate.lock().unwrap();
        stream
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .unwrap();
        kernel.run(&stream).unwrap();
        started_tx.send(()).unwrap();
        assert!(matches!(
            done_rx.recv_timeout(Duration::from_millis(100)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ));
        let graph = stream
            .end_capture(
                sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            )
            .unwrap()
            .unwrap();
        drop(graph);
        drop(guard);
        done_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    });
    started_rx.recv().unwrap();
    drop(executor);
    done_tx.send(()).unwrap();
    task.join().unwrap();
}

#[test]
fn mixed_preparation_waits_for_shared_stream_capture() {
    use std::{sync::mpsc, time::Duration};
    if !cuda_available() {
        return;
    }
    let graph: TensorGraph<f32> = TensorExpr::input("x", vec![4])
        .cast(ValueDType::BF16)
        .into();
    let executor = MixedExecutor::new(&graph, MatMulPrecision::Tf32x3, true).unwrap();
    let gate = executor.gate.clone();
    let stream = executor.stream.clone();
    let kernel = executor.kernels[0].clone();
    let bindings = ResidentBindings {
        stream: stream.clone(),
        gate: gate.clone(),
        parameters: HashMap::new(),
    };
    let (started_tx, started_rx) = mpsc::channel();
    let (done_tx, done_rx) = mpsc::channel();
    let task = std::thread::spawn(move || {
        let guard = gate.lock().unwrap();
        stream
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .unwrap();
        kernel.run(&stream).unwrap();
        started_tx.send(()).unwrap();
        assert!(matches!(
            done_rx.recv_timeout(Duration::from_millis(100)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ));
        drop(
            stream
                .end_capture(
                    sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
                )
                .unwrap()
                .unwrap(),
        );
        drop(guard);
        done_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    });
    started_rx.recv().unwrap();
    let prepared =
        MixedExecutor::for_resident(&graph, MatMulPrecision::Tf32x3, true, bindings).unwrap();
    done_tx.send(()).unwrap();
    task.join().unwrap();
    drop(prepared);
}

#[test]
fn mixed_transposed_bf16_async_replay_handles_changed_feeds() {
    if !cuda_available() {
        return;
    }
    for (m, k, n, ta, tb) in [
        (264, 136, 280, true, true),
        (267, 136, 280, true, true),
        (264, 136, 283, true, false),
        (264, 137, 280, false, false),
    ] {
        let a =
            TensorExpr::input("a", if ta { vec![k, m] } else { vec![m, k] }).cast(ValueDType::BF16);
        let b =
            TensorExpr::input("b", if tb { vec![n, k] } else { vec![k, n] }).cast(ValueDType::BF16);
        let a = if ta { a.transpose() } else { a };
        let b = if tb { b.transpose() } else { b };
        let mut graph = TensorGraph::new();
        let out = graph.add_expr(&a.matmul(b), false);
        for replay in [false, true] {
            let mut gpu = MixedExecutor::new(&graph, MatMulPrecision::Tf32x3, replay).unwrap();
            for revision in 0..3 {
                let values = |count, period| {
                    (0..count)
                        .map(|i| {
                            half::bf16::from_f32(((i + revision) % period) as f32 / 32. - 0.125)
                                .to_f32()
                        })
                        .collect::<Vec<_>>()
                };
                let av = values(k * m, 7);
                let bv = values(n * k, 5);
                gpu.upload(HashMap::from([
                    ("a".into(), av.clone()),
                    ("b".into(), bv.clone()),
                ]))
                .unwrap();
                gpu.run().unwrap();
                let actual = gpu.read(out).unwrap();
                for row in [0, 31, 32, 255, m - 1] {
                    for col in [0, 31, 32, 255, n - 1] {
                        let expected: f64 = (0..k)
                            .map(|inner| {
                                f64::from(av[if ta { inner * m + row } else { row * k + inner }])
                                    * f64::from(
                                        bv[if tb { col * k + inner } else { inner * n + col }],
                                    )
                            })
                            .sum();
                        assert!(
                            (actual[row * n + col] - half::bf16::from_f64(expected).to_f32()).abs()
                                <= f32::EPSILON,
                            "replay={replay} revision={revision} row={row} col={col}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn mixed_hybrid_split_k_preserves_ragged_edges_and_replay() {
    if !cuda_available() {
        return;
    }
    let k = 4096;
    for (m, n) in [(267, 280), (264, 283)] {
        let a = TensorExpr::input("a", vec![k, m])
            .cast(ValueDType::BF16)
            .transpose();
        let b = TensorExpr::input("b", vec![k, n]).cast(ValueDType::BF16);
        let mut graph = TensorGraph::new();
        let out = graph.add_expr(&a.matmul(b), false);
        for replay in [false, true] {
            let mut gpu = MixedExecutor::new(&graph, MatMulPrecision::Tf32x3, replay).unwrap();
            for revision in 0..2 {
                let values = |count, period| {
                    (0..count)
                        .map(|i| {
                            half::bf16::from_f32(((i + revision) % period) as f32 / 32. - 0.125)
                                .to_f32()
                        })
                        .collect::<Vec<_>>()
                };
                let av = values(k * m, 7);
                let bv = values(k * n, 5);
                gpu.upload(HashMap::from([
                    ("a".into(), av.clone()),
                    ("b".into(), bv.clone()),
                ]))
                .unwrap();
                gpu.run().unwrap();
                let actual = gpu.read(out).unwrap();
                for row in [0, 31, 32, 63, 64, m - 1] {
                    for col in [0, 31, 32, 63, 64, n - 1] {
                        let expected: f64 = (0..k)
                            .map(|inner| {
                                f64::from(av[inner * m + row]) * f64::from(bv[inner * n + col])
                            })
                            .sum();
                        assert!(
                            (actual[row * n + col] - half::bf16::from_f64(expected).to_f32()).abs()
                                <= f32::EPSILON,
                            "m={m} n={n} replay={replay} revision={revision} row={row} col={col}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn mixed_bf16_shared_reuse_schedule_and_codegen_without_device() -> anyhow::Result<()> {
    use crate::tile::{MatMulPipeline, MatMulRegion, MatMulSharedLayout};

    assert_eq!(
        MixedOptions::default(),
        MixedOptions {
            layer_norm_pair_fusion: false,
            qkv_grad_pack_fusion: false,
            direct_master_bindings: false,
            physical_order_casts: false,
            shared_memory_reuse: false,
            incremental_allocation_reuse: false,
            attention_softmax_fusion: false,
            ..MixedOptions::default()
        }
    );
    let target = PtxTarget::sm80();
    for (m, k, n) in [(128, 64, 128), (33, 65, 35)] {
        let mut graph = TensorGraph::<half::bf16>::new();
        let a = graph.graph.add_node(TensorGraphNode::Input {
            name: "a",
            shape: vec![2, m, k],
        });
        let b = graph.graph.add_node(TensorGraphNode::Input {
            name: "b",
            shape: vec![1, k, n],
        });
        let out = graph.graph.add_node(TensorGraphNode::MatMul {
            shape: vec![2, m, n],
        });
        graph.graph.add_edge(a, out, 0);
        graph.graph.add_edge(b, out, 1);
        let mut plan = crate::ptx::plan::PtxExecutionPlan::build(&graph)?;
        plan.schedule_matmuls(
            target.matmul_capabilities(),
            MatMulPrecision::NativeHalf,
            target.matmul_resources,
            MatMulSharedLayout::Padded,
            MatMulPipeline::Auto,
            target.supports_async_copy(),
            false,
        )?;
        let baseline = plan.matmul_regions().to_vec();
        assert!(!baseline.is_empty());
        plan.set_matmul_shared_memory_reuse(true);
        for (before, after) in baseline.iter().zip(plan.matmul_regions()) {
            assert!(!before.schedule.reuse_shared_memory);
            assert!(after.schedule.reuse_shared_memory);
            assert_eq!(
                after.schedule.with_shared_memory_reuse(false),
                before.schedule
            );
            assert!(after.schedule.shared_mem_bytes() < before.schedule.shared_mem_bytes());
            let emit = |region: &MatMulRegion| -> anyhow::Result<String> {
                let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
                tiles.add_matmul_regions(std::slice::from_ref(region))?;
                tiles.optimize_indices()?;
                Ok(PtxGraph::from(tiles).with_target(target).module_source())
            };
            let old = emit(before)?;
            let new = emit(after)?;
            for opcode in [
                "wmma.mma.sync",
                "wmma.load.a.sync",
                "wmma.load.b.sync",
                "wmma.store.d.sync",
                "cp.async",
                "bar.sync",
            ] {
                assert_eq!(old.matches(opcode).count(), new.matches(opcode).count());
            }
            assert!(new.contains(&format!(
                ".shared .align 32 .b8 shared[{}];",
                after.schedule.shared_mem_bytes()
            )));
        }
    }
    Ok(())
}

#[test]
fn mixed_shared_reuse_preserves_bf16_forward_backward_and_replay() {
    if !cuda_available() {
        return;
    }
    // Aligned tensor-core case, ragged case and small fallback. The option
    // changes only scratch allocation, so finite outputs must match bitwise.
    for (rows, input, output) in [(128, 64, 128), (33, 65, 35), (1, 7, 5)] {
        let weight = Parameter::new(
            (0..input * output)
                .map(|i| (i % 13) as f32 / 64. - 0.0625)
                .collect(),
            vec![input, output],
        );
        let bias = Parameter::new(vec![0.03125; output], vec![output]);
        let y = TensorExpr::input("x", vec![rows, input])
            .linear_bf16(weight.clone().into(), bias.clone().into())
            .unwrap()
            .cast(ValueDType::F32);
        let mut graph = TensorGraph::new();
        let forward = graph.add_expr(&y, false);
        let loss = graph.add_expr(&y.mean_all(), false);
        let graph = graph.with_gradients(loss);
        for replay in [false, true] {
            let mut baseline =
                MixedExecutor::new(&graph, MatMulPrecision::NativeHalf, replay).unwrap();
            let mut candidate = MixedExecutor::new_with_options(
                &graph,
                MatMulPrecision::NativeHalf,
                replay,
                MixedOptions {
                    shared_memory_reuse: true,
                    ..MixedOptions::default()
                },
            )
            .unwrap();
            if rows == 128 {
                // Check the constructor reached the BF16 codegen, not just
                // numerical equality through an accidentally ignored option.
                let shared_bytes = |executor: &MixedExecutor| -> usize {
                    executor
                        .programs
                        .iter()
                        .filter_map(|p| p.downcast_ref::<Program<BF16>>())
                        .flat_map(|p| p.ptx.lines())
                        .filter_map(|line| line.split(".b8 shared[").nth(1))
                        .filter_map(|s| s.split(']').next()?.parse::<usize>().ok())
                        .sum()
                };
                assert!(candidate.info().bf16_tensor_core_programs > 0);
                assert!(shared_bytes(&candidate) < shared_bytes(&baseline));
            }
            for revision in 0..3 {
                let data = (0..rows * input)
                    .map(|i| ((i + revision) % 17) as f32 / 32. - 0.125)
                    .collect();
                let feed = HashMap::from([("x".into(), data)]);
                baseline.upload(feed.clone()).unwrap();
                candidate.upload(feed).unwrap();
                baseline.run().unwrap();
                candidate.run().unwrap();
                for node in [forward, loss] {
                    let bits = |v: Vec<f32>| v.into_iter().map(f32::to_bits).collect::<Vec<_>>();
                    assert_eq!(
                        bits(baseline.read(node).unwrap()),
                        bits(candidate.read(node).unwrap())
                    );
                }
                let old = baseline.gradients(&graph).unwrap();
                let new = candidate.gradients(&graph).unwrap();
                for id in [weight.id(), bias.id()] {
                    assert!(old[&id].iter().all(|v| v.is_finite()));
                    assert_eq!(
                        old[&id].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        new[&id].iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                    );
                }
            }
        }
    }
}

#[test]
fn mixed_incremental_allocation_reuse_preserves_aliases_scatter_gradients_and_refresh() {
    use crate::tensor::IndexTensor;
    if !cuda_available() {
        return;
    }
    let table = Parameter::new(
        (0..28).map(|i| (i % 13) as f32 / 8. - 0.5).collect(),
        vec![7, 4],
    );
    let primal = TensorExpr::from(table.clone())
        .embedding_indices(IndexTensor::input("ids", vec![3]))
        .cast(ValueDType::BF16)
        .transpose()
        .cast(ValueDType::F32);
    // A physical alias is a separately retained sink, while F32 regions save
    // primals across multiple casts and a later branch consumes the same view.
    let y = crate::nn::silu(primal.clone()).cast(ValueDType::BF16);
    let loss = (y.clone().cast(ValueDType::F32) * TensorExpr::input("g", vec![4, 3])
        + primal * y.cast(ValueDType::F32))
    .mean_all();
    let mut graph = TensorGraph::new();
    let alias = graph.add_expr(&TensorExpr::from(table.clone()).transpose(), false);
    let output = graph.add_expr(&loss, false);
    let graph = graph.with_gradients(output);
    let parameter_nodes: Vec<_> = graph.graph.node_indices().filter(|&node| {
        matches!(&graph[node], TensorGraphNode::Parameter { id, .. } if *id == table.id())
    }).collect();
    let parameter_node = parameter_nodes[0];
    for replay in [false, true] {
        let make = |incremental| {
            let stream = CudaContext::new(0).unwrap().new_stream().unwrap();
            let master = Arc::new(stream.memcpy_stod(&table.values().unwrap()).unwrap());
            MixedExecutor::for_resident_with_options(
                &graph,
                MatMulPrecision::Tf32x3,
                replay,
                ResidentBindings {
                    stream,
                    gate: Arc::new(Mutex::new(())),
                    parameters: parameter_nodes
                        .iter()
                        .map(|&n| (n, master.clone()))
                        .collect(),
                },
                MixedOptions {
                    incremental_allocation_reuse: incremental,
                    ..MixedOptions::default()
                },
            )
            .unwrap()
        };
        let mut baseline = make(false);
        let mut candidate = make(true);
        assert_eq!(
            baseline.info().unpooled_bytes,
            candidate.info().unpooled_bytes
        );
        assert_eq!(baseline.info().kernels, candidate.info().kernels);
        assert!(candidate.info().reused_allocations > 0);
        assert_eq!(candidate.info().replay, replay);
        assert_eq!(
            baseline
                .kernels
                .iter()
                .map(|k| &k.label)
                .collect::<Vec<_>>(),
            candidate
                .kernels
                .iter()
                .map(|k| &k.label)
                .collect::<Vec<_>>()
        );
        assert!(candidate.kernels.iter().any(|k| k.zero_before.is_some()));
        let pointers: Vec<_> = [&baseline, &candidate]
            .iter()
            .map(|e| e.values[&parameter_node].binding.pointer)
            .collect();
        for revision in 0..3 {
            let master: Vec<_> = (0..28)
                .map(|i| ((i + revision) % 11) as f32 / 8. - 0.25)
                .collect();
            let indices = HashMap::from([(
                "ids".into(),
                if revision % 2 == 0 {
                    vec![6, 6, 1]
                } else {
                    vec![2, 3, 2]
                },
            )]);
            let inputs = HashMap::from([("g".into(), vec![0.25 + revision as f32 / 4.; 12])]);
            for (index, e) in [&mut baseline, &mut candidate].into_iter().enumerate() {
                let latest = Arc::new(e.stream.memcpy_stod(&master).unwrap());
                e.refresh_parameters(
                    &parameter_nodes
                        .iter()
                        .map(|&n| (n, latest.clone()))
                        .collect(),
                )
                .unwrap();
                assert_eq!(e.values[&parameter_node].binding.pointer, pointers[index]);
                e.upload_with_indices(inputs.clone(), indices.clone())
                    .unwrap();
                // Repeated scatter must zero its logical range before every replay.
                for _ in 0..3 {
                    e.run().unwrap();
                }
            }
            let bits = |v: Vec<f32>| v.into_iter().map(f32::to_bits).collect::<Vec<_>>();
            for node in [alias, output] {
                assert_eq!(
                    bits(baseline.read(node).unwrap()),
                    bits(candidate.read(node).unwrap())
                );
            }
            let expected_alias: Vec<_> = (0..4)
                .flat_map(|col| (0..7).map(move |row| (row, col)))
                .map(|(row, col)| master[row * 4 + col])
                .collect();
            assert_eq!(candidate.read(alias).unwrap(), expected_alias);
            let old = baseline.gradients(&graph).unwrap();
            let new = candidate.gradients(&graph).unwrap();
            assert!(new[&table.id()].iter().all(|v| v.is_finite()));
            assert_eq!(
                bits(old[&table.id()].clone()),
                bits(new[&table.id()].clone())
            );
            for row in 0..7 {
                if !indices["ids"].contains(&(row as u32)) {
                    assert!(
                        new[&table.id()][row * 4..(row + 1) * 4]
                            .iter()
                            .all(|v| *v == 0.0)
                    );
                }
            }
        }
    }
}

#[test]
fn bf16_eight_compute_warps_preserves_linear_gradients_replay_and_fallback() -> Result<()> {
    if !cuda_available() {
        return Ok(());
    }
    for (rows, transpose_weight) in [(2048, false), (2048, true), (2049, true), (512, true)] {
        let x = Parameter::new(vec![0.125; rows * 64], vec![rows, 64]);
        let weight = Parameter::new(
            (0..64 * 256)
                .map(|i| (i % 13) as f32 / 64. - 0.0625)
                .collect(),
            if transpose_weight {
                vec![256, 64]
            } else {
                vec![64, 256]
            },
        );
        let weight_expr = TensorExpr::from(weight.clone());
        let weight_expr = if transpose_weight {
            weight_expr.transpose()
        } else {
            weight_expr
        };
        let bias = Parameter::new(vec![0.03125; 256], vec![256]);
        let output = (TensorExpr::from(x.clone()) + TensorExpr::input("delta", vec![rows, 64]))
            .linear_bf16(weight_expr, bias.clone().into())?
            .cast(ValueDType::F32);
        let mut graph = TensorGraph::new();
        let forward = graph.add_expr(&output, false);
        let loss = graph.add_expr(
            &(output * TensorExpr::input("seed", vec![rows, 256])).mean_all(),
            false,
        );
        let graph = graph.with_gradients(loss);
        for replay in [false, true] {
            let options = MixedOptions {
                physical_order_casts: true,
                shared_memory_reuse: true,
                narrow_bf16_tile: true,
                transposed_bf16_outputs: true,
                ..MixedOptions::default()
            };
            let mut old = MixedExecutor::new_with_options(
                &graph,
                MatMulPrecision::NativeHalf,
                replay,
                options,
            )?;
            let mut new = MixedExecutor::new_with_options(
                &graph,
                MatMulPrecision::NativeHalf,
                replay,
                MixedOptions {
                    bf16_eight_compute_warps: true,
                    ..options
                },
            )?;
            assert_eq!(old.info().bf16_eight_compute_warp_programs, 0);
            // Adding an explicit forward output and the loss can retain two
            // equivalent Linear nodes. Count eligible graph operations rather
            // than assuming cross-add_expr common-subexpression elimination.
            let expected_programs = graph.graph.node_weights().filter(|node| {
                matches!(node, TensorGraphNode::Linear { shape } if shape[0] >= 2048 && shape[1] >= 256)
            }).count();
            assert_eq!(
                new.info().bf16_eight_compute_warp_programs,
                expected_programs
            );
            assert_eq!(expected_programs > 0, rows >= 2048);
            assert_eq!(old.info().bf16_bytes, new.info().bf16_bytes);
            assert_eq!(old.info().f32_bytes, new.info().f32_bytes);
            assert_eq!(old.info().kernels, new.info().kernels);
            let old_shared: usize = old
                .programs
                .iter()
                .filter_map(|p| p.downcast_ref::<Program<BF16>>())
                .flat_map(|p| p.kernels.iter())
                .map(|k| k.config.shared_mem_bytes as usize)
                .sum();
            let new_shared: usize = new
                .programs
                .iter()
                .filter_map(|p| p.downcast_ref::<Program<BF16>>())
                .flat_map(|p| p.kernels.iter())
                .map(|k| k.config.shared_mem_bytes as usize)
                .sum();
            assert_eq!(old_shared, new_shared);
            for p in new
                .programs
                .iter()
                .filter_map(|p| p.downcast_ref::<Program<BF16>>())
                .filter(|p| p.bf16_eight_compute_warps)
            {
                assert_eq!(p.ptx.matches("wmma.mma").count(), 4);
                assert_eq!(p.ptx.matches("wmma.load.b").count(), 2);
            }
            for phase in 0..3 {
                let feed = HashMap::from([
                    (
                        "delta".into(),
                        (0..rows * 64)
                            .map(|i| ((i + phase) % 7) as f32 / 64.)
                            .collect(),
                    ),
                    (
                        "seed".into(),
                        (0..rows * 256)
                            .map(|i| ((i + phase) % 3) as f32 / 4.)
                            .collect(),
                    ),
                ]);
                old.upload(feed.clone())?;
                new.upload(feed)?;
                old.run()?;
                new.run()?;
                for node in [forward, loss] {
                    let a = old.read(node)?;
                    let b = new.read(node)?;
                    assert!(a.iter().all(|v| v.is_finite()));
                    assert_eq!(
                        a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                    );
                }
                let a = old.gradients(&graph)?;
                let b = new.gradients(&graph)?;
                for id in [x.id(), weight.id(), bias.id()] {
                    assert!(a[&id].iter().all(|v| v.is_finite()));
                    assert_eq!(
                        a[&id].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        b[&id].iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                    );
                }
            }
            assert_eq!(new.captured.is_some(), replay);
        }
    }
    Ok(())
}

#[test]
fn medium_bf16_policy_preserves_gradients_replay_and_actual_activation() -> Result<()> {
    use crate::tile::MediumBf16TilePolicy as Policy;
    if !cuda_available() {
        return Ok(());
    }
    for (rows, columns, policy, square, tall) in [
        (128, 4096, Policy::SquareOnly, true, false),
        (128, 2048, Policy::SquareOnly, false, false),
        (128, 2048, Policy::SquareThenTall, false, true),
        (128, 256, Policy::SquareThenTall, false, false),
        (129, 4096, Policy::SquareThenTall, false, false),
    ] {
        let k = 128;
        let x = Parameter::new(vec![0.125; rows * k], vec![rows, k]);
        let w = Parameter::new(
            (0..columns * k)
                .map(|i| (i % 13) as f32 / 64. - 0.0625)
                .collect(),
            vec![columns, k],
        );
        let bias = Parameter::new(vec![0.03125; columns], vec![columns]);
        let y = (TensorExpr::from(x.clone()) + TensorExpr::input("delta", vec![rows, k]))
            .linear_bf16(TensorExpr::from(w.clone()).transpose(), bias.clone().into())?
            .cast(ValueDType::F32);
        let mut graph = TensorGraph::new();
        let output = graph.add_expr(&y, false);
        let loss = graph.add_expr(
            &(y * TensorExpr::input("seed", vec![rows, columns])).mean_all(),
            false,
        );
        let graph = graph.with_gradients(loss);
        for replay in [false, true] {
            let options = MixedOptions {
                physical_order_casts: true,
                shared_memory_reuse: true,
                narrow_bf16_tile: true,
                transposed_bf16_outputs: true,
                ..MixedOptions::default()
            };
            let mut old = MixedExecutor::new_with_options(
                &graph,
                MatMulPrecision::NativeHalf,
                replay,
                options,
            )?;
            let mut new = MixedExecutor::new_with_options(
                &graph,
                MatMulPrecision::NativeHalf,
                replay,
                MixedOptions {
                    medium_bf16_tile_policy: policy,
                    ..options
                },
            )?;
            assert_eq!(
                (
                    old.info().medium_bf16_square_programs,
                    old.info().medium_bf16_tall_programs
                ),
                (0, 0)
            );
            assert_eq!(new.info().medium_bf16_square_programs > 0, square);
            assert_eq!(new.info().medium_bf16_tall_programs > 0, tall);
            for program in new
                .programs
                .iter()
                .filter_map(|p| p.downcast_ref::<Program<BF16>>())
            {
                if program.medium_bf16_tile_counts.0 > 0 {
                    assert_eq!(program.ptx.matches("wmma.mma").count(), 8);
                }
                if program.medium_bf16_tile_counts.1 > 0 {
                    assert_eq!(program.ptx.matches("wmma.mma").count(), 4);
                }
            }
            for phase in 0..3 {
                let feeds = HashMap::from([
                    (
                        "delta".into(),
                        (0..rows * k)
                            .map(|i| ((i + phase) % 7) as f32 / 64.)
                            .collect(),
                    ),
                    (
                        "seed".into(),
                        (0..rows * columns)
                            .map(|i| ((i + phase) % 3) as f32 / 4.)
                            .collect(),
                    ),
                ]);
                old.upload(feeds.clone())?;
                new.upload(feeds)?;
                old.run()?;
                new.run()?;
                let bits = |v: Vec<f32>| v.into_iter().map(f32::to_bits).collect::<Vec<_>>();
                for node in [output, loss] {
                    let a = old.read(node)?;
                    assert!(a.iter().all(|v| v.is_finite()));
                    assert_eq!(bits(a), bits(new.read(node)?));
                }
                let a = old.gradients(&graph)?;
                let b = new.gradients(&graph)?;
                for id in [x.id(), w.id(), bias.id()] {
                    assert!(a[&id].iter().all(|v| v.is_finite()));
                    assert_eq!(bits(a[&id].clone()), bits(b[&id].clone()));
                }
            }
            assert_eq!(new.captured.is_some(), replay);
        }
    }
    Ok(())
}

#[test]
fn medium_bf16_linear_keeps_bias_before_only_output_rounding() -> Result<()> {
    use crate::tile::MediumBf16TilePolicy as Policy;
    if !cuda_available() {
        return Ok(());
    }
    for (columns, policy, square) in [
        (4096, Policy::SquareOnly, true),
        (2048, Policy::SquareThenTall, false),
    ] {
        let rows = 128;
        let k = 128;
        let w = Parameter::new(
            (0..columns * k)
                .map(|i| if i % k == 0 { 1.0078125 } else { 0. })
                .collect(),
            vec![columns, k],
        );
        let bias = Parameter::new(vec![-1.015625; columns], vec![columns]);
        let y = TensorExpr::input("x", vec![rows, k])
            .linear_bf16(TensorExpr::from(w).transpose(), bias.into())?
            .cast(ValueDType::F32);
        let graph: TensorGraph<f32> = y.into();
        let out = *graph.toposort().last().unwrap();
        let mut executor = MixedExecutor::new_with_options(
            &graph,
            MatMulPrecision::NativeHalf,
            true,
            MixedOptions {
                medium_bf16_tile_policy: policy,
                physical_order_casts: true,
                shared_memory_reuse: true,
                ..MixedOptions::default()
            },
        )?;
        assert_eq!(
            executor.info().medium_bf16_square_programs,
            usize::from(square)
        );
        assert_eq!(
            executor.info().medium_bf16_tall_programs,
            usize::from(!square)
        );
        executor.upload(HashMap::from([(
            "x".into(),
            (0..rows * k)
                .map(|i| if i % k == 0 { 1.0078125 } else { 0. })
                .collect(),
        )]))?;
        executor.run()?;
        // Exact f32 product is 1.01568603515625. An intermediate BF16
        // matmul store would round it to 1.015625 and incorrectly produce zero.
        assert_eq!(executor.read(out)?, vec![1. / 16384.; rows * columns]);
    }
    Ok(())
}
