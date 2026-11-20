use super::*;
use crate::tensor::{Parameter, TensorExpr};

fn fixture(
    m: usize,
    n: usize,
    k: usize,
) -> (
    TensorGraph<f32>,
    HashMap<NodeIndex, ValueDType>,
    [NodeIndex; 5],
) {
    let mut g = TensorGraph::new();
    let a = g.graph.add_node(TensorGraphNode::Input {
        name: "a",
        shape: vec![m, k],
    });
    let b = g.graph.add_node(TensorGraphNode::Input {
        name: "b",
        shape: vec![k, n],
    });
    let product = g
        .graph
        .add_node(TensorGraphNode::MatMul { shape: vec![m, n] });
    let cast = g.graph.add_node(TensorGraphNode::Cast {
        dtype: ValueDType::F32,
        shape: vec![m, n],
    });
    let trans = g
        .graph
        .add_node(TensorGraphNode::Transpose { shape: vec![n, m] });
    g.graph.add_edge(a, product, 0);
    g.graph.add_edge(b, product, 1);
    g.graph.add_edge(product, cast, 0);
    g.graph.add_edge(cast, trans, 0);
    let types = HashMap::from([
        (a, ValueDType::BF16),
        (b, ValueDType::BF16),
        (product, ValueDType::BF16),
        (cast, ValueDType::F32),
        (trans, ValueDType::F32),
    ]);
    (g, types, [a, b, product, cast, trans])
}

#[test]
fn closed_transposed_output_matcher_rejects_cast_fanout_rank_and_k_changes() {
    assert!(!MixedOptions::default().transposed_bf16_outputs);
    for (m, n, k, expected) in [
        (32, 64, 64, true),
        (65, 128, 96, true),
        (16, 16, 64, false),
        (16, 32, 64, true),
        (15, 64, 64, false),
        (64, 15, 64, false),
        (32, 64, 32, false),
        (32, 64, 65, false),
        (32, 64, 80, false),
        (128, 128, 2048, false),
    ] {
        let (g, t, nodes) = fixture(m, n, k);
        assert_eq!(eligible(&g, nodes[2], &t), expected);
    }
    for where_fanout in [2, 3] {
        let (mut g, t, nodes) = fixture(32, 64, 64);
        let extra = g.graph.add_node(TensorGraphNode::Transpose {
            shape: vec![64, 32],
        });
        g.graph.add_edge(nodes[where_fanout], extra, 0);
        assert!(!eligible(&g, nodes[2], &t));
    }
    let (g, mut t, nodes) = fixture(32, 64, 64);
    t.insert(nodes[0], ValueDType::F32);
    assert!(!eligible(&g, nodes[2], &t));
    t.insert(nodes[0], ValueDType::BF16);
    t.insert(nodes[2], ValueDType::F32);
    assert!(!eligible(&g, nodes[2], &t));
    let (mut g, t, nodes) = fixture(32, 64, 64);
    g.graph[nodes[3]] = TensorGraphNode::Cast {
        dtype: ValueDType::BF16,
        shape: vec![32, 64],
    };
    assert!(!eligible(&g, nodes[2], &t));
    let (mut g, t, nodes) = fixture(32, 64, 64);
    g.graph[nodes[2]] = TensorGraphNode::Linear {
        shape: vec![32, 64],
    };
    assert!(!eligible(&g, nodes[2], &t));
    let (mut g, t, nodes) = fixture(32, 64, 64);
    g.graph[nodes[2]] = TensorGraphNode::MatMul {
        shape: vec![1, 32, 64],
    };
    assert!(!eligible(&g, nodes[2], &t));
    let (mut g, t, nodes) = fixture(32, 64, 64);
    g.graph[nodes[4]] = TensorGraphNode::Reshape {
        shape: vec![64, 32],
    };
    assert!(!eligible(&g, nodes[2], &t));
}

#[test]
fn physical_transpose_value_retains_chain_and_logical_indices() -> Result<()> {
    let physical = Value {
        binding: Binding {
            pointer: 128,
            len: 65 * 32,
            dtype: DType::BF16,
        },
        view: VirtualTensor::identity(NodeIndex::new(7), vec![65, 32], DType::BF16),
        source_shape: vec![65, 32],
        chain: vec![],
    };
    let logical = transpose_value(&physical)?;
    assert_eq!(logical.source_shape, vec![65, 32]);
    assert_eq!(logical.view.shape, vec![32, 65]);
    assert_eq!(logical.binding.pointer, 128);
    assert!(matches!(&logical.chain[0],TensorGraphNode::Transpose{shape} if shape==&vec![32,65]));
    let restored = transpose_value(&logical)?;
    assert_eq!(restored.view.access, physical.view.access);
    assert_eq!(restored.view.shape, physical.view.shape);
    assert_eq!(restored.chain.len(), 2);
    for row in 0..32 {
        for col in 0..65 {
            assert_eq!(
                logical.view.source_index(&[row, col]).unwrap(),
                vec![col, row]
            );
            assert_eq!(
                restored.view.source_index(&[col, row]).unwrap(),
                vec![col, row]
            );
        }
    }
    Ok(())
}

fn cuda_available() -> bool {
    let available = CudaContext::new(0).is_ok();
    assert!(
        available || std::env::var("TNSR_REQUIRE_CUDA").as_deref() != Ok("1"),
        "CUDA required"
    );
    available
}

#[test]
fn transposed_output_matches_logical_reads_gradients_and_replay() -> Result<()> {
    if !cuda_available() {
        return Ok(());
    }
    // Both input orientations, rectangular M/N tails, and a swapped M that
    // crosses the existing 2048-row register-fragment schedule threshold.
    for (m, n, k) in [(32, 64, 64), (65, 33, 96), (32, 2048, 64)] {
        for transpose_a in [false, true] {
            let a_shape = if transpose_a { vec![k, m] } else { vec![m, k] };
            let a = Parameter::new(vec![0.125; m * k], a_shape.clone());
            let b = Parameter::new(
                (0..k * n).map(|i| (i % 7) as f32 / 32. - 0.0625).collect(),
                vec![k, n],
            );
            let a_expr = (TensorExpr::from(a.clone()) + TensorExpr::input("delta", a_shape))
                .cast(ValueDType::BF16);
            let a_expr = if transpose_a {
                a_expr.transpose()
            } else {
                a_expr
            };
            let raw = a_expr.matmul(TensorExpr::from(b.clone()).cast(ValueDType::BF16));
            let output = raw.cast(ValueDType::F32).transpose();
            let loss = (output * TensorExpr::input("seed", vec![n, m])).mean_all();
            let mut graph = TensorGraph::new();
            let loss = graph.add_expr(&loss, false);
            let graph = graph.with_gradients(loss);
            let types = value_dtypes(&graph);
            let matched: Vec<_> = graph
                .toposort()
                .into_iter()
                .filter(|&node| eligible(&graph, node, &types))
                .collect();
            assert!(!matched.is_empty());
            for (replay, combined) in [(false, false), (true, false), (true, true)] {
                let base_options = MixedOptions {
                    physical_order_casts: combined,
                    shared_memory_reuse: combined,
                    narrow_bf16_tile: combined,
                    ..MixedOptions::default()
                };
                let mut old = MixedExecutor::new_with_options(
                    &graph,
                    MatMulPrecision::NativeHalf,
                    replay,
                    base_options,
                )?;
                let mut new = MixedExecutor::new_with_options(
                    &graph,
                    MatMulPrecision::NativeHalf,
                    replay,
                    MixedOptions {
                        transposed_bf16_outputs: true,
                        ..base_options
                    },
                )?;
                assert_eq!(old.info().transposed_bf16_output_programs, 0);
                assert_eq!(new.info().transposed_bf16_output_programs, matched.len());
                let original = matched[0];
                let (_, cast) = after_identity_casts(
                    &graph,
                    original,
                    ValueDType::BF16,
                    graph[original].shape(),
                    &types,
                )
                .unwrap();
                let (_, trans) = after_identity_casts(
                    &graph,
                    cast,
                    ValueDType::F32,
                    graph[cast].shape(),
                    &types,
                )
                .unwrap();
                assert_eq!(
                    new.values[&original].source_shape,
                    vec![graph[original].shape()[1], graph[original].shape()[0]]
                );
                assert_eq!(new.values[&original].view.shape, *graph[original].shape());
                assert!(new.values[&trans].view.access.is_identity());
                assert_eq!(new.values[&trans].source_shape, *graph[trans].shape());
                assert_eq!(new.values[&trans].binding.dtype, DType::BF16);
                let bits =
                    |values: Vec<f32>| values.into_iter().map(f32::to_bits).collect::<Vec<_>>();
                for phase in 0..3 {
                    let feed = HashMap::from([
                        (
                            "delta".into(),
                            (0..m * k).map(|i| ((i + phase) % 5) as f32 / 64.).collect(),
                        ),
                        (
                            "seed".into(),
                            (0..n * m).map(|i| ((i + phase) % 3) as f32 / 4.).collect(),
                        ),
                    ]);
                    old.upload(feed.clone())?;
                    new.upload(feed)?;
                    old.run()?;
                    new.run()?;
                    // Read every logical rounding/view boundary, not only loss.
                    for node in [original, cast, trans, loss] {
                        let reference = old.read(node)?;
                        assert!(reference.iter().all(|v| v.is_finite()));
                        assert_eq!(bits(reference), bits(new.read(node)?));
                    }
                    let reference = old.gradients(&graph)?;
                    let actual = new.gradients(&graph)?;
                    for id in [a.id(), b.id()] {
                        assert!(reference[&id].iter().all(|v| v.is_finite()));
                        assert_eq!(bits(reference[&id].clone()), bits(actual[&id].clone()));
                    }
                }
                assert_eq!(new.info().replay, replay);
            }
        }
    }
    Ok(())
}

#[test]
fn transposed_output_nonfinite_class_and_zero_sign_diagnostics() -> Result<()> {
    if !cuda_available() {
        return Ok(());
    }
    let a = TensorExpr::input("a", vec![32, 64]).cast(ValueDType::BF16);
    let b = TensorExpr::input("b", vec![64, 64]).cast(ValueDType::BF16);
    let graph: TensorGraph<f32> = a.matmul(b).cast(ValueDType::F32).transpose().into();
    let out = *graph.toposort().last().unwrap();
    let mut old = MixedExecutor::new(&graph, MatMulPrecision::NativeHalf, true)?;
    let mut new = MixedExecutor::new_with_options(
        &graph,
        MatMulPrecision::NativeHalf,
        true,
        MixedOptions {
            transposed_bf16_outputs: true,
            ..MixedOptions::default()
        },
    )?;
    for special in [0.0, -0.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
        let mut values = if special == 0. {
            vec![special; 32 * 64]
        } else {
            vec![1.; 32 * 64]
        };
        values[0] = special;
        let feed = HashMap::from([("a".into(), values), ("b".into(), vec![1.; 64 * 64])]);
        old.upload(feed.clone())?;
        new.upload(feed)?;
        old.run()?;
        new.run()?;
        let (a, b) = (old.read(out)?, new.read(out)?);
        let mut zero_sign_differences = 0;
        for (x, y) in a.into_iter().zip(b) {
            assert_eq!(x.is_nan(), y.is_nan());
            assert_eq!(x.is_infinite(), y.is_infinite());
            if x.is_infinite() {
                assert_eq!(x.is_sign_negative(), y.is_sign_negative());
            }
            if x == 0. && y == 0. {
                zero_sign_differences += usize::from(x.to_bits() != y.to_bits());
            }
            if x.is_finite() {
                assert!((x - y).abs() <= 1e-6, "finite difference {x} vs {y}");
            }
        }
        eprintln!(
            "transposed BF16 output special={special:?}: zero-sign differences={zero_sign_differences}; NaN payload identity is not required"
        );
    }
    Ok(())
}

#[test]
fn transposed_output_fanout_stays_on_original_physical_layout() -> Result<()> {
    if !cuda_available() {
        return Ok(());
    }
    let raw = TensorExpr::input("a", vec![32, 64])
        .cast(ValueDType::BF16)
        .matmul(TensorExpr::input("b", vec![64, 64]).cast(ValueDType::BF16));
    let promoted = raw.cast(ValueDType::F32);
    let loss = promoted.clone().transpose().mean_all() + promoted.mean_all();
    let graph: TensorGraph<f32> = loss.into();
    let types = value_dtypes(&graph);
    assert!(
        !graph
            .toposort()
            .into_iter()
            .any(|n| eligible(&graph, n, &types))
    );
    let mut old = MixedExecutor::new(&graph, MatMulPrecision::NativeHalf, true)?;
    let mut new = MixedExecutor::new_with_options(
        &graph,
        MatMulPrecision::NativeHalf,
        true,
        MixedOptions {
            transposed_bf16_outputs: true,
            ..MixedOptions::default()
        },
    )?;
    assert_eq!(new.info().transposed_bf16_output_programs, 0);
    for phase in 0..2 {
        let feed = HashMap::from([
            ("a".into(), vec![0.125 + phase as f32 / 32.; 32 * 64]),
            ("b".into(), vec![0.25; 64 * 64]),
        ]);
        old.upload(feed.clone())?;
        new.upload(feed)?;
        old.run()?;
        new.run()?;
        for n in graph.toposort() {
            if matches!(graph[n], TensorGraphNode::MatMul { .. }) {
                assert_eq!(old.values[&n].source_shape, new.values[&n].source_shape);
                assert_eq!(old.values[&n].view.access, new.values[&n].view.access);
                assert_eq!(old.read(n)?, new.read(n)?);
            }
        }
    }
    Ok(())
}

#[test]
fn transposed_output_strict_precision_keeps_original_layout() -> Result<()> {
    if !cuda_available() {
        return Ok(());
    }
    let graph: TensorGraph<f32> = TensorExpr::input("a", vec![32, 64])
        .cast(ValueDType::BF16)
        .matmul(TensorExpr::input("b", vec![64, 64]).cast(ValueDType::BF16))
        .cast(ValueDType::F32)
        .transpose()
        .into();
    let types = value_dtypes(&graph);
    let product = graph
        .toposort()
        .into_iter()
        .find(|&n| eligible(&graph, n, &types))
        .unwrap();
    let mut candidate = MixedExecutor::new_with_options(
        &graph,
        MatMulPrecision::StrictF32,
        true,
        MixedOptions {
            transposed_bf16_outputs: true,
            ..MixedOptions::default()
        },
    )?;
    assert_eq!(candidate.info().transposed_bf16_output_programs, 0);
    assert_eq!(candidate.values[&product].source_shape, vec![32, 64]);
    let feed = HashMap::from([
        ("a".into(), vec![0.125; 32 * 64]),
        ("b".into(), vec![0.25; 64 * 64]),
    ]);
    candidate.upload(feed)?;
    candidate.run()?;
    assert_eq!(candidate.read(product)?, vec![2.; 32 * 64]);
    Ok(())
}

#[test]
fn actual_linear_bf16_autograd_identity_cast_chain_is_eligible() {
    let weight = Parameter::new(vec![0.125; 64 * 32], vec![64, 32]);
    let bias = Parameter::new(vec![0.03125; 64], vec![64]);
    let output = TensorExpr::input("x", vec![128, 32])
        .linear_bf16(TensorExpr::from(weight.clone()).transpose(), bias.into())
        .unwrap();
    let mut graph = TensorGraph::new();
    let loss = graph.add_expr(&output.cast(ValueDType::F32).mean_all(), false);
    let graph = graph.with_gradients(loss);
    let types = value_dtypes(&graph);
    let products: Vec<_> = graph
        .toposort()
        .into_iter()
        .filter(|&n| matches!(&graph[n],TensorGraphNode::MatMul{shape} if shape==&vec![32,64]))
        .collect();
    assert_eq!(products.len(), 1);
    let product = products[0];
    let wrapper = sole_consumer(&graph, product).unwrap();
    assert!(matches!(
        &graph[wrapper],
        TensorGraphNode::Cast {
            dtype: ValueDType::BF16,
            ..
        }
    ));
    assert!(eligible(&graph, product, &types));
    let (_, promotion) =
        after_identity_casts(&graph, product, ValueDType::BF16, &[32, 64], &types).unwrap();
    assert!(matches!(
        &graph[promotion],
        TensorGraphNode::Cast {
            dtype: ValueDType::F32,
            ..
        }
    ));
    let (_, transpose) =
        after_identity_casts(&graph, promotion, ValueDType::F32, &[32, 64], &types).unwrap();
    assert!(matches!(&graph[transpose],TensorGraphNode::Transpose{shape} if shape==&vec![64,32]));
    assert_eq!(
        graph[graph.gradient_metadata().param_to_grad[&weight.id()]].shape(),
        &vec![64, 32]
    );
}

#[test]
fn identity_wrapper_chains_require_single_consumers_and_exact_shapes_and_types() {
    for before in 0..3 {
        for after in 0..3 {
            let (mut graph, mut types, nodes) = fixture(32, 64, 64);
            let [_, _, product, promotion, transpose] = nodes;
            graph
                .graph
                .remove_edge(graph.graph.find_edge(product, promotion).unwrap());
            graph
                .graph
                .remove_edge(graph.graph.find_edge(promotion, transpose).unwrap());
            let mut cursor = product;
            let mut wrappers = vec![];
            for _ in 0..before {
                let next = graph.graph.add_node(TensorGraphNode::Cast {
                    dtype: ValueDType::BF16,
                    shape: vec![32, 64],
                });
                types.insert(next, ValueDType::BF16);
                graph.graph.add_edge(cursor, next, 0);
                cursor = next;
                wrappers.push(next);
            }
            graph.graph.add_edge(cursor, promotion, 0);
            cursor = promotion;
            for _ in 0..after {
                let next = graph.graph.add_node(TensorGraphNode::Cast {
                    dtype: ValueDType::F32,
                    shape: vec![32, 64],
                });
                types.insert(next, ValueDType::F32);
                graph.graph.add_edge(cursor, next, 0);
                cursor = next;
                wrappers.push(next);
            }
            graph.graph.add_edge(cursor, transpose, 0);
            assert!(eligible(&graph, product, &types));
            for &wrapper in &wrappers {
                let extra = graph.graph.add_node(TensorGraphNode::Reshape {
                    shape: vec![32, 64],
                });
                let edge = graph.graph.add_edge(wrapper, extra, 0);
                assert!(!eligible(&graph, product, &types));
                graph.graph.remove_edge(edge);
                graph.graph.remove_node(extra);
                let saved = graph[wrapper].clone();
                let dtype = types[&wrapper];
                graph.graph[wrapper] = TensorGraphNode::Cast {
                    dtype,
                    shape: vec![16, 128],
                };
                assert!(!eligible(&graph, product, &types));
                graph.graph[wrapper] = saved;
                types.remove(&wrapper);
                assert!(!eligible(&graph, product, &types));
                types.insert(wrapper, dtype);
            }
        }
    }
}

#[test]
fn actual_linear_weight_gradient_transpose_is_contiguous_and_matches_replay() -> Result<()> {
    if !cuda_available() {
        return Ok(());
    }
    for rows in [128, 160, 129] {
        let weight = Parameter::new(
            (0..64 * 32)
                .map(|i| (i % 11) as f32 / 64. - 0.0625)
                .collect(),
            vec![64, 32],
        );
        let bias = Parameter::new(vec![0.03125; 64], vec![64]);
        let output = TensorExpr::input("x", vec![rows, 32])
            .linear_bf16(
                TensorExpr::from(weight.clone()).transpose(),
                bias.clone().into(),
            )
            .unwrap();
        let mut graph = TensorGraph::new();
        let loss = graph.add_expr(&output.cast(ValueDType::F32).mean_all(), false);
        let graph = graph.with_gradients(loss);
        let types = value_dtypes(&graph);
        let matches = graph
            .toposort()
            .into_iter()
            .filter(|&n| eligible(&graph, n, &types))
            .count();
        assert_eq!(matches, usize::from(rows != 129));
        let base_options = MixedOptions {
            physical_order_casts: true,
            shared_memory_reuse: true,
            narrow_bf16_tile: true,
            ..MixedOptions::default()
        };
        let mut old =
            MixedExecutor::new_with_options(&graph, MatMulPrecision::Tf32x3, true, base_options)?;
        let mut new = MixedExecutor::new_with_options(
            &graph,
            MatMulPrecision::Tf32x3,
            true,
            MixedOptions {
                transposed_bf16_outputs: true,
                ..base_options
            },
        )?;
        assert_eq!(new.info().transposed_bf16_output_programs, matches);
        if matches > 0 {
            let gradient = graph.gradient_metadata().param_to_grad[&weight.id()];
            assert!(new.values[&gradient].view.access.is_identity());
            assert_eq!(new.values[&gradient].source_shape, vec![64, 32]);
            assert_eq!(new.values[&gradient].binding.dtype, DType::BF16);
        }
        for phase in 0..3 {
            let feed = HashMap::from([(
                "x".into(),
                (0..rows * 32)
                    .map(|i| ((i + phase) % 17) as f32 / 32. - 0.125)
                    .collect(),
            )]);
            old.upload(feed.clone())?;
            new.upload(feed)?;
            old.run()?;
            new.run()?;
            assert_eq!(old.read(loss)?, new.read(loss)?);
            let expected = old.gradients(&graph)?;
            let actual = new.gradients(&graph)?;
            for id in [weight.id(), bias.id()] {
                assert!(expected[&id].iter().all(|v| v.is_finite()));
                assert_eq!(
                    expected[&id]
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    actual[&id].iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
            }
        }
        assert!(new.info().replay);
    }
    Ok(())
}
