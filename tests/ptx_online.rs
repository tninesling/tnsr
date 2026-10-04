#![cfg(feature = "cuda")]
use std::collections::HashMap;
use tnsr::graph::TensorGraph;
use tnsr::nn::{scaled_dot_product_attention, softmax};
use tnsr::ptx::{PtxExecutor, PtxReductionMode};
use tnsr::tensor::TensorExpr;
use tnsr::{Executor, SimpleExecutor};

fn executor(mode: PtxReductionMode) -> Option<PtxExecutor> {
    match PtxExecutor::try_new_with_reduction_mode(mode) {
        Ok(executor) => Some(executor),
        Err(error)
            if std::env::var("TNSR_REQUIRE_CUDA").as_deref() != Ok("1")
                && error
                    .to_string()
                    .starts_with("Failed to initialize CUDA device 0") =>
        {
            eprintln!("skipping CUDA: {error:#}");
            None
        }
        Err(error) => panic!("{error:#}"),
    }
}
fn values(shape: Vec<usize>, phase: usize) -> TensorExpr<f32> {
    TensorExpr::constant(
        (0..shape.iter().product())
            .map(|i| (((i + phase) % 23) as f32 - 11.0) * 0.09)
            .collect(),
        shape,
    )
}
fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - b).abs() < 3e-4 || (a.is_nan() && b.is_nan()),
            "index {i}: {a} != {b}"
        );
    }
}
fn check(executor: &mut PtxExecutor, graph: &TensorGraph<f32>, regions: usize) -> Vec<f32> {
    let expected = SimpleExecutor::new()
        .execute(graph, HashMap::new())
        .unwrap();
    let actual = executor.execute(graph, HashMap::new()).unwrap();
    close(&actual, &expected);
    let plan = executor.execution_plan().unwrap();
    assert_eq!(
        plan.online_regions().len(),
        regions,
        "{}",
        executor.describe_plan(graph).unwrap()
    );
    actual
}

#[test]
fn online_attention_matches_primitive_reference_without_score_buffers() {
    let Some(mut online) = executor(PtxReductionMode::Online) else {
        return;
    };
    for (batch, heads, queries, keys, features, width) in [
        (1, 1, 3, 5, 7, 6),
        (2, 3, 9, 9, 8, 11),
        (1, 2, 17, 13, 33, 35),
    ] {
        for causal in [false, true] {
            let graph: TensorGraph<f32> = scaled_dot_product_attention(
                values(vec![batch, heads, queries, features], 0),
                values(vec![batch, heads, keys, features], 3),
                values(vec![batch, heads, keys, width], 7),
                causal,
            )
            .into();
            check(&mut online, &graph, 1);
            assert_eq!(online.execution_metrics().kernel_launches, 1);
        }
    }
}
#[test]
fn online_weighted_normalization_handles_extreme_and_masked_scores() {
    let Some(mut online) = executor(PtxReductionMode::Online) else {
        return;
    };
    let scores = TensorExpr::constant(
        vec![
            1000.,
            -1000.,
            999.,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            3.,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
        ],
        vec![3, 3],
    );
    let graph: TensorGraph<f32> = softmax(scores, 1).matmul(values(vec![3, 5], 0)).into();
    check(&mut online, &graph, 1);
}
#[test]
fn online_fusion_preserves_escaping_probabilities_and_strict_policy() {
    let Some(mut online) = executor(PtxReductionMode::Online) else {
        return;
    };
    let probabilities = softmax(values(vec![3, 3], 0), 1);
    let output = probabilities.clone().matmul(values(vec![3, 3], 2)) + probabilities;
    let graph: TensorGraph<f32> = output.into();
    check(&mut online, &graph, 0);
    let graph: TensorGraph<f32> = softmax(values(vec![3, 3], 0), 1)
        .matmul(values(vec![3, 3], 2))
        .into();
    let Some(mut strict) = executor(PtxReductionMode::Strict) else {
        return;
    };
    check(&mut strict, &graph, 0);
}

#[test]
fn online_serial_schedule_and_nonfinite_inputs_match_reference() {
    let Some(mut online) = executor(PtxReductionMode::Online) else {
        return;
    };
    for scores in [
        vec![f32::NEG_INFINITY, 2., 1., 3., f32::NAN, -1.],
        vec![f32::INFINITY, 2., 1., -3., -2., -1.],
        vec![f32::NEG_INFINITY, 2., 1., -3., -2., -1.],
    ] {
        let weights = TensorExpr::constant(
            (0..3 * 129)
                .map(|i| {
                    if i == 0 {
                        f32::NAN
                    } else {
                        (i % 11) as f32 * 0.03
                    }
                })
                .collect(),
            vec![3, 129],
        );
        let graph: TensorGraph<f32> = softmax(TensorExpr::constant(scores, vec![2, 3]), 1)
            .matmul(weights)
            .into();
        check(&mut online, &graph, 1);
        assert!(
            online.execution_plan().unwrap().online_regions()[0]
                .block_threads()
                .is_none()
        );
    }
}

#[test]
fn online_policy_preserves_attention_training_gradients() {
    use tnsr::tensor::Parameter;
    let Some(mut online) = executor(PtxReductionMode::Online) else {
        return;
    };
    let query = Parameter::new(vec![0.2, 0.1, -0.1, 0.3, 0.5, -0.2], vec![1, 3, 2]);
    let key = Parameter::new(vec![0.4, -0.2, 0.1, 0.2, -0.3, 0.5], vec![1, 3, 2]);
    let value = Parameter::new(vec![0.5, -0.2, 0.4, 0.1, 0.6, -0.1], vec![1, 3, 2]);
    let graph: TensorGraph<f32> = scaled_dot_product_attention(query, key, value, true)
        .mean_all()
        .into();
    let loss = *graph.toposort().last().unwrap();
    let graph = graph.with_gradients(loss);
    let mut cpu = SimpleExecutor::new();
    cpu.execute(&graph, HashMap::new()).unwrap();
    online.compile(&graph).unwrap();
    // Backward consumers retain the forward intermediates; the matcher declines
    // contraction rather than silently discarding values needed by autograd.
    assert!(online.execution_plan().unwrap().online_regions().is_empty());
    online.execute_compiled(&graph, HashMap::new()).unwrap();
    let reference = cpu.get_gradients(&graph);
    let actual = online.get_gradients(&graph);
    assert_eq!(actual.len(), reference.len());
    for (parameter, expected) in reference {
        close(&actual[&parameter], &expected);
    }
}

#[test]
fn online_regions_compose_and_keep_input_views_virtual() {
    let Some(mut online) = executor(PtxReductionMode::Online) else {
        return;
    };
    let first = softmax(values(vec![4, 5], 0), 1).matmul(values(vec![5, 3], 2));
    let second = softmax(first, 1).matmul(values(vec![3, 7], 7));
    let graph: TensorGraph<f32> = second.into();
    check(&mut online, &graph, 2);
    assert_eq!(online.execution_plan().unwrap().online_regions().len(), 2);
    assert_eq!(online.execution_metrics().kernel_launches, 2);
}

#[test]
fn online_policy_fuses_attention_inside_transformer_blocks() {
    let Some(mut online) = executor(PtxReductionMode::Online) else {
        return;
    };
    let block = tnsr::nn::TransformerBlock::new(8, 2, 16, 1e-5);
    let graph: TensorGraph<f32> = block.forward(values(vec![2, 5, 8], 4), true).into();
    check(&mut online, &graph, 1);
}

#[test]
fn online_normalizer_rewrite_works_without_normalized_or_weighted_consumers() {
    use tnsr::tile::OnlineConsumer;
    let Some(mut online) = executor(PtxReductionMode::Online) else {
        return;
    };
    for shape in [vec![7], vec![3, 37], vec![2, 3, 65]] {
        let axis = shape.len() - 1;
        let scores = values(shape.clone(), 4);
        let shifted = scores.clone() - scores.reduce_max(axis).broadcast_axis(axis, shape[axis]);
        let graph: TensorGraph<f32> = shifted.exp().reduce_sum(axis).into();
        check(&mut online, &graph, 1);
        let regions = online.execution_plan().unwrap().online_regions();

        assert!(matches!(regions[0].consumer, OnlineConsumer::Normalizer));
        assert_eq!(online.execution_metrics().kernel_launches, 1);
        assert_eq!(
            online.execution_metrics().intermediate_materialized_bytes,
            0
        );
    }
    let scores = TensorExpr::constant(vec![f32::NEG_INFINITY; 35], vec![1, 35]);
    let graph: TensorGraph<f32> = (scores.clone() - scores.reduce_max(1).broadcast_axis(1, 35))
        .exp()
        .reduce_sum(1)
        .into();
    assert!(online.execute(&graph, HashMap::new()).unwrap()[0].is_nan());
}

#[test]
fn online_consumer_fuses_elementwise_weighted_sum_without_matmuls() {
    use tnsr::tile::OnlineConsumer;
    let Some(mut online) = executor(PtxReductionMode::Online) else {
        return;
    };
    for shape in [vec![35], vec![3, 37], vec![2, 3, 65]] {
        let axis = shape.len() - 1;
        for reversed in [false, true] {
            let probability = softmax(values(shape.clone(), 0), axis);
            let weights = values(shape.clone(), 7);
            let product = if reversed {
                weights * probability
            } else {
                probability * weights
            };
            let graph: TensorGraph<f32> = product.reduce_sum(axis).into();
            check(&mut online, &graph, 1);
            let regions = online.execution_plan().unwrap().online_regions();
            assert!(matches!(regions[0].consumer, OnlineConsumer::Sum(_)));
            assert_eq!(online.execution_metrics().kernel_launches, 1);
        }
    }
}

#[test]
fn online_matmul_and_explicit_indexed_sums_share_contractions() {
    let Some(mut online) = executor(PtxReductionMode::Online) else {
        return;
    };
    let (queries, keys, features, width) = (3, 35, 7, 5);
    for prefix in [vec![], vec![2]] {
        let shape = |tail: &[usize]| [prefix.as_slice(), tail].concat();
        let axis = prefix.len();
        let q = values(shape(&[queries, features]), 0);
        let k = values(shape(&[keys, features]), 3);
        let v = values(shape(&[keys, width]), 7);
        let mut canonical: Option<Vec<f32>> = None;
        for explicit_score in [false, true] {
            for explicit_output in [false, true] {
                let scores = if explicit_score {
                    (q.clone()
                        .reshape(shape(&[queries, 1, features]))
                        .broadcast_axis(axis + 1, keys)
                        * k.clone()
                            .reshape(shape(&[1, keys, features]))
                            .broadcast_axis(axis, queries))
                    .reduce_sum(axis + 2)
                    .reshape(shape(&[queries, keys]))
                } else {
                    q.clone().matmul(k.clone().transpose())
                };
                let probability = softmax(scores, axis + 1);
                let output = if explicit_output {
                    (probability
                        .reshape(shape(&[queries, keys, 1]))
                        .broadcast_axis(axis + 2, width)
                        * v.clone()
                            .reshape(shape(&[1, keys, width]))
                            .broadcast_axis(axis, queries))
                    .reduce_sum(axis + 1)
                    .reshape(shape(&[queries, width]))
                } else {
                    probability.matmul(v.clone())
                };
                let graph: TensorGraph<f32> = output.into();
                let actual = check(&mut online, &graph, 1);
                if let Some(reference) = &canonical {
                    close(&actual, reference);
                } else {
                    canonical = Some(actual);
                }
                assert_eq!(online.execution_metrics().kernel_launches, 1);
                assert_eq!(
                    online.execution_metrics().intermediate_materialized_bytes,
                    0
                );
            }
        }
    }
}

#[test]
fn online_indexed_consumer_requires_an_invariant_normalizer() {
    let Some(mut online) = executor(PtxReductionMode::Online) else {
        return;
    };
    // Transposition moves the row-specific denominator onto the reduction axis.
    // Moving this division outside the contraction would change the result.
    let graph: TensorGraph<f32> = softmax(values(vec![3, 3], 0), 1)
        .transpose()
        .matmul(values(vec![3, 5], 7))
        .into();
    check(&mut online, &graph, 0);
}
