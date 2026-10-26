use petgraph::visit::EdgeRef;
use std::collections::{HashMap, HashSet};
use tnsr::graph::{GradientOptions, NodeIndex, TensorGraph};
use tnsr::nn::{scaled_dot_product_attention, softmax};
use tnsr::tensor::{Parameter, TensorExpr, ValueDType};
use tnsr::{Executor, SimpleExecutor};

fn options() -> GradientOptions {
    GradientOptions {
        compact_softmax: true,
    }
}

#[test]
fn escaped_sum_maximum_and_unstabilized_division_are_not_rewritten() {
    for escaping in 0..3 {
        let p = Parameter::new(vec![0.1_f32, 0.3, -0.2, 0.8], vec![2, 2]);
        let x: TensorExpr<f32> = TensorExpr::from(p);
        let maximum = x.clone().reduce_max(1);
        let exp = (x - maximum.clone().broadcast(vec![2, 2])).exp();
        let sum = exp.clone().reduce_sum(1);
        let probability = exp.clone() / sum.clone().broadcast(vec![2, 2]);
        let escaped = match escaping {
            0 => maximum.mean_all(),
            1 => sum.mean_all(),
            _ => exp.mean_all(),
        };
        let mut graph = TensorGraph::new();
        let loss = graph.add_expr(&(probability.mean_all() + escaped), false);
        assert_eq!(
            graph.clone().with_gradients(loss).len(),
            graph.with_gradients_with_options(loss, options()).len()
        );
    }
    let p = Parameter::new(vec![0.1_f32, 0.3, -0.2, 0.8], vec![2, 2]);
    let exp = TensorExpr::from(p).exp();
    let loss = (exp.clone() / exp.reduce_sum(1).broadcast(vec![2, 2])).mean_all();
    let graph: TensorGraph<f32> = loss.into();
    let loss = *graph.toposort().last().unwrap();
    assert_eq!(
        graph.clone().with_gradients(loss).len(),
        graph.with_gradients_with_options(loss, options()).len()
    );
}

#[cfg(feature = "cuda")]
fn cuda_available() -> bool {
    match cudarc::driver::CudaContext::new(0) {
        Ok(_) => true,
        Err(error) if std::env::var("TNSR_REQUIRE_CUDA").as_deref() != Ok("1") => {
            eprintln!("skipping CUDA: {error:#}");
            false
        }
        Err(error) => panic!("{error:#}"),
    }
}

#[cfg(feature = "cuda")]
#[test]
fn compact_attention_cuda_qkv_gradients_and_replay_match_f64_oracle() {
    use tnsr::ptx::mixed::MixedExecutor;
    use tnsr::tile::MatMulPrecision;
    if !cuda_available() {
        return;
    }
    for causal in [false, true] {
        let mut fixture = attention(causal);
        let graph = fixture
            .forward
            .clone()
            .with_gradients_with_options(fixture.loss, options());
        let mut gpu = MixedExecutor::new(&graph, MatMulPrecision::StrictF32, true).unwrap();
        assert!(gpu.info().replay);
        for iteration in 0..3 {
            if iteration > 0 {
                for (parameter, values) in fixture.parameters.iter().zip(&mut fixture.values) {
                    parameter
                        .set_values(
                            values
                                .iter()
                                .map(|&value| (value * 0.7 + 0.03) as f32)
                                .collect(),
                        )
                        .unwrap();
                    *values = parameter
                        .values()
                        .unwrap()
                        .into_iter()
                        .map(f64::from)
                        .collect();
                }
            }
            let expected = finite_differences(&fixture);
            gpu.upload(HashMap::new()).unwrap();
            gpu.run().unwrap();
            let gradients = gpu.gradients(&graph).unwrap();
            for (parameter, expected) in fixture.parameters.iter().zip(&expected) {
                compare(&gradients[&parameter.id()], expected, 3e-7);
            }
            assert!(
                (f64::from(gpu.read(fixture.loss).unwrap()[0])
                    - objective(&fixture.values, causal))
                .abs()
                    < 1e-7
            );
        }
    }
}

/// Manual diagnostic: compilation/uploads excluded; alternating replay timings.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "manual GPU timing; run with exclusive GPU access"]
fn compact_softmax_cuda_replay_timing() {
    use tnsr::ptx::mixed::MixedExecutor;
    use tnsr::tile::MatMulPrecision;
    if !cuda_available() {
        return;
    }
    let shape = vec![16, 512];
    let p = Parameter::new(
        (0..8192)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.2)
            .collect(),
        shape.clone(),
    );
    let weights = TensorExpr::constant(
        (0..8192)
            .map(|i| ((i * 17 % 101) as f32 - 50.0) * 0.01)
            .collect(),
        shape,
    );
    let mut graph = TensorGraph::new();
    let loss = graph.add_expr(&(softmax(p.clone(), 1) * weights).mean_all(), false);
    let primitive = graph.clone().with_gradients(loss);
    let compact = graph.with_gradients_with_options(loss, options());
    let mut baseline = MixedExecutor::new(&primitive, MatMulPrecision::StrictF32, true).unwrap();
    let mut candidate = MixedExecutor::new(&compact, MatMulPrecision::StrictF32, true).unwrap();
    baseline.upload(HashMap::new()).unwrap();
    candidate.upload(HashMap::new()).unwrap();
    for _ in 0..10 {
        baseline.run().unwrap();
        candidate.run().unwrap();
    }
    let mut baseline_us = Vec::new();
    let mut candidate_us = Vec::new();
    for round in 0..9 {
        if round % 2 == 0 {
            baseline_us.push(baseline.time_runs(200).unwrap());
            candidate_us.push(candidate.time_runs(200).unwrap());
        } else {
            candidate_us.push(candidate.time_runs(200).unwrap());
            baseline_us.push(baseline.time_runs(200).unwrap());
        }
    }
    compare(
        &candidate.gradients(&compact).unwrap()[&p.id()],
        &baseline.gradients(&primitive).unwrap()[&p.id()],
        1e-8,
    );
    let mut baseline_sorted = baseline_us.clone();
    let mut candidate_sorted = candidate_us.clone();
    let mut paired_ratios: Vec<_> = baseline_us
        .iter()
        .zip(&candidate_us)
        .map(|(a, b)| a / b)
        .collect();
    baseline_sorted.sort_by(f64::total_cmp);
    candidate_sorted.sort_by(f64::total_cmp);
    paired_ratios.sort_by(f64::total_cmp);
    eprintln!(
        "compact_softmax_replay baseline_us={baseline_us:?} compact_us={candidate_us:?} median_speedup={} baseline_info={:?} compact_info={:?}",
        baseline_sorted[4] / candidate_sorted[4],
        baseline.info(),
        candidate.info()
    );
    eprintln!(
        "paired_ratio_median={} even_rounds=baseline_first odd_rounds=compact_first",
        paired_ratios[4]
    );
}

fn compare(a: &[f32], b: &[f32], tolerance: f32) {
    assert_eq!(a.len(), b.len());
    for (index, (&a, &b)) in a.iter().zip(b).enumerate() {
        assert!(
            (a - b).abs() <= tolerance || a.is_nan() && b.is_nan(),
            "element {index}: {a} != {b}"
        );
    }
}

#[test]
fn compact_softmax_is_opt_in_handles_axes_and_probability_fanout_and_roundtrips() {
    for (shape, axis) in [(vec![3, 5], 1), (vec![3, 5], 0), (vec![2, 1, 3], 1)] {
        let len: usize = shape.iter().product();
        let p = Parameter::new(
            (0..len).map(|i| (i as f32 - 4.0) * 0.2).collect(),
            shape.clone(),
        );
        let probabilities = softmax(p.clone(), axis);
        let weights = TensorExpr::constant((0..len).map(|i| (i % 7) as f32 - 3.0).collect(), shape);
        let loss =
            (probabilities.clone() * weights + probabilities.clone() * probabilities).mean_all();
        let mut forward = TensorGraph::new();
        let loss_node = forward.add_expr(&loss, false);
        let default = forward.clone().with_gradients(loss_node);
        let explicit_default = forward
            .clone()
            .with_gradients_with_options(loss_node, GradientOptions::default());
        let compact = forward
            .clone()
            .with_gradients_with_options(loss_node, options());
        assert_eq!(default.len(), explicit_default.len());
        assert!(compact.len() < default.len());
        let mut cpu = SimpleExecutor::new();
        cpu.execute(&default, HashMap::new()).unwrap();
        let reference = cpu.get_gradients(&default);
        let loss = cpu.get_value(loss_node).unwrap();
        cpu.execute(&compact, HashMap::new()).unwrap();
        compare(
            &cpu.get_gradients(&compact)[&p.id()],
            &reference[&p.id()],
            2e-7,
        );
        assert_eq!(cpu.get_value(loss_node).unwrap(), loss);
        let restored = compact.without_gradients();
        assert_eq!(forward.len(), restored.len());
        for node in forward.graph.node_indices() {
            assert_eq!(forward[node].shape(), restored[node].shape());
            assert_eq!(
                std::mem::discriminant(&forward[node]),
                std::mem::discriminant(&restored[node])
            );
        }
        let edges = |g: &TensorGraph<f32>| {
            g.graph
                .edge_references()
                .map(|e| (e.source(), e.target(), *e.weight()))
                .collect::<HashSet<_>>()
        };
        assert_eq!(edges(&forward), edges(&restored));
    }
}

#[test]
fn escaping_normalizer_and_any_cast_keep_primitive_derivatives() {
    let p = Parameter::new(vec![0.1, 0.3, -0.2, 0.8], vec![2, 2]);
    let x = TensorExpr::from(p);
    let exp = (x.clone() - x.reduce_max(1).broadcast(vec![2, 2])).exp();
    let probabilities = exp.clone() / exp.clone().reduce_sum(1).broadcast(vec![2, 2]);
    for loss in [
        (probabilities + exp).mean_all(),
        softmax(Parameter::new(vec![0.1, 0.2], vec![1, 2]), 1)
            .cast(ValueDType::BF16)
            .mean_all(),
    ] {
        let mut graph = TensorGraph::new();
        let loss = graph.add_expr(&loss, false);
        let reference = graph.clone().with_gradients(loss);
        let compact = graph.with_gradients_with_options(loss, options());
        assert_eq!(reference.len(), compact.len());
        assert_eq!(
            tnsr::graph::mixed::value_dtypes(&reference),
            tnsr::graph::mixed::value_dtypes(&compact)
        );
        let mut cpu = SimpleExecutor::new();
        cpu.execute(&reference, HashMap::new()).unwrap();
        let expected = cpu.get_gradients(&reference);
        cpu.execute(&compact, HashMap::new()).unwrap();
        for (id, values) in expected {
            compare(&values, &cpu.get_gradients(&compact)[&id], 0.0);
        }
    }
    let p = Parameter::new(vec![0.1_f64, 0.2], vec![1, 2]);
    let graph: TensorGraph<f64> = softmax(p, 1).mean_all().into();
    let loss = *graph.toposort().last().unwrap();
    assert_eq!(
        graph.clone().with_gradients(loss).len(),
        graph.with_gradients_with_options(loss, options()).len()
    );
}

#[test]
fn compact_softmax_defines_masked_and_nonfinite_rows() {
    for values in [
        vec![f32::NEG_INFINITY, 2.0, 3.0],
        vec![f32::NEG_INFINITY; 3],
        vec![f32::INFINITY, 2.0, 3.0],
        vec![f32::NAN, 2.0, 3.0],
    ] {
        let p = Parameter::new(values.clone(), vec![1, 3]);
        let probabilities: TensorExpr<f32> = softmax(p.clone(), 1);
        let loss =
            (probabilities * TensorExpr::constant(vec![1.0, -2.0, 0.5], vec![1, 3])).mean_all();
        let mut graph = TensorGraph::new();
        let loss = graph.add_expr(&loss, false);
        let graph = graph.with_gradients_with_options(loss, options());
        let mut cpu = SimpleExecutor::new();
        cpu.execute(&graph, HashMap::new()).unwrap();
        let gradients = cpu.get_gradients(&graph)[&p.id()].clone();
        if values[1].is_finite() && values[0] == f32::NEG_INFINITY {
            assert_eq!(gradients[0], 0.0);
            assert!(gradients.iter().all(|g| g.is_finite()));
        } else {
            assert!(gradients.iter().all(|g| g.is_nan()));
        }
    }
}

#[test]
fn compact_softmax_nonfinite_incoming_gradient_uses_documented_formula() {
    for (incoming, expected) in [
        (vec![f32::INFINITY, 0.0], vec![f32::NAN, f32::NEG_INFINITY]),
        (vec![f32::NAN, 0.0], vec![f32::NAN, f32::NAN]),
    ] {
        let p = Parameter::new(vec![0.2_f32, 0.3], vec![1, 2]);
        let probability = softmax(p.clone(), 1);
        let mut graph = TensorGraph::<f32>::new();
        let loss = graph.add_expr(
            &(probability * TensorExpr::constant(incoming, vec![1, 2])).reduce_sum(1),
            false,
        );
        let graph = graph.with_gradients_with_options(loss, options());
        let mut cpu = SimpleExecutor::new();
        cpu.execute(&graph, HashMap::new()).unwrap();
        let actual = cpu.get_gradients(&graph)[&p.id()].clone();
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!(actual == expected || actual.is_nan() && expected.is_nan());
        }
    }
}

struct Attention {
    forward: TensorGraph<f32>,
    loss: NodeIndex,
    parameters: Vec<Parameter<f32>>,
    values: Vec<Vec<f64>>,
    causal: bool,
}

fn attention(causal: bool) -> Attention {
    let shapes = [vec![1, 3, 3], vec![1, 4, 3], vec![1, 4, 2]];
    let parameters: Vec<_> = shapes
        .into_iter()
        .enumerate()
        .map(|(phase, shape)| {
            Parameter::new(
                (0..shape.iter().product())
                    .map(|i| (((i * 7 + phase * 3) % 19) as f32 - 9.0) * 0.07)
                    .collect(),
                shape,
            )
        })
        .collect();
    let values = parameters
        .iter()
        .map(|p| p.values().unwrap().iter().map(|&x| f64::from(x)).collect())
        .collect();
    let output = scaled_dot_product_attention(
        parameters[0].clone(),
        parameters[1].clone(),
        parameters[2].clone(),
        causal,
    );
    let weights = TensorExpr::constant(vec![0.3, -0.7, 0.2, 0.5, -0.6, 0.4], vec![1, 3, 2]);
    let mut forward = TensorGraph::new();
    let loss = forward.add_expr(&(output * weights).mean_all(), false);
    Attention {
        forward,
        loss,
        parameters,
        values,
        causal,
    }
}

/// Independent f64 loops, without tnsr tensor expressions or autograd.
fn objective(values: &[Vec<f64>], causal: bool) -> f64 {
    let weights = [0.3, -0.7, 0.2, 0.5, -0.6, 0.4];
    let mut total = 0.0;
    for query in 0..3 {
        let mut scores = [f64::NEG_INFINITY; 4];
        for (key, score) in scores.iter_mut().enumerate() {
            if !causal || key <= query {
                *score = (0..3)
                    .map(|d| values[0][query * 3 + d] * values[1][key * 3 + d])
                    .sum::<f64>()
                    / 3_f64.sqrt();
            }
        }
        let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exp = scores.map(|s| (s - maximum).exp());
        let sum: f64 = exp.iter().sum();
        for d in 0..2 {
            let output: f64 = (0..4)
                .map(|key| exp[key] / sum * values[2][key * 2 + d])
                .sum();
            total += output * weights[query * 2 + d] / 6.0;
        }
    }
    total
}

fn finite_differences(fixture: &Attention) -> Vec<Vec<f32>> {
    let mut values = fixture.values.clone();
    (0..3)
        .map(|parameter| {
            (0..values[parameter].len())
                .map(|index| {
                    let original = values[parameter][index];
                    values[parameter][index] = original + 1e-6;
                    let plus = objective(&values, fixture.causal);
                    values[parameter][index] = original - 1e-6;
                    let minus = objective(&values, fixture.causal);
                    values[parameter][index] = original;
                    ((plus - minus) / 2e-6) as f32
                })
                .collect()
        })
        .collect()
}

#[test]
fn compact_attention_qkv_gradients_match_independent_f64_finite_differences() {
    for causal in [false, true] {
        let fixture = attention(causal);
        let expected = finite_differences(&fixture);
        let original_count = fixture.forward.clone().with_gradients(fixture.loss).len();
        let graph = fixture
            .forward
            .with_gradients_with_options(fixture.loss, options());
        assert!(graph.len() < original_count);
        let mut cpu = SimpleExecutor::new();
        cpu.execute(&graph, HashMap::new()).unwrap();
        let gradients = cpu.get_gradients(&graph);
        for (parameter, expected) in fixture.parameters.iter().zip(expected) {
            compare(&gradients[&parameter.id()], &expected, 2e-7);
        }
        assert!(
            (f64::from(cpu.get_value(fixture.loss).unwrap()[0])
                - objective(&fixture.values, causal))
            .abs()
                < 1e-7
        );
    }
}
