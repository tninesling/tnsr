use super::{TensorGraph, WithGrad};
use crate::tensor::{Parameter, Shape, TensorExpr};
use crate::{Executor, SimpleExecutor};
use std::collections::HashMap;

struct MaskCase {
    name: &'static str,
    graph: TensorGraph<f32, WithGrad>,
    values_id: usize,
    condition_id: Option<usize>,
    expected_loss: f32,
    expected_gradient: Vec<f32>,
}

fn direct_case(
    name: &'static str,
    values: (Vec<f32>, Shape),
    condition: (Vec<f32>, Shape),
    weights: Vec<f32>,
    expected_loss: f32,
    expected_gradient: Vec<f32>,
) -> MaskCase {
    let values = Parameter::new(values.0, values.1);
    let condition = Parameter::new(condition.0, condition.1);
    let masked = TensorExpr::from(values.clone()).mask(condition.clone());
    let shape = masked.shape().clone();
    let rank = shape.len();
    let weighted = masked * TensorExpr::constant(weights, shape);
    let loss = (0..rank).fold(weighted, |expression, axis| expression.reduce_sum(axis));
    let graph: TensorGraph<f32> = loss.into();
    let loss_node = *graph.toposort().last().unwrap();
    MaskCase {
        name,
        graph: graph.with_gradients(loss_node),
        values_id: values.id(),
        condition_id: Some(condition.id()),
        expected_loss,
        expected_gradient,
    }
}

fn cases() -> Vec<MaskCase> {
    let mut cases = vec![
        direct_case(
            "nonzero conditions and masked NaN",
            (vec![f32::NAN, 3.0, 4.0, -1.0], vec![4]),
            (vec![0.0, -2.0, 0.5, -0.0], vec![4]),
            vec![7.0, 2.0, -3.0, 11.0],
            -6.0,
            vec![0.0, 2.0, -3.0, 0.0],
        ),
        direct_case(
            "broadcast values accumulate selected rows",
            (vec![1.0, 2.0, 4.0], vec![1, 3]),
            (vec![1.0, 0.0, -1.0, 0.0, 2.0, 1.0], vec![2, 3]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            47.0,
            vec![1.0, 5.0, 9.0],
        ),
        direct_case(
            "broadcast condition preserves values shape",
            (vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]),
            (vec![-1.0, 0.0, 1.0], vec![3]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            62.0,
            vec![1.0, 0.0, 3.0, 4.0, 0.0, 6.0],
        ),
        direct_case(
            "two-sided broadcasting",
            (vec![2.0, 5.0], vec![2, 1]),
            (vec![1.0, 0.0, -2.0], vec![1, 3]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            58.0,
            vec![4.0, 10.0],
        ),
        direct_case(
            "all values masked",
            (vec![2.0, 5.0], vec![2]),
            (vec![0.0, 0.0], vec![2]),
            vec![3.0, 7.0],
            0.0,
            vec![0.0, 0.0],
        ),
    ];

    let values = Parameter::new(vec![0.0, std::f32::consts::LN_2, 3.0f32.ln()], vec![3]);
    let condition = Parameter::new(vec![0.0, 1.0, -1.0], vec![3]);
    let shared = TensorExpr::from(values.clone()).exp();
    let graph: TensorGraph<f32> = (shared.clone().mask(condition.clone()) + shared * 2.0)
        .reduce_sum(0)
        .into();
    let loss_node = *graph.toposort().last().unwrap();
    cases.push(MaskCase {
        name: "shared producer chain accumulates masked and unmasked gradients",
        graph: graph.with_gradients(loss_node),
        values_id: values.id(),
        condition_id: Some(condition.id()),
        expected_loss: 17.0,
        expected_gradient: vec![2.0, 6.0, 9.0],
    });

    let values = Parameter::new(vec![-2.0, 0.0, 3.0], vec![3]);
    let graph: TensorGraph<f32> = TensorExpr::from(values.clone())
        .mask(values.clone())
        .reduce_sum(0)
        .into();
    let loss_node = *graph.toposort().last().unwrap();
    cases.push(MaskCase {
        name: "same tensor supplies values and condition",
        graph: graph.with_gradients(loss_node),
        values_id: values.id(),
        condition_id: None,
        expected_loss: 1.0,
        expected_gradient: vec![1.0, 0.0, 1.0],
    });
    cases
}

fn check_cases(executor: &mut impl Executor<f32>) {
    for case in cases() {
        if let Some(condition_id) = case.condition_id {
            assert!(
                !case
                    .graph
                    .gradient_metadata()
                    .param_to_grad
                    .contains_key(&condition_id),
                "{}: the condition must remain nondifferentiable",
                case.name
            );
        }
        let output = executor.execute(&case.graph, HashMap::new()).unwrap();
        assert_eq!(output.len(), 1);
        assert!(
            (output[0] - case.expected_loss).abs() < 1e-5,
            "{}: expected loss {}, got {output:?}",
            case.name,
            case.expected_loss
        );
        let gradients = executor.get_gradients(&case.graph);
        let actual = gradients
            .get(&case.values_id)
            .unwrap_or_else(|| panic!("{}: values gradient is missing", case.name));
        assert_eq!(actual.len(), case.expected_gradient.len());
        for (&actual, &expected) in actual.iter().zip(&case.expected_gradient) {
            assert!(
                (actual - expected).abs() < 1e-5,
                "{}: expected gradient {expected}, got {actual}",
                case.name
            );
        }
        if let Some(condition_id) = case.condition_id {
            assert!(!gradients.contains_key(&condition_id));
        }
    }
}

#[test]
fn cpu_mask_values_gradients() {
    check_cases(&mut SimpleExecutor::new());
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_mask_values_gradients() {
    let mut executor = match crate::ptx::PtxExecutor::try_new() {
        Ok(executor) => executor,
        Err(error)
            if std::env::var("TNSR_REQUIRE_CUDA").as_deref() != Ok("1")
                && error
                    .to_string()
                    .starts_with("Failed to initialize CUDA device 0") =>
        {
            eprintln!("skipping CUDA test: {error:#}");
            return;
        }
        Err(error) => panic!("PTX initialization failed: {error:#}"),
    };
    check_cases(&mut executor);
}
