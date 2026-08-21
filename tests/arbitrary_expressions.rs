use std::collections::HashMap;

use arbitrary::{Arbitrary, Unstructured};
use rand::rngs::StdRng;
use rand::{Rng, RngCore, SeedableRng};
use tnsr::graph::TensorGraph;
use tnsr::tensor::{BinaryOp, ExprKind, TensorExpr, UnaryOp};
use tnsr::{Executor, SimpleExecutor};

const CASES: usize = 512;
const CASE_BYTES: usize = 4096;

fn evaluate(expr: &TensorExpr<f32>, inputs: &HashMap<String, Vec<f32>>) -> Vec<f32> {
    match expr.kind() {
        ExprKind::Input { name } => inputs[*name].clone(),
        ExprKind::Unary { op, x } => evaluate(x, inputs)
            .into_iter()
            .map(|value| match op {
                UnaryOp::Neg => -value,
                UnaryOp::Relu => value.max(0.0),
                _ => panic!("arbitrary expressions must use finite-domain unary operations"),
            })
            .collect(),
        ExprKind::Binary { op, a, b } => evaluate(a, inputs)
            .into_iter()
            .zip(evaluate(b, inputs))
            .map(|(a, b)| match op {
                BinaryOp::Add => a + b,
                BinaryOp::Sub => a - b,
                BinaryOp::Mul => a * b,
                BinaryOp::Div => {
                    panic!("arbitrary expressions must not contain division")
                }
            })
            .collect(),
        _ => panic!("arbitrary expression contains an unsupported operation"),
    }
}

#[test]
fn arbitrary_expressions_lower_and_match_reference() {
    let mut rng = StdRng::seed_from_u64(0x0F02_21A6);

    for case in 0..CASES {
        let mut bytes = [0u8; CASE_BYTES];
        rng.fill_bytes(&mut bytes);
        let mut unstructured = Unstructured::new(&bytes);
        let expr = TensorExpr::<f32>::arbitrary(&mut unstructured).unwrap_or_else(|error| {
            panic!("case {case} could not generate an expression: {error}")
        });

        let element_count = expr.shape().iter().product();
        let mut inputs = HashMap::new();
        for name in ["x", "y"] {
            let values = (0..element_count)
                .map(|_| f32::from(rng.random_range(-16i8..=16)) / 16.0)
                .collect();
            inputs.insert(name.to_string(), values);
        }

        let expected = evaluate(&expr, &inputs);
        let graph: TensorGraph<f32> = expr.into();
        let actual = SimpleExecutor::new()
            .execute(&graph, inputs)
            .unwrap_or_else(|error| panic!("case {case} failed to execute: {error:#}"));

        assert_eq!(actual.len(), expected.len(), "case {case} length mismatch");
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                actual.is_finite() && expected.is_finite(),
                "case {case}, element {index}: non-finite result ({actual} vs {expected})"
            );
            let tolerance = 1e-5 * (1.0 + expected.abs());
            assert!(
                (actual - expected).abs() <= tolerance,
                "case {case}, element {index}: {actual} != {expected} (tolerance {tolerance})"
            );
        }
    }
}
