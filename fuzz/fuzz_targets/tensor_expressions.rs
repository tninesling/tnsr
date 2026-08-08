#![no_main]

use std::collections::HashMap;

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use tnsr::graph::TensorGraph;
use tnsr::tensor::{BinaryOp, ExprKind, TensorExpr, UnaryOp};
use tnsr::{Executor, SimpleExecutor};

fn evaluate(expr: &TensorExpr<f32>, inputs: &HashMap<String, Vec<f32>>) -> Vec<f32> {
    match expr.kind() {
        ExprKind::Input { name } => inputs[*name].clone(),
        ExprKind::Unary { op, x } => evaluate(x, inputs)
            .into_iter()
            .map(|value| match op {
                UnaryOp::Neg => -value,
                UnaryOp::Relu => value.max(0.0),
                _ => unreachable!("the generator only emits finite-domain unary operations"),
            })
            .collect(),
        ExprKind::Binary { op, a, b } => evaluate(a, inputs)
            .into_iter()
            .zip(evaluate(b, inputs))
            .map(|(a, b)| match op {
                BinaryOp::Add => a + b,
                BinaryOp::Sub => a - b,
                BinaryOp::Mul => a * b,
                BinaryOp::Div => unreachable!("the generator does not emit division"),
            })
            .collect(),
        _ => unreachable!("the generator only emits pointwise expressions"),
    }
}

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }

    let mut unstructured = Unstructured::new(data);
    let Ok(expr) = TensorExpr::<f32>::arbitrary(&mut unstructured) else {
        return;
    };

    let element_count = expr.shape().iter().product();
    let mut inputs = HashMap::new();
    for name in ["x", "y"] {
        let values = (0..element_count)
            .map(|index| f32::from(data[index % data.len()] as i8) / 128.0)
            .collect();
        inputs.insert(name.to_string(), values);
    }

    let expected = evaluate(&expr, &inputs);
    let graph: TensorGraph<f32> = expr.into();
    let actual = SimpleExecutor::new()
        .execute(&graph, inputs)
        .expect("generated expression must execute");

    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(actual.is_finite() && expected.is_finite());
        let tolerance = 1e-5 * (1.0 + expected.abs());
        assert!((actual - expected).abs() <= tolerance);
    }
});
