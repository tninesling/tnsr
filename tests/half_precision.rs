use half::{bf16, f16};
use std::collections::HashMap;
use tnsr::tensor::TensorExpr;
use tnsr::{Executor, SimpleExecutor};

/// Helper to convert f32 slice to generic type
fn to_dtype<D: num_traits::Float>(vals: &[f32]) -> Vec<D> {
    vals.iter().map(|&v| D::from(v).unwrap()).collect()
}

/// Helper to convert generic type back to f32 for comparison
fn to_f32<D: num_traits::Float>(vals: &[D]) -> Vec<f32> {
    vals.iter().map(|&v| v.to_f32().unwrap()).collect()
}

/// Compare results with tolerance appropriate for the dtype
fn assert_close<D: num_traits::Float>(result: &[D], expected: &[f32], tolerance: f32) {
    let result_f32 = to_f32(result);
    assert_eq!(result_f32.len(), expected.len());
    for (i, (&r, &e)) in result_f32.iter().zip(expected.iter()).enumerate() {
        let diff = (r - e).abs();
        assert!(
            diff <= tolerance,
            "Mismatch at index {}: got {}, expected {}, diff {}",
            i,
            r,
            e,
            diff
        );
    }
}

#[test]
fn test_f16_basic_ops() {
    // Create a simple computation: x + 2.0
    let x = TensorExpr::<f16>::input("x", vec![4]);
    let two = TensorExpr::<f16>::constant(to_dtype(&[2.0, 2.0, 2.0, 2.0]), vec![4]);
    let y = x + two;

    let mut executor = SimpleExecutor::<f16>::new();
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), to_dtype(&[1.0, 2.0, 3.0, 4.0]));

    let result = executor.execute(&y.into(), inputs).unwrap();

    // f16 has ~3 decimal digits of precision
    assert_close(&result, &[3.0, 4.0, 5.0, 6.0], 0.001);
}

#[test]
fn test_bf16_basic_ops() {
    // Create a simple computation: x * 2.0
    let x = TensorExpr::<bf16>::input("x", vec![4]);
    let two = TensorExpr::<bf16>::constant(to_dtype(&[2.0, 2.0, 2.0, 2.0]), vec![4]);
    let y = x * two;

    let mut executor = SimpleExecutor::<bf16>::new();
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), to_dtype(&[1.0, 2.0, 3.0, 4.0]));

    let result = executor.execute(&y.into(), inputs).unwrap();

    // bf16 has ~2-3 decimal digits of precision
    assert_close(&result, &[2.0, 4.0, 6.0, 8.0], 0.01);
}

#[test]
fn test_f16_transcendental() {
    // Test exp on f16
    let x = TensorExpr::<f16>::input("x", vec![3]);
    let y = x.exp();

    let mut executor = SimpleExecutor::<f16>::new();
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), to_dtype(&[0.0, 1.0, 2.0]));

    let result = executor.execute(&y.into(), inputs).unwrap();

    // Expected: [e^0, e^1, e^2] = [1.0, 2.718..., 7.389...]
    assert_close(&result, &[1.0, std::f32::consts::E, 7.38906], 0.01);
}

#[test]
fn test_bf16_matmul() {
    // Test 2x2 matmul with bf16
    let a = TensorExpr::<bf16>::input("a", vec![2, 2]);
    let b = TensorExpr::<bf16>::input("b", vec![2, 2]);
    let c = a.matmul(b);

    let mut executor = SimpleExecutor::<bf16>::new();
    let mut inputs = HashMap::new();
    // A = [[1, 2], [3, 4]]
    inputs.insert("a".to_string(), to_dtype(&[1.0, 2.0, 3.0, 4.0]));
    // B = [[5, 6], [7, 8]]
    inputs.insert("b".to_string(), to_dtype(&[5.0, 6.0, 7.0, 8.0]));

    let result = executor.execute(&c.into(), inputs).unwrap();

    // Expected: [[19, 22], [43, 50]]
    assert_close(&result, &[19.0, 22.0, 43.0, 50.0], 0.1);
}

#[test]
fn test_f16_relu() {
    let x = TensorExpr::<f16>::input("x", vec![6]);
    let y = x.relu();

    let mut executor = SimpleExecutor::<f16>::new();
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), to_dtype(&[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]));

    let result = executor.execute(&y.into(), inputs).unwrap();

    assert_close(&result, &[0.0, 0.0, 0.0, 1.0, 2.0, 3.0], 0.001);
}

#[test]
fn test_dtype_comparison() {
    // Same computation in different dtypes
    let input_vals = vec![1.0, 2.0, 3.0, 4.0];

    // f32 (baseline)
    let x_f32 = TensorExpr::<f32>::input("x", vec![4]);
    let y_f32 = x_f32.exp();
    let mut exec_f32 = SimpleExecutor::<f32>::new();
    let mut inputs_f32 = HashMap::new();
    inputs_f32.insert("x".to_string(), input_vals.clone());
    let result_f32 = exec_f32.execute(&y_f32.into(), inputs_f32).unwrap();

    // f16
    let x_f16 = TensorExpr::input("x", vec![4]);
    let y_f16 = x_f16.exp();
    let mut exec_f16 = SimpleExecutor::<f16>::new();
    let mut inputs_f16 = HashMap::new();
    inputs_f16.insert("x".to_string(), to_dtype(&input_vals));
    let result_f16 = exec_f16.execute(&y_f16.into(), inputs_f16).unwrap();

    // bf16
    let x_bf16 = TensorExpr::input("x", vec![4]);
    let y_bf16 = x_bf16.exp();
    let mut exec_bf16 = SimpleExecutor::<bf16>::new();
    let mut inputs_bf16 = HashMap::new();
    inputs_bf16.insert("x".to_string(), to_dtype(&input_vals));
    let result_bf16 = exec_bf16.execute(&y_bf16.into(), inputs_bf16).unwrap();

    // All should be close to f32 baseline, within appropriate tolerances
    assert_close(&result_f16, &result_f32, 0.1);
    assert_close(&result_bf16, &result_f32, 0.1);
}
