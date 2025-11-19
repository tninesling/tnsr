//! Integration tests for graph execution.
//!
//! Tests forward and backward passes on computation graphs using both CPU and GPU executors.
//! The same test logic runs on both executors to ensure consistency across platforms.

use std::collections::HashMap;

use runtime::{Executor, SimpleExecutor};
use tensor::graph::TensorGraph;
use tensor::{Constant, Parameter, TensorExpr};

#[cfg(feature = "cuda")]
use runtime::cuda::CudaExecutor;

const EPSILON: f32 = 1e-5;

/// Helper macro to assert approximate equality for float vectors.
macro_rules! assert_approx_eq {
    ($a:expr, $b:expr) => {
        if (&$a)
            .iter()
            .zip((&$b).iter())
            .any(|(a, b)| (a - b).abs() > EPSILON)
        {
            panic!(
                "assertion failed: `(left ~= right)` (left: `{:?}`, right: `{:?}`)",
                $a, $b
            );
        }
    };
}

/// Creates an executor based on the current feature flags.
///
/// Returns a CudaExecutor if the cuda feature is enabled, otherwise returns SimpleExecutor.
#[cfg(feature = "cuda")]
fn create_executor() -> impl Executor<f32> {
    CudaExecutor::new()
}

/// Creates an executor based on the current feature flags.
///
/// Returns a CudaExecutor if the cuda feature is enabled, otherwise returns SimpleExecutor.
#[cfg(not(feature = "cuda"))]
fn create_executor() -> impl Executor<f32> {
    SimpleExecutor::new()
}

// ============================================================================
// Forward Pass Tests
// ============================================================================

#[test]
fn forward_matmul_simple() {
    let mut executor = create_executor();

    // A[2,3] @ B[3,2] = C[2,2]
    let a = Constant::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = Constant::new(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);
    let node = TensorExpr::from(a).matmul(b);

    let graph: TensorGraph<f32> = node.into();
    let result = executor.forward(&graph, HashMap::new()).unwrap();

    // Expected: [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
    //         = [[58, 64], [139, 154]]
    let expected = vec![58.0, 64.0, 139.0, 154.0];
    assert_approx_eq!(result, expected);
}

#[test]
fn forward_broadcast_then_reduce() {
    let mut executor = create_executor();

    // Broadcast [2,1] to [2,3], then reduce_sum along axis 1
    let a = Constant::new(vec![1.0, 2.0], vec![2, 1]);
    let node = TensorExpr::from(a).broadcast(vec![2, 3]).reduce_sum(1);

    let graph: TensorGraph<f32> = node.into();
    let result = executor.forward(&graph, HashMap::new()).unwrap();

    // Expected: [[1,1,1],[2,2,2]] summed along axis 1 = [3, 6]
    let expected = vec![3.0, 6.0];
    assert_approx_eq!(result, expected);
}

#[test]
fn forward_reduce_max() {
    let mut executor = create_executor();

    let a = Constant::new(vec![1.0, -2.0, 3.5, 0.5, 10.0, -1.0], vec![2, 3]);
    let node = TensorExpr::from(a).reduce_max(1);

    let graph: TensorGraph<f32> = node.into();
    let result = executor.forward(&graph, HashMap::new()).unwrap();

    // Expected: max of each row = [3.5, 10.0]
    let expected = vec![3.5, 10.0];
    assert_approx_eq!(result, expected);
}

#[test]
fn forward_reduce_mean() {
    let mut executor = create_executor();

    let a = Constant::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let node = TensorExpr::from(a).reduce_mean(1);

    let graph: TensorGraph<f32> = node.into();
    let result = executor.forward(&graph, HashMap::new()).unwrap();

    // Expected: mean of each row = [2.0, 5.0]
    let expected = vec![2.0, 5.0];
    assert_approx_eq!(result, expected);
}

#[test]
fn forward_chained_operations() {
    let mut executor = create_executor();

    // Complex chain: (a * 2) + (b - 1)
    let a = Constant::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Constant::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
    let two = Constant::new(vec![2.0, 2.0, 2.0, 2.0], vec![2, 2]);
    let one = Constant::new(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);

    let node = (TensorExpr::from(a) * TensorExpr::from(two))
        + (TensorExpr::from(b) - TensorExpr::from(one));

    let graph: TensorGraph<f32> = node.into();
    let result = executor.forward(&graph, HashMap::new()).unwrap();

    // Expected: [2,4,6,8] + [4,5,6,7] = [6,9,12,15]
    let expected = vec![6.0, 9.0, 12.0, 15.0];
    assert_approx_eq!(result, expected);
}

// ============================================================================
// Gradient Tests
// ============================================================================

#[test]
fn gradient_simple_square() {
    let mut executor = create_executor();

    // f(x) = x^2 where x = [2.0]
    // df/dx = 2x = 4.0
    let x = Parameter::new(vec![2.0f32], vec![1]);
    let x_id = x.id();
    let x_sq = TensorExpr::from(x.clone()) * TensorExpr::from(x);

    let graph: TensorGraph<f32> = x_sq.into();
    let loss_node = *graph.toposort().last().unwrap();
    let grad_graph = graph.with_gradients(loss_node);

    executor.forward(&grad_graph, HashMap::new()).unwrap();
    let grads = executor.get_gradients(&grad_graph);

    let grad = grads.get(&x_id).expect("Parameter gradient missing");
    assert_eq!(grad.len(), 1);
    assert!(
        (grad[0] - 4.0).abs() < EPSILON,
        "Expected gradient 4.0, got {}",
        grad[0]
    );
}

#[test]
fn gradient_exp_function() {
    let mut executor = create_executor();

    // f(x) = exp(x) where x = [1.0]
    // df/dx = exp(x) = e ≈ 2.718
    let x = Parameter::new(vec![1.0f32], vec![1]);
    let x_id = x.id();
    let exp_x = TensorExpr::from(x).exp();

    let graph: TensorGraph<f32> = exp_x.into();
    let loss_node = *graph.toposort().last().unwrap();
    let grad_graph = graph.with_gradients(loss_node);

    executor.forward(&grad_graph, HashMap::new()).unwrap();
    let grads = executor.get_gradients(&grad_graph);

    let grad = grads.get(&x_id).expect("Parameter gradient missing");
    assert_eq!(grad.len(), 1);
    let expected = 1.0f32.exp();
    assert!(
        (grad[0] - expected).abs() < EPSILON,
        "Expected gradient {}, got {}",
        expected,
        grad[0]
    );
}

#[test]
fn gradient_matmul_rectangular() {
    let mut executor = create_executor();

    // Test with rectangular matrices: A[2,3] @ B[3,2]
    let a = Parameter::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let a_id = a.id();
    let b = Parameter::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
    let b_id = b.id();

    let node = TensorExpr::from(a).matmul(b).reduce_sum(1).reduce_sum(0);

    let graph: TensorGraph<f32> = node.into();
    let loss_node = *graph.toposort().last().unwrap();
    let grad_graph = graph.with_gradients(loss_node);

    executor.forward(&grad_graph, HashMap::new()).unwrap();
    let grads = executor.get_gradients(&grad_graph);

    // Verify we have gradients for both parameters
    assert_eq!(grads.len(), 2);

    let grad_a = grads.get(&a_id).expect("dA missing");
    let grad_b = grads.get(&b_id).expect("dB missing");

    assert_eq!(grad_a.len(), 6, "dA should have shape [2,3]");
    assert_eq!(grad_b.len(), 6, "dB should have shape [3,2]");

    // Verify all gradients are finite
    for &val in grad_a.iter().chain(grad_b.iter()) {
        assert!(val.is_finite(), "Gradient contains non-finite value");
    }
}

#[test]
fn gradient_matmul_chain_rule() {
    let mut executor = create_executor();

    // z = (A @ B) @ C where B=I and C=2I
    // Result: z = 2A, so dz/dA should be 2*ones
    let a = Parameter::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
    let a_id = a.id();
    let b = Parameter::new(vec![1.0f32, 0.0, 0.0, 1.0], vec![2, 2]); // Identity
    let c = Parameter::new(vec![2.0f32, 0.0, 0.0, 2.0], vec![2, 2]); // 2*Identity

    let ab = TensorExpr::from(a).matmul(b);
    let z = ab.matmul(c).reduce_sum(1).reduce_sum(0);

    let graph: TensorGraph<f32> = z.into();
    let loss_node = *graph.toposort().last().unwrap();
    let grad_graph = graph.with_gradients(loss_node);

    executor.forward(&grad_graph, HashMap::new()).unwrap();
    let grads = executor.get_gradients(&grad_graph);

    let grad_a = grads.get(&a_id).expect("Parameter gradient missing");
    assert_eq!(grad_a.len(), 4);

    // With B=I and C=2I, z = sum(2A), so dz/dA = 2*ones
    for &val in grad_a.iter() {
        assert!(
            (val - 2.0).abs() < EPSILON,
            "Expected gradient 2.0, got {}",
            val
        );
    }
}

#[test]
fn gradient_broadcast_and_reduce() {
    let mut executor = create_executor();

    // Parameter [2,1] broadcast to [2,3], then reduce_sum to scalar
    let x = Parameter::new(vec![1.0, 2.0], vec![2, 1]);
    let x_id = x.id();
    let node = TensorExpr::from(x)
        .broadcast(vec![2, 3])
        .reduce_sum(1)
        .reduce_sum(0);

    let graph: TensorGraph<f32> = node.into();
    let loss_node = *graph.toposort().last().unwrap();
    let grad_graph = graph.with_gradients(loss_node);

    executor.forward(&grad_graph, HashMap::new()).unwrap();
    let grads = executor.get_gradients(&grad_graph);

    let grad = grads.get(&x_id).expect("Parameter gradient missing");
    assert_eq!(grad.len(), 2);

    // Broadcasting replicates each element 3 times, gradient should be [3, 3]
    for &g in grad.iter() {
        assert!(
            (g - 3.0).abs() < EPSILON,
            "Expected gradient 3.0, got {}",
            g
        );
    }
}

#[test]
fn gradient_relu_activation() {
    let mut executor = create_executor();

    // f(x) = relu(x) where x = [-1, 2, -3, 4]
    // df/dx = [0, 1, 0, 1]
    let x = Parameter::new(vec![-1.0, 2.0, -3.0, 4.0], vec![2, 2]);
    let x_id = x.id();
    let node = TensorExpr::from(x).relu().reduce_sum(1).reduce_sum(0);

    let graph: TensorGraph<f32> = node.into();
    let loss_node = *graph.toposort().last().unwrap();
    let grad_graph = graph.with_gradients(loss_node);

    executor.forward(&grad_graph, HashMap::new()).unwrap();
    let grads = executor.get_gradients(&grad_graph);

    let grad = grads.get(&x_id).expect("Parameter gradient missing");
    assert_eq!(grad.len(), 4);

    let expected = vec![0.0, 1.0, 0.0, 1.0];
    assert_approx_eq!(grad, &expected);
}

#[test]
fn gradient_log_function() {
    let mut executor = create_executor();

    // f(x) = log(x) where x = [1, 2, 3, 4]
    // df/dx = 1/x = [1, 0.5, 0.333..., 0.25]
    let x_data = vec![1.0, 2.0, 3.0, 4.0];
    let x = Parameter::new(x_data.clone(), vec![2, 2]);
    let x_id = x.id();
    let node = TensorExpr::from(x).log().reduce_sum(1).reduce_sum(0);

    let graph: TensorGraph<f32> = node.into();
    let loss_node = *graph.toposort().last().unwrap();
    let grad_graph = graph.with_gradients(loss_node);

    executor.forward(&grad_graph, HashMap::new()).unwrap();
    let grads = executor.get_gradients(&grad_graph);

    let grad = grads.get(&x_id).expect("Parameter gradient missing");
    assert_eq!(grad.len(), 4);

    let expected: Vec<f32> = x_data.iter().map(|&v| 1.0 / v).collect();
    assert_approx_eq!(grad, &expected);
}

#[test]
fn gradient_division_operation() {
    let mut executor = create_executor();

    // f(a, b) = sum(a / b) where a = [4, 6], b = [2, 3]
    // df/da = 1/b = [0.5, 0.333...]
    // df/db = -a/b^2 = [-1, -0.666...]
    let a = Parameter::new(vec![4.0, 6.0], vec![2]);
    let a_id = a.id();
    let b = Parameter::new(vec![2.0, 3.0], vec![2]);
    let b_id = b.id();

    let node = (TensorExpr::from(a) / TensorExpr::from(b)).reduce_sum(0);

    let graph: TensorGraph<f32> = node.into();
    let loss_node = *graph.toposort().last().unwrap();
    let grad_graph = graph.with_gradients(loss_node);

    executor.forward(&grad_graph, HashMap::new()).unwrap();
    let grads = executor.get_gradients(&grad_graph);

    let grad_a = grads.get(&a_id).expect("dA missing");
    let grad_b = grads.get(&b_id).expect("dB missing");

    let expected_grad_a = vec![0.5, 1.0 / 3.0];
    let expected_grad_b = vec![-1.0, -2.0 / 3.0];

    assert_approx_eq!(grad_a, &expected_grad_a);
    assert_approx_eq!(grad_b, &expected_grad_b);
}

#[test]
fn gradient_transpose_operation() {
    let mut executor = create_executor();

    // f(x) = sum(transpose(x))
    // Gradient should flow back through transpose
    let x = Parameter::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let x_id = x.id();
    let node = TensorExpr::from(x)
        .transpose()
        .reduce_sum(1)
        .reduce_sum(0);

    let graph: TensorGraph<f32> = node.into();
    let loss_node = *graph.toposort().last().unwrap();
    let grad_graph = graph.with_gradients(loss_node);

    executor.forward(&grad_graph, HashMap::new()).unwrap();
    let grads = executor.get_gradients(&grad_graph);

    let grad = grads.get(&x_id).expect("Parameter gradient missing");
    assert_eq!(grad.len(), 6);

    // Gradient should be all ones (transpose doesn't affect sum gradient)
    for &g in grad.iter() {
        assert!(
            (g - 1.0).abs() < EPSILON,
            "Expected gradient 1.0, got {}",
            g
        );
    }
}
