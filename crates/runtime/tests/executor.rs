//! Integration tests for graph execution.
//!
//! Tests graph execution and gradient computation using the unified Runtime.
//! The Runtime automatically selects the best available backend (CUDA or CPU).

use std::collections::HashMap;
use std::ops::Neg;

use petgraph::visit::IntoNodeReferences;
use runtime::{Executor, Runtime};
use tensor::graph::TensorGraph;
use tensor::{Constant, Parameter, TensorExpr};

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

/// Creates a Runtime with automatic backend selection.
///
/// The Runtime will prefer CUDA if available, otherwise falls back to CPU.
fn create_runtime() -> Runtime {
    Runtime::new()
}

// ============================================================================
// Execution Tests
// ============================================================================

#[test]
fn forward_matmul_2x2() {
    let mut runtime = create_runtime();

    // Simple 2x2 matrix multiplication test
    // A = [[1, 2],    B = [[5, 6],
    //      [3, 4]]         [7, 8]]
    // Expected C = A @ B = [[19, 22],
    //                       [43, 50]]
    let a = Constant::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Constant::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
    let node = TensorExpr::from(a).matmul(b);

    let graph: TensorGraph<f32> = node.into();
    let result = runtime.execute(&graph, HashMap::new()).unwrap();

    // C[0,0] = 1*5 + 2*7 = 19
    // C[0,1] = 1*6 + 2*8 = 22
    // C[1,0] = 3*5 + 4*7 = 43
    // C[1,1] = 3*6 + 4*8 = 50
    let expected = vec![19.0, 22.0, 43.0, 50.0];
    assert_approx_eq!(result, expected);
}

#[test]
fn forward_matmul_simple() {
    let mut runtime = create_runtime();

    // A[16,16] @ B[16,16] = C[16,16]
    // Use identity matrices with a scalar multiplier for easy verification
    let mut a_data = vec![0.0; 16 * 16];
    let mut b_data = vec![0.0; 16 * 16];

    // A = 2 * I (identity matrix scaled by 2)
    // B = 3 * I (identity matrix scaled by 3)
    for i in 0..16 {
        a_data[i * 16 + i] = 2.0;
        b_data[i * 16 + i] = 3.0;
    }

    let a = Constant::new(a_data, vec![16, 16]);
    let b = Constant::new(b_data, vec![16, 16]);
    let node = TensorExpr::from(a).matmul(b);

    let graph: TensorGraph<f32> = node.into();
    let result = runtime.execute(&graph, HashMap::new()).unwrap();

    // Expected: 2*I @ 3*I = 6*I (identity matrix scaled by 6)
    let mut expected = vec![0.0; 16 * 16];
    for i in 0..16 {
        expected[i * 16 + i] = 6.0;
    }
    assert_approx_eq!(result, expected);
}

#[test]
fn forward_matmul_32x32() {
    let mut runtime = create_runtime();

    // A[32,32] @ B[32,32] = C[32,32]
    // Use identity matrices with a scalar multiplier for easy verification
    let mut a_data = vec![0.0; 32 * 32];
    let mut b_data = vec![0.0; 32 * 32];

    // A = 2 * I (identity matrix scaled by 2)
    // B = 3 * I (identity matrix scaled by 3)
    for i in 0..32 {
        a_data[i * 32 + i] = 2.0;
        b_data[i * 32 + i] = 3.0;
    }

    let a = Constant::new(a_data, vec![32, 32]);
    let b = Constant::new(b_data, vec![32, 32]);
    let node = TensorExpr::from(a).matmul(b);

    let graph: TensorGraph<f32> = node.into();
    let result = runtime.execute(&graph, HashMap::new()).unwrap();

    // Expected: 2*I @ 3*I = 6*I (identity matrix scaled by 6)
    let mut expected = vec![0.0; 32 * 32];
    for i in 0..32 {
        expected[i * 32 + i] = 6.0;
    }
    assert_approx_eq!(result, expected);
}

#[test]
fn forward_broadcast_then_reduce() {
    let mut runtime = create_runtime();

    // Broadcast [2,1] to [2,3], then reduce_sum along axis 1
    let a = Constant::new(vec![1.0, 2.0], vec![2, 1]);
    let node = TensorExpr::from(a).broadcast(vec![2, 3]).reduce_sum(1);

    let graph: TensorGraph<f32> = node.into();
    let result = runtime.execute(&graph, HashMap::new()).unwrap();

    // Expected: [[1,1,1],[2,2,2]] summed along axis 1 = [3, 6]
    let expected = vec![3.0, 6.0];
    assert_approx_eq!(result, expected);
}

#[test]
fn forward_reduce_max() {
    let mut runtime = create_runtime();

    let a = Constant::new(vec![1.0, -2.0, 3.5, 0.5, 10.0, -1.0], vec![2, 3]);
    let node = TensorExpr::from(a).reduce_max(1);

    let graph: TensorGraph<f32> = node.into();
    let result = runtime.execute(&graph, HashMap::new()).unwrap();

    // Expected: max of each row = [3.5, 10.0]
    let expected = vec![3.5, 10.0];
    assert_approx_eq!(result, expected);
}

#[test]
fn forward_reduce_mean() {
    let mut runtime = create_runtime();

    let a = Constant::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let node = TensorExpr::from(a).reduce_mean(1);

    let graph: TensorGraph<f32> = node.into();
    let result = runtime.execute(&graph, HashMap::new()).unwrap();

    // Expected: mean of each row = [2.0, 5.0]
    let expected = vec![2.0, 5.0];
    assert_approx_eq!(result, expected);
}

#[test]
fn forward_chained_operations() {
    let mut runtime = create_runtime();

    // Complex chain: (a * 2) + (b - 1)
    let a = Constant::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Constant::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
    let two = Constant::new(vec![2.0, 2.0, 2.0, 2.0], vec![2, 2]);
    let one = Constant::new(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);

    let node = (TensorExpr::from(a) * TensorExpr::from(two))
        + (TensorExpr::from(b) - TensorExpr::from(one));

    let graph: TensorGraph<f32> = node.into();
    let result = runtime.execute(&graph, HashMap::new()).unwrap();

    // Expected: [2,4,6,8] + [4,5,6,7] = [6,9,12,15]
    let expected = vec![6.0, 9.0, 12.0, 15.0];
    assert_approx_eq!(result, expected);
}

// ============================================================================
// Gradient Tests
// ============================================================================

#[test]
fn gradient_simple_square() {
    let mut runtime = create_runtime();

    // f(x) = x^2 where x = [2.0]
    // df/dx = 2x = 4.0
    let x = Parameter::new(vec![2.0f32], vec![1]);
    let x_id = x.id();
    let x_sq = TensorExpr::from(x.clone()) * TensorExpr::from(x);

    let graph: TensorGraph<f32> = x_sq.into();
    let loss_node = *graph.toposort().last().unwrap();
    let grad_graph = graph.with_gradients(loss_node);

    runtime.execute(&grad_graph, HashMap::new()).unwrap();
    let grads = runtime.get_gradients(&grad_graph);

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
    let mut runtime = create_runtime();

    // f(x) = exp(x) where x = [1.0]
    // df/dx = exp(x) = e ≈ 2.718
    let x = Parameter::new(vec![1.0f32], vec![1]);
    let x_id = x.id();
    let exp_x = TensorExpr::from(x).exp();

    let graph: TensorGraph<f32> = exp_x.into();
    let loss_node = *graph.toposort().last().unwrap();
    let grad_graph = graph.with_gradients(loss_node);

    runtime.execute(&grad_graph, HashMap::new()).unwrap();
    let grads = runtime.get_gradients(&grad_graph);

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
    let mut runtime = create_runtime();

    // Test with rectangular matrices: A[2,3] @ B[3,2]
    let a = Parameter::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let a_id = a.id();
    let b = Parameter::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
    let b_id = b.id();

    let node = TensorExpr::from(a).matmul(b).reduce_sum(1).reduce_sum(0);

    let graph: TensorGraph<f32> = node.into();
    let loss_node = *graph.toposort().last().unwrap();
    let grad_graph = graph.with_gradients(loss_node);

    runtime.execute(&grad_graph, HashMap::new()).unwrap();
    let grads = runtime.get_gradients(&grad_graph);

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
    let mut runtime = create_runtime();

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

    eprintln!("\n=== Forward Graph ===");
    for (idx, node) in graph.graph.node_references() {
        let shape = graph
            .shapes
            .get(&idx)
            .map(|s| format!("{:?}", s))
            .unwrap_or("N/A".to_string());
        eprintln!("Node {:?}: {} shape {}", idx.index(), node.name(), shape);
    }

    let grad_graph = graph.with_gradients(loss_node);

    eprintln!("\n=== Gradient Graph ===");
    for (idx, node) in grad_graph.graph.node_references() {
        let shape = grad_graph
            .shapes
            .get(&idx)
            .map(|s| format!("{:?}", s))
            .unwrap_or("N/A".to_string());
        let inputs = grad_graph.inputs(idx);
        eprintln!(
            "Node {:?}: {} shape {} inputs: {:?}",
            idx.index(),
            node.name(),
            shape,
            inputs.iter().map(|i| i.index()).collect::<Vec<_>>()
        );
    }
    eprintln!(
        "\nParameter {} gradient node: {:?}\n",
        a_id,
        grad_graph.gradient_metadata().param_to_grad.get(&a_id)
    );

    runtime.execute(&grad_graph, HashMap::new()).unwrap();

    let grads = runtime.get_gradients(&grad_graph);

    let grad_a = grads.get(&a_id).expect("Parameter gradient missing");
    assert_eq!(grad_a.len(), 4);

    // Debug: print all gradient values
    eprintln!("Gradient values for parameter {}: {:?}", a_id, grad_a);

    // With B=I and C=2I, z = sum(2A), so dz/dA = 2*ones
    for (i, &val) in grad_a.iter().enumerate() {
        assert!(
            (val - 2.0).abs() < EPSILON,
            "Expected gradient 2.0, got {} at index {}",
            val,
            i
        );
    }
}

#[test]
fn gradient_broadcast_and_reduce() {
    let mut runtime = create_runtime();

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

    runtime.execute(&grad_graph, HashMap::new()).unwrap();
    let grads = runtime.get_gradients(&grad_graph);

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
    let mut runtime = create_runtime();

    // f(x) = relu(x) where x = [-1, 2, -3, 4]
    // df/dx = [0, 1, 0, 1]
    let x = Parameter::new(vec![-1.0, 2.0, -3.0, 4.0], vec![2, 2]);
    let x_id = x.id();
    let node = TensorExpr::from(x).relu().reduce_sum(1).reduce_sum(0);

    let graph: TensorGraph<f32> = node.into();
    let loss_node = *graph.toposort().last().unwrap();
    let grad_graph = graph.with_gradients(loss_node);

    runtime.execute(&grad_graph, HashMap::new()).unwrap();
    let grads = runtime.get_gradients(&grad_graph);

    let grad = grads.get(&x_id).expect("Parameter gradient missing");
    assert_eq!(grad.len(), 4);

    let expected = vec![0.0, 1.0, 0.0, 1.0];
    assert_approx_eq!(grad, &expected);
}

#[test]
fn gradient_log_function() {
    let mut runtime = create_runtime();

    // f(x) = log(x) where x = [1, 2, 3, 4]
    // df/dx = 1/x = [1, 0.5, 0.333..., 0.25]
    let x_data = vec![1.0, 2.0, 3.0, 4.0];
    let x = Parameter::new(x_data.clone(), vec![2, 2]);
    let x_id = x.id();
    let node = TensorExpr::from(x).log().reduce_sum(1).reduce_sum(0);

    let graph: TensorGraph<f32> = node.into();
    let loss_node = *graph.toposort().last().unwrap();
    let grad_graph = graph.with_gradients(loss_node);

    runtime.execute(&grad_graph, HashMap::new()).unwrap();
    let grads = runtime.get_gradients(&grad_graph);

    let grad = grads.get(&x_id).expect("Parameter gradient missing");
    assert_eq!(grad.len(), 4);

    let expected: Vec<f32> = x_data.iter().map(|&v| 1.0 / v).collect();
    assert_approx_eq!(grad, &expected);
}

#[test]
fn gradient_division_operation() {
    let mut runtime = create_runtime();

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

    runtime.execute(&grad_graph, HashMap::new()).unwrap();
    let grads = runtime.get_gradients(&grad_graph);

    let grad_a = grads.get(&a_id).expect("dA missing");
    let grad_b = grads.get(&b_id).expect("dB missing");

    let expected_grad_a = vec![0.5, 1.0 / 3.0];
    let expected_grad_b = vec![-1.0, -2.0 / 3.0];

    assert_approx_eq!(grad_a, &expected_grad_a);
    assert_approx_eq!(grad_b, &expected_grad_b);
}

#[test]
fn gradient_transpose_operation() {
    let mut runtime = create_runtime();

    // f(x) = sum(transpose(x))
    // Gradient should flow back through transpose
    let x = Parameter::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let x_id = x.id();
    let node = TensorExpr::from(x).transpose().reduce_sum(1).reduce_sum(0);

    let graph: TensorGraph<f32> = node.into();
    let loss_node = *graph.toposort().last().unwrap();
    let grad_graph = graph.with_gradients(loss_node);

    runtime.execute(&grad_graph, HashMap::new()).unwrap();
    let grads = runtime.get_gradients(&grad_graph);

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

#[test]
fn gradient_matmul_with_transpose() {
    let mut runtime = create_runtime();

    // Simple test: z = sum(A @ B^T)
    // This tests the gradient pattern used in matmul backprop
    let a = Parameter::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let a_id = a.id();
    let b = Constant::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]); // Identity

    let node = TensorExpr::from(a)
        .matmul(TensorExpr::from(b).transpose())
        .reduce_sum(1)
        .reduce_sum(0);

    let graph: TensorGraph<f32> = node.into();
    let loss_node = *graph.toposort().last().unwrap();

    eprintln!("\n=== FORWARD GRAPH ===");
    for idx in graph.toposort().iter() {
        eprintln!("Node {:?}: {:?}", idx.index(), graph[*idx].name());
    }

    let grad_graph = graph.with_gradients(loss_node);

    eprintln!(
        "\n=== GRADIENT GRAPH (total {} nodes) ===",
        grad_graph.len()
    );
    for idx in grad_graph.toposort().iter() {
        eprintln!("Node {:?}: {:?}", idx.index(), grad_graph[*idx].name());
    }

    runtime.execute(&grad_graph, HashMap::new()).unwrap();

    // Debug: Check what Node 10 contains after execution
    eprintln!("\n=== DEBUG: Checking Node 10 after execution ===");
    if let Some(node10_val) = runtime.get_value(petgraph::graph::NodeIndex::new(10)) {
        eprintln!("Node 10 value: {:?}", node10_val);
    } else {
        eprintln!("Node 10: NOT FOUND in cache");
    }

    let grads = runtime.get_gradients(&grad_graph);

    let grad = grads.get(&a_id).expect("Parameter gradient missing");
    assert_eq!(grad.len(), 4);

    eprintln!("Gradient for A @ I^T: {:?}", grad);

    // A @ I^T = A, so sum(A) has gradient of all ones
    for (i, &g) in grad.iter().enumerate() {
        assert!(
            (g - 1.0).abs() < EPSILON,
            "Expected gradient 1.0, got {} at index {}",
            g,
            i
        );
    }
}

#[test]
fn gradient_transpose_then_matmul() {
    let mut runtime = create_runtime();

    // Test: z = sum(A^T @ B) where B is a constant
    // This tests if we can compute gradients through transpose followed by matmul
    let a = Parameter::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let a_id = a.id();
    let b = Constant::new(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]); // All ones

    let node = TensorExpr::from(a)
        .transpose()
        .matmul(b)
        .reduce_sum(1)
        .reduce_sum(0);

    let graph: TensorGraph<f32> = node.into();
    let loss_node = *graph.toposort().last().unwrap();
    let grad_graph = graph.with_gradients(loss_node);

    runtime.execute(&grad_graph, HashMap::new()).unwrap();
    let grads = runtime.get_gradients(&grad_graph);

    let grad = grads.get(&a_id).expect("Parameter gradient missing");
    assert_eq!(grad.len(), 4);

    eprintln!("Gradient for A^T @ ones: {:?}", grad);

    // Expected gradient should be all 2.0
    // d/dA sum(A^T @ ones) = d/dA sum([a+c, b+d], [a+c, b+d])
    // = d/dA [2a+2c, 2b+2d] = [[2, 2], [2, 2]]
    for (i, &g) in grad.iter().enumerate() {
        assert!(
            (g - 2.0).abs() < EPSILON,
            "Expected gradient 2.0, got {} at index {}",
            g,
            i
        );
    }
}

// ============================================================================
// Focused Tests to Isolate MatMul Bug
// ============================================================================

#[test]
fn forward_matmul_result_reused() {
    let mut runtime = create_runtime();

    // Test: (A @ B) used in two different operations
    // This tests if matmul result can be read multiple times
    // C = A @ B (2x2 @ 2x2 = 2x2)
    // D = C + C
    let a = Constant::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]); // Identity
    let b = Constant::new(vec![2.0, 0.0, 0.0, 2.0], vec![2, 2]); // 2*I

    let c = TensorExpr::from(a).matmul(b);
    let node = c.clone() + c;

    let graph: TensorGraph<f32> = node.into();
    let result = runtime.execute(&graph, HashMap::new()).unwrap();

    // Expected: (I @ 2I) + (I @ 2I) = 2I + 2I = 4I
    let mut expected = vec![0.0; 4];
    expected[0] = 4.0;
    expected[3] = 4.0;
    assert_approx_eq!(result, expected);
}

#[test]
fn forward_chained_matmuls() {
    let mut runtime = create_runtime();

    // Test: (A @ B) @ C - tests if matmul result can be input to another matmul
    // All 2x2 matrices
    let a = Constant::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]); // Identity
    let b = Constant::new(vec![2.0, 0.0, 0.0, 2.0], vec![2, 2]); // 2*I
    let c = Constant::new(vec![3.0, 0.0, 0.0, 3.0], vec![2, 2]); // 3*I

    let ab = TensorExpr::from(a).matmul(b);
    let node = ab.matmul(c);

    let graph: TensorGraph<f32> = node.into();
    let result = runtime.execute(&graph, HashMap::new()).unwrap();

    // Expected: (I @ 2I) @ 3I = 2I @ 3I = 6I
    let mut expected = vec![0.0; 4];
    expected[0] = 6.0;
    expected[3] = 6.0;
    assert_approx_eq!(result, expected);
}

#[test]
fn forward_transpose_2x2() {
    let mut runtime = create_runtime();

    // Test: Basic 2x2 transpose
    // Input:  [[1, 2],    Output: [[1, 3],
    //          [3, 4]]             [2, 4]]
    let a = Constant::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let node = TensorExpr::from(a).transpose();

    let graph: TensorGraph<f32> = node.into();
    let result = runtime.execute(&graph, HashMap::new()).unwrap();

    // Row-major: [[1, 3], [2, 4]] = [1, 3, 2, 4]
    let expected = vec![1.0, 3.0, 2.0, 4.0];
    assert_approx_eq!(result, expected);
}

#[test]
fn forward_transpose_2x3() {
    let mut runtime = create_runtime();

    // Test: 2x3 -> 3x2 transpose
    // Input:  [[1, 2, 3],     Output: [[1, 4],
    //          [4, 5, 6]]              [2, 5],
    //                                  [3, 6]]
    let a = Constant::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let node = TensorExpr::from(a).transpose();

    let graph: TensorGraph<f32> = node.into();
    let result = runtime.execute(&graph, HashMap::new()).unwrap();

    // Row-major: [[1, 4], [2, 5], [3, 6]] = [1, 4, 2, 5, 3, 6]
    let expected = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    assert_approx_eq!(result, expected);
}

#[test]
fn forward_transpose_3x2() {
    let mut runtime = create_runtime();

    // Test: 3x2 -> 2x3 transpose
    // Input:  [[1, 2],       Output: [[1, 3, 5],
    //          [3, 4],                [2, 4, 6]]
    //          [5, 6]]
    let a = Constant::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
    let node = TensorExpr::from(a).transpose();

    let graph: TensorGraph<f32> = node.into();
    let result = runtime.execute(&graph, HashMap::new()).unwrap();

    // Row-major: [[1, 3, 5], [2, 4, 6]] = [1, 3, 5, 2, 4, 6]
    let expected = vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0];
    assert_approx_eq!(result, expected);
}

#[test]
fn forward_transpose_double() {
    let mut runtime = create_runtime();

    // Test: transpose(transpose(A)) = A
    let a = Constant::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let node = TensorExpr::from(a.clone()).transpose().transpose();

    let graph: TensorGraph<f32> = node.into();
    let result = runtime.execute(&graph, HashMap::new()).unwrap();

    // Should get back the original
    let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    assert_approx_eq!(result, expected);
}

#[test]
fn forward_matmul_then_transpose() {
    let mut runtime = create_runtime();

    // Test: transpose(A @ B) - tests if we can transpose a matmul result
    let a = Constant::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Constant::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

    let ab = TensorExpr::from(a).matmul(b);
    let node = ab.transpose();

    let graph: TensorGraph<f32> = node.into();
    let result = runtime.execute(&graph, HashMap::new()).unwrap();

    // A @ B = [[19, 22], [43, 50]] (row-major: [19, 22, 43, 50])
    // Transposed = [[19, 43], [22, 50]] (row-major: [19, 43, 22, 50])
    let expected = vec![19.0, 43.0, 22.0, 50.0];
    assert_approx_eq!(result, expected);
}

#[test]
fn forward_matmul_with_transpose_input() {
    let mut runtime = create_runtime();

    // Test: A @ (B^T) - tests matmul with a transposed input
    let a = Constant::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Constant::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

    let node = TensorExpr::from(a).matmul(TensorExpr::from(b).transpose());

    let graph: TensorGraph<f32> = node.into();
    let result = runtime.execute(&graph, HashMap::new()).unwrap();

    // B^T = [[5, 7], [6, 8]] (row-major: [5, 7, 6, 8])
    // A @ B^T = [[1*5+2*6, 1*7+2*8], [3*5+4*6, 3*7+4*8]]
    //         = [[17, 23], [39, 53]]
    let expected = vec![17.0, 23.0, 39.0, 53.0];
    assert_approx_eq!(result, expected);
}

#[test]
fn forward_gt_operation() {
    let mut runtime = create_runtime();

    // Test: x > 0 where x = [-2.0, -1.0, 0.0, 1.0, 2.0]
    // Expected: [0.0, 0.0, 0.0, 1.0, 1.0]
    let x = Constant::new(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]);
    let zero = Constant::new(vec![0.0, 0.0, 0.0, 0.0, 0.0], vec![5]);
    let node = TensorExpr::from(x).gt(TensorExpr::from(zero));

    let graph: TensorGraph<f32> = node.into();
    let result = runtime
        .execute(&graph, std::collections::HashMap::new())
        .unwrap();

    let expected = [0.0, 0.0, 0.0, 1.0, 1.0];
    assert_eq!(result.len(), 5);
    for (i, (&actual, &exp)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - exp).abs() < EPSILON,
            "Gt mismatch at index {}: expected {}, got {}",
            i,
            exp,
            actual
        );
    }
}

#[test]
fn forward_mask_operation() {
    let mut runtime = create_runtime();

    // Test: mask([10, 20, 30, 40], [0, 1, 0, 1])
    // Expected: [0, 20, 0, 40]
    let values = Constant::new(vec![10.0, 20.0, 30.0, 40.0], vec![4]);
    let condition = Constant::new(vec![0.0, 1.0, 0.0, 1.0], vec![4]);
    let node = TensorExpr::from(values).mask(TensorExpr::from(condition));

    let graph: TensorGraph<f32> = node.into();
    let result = runtime
        .execute(&graph, std::collections::HashMap::new())
        .unwrap();
    eprintln!("Result: {:?}", result);

    let expected = [0.0, 20.0, 0.0, 40.0];
    assert_eq!(result.len(), 4);
    for (i, (&actual, &exp)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - exp).abs() < EPSILON,
            "Mask mismatch at index {}: expected {}, got {}",
            i,
            exp,
            actual
        );
    }
}

#[test]
fn forward_relu_detailed() {
    let mut runtime = create_runtime();

    // Test ReLU with various values including negatives, zero, and positives
    let x = Constant::new(vec![-5.0, -2.0, -0.5, 0.0, 0.5, 2.0, 5.0], vec![7]);
    let node = TensorExpr::from(x).relu();

    let graph: TensorGraph<f32> = node.into();
    let result = runtime
        .execute(&graph, std::collections::HashMap::new())
        .unwrap();

    let expected = [0.0, 0.0, 0.0, 0.0, 0.5, 2.0, 5.0];
    assert_eq!(result.len(), 7);
    for (i, (&actual, &exp)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - exp).abs() < EPSILON,
            "ReLU mismatch at index {}: expected {}, got {}",
            i,
            exp,
            actual
        );
    }
}

#[test]
fn gradient_relu_detailed() {
    let mut runtime = create_runtime();

    // Test ReLU gradient with various values
    // f(x) = sum(relu(x)) where x = [-5, -2, -0.5, 0, 0.5, 2, 5]
    // relu(x) = [0, 0, 0, 0, 0.5, 2, 5]
    // df/dx = [0, 0, 0, 0, 1, 1, 1] (gradient is 1 where x > 0, 0 elsewhere)
    let x = Parameter::new(vec![-5.0, -2.0, -0.5, 0.0, 0.5, 2.0, 5.0], vec![7]);
    let x_id = x.id();
    let node = TensorExpr::from(x).relu().reduce_sum(0);

    let graph: TensorGraph<f32> = node.into();
    let loss_node = *graph.toposort().last().unwrap();
    let grad_graph = graph.with_gradients(loss_node);

    runtime
        .execute(&grad_graph, std::collections::HashMap::new())
        .unwrap();
    let grads = runtime.get_gradients(&grad_graph);

    let grad = grads.get(&x_id).expect("Parameter gradient missing");
    assert_eq!(grad.len(), 7);

    let expected = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    for (i, (&actual, &exp)) in grad.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - exp).abs() < EPSILON,
            "ReLU gradient mismatch at index {}: expected {}, got {}",
            i,
            exp,
            actual
        );
    }
}

// ============================================================================
// Fusion Tests
// ============================================================================

#[test]
#[cfg(feature = "fusion")]
fn forward_fused_unary_chain() {
    let mut runtime = create_runtime();

    // Test: exp(log(relu(x))) where x = [1.0, 2.0, 3.0, 4.0]
    // relu([1, 2, 3, 4]) = [1, 2, 3, 4]
    // log([1, 2, 3, 4]) = [0, ln(2), ln(3), ln(4)]
    // exp([0, ln(2), ln(3), ln(4)]) = [1, 2, 3, 4]
    let x = Constant::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
    let node = TensorExpr::from(x).relu().log().exp();

    let mut graph: TensorGraph<f32> = node.into();

    // Apply fusion optimization
    graph.apply_fusion();

    let result = runtime.execute(&graph, HashMap::new()).unwrap();

    let expected = vec![1.0, 2.0, 3.0, 4.0];
    assert_approx_eq!(result, expected);
}

#[test]
#[cfg(feature = "fusion")]
fn forward_fused_vs_unfused_same_result() {
    let mut runtime_fused = create_runtime();
    let mut runtime_unfused = create_runtime();

    // Test that fused and unfused graphs produce the same result
    // Chain: neg(exp(x)) where x = [0.0, 1.0, 2.0]
    let x1 = Constant::new(vec![0.0, 1.0, 2.0], vec![3]);
    let node1 = TensorExpr::from(x1).exp().neg();

    let x2 = Constant::new(vec![0.0, 1.0, 2.0], vec![3]);
    let node2 = TensorExpr::from(x2).exp().neg();

    let mut graph_fused: TensorGraph<f32> = node1.into();
    let graph_unfused: TensorGraph<f32> = node2.into();

    // Apply fusion to one graph only
    graph_fused.apply_fusion();

    let result_fused = runtime_fused.execute(&graph_fused, HashMap::new()).unwrap();
    let result_unfused = runtime_unfused
        .execute(&graph_unfused, HashMap::new())
        .unwrap();

    assert_approx_eq!(result_fused, result_unfused);
}

#[test]
#[cfg(feature = "fusion")]
fn forward_fused_with_branching_preserved() {
    let mut runtime = create_runtime();

    // Test that branching prevents fusion where it should
    // x -> exp -> [relu, log]
    // The exp node has multiple consumers, so it shouldn't be fused
    let x = Constant::new(vec![1.0, 2.0, 3.0], vec![3]);
    let exp_node = TensorExpr::from(x).exp();
    let relu_branch = exp_node.clone().relu();
    let log_branch = exp_node.log();
    let result = relu_branch + log_branch;

    let mut graph: TensorGraph<f32> = result.into();

    // Apply fusion - should not fuse exp with either relu or log
    graph.apply_fusion();

    let output = runtime.execute(&graph, HashMap::new()).unwrap();

    // Expected: relu(exp([1,2,3])) + log(exp([1,2,3]))
    //         = relu([e, e^2, e^3]) + log([e, e^2, e^3])
    //         = [e, e^2, e^3] + [1, 2, 3]
    //         = [e+1, e^2+2, e^3+3]
    let e = std::f32::consts::E;
    let expected = vec![e + 1.0, e.powi(2) + 2.0, e.powi(3) + 3.0];
    assert_approx_eq!(output, expected);
}

// ============================================================================
// Fusion Gradient Tests
// ============================================================================

#[test]
#[cfg(feature = "fusion")]
fn gradient_fused_exp_log_chain() {
    let mut runtime = create_runtime();

    // Test: f(x) = log(exp(x)) with fusion
    // Mathematically: log(exp(x)) = x, so df/dx = 1
    // Chain rule: df/dx = (1/exp(x)) * exp(x) = 1
    let x = Parameter::new(vec![1.0, 2.0, 3.0], vec![3]);
    let x_id = x.id();

    let node = TensorExpr::from(x).exp().log().reduce_sum(0);

    let mut graph: TensorGraph<f32> = node.into();

    // Apply fusion
    let num_fused = graph.apply_fusion();
    assert_eq!(num_fused, 1, "Should have fused exp->log chain");

    let loss_node = *graph.toposort().last().unwrap();
    let grad_graph = graph.with_gradients(loss_node);

    runtime.execute(&grad_graph, HashMap::new()).unwrap();
    let grads = runtime.get_gradients(&grad_graph);

    let grad = grads.get(&x_id).expect("Parameter gradient missing");
    assert_eq!(grad.len(), 3);

    // Expected: all gradients should be 1.0
    for (i, &g) in grad.iter().enumerate() {
        assert!(
            (g - 1.0).abs() < EPSILON,
            "Expected gradient 1.0, got {} at index {}",
            g,
            i
        );
    }
}

#[test]
#[cfg(feature = "fusion")]
fn gradient_fused_vs_unfused_exp_log() {
    let mut runtime_fused = create_runtime();
    let mut runtime_unfused = create_runtime();

    // Compare gradient values between fused and unfused versions
    // They should produce identical numerical results
    let x_fused = Parameter::new(vec![1.0, 2.0, 3.0], vec![3]);
    let x_fused_id = x_fused.id();
    let x_unfused = Parameter::new(vec![1.0, 2.0, 3.0], vec![3]);
    let x_unfused_id = x_unfused.id();

    // Fused version
    let node_fused = TensorExpr::from(x_fused).exp().log().reduce_sum(0);
    let mut graph_fused: TensorGraph<f32> = node_fused.into();
    graph_fused.apply_fusion();
    let loss_fused = *graph_fused.toposort().last().unwrap();
    let grad_graph_fused = graph_fused.with_gradients(loss_fused);

    // Unfused version
    let node_unfused = TensorExpr::from(x_unfused).exp().log().reduce_sum(0);
    let graph_unfused: TensorGraph<f32> = node_unfused.into();
    let loss_unfused = *graph_unfused.toposort().last().unwrap();
    let grad_graph_unfused = graph_unfused.with_gradients(loss_unfused);

    // Execute both
    runtime_fused
        .execute(&grad_graph_fused, HashMap::new())
        .unwrap();
    runtime_unfused
        .execute(&grad_graph_unfused, HashMap::new())
        .unwrap();

    let grads_fused = runtime_fused.get_gradients(&grad_graph_fused);
    let grads_unfused = runtime_unfused.get_gradients(&grad_graph_unfused);

    let grad_fused = grads_fused
        .get(&x_fused_id)
        .expect("Fused gradient missing");
    let grad_unfused = grads_unfused
        .get(&x_unfused_id)
        .expect("Unfused gradient missing");

    // Gradients should be identical
    assert_approx_eq!(grad_fused, grad_unfused);
}

#[test]
#[cfg(feature = "fusion")]
fn gradient_fused_neg_exp_chain() {
    let mut runtime = create_runtime();

    // Test: f(x) = sum(exp(neg(x))) where x = [1, 2, 3]
    // exp(neg(x)) = exp(-x) = [e^-1, e^-2, e^-3]
    // df/dx = -exp(-x) = [-e^-1, -e^-2, -e^-3]
    let x_data = vec![1.0, 2.0, 3.0];
    let x = Parameter::new(x_data.clone(), vec![3]);
    let x_id = x.id();

    let node = (-TensorExpr::from(x)).exp().reduce_sum(0);

    let mut graph: TensorGraph<f32> = node.into();
    graph.apply_fusion();

    let loss_node = *graph.toposort().last().unwrap();
    let grad_graph = graph.with_gradients(loss_node);

    runtime.execute(&grad_graph, HashMap::new()).unwrap();
    let grads = runtime.get_gradients(&grad_graph);

    let grad = grads.get(&x_id).expect("Parameter gradient missing");
    assert_eq!(grad.len(), 3);

    // Expected: -exp(-x)
    let expected: Vec<f32> = x_data.iter().map(|&v| -(-v).exp()).collect();
    assert_approx_eq!(grad, &expected);
}

#[test]
#[cfg(feature = "fusion")]
fn gradient_fused_relu_chain() {
    let mut runtime = create_runtime();

    // Test: f(x) = sum(relu(neg(x))) where x = [-2, -1, 0, 1, 2]
    // neg(x) = [2, 1, 0, -1, -2]
    // relu(neg(x)) = [2, 1, 0, 0, 0]
    // df/dx: gradient is -1 where neg(x) > 0, else 0
    //        = [-1, -1, 0, 0, 0]
    let x = Parameter::new(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]);
    let x_id = x.id();

    let node = (-TensorExpr::from(x)).relu().reduce_sum(0);

    let mut graph: TensorGraph<f32> = node.into();
    graph.apply_fusion();

    let loss_node = *graph.toposort().last().unwrap();
    let grad_graph = graph.with_gradients(loss_node);

    runtime.execute(&grad_graph, HashMap::new()).unwrap();
    let grads = runtime.get_gradients(&grad_graph);

    let grad = grads.get(&x_id).expect("Parameter gradient missing");
    assert_eq!(grad.len(), 5);

    let expected = vec![-1.0, -1.0, 0.0, 0.0, 0.0];
    assert_approx_eq!(grad, &expected);
}

#[test]
#[cfg(feature = "fusion")]
fn gradient_fused_log_relu_chain() {
    let mut runtime = create_runtime();

    // Test: f(x) = sum(log(relu(x))) where x = [0.5, 1, 2, 3]
    // relu(x) = [0.5, 1, 2, 3]
    // log(relu(x)) = [ln(0.5), 0, ln(2), ln(3)]
    // df/dx = (1/relu(x)) * relu'(x) = 1/x where x > 0
    let x_data = vec![0.5, 1.0, 2.0, 3.0];
    let x = Parameter::new(x_data.clone(), vec![4]);
    let x_id = x.id();

    let node = TensorExpr::from(x).relu().log().reduce_sum(0);

    let mut graph: TensorGraph<f32> = node.into();
    graph.apply_fusion();

    let loss_node = *graph.toposort().last().unwrap();
    let grad_graph = graph.with_gradients(loss_node);

    runtime.execute(&grad_graph, HashMap::new()).unwrap();
    let grads = runtime.get_gradients(&grad_graph);

    let grad = grads.get(&x_id).expect("Parameter gradient missing");
    assert_eq!(grad.len(), 4);

    // Expected: 1/x for all positive x
    let expected: Vec<f32> = x_data.iter().map(|&v| 1.0 / v).collect();
    assert_approx_eq!(grad, &expected);
}

#[test]
#[cfg(feature = "fusion")]
fn gradient_fused_longer_chain() {
    let mut runtime = create_runtime();

    // Test: f(x) = sum(exp(log(exp(x)))) where x = [1, 2]
    // exp(log(exp(x))) = exp(x) mathematically
    // df/dx = exp(x)
    let x_data = vec![1.0, 2.0];
    let x = Parameter::new(x_data.clone(), vec![2]);
    let x_id = x.id();

    let node = TensorExpr::from(x).exp().log().exp().reduce_sum(0);

    let mut graph: TensorGraph<f32> = node.into();
    let num_fused = graph.apply_fusion();
    assert!(num_fused > 0, "Should have fused at least one chain");

    let loss_node = *graph.toposort().last().unwrap();
    let grad_graph = graph.with_gradients(loss_node);

    runtime.execute(&grad_graph, HashMap::new()).unwrap();
    let grads = runtime.get_gradients(&grad_graph);

    let grad = grads.get(&x_id).expect("Parameter gradient missing");
    assert_eq!(grad.len(), 2);

    // Expected: exp(x)
    let expected: Vec<f32> = x_data.iter().map(|&v| v.exp()).collect();
    assert_approx_eq!(grad, &expected);
}

#[test]
#[cfg(feature = "fusion")]
fn gradient_fused_in_larger_graph() {
    let mut runtime = create_runtime();

    // Test: gradient when fused chain is part of larger computation
    // f(x, y) = sum((exp(log(x))) + y^2)
    // df/dx = 1 (from exp(log(x)) = x)
    // df/dy = 2y
    let x = Parameter::new(vec![2.0, 3.0], vec![2]);
    let x_id = x.id();
    let y_data = vec![1.0, 2.0];
    let y = Parameter::new(y_data.clone(), vec![2]);
    let y_id = y.id();

    let exp_log_x = TensorExpr::from(x).exp().log();
    let y_sq = TensorExpr::from(y.clone()) * TensorExpr::from(y);
    let node = (exp_log_x + y_sq).reduce_sum(0);

    let mut graph: TensorGraph<f32> = node.into();
    graph.apply_fusion();

    let loss_node = *graph.toposort().last().unwrap();
    let grad_graph = graph.with_gradients(loss_node);

    runtime.execute(&grad_graph, HashMap::new()).unwrap();
    let grads = runtime.get_gradients(&grad_graph);

    let grad_x = grads.get(&x_id).expect("x gradient missing");
    let grad_y = grads.get(&y_id).expect("y gradient missing");

    assert_eq!(grad_x.len(), 2);
    assert_eq!(grad_y.len(), 2);

    // Check x gradient (should be all 1.0)
    for (i, &g) in grad_x.iter().enumerate() {
        assert!(
            (g - 1.0).abs() < EPSILON,
            "Expected x gradient 1.0, got {} at index {}",
            g,
            i
        );
    }

    // Check y gradient (should be 2y)
    let expected_y: Vec<f32> = y_data.iter().map(|&v| 2.0 * v).collect();
    assert_approx_eq!(grad_y, &expected_y);
}

#[test]
#[cfg(feature = "fusion")]
fn gradient_fused_training_step() {
    let mut runtime = create_runtime();

    // Simulate a simple training step with fused operations
    // Model: y_pred = exp(log(W * x + b))  (simplified to W*x + b after fusion)
    // Loss: L = sum((y_pred - y_true)^2)
    // This tests that fusion works in a realistic training scenario

    // Initialize parameters
    let w = Parameter::new(vec![2.0, 3.0], vec![2]);
    let w_id = w.id();
    let b = Parameter::new(vec![0.5], vec![1]);
    let b_id = b.id();

    // Input and target
    let x = Constant::new(vec![1.0, 2.0], vec![2]);
    let y_true = Constant::new(vec![10.0], vec![1]);

    // Forward pass: y_pred = exp(log(W*x + b))
    let wx = TensorExpr::from(w) * TensorExpr::from(x);
    let wx_sum = wx.reduce_sum(0); // Sum to get scalar-like shape [1]
    let linear = wx_sum + TensorExpr::from(b);

    // Apply fused chain: exp(log(z)) which should optimize to z
    let y_pred = linear.exp().log();

    // Loss: (y_pred - y_true)^2
    let diff = y_pred - TensorExpr::from(y_true);
    let loss = diff.clone() * diff;

    let mut graph: TensorGraph<f32> = loss.into();

    // Apply fusion (should fuse exp->log chain)
    let num_fused = graph.apply_fusion();
    assert!(num_fused >= 1, "Should have fused at least one chain");

    // Compute gradients
    let loss_node = *graph.toposort().last().unwrap();
    let grad_graph = graph.with_gradients(loss_node);

    runtime.execute(&grad_graph, HashMap::new()).unwrap();
    let grads = runtime.get_gradients(&grad_graph);

    // Verify we got gradients for both parameters
    let grad_w = grads.get(&w_id).expect("W gradient missing");
    let grad_b = grads.get(&b_id).expect("b gradient missing");

    assert_eq!(grad_w.len(), 2);
    assert_eq!(grad_b.len(), 1);

    // Expected forward values:
    // W*x = [2*1, 3*2] = [2, 6]
    // sum(W*x) = 8
    // linear = 8 + 0.5 = 8.5
    // y_pred = exp(log(8.5)) = 8.5
    // diff = 8.5 - 10 = -1.5
    // loss = (-1.5)^2 = 2.25

    // Expected gradients:
    // dL/dy_pred = 2 * (y_pred - y_true) = 2 * (-1.5) = -3.0
    // dy_pred/d(linear) = 1 (from exp(log(z)) = z)
    // dL/d(linear) = -3.0
    // d(linear)/dW = x = [1, 2]
    // dL/dW = -3.0 * [1, 2] = [-3.0, -6.0]
    // dL/db = -3.0

    let expected_grad_w = vec![-3.0, -6.0];
    let expected_grad_b = vec![-3.0];

    assert_approx_eq!(grad_w, &expected_grad_w);
    assert_approx_eq!(grad_b, &expected_grad_b);

    // Simulate parameter update (gradient descent with lr=0.1)
    let lr = 0.1;
    let w_new: Vec<f32> = vec![2.0, 3.0]
        .iter()
        .zip(grad_w.iter())
        .map(|(w, g)| w - lr * g)
        .collect();
    let b_new: Vec<f32> = vec![0.5]
        .iter()
        .zip(grad_b.iter())
        .map(|(b, g)| b - lr * g)
        .collect();

    // Expected updates:
    // W_new = [2.0, 3.0] - 0.1 * [-3.0, -6.0] = [2.3, 3.6]
    // b_new = 0.5 - 0.1 * (-3.0) = 0.8
    let expected_w_new = vec![2.3, 3.6];
    let expected_b_new = vec![0.8];

    assert_approx_eq!(&w_new, &expected_w_new);
    assert_approx_eq!(&b_new, &expected_b_new);
}

#[test]
#[cfg(feature = "fusion")]
fn gradient_fused_multi_step_training() {
    // Test multiple training steps to ensure fusion works across iterations
    // This verifies that the fused gradient computation is stable and correct
    let mut runtime = create_runtime();

    // Simple model: minimize sum(exp(log(x)))^2 where target is 0
    // With fusion: minimize sum(x^2)
    let mut x_data = vec![5.0, -3.0, 2.0];

    for step in 0..5 {
        let x = Parameter::new(x_data.clone(), vec![3]);
        let x_id = x.id();

        // Apply fused transformation
        let transformed = TensorExpr::from(x).exp().log();

        // Loss: sum(transformed^2)
        let loss = (transformed.clone() * transformed).reduce_sum(0);

        let mut graph: TensorGraph<f32> = loss.into();
        graph.apply_fusion();

        let loss_node = *graph.toposort().last().unwrap();
        let grad_graph = graph.with_gradients(loss_node);

        runtime.execute(&grad_graph, HashMap::new()).unwrap();

        // Get loss value and gradient
        let loss_value = runtime
            .get_value(loss_node)
            .expect("Loss value not found")
            .to_vec();
        let grads = runtime.get_gradients(&grad_graph);
        let grad = grads.get(&x_id).expect("Gradient missing");

        // Expected gradient: 2*x
        let expected_grad: Vec<f32> = x_data.iter().map(|&v| 2.0 * v).collect();
        assert_approx_eq!(grad, &expected_grad);

        // Update parameters
        let lr = 0.1;
        x_data = x_data
            .iter()
            .zip(grad.iter())
            .map(|(x, g)| x - lr * g)
            .collect();

        eprintln!("Step {}: loss = {:?}, x = {:?}", step, loss_value, x_data);

        // Verify loss is positive (unless x is near zero)
        if step > 0 {
            assert!(
                loss_value[0] > 0.0 || x_data.iter().all(|&v| v.abs() < EPSILON),
                "Loss should be positive unless x is near zero"
            );
        }
    }

    // After 5 steps, parameters should be closer to 0
    // With lr=0.1 and gradient=2x, we decay by factor of 0.8 per step
    // After 5 steps: x *= 0.8^5 ≈ 0.328
    // Initial max was 5, so final should be around 5*0.328 ≈ 1.64
    for &v in x_data.iter() {
        assert!(
            v.abs() < 2.0,
            "Parameters should have decreased in magnitude after 5 steps: {:?}",
            x_data
        );
    }
}
