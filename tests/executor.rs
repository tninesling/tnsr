use std::collections::HashMap;

use petgraph::visit::IntoNodeReferences;
use tnsr::graph::TensorGraph;
use tnsr::tensor::{Constant, Parameter, TensorExpr};
use tnsr::{Executor, Runtime};

const EPSILON: f32 = 1e-5;

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
