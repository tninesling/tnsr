use super::*;
use crate::{Executor, SimpleExecutor, tensor::ValueDType};

fn source_bytes<G>(graph: &TensorGraph<f32, G>) -> usize {
    graph
        .graph
        .node_weights()
        .filter_map(|node| match node {
            TensorGraphNode::Constant { data, .. } => Some(data.len() * size_of::<f32>()),
            _ => None,
        })
        .sum()
}

#[test]
fn mean_and_relu_gradients_use_scalar_sources_with_shared_bf16_boundaries() {
    for bf16 in [false, true] {
        let parameter = crate::tensor::Parameter::new(vec![-2.0, -0.0, 1.0, 3.0], vec![2, 2]);
        let mut x = TensorExpr::from(parameter.clone());
        if bf16 {
            x = x.cast(ValueDType::BF16);
        }
        // Fanout through the same narrowed primal tests accumulation rounding.
        let loss = (x.clone().relu() + x.clone().relu() * x).mean_all();
        let mut graph = TensorGraph::new();
        let loss = graph.add_expr(&loss, false);
        let graph = graph.with_gradients(loss);
        assert!(graph.graph.node_weights().all(|node| {
            !matches!(node, TensorGraphNode::Constant { data, .. } if data.len() > 1)
        }));
        let mut cpu = SimpleExecutor::new();
        cpu.execute(&graph, HashMap::new()).unwrap();
        assert_eq!(cpu.get_value(loss).unwrap(), vec![3.5]);
        assert_eq!(
            cpu.get_gradients(&graph)[&parameter.id()],
            vec![0.0, 0.0, 0.75, 1.75]
        );
        if bf16 {
            let dtypes = mixed::value_dtypes(&graph);
            let gradient = graph.gradient_metadata().param_to_grad[&parameter.id()];
            assert_eq!(dtypes[&gradient], ValueDType::F32);
            assert!(dtypes.values().any(|&dtype| dtype == ValueDType::BF16));
        }
    }
}

#[test]
fn mean_backward_constant_payload_is_independent_of_tensor_extent() {
    for shape in [vec![1, 1], vec![16, 512], vec![64, 2048]] {
        let parameter = crate::tensor::Parameter::new(vec![2.0; shape.iter().product()], shape);
        let x = TensorExpr::from(parameter);
        let mut graph = TensorGraph::new();
        let loss = graph.add_expr(&(x.clone() * x).mean_all(), false);
        let graph = graph.with_gradients(loss);
        // One scalar seed and one scalar reciprocal per reduction axis.
        assert_eq!(source_bytes(&graph), 3 * size_of::<f32>());
    }
}

#[test]
fn scalar_sources_preserve_singleton_and_empty_gradients() {
    for (shape, values, expected) in [
        (vec![1, 1], vec![3.0], vec![1.0]),
        (vec![0, 2], vec![], vec![]),
    ] {
        let parameter = crate::tensor::Parameter::new(values, shape);
        let x: TensorExpr<f32> = TensorExpr::from(parameter.clone());
        let mut graph = TensorGraph::new();
        let loss = graph.add_expr(&x.relu().mean_all(), false);
        let graph = graph.with_gradients(loss);
        let mut cpu = SimpleExecutor::new();
        cpu.execute(&graph, HashMap::new()).unwrap();
        assert_eq!(cpu.get_gradients(&graph)[&parameter.id()], expected);
    }
}

#[cfg(feature = "fusion")]
#[test]
fn fused_relu_derivative_also_uses_scalar_zero() {
    let parameter = crate::tensor::Parameter::new(vec![-1.0_f32, 2.0], vec![2]);
    let mut graph = TensorGraph::<f32>::new();
    let input = graph.add_expr(&TensorExpr::from(parameter.clone()), false);
    let fused = graph.graph.add_node(TensorGraphNode::FusedUnary {
        ops: vec![UnaryOp::Relu, UnaryOp::Neg],
        shape: vec![2],
    });
    graph.graph.add_edge(input, fused, 0);
    let loss = graph.add_expr(&TensorExpr::node_ref(fused, vec![2]).mean_all(), false);
    let graph = graph.with_gradients(loss);
    assert!(
        graph.graph.node_weights().all(|node| {
            !matches!(node, TensorGraphNode::Constant { data, .. } if data.len() > 1)
        })
    );
    let mut cpu = SimpleExecutor::new();
    cpu.execute(&graph, HashMap::new()).unwrap();
    assert_eq!(cpu.get_gradients(&graph)[&parameter.id()], vec![0.0, -0.5]);
}
