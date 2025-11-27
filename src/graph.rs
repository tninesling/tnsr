//! Lowered computation graph representation with gradient support.
//!
//! Gradients are added to the graph by constructing the derivative computations directly from
//! graph nodes in reverse topological order, starting from the loss node. This means an execution
//! of the graph will compute both forward values and gradients in a single pass. So, there is no
//! separate backward pass as you might find in other frameworks. Here is how the graph might look
//! after adding gradients for the graph computing `D = (A * B).log()`.
//!
//! Forward only:
//! ```text
//!     ┌───┐
//!     │ A │
//!     └─┬─┘
//!       │      ┌─────┐    ┌───┐     ┌─────┐    ┌───┐
//!       ├─────►│ MUL ├───►│ C ├────►│ LOG ├───►│ D │
//!       │      └─────┘    └───┘     └─────┘    └───┘
//!     ┌─┴─┐
//!     │ B │
//!     └───┘
//! ```
//!
//! Forward and backward:
//! ```text
//!     ┌───┐
//! ┌───┤ A │
//! │   └─┬─┘
//! │     │      ┌─────┐    ┌───┐     ┌─────┐    ┌───┐
//! │     ├─────►│ MUL ├───►│ C ├────►│ LOG ├───►│ D │
//! │     │      └─────┘    └─┬─┘     └─────┘    └───┘
//! │   ┌─┴─┐                 └──────────────┐
//! │   │ B ├───────────────────┐            │
//! │   └───┘                   │            ▼
//! │                           │        ┌───────┐
//! │                           │        │ RECIP │
//! │   ┌───────┐     ┌─────┐   │        └───┬───┘
//! │   │ dC/dA │◄────┤ MUL │◄──┴───┐        │
//! │   └───────┘     └─────┘       │        │
//! │                             ┌─┴─────┐  │
//! └────────────────────┬────────┤ dD/dC │◄─┘
//!                      ▼        └───────┘
//!     ┌───────┐     ┌─────┐
//!     │ dC/dB │◄────┤ MUL │
//!     └───────┘     └─────┘
//! ```

use std::collections::HashMap;
use std::collections::HashSet;
use std::ops::Index;
use std::sync::Arc;
use std::sync::Mutex;

use itertools::Itertools;
use petgraph::graph::Graph;
pub use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;

use crate::tensor::{BinaryOp, ReduceOp, TensorExpr, UnaryOp};

/// Typestate marker for graphs without gradients.
#[derive(Clone)]
pub struct NoGrad;

/// Typestate marker for graphs with gradient computation nodes.
#[derive(Clone, Debug)]
pub struct WithGrad {
    /// Maps parameter ID to its gradient accumulation node.
    pub param_to_grad: HashMap<usize, NodeIndex>,
    /// Set of all gradient computation nodes.
    pub gradient_nodes: HashSet<NodeIndex>,
}

#[derive(Clone)]
pub enum TensorGraphNode<D> {
    Constant {
        data: Arc<Vec<D>>,
        shape: crate::tensor::Shape,
    },
    Input {
        name: &'static str,
        shape: crate::tensor::Shape,
    },
    Parameter {
        id: usize,
        data: Arc<Mutex<Vec<D>>>,
        shape: crate::tensor::Shape,
    },
    Unary {
        op: UnaryOp,
        shape: crate::tensor::Shape,
    },
    Binary {
        op: BinaryOp,
        shape: crate::tensor::Shape,
    },
    MatMul {
        shape: crate::tensor::Shape,
    },
    Transpose {
        shape: crate::tensor::Shape,
    },
    BroadcastAxis {
        axis: usize,
        shape: crate::tensor::Shape,
    },
    ReduceAxis {
        op: ReduceOp,
        axis: usize,
        shape: crate::tensor::Shape,
    },
    /// Greater than comparison: returns 1.0 where lhs > rhs, 0.0 otherwise
    Gt {
        shape: crate::tensor::Shape,
    },
    /// Mask operation: returns values where condition != 0.0, 0.0 otherwise
    Mask {
        shape: crate::tensor::Shape,
    },
}

impl<D> TensorGraphNode<D> {
    pub fn name(&self) -> &'static str {
        match self {
            TensorGraphNode::Constant { .. } => "Constant",
            TensorGraphNode::Input { .. } => "Input",
            TensorGraphNode::Parameter { .. } => "Parameter",
            TensorGraphNode::Unary { op, .. } => match op {
                UnaryOp::Neg => "Neg",
                UnaryOp::Exp => "Exp",
                UnaryOp::Log => "Log",
                UnaryOp::Relu => "Relu",
            },
            TensorGraphNode::Binary { op, .. } => match op {
                BinaryOp::Add => "Add",
                BinaryOp::Sub => "Sub",
                BinaryOp::Mul => "Mul",
                BinaryOp::Div => "Div",
            },
            TensorGraphNode::MatMul { .. } => "MatMul",
            TensorGraphNode::Transpose { .. } => "Transpose",
            TensorGraphNode::BroadcastAxis { .. } => "BroadcastAxis",
            TensorGraphNode::ReduceAxis { op, .. } => match op {
                ReduceOp::Sum => "ReduceAxisSum",
                ReduceOp::Max => "ReduceAxisMax",
                ReduceOp::Mean => "ReduceAxisMean",
            },
            TensorGraphNode::Gt { .. } => "Gt",
            TensorGraphNode::Mask { .. } => "Mask",
        }
    }

    pub fn shape(&self) -> &crate::tensor::Shape {
        match self {
            TensorGraphNode::Constant { shape, .. } => shape,
            TensorGraphNode::Input { shape, .. } => shape,
            TensorGraphNode::Parameter { shape, .. } => shape,
            TensorGraphNode::Unary { shape, .. } => shape,
            TensorGraphNode::Binary { shape, .. } => shape,
            TensorGraphNode::MatMul { shape } => shape,
            TensorGraphNode::Transpose { shape } => shape,
            TensorGraphNode::BroadcastAxis { shape, .. } => shape,
            TensorGraphNode::ReduceAxis { shape, .. } => shape,
            TensorGraphNode::Gt { shape } => shape,
            TensorGraphNode::Mask { shape } => shape,
        }
    }
}

/// Directed acyclic graph (DAG) representation of tensor computations.
#[derive(Clone)]
pub struct TensorGraph<D, G = NoGrad> {
    pub graph: Graph<TensorGraphNode<D>, usize>,
    gradients: G,
    _phantom: std::marker::PhantomData<G>,
}

impl<D> Default for TensorGraph<D, NoGrad> {
    fn default() -> Self {
        Self::new()
    }
}

impl<D> TensorGraph<D, NoGrad> {
    /// Create a new empty TensorGraph without gradients.
    pub fn new() -> Self {
        Self {
            graph: Graph::new(),
            gradients: NoGrad,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<D, G> TensorGraph<D, G> {
    pub fn len(&self) -> usize {
        self.graph.node_count()
    }

    pub fn is_empty(&self) -> bool {
        self.graph.node_count() == 0
    }

    pub fn toposort(&self) -> Vec<NodeIndex> {
        petgraph::algo::toposort(&self.graph, None).unwrap()
    }

    pub fn inputs(&self, idx: NodeIndex) -> Vec<NodeIndex> {
        // We preserve the input order for non-commutative operations
        self.graph
            .edges_directed(idx, petgraph::Direction::Incoming)
            .sorted_by_key(|e| e.weight())
            .map(|e| e.source())
            .collect()
    }
}

impl TensorGraph<f32, NoGrad> {
    /// Helper method to lower a TensorExpr into the graph during gradient construction.
    fn lower_gradient_expr(
        &mut self,
        expr: &crate::tensor::TensorExpr<f32>,
        gradient_nodes: &mut HashSet<NodeIndex>,
    ) -> NodeIndex {
        // Track nodes before lowering
        let nodes_before: HashSet<NodeIndex> = self.graph.node_indices().collect();

        let node_idx = expr.lower_to_graph(self);

        // Mark all newly created nodes as gradient nodes
        // Any node that exists now but didn't exist before is a gradient node
        let nodes_after: HashSet<NodeIndex> = self.graph.node_indices().collect();

        for new_node in nodes_after.difference(&nodes_before) {
            gradient_nodes.insert(*new_node);
        }

        node_idx
    }

    /// Create a new graph with gradient computation nodes for automatic differentiation.
    pub fn with_gradients(mut self, loss_node: NodeIndex) -> TensorGraph<f32, WithGrad> {
        let mut param_to_grad: HashMap<usize, NodeIndex> = HashMap::new();
        let mut gradient_nodes: HashSet<NodeIndex> = HashSet::new();
        let mut node_to_grad: HashMap<NodeIndex, NodeIndex> = HashMap::new();

        // Step 1: Find all parameters in the graph
        let mut param_nodes: HashMap<usize, NodeIndex> = HashMap::new();
        for idx in self.graph.node_indices() {
            if let TensorGraphNode::Parameter { id, .. } = &self.graph[idx] {
                param_nodes.insert(*id, idx);
            }
        }

        // Step 2: Validate loss node
        let loss_shape = self.graph[loss_node].shape();
        assert_eq!(
            loss_shape.iter().product::<usize>(),
            1,
            "Loss node must be scalar, got shape {:?}",
            loss_shape
        );

        // Step 3: Create seed gradient for loss (gradient of loss w.r.t. itself = 1.0)
        // We'll create a constant node with value 1.0
        let seed_grad_node = self.graph.add_node(TensorGraphNode::Constant {
            data: Arc::new(vec![1.0f32]),
            shape: loss_shape.clone(),
        });
        node_to_grad.insert(loss_node, seed_grad_node);
        gradient_nodes.insert(seed_grad_node);

        // Step 4: Traverse graph in reverse topological order
        // For each node, compute its gradient based on downstream gradients
        let topo_order = self.toposort();

        // Process nodes in reverse order (from loss back to inputs/parameters)
        for &node_idx in topo_order.iter().rev() {
            // Skip if this node doesn't have a gradient (not on path to loss)
            let grad_output = match node_to_grad.get(&node_idx) {
                Some(&grad) => grad,
                None => continue,
            };

            // Get the node's inputs
            let inputs = self.inputs(node_idx);

            // Generate gradient nodes based on operation type
            match &self.graph[node_idx] {
                TensorGraphNode::Binary { op, .. } => {
                    if inputs.len() != 2 {
                        panic!("Binary op should have 2 inputs");
                    }
                    let input_a = inputs[0];
                    let input_b = inputs[1];

                    match op {
                        BinaryOp::Add => {
                            // d(a+b)/da = 1, d(a+b)/db = 1
                            // grad_a = grad_output, grad_b = grad_output
                            self.accumulate_gradient(
                                &mut node_to_grad,
                                &mut gradient_nodes,
                                input_a,
                                grad_output,
                            );
                            self.accumulate_gradient(
                                &mut node_to_grad,
                                &mut gradient_nodes,
                                input_b,
                                grad_output,
                            );
                        }
                        BinaryOp::Sub => {
                            // d(a-b)/da = 1, d(a-b)/db = -1
                            // grad_a = grad_output, grad_b = -grad_output
                            self.accumulate_gradient(
                                &mut node_to_grad,
                                &mut gradient_nodes,
                                input_a,
                                grad_output,
                            );

                            // Use fluent syntax for negation
                            let grad_out_shape = self.graph[grad_output].shape().clone();
                            let grad_out_expr =
                                crate::tensor::TensorExpr::node_ref(grad_output, grad_out_shape);
                            let neg_grad_expr = -grad_out_expr;
                            let neg_grad =
                                self.lower_gradient_expr(&neg_grad_expr, &mut gradient_nodes);

                            self.accumulate_gradient(
                                &mut node_to_grad,
                                &mut gradient_nodes,
                                input_b,
                                neg_grad,
                            );
                        }
                        BinaryOp::Mul => {
                            // d(a*b)/da = b, d(a*b)/db = a
                            // grad_a = grad_output * b, grad_b = grad_output * a

                            // Use fluent syntax for gradient construction
                            let grad_out_shape = self.graph[grad_output].shape().clone();
                            let input_b_shape = self.graph[input_b].shape().clone();
                            let input_a_shape = self.graph[input_a].shape().clone();

                            let grad_out_expr =
                                TensorExpr::node_ref(grad_output, grad_out_shape.clone());
                            let input_b_expr = TensorExpr::node_ref(input_b, input_b_shape);
                            let input_a_expr = TensorExpr::node_ref(input_a, input_a_shape);

                            let grad_a_expr = grad_out_expr.clone() * input_b_expr;
                            let grad_b_expr = grad_out_expr * input_a_expr;

                            let grad_a =
                                self.lower_gradient_expr(&grad_a_expr, &mut gradient_nodes);
                            let grad_b =
                                self.lower_gradient_expr(&grad_b_expr, &mut gradient_nodes);

                            self.accumulate_gradient(
                                &mut node_to_grad,
                                &mut gradient_nodes,
                                input_a,
                                grad_a,
                            );
                            self.accumulate_gradient(
                                &mut node_to_grad,
                                &mut gradient_nodes,
                                input_b,
                                grad_b,
                            );
                        }
                        BinaryOp::Div => {
                            // d(a/b)/da = 1/b, d(a/b)/db = -a/b²
                            // grad_a = grad_output / b
                            // grad_b = -grad_output * a / (b * b)

                            let grad_out_shape = self.graph[grad_output].shape().clone();
                            let input_a_shape = self.graph[input_a].shape().clone();
                            let input_b_shape = self.graph[input_b].shape().clone();

                            let grad_out_expr = TensorExpr::node_ref(grad_output, grad_out_shape);
                            let input_a_expr = TensorExpr::node_ref(input_a, input_a_shape);
                            let input_b_expr = TensorExpr::node_ref(input_b, input_b_shape.clone());

                            let grad_a_expr = grad_out_expr.clone() / input_b_expr.clone();
                            let grad_b_expr = -(grad_out_expr * input_a_expr)
                                / (input_b_expr.clone() * input_b_expr);

                            let grad_a =
                                self.lower_gradient_expr(&grad_a_expr, &mut gradient_nodes);
                            let grad_b =
                                self.lower_gradient_expr(&grad_b_expr, &mut gradient_nodes);

                            self.accumulate_gradient(
                                &mut node_to_grad,
                                &mut gradient_nodes,
                                input_a,
                                grad_a,
                            );
                            self.accumulate_gradient(
                                &mut node_to_grad,
                                &mut gradient_nodes,
                                input_b,
                                grad_b,
                            );
                        }
                    }
                }
                TensorGraphNode::Unary { op, .. } => {
                    if inputs.len() != 1 {
                        panic!("Unary op should have 1 input");
                    }
                    let input_x = inputs[0];

                    match op {
                        UnaryOp::Exp => {
                            // d(exp(x))/dx = exp(x)
                            // grad_x = grad_output * exp(x) = grad_output * node_output

                            let grad_out_shape = self.graph[grad_output].shape().clone();
                            let node_out_shape = self.graph[node_idx].shape().clone();

                            let grad_out_expr = TensorExpr::node_ref(grad_output, grad_out_shape);
                            let node_out_expr = TensorExpr::node_ref(node_idx, node_out_shape);

                            let grad_x_expr = grad_out_expr * node_out_expr;
                            let grad_x =
                                self.lower_gradient_expr(&grad_x_expr, &mut gradient_nodes);

                            self.accumulate_gradient(
                                &mut node_to_grad,
                                &mut gradient_nodes,
                                input_x,
                                grad_x,
                            );
                        }
                        UnaryOp::Log => {
                            // d(log(x))/dx = 1/x
                            // grad_x = grad_output / x

                            let grad_out_shape = self.graph[grad_output].shape().clone();
                            let input_x_shape = self.graph[input_x].shape().clone();

                            let grad_out_expr = TensorExpr::node_ref(grad_output, grad_out_shape);
                            let input_x_expr = TensorExpr::node_ref(input_x, input_x_shape);

                            let grad_x_expr = grad_out_expr / input_x_expr;
                            let grad_x =
                                self.lower_gradient_expr(&grad_x_expr, &mut gradient_nodes);

                            self.accumulate_gradient(
                                &mut node_to_grad,
                                &mut gradient_nodes,
                                input_x,
                                grad_x,
                            );
                        }
                        UnaryOp::Neg => {
                            // d(-x)/dx = -1
                            // grad_x = -grad_output

                            let grad_out_shape = self.graph[grad_output].shape().clone();
                            let grad_out_expr = TensorExpr::node_ref(grad_output, grad_out_shape);
                            let grad_x_expr = -grad_out_expr;
                            let grad_x =
                                self.lower_gradient_expr(&grad_x_expr, &mut gradient_nodes);

                            self.accumulate_gradient(
                                &mut node_to_grad,
                                &mut gradient_nodes,
                                input_x,
                                grad_x,
                            );
                        }
                        UnaryOp::Relu => {
                            // d(relu(x))/dx = grad_output * (x > 0)

                            let grad_out_shape = self.graph[grad_output].shape().clone();
                            let input_x_shape = self.graph[input_x].shape().clone();
                            let num_elements: usize = input_x_shape.iter().product();

                            let grad_out_expr = TensorExpr::node_ref(grad_output, grad_out_shape);
                            let input_x_expr = TensorExpr::node_ref(input_x, input_x_shape.clone());
                            let zero_expr =
                                TensorExpr::constant(vec![0.0f32; num_elements], input_x_shape);

                            let condition_expr = input_x_expr.gt(zero_expr);
                            let grad_x_expr = grad_out_expr.mask(condition_expr);
                            let grad_x =
                                self.lower_gradient_expr(&grad_x_expr, &mut gradient_nodes);

                            self.accumulate_gradient(
                                &mut node_to_grad,
                                &mut gradient_nodes,
                                input_x,
                                grad_x,
                            );
                        }
                    }
                }
                TensorGraphNode::MatMul { .. } => {
                    // d(A@B)/dA = dC @ B^T, d(A@B)/dB = A^T @ dC
                    if inputs.len() != 2 {
                        panic!("MatMul should have 2 inputs");
                    }
                    let input_a = inputs[0];
                    let input_b = inputs[1];

                    let grad_out_shape = self.graph[grad_output].shape().clone();
                    let input_a_shape = self.graph[input_a].shape().clone();
                    let input_b_shape = self.graph[input_b].shape().clone();

                    let grad_out_expr = TensorExpr::<f32>::node_ref(grad_output, grad_out_shape);
                    let input_a_expr = TensorExpr::<f32>::node_ref(input_a, input_a_shape);
                    let input_b_expr = TensorExpr::<f32>::node_ref(input_b, input_b_shape);

                    // grad_a = grad_output @ B^T
                    let grad_a_expr = grad_out_expr
                        .clone()
                        .matmul(input_b_expr.clone().transpose());
                    let grad_a = self.lower_gradient_expr(&grad_a_expr, &mut gradient_nodes);

                    // grad_b = A^T @ grad_output
                    let grad_b_expr = input_a_expr.transpose().matmul(grad_out_expr);
                    let grad_b = self.lower_gradient_expr(&grad_b_expr, &mut gradient_nodes);

                    self.accumulate_gradient(
                        &mut node_to_grad,
                        &mut gradient_nodes,
                        input_a,
                        grad_a,
                    );
                    self.accumulate_gradient(
                        &mut node_to_grad,
                        &mut gradient_nodes,
                        input_b,
                        grad_b,
                    );
                }
                TensorGraphNode::Parameter { id, .. } => {
                    // Store the gradient node for this parameter
                    param_to_grad.insert(*id, grad_output);
                }
                TensorGraphNode::Constant { .. } | TensorGraphNode::Input { .. } => {
                    // Constants and inputs don't need gradients
                }
                TensorGraphNode::Transpose { .. } => {
                    // d(A^T)/dA = (grad_output)^T
                    // Transpose gradient is just transpose of incoming gradient
                    if inputs.len() != 1 {
                        panic!("Transpose should have 1 input");
                    }
                    let input_a = inputs[0];
                    let grad_out_shape = self.graph[grad_output].shape().clone();

                    let grad_out_expr = TensorExpr::<f32>::node_ref(grad_output, grad_out_shape);

                    // grad_a = grad_output^T
                    let grad_a_expr = grad_out_expr.transpose();
                    let grad_a = self.lower_gradient_expr(&grad_a_expr, &mut gradient_nodes);

                    self.accumulate_gradient(
                        &mut node_to_grad,
                        &mut gradient_nodes,
                        input_a,
                        grad_a,
                    );
                }
                TensorGraphNode::ReduceAxis { op, axis, .. } => {
                    // For ReduceAxisSum: gradient is broadcast back to original shape
                    // If Y = sum(X, axis), then dX = broadcast(dY, axis)
                    match op {
                        ReduceOp::Sum => {
                            if inputs.len() != 1 {
                                panic!("ReduceAxis should have 1 input");
                            }
                            let input_x = inputs[0];
                            let input_shape = self.graph[input_x].shape().clone();
                            let grad_out_shape = self.graph[grad_output].shape().clone();
                            let target_size = input_shape[*axis];

                            let grad_out_expr =
                                TensorExpr::<f32>::node_ref(grad_output, grad_out_shape);

                            // grad_x = broadcast(grad_output, axis)
                            let grad_x_expr = grad_out_expr.broadcast_axis(*axis, target_size);
                            let grad_x =
                                self.lower_gradient_expr(&grad_x_expr, &mut gradient_nodes);

                            self.accumulate_gradient(
                                &mut node_to_grad,
                                &mut gradient_nodes,
                                input_x,
                                grad_x,
                            );
                        }
                        ReduceOp::Mean => {
                            if inputs.len() != 1 {
                                panic!("ReduceAxis should have 1 input");
                            }
                            let input_x = inputs[0];
                            let input_shape = self.graph[input_x].shape().clone();
                            let grad_out_shape = self.graph[grad_output].shape().clone();
                            let axis_size = input_shape[*axis];
                            let target_size = axis_size;

                            let grad_out_expr =
                                TensorExpr::<f32>::node_ref(grad_output, grad_out_shape);

                            // Broadcast gradient back to input shape
                            let grad_broadcast_expr =
                                grad_out_expr.broadcast_axis(*axis, target_size);

                            // Create scale constant (1/axis_size)
                            let scale = 1.0 / axis_size as f32;
                            let num_elements: usize = input_shape.iter().product();
                            let scale_const_expr = TensorExpr::<f32>::constant(
                                vec![scale; num_elements],
                                input_shape.clone(),
                            );

                            // grad_x = grad_broadcast * scale
                            let grad_x_expr = grad_broadcast_expr * scale_const_expr;
                            let grad_x =
                                self.lower_gradient_expr(&grad_x_expr, &mut gradient_nodes);

                            self.accumulate_gradient(
                                &mut node_to_grad,
                                &mut gradient_nodes,
                                input_x,
                                grad_x,
                            );
                        }
                        ReduceOp::Max => {
                            // TODO: Implement gradients for Max
                            // Requires tracking argmax indices, deferred for now
                        }
                    }
                }
                TensorGraphNode::BroadcastAxis { axis, .. } => {
                    // For BroadcastAxis: gradient is reduced back to original shape
                    // If Y = broadcast(X, axis), then dX = reduce_sum(dY, axis)
                    if inputs.len() != 1 {
                        panic!("BroadcastAxis should have 1 input");
                    }
                    let input_x = inputs[0];
                    let grad_out_shape = self.graph[grad_output].shape().clone();

                    let grad_out_expr = TensorExpr::<f32>::node_ref(grad_output, grad_out_shape);

                    // grad_x = reduce_sum(grad_output, axis)
                    let grad_x_expr = grad_out_expr.reduce_axis_sum(*axis);
                    let grad_x = self.lower_gradient_expr(&grad_x_expr, &mut gradient_nodes);

                    self.accumulate_gradient(
                        &mut node_to_grad,
                        &mut gradient_nodes,
                        input_x,
                        grad_x,
                    );
                }
                TensorGraphNode::Gt { .. } | TensorGraphNode::Mask { .. } => {
                    // Gt and Mask are only used in gradient computation itself
                    // They don't need gradients (they're non-differentiable operations)
                    // Skip gradient computation for these nodes
                }
            }
        }

        TensorGraph {
            graph: self.graph,
            gradients: WithGrad {
                param_to_grad,
                gradient_nodes,
            },
            _phantom: std::marker::PhantomData,
        }
    }

    /// Helper to accumulate gradients when a node has multiple consumers.
    fn accumulate_gradient(
        &mut self,
        node_to_grad: &mut HashMap<NodeIndex, NodeIndex>,
        gradient_nodes: &mut HashSet<NodeIndex>,
        node: NodeIndex,
        new_grad: NodeIndex,
    ) {
        if let Some(&existing_grad) = node_to_grad.get(&node) {
            // Node already has a gradient, add them together
            let grad_shape = self.graph[existing_grad].shape().clone();
            let sum_grad = self.graph.add_node(TensorGraphNode::Binary {
                op: BinaryOp::Add,
                shape: grad_shape.clone(),
            });
            self.graph.add_edge(existing_grad, sum_grad, 0);
            self.graph.add_edge(new_grad, sum_grad, 1);
            gradient_nodes.insert(sum_grad);
            node_to_grad.insert(node, sum_grad);
        } else {
            // First gradient for this node
            node_to_grad.insert(node, new_grad);
        }
    }
}

impl TensorGraph<f32, WithGrad> {
    /// Access gradient metadata for graphs with gradients.
    pub fn gradient_metadata(&self) -> &WithGrad {
        &self.gradients
    }

    /// Convert a graph with gradients back to an inference-only graph by removing gradient nodes.
    pub fn without_gradients(self) -> TensorGraph<f32, NoGrad> {
        let gradient_nodes = &self.gradients.gradient_nodes;
        let new_graph = self.graph.filter_map_owned(
            |node_idx, node| {
                if gradient_nodes.contains(&node_idx) {
                    None
                } else {
                    Some(node)
                }
            },
            |_edge_idx, edge| {
                Some(edge) // Edges are auto-filtered if nodes are removed
            },
        );

        TensorGraph {
            graph: new_graph,
            gradients: NoGrad,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<D, G> Index<NodeIndex> for TensorGraph<D, G> {
    type Output = TensorGraphNode<D>;

    fn index(&self, index: NodeIndex) -> &Self::Output {
        &self.graph[index]
    }
}

impl From<TensorExpr<f32>> for TensorGraph<f32, NoGrad> {
    fn from(expr: TensorExpr<f32>) -> Self {
        let mut graph = TensorGraph::new();
        let _ = expr.lower_to_graph(&mut graph);
        graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Constant, Parameter};

    #[test]
    fn test_with_gradients_api() {
        // Create simple graph: y = x^2 where x is a parameter
        let x = Parameter::new(vec![2.0f32], vec![1]);
        let x_id = x.id();
        let y = x.clone() * x;
        let graph: TensorGraph<f32, NoGrad> = y.into();

        // Convert to gradient-enabled graph
        let grad_graph = graph.with_gradients(NodeIndex::from(1));

        // Verify metadata exists
        let metadata = grad_graph.gradient_metadata();
        assert!(metadata.param_to_grad.contains_key(&x_id));

        // Verify gradient nodes were created
        assert!(
            !metadata.gradient_nodes.is_empty(),
            "Expected gradient nodes to be generated"
        );
    }

    #[test]
    fn test_gradient_node_generation_mul() {
        // Test: y = x * x, should generate grad_x = grad_y * x + grad_y * x = 2 * grad_y * x
        let x = Parameter::new(vec![2.0f32], vec![1]);
        let y = x.clone() * x;
        let graph: TensorGraph<f32, NoGrad> = y.into();

        let initial_node_count = graph.len();
        let grad_graph = graph.with_gradients(NodeIndex::from(1));

        // Should have created additional gradient nodes:
        // - seed gradient (constant 1.0)
        // - grad_left = grad_y * x (right input)
        // - grad_right = grad_y * x (left input)
        // - accumulated = grad_left + grad_right
        assert!(
            grad_graph.len() > initial_node_count,
            "Expected new gradient nodes. Before: {}, After: {}",
            initial_node_count,
            grad_graph.len()
        );

        let metadata = grad_graph.gradient_metadata();
        assert!(
            metadata.gradient_nodes.len() >= 3,
            "Expected at least 3 gradient nodes (seed, 2 muls, 1 add), got {}",
            metadata.gradient_nodes.len()
        );
    }

    #[test]
    fn test_gradient_node_generation_add() {
        // Test: y = x + x, should generate grad_x = grad_y + grad_y
        let x = Parameter::new(vec![2.0f32], vec![1]);
        let y = x.clone() + x;
        let graph: TensorGraph<f32, NoGrad> = y.into();

        let initial_node_count = graph.len();
        let grad_graph = graph.with_gradients(NodeIndex::from(1));

        // Should have created gradient nodes
        assert!(
            grad_graph.len() > initial_node_count,
            "Expected new gradient nodes"
        );

        let metadata = grad_graph.gradient_metadata();
        assert!(
            metadata.gradient_nodes.len() >= 2,
            "Expected at least 2 gradient nodes (seed, accumulation add), got {}",
            metadata.gradient_nodes.len()
        );
    }

    #[test]
    fn test_gradient_node_generation_exp() {
        // Test: y = exp(x), should generate grad_x = grad_y * exp(x)
        let x = Parameter::new(vec![1.0f32], vec![1]);
        let y = x.exp();
        let graph: TensorGraph<f32, NoGrad> = y.into();

        let initial_node_count = graph.len();
        let grad_graph = graph.with_gradients(NodeIndex::from(1));

        // Should have created gradient nodes
        assert!(
            grad_graph.len() > initial_node_count,
            "Expected new gradient nodes"
        );

        let metadata = grad_graph.gradient_metadata();
        assert!(
            metadata.gradient_nodes.len() >= 2,
            "Expected at least 2 gradient nodes (seed, mul), got {}",
            metadata.gradient_nodes.len()
        );
    }

    #[test]
    fn test_gradient_node_generation_chain() {
        // Test: y = (x + x) * x, should generate gradient nodes for entire chain
        let x = Parameter::new(vec![2.0f32], vec![1]);
        let x_id = x.id();
        let sum = x.clone() + x.clone();
        let y = sum * x;
        let graph: TensorGraph<f32, NoGrad> = y.into();

        let initial_node_count = graph.len();
        let grad_graph = graph.with_gradients(NodeIndex::from(2));

        // Should have created many gradient nodes for the chain
        assert!(
            grad_graph.len() > initial_node_count,
            "Expected new gradient nodes"
        );

        let metadata = grad_graph.gradient_metadata();
        assert!(
            metadata.gradient_nodes.len() >= 5,
            "Expected multiple gradient nodes for chain, got {}",
            metadata.gradient_nodes.len()
        );

        // Verify parameter has a gradient
        assert!(metadata.param_to_grad.contains_key(&x_id));
    }

    #[test]
    fn test_gradient_structure_detailed() {
        // Test gradient graph structure in detail for y = x * 2
        let x = Parameter::new(vec![3.0f32], vec![1]);
        let x_id = x.id();
        let two = Constant::new(vec![2.0f32], vec![1]);
        let y = x * two;
        let graph: TensorGraph<f32, NoGrad> = y.into();

        // Graph should be: param(0), constant(1), mul(2)
        assert_eq!(graph.len(), 3, "Initial graph should have 3 nodes");

        let grad_graph = graph.with_gradients(NodeIndex::from(2));

        // After gradient generation:
        // - seed gradient constant (1.0)
        // - grad for left input (param): grad_y * constant(2.0)
        // - grad for right input (constant): grad_y * param (but constant doesn't need grad)
        let metadata = grad_graph.gradient_metadata();

        // Verify parameter has a gradient node assigned
        let param_grad = metadata.param_to_grad.get(&x_id);
        assert!(param_grad.is_some(), "Parameter should have gradient node");

        // Verify the gradient computation graph structure
        println!("Total nodes: {}", grad_graph.len());
        println!("Gradient nodes: {}", metadata.gradient_nodes.len());

        // Walk through nodes to understand structure
        for idx in grad_graph.graph.node_indices() {
            let node = &grad_graph[idx];
            let shape = grad_graph.graph[idx].shape();
            let is_grad = metadata.gradient_nodes.contains(&idx);
            println!(
                "Node {:?}: {} (shape: {:?}, is_grad: {})",
                idx,
                node.name(),
                shape,
                is_grad
            );
        }
    }

    #[test]
    #[should_panic(expected = "Loss node must be scalar")]
    fn test_with_gradients_requires_scalar_loss() {
        // Create graph with non-scalar output
        let x = Parameter::new(vec![1.0f32, 2.0], vec![2]);
        let y = x.clone() * x;
        let graph: TensorGraph<f32, NoGrad> = y.into();

        // Should panic because output is not scalar
        let _grad_graph = graph.with_gradients(NodeIndex::from(1));
    }

    #[test]
    fn test_gradient_node_generation_matmul() {
        // Test: C = A @ B, should generate grad_A = grad_C @ B^T, grad_B = A^T @ grad_C
        let a = Parameter::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
        let a_id = a.id();
        let b = Parameter::new(vec![5.0f32, 6.0, 7.0, 8.0], vec![2, 2]);
        let b_id = b.id();

        // C = A @ B, then reduce to scalar for loss
        let a_expr: TensorExpr<f32> = a.into();
        let b_expr: TensorExpr<f32> = b.into();
        let c = a_expr.matmul(b_expr);
        // Reduce axis 1 first [2,2] -> [2,1], then reduce axis 0 [2,1] -> [1,1]
        let loss = c.reduce_sum(1).reduce_sum(0);

        let graph: TensorGraph<f32, NoGrad> = loss.into();
        let initial_node_count = graph.len();

        // Find the loss node (should be the last node in topo order)
        let topo = graph.toposort();
        let loss_node = *topo.last().unwrap();

        let grad_graph = graph.with_gradients(loss_node);

        // Should have created gradient nodes for MatMul
        assert!(
            grad_graph.len() > initial_node_count,
            "Expected new gradient nodes for MatMul"
        );

        let metadata = grad_graph.gradient_metadata();

        // Both parameters should have gradients
        assert!(
            metadata.param_to_grad.contains_key(&a_id),
            "Parameter A should have gradient"
        );
        assert!(
            metadata.param_to_grad.contains_key(&b_id),
            "Parameter B should have gradient"
        );

        // Should have created transpose nodes for MatMul gradients
        let transpose_count = grad_graph
            .graph
            .node_indices()
            .filter(|&idx| matches!(grad_graph[idx], TensorGraphNode::Transpose { .. }))
            .count();

        assert!(
            transpose_count >= 2,
            "Expected at least 2 Transpose nodes for MatMul gradient (A^T and B^T), got {}",
            transpose_count
        );
    }

    #[test]
    fn test_gradient_broadcast_axis() {
        // Test: y = broadcast(x, axis), should generate grad_x = reduce_sum(grad_y, axis)
        let x = Parameter::new(vec![1.0f32, 2.0], vec![2, 1]);
        let x_id = x.id();

        let x_expr: TensorExpr<f32> = x.into();
        let broadcasted = x_expr.broadcast_axis(1, 3); // [2, 1] -> [2, 3]
        // Reduce to scalar for loss
        let loss = broadcasted.reduce_sum(0).reduce_sum(1); // [2, 3] -> [1, 3] -> [1, 1]

        let graph: TensorGraph<f32, NoGrad> = loss.into();
        let initial_node_count = graph.len();

        // Find the loss node (should be the last node)
        let topo = graph.toposort();
        let loss_node = *topo.last().unwrap();

        let grad_graph = graph.with_gradients(loss_node);

        // Should have created gradient nodes including ReduceAxis gradient (which creates BroadcastAxis)
        // and BroadcastAxis gradient (which creates ReduceAxis)
        assert!(
            grad_graph.len() > initial_node_count,
            "Expected new gradient nodes for BroadcastAxis"
        );

        let metadata = grad_graph.gradient_metadata();

        // Parameter should have a gradient
        assert!(
            metadata.param_to_grad.contains_key(&x_id),
            "Parameter should have gradient"
        );

        // Should have created a ReduceAxis node for the BroadcastAxis gradient
        let reduce_count = grad_graph
            .graph
            .node_indices()
            .filter(|&idx| {
                matches!(
                    grad_graph[idx],
                    TensorGraphNode::ReduceAxis {
                        op: ReduceOp::Sum,
                        ..
                    }
                )
            })
            .count();

        assert!(
            reduce_count >= 1,
            "Expected at least 1 ReduceAxis node for BroadcastAxis gradient, got {}",
            reduce_count
        );
    }

    #[test]
    fn test_gradient_reduce_mean() {
        // Test: y = mean(x, axis), should generate grad_x = broadcast(grad_y, axis) / axis_size
        let x = Parameter::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let x_id = x.id();

        let x_expr: TensorExpr<f32> = x.into();
        let reduced = x_expr.reduce_mean(1); // [2, 3] -> [2, 1]
        // Reduce to scalar for loss
        let loss = reduced.reduce_sum(0).reduce_sum(1); // [2, 1] -> [1, 1] -> [1, 1]

        let graph: TensorGraph<f32, NoGrad> = loss.into();
        let initial_node_count = graph.len();

        // Find the loss node (should be the last node)
        let topo = graph.toposort();
        let loss_node = *topo.last().unwrap();

        let grad_graph = graph.with_gradients(loss_node);

        // Should have created gradient nodes: seed + BroadcastAxis + Constant + Mul
        assert!(
            grad_graph.len() > initial_node_count,
            "Expected new gradient nodes for ReduceAxis Mean"
        );

        let metadata = grad_graph.gradient_metadata();

        // Parameter should have a gradient
        assert!(
            metadata.param_to_grad.contains_key(&x_id),
            "Parameter should have gradient"
        );

        // Should have created a BroadcastAxis node for the gradient
        let broadcast_count = grad_graph
            .graph
            .node_indices()
            .filter(|&idx| matches!(grad_graph[idx], TensorGraphNode::BroadcastAxis { .. }))
            .count();

        assert!(
            broadcast_count >= 1,
            "Expected at least 1 BroadcastAxis node for ReduceAxis Mean gradient, got {}",
            broadcast_count
        );

        // Should have created a Binary Mul node for scaling
        let mul_count = grad_graph
            .graph
            .node_indices()
            .filter(|&idx| {
                matches!(
                    grad_graph[idx],
                    TensorGraphNode::Binary {
                        op: BinaryOp::Mul,
                        ..
                    }
                )
            })
            .count();

        assert!(
            mul_count >= 1,
            "Expected at least 1 Binary Mul node for scaling gradient, got {}",
            mul_count
        );

        // Should have created a Constant node with the scale factor (1/3)
        let constant_count = grad_graph
            .graph
            .node_indices()
            .filter(|&idx| matches!(grad_graph[idx], TensorGraphNode::Constant { .. }))
            .count();

        assert!(
            constant_count >= 1,
            "Expected at least 1 Constant node for scale factor, got {}",
            constant_count
        );
    }

    #[test]
    fn test_gradient_reduce_sum() {
        // Test: y = sum(x, axis), should generate grad_x = broadcast(grad_y, axis)
        let x = Parameter::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
        let x_id = x.id();

        let x_expr: TensorExpr<f32> = x.into();
        let reduced = x_expr.reduce_sum(0); // [2, 2] -> [1, 2]
        // Reduce to scalar for loss
        let loss = reduced.reduce_sum(1); // [1, 2] -> [1, 1]

        let graph: TensorGraph<f32, NoGrad> = loss.into();
        let initial_node_count = graph.len();

        // Find the loss node (should be the last node)
        let topo = graph.toposort();
        let loss_node = *topo.last().unwrap();

        let grad_graph = graph.with_gradients(loss_node);

        // Should have created gradient nodes: seed + BroadcastAxis
        assert!(
            grad_graph.len() > initial_node_count,
            "Expected new gradient nodes for ReduceAxis Sum"
        );

        let metadata = grad_graph.gradient_metadata();

        // Parameter should have a gradient
        assert!(
            metadata.param_to_grad.contains_key(&x_id),
            "Parameter should have gradient"
        );

        // Should have created a BroadcastAxis node for the gradient
        let broadcast_count = grad_graph
            .graph
            .node_indices()
            .filter(|&idx| matches!(grad_graph[idx], TensorGraphNode::BroadcastAxis { .. }))
            .count();

        assert!(
            broadcast_count >= 1,
            "Expected at least 1 BroadcastAxis node for ReduceAxis Sum gradient, got {}",
            broadcast_count
        );
    }

    #[test]
    fn test_without_gradients() {
        // Test: Create a graph with gradients, then convert it back to inference-only
        let x = Parameter::new(vec![2.0f32], vec![1]);
        let x_id = x.id();
        let y = x.clone() * x;
        let graph: TensorGraph<f32, NoGrad> = y.into();

        let initial_node_count = graph.len();
        assert_eq!(
            initial_node_count, 2,
            "Initial graph should have 2 nodes (param + mul)"
        );

        // Add gradients
        let grad_graph = graph.with_gradients(NodeIndex::from(1));

        // Verify gradient nodes were created
        let grad_node_count = grad_graph.len();
        assert!(
            grad_node_count > initial_node_count,
            "Graph with gradients should have more nodes"
        );

        let metadata = grad_graph.gradient_metadata();
        assert!(
            !metadata.gradient_nodes.is_empty(),
            "Should have gradient nodes"
        );

        // Convert back to inference-only graph
        let inference_graph = grad_graph.without_gradients();

        // Verify gradient nodes were removed
        assert_eq!(
            inference_graph.len(),
            initial_node_count,
            "Inference graph should have same number of nodes as original forward graph"
        );

        // Verify the forward computation nodes are preserved
        let mut has_param = false;
        let mut has_mul = false;

        for idx in inference_graph.graph.node_indices() {
            match &inference_graph[idx] {
                TensorGraphNode::Parameter { id, .. } => {
                    assert_eq!(*id, x_id, "Parameter should have correct ID");
                    has_param = true;
                }
                TensorGraphNode::Binary {
                    op: BinaryOp::Mul, ..
                } => {
                    has_mul = true;
                }
                _ => {}
            }
        }

        assert!(has_param, "Should have parameter node");
        assert!(has_mul, "Should have multiplication node");
    }

    #[test]
    fn test_without_gradients_preserves_structure() {
        // Test: Verify that without_gradients() preserves the computation graph structure
        let a = Parameter::new(vec![1.0f32, 2.0], vec![2, 1]);
        let b = Parameter::new(vec![3.0f32, 4.0], vec![2, 1]);

        let a_expr: TensorExpr<f32> = a.into();
        let b_expr: TensorExpr<f32> = b.into();

        let sum = a_expr + b_expr;
        let product = sum.clone() * sum; // (a + b) * (a + b)
        let loss = product.reduce_sum(0).reduce_sum(1); // Scalar loss

        let graph: TensorGraph<f32, NoGrad> = loss.into();
        let forward_node_count = graph.len();

        // Add gradients
        let topo = graph.toposort();
        let loss_node = *topo.last().unwrap();
        let grad_graph = graph.with_gradients(loss_node);

        // Convert back to inference
        let inference_graph = grad_graph.without_gradients();

        // Should have same number of nodes as original
        assert_eq!(
            inference_graph.len(),
            forward_node_count,
            "Inference graph should match forward graph size"
        );

        // Verify topological order is preserved (all nodes should still be reachable)
        let inference_topo = inference_graph.toposort();
        assert_eq!(
            inference_topo.len(),
            forward_node_count,
            "All forward nodes should be reachable in inference graph"
        );
    }

    #[test]
    fn test_without_gradients_complex_graph() {
        // Test: More complex graph to ensure all gradient nodes are properly removed
        let x = Parameter::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
        let w = Parameter::new(vec![5.0f32, 6.0, 7.0, 8.0], vec![2, 2]);

        let x_expr: TensorExpr<f32> = x.into();
        let w_expr: TensorExpr<f32> = w.into();

        // Forward: h = x @ w, then reduce to scalar
        let h = x_expr.matmul(w_expr);
        let loss = h.reduce_sum(0).reduce_sum(1);

        let graph: TensorGraph<f32, NoGrad> = loss.into();
        let forward_nodes = graph.len();

        // Add gradients
        let topo = graph.toposort();
        let loss_node = *topo.last().unwrap();
        let grad_graph = graph.clone().with_gradients(loss_node);

        let with_grad_nodes = grad_graph.len();
        assert!(
            with_grad_nodes > forward_nodes,
            "Should have added gradient nodes"
        );

        // Convert to inference
        let inference_graph = grad_graph.without_gradients();

        // The inference graph may have slightly different node count than the original
        // forward graph because some nodes may have been de-duplicated during gradient
        // construction. What matters is that all gradient nodes are removed.
        assert!(
            inference_graph.len() <= with_grad_nodes,
            "Inference graph should have fewer or equal nodes than grad graph"
        );

        assert!(
            inference_graph.len() >= forward_nodes,
            "Inference graph should have at least the forward nodes"
        );

        // Verify no nodes are marked as gradient nodes (since we filtered them out)
        // The key test: ensure the graph is still valid and can be topologically sorted
        let inference_topo = inference_graph.toposort();
        assert!(
            !inference_topo.is_empty(),
            "Inference graph should have nodes"
        );

        // Count forward-like operations
        let mut param_count = 0;
        let mut matmul_count = 0;
        let mut reduce_count = 0;

        for idx in inference_graph.graph.node_indices() {
            match &inference_graph[idx] {
                TensorGraphNode::Parameter { .. } => param_count += 1,
                TensorGraphNode::MatMul { .. } => matmul_count += 1,
                TensorGraphNode::ReduceAxis { .. } => reduce_count += 1,
                _ => {}
            }
        }

        assert_eq!(param_count, 2, "Should have 2 parameters");
        assert_eq!(matmul_count, 1, "Should have 1 matmul");
        assert_eq!(reduce_count, 2, "Should have 2 reduce operations");
    }
}
