use std::collections::HashMap;
use std::collections::HashSet;
use std::ops::Index;
use std::sync::Arc;
use std::sync::Mutex;

use itertools::Itertools;
use petgraph::graph::Graph;
pub use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;

use crate::BinaryOp;
use crate::ReduceOp;
use crate::TensorExpr;
use crate::UnaryOp;

/// Typestate marker for graphs without gradients.
#[derive(Clone)]
pub struct NoGrad;

/// Typestate marker for graphs with gradient computation nodes.
#[derive(Clone)]
pub struct WithGrad {
    data: GradientMetadata,
}

/// Metadata tracking gradient nodes and parameter mappings.
#[derive(Clone, Debug)]
pub struct GradientMetadata {
    /// Maps parameter ID to its gradient accumulation node.
    pub param_to_grad: HashMap<usize, NodeIndex>,
    /// Set of all gradient computation nodes.
    pub gradient_nodes: HashSet<NodeIndex>,
}

#[derive(Clone)]
pub enum TensorGraphNode<D> {
    Constant { data: Arc<Vec<D>> },
    Input { name: &'static str },
    Parameter { id: usize, data: Arc<Mutex<Vec<D>>> },
    Unary { op: UnaryOp },
    Binary { op: BinaryOp },
    MatMul,
    Transpose,
    BroadcastAxis { axis: usize },
    ReduceAxis { op: ReduceOp, axis: usize },
    /// Greater than comparison: returns 1.0 where lhs > rhs, 0.0 otherwise
    Gt,
    /// Mask operation: returns values where condition != 0.0, 0.0 otherwise
    Mask,
}

impl<D> TensorGraphNode<D> {
    pub fn name(&self) -> &'static str {
        match self {
            TensorGraphNode::Constant { .. } => "Constant",
            TensorGraphNode::Input { .. } => "Input",
            TensorGraphNode::Parameter { .. } => "Parameter",
            TensorGraphNode::Unary { op } => match op {
                UnaryOp::Neg => "Neg",
                UnaryOp::Exp => "Exp",
                UnaryOp::Log => "Log",
                UnaryOp::Relu => "Relu",
            },
            TensorGraphNode::Binary { op } => match op {
                BinaryOp::Add => "Add",
                BinaryOp::Sub => "Sub",
                BinaryOp::Mul => "Mul",
                BinaryOp::Div => "Div",
            },
            TensorGraphNode::MatMul => "MatMul",
            TensorGraphNode::Transpose => "Transpose",
            TensorGraphNode::BroadcastAxis { .. } => "BroadcastAxis",
            TensorGraphNode::ReduceAxis { op, .. } => match op {
                ReduceOp::Sum => "ReduceAxisSum",
                ReduceOp::Max => "ReduceAxisMax",
                ReduceOp::Mean => "ReduceAxisMean",
            },
            TensorGraphNode::Gt => "Gt",
            TensorGraphNode::Mask => "Mask",
        }
    }
}

impl<D> From<UnaryOp> for TensorGraphNode<D> {
    fn from(op: UnaryOp) -> Self {
        TensorGraphNode::Unary { op }
    }
}

impl<D> From<BinaryOp> for TensorGraphNode<D> {
    fn from(op: BinaryOp) -> Self {
        TensorGraphNode::Binary { op }
    }
}

#[derive(Clone)]
pub struct TensorGraph<D, G = NoGrad> {
    pub graph: Graph<TensorGraphNode<D>, usize>,
    pub shapes: HashMap<NodeIndex, crate::Shape>,
    gradients: G,
    _phantom: std::marker::PhantomData<G>,
}

impl<D> TensorGraph<D, NoGrad> {
    /// Create a new empty TensorGraph without gradients.
    pub fn new() -> Self {
        Self {
            graph: Graph::new(),
            shapes: HashMap::new(),
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
    /// Create a new graph with gradient computation nodes for automatic differentiation.
    ///
    /// This method transforms a forward-only graph into one that computes gradients
    /// during the forward pass. Gradient nodes are added for each operation that
    /// depends (directly or transitively) on parameters.
    ///
    /// # Arguments
    ///
    /// * `loss_node` - The scalar loss node to differentiate with respect to
    ///
    /// # Returns
    ///
    /// A `TensorGraph<f32, WithGrad>` containing both forward and gradient computation nodes.
    ///
    /// # Gradient Computation Strategy
    ///
    /// For each node in topological order from loss back to parameters:
    /// - Binary Add: d(a+b)/da = 1, d(a+b)/db = 1 → grad_a = grad_out, grad_b = grad_out
    /// - Binary Mul: d(a*b)/da = b, d(a*b)/db = a → grad_a = grad_out * b, grad_b = grad_out * a
    /// - Unary Exp: d(exp(x))/dx = exp(x) → grad_x = grad_out * exp(x)
    /// - MatMul: d(A@B)/dA = dC @ B^T, d(A@B)/dB = A^T @ dC
    ///
    /// # Example
    ///
    /// ```ignore
    /// let x = Parameter::new(vec![2.0], vec![1]);
    /// let y = x.clone() * x; // y = x^2
    /// let graph: TensorGraph<f32, NoGrad> = y.into();
    /// let grad_graph = graph.with_gradients(NodeIndex::from(1));
    /// // grad_graph now contains nodes to compute dy/dx = 2*x
    /// ```
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
        let loss_shape = self
            .shapes
            .get(&loss_node)
            .expect("loss node shape missing");
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
        });
        self.shapes.insert(seed_grad_node, loss_shape.clone());
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
                TensorGraphNode::Binary { op } => {
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

                            let neg_grad = self
                                .graph
                                .add_node(TensorGraphNode::Unary { op: UnaryOp::Neg });
                            self.graph.add_edge(grad_output, neg_grad, 0);
                            let grad_shape = self.shapes.get(&grad_output).unwrap().clone();
                            self.shapes.insert(neg_grad, grad_shape);
                            gradient_nodes.insert(neg_grad);

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
                            let grad_a = self
                                .graph
                                .add_node(TensorGraphNode::Binary { op: BinaryOp::Mul });
                            self.graph.add_edge(grad_output, grad_a, 0);
                            self.graph.add_edge(input_b, grad_a, 1);
                            let grad_shape = self.shapes.get(&grad_output).unwrap().clone();
                            self.shapes.insert(grad_a, grad_shape);
                            gradient_nodes.insert(grad_a);

                            let grad_b = self
                                .graph
                                .add_node(TensorGraphNode::Binary { op: BinaryOp::Mul });
                            self.graph.add_edge(grad_output, grad_b, 0);
                            self.graph.add_edge(input_a, grad_b, 1);
                            let grad_shape = self.shapes.get(&grad_output).unwrap().clone();
                            self.shapes.insert(grad_b, grad_shape);
                            gradient_nodes.insert(grad_b);

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
                            let grad_a = self
                                .graph
                                .add_node(TensorGraphNode::Binary { op: BinaryOp::Div });
                            self.graph.add_edge(grad_output, grad_a, 0);
                            self.graph.add_edge(input_b, grad_a, 1);
                            let grad_shape = self.shapes.get(&grad_output).unwrap().clone();
                            self.shapes.insert(grad_a, grad_shape);
                            gradient_nodes.insert(grad_a);

                            // grad_b = -grad_output * a / (b * b)
                            let b_sq = self
                                .graph
                                .add_node(TensorGraphNode::Binary { op: BinaryOp::Mul });
                            self.graph.add_edge(input_b, b_sq, 0);
                            self.graph.add_edge(input_b, b_sq, 1);
                            let b_shape = self.shapes.get(&input_b).unwrap().clone();
                            self.shapes.insert(b_sq, b_shape);
                            gradient_nodes.insert(b_sq);

                            let grad_times_a = self
                                .graph
                                .add_node(TensorGraphNode::Binary { op: BinaryOp::Mul });
                            self.graph.add_edge(grad_output, grad_times_a, 0);
                            self.graph.add_edge(input_a, grad_times_a, 1);
                            let grad_shape = self.shapes.get(&grad_output).unwrap().clone();
                            self.shapes.insert(grad_times_a, grad_shape);
                            gradient_nodes.insert(grad_times_a);

                            let grad_b_pos = self
                                .graph
                                .add_node(TensorGraphNode::Binary { op: BinaryOp::Div });
                            self.graph.add_edge(grad_times_a, grad_b_pos, 0);
                            self.graph.add_edge(b_sq, grad_b_pos, 1);
                            let grad_shape = self.shapes.get(&grad_output).unwrap().clone();
                            self.shapes.insert(grad_b_pos, grad_shape);
                            gradient_nodes.insert(grad_b_pos);

                            let grad_b = self
                                .graph
                                .add_node(TensorGraphNode::Unary { op: UnaryOp::Neg });
                            self.graph.add_edge(grad_b_pos, grad_b, 0);
                            let grad_shape = self.shapes.get(&grad_output).unwrap().clone();
                            self.shapes.insert(grad_b, grad_shape);
                            gradient_nodes.insert(grad_b);

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
                TensorGraphNode::Unary { op } => {
                    if inputs.len() != 1 {
                        panic!("Unary op should have 1 input");
                    }
                    let input_x = inputs[0];

                    match op {
                        UnaryOp::Exp => {
                            // d(exp(x))/dx = exp(x)
                            // grad_x = grad_output * exp(x) = grad_output * node_output
                            let grad_x = self
                                .graph
                                .add_node(TensorGraphNode::Binary { op: BinaryOp::Mul });
                            self.graph.add_edge(grad_output, grad_x, 0);
                            self.graph.add_edge(node_idx, grad_x, 1); // Use the exp output
                            let grad_shape = self.shapes.get(&grad_output).unwrap().clone();
                            self.shapes.insert(grad_x, grad_shape);
                            gradient_nodes.insert(grad_x);

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
                            let grad_x = self
                                .graph
                                .add_node(TensorGraphNode::Binary { op: BinaryOp::Div });
                            self.graph.add_edge(grad_output, grad_x, 0);
                            self.graph.add_edge(input_x, grad_x, 1);
                            let grad_shape = self.shapes.get(&grad_output).unwrap().clone();
                            self.shapes.insert(grad_x, grad_shape);
                            gradient_nodes.insert(grad_x);

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
                            let grad_x = self
                                .graph
                                .add_node(TensorGraphNode::Unary { op: UnaryOp::Neg });
                            self.graph.add_edge(grad_output, grad_x, 0);
                            let grad_shape = self.shapes.get(&grad_output).unwrap().clone();
                            self.shapes.insert(grad_x, grad_shape);
                            gradient_nodes.insert(grad_x);

                            self.accumulate_gradient(
                                &mut node_to_grad,
                                &mut gradient_nodes,
                                input_x,
                                grad_x,
                            );
                        }
                        UnaryOp::Relu => {
                            // d(relu(x))/dx = grad_output * (x > 0)
                            // We need to create: condition = (x > 0), then mask(grad_output, condition)
                            
                            // Create a constant zero tensor with same shape as input
                            let input_shape = self.shapes.get(&input_x).unwrap().clone();
                            let num_elements: usize = input_shape.iter().product();
                            let zero_const = self.graph.add_node(TensorGraphNode::Constant {
                                data: Arc::new(vec![0.0f32; num_elements]),
                            });
                            self.shapes.insert(zero_const, input_shape.clone());

                            // Create Gt node: condition = (x > 0)
                            let condition = self.graph.add_node(TensorGraphNode::Gt);
                            self.graph.add_edge(input_x, condition, 0);
                            self.graph.add_edge(zero_const, condition, 1);
                            self.shapes.insert(condition, input_shape.clone());
                            gradient_nodes.insert(condition);

                            // Create Mask node: grad_x = mask(grad_output, condition)
                            let grad_x = self.graph.add_node(TensorGraphNode::Mask);
                            self.graph.add_edge(grad_output, grad_x, 0);
                            self.graph.add_edge(condition, grad_x, 1);
                            let grad_shape = self.shapes.get(&grad_output).unwrap().clone();
                            self.shapes.insert(grad_x, grad_shape);
                            gradient_nodes.insert(grad_x);

                            self.accumulate_gradient(
                                &mut node_to_grad,
                                &mut gradient_nodes,
                                input_x,
                                grad_x,
                            );
                        }
                    }
                }
                TensorGraphNode::MatMul => {
                    // d(A@B)/dA = dC @ B^T, d(A@B)/dB = A^T @ dC
                    if inputs.len() != 2 {
                        panic!("MatMul should have 2 inputs");
                    }
                    let input_a = inputs[0];
                    let input_b = inputs[1];

                    // Create B^T
                    let b_t = self.graph.add_node(TensorGraphNode::Transpose);
                    self.graph.add_edge(input_b, b_t, 0);
                    // Transpose swaps last two dimensions: [m, n] -> [n, m]
                    let b_shape = self.shapes.get(&input_b).unwrap();
                    let mut b_t_shape = b_shape.clone();
                    if b_t_shape.len() >= 2 {
                        let len = b_t_shape.len();
                        b_t_shape.swap(len - 2, len - 1);
                    }
                    self.shapes.insert(b_t, b_t_shape.clone());
                    gradient_nodes.insert(b_t);

                    // grad_a = grad_output @ B^T
                    let grad_a = self.graph.add_node(TensorGraphNode::MatMul);
                    self.graph.add_edge(grad_output, grad_a, 0);
                    self.graph.add_edge(b_t, grad_a, 1);
                    let a_shape = self.shapes.get(&input_a).unwrap().clone();
                    self.shapes.insert(grad_a, a_shape);
                    gradient_nodes.insert(grad_a);

                    // Create A^T
                    let a_t = self.graph.add_node(TensorGraphNode::Transpose);
                    self.graph.add_edge(input_a, a_t, 0);
                    let mut a_t_shape = self.shapes.get(&input_a).unwrap().clone();
                    if a_t_shape.len() >= 2 {
                        let len = a_t_shape.len();
                        a_t_shape.swap(len - 2, len - 1);
                    }
                    self.shapes.insert(a_t, a_t_shape);
                    gradient_nodes.insert(a_t);

                    // grad_b = A^T @ grad_output
                    let grad_b = self.graph.add_node(TensorGraphNode::MatMul);
                    self.graph.add_edge(a_t, grad_b, 0);
                    self.graph.add_edge(grad_output, grad_b, 1);
                    let b_shape = self.shapes.get(&input_b).unwrap().clone();
                    self.shapes.insert(grad_b, b_shape);
                    gradient_nodes.insert(grad_b);

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
                TensorGraphNode::Transpose => {
                    // d(A^T)/dA = (grad_output)^T
                    // Transpose gradient is just transpose of incoming gradient
                    if inputs.len() != 1 {
                        panic!("Transpose should have 1 input");
                    }
                    let input_a = inputs[0];

                    let grad_a = self.graph.add_node(TensorGraphNode::Transpose);
                    self.graph.add_edge(grad_output, grad_a, 0);
                    let input_shape = self.shapes.get(&input_a).unwrap().clone();
                    self.shapes.insert(grad_a, input_shape);
                    gradient_nodes.insert(grad_a);

                    self.accumulate_gradient(
                        &mut node_to_grad,
                        &mut gradient_nodes,
                        input_a,
                        grad_a,
                    );
                }
                TensorGraphNode::ReduceAxis { op, axis } => {
                    // For ReduceAxisSum: gradient is broadcast back to original shape
                    // If Y = sum(X, axis), then dX = broadcast(dY, axis)
                    match op {
                        ReduceOp::Sum => {
                            if inputs.len() != 1 {
                                panic!("ReduceAxis should have 1 input");
                            }
                            let input_x = inputs[0];
                            let input_shape = self.shapes.get(&input_x).unwrap();

                            // Create broadcast node to expand gradient back to input shape
                            let grad_x = self
                                .graph
                                .add_node(TensorGraphNode::BroadcastAxis { axis: *axis });
                            self.graph.add_edge(grad_output, grad_x, 0);
                            self.shapes.insert(grad_x, input_shape.clone());
                            gradient_nodes.insert(grad_x);

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
                            let input_shape = self.shapes.get(&input_x).unwrap().clone();
                            let axis_size = input_shape[*axis];

                            // Broadcast gradient back to input shape
                            let grad_broadcast = self
                                .graph
                                .add_node(TensorGraphNode::BroadcastAxis { axis: *axis });
                            self.graph.add_edge(grad_output, grad_broadcast, 0);
                            self.shapes.insert(grad_broadcast, input_shape.clone());
                            gradient_nodes.insert(grad_broadcast);

                            // Divide by axis_size (multiply by 1/axis_size)
                            let scale = 1.0 / axis_size as f32;
                            let num_elements: usize = input_shape.iter().product();
                            let scale_const = self.graph.add_node(TensorGraphNode::Constant {
                                data: Arc::new(vec![scale; num_elements]),
                            });
                            self.shapes.insert(scale_const, input_shape.clone());

                            let grad_x = self
                                .graph
                                .add_node(TensorGraphNode::Binary { op: BinaryOp::Mul });
                            self.graph.add_edge(grad_broadcast, grad_x, 0);
                            self.graph.add_edge(scale_const, grad_x, 1);
                            self.shapes.insert(grad_x, input_shape.clone());
                            gradient_nodes.insert(grad_x);

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
                TensorGraphNode::BroadcastAxis { axis } => {
                    // For BroadcastAxis: gradient is reduced back to original shape
                    // If Y = broadcast(X, axis), then dX = reduce_sum(dY, axis)
                    if inputs.len() != 1 {
                        panic!("BroadcastAxis should have 1 input");
                    }
                    let input_x = inputs[0];
                    let input_shape = self.shapes.get(&input_x).unwrap();

                    // Create reduce node to collapse gradient back to input shape
                    let grad_x = self.graph.add_node(TensorGraphNode::ReduceAxis {
                        op: ReduceOp::Sum,
                        axis: *axis,
                    });
                    self.graph.add_edge(grad_output, grad_x, 0);
                    self.shapes.insert(grad_x, input_shape.clone());
                    gradient_nodes.insert(grad_x);

                    self.accumulate_gradient(
                        &mut node_to_grad,
                        &mut gradient_nodes,
                        input_x,
                        grad_x,
                    );
                }
                TensorGraphNode::Gt | TensorGraphNode::Mask => {
                    // Gt and Mask are only used in gradient computation itself
                    // They don't need gradients (they're non-differentiable operations)
                    // Skip gradient computation for these nodes
                }
            }
        }

        TensorGraph {
            graph: self.graph,
            shapes: self.shapes,
            gradients: WithGrad {
                data: GradientMetadata {
                    param_to_grad,
                    gradient_nodes,
                },
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
            let sum_grad = self
                .graph
                .add_node(TensorGraphNode::Binary { op: BinaryOp::Add });
            self.graph.add_edge(existing_grad, sum_grad, 0);
            self.graph.add_edge(new_grad, sum_grad, 1);
            let grad_shape = self.shapes.get(&existing_grad).unwrap().clone();
            self.shapes.insert(sum_grad, grad_shape);
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
    pub fn gradient_metadata(&self) -> &GradientMetadata {
        &self.gradients.data
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
    use crate::Constant;
    use crate::Parameter;
    use crate::ptx;
    use crate::tile;

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
            let shape = grad_graph.shapes.get(&idx);
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
            .filter(|&idx| matches!(grad_graph[idx], TensorGraphNode::Transpose))
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

        // Should have created gradient nodes including ReduceAxis backward (which creates BroadcastAxis)
        // and BroadcastAxis backward (which creates ReduceAxis)
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
                    TensorGraphNode::Binary { op: BinaryOp::Mul }
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
    #[ignore = "PTX lowering not yet implemented"]
    fn lowers_add_to_ptx() {
        let a = Constant::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![4]);
        let b = Constant::new(vec![10.0f32, 20.0, 30.0, 40.0], vec![4]);
        let c = a + b;
        println!("Expr: {c:?}");

        let tensor_graph: TensorGraph<f32> = c.into();
        let tile_graph: tile::TileGraph = tensor_graph.into();
        let ptx_graph: ptx::PtxGraph = tile_graph.into();
        let ptx_module: ptx::Module = ptx_graph.into();

        insta::assert_snapshot!(ptx_module);
    }
}
