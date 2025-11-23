//! Lowered computation graph representation with gradient support.
//!
//! This module provides the lowered DAG representation of tensor computations
//! that executors use to perform actual computation. It includes automatic
//! differentiation support through a typestate pattern.
//!
//! # Key Types
//!
//! - [`TensorGraph`] - The main computation graph structure
//! - [`TensorGraphNode`] - Individual operations in the graph (constants, parameters, ops)
//! - [`NoGrad`] - Typestate marker for forward-only graphs
//! - [`WithGrad`] - Typestate marker for graphs with gradient computation
//!
//! # Gradient Computation
//!
//! Convert a forward-only graph to one with gradients using [`TensorGraph::with_gradients`]:
//!
//! ```ignore
//! let forward_graph: TensorGraph<f32, NoGrad> = expr.into();
//! let grad_graph = forward_graph.with_gradients(loss_node);
//! // grad_graph now contains both forward and gradient computation nodes
//! ```
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
pub use petgraph::graph::NodeIndex;
use petgraph::stable_graph::StableGraph;
use petgraph::visit::EdgeRef;

use crate::BinaryOp;
use crate::ReduceOp;
use crate::TensorExpr;
use crate::UnaryOp;

/// Typestate marker for graphs without gradients.
#[derive(Clone)]
pub struct NoGrad;

/// Typestate marker for graphs with gradient computation nodes.
///
/// Tracks gradient nodes and parameter mappings for automatic differentiation.
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
    },
    Input {
        name: &'static str,
    },
    Parameter {
        id: usize,
        data: Arc<Mutex<Vec<D>>>,
    },
    Unary {
        op: UnaryOp,
    },
    /// Fused sequence of unary operations (for operator fusion optimization).
    #[cfg(feature = "fusion")]
    FusedUnary {
        ops: Vec<UnaryOp>,
    },
    Binary {
        op: BinaryOp,
    },
    MatMul,
    Transpose,
    BroadcastAxis {
        axis: usize,
    },
    ReduceAxis {
        op: ReduceOp,
        axis: usize,
    },
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
            #[cfg(feature = "fusion")]
            TensorGraphNode::FusedUnary { .. } => "FusedUnary",
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

/// Directed acyclic graph (DAG) representation of tensor computations.
///
/// `TensorGraph` is the lowered representation of [`TensorExpr`] that's used by executors
/// to perform actual computation. It uses a typestate pattern with generic parameter `G`
/// to track whether gradient computation nodes have been added.
///
/// # Type States
///
/// - `TensorGraph<D, NoGrad>` - Forward-only graph without gradient computation
/// - `TensorGraph<D, WithGrad>` - Graph with gradient nodes for automatic differentiation
///
/// # Structure
///
/// Each node in the graph represents an operation (constant, input, parameter, unary, binary,
/// matmul, etc.), and edges represent data flow between operations. The graph maintains:
/// - Node operations and their relationships
/// - Shape information for each node
/// - Gradient metadata (when `G = WithGrad`)
///
/// # Usage
///
/// Create a graph by converting from a [`TensorExpr`]:
///
/// ```rust
/// use tensor::{TensorExpr, graph::TensorGraph};
///
/// let x = TensorExpr::<f32>::input("x", vec![2, 3]);
/// let y = x.relu();
/// let graph: TensorGraph<f32> = y.into();
/// ```
///
/// For training with gradients, use [`with_gradients`](TensorGraph::with_gradients):
///
/// ```ignore
/// use tensor::{Parameter, graph::TensorGraph};
///
/// let w = Parameter::new(vec![1.0, 2.0], vec![2, 1]);
/// let x = TensorExpr::<f32>::input("x", vec![2, 1]);
/// let y = w * x;  // Forward computation
/// let loss = y.reduce_sum(0);  // Scalar loss
///
/// let graph: TensorGraph<f32> = loss.into();
/// let grad_graph = graph.with_gradients(loss_node);
/// // grad_graph now contains both forward and gradient computation nodes
/// ```
///
/// # Execution
///
/// Graphs are executed by implementors of the `Executor` trait from the `runtime` crate:
/// - `SimpleExecutor` for CPU execution
/// - `CudaExecutor` for GPU execution (with `cuda` feature)
///
/// See the `runtime` crate documentation for execution details.
#[derive(Clone)]
pub struct TensorGraph<D, G = NoGrad> {
    pub graph: StableGraph<TensorGraphNode<D>, usize>,
    pub shapes: HashMap<NodeIndex, crate::Shape>,
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
            graph: StableGraph::new(),
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

    /// Apply operator fusion optimization to the graph.
    ///
    /// This pass identifies chains of consecutive unary operations and fuses them
    /// into single `FusedUnary` nodes, reducing kernel launch overhead and memory traffic.
    ///
    /// # Returns
    ///
    /// The number of fusion transformations applied.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let x = TensorExpr::<f32>::input("x", vec![4]);
    /// let y = x.exp().log().relu();
    /// let mut graph: TensorGraph<f32> = y.into();
    /// let num_fused = graph.apply_fusion();
    /// // Graph now contains a single FusedUnary(Exp, Log, Relu) node
    /// assert_eq!(num_fused, 1);
    /// ```
    #[cfg(feature = "fusion")]
    pub fn apply_fusion(&mut self) -> usize
    where
        D: Clone,
    {
        use crate::fusion::FusionAnalyzer;

        let analyzer = FusionAnalyzer::new(self);
        let chains = analyzer.find_fusible_chains();
        let num_chains = chains.len();

        let mut nodes_to_remove = Vec::new();

        // For each chain, replace it with a FusedUnary node
        for chain in chains {
            // Create the fused node
            let fused_node = self.graph.add_node(TensorGraphNode::FusedUnary {
                ops: chain.ops.clone(),
            });

            // Copy the shape from the end node of the chain
            if let Some(shape) = self.shapes.get(&chain.end_node) {
                self.shapes.insert(fused_node, shape.clone());
            }

            // Add edge from start_node to fused_node
            self.graph.add_edge(chain.start_node, fused_node, 0);

            // Redirect all edges from end_node to fused_node
            let outgoing_edges: Vec<_> = self
                .graph
                .edges_directed(chain.end_node, petgraph::Direction::Outgoing)
                .map(|e| (e.target(), *e.weight()))
                .collect();

            for (target, weight) in outgoing_edges {
                self.graph.add_edge(fused_node, target, weight);
            }

            // Mark nodes for removal (but don't remove yet to avoid index shifting)
            nodes_to_remove.extend(chain.chain_nodes.iter().copied());
        }

        // Remove all marked nodes after all fused nodes have been created
        for node in nodes_to_remove {
            self.graph.remove_node(node);
            self.shapes.remove(&node);
        }

        num_chains
    }
}

impl TensorGraph<f32, NoGrad> {
    /// Helper method to lower a TensorExpr into the graph during gradient construction.
    ///
    /// This method is used internally by `with_gradients` to convert fluent operator
    /// expressions into graph nodes.
    fn lower_gradient_expr(
        &mut self,
        expr: &crate::TensorExpr<f32>,
        gradient_nodes: &mut HashSet<NodeIndex>,
    ) -> NodeIndex {
        let node_idx = expr.lower_to_graph(self);

        // Mark all newly created nodes as gradient nodes
        // We need to traverse the expression tree and mark all nodes
        fn mark_gradient_nodes(expr: &crate::TensorExpr<f32>, visited: &mut HashSet<NodeIndex>) {
            match &expr.0.kind {
                crate::ExprKind::NodeRef { idx } => {
                    // Don't mark NodeRef nodes as they reference existing nodes
                    visited.insert(*idx);
                }
                crate::ExprKind::Constant { .. } => {
                    // Constants created during gradient construction are gradient nodes
                }
                crate::ExprKind::Input { .. } | crate::ExprKind::Parameter { .. } => {
                    // These already exist in the graph
                }
                crate::ExprKind::Unary { x, .. } => {
                    mark_gradient_nodes(x, visited);
                }
                crate::ExprKind::Binary { a, b, .. } => {
                    mark_gradient_nodes(a, visited);
                    mark_gradient_nodes(b, visited);
                }
                crate::ExprKind::MatMul { a, b } => {
                    mark_gradient_nodes(a, visited);
                    mark_gradient_nodes(b, visited);
                }
                crate::ExprKind::Transpose { x } => {
                    mark_gradient_nodes(x, visited);
                }
                crate::ExprKind::BroadcastAxis { x, .. } => {
                    mark_gradient_nodes(x, visited);
                }
                crate::ExprKind::ReduceAxis { x, .. } => {
                    mark_gradient_nodes(x, visited);
                }
                crate::ExprKind::Gt { a, b } => {
                    mark_gradient_nodes(a, visited);
                    mark_gradient_nodes(b, visited);
                }
                crate::ExprKind::Mask { values, condition } => {
                    mark_gradient_nodes(values, visited);
                    mark_gradient_nodes(condition, visited);
                }
            }
        }

        let mut visited = HashSet::new();
        mark_gradient_nodes(expr, &mut visited);
        gradient_nodes.insert(node_idx);

        node_idx
    }

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

                            // Use fluent syntax for negation
                            let grad_out_shape = self.shapes.get(&grad_output).unwrap().clone();
                            let grad_out_expr =
                                crate::TensorExpr::node_ref(grad_output, grad_out_shape);
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
                            let grad_out_shape = self.shapes.get(&grad_output).unwrap().clone();
                            let input_b_shape = self.shapes.get(&input_b).unwrap().clone();
                            let input_a_shape = self.shapes.get(&input_a).unwrap().clone();

                            let grad_out_expr =
                                crate::TensorExpr::node_ref(grad_output, grad_out_shape.clone());
                            let input_b_expr = crate::TensorExpr::node_ref(input_b, input_b_shape);
                            let input_a_expr = crate::TensorExpr::node_ref(input_a, input_a_shape);

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

                            let grad_out_shape = self.shapes.get(&grad_output).unwrap().clone();
                            let input_a_shape = self.shapes.get(&input_a).unwrap().clone();
                            let input_b_shape = self.shapes.get(&input_b).unwrap().clone();

                            let grad_out_expr =
                                crate::TensorExpr::node_ref(grad_output, grad_out_shape);
                            let input_a_expr = crate::TensorExpr::node_ref(input_a, input_a_shape);
                            let input_b_expr =
                                crate::TensorExpr::node_ref(input_b, input_b_shape.clone());

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
                TensorGraphNode::Unary { op } => {
                    if inputs.len() != 1 {
                        panic!("Unary op should have 1 input");
                    }
                    let input_x = inputs[0];

                    match op {
                        UnaryOp::Exp => {
                            // d(exp(x))/dx = exp(x)
                            // grad_x = grad_output * exp(x) = grad_output * node_output

                            let grad_out_shape = self.shapes.get(&grad_output).unwrap().clone();
                            let node_out_shape = self.shapes.get(&node_idx).unwrap().clone();

                            let grad_out_expr =
                                crate::TensorExpr::node_ref(grad_output, grad_out_shape);
                            let node_out_expr =
                                crate::TensorExpr::node_ref(node_idx, node_out_shape);

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

                            let grad_out_shape = self.shapes.get(&grad_output).unwrap().clone();
                            let input_x_shape = self.shapes.get(&input_x).unwrap().clone();

                            let grad_out_expr =
                                crate::TensorExpr::node_ref(grad_output, grad_out_shape);
                            let input_x_expr = crate::TensorExpr::node_ref(input_x, input_x_shape);

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

                            let grad_out_shape = self.shapes.get(&grad_output).unwrap().clone();
                            let grad_out_expr =
                                crate::TensorExpr::node_ref(grad_output, grad_out_shape);
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

                            let grad_out_shape = self.shapes.get(&grad_output).unwrap().clone();
                            let input_x_shape = self.shapes.get(&input_x).unwrap().clone();
                            let num_elements: usize = input_x_shape.iter().product();

                            let grad_out_expr =
                                crate::TensorExpr::node_ref(grad_output, grad_out_shape);
                            let input_x_expr =
                                crate::TensorExpr::node_ref(input_x, input_x_shape.clone());
                            let zero_expr = crate::TensorExpr::constant(
                                vec![0.0f32; num_elements],
                                input_x_shape,
                            );

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
                TensorGraphNode::MatMul => {
                    // d(A@B)/dA = dC @ B^T, d(A@B)/dB = A^T @ dC
                    if inputs.len() != 2 {
                        panic!("MatMul should have 2 inputs");
                    }
                    let input_a = inputs[0];
                    let input_b = inputs[1];

                    let grad_out_shape = self.shapes.get(&grad_output).unwrap().clone();
                    let input_a_shape = self.shapes.get(&input_a).unwrap().clone();
                    let input_b_shape = self.shapes.get(&input_b).unwrap().clone();

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
                TensorGraphNode::Transpose => {
                    // d(A^T)/dA = (grad_output)^T
                    // Transpose gradient is just transpose of incoming gradient
                    if inputs.len() != 1 {
                        panic!("Transpose should have 1 input");
                    }
                    let input_a = inputs[0];
                    let grad_out_shape = self.shapes.get(&grad_output).unwrap().clone();

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
                TensorGraphNode::ReduceAxis { op, axis } => {
                    // For ReduceAxisSum: gradient is broadcast back to original shape
                    // If Y = sum(X, axis), then dX = broadcast(dY, axis)
                    match op {
                        ReduceOp::Sum => {
                            if inputs.len() != 1 {
                                panic!("ReduceAxis should have 1 input");
                            }
                            let input_x = inputs[0];
                            let input_shape = self.shapes.get(&input_x).unwrap().clone();
                            let grad_out_shape = self.shapes.get(&grad_output).unwrap().clone();
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
                            let input_shape = self.shapes.get(&input_x).unwrap().clone();
                            let grad_out_shape = self.shapes.get(&grad_output).unwrap().clone();
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
                TensorGraphNode::BroadcastAxis { axis } => {
                    // For BroadcastAxis: gradient is reduced back to original shape
                    // If Y = broadcast(X, axis), then dX = reduce_sum(dY, axis)
                    if inputs.len() != 1 {
                        panic!("BroadcastAxis should have 1 input");
                    }
                    let input_x = inputs[0];
                    let grad_out_shape = self.shapes.get(&grad_output).unwrap().clone();

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
                TensorGraphNode::Gt | TensorGraphNode::Mask => {
                    // Gt and Mask are only used in gradient computation itself
                    // They don't need gradients (they're non-differentiable operations)
                    // Skip gradient computation for these nodes
                }
                #[cfg(feature = "fusion")]
                TensorGraphNode::FusedUnary { .. } => {
                    // TODO: Implement gradient computation for fused unary operations
                    // For now, FusedUnary should only be used in inference graphs without gradients
                    panic!("Gradient computation for FusedUnary not yet implemented");
                }
            }
        }

        TensorGraph {
            graph: self.graph,
            shapes: self.shapes,
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
    pub fn gradient_metadata(&self) -> &WithGrad {
        &self.gradients
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
    #[cfg(feature = "fusion")]
    fn test_apply_fusion_simple_chain() {
        // Create graph: x -> exp -> log
        let x = TensorExpr::<f32>::input("x", vec![4]);
        let y = x.exp().log();
        let mut graph: TensorGraph<f32> = y.into();

        let initial_node_count = graph.len();
        let num_fused = graph.apply_fusion();

        assert_eq!(num_fused, 1, "Should fuse one chain");
        // Should have 2 nodes: input and fused
        assert_eq!(graph.len(), 2, "Should have input + fused node");
        assert!(
            graph.len() < initial_node_count,
            "Should have fewer nodes after fusion"
        );

        // Verify the fused node exists and has the right ops
        let fused_node = graph
            .graph
            .node_indices()
            .find(|&idx| matches!(graph[idx], TensorGraphNode::FusedUnary { .. }));

        assert!(fused_node.is_some(), "Should have a FusedUnary node");

        if let TensorGraphNode::FusedUnary { ops } = &graph[fused_node.unwrap()] {
            assert_eq!(ops.len(), 2, "Should have 2 ops");
            assert_eq!(ops[0], UnaryOp::Exp);
            assert_eq!(ops[1], UnaryOp::Log);
        }
    }

    #[test]
    #[cfg(feature = "fusion")]
    fn test_apply_fusion_longer_chain() {
        // Create graph: x -> neg -> exp -> log -> relu
        let x = TensorExpr::<f32>::input("x", vec![4]);
        let y = (-x).exp().log().relu();
        let mut graph: TensorGraph<f32> = y.into();

        let num_fused = graph.apply_fusion();

        assert_eq!(num_fused, 1, "Should fuse one chain");

        // Verify the fused node has all 4 ops
        let fused_node = graph
            .graph
            .node_indices()
            .find(|&idx| matches!(graph[idx], TensorGraphNode::FusedUnary { .. }));

        assert!(fused_node.is_some(), "Should have a FusedUnary node");

        if let TensorGraphNode::FusedUnary { ops } = &graph[fused_node.unwrap()] {
            assert_eq!(ops.len(), 4, "Should have 4 ops");
            assert_eq!(ops[0], UnaryOp::Neg);
            assert_eq!(ops[1], UnaryOp::Exp);
            assert_eq!(ops[2], UnaryOp::Log);
            assert_eq!(ops[3], UnaryOp::Relu);
        }
    }

    #[test]
    #[cfg(feature = "fusion")]
    fn test_apply_fusion_no_single_op() {
        // Create graph: x -> exp (single op, should not fuse)
        let x = TensorExpr::<f32>::input("x", vec![4]);
        let y = x.exp();
        let mut graph: TensorGraph<f32> = y.into();

        let num_fused = graph.apply_fusion();

        assert_eq!(num_fused, 0, "Should not fuse single op");

        // Verify no FusedUnary node exists
        let has_fused = graph
            .graph
            .node_indices()
            .any(|idx| matches!(graph[idx], TensorGraphNode::FusedUnary { .. }));

        assert!(!has_fused, "Should not have FusedUnary node");
    }

    #[test]
    #[cfg(feature = "fusion")]
    fn test_apply_fusion_with_branching() {
        // Create graph with branching: x -> exp -> (log, relu) -> add
        let x = TensorExpr::<f32>::input("x", vec![4]);
        let exp_x = x.exp();
        let log_exp = exp_x.clone().log();
        let relu_exp = exp_x.relu();
        let y = log_exp + relu_exp;
        let mut graph: TensorGraph<f32> = y.into();

        let num_fused = graph.apply_fusion();

        // Should not fuse because exp has multiple consumers
        assert_eq!(num_fused, 0, "Should not fuse with branching");
    }

    #[test]
    #[cfg(feature = "fusion")]
    fn test_apply_fusion_multiple_independent_chains() {
        // Create two independent chains: x -> exp -> log, y -> neg -> relu
        let x = TensorExpr::<f32>::input("x", vec![4]);
        let y = TensorExpr::<f32>::input("y", vec![4]);
        let a = x.exp().log();
        let b = (-y).relu();
        let result = a + b;
        let mut graph: TensorGraph<f32> = result.into();

        let num_fused = graph.apply_fusion();

        assert_eq!(num_fused, 2, "Should fuse two independent chains");

        // Verify two FusedUnary nodes exist
        let fused_count = graph
            .graph
            .node_indices()
            .filter(|&idx| matches!(graph[idx], TensorGraphNode::FusedUnary { .. }))
            .count();

        assert_eq!(fused_count, 2, "Should have 2 FusedUnary nodes");
    }
}
