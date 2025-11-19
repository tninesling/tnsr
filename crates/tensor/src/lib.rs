//! Tensor expression library providing lazy evaluation and automatic differentiation.
//!
//! This crate implements a **lazy evaluation** system for tensor computations. Graph construction
//! is infallible by design—errors are deferred until execution time when graphs are lowered
//! and executed.
//!
//! # Architecture
//!
//! The crate is organized into three main components:
//!
//! - **Expression Layer** (`TensorExpr`, `Input`, `Parameter`, `Constant`): High-level API for
//!   building computation graphs. Operations like `+`, `*`, `matmul()`, and `broadcast()` construct
//!   expression trees without performing any computation.
//!
//! - **Graph Layer** (`graph` module): Intermediate representation as a directed acyclic graph (DAG)
//!   of operations. Expression trees are lowered to this representation for optimization and execution.
//!
//! - **Compilation Layer** (`ptx` module): PTX (CUDA assembly) code generation for GPU execution.
//!
//! # Design Principles
//!
//! - **Lazy Evaluation**: Expressions are built as immutable trees; no computation happens until
//!   graphs are lowered and executed by a runtime executor.
//!
//! - **Deferred Validation**: Graph construction never fails. Shape mismatches, invalid operations,
//!   and resource allocation errors are detected during lowering or execution.
//!
//! - **Explicit Broadcasting**: Users must call `.broadcast()` to adjust tensor shapes for operations
//!   requiring matching dimensions (when `implicit_broadcast` feature is disabled).
//!
//! # Example
//!
//! ```rust
//! use tensor::{TensorExpr, Parameter};
//!
//! // Build expression graph (infallible)
//! let x = TensorExpr::<f32>::input("x", vec![64, 10]);
//! let w = Parameter::new(vec![0.1; 100], vec![10, 10]);
//! let y = x.matmul(w).relu();
//!
//! // Lowering and execution may fail (handled by runtime)
//! // let result = executor.forward(&y, inputs)?;
//! ```

pub mod graph;
pub mod ptx;
pub mod tile;

use std::ops::Add;
use std::ops::Div;
use std::ops::Mul;
use std::ops::Sub;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use graph::TensorGraphNode;
use petgraph::graph::NodeIndex;

static PARAM_ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// Marker trait for supported data types in tensor operations.
///
/// Currently implemented for `f32`, `f64`, and `i32`.
pub trait DType: Clone {}
impl DType for f32 {}
impl DType for f64 {}
impl DType for i32 {}

/// Type alias for tensor shapes represented as dimension vectors.
///
/// For example, `vec![2, 3, 4]` represents a 3D tensor with shape 2×3×4.
pub type Shape = Vec<usize>;

/// Compute the output shape when broadcasting two tensors.
///
/// Follows NumPy-style broadcasting rules: dimensions are compared from right to left,
/// and are compatible if they are equal or one of them is 1.
///
/// # Arguments
///
/// * `a` - Shape of the first tensor
/// * `b` - Shape of the second tensor
///
/// # Returns
///
/// The broadcasted output shape
///
/// # Panics
///
/// Panics if shapes are incompatible for broadcasting
///
/// # Example
///
/// ```
/// use tensor::broadcast_output_shape;
///
/// let shape_a = vec![3, 1, 5];
/// let shape_b = vec![1, 4, 5];
/// let result = broadcast_output_shape(&shape_a, &shape_b);
/// assert_eq!(result, vec![3, 4, 5]);
/// ```
pub fn broadcast_output_shape(a: &Shape, b: &Shape) -> Shape {
    let max_len = a.len().max(b.len());
    let mut out = Vec::with_capacity(max_len);
    for i in 0..max_len {
        let ai = if i < max_len - a.len() {
            1
        } else {
            a[i - (max_len - a.len())]
        };
        let bi = if i < max_len - b.len() {
            1
        } else {
            b[i - (max_len - b.len())]
        };
        if ai == bi || ai == 1 || bi == 1 {
            out.push(ai.max(bi));
        } else {
            panic!(
                "Cannot broadcast shapes {:?} and {:?}: dim mismatch {} vs {} at axis {} (from right)",
                a,
                b,
                ai,
                bi,
                max_len - 1 - i
            );
        }
    }
    out
}

/// A constant tensor with fixed data known at graph construction time.
///
/// Constants are embedded directly into the computation graph and their values
/// cannot change during training. Useful for non-trainable values like bias initializations.
///
/// # Example
///
/// ```
/// use tensor::Constant;
///
/// let bias = Constant::new(vec![0.1, 0.2, 0.3], vec![3]);
/// ```
#[derive(Clone)]
pub struct Constant<D: DType> {
    data: Arc<Vec<D>>,
    shape: Shape,
}

impl<D: DType> Constant<D> {
    pub fn new(data: Vec<D>, shape: Shape) -> Self {
        Self {
            data: Arc::new(data),
            shape,
        }
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }
}

impl<D: DType, R: Into<TensorExpr<D>>> Add<R> for Constant<D> {
    type Output = TensorExpr<D>;
    fn add(self, rhs: R) -> Self::Output {
        let lhs_expr: TensorExpr<D> = self.into();
        lhs_expr + rhs
    }
}

/// An input tensor whose data is provided at execution time.
///
/// Inputs represent external data fed into the computation graph (e.g., training batches,
/// inference inputs). Values are supplied via the `inputs` parameter to the Executor.
///
/// # Example
///
/// ```
/// use tensor::{TensorExpr, Input};
///
/// let x = TensorExpr::<f32>::input("batch", vec![32, 784]);
/// ```
#[derive(Clone, Debug)]
pub struct Input<D: DType> {
    name: &'static str,
    shape: Shape,
    _marker: std::marker::PhantomData<D>,
}

impl<D: DType> Input<D> {
    pub fn new(name: &'static str, shape: Shape) -> Self {
        Self {
            name,
            shape,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }
}

/// Element-wise unary operations on tensors.
#[derive(Clone, Debug)]
pub enum UnaryOp {
    /// Negation: `-x`
    Neg,
    /// Exponential: `e^x`
    Exp,
    /// Natural logarithm: `ln(x)`
    Log,
    /// Rectified Linear Unit: `max(0, x)`
    Relu,
}

/// Element-wise binary operations on tensors.
#[derive(Clone, Debug)]
pub enum BinaryOp {
    /// Addition: `a + b`
    Add,
    /// Subtraction: `a - b`
    Sub,
    /// Multiplication: `a * b`
    Mul,
    /// Division: `a / b`
    Div,
}

/// Reduction operations along a tensor axis.
#[derive(Clone, Debug)]
pub enum ReduceOp {
    /// Sum all elements along an axis
    Sum,
    /// Maximum element along an axis
    Max,
    /// Mean (average) of elements along an axis
    Mean,
}

/// A trainable parameter with mutable data updated during optimization.
///
/// Parameters are updated by optimizers based on computed gradients. Each parameter
/// has a unique ID used to track gradients across the computation graph.
///
/// # Example
///
/// ```
/// use tensor::Parameter;
///
/// let weights = Parameter::new(vec![0.1; 100], vec![10, 10]);
/// let id = weights.id(); // Unique parameter ID
/// ```
#[derive(Clone)]
pub struct Parameter<D: DType> {
    id: usize,
    data: Arc<Mutex<Vec<D>>>,
    shape: Shape,
}

impl<D: DType> Parameter<D> {
    pub fn new(data: Vec<D>, shape: Shape) -> Self {
        let id = PARAM_ID_COUNTER.fetch_add(1, Ordering::Relaxed);
        Self {
            id,
            data: Arc::new(Mutex::new(data)),
            shape,
        }
    }
    pub fn new_with_id(id: usize, data: Vec<D>, shape: Shape) -> Self {
        Self {
            id,
            data: Arc::new(Mutex::new(data)),
            shape,
        }
    }

    pub fn id(&self) -> usize {
        self.id
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn exp(self) -> TensorExpr<D>
    where
        D: 'static,
    {
        let expr: TensorExpr<D> = self.into();
        expr.exp()
    }
}

impl<D: DType, R: Into<TensorExpr<D>>> Add<R> for Parameter<D> {
    type Output = TensorExpr<D>;
    fn add(self, rhs: R) -> Self::Output {
        let lhs_expr: TensorExpr<D> = self.into();
        lhs_expr + rhs
    }
}

impl<D: DType, R: Into<TensorExpr<D>>> Sub<R> for Parameter<D> {
    type Output = TensorExpr<D>;
    fn sub(self, rhs: R) -> Self::Output {
        let lhs_expr: TensorExpr<D> = self.into();
        lhs_expr - rhs
    }
}

impl<D: DType, R: Into<TensorExpr<D>>> Mul<R> for Parameter<D> {
    type Output = TensorExpr<D>;
    fn mul(self, rhs: R) -> Self::Output {
        let lhs_expr: TensorExpr<D> = self.into();
        lhs_expr * rhs
    }
}

impl<D: DType, R: Into<TensorExpr<D>>> Div<R> for Parameter<D> {
    type Output = TensorExpr<D>;
    fn div(self, rhs: R) -> Self::Output {
        let lhs_expr: TensorExpr<D> = self.into();
        lhs_expr / rhs
    }
}

#[derive(Clone, Debug)]
pub struct TensorExpr<D: DType>(Arc<ExprNode<D>>);

#[derive(Clone, Debug)]
struct ExprNode<D: DType> {
    shape: Shape,
    kind: ExprKind<D>,
}

#[derive(Clone, Debug)]
enum ExprKind<D: DType> {
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
    /// Reference to an existing node in a graph (used for gradients)
    NodeRef {
        idx: NodeIndex,
    },
    Unary {
        op: UnaryOp,
        x: TensorExpr<D>,
    },
    Binary {
        op: BinaryOp,
        a: TensorExpr<D>,
        b: TensorExpr<D>,
    },
    MatMul {
        a: TensorExpr<D>,
        b: TensorExpr<D>,
    },
    BroadcastAxis {
        x: TensorExpr<D>,
        axis: usize,
    },
    ReduceAxis {
        op: ReduceOp,
        x: TensorExpr<D>,
        axis: usize,
    },
    Transpose {
        x: TensorExpr<D>,
    },
    /// Greater than comparison
    Gt {
        a: TensorExpr<D>,
        b: TensorExpr<D>,
    },
    /// Mask operation
    Mask {
        values: TensorExpr<D>,
        condition: TensorExpr<D>,
    },
}

impl<D: DType> TensorExpr<D> {
    pub fn shape(&self) -> &Shape {
        &self.0.shape
    }

    pub fn input(name: &'static str, shape: Shape) -> Self {
        Self(Arc::new(ExprNode {
            shape,
            kind: ExprKind::Input { name },
        }))
    }

    pub fn constant(data: Vec<D>, shape: Shape) -> Self {
        Self(Arc::new(ExprNode {
            shape,
            kind: ExprKind::Constant {
                data: Arc::new(data),
            },
        }))
    }

    pub fn parameter(data: Vec<D>, shape: Shape) -> Self {
        let id = PARAM_ID_COUNTER.fetch_add(1, Ordering::Relaxed);
        Self(Arc::new(ExprNode {
            shape,
            kind: ExprKind::Parameter {
                id,
                data: Arc::new(Mutex::new(data)),
            },
        }))
    }

    pub fn parameter_with_id(id: usize, data: Vec<D>, shape: Shape) -> Self {
        Self(Arc::new(ExprNode {
            shape,
            kind: ExprKind::Parameter {
                id,
                data: Arc::new(Mutex::new(data)),
            },
        }))
    }

    /// Create a reference to an existing node in a graph.
    ///
    /// This is used during gradient construction to build tensor expressions
    /// that reference already-lowered graph nodes, enabling fluent operator
    /// syntax for creating gradient computation nodes.
    ///
    /// # Arguments
    ///
    /// * `idx` - The NodeIndex of the existing graph node to reference
    /// * `shape` - The shape of the tensor at this node
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Instead of manually adding nodes and edges:
    /// let grad_a = graph.add_node(TensorGraphNode::Binary { op: BinaryOp::Mul });
    /// graph.add_edge(grad_output, grad_a, 0);
    /// graph.add_edge(input_b, grad_a, 1);
    ///
    /// // Use fluent syntax:
    /// let grad_output_expr = TensorExpr::node_ref(grad_output, shape);
    /// let input_b_expr = TensorExpr::node_ref(input_b, shape);
    /// let grad_a_expr = grad_output_expr * input_b_expr;
    /// ```
    pub fn node_ref(idx: NodeIndex, shape: Shape) -> Self {
        Self(Arc::new(ExprNode {
            shape,
            kind: ExprKind::NodeRef { idx },
        }))
    }

    pub fn exp(self) -> Self
    where
        D: 'static,
    {
        Self(Arc::new(ExprNode {
            shape: self.shape().clone(),
            kind: ExprKind::Unary {
                op: UnaryOp::Exp,
                x: self,
            },
        }))
    }

    pub fn log(self) -> Self
    where
        D: 'static,
    {
        Self(Arc::new(ExprNode {
            shape: self.shape().clone(),
            kind: ExprKind::Unary {
                op: UnaryOp::Log,
                x: self,
            },
        }))
    }

    pub fn relu(self) -> Self
    where
        D: 'static,
    {
        Self(Arc::new(ExprNode {
            shape: self.shape().clone(),
            kind: ExprKind::Unary {
                op: UnaryOp::Relu,
                x: self,
            },
        }))
    }

    pub fn matmul(self, rhs: impl Into<TensorExpr<D>>) -> Self
    where
        D: 'static,
    {
        let rhs = rhs.into();
        let lshape = self.shape().clone();
        let rshape = rhs.shape().clone();
        if lshape.len() != 2 || rshape.len() != 2 {
            panic!("MatMul only supports 2D tensors for now");
        }
        if lshape[1] != rshape[0] {
            panic!("MatMul inner dimensions must match: got {lshape:?} and {rshape:?}");
        }
        let shape = vec![lshape[0], rshape[1]];
        Self(Arc::new(ExprNode {
            shape,
            kind: ExprKind::MatMul { a: self, b: rhs },
        }))
    }

    pub fn broadcast(self, to: Shape) -> Self
    where
        D: 'static + Clone,
    {
        // Validate broadcasting compatibility
        let in_shape = self.shape().to_vec(); // Clone the shape to avoid borrowing issues
        if in_shape.len() > to.len() {
            panic!("Cannot broadcast to smaller rank");
        }
        for (i, &dim) in in_shape.iter().rev().enumerate() {
            let target_dim = to[to.len() - 1 - i];
            if dim != target_dim && dim != 1 {
                panic!("Cannot broadcast dim {dim} -> {target_dim}");
            }
        }

        // Decompose generic broadcast into per-axis broadcasts
        // This allows us to use optimized CUDA kernels instead of reduce_like fallback
        let mut result = self;
        let mut current_shape = in_shape.clone();

        // First, handle rank difference by prepending 1s (implicit dimensions)
        if current_shape.len() < to.len() {
            // Pad with 1s on the left to match target rank
            let rank_diff = to.len() - current_shape.len();
            let mut padded_shape = vec![1; rank_diff];
            padded_shape.extend_from_slice(&current_shape);
            current_shape = padded_shape;

            // Update the expression's shape to match the new rank
            result = Self(Arc::new(ExprNode {
                shape: current_shape.clone(),
                kind: result.0.kind.clone(),
            }));
        }

        // Broadcast each axis that differs
        for axis in 0..to.len() {
            if current_shape[axis] != to[axis] {
                if current_shape[axis] != 1 {
                    panic!(
                        "Cannot broadcast axis {}: {} -> {}",
                        axis, current_shape[axis], to[axis]
                    );
                }
                result = result.broadcast_axis(axis, to[axis]);
                current_shape[axis] = to[axis];
            }
        }

        result
    }

    pub fn broadcast_axis(self, axis: usize, target_size: usize) -> Self
    where
        D: 'static,
    {
        let rank = self.shape().len();
        assert!(axis < rank, "broadcast axis out of bounds");
        assert_eq!(
            self.shape()[axis],
            1,
            "can only broadcast axis of size 1, got {}",
            self.shape()[axis]
        );
        let mut out_shape = self.shape().clone();
        out_shape[axis] = target_size;
        Self(Arc::new(ExprNode {
            shape: out_shape,
            kind: ExprKind::BroadcastAxis { x: self, axis },
        }))
    }

    pub fn reduce_axis_sum(self, axis: usize) -> Self {
        let rank = self.shape().len();
        assert!(axis < rank, "reduce axis out of bounds");
        let mut out_shape = self.shape().clone();
        out_shape[axis] = 1;
        Self(Arc::new(ExprNode {
            shape: out_shape,
            kind: ExprKind::ReduceAxis {
                op: ReduceOp::Sum,
                x: self,
                axis,
            },
        }))
    }

    pub fn reduce_axis_mean(self, axis: usize) -> Self {
        let rank = self.shape().len();
        assert!(axis < rank, "reduce axis out of bounds");
        let mut out_shape = self.shape().clone();
        out_shape[axis] = 1;
        Self(Arc::new(ExprNode {
            shape: out_shape,
            kind: ExprKind::ReduceAxis {
                op: ReduceOp::Mean,
                x: self,
                axis,
            },
        }))
    }

    pub fn reduce_axis_max(self, axis: usize) -> Self {
        let rank = self.shape().len();
        assert!(axis < rank, "reduce axis out of bounds");
        let mut out_shape = self.shape().clone();
        out_shape[axis] = 1;
        Self(Arc::new(ExprNode {
            shape: out_shape,
            kind: ExprKind::ReduceAxis {
                op: ReduceOp::Max,
                x: self,
                axis,
            },
        }))
    }

    pub fn reduce_sum(self, axis: usize) -> Self {
        let rank = self.shape().len();
        assert!(axis < rank, "reduce axis out of bounds");
        let mut out_shape = self.shape().clone();
        out_shape[axis] = 1;
        Self(Arc::new(ExprNode {
            shape: out_shape,
            kind: ExprKind::ReduceAxis {
                op: ReduceOp::Sum,
                x: self,
                axis,
            },
        }))
    }

    pub fn reduce_mean(self, axis: usize) -> Self {
        let rank = self.shape().len();
        assert!(axis < rank, "reduce axis out of bounds");
        let mut out_shape = self.shape().clone();
        out_shape[axis] = 1;
        Self(Arc::new(ExprNode {
            shape: out_shape,
            kind: ExprKind::ReduceAxis {
                op: ReduceOp::Mean,
                x: self,
                axis,
            },
        }))
    }

    pub fn reduce_max(self, axis: usize) -> Self {
        let rank = self.shape().len();
        assert!(axis < rank, "reduce axis out of bounds");
        let mut out_shape = self.shape().clone();
        out_shape[axis] = 1;
        Self(Arc::new(ExprNode {
            shape: out_shape,
            kind: ExprKind::ReduceAxis {
                op: ReduceOp::Max,
                x: self,
                axis,
            },
        }))
    }

    pub fn mean_all(self) -> Self {
        let mut expr = self;
        for axis in (0..expr.shape().len()).rev() {
            expr = expr.reduce_mean(axis);
        }
        expr
    }

    pub fn transpose(self) -> Self
    where
        D: 'static,
    {
        let shape = self.shape().clone();
        assert!(
            shape.len() >= 2,
            "Transpose requires at least 2D tensor, got {}D",
            shape.len()
        );
        let mut out_shape = shape.clone();
        let len = out_shape.len();
        out_shape.swap(len - 2, len - 1);
        Self(Arc::new(ExprNode {
            shape: out_shape,
            kind: ExprKind::Transpose { x: self },
        }))
    }
}

impl<D: DType> std::ops::Neg for TensorExpr<D> {
    type Output = TensorExpr<D>;
    fn neg(self) -> Self::Output {
        TensorExpr(Arc::new(ExprNode {
            shape: self.shape().clone(),
            kind: ExprKind::Unary {
                op: UnaryOp::Neg,
                x: self,
            },
        }))
    }
}

impl<D: DType, R: Into<TensorExpr<D>>> Add<R> for TensorExpr<D> {
    type Output = TensorExpr<D>;
    fn add(self, rhs: R) -> Self::Output {
        let rhs = rhs.into();
        #[cfg(feature = "implicit_broadcast")]
        let out_shape = broadcast_output_shape(self.shape(), rhs.shape());
        #[cfg(not(feature = "implicit_broadcast"))]
        let out_shape = self.shape().clone();
        TensorExpr(Arc::new(ExprNode {
            shape: out_shape,
            kind: ExprKind::Binary {
                op: BinaryOp::Add,
                a: self,
                b: rhs,
            },
        }))
    }
}

impl<D: DType, R: Into<TensorExpr<D>>> Sub<R> for TensorExpr<D> {
    type Output = TensorExpr<D>;
    fn sub(self, rhs: R) -> Self::Output {
        let rhs = rhs.into();
        #[cfg(feature = "implicit_broadcast")]
        let out_shape = broadcast_output_shape(self.shape(), rhs.shape());
        #[cfg(not(feature = "implicit_broadcast"))]
        let out_shape = self.shape().clone();
        TensorExpr(Arc::new(ExprNode {
            shape: out_shape,
            kind: ExprKind::Binary {
                op: BinaryOp::Sub,
                a: self,
                b: rhs,
            },
        }))
    }
}

impl<D: DType, R: Into<TensorExpr<D>>> Mul<R> for TensorExpr<D> {
    type Output = TensorExpr<D>;
    fn mul(self, rhs: R) -> Self::Output {
        let rhs = rhs.into();
        #[cfg(feature = "implicit_broadcast")]
        let out_shape = broadcast_output_shape(self.shape(), rhs.shape());
        #[cfg(not(feature = "implicit_broadcast"))]
        let out_shape = self.shape().clone();
        TensorExpr(Arc::new(ExprNode {
            shape: out_shape,
            kind: ExprKind::Binary {
                op: BinaryOp::Mul,
                a: self,
                b: rhs,
            },
        }))
    }
}

impl<D: DType, R: Into<TensorExpr<D>>> Div<R> for TensorExpr<D> {
    type Output = TensorExpr<D>;
    fn div(self, rhs: R) -> Self::Output {
        let rhs = rhs.into();
        #[cfg(feature = "implicit_broadcast")]
        let out_shape = broadcast_output_shape(self.shape(), rhs.shape());
        #[cfg(not(feature = "implicit_broadcast"))]
        let out_shape = self.shape().clone();
        TensorExpr(Arc::new(ExprNode {
            shape: out_shape,
            kind: ExprKind::Binary {
                op: BinaryOp::Div,
                a: self,
                b: rhs,
            },
        }))
    }
}

impl<D: DType> TensorExpr<D> {
    /// Greater than comparison: returns 1.0 where `self > other`, 0.0 elsewhere.
    pub fn gt<R: Into<TensorExpr<D>>>(self, other: R) -> Self
    where
        D: 'static,
    {
        let other = other.into();
        #[cfg(feature = "implicit_broadcast")]
        let out_shape = broadcast_output_shape(self.shape(), other.shape());
        #[cfg(not(feature = "implicit_broadcast"))]
        let out_shape = self.shape().clone();
        TensorExpr(Arc::new(ExprNode {
            shape: out_shape,
            kind: ExprKind::Gt { a: self, b: other },
        }))
    }

    /// Mask operation: returns `self` where `condition != 0`, else 0.
    pub fn mask<R: Into<TensorExpr<D>>>(self, condition: R) -> Self
    where
        D: 'static,
    {
        let condition = condition.into();
        #[cfg(feature = "implicit_broadcast")]
        let out_shape = broadcast_output_shape(self.shape(), condition.shape());
        #[cfg(not(feature = "implicit_broadcast"))]
        let out_shape = self.shape().clone();
        TensorExpr(Arc::new(ExprNode {
            shape: out_shape,
            kind: ExprKind::Mask {
                values: self,
                condition,
            },
        }))
    }
}

impl<D: DType> TensorExpr<D> {
    pub fn lower_to_graph<G>(&self, graph: &mut graph::TensorGraph<D, G>) -> NodeIndex {
        fn lower_rec<D: DType, G>(
            expr: &TensorExpr<D>,
            g: &mut graph::TensorGraph<D, G>,
        ) -> NodeIndex {
            match &expr.0.kind {
                ExprKind::NodeRef { idx } => {
                    // NodeRef just returns the existing node index
                    // No new node is created - this allows referencing nodes during gradient construction
                    *idx
                }
                ExprKind::Constant { data } => {
                    let idx = g
                        .graph
                        .add_node(TensorGraphNode::Constant { data: data.clone() });
                    g.shapes.insert(idx, expr.shape().clone());
                    idx
                }
                ExprKind::Input { name } => {
                    // Reuse semantics like Input::lower_to_graph
                    for idx in g.graph.node_indices() {
                        if let TensorGraphNode::Input { name: n } = &g[idx]
                            && n == name
                        {
                            let existing_shape =
                                g.shapes.get(&idx).expect("shape missing for input");
                            assert_eq!(
                                existing_shape,
                                expr.shape(),
                                "Input '{:?}' shape mismatch: {:?} vs {:?}",
                                name,
                                existing_shape,
                                expr.shape()
                            );
                            return idx;
                        }
                    }
                    let idx = g.graph.add_node(TensorGraphNode::Input { name });
                    g.shapes.insert(idx, expr.shape().clone());
                    idx
                }
                ExprKind::Parameter { id, data } => {
                    for idx in g.graph.node_indices() {
                        if let TensorGraphNode::Parameter { id: pid, .. } = &g[idx]
                            && pid == id
                        {
                            let existing_shape =
                                g.shapes.get(&idx).expect("shape missing for parameter");
                            assert_eq!(
                                existing_shape,
                                expr.shape(),
                                "Parameter id {} shape mismatch: {:?} vs {:?}",
                                id,
                                existing_shape,
                                expr.shape()
                            );
                            return idx;
                        }
                    }
                    let idx = g.graph.add_node(TensorGraphNode::Parameter {
                        id: *id,
                        data: data.clone(),
                    });
                    g.shapes.insert(idx, expr.shape().clone());
                    idx
                }
                ExprKind::Unary { op, x } => {
                    let x_idx = lower_rec(x, g);
                    let node_idx = g.graph.add_node(op.clone().into());
                    g.shapes.insert(node_idx, expr.shape().clone());
                    g.graph.add_edge(x_idx, node_idx, 0);
                    node_idx
                }
                ExprKind::Binary { op, a, b } => {
                    let a_idx = lower_rec(a, g);
                    let b_idx = lower_rec(b, g);
                    // Implicit broadcasting is not supported with the new axis-based broadcast system.
                    // Users must explicitly broadcast operands to matching shapes before binary ops.
                    #[cfg(feature = "implicit_broadcast")]
                    {
                        if a.shape() != expr.shape() || b.shape() != expr.shape() {
                            panic!(
                                "Binary op requires matching shapes. Use .broadcast() explicitly.\n\
                                 Left shape: {:?}, Right shape: {:?}, Output shape: {:?}",
                                a.shape(),
                                b.shape(),
                                expr.shape()
                            );
                        }
                    }
                    let node_idx = g.graph.add_node(op.clone().into());
                    g.shapes.insert(node_idx, expr.shape().clone());
                    g.graph.add_edge(a_idx, node_idx, 0);
                    g.graph.add_edge(b_idx, node_idx, 1);
                    node_idx
                }
                ExprKind::MatMul { a, b } => {
                    let a_idx = lower_rec(a, g);
                    let b_idx = lower_rec(b, g);
                    let node_idx = g.graph.add_node(TensorGraphNode::MatMul);
                    g.shapes.insert(node_idx, expr.shape().clone());
                    g.graph.add_edge(a_idx, node_idx, 0);
                    g.graph.add_edge(b_idx, node_idx, 1);
                    node_idx
                }
                ExprKind::BroadcastAxis { x, axis } => {
                    let x_idx = lower_rec(x, g);
                    let node_idx = g
                        .graph
                        .add_node(TensorGraphNode::BroadcastAxis { axis: *axis });
                    g.shapes.insert(node_idx, expr.shape().clone());
                    g.graph.add_edge(x_idx, node_idx, 0);
                    node_idx
                }
                ExprKind::ReduceAxis { op, x, axis } => {
                    let x_idx = lower_rec(x, g);
                    let node_idx = g.graph.add_node(TensorGraphNode::ReduceAxis {
                        op: op.clone(),
                        axis: *axis,
                    });
                    g.shapes.insert(node_idx, expr.shape().clone());
                    g.graph.add_edge(x_idx, node_idx, 0);
                    node_idx
                }
                ExprKind::Transpose { x } => {
                    let x_idx = lower_rec(x, g);
                    let node_idx = g.graph.add_node(TensorGraphNode::Transpose);
                    g.shapes.insert(node_idx, expr.shape().clone());
                    g.graph.add_edge(x_idx, node_idx, 0);
                    node_idx
                }
                ExprKind::Gt { a, b } => {
                    let a_idx = lower_rec(a, g);
                    let b_idx = lower_rec(b, g);
                    let node_idx = g.graph.add_node(TensorGraphNode::Gt);
                    g.shapes.insert(node_idx, expr.shape().clone());
                    g.graph.add_edge(a_idx, node_idx, 0);
                    g.graph.add_edge(b_idx, node_idx, 1);
                    node_idx
                }
                ExprKind::Mask { values, condition } => {
                    let values_idx = lower_rec(values, g);
                    let condition_idx = lower_rec(condition, g);
                    let node_idx = g.graph.add_node(TensorGraphNode::Mask);
                    g.shapes.insert(node_idx, expr.shape().clone());
                    g.graph.add_edge(values_idx, node_idx, 0);
                    g.graph.add_edge(condition_idx, node_idx, 1);
                    node_idx
                }
            }
        }
        lower_rec(self, graph)
    }
}

impl<D: DType + Clone> From<D> for TensorExpr<D> {
    fn from(v: D) -> Self {
        TensorExpr::constant(vec![v], vec![])
    }
}

impl<D: DType> From<Constant<D>> for TensorExpr<D> {
    fn from(c: Constant<D>) -> Self {
        TensorExpr(Arc::new(ExprNode {
            shape: c.shape().clone(),
            kind: ExprKind::Constant { data: c.data },
        }))
    }
}

impl<D: DType> From<Input<D>> for TensorExpr<D> {
    fn from(i: Input<D>) -> Self {
        TensorExpr(Arc::new(ExprNode {
            shape: i.shape().clone(),
            kind: ExprKind::Input { name: i.name },
        }))
    }
}

impl<D: DType> From<Parameter<D>> for TensorExpr<D> {
    fn from(p: Parameter<D>) -> Self {
        TensorExpr(Arc::new(ExprNode {
            shape: p.shape().clone(),
            kind: ExprKind::Parameter {
                id: p.id,
                data: p.data,
            },
        }))
    }
}
