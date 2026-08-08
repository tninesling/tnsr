use crate::graph;
use crate::graph::TensorGraphNode;
use petgraph::graph::NodeIndex;
use std::ops::{Add, Div, Mul, Sub};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

static PARAM_ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// Marker trait for supported data types in tensor operations.
pub trait DType: Clone {}

// Blanket implementation for all Clone types
impl<T: Clone> DType for T {}

/// Type alias for tensor shapes represented as dimension vectors.
pub type Shape = Vec<usize>;

/// Compute the output shape when broadcasting two tensors using NumPy-style rules.
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
            out.push(if ai == 1 { bi } else { ai });
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
#[derive(Clone, Debug, Copy, PartialEq, Eq)]
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

impl<D: DType> TensorExpr<D> {
    pub fn kind(&self) -> &ExprKind<D> {
        &self.0.kind
    }

    /// Returns references to child expressions for tree traversal.
    pub fn children(&self) -> Vec<&TensorExpr<D>> {
        match &self.0.kind {
            ExprKind::Unary { x, .. } => vec![x],
            ExprKind::Binary { a, b, .. } => vec![a, b],
            ExprKind::MatMul { a, b } => vec![a, b],
            ExprKind::Transpose { x } => vec![x],
            ExprKind::Reshape { x } => vec![x],
            ExprKind::Permute { x, .. } => vec![x],
            ExprKind::BroadcastAxis { x, .. } => vec![x],
            ExprKind::ReduceAxis { x, .. } => vec![x],
            ExprKind::Gt { a, b } => vec![a, b],
            ExprKind::Mask { values, condition } => vec![values, condition],
            ExprKind::Conv2d { input, weight, .. } => vec![input, weight],
            ExprKind::ConvTranspose2d {
                grad_output,
                weight,
                ..
            } => vec![grad_output, weight],
            ExprKind::Conv2dBackwardWeight {
                input, grad_output, ..
            } => vec![input, grad_output],
            ExprKind::MaxPool2d { x, .. } => vec![x],
            ExprKind::MaxPool2dBackward {
                input,
                output,
                grad_output,
                ..
            } => vec![input, output, grad_output],
            ExprKind::Flatten { x } => vec![x],
            ExprKind::NodeRef { .. }
            | ExprKind::Constant { .. }
            | ExprKind::Input { .. }
            | ExprKind::Parameter { .. } => vec![],
        }
    }
}

/// Generates small, shape-valid pointwise expressions suitable for fuzz testing.
impl<'a> arbitrary::Arbitrary<'a> for TensorExpr<f32> {
    fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        fn generate(
            u: &mut arbitrary::Unstructured<'_>,
            shape: &Shape,
            depth: usize,
        ) -> arbitrary::Result<TensorExpr<f32>> {
            if depth == 0 {
                let name = if u.arbitrary::<bool>()? { "x" } else { "y" };
                return Ok(TensorExpr::input(name, shape.clone()));
            }

            let next_depth = depth - 1;
            match u.int_in_range(0u8..=6)? {
                0 => Ok(TensorExpr::input("x", shape.clone())),
                1 => Ok(TensorExpr::input("y", shape.clone())),
                2 => Ok(-generate(u, shape, next_depth)?),
                3 => Ok(generate(u, shape, next_depth)?.relu()),
                4 => Ok(generate(u, shape, next_depth)? + generate(u, shape, next_depth)?),
                5 => Ok(generate(u, shape, next_depth)? - generate(u, shape, next_depth)?),
                _ => Ok(generate(u, shape, next_depth)? * generate(u, shape, next_depth)?),
            }
        }

        let rank = u.int_in_range(1usize..=3)?;
        let shape = (0..rank)
            .map(|_| u.int_in_range(1usize..=4))
            .collect::<arbitrary::Result<Shape>>()?;
        let depth = u.int_in_range(0usize..=5)?;
        generate(u, &shape, depth)
    }
}

#[derive(Clone, Debug)]
struct ExprNode<D: DType> {
    shape: Shape,
    kind: ExprKind<D>,
}

#[derive(Clone, Debug)]
pub enum ExprKind<D: DType> {
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
    Reshape {
        x: TensorExpr<D>,
    },
    Permute {
        x: TensorExpr<D>,
        axes: Vec<usize>,
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
    /// 2D convolution: input [N, C_in, H, W] * weight [C_out, C_in, kH, kW] -> [N, C_out, H_out, W_out]
    Conv2d {
        input: TensorExpr<D>,
        weight: TensorExpr<D>,
        stride: usize,
        padding: usize,
    },
    /// Transposed 2D convolution (backward w.r.t. input).
    ConvTranspose2d {
        grad_output: TensorExpr<D>,
        weight: TensorExpr<D>,
        stride: usize,
        padding: usize,
    },
    /// Convolution backward w.r.t. weight.
    Conv2dBackwardWeight {
        input: TensorExpr<D>,
        grad_output: TensorExpr<D>,
        stride: usize,
        padding: usize,
    },
    /// 2D max pooling: input [N, C, H, W] -> [N, C, H_out, W_out]
    MaxPool2d {
        x: TensorExpr<D>,
        kernel_size: usize,
        stride: usize,
    },
    /// Backward for max pooling: scatters grad to max elements.
    MaxPool2dBackward {
        input: TensorExpr<D>,
        output: TensorExpr<D>,
        grad_output: TensorExpr<D>,
        kernel_size: usize,
        stride: usize,
    },
    /// Flatten: reshape [N, C, H, W] -> [N, C*H*W]. Data is unchanged.
    Flatten {
        x: TensorExpr<D>,
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

    /// Create a reference to an existing node in a graph for gradient construction.
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
        assert!(
            lshape.len() >= 2 && rshape.len() >= 2,
            "MatMul requires tensors with rank >= 2, got {lshape:?} and {rshape:?}"
        );
        if lshape[lshape.len() - 1] != rshape[rshape.len() - 2] {
            panic!("MatMul inner dimensions must match: got {lshape:?} and {rshape:?}");
        }
        let mut shape = broadcast_output_shape(
            &lshape[..lshape.len() - 2].to_vec(),
            &rshape[..rshape.len() - 2].to_vec(),
        );
        shape.push(lshape[lshape.len() - 2]);
        shape.push(rshape[rshape.len() - 1]);
        Self(Arc::new(ExprNode {
            shape,
            kind: ExprKind::MatMul { a: self, b: rhs },
        }))
    }

    /// 2D convolution. Input [N, C_in, H, W], weight [C_out, C_in, kH, kW].
    /// Output [N, C_out, H_out, W_out] where H_out = (H + 2*padding - kH) / stride + 1.
    pub fn conv2d(self, weight: impl Into<TensorExpr<D>>, stride: usize, padding: usize) -> Self
    where
        D: 'static,
    {
        let weight = weight.into();
        let in_shape = self.shape().clone();
        let w_shape = weight.shape().clone();
        assert_eq!(
            in_shape.len(),
            4,
            "Conv2d input must be 4D [N, C_in, H, W], got {in_shape:?}"
        );
        assert_eq!(
            w_shape.len(),
            4,
            "Conv2d weight must be 4D [C_out, C_in, kH, kW], got {w_shape:?}"
        );
        assert_eq!(
            in_shape[1], w_shape[1],
            "Conv2d channel mismatch: input has {} channels, weight expects {}",
            in_shape[1], w_shape[1]
        );
        let h_out = (in_shape[2] + 2 * padding - w_shape[2]) / stride + 1;
        let w_out = (in_shape[3] + 2 * padding - w_shape[3]) / stride + 1;
        let shape = vec![in_shape[0], w_shape[0], h_out, w_out];
        Self(Arc::new(ExprNode {
            shape,
            kind: ExprKind::Conv2d {
                input: self,
                weight,
                stride,
                padding,
            },
        }))
    }

    pub(crate) fn conv_transpose_2d(
        grad_output: impl Into<TensorExpr<D>>,
        weight: impl Into<TensorExpr<D>>,
        input_shape: Shape,
        stride: usize,
        padding: usize,
    ) -> Self
    where
        D: 'static,
    {
        Self(Arc::new(ExprNode {
            shape: input_shape,
            kind: ExprKind::ConvTranspose2d {
                grad_output: grad_output.into(),
                weight: weight.into(),
                stride,
                padding,
            },
        }))
    }

    pub(crate) fn conv2d_backward_weight(
        input: impl Into<TensorExpr<D>>,
        grad_output: impl Into<TensorExpr<D>>,
        weight_shape: Shape,
        stride: usize,
        padding: usize,
    ) -> Self
    where
        D: 'static,
    {
        Self(Arc::new(ExprNode {
            shape: weight_shape,
            kind: ExprKind::Conv2dBackwardWeight {
                input: input.into(),
                grad_output: grad_output.into(),
                stride,
                padding,
            },
        }))
    }

    /// 2D max pooling. Input [N, C, H, W]. Output [N, C, H_out, W_out].
    pub fn max_pool2d(self, kernel_size: usize, stride: usize) -> Self
    where
        D: 'static,
    {
        let in_shape = self.shape().clone();
        assert_eq!(
            in_shape.len(),
            4,
            "MaxPool2d input must be 4D [N, C, H, W], got {in_shape:?}"
        );
        let h_out = (in_shape[2] - kernel_size) / stride + 1;
        let w_out = (in_shape[3] - kernel_size) / stride + 1;
        let shape = vec![in_shape[0], in_shape[1], h_out, w_out];
        Self(Arc::new(ExprNode {
            shape,
            kind: ExprKind::MaxPool2d {
                x: self,
                kernel_size,
                stride,
            },
        }))
    }

    pub(crate) fn max_pool2d_backward(
        input: impl Into<TensorExpr<D>>,
        output: impl Into<TensorExpr<D>>,
        grad_output: impl Into<TensorExpr<D>>,
        input_shape: Shape,
        kernel_size: usize,
        stride: usize,
    ) -> Self
    where
        D: 'static,
    {
        Self(Arc::new(ExprNode {
            shape: input_shape,
            kind: ExprKind::MaxPool2dBackward {
                input: input.into(),
                output: output.into(),
                grad_output: grad_output.into(),
                kernel_size,
                stride,
            },
        }))
    }

    /// Reshape without changing the underlying row-major data layout.
    pub fn reshape(self, shape: Shape) -> Self {
        assert_eq!(
            self.shape().iter().product::<usize>(),
            shape.iter().product::<usize>(),
            "reshape must preserve element count"
        );
        Self(Arc::new(ExprNode {
            shape,
            kind: ExprKind::Reshape { x: self },
        }))
    }

    /// Flatten a tensor to 2D [N, rest]. Data unchanged.
    pub fn flatten(self) -> Self
    where
        D: 'static,
    {
        let in_shape = self.shape().clone();
        assert!(!in_shape.is_empty(), "Flatten requires at least 1D tensor");
        let rest: usize = in_shape[1..].iter().product();
        let shape = vec![in_shape[0], rest];
        Self(Arc::new(ExprNode {
            shape,
            kind: ExprKind::Flatten { x: self },
        }))
    }

    pub fn broadcast(self, to: Shape) -> Self {
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

            result = result.reshape(current_shape.clone());
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

    pub fn broadcast_axis(self, axis: usize, target_size: usize) -> Self {
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

    /// Reorder dimensions according to `axes`.
    pub fn permute(self, axes: Vec<usize>) -> Self {
        let shape = self.shape().clone();
        assert_eq!(
            axes.len(),
            shape.len(),
            "Permutation rank mismatch: got {} axes for {}D tensor",
            axes.len(),
            shape.len()
        );
        let mut seen = vec![false; axes.len()];
        for &axis in &axes {
            assert!(axis < axes.len(), "Permutation axis {axis} out of bounds");
            assert!(!seen[axis], "Permutation contains duplicate axis {axis}");
            seen[axis] = true;
        }
        let out_shape = axes.iter().map(|&axis| shape[axis]).collect();
        Self(Arc::new(ExprNode {
            shape: out_shape,
            kind: ExprKind::Permute { x: self, axes },
        }))
    }

    /// Swap two dimensions.
    pub fn swap_axes(self, axis_a: usize, axis_b: usize) -> Self {
        let rank = self.shape().len();
        assert!(axis_a < rank, "swap axis {axis_a} out of bounds");
        assert!(axis_b < rank, "swap axis {axis_b} out of bounds");
        let mut axes: Vec<usize> = (0..rank).collect();
        axes.swap(axis_a, axis_b);
        self.permute(axes)
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
        let out_shape = broadcast_output_shape(self.shape(), rhs.shape());
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
        let out_shape = broadcast_output_shape(self.shape(), rhs.shape());
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
        let out_shape = broadcast_output_shape(self.shape(), rhs.shape());
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
        let out_shape = broadcast_output_shape(self.shape(), rhs.shape());
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
        let out_shape = broadcast_output_shape(self.shape(), other.shape());
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
        let out_shape = broadcast_output_shape(self.shape(), condition.shape());
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
    /// Optimize this expression using e-graph rewrites.
    pub fn optimize(&self) -> Self
    where
        D: Clone + Default + 'static,
    {
        crate::graph::rewrite::optimize_expr(self, crate::graph::rewrite::RewriteConfig::default())
    }

    /// Optimize this expression with custom configuration.
    pub fn optimize_with(&self, config: crate::graph::rewrite::RewriteConfig) -> Self
    where
        D: Clone + Default + 'static,
    {
        crate::graph::rewrite::optimize_expr(self, config)
    }

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
                ExprKind::Constant { data } => g.graph.add_node(TensorGraphNode::Constant {
                    data: data.clone(),
                    shape: expr.shape().clone(),
                }),
                ExprKind::Input { name } => {
                    // Reuse semantics like Input::lower_to_graph
                    for idx in g.graph.node_indices() {
                        if let TensorGraphNode::Input { name: n, shape } = &g[idx]
                            && n == name
                        {
                            assert_eq!(
                                shape,
                                expr.shape(),
                                "Input '{:?}' shape mismatch: {:?} vs {:?}",
                                name,
                                shape,
                                expr.shape()
                            );
                            return idx;
                        }
                    }
                    g.graph.add_node(TensorGraphNode::Input {
                        name,
                        shape: expr.shape().clone(),
                    })
                }
                ExprKind::Parameter { id, data } => {
                    for idx in g.graph.node_indices() {
                        if let TensorGraphNode::Parameter { id: pid, .. } = &g[idx]
                            && pid == id
                        {
                            let existing_shape = g.graph[idx].shape();
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
                    g.graph.add_node(TensorGraphNode::Parameter {
                        id: *id,
                        data: data.clone(),
                        shape: expr.shape().clone(),
                    })
                }
                ExprKind::Unary { op, x } => {
                    let x_idx = lower_rec(x, g);
                    let node_idx = g.graph.add_node(TensorGraphNode::Unary {
                        op: *op,
                        shape: expr.shape().clone(),
                    });
                    g.graph.add_edge(x_idx, node_idx, 0);
                    node_idx
                }
                ExprKind::Binary { op, a, b } => {
                    let a_idx = if a.shape() == expr.shape() {
                        lower_rec(a, g)
                    } else {
                        lower_rec(&a.clone().broadcast(expr.shape().clone()), g)
                    };
                    let b_idx = if b.shape() == expr.shape() {
                        lower_rec(b, g)
                    } else {
                        lower_rec(&b.clone().broadcast(expr.shape().clone()), g)
                    };
                    let node_idx = g.graph.add_node(TensorGraphNode::Binary {
                        op: op.clone(),
                        shape: expr.shape().clone(),
                    });
                    g.graph.add_edge(a_idx, node_idx, 0);
                    g.graph.add_edge(b_idx, node_idx, 1);
                    node_idx
                }
                ExprKind::MatMul { a, b } => {
                    let a_idx = lower_rec(a, g);
                    let b_idx = lower_rec(b, g);
                    let node_idx = g.graph.add_node(TensorGraphNode::MatMul {
                        shape: expr.shape().clone(),
                    });
                    g.graph.add_edge(a_idx, node_idx, 0);
                    g.graph.add_edge(b_idx, node_idx, 1);
                    node_idx
                }
                ExprKind::BroadcastAxis { x, axis } => {
                    let x_idx = lower_rec(x, g);
                    let node_idx = g.graph.add_node(TensorGraphNode::BroadcastAxis {
                        axis: *axis,
                        shape: expr.shape().clone(),
                    });
                    g.graph.add_edge(x_idx, node_idx, 0);
                    node_idx
                }
                ExprKind::ReduceAxis { op, x, axis } => {
                    let x_idx = lower_rec(x, g);
                    let node_idx = g.graph.add_node(TensorGraphNode::ReduceAxis {
                        op: op.clone(),
                        axis: *axis,
                        shape: expr.shape().clone(),
                    });
                    g.graph.add_edge(x_idx, node_idx, 0);
                    node_idx
                }
                ExprKind::Transpose { x } => {
                    let x_idx = lower_rec(x, g);
                    let node_idx = g.graph.add_node(TensorGraphNode::Transpose {
                        shape: expr.shape().clone(),
                    });
                    g.graph.add_edge(x_idx, node_idx, 0);
                    node_idx
                }
                ExprKind::Reshape { x } => {
                    let x_idx = lower_rec(x, g);
                    let node_idx = g.graph.add_node(TensorGraphNode::Reshape {
                        shape: expr.shape().clone(),
                    });
                    g.graph.add_edge(x_idx, node_idx, 0);
                    node_idx
                }
                ExprKind::Permute { x, axes } => {
                    let x_idx = lower_rec(x, g);
                    let node_idx = g.graph.add_node(TensorGraphNode::Permute {
                        axes: axes.clone(),
                        shape: expr.shape().clone(),
                    });
                    g.graph.add_edge(x_idx, node_idx, 0);
                    node_idx
                }
                ExprKind::Gt { a, b } => {
                    let a_idx = if a.shape() == expr.shape() {
                        lower_rec(a, g)
                    } else {
                        lower_rec(&a.clone().broadcast(expr.shape().clone()), g)
                    };
                    let b_idx = if b.shape() == expr.shape() {
                        lower_rec(b, g)
                    } else {
                        lower_rec(&b.clone().broadcast(expr.shape().clone()), g)
                    };
                    let node_idx = g.graph.add_node(TensorGraphNode::Gt {
                        shape: expr.shape().clone(),
                    });
                    g.graph.add_edge(a_idx, node_idx, 0);
                    g.graph.add_edge(b_idx, node_idx, 1);
                    node_idx
                }
                ExprKind::Mask { values, condition } => {
                    let values_idx = if values.shape() == expr.shape() {
                        lower_rec(values, g)
                    } else {
                        lower_rec(&values.clone().broadcast(expr.shape().clone()), g)
                    };
                    let condition_idx = if condition.shape() == expr.shape() {
                        lower_rec(condition, g)
                    } else {
                        lower_rec(&condition.clone().broadcast(expr.shape().clone()), g)
                    };
                    let node_idx = g.graph.add_node(TensorGraphNode::Mask {
                        shape: expr.shape().clone(),
                    });
                    g.graph.add_edge(values_idx, node_idx, 0);
                    g.graph.add_edge(condition_idx, node_idx, 1);
                    node_idx
                }
                ExprKind::Conv2d {
                    input,
                    weight,
                    stride,
                    padding,
                } => {
                    let input_idx = lower_rec(input, g);
                    let weight_idx = lower_rec(weight, g);
                    let node_idx = g.graph.add_node(TensorGraphNode::Conv2d {
                        stride: *stride,
                        padding: *padding,
                        shape: expr.shape().clone(),
                    });
                    g.graph.add_edge(input_idx, node_idx, 0);
                    g.graph.add_edge(weight_idx, node_idx, 1);
                    node_idx
                }
                ExprKind::ConvTranspose2d {
                    grad_output,
                    weight,
                    stride,
                    padding,
                } => {
                    let grad_idx = lower_rec(grad_output, g);
                    let weight_idx = lower_rec(weight, g);
                    let node_idx = g.graph.add_node(TensorGraphNode::ConvTranspose2d {
                        stride: *stride,
                        padding: *padding,
                        shape: expr.shape().clone(),
                    });
                    g.graph.add_edge(grad_idx, node_idx, 0);
                    g.graph.add_edge(weight_idx, node_idx, 1);
                    node_idx
                }
                ExprKind::Conv2dBackwardWeight {
                    input,
                    grad_output,
                    stride,
                    padding,
                } => {
                    let input_idx = lower_rec(input, g);
                    let grad_idx = lower_rec(grad_output, g);
                    let node_idx = g.graph.add_node(TensorGraphNode::Conv2dBackwardWeight {
                        stride: *stride,
                        padding: *padding,
                        shape: expr.shape().clone(),
                    });
                    g.graph.add_edge(input_idx, node_idx, 0);
                    g.graph.add_edge(grad_idx, node_idx, 1);
                    node_idx
                }
                ExprKind::MaxPool2d {
                    x,
                    kernel_size,
                    stride,
                } => {
                    let x_idx = lower_rec(x, g);
                    let node_idx = g.graph.add_node(TensorGraphNode::MaxPool2d {
                        kernel_size: *kernel_size,
                        stride: *stride,
                        shape: expr.shape().clone(),
                    });
                    g.graph.add_edge(x_idx, node_idx, 0);
                    node_idx
                }
                ExprKind::MaxPool2dBackward {
                    input,
                    output,
                    grad_output,
                    kernel_size,
                    stride,
                } => {
                    let input_idx = lower_rec(input, g);
                    let output_idx = lower_rec(output, g);
                    let grad_idx = lower_rec(grad_output, g);
                    let node_idx = g.graph.add_node(TensorGraphNode::MaxPool2dBackward {
                        kernel_size: *kernel_size,
                        stride: *stride,
                        shape: expr.shape().clone(),
                    });
                    g.graph.add_edge(input_idx, node_idx, 0);
                    g.graph.add_edge(output_idx, node_idx, 1);
                    g.graph.add_edge(grad_idx, node_idx, 2);
                    node_idx
                }
                ExprKind::Flatten { x } => {
                    let x_idx = lower_rec(x, g);
                    let node_idx = g.graph.add_node(TensorGraphNode::Flatten {
                        shape: expr.shape().clone(),
                    });
                    g.graph.add_edge(x_idx, node_idx, 0);
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
