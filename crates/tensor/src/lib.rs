pub mod graph;

use std::ops::Add;
use std::ops::Div;
use std::ops::Mul;
use std::ops::Sub;
use std::sync::Arc;

use graph::TensorGraphNode;
use petgraph::graph::NodeIndex;


pub trait DType {}
impl DType for f32 {}
impl DType for f64 {}
impl DType for i32 {}

pub type Shape = Vec<usize>;

pub fn broadcast_output_shape(a: &Shape, b: &Shape) -> Shape {
    let max_len = a.len().max(b.len());
    let mut out = Vec::with_capacity(max_len);
    for i in 0..max_len {
        let ai = if i < max_len - a.len() { 1 } else { a[i - (max_len - a.len())] };
        let bi = if i < max_len - b.len() { 1 } else { b[i - (max_len - b.len())] };
        if ai == bi || ai == 1 || bi == 1 {
            out.push(ai.max(bi));
        } else {
            panic!("Cannot broadcast shapes {:?} and {:?}: dim mismatch {} vs {} at axis {} (from right)", a, b, ai, bi, max_len - 1 - i);
        }
    }
    out
}


pub trait TensorOps<D: DType>: Tensor<D> + Sized {
    fn matmul(self, rhs: impl Tensor<D> + 'static) -> MatMulNode<D>;
    fn broadcast(self, shape: Shape) -> BroadcastNode<D>;
}

impl<D, T> TensorOps<D> for T
where
    D: DType + 'static,
    T: Tensor<D> + Sized + 'static,
{
    fn matmul(self, rhs: impl Tensor<D> + 'static) -> MatMulNode<D> {
        MatMulNode::new(Box::new(self), Box::new(rhs))
    }

    fn broadcast(self, shape: Shape) -> BroadcastNode<D> {
        BroadcastNode::new(Box::new(self), shape)
    }
}

pub trait Tensor<D: DType> {
    fn shape(&self) -> &Shape;

    fn lower_to_graph(&self, graph: &mut graph::TensorGraph<D>) -> NodeIndex;
}

impl<D: DType> Tensor<D> for Box<dyn Tensor<D>> {
    fn shape(&self) -> &Shape {
        (**self).shape()
    }

    fn lower_to_graph(&self, graph: &mut graph::TensorGraph<D>) -> NodeIndex {
        (**self).lower_to_graph(graph)
    }
}

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
}

impl<D: DType> Tensor<D> for Constant<D> {
    fn shape(&self) -> &Shape {
        &self.shape
    }

    fn lower_to_graph(&self, graph: &mut graph::TensorGraph<D>) -> NodeIndex {
        let idx = graph.graph.add_node(TensorGraphNode::Constant {
            data: self.data.clone(),
        });
        graph.shapes.insert(idx, self.shape.clone());
        idx
    }
}

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
}

impl<D: DType> Tensor<D> for Input<D> {
    fn shape(&self) -> &Shape {
        &self.shape
    }

    fn lower_to_graph(&self, graph: &mut graph::TensorGraph<D>) -> NodeIndex {
        let idx = graph
            .graph
            .add_node(TensorGraphNode::Input { name: self.name });
        graph.shapes.insert(idx, self.shape.clone());
        idx
    }
}

#[derive(Clone, Debug)]
pub enum UnaryOp {
    Neg,
    Exp,
    Log,
    Relu,
}

pub struct UnaryOpNode<D: DType> {
    op: UnaryOp,
    input: Box<dyn Tensor<D>>,
    shape: Shape,
}

impl<D: DType> Tensor<D> for UnaryOpNode<D> {
    fn shape(&self) -> &Shape {
        &self.shape
    }

    fn lower_to_graph(&self, graph: &mut graph::TensorGraph<D>) -> NodeIndex {
        let input_idx = self.input.lower_to_graph(graph);
        let node_idx = graph.graph.add_node(self.op.clone().into());
        graph.graph.add_edge(input_idx, node_idx, 0);
        node_idx
    }
}

#[derive(Clone, Debug)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

pub struct BinaryOpNode<D: DType> {
    op: BinaryOp,
    lhs: Box<dyn Tensor<D>>,
    rhs: Box<dyn Tensor<D>>,
    shape: Shape,
}

impl<D: DType> Tensor<D> for BinaryOpNode<D> {
    fn shape(&self) -> &Shape {
        &self.shape
    }

    fn lower_to_graph(&self, graph: &mut graph::TensorGraph<D>) -> NodeIndex {
        let lhs_idx = self.lhs.lower_to_graph(graph);
        let rhs_idx = self.rhs.lower_to_graph(graph);
        let node_idx = graph.graph.add_node(self.op.clone().into());
        graph.graph.add_edge(lhs_idx, node_idx, 0);
        graph.graph.add_edge(rhs_idx, node_idx, 1);
        node_idx
    }
}

#[derive(Clone, Debug)]
pub enum ReduceOp {
    Sum,
    Max,
    Mean,
}

pub struct ReduceOpNode<D: DType> {
    op: ReduceOp,
    input: Box<dyn Tensor<D>>,
    axis: usize,
    shape: Shape,
}

impl<D: DType> Tensor<D> for ReduceOpNode<D> {
    fn shape(&self) -> &Shape {
        &self.shape
    }

    fn lower_to_graph(&self, graph: &mut graph::TensorGraph<D>) -> NodeIndex {
        let input_idx = self.input.lower_to_graph(graph);
        let node_idx = graph.graph.add_node(TensorGraphNode::Reduce {
            op: self.op.clone().into(),
            axis: self.axis,
        });
        graph.graph.add_edge(input_idx, node_idx, 0);
        node_idx
    }
}

macro_rules! impl_un_ops_for_type {
    ($Type:ident) => {
        impl<D> std::ops::Neg for $Type<D>
        where
            D: DType + 'static,
        {
            type Output = UnaryOpNode<D>;
            fn neg(self) -> Self::Output {
                let shape = self.shape().clone();
                UnaryOpNode {
                    op: UnaryOp::Neg,
                    input: Box::new(self),
                    shape,
                }
            }
        }

        impl<D> $Type<D>
        where
            D: DType + 'static,
        {
            pub fn exp(self) -> UnaryOpNode<D> {
                let shape = self.shape().clone();
                UnaryOpNode {
                    op: UnaryOp::Exp,
                    input: Box::new(self),
                    shape,
                }
            }

            pub fn log(self) -> UnaryOpNode<D> {
                let shape = self.shape().clone();
                UnaryOpNode {
                    op: UnaryOp::Log,
                    input: Box::new(self),
                    shape,
                }
            }

            pub fn relu(self) -> UnaryOpNode<D> {
                let shape = self.shape().clone();
                UnaryOpNode {
                    op: UnaryOp::Relu,
                    input: Box::new(self),
                    shape,
                }
            }
        }
    };
}

impl_un_ops_for_type!(Constant);
impl_un_ops_for_type!(Input);
impl_un_ops_for_type!(UnaryOpNode);
impl_un_ops_for_type!(BinaryOpNode);
impl_un_ops_for_type!(ReduceOpNode);

macro_rules! impl_bin_ops_for_type {
    ($Type:ident) => {
        impl<D, T> Add<T> for $Type<D>
        where
            D: DType + 'static,
            T: Tensor<D> + 'static,
        {
            type Output = BinaryOpNode<D>;
            fn add(self, rhs: T) -> Self::Output {
                #[cfg(feature = "implicit_broadcast")]
                {
                    let out_shape = broadcast_output_shape(self.shape(), rhs.shape());
                    let lhs_box: Box<dyn Tensor<D>> = if self.shape() == &out_shape {
                        Box::new(self)
                    } else {
                        Box::new(BroadcastNode::new(Box::new(self), out_shape.clone()))
                    };
                    let rhs_box: Box<dyn Tensor<D>> = if rhs.shape() == &out_shape {
                        Box::new(rhs)
                    } else {
                        Box::new(BroadcastNode::new(Box::new(rhs), out_shape.clone()))
                    };
                    return BinaryOpNode { op: BinaryOp::Add, lhs: lhs_box, rhs: rhs_box, shape: out_shape };
                }
                #[cfg(not(feature = "implicit_broadcast"))]
                {
                    let shape = self.shape().clone();
                    BinaryOpNode {
                        op: BinaryOp::Add,
                        lhs: Box::new(self),
                        rhs: Box::new(rhs),
                        shape,
                    }
                }
            }
        }

        impl<D, T> Sub<T> for $Type<D>
        where
            D: DType + 'static,
            T: Tensor<D> + 'static,
        {
            type Output = BinaryOpNode<D>;
            fn sub(self, rhs: T) -> Self::Output {
                #[cfg(feature = "implicit_broadcast")]
                {
                    let out_shape = broadcast_output_shape(self.shape(), rhs.shape());
                    let lhs_box: Box<dyn Tensor<D>> = if self.shape() == &out_shape {
                        Box::new(self)
                    } else {
                        Box::new(BroadcastNode::new(Box::new(self), out_shape.clone()))
                    };
                    let rhs_box: Box<dyn Tensor<D>> = if rhs.shape() == &out_shape {
                        Box::new(rhs)
                    } else {
                        Box::new(BroadcastNode::new(Box::new(rhs), out_shape.clone()))
                    };
                    return BinaryOpNode { op: BinaryOp::Sub, lhs: lhs_box, rhs: rhs_box, shape: out_shape };
                }
                #[cfg(not(feature = "implicit_broadcast"))]
                {
                    let shape = self.shape().clone();
                    BinaryOpNode {
                        op: BinaryOp::Sub,
                        lhs: Box::new(self),
                        rhs: Box::new(rhs),
                        shape,
                    }
                }
            }
        }

        impl<D, T> Mul<T> for $Type<D>
        where
            D: DType + 'static,
            T: Tensor<D> + 'static,
        {
            type Output = BinaryOpNode<D>;
            fn mul(self, rhs: T) -> Self::Output {
                #[cfg(feature = "implicit_broadcast")]
                {
                    let out_shape = broadcast_output_shape(self.shape(), rhs.shape());
                    let lhs_box: Box<dyn Tensor<D>> = if self.shape() == &out_shape {
                        Box::new(self)
                    } else {
                        Box::new(BroadcastNode::new(Box::new(self), out_shape.clone()))
                    };
                    let rhs_box: Box<dyn Tensor<D>> = if rhs.shape() == &out_shape {
                        Box::new(rhs)
                    } else {
                        Box::new(BroadcastNode::new(Box::new(rhs), out_shape.clone()))
                    };
                    return BinaryOpNode { op: BinaryOp::Mul, lhs: lhs_box, rhs: rhs_box, shape: out_shape };
                }
                #[cfg(not(feature = "implicit_broadcast"))]
                {
                    let shape = self.shape().clone();
                    BinaryOpNode {
                        op: BinaryOp::Mul,
                        lhs: Box::new(self),
                        rhs: Box::new(rhs),
                        shape,
                    }
                }
            }
        }

        impl<D, T> Div<T> for $Type<D>
        where
            D: DType + 'static,
            T: Tensor<D> + 'static,
        {
            type Output = BinaryOpNode<D>;
            fn div(self, rhs: T) -> Self::Output {
                #[cfg(feature = "implicit_broadcast")]
                {
                    let out_shape = broadcast_output_shape(self.shape(), rhs.shape());
                    let lhs_box: Box<dyn Tensor<D>> = if self.shape() == &out_shape {
                        Box::new(self)
                    } else {
                        Box::new(BroadcastNode::new(Box::new(self), out_shape.clone()))
                    };
                    let rhs_box: Box<dyn Tensor<D>> = if rhs.shape() == &out_shape {
                        Box::new(rhs)
                    } else {
                        Box::new(BroadcastNode::new(Box::new(rhs), out_shape.clone()))
                    };
                    return BinaryOpNode { op: BinaryOp::Div, lhs: lhs_box, rhs: rhs_box, shape: out_shape };
                }
                #[cfg(not(feature = "implicit_broadcast"))]
                {
                    let shape = self.shape().clone();
                    BinaryOpNode {
                        op: BinaryOp::Div,
                        lhs: Box::new(self),
                        rhs: Box::new(rhs),
                        shape,
                    }
                }
            }
        }

        impl<D> $Type<D>
        where
            D: DType + 'static,
        {
            pub fn matmul(self, rhs: impl Tensor<D> + 'static) -> MatMulNode<D> {
                MatMulNode::new(Box::new(self), Box::new(rhs))
            }

            pub fn broadcast(self, shape: Shape) -> BroadcastNode<D> {
                BroadcastNode::new(Box::new(self), shape)
            }
        }
    };
}

impl_bin_ops_for_type!(Constant);
impl_bin_ops_for_type!(Input);
impl_bin_ops_for_type!(UnaryOpNode);
impl_bin_ops_for_type!(BinaryOpNode);
impl_bin_ops_for_type!(ReduceOpNode);
impl_bin_ops_for_type!(MatMulNode);
impl_bin_ops_for_type!(BroadcastNode);

pub struct MatMulNode<D: DType> {
    lhs: Box<dyn Tensor<D>>,
    rhs: Box<dyn Tensor<D>>,
    shape: Shape,
}

impl<D: DType> MatMulNode<D> {
    pub fn new(lhs: Box<dyn Tensor<D>>, rhs: Box<dyn Tensor<D>>) -> Self {
        // Basic shape inference for 2D matmul: [M,K] x [K,N] -> [M,N]
        let lshape = lhs.shape().clone();
        let rshape = rhs.shape().clone();
        if lshape.len() != 2 || rshape.len() != 2 {
            panic!("MatMul only supports 2D tensors for now");
        }
        if lshape[1] != rshape[0] {
            panic!("MatMul inner dimensions must match: got {:?} and {:?}", lshape, rshape);
        }
        let shape = vec![lshape[0], rshape[1]];
        Self { lhs, rhs, shape }
    }
}

impl<D: DType> Tensor<D> for MatMulNode<D> {
    fn shape(&self) -> &Shape {
        &self.shape
    }

    fn lower_to_graph(&self, graph: &mut graph::TensorGraph<D>) -> NodeIndex {
        let lhs_idx = self.lhs.lower_to_graph(graph);
        let rhs_idx = self.rhs.lower_to_graph(graph);
        let node_idx = graph.graph.add_node(TensorGraphNode::MatMul);
        // insert inferred shape
        graph.shapes.insert(node_idx, self.shape.clone());
        graph.graph.add_edge(lhs_idx, node_idx, 0);
        graph.graph.add_edge(rhs_idx, node_idx, 1);
        node_idx
    }
}

pub struct BroadcastNode<D: DType> {
    input: Box<dyn Tensor<D>>,
    shape: Shape,
}

impl<D: DType> BroadcastNode<D> {
    pub fn new(input: Box<dyn Tensor<D>>, shape: Shape) -> Self {
        // Basic validation: input can broadcast to target shape if input rank <= target rank and
        // trailing dims of input match or are 1.
        let in_shape = input.shape().clone();
        if in_shape.len() > shape.len() {
            panic!("Cannot broadcast to smaller rank");
        }
        // align right
        for (i, &dim) in in_shape.iter().rev().enumerate() {
            let target_dim = shape[shape.len() - 1 - i];
            if dim != target_dim && dim != 1 {
                panic!("Cannot broadcast dim {} -> {}", dim, target_dim);
            }
        }
        Self { input, shape }
    }
}

impl<D: DType> Tensor<D> for BroadcastNode<D> {
    fn shape(&self) -> &Shape {
        &self.shape
    }

    fn lower_to_graph(&self, graph: &mut graph::TensorGraph<D>) -> NodeIndex {
        let input_idx = self.input.lower_to_graph(graph);
        let node_idx = graph.graph.add_node(TensorGraphNode::Broadcast);
        graph.shapes.insert(node_idx, self.shape.clone());
        graph.graph.add_edge(input_idx, node_idx, 0);
        node_idx
    }
}

