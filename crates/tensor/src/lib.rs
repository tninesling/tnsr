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

pub trait Tensor<D: DType> {
    fn shape(&self) -> &Shape;

    fn lower_to_graph(&self, graph: &mut graph::TensorGraph<D>) -> NodeIndex;
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
        graph.graph.add_node(TensorGraphNode::Constant {
            data: self.data.clone(),
        })
    }
}

#[derive(Clone, Debug)]
pub struct Input<D: DType> {
    name: &'static str,
    shape: Shape,
    _marker: std::marker::PhantomData<D>,
}

impl<D: DType> Tensor<D> for Input<D> {
    fn shape(&self) -> &Shape {
        &self.shape
    }

    fn lower_to_graph(&self, graph: &mut graph::TensorGraph<D>) -> NodeIndex {
        graph
            .graph
            .add_node(TensorGraphNode::Input { name: self.name })
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
                let shape = self.shape().clone();
                BinaryOpNode {
                    op: BinaryOp::Add,
                    lhs: Box::new(self),
                    rhs: Box::new(rhs),
                    shape,
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
                let shape = self.shape().clone();
                BinaryOpNode {
                    op: BinaryOp::Sub,
                    lhs: Box::new(self),
                    rhs: Box::new(rhs),
                    shape,
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
                let shape = self.shape().clone();
                BinaryOpNode {
                    op: BinaryOp::Mul,
                    lhs: Box::new(self),
                    rhs: Box::new(rhs),
                    shape,
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
                let shape = self.shape().clone();
                BinaryOpNode {
                    op: BinaryOp::Div,
                    lhs: Box::new(self),
                    rhs: Box::new(rhs),
                    shape,
                }
            }
        }
    };
}

impl_bin_ops_for_type!(Constant);
impl_bin_ops_for_type!(Input);
impl_bin_ops_for_type!(UnaryOpNode);
impl_bin_ops_for_type!(BinaryOpNode);
impl_bin_ops_for_type!(ReduceOpNode);
