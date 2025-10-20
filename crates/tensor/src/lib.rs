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

pub trait DType {}
impl DType for f32 {}
impl DType for f64 {}
impl DType for i32 {}

pub type Shape = Vec<usize>;

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

#[derive(Clone, Debug)]
pub enum UnaryOp {
    Neg,
    Exp,
    Log,
    Relu,
}

#[derive(Clone, Debug)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Clone, Debug)]
pub enum ReduceOp {
    Sum,
    Max,
    Mean,
}

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
    Broadcast {
        x: TensorExpr<D>,
    }, // shape stored in node
    Reduce {
        op: ReduceOp,
        x: TensorExpr<D>,
        axis: usize,
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
        D: 'static,
    {
        // Validate broadcasting compatibility
        let in_shape = self.shape();
        if in_shape.len() > to.len() {
            panic!("Cannot broadcast to smaller rank");
        }
        for (i, &dim) in in_shape.iter().rev().enumerate() {
            let target_dim = to[to.len() - 1 - i];
            if dim != target_dim && dim != 1 {
                panic!("Cannot broadcast dim {dim} -> {target_dim}");
            }
        }
        Self(Arc::new(ExprNode {
            shape: to,
            kind: ExprKind::Broadcast { x: self },
        }))
    }

    pub fn reduce_sum(self, axis: usize) -> Self {
        let rank = self.shape().len();
        assert!(axis < rank, "reduce axis out of bounds");
        let mut out_shape = self.shape().clone();
        out_shape.remove(axis);
        Self(Arc::new(ExprNode {
            shape: out_shape,
            kind: ExprKind::Reduce {
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
        out_shape.remove(axis);
        Self(Arc::new(ExprNode {
            shape: out_shape,
            kind: ExprKind::Reduce {
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
        out_shape.remove(axis);
        Self(Arc::new(ExprNode {
            shape: out_shape,
            kind: ExprKind::Reduce {
                op: ReduceOp::Max,
                x: self,
                axis,
            },
        }))
    }

    pub fn mean_all(self) -> Self {
        let mut expr = self;
        while !expr.shape().is_empty() {
            let axis = expr.shape().len() - 1;
            expr = expr.reduce_mean(axis);
        }
        expr
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
    pub fn lower_to_graph(&self, graph: &mut graph::TensorGraph<D>) -> NodeIndex {
        fn lower_rec<D: DType>(expr: &TensorExpr<D>, g: &mut graph::TensorGraph<D>) -> NodeIndex {
            match &expr.0.kind {
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
                    // If implicit broadcasting is enabled and input shapes differ from out, insert broadcast nodes
                    #[cfg(feature = "implicit_broadcast")]
                    let a_idx = {
                        if a.shape() != expr.shape() {
                            let bnode = g.graph.add_node(TensorGraphNode::Broadcast);
                            g.shapes.insert(bnode, expr.shape().clone());
                            g.graph.add_edge(a_idx, bnode, 0);
                            bnode
                        } else {
                            a_idx
                        }
                    };
                    #[cfg(feature = "implicit_broadcast")]
                    let b_idx = {
                        if b.shape() != expr.shape() {
                            let bnode = g.graph.add_node(TensorGraphNode::Broadcast);
                            g.shapes.insert(bnode, expr.shape().clone());
                            g.graph.add_edge(b_idx, bnode, 0);
                            bnode
                        } else {
                            b_idx
                        }
                    };
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
                ExprKind::Broadcast { x } => {
                    let x_idx = lower_rec(x, g);
                    let node_idx = g.graph.add_node(TensorGraphNode::Broadcast);
                    g.shapes.insert(node_idx, expr.shape().clone());
                    g.graph.add_edge(x_idx, node_idx, 0);
                    node_idx
                }
                ExprKind::Reduce { op, x, axis } => {
                    let x_idx = lower_rec(x, g);
                    let node_idx = g.graph.add_node(TensorGraphNode::Reduce {
                        op: op.clone(),
                        axis: *axis,
                    });
                    g.shapes.insert(node_idx, expr.shape().clone());
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
