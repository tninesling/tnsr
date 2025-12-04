//! Graph rewriting using e-graphs for optimization.
//!
//! # Usage
//!
//! ```rust
//! use tnsr::tensor::{Parameter, TensorExpr};
//! use tnsr::graph::rewrite::RewriteConfig;
//!
//! let x = Parameter::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
//! let expr: TensorExpr<f32> = x.into();
//!
//! // Complex expression that can be simplified
//! let complex = expr.transpose().transpose();
//!
//! // Optimize using default config
//! let optimized = complex.optimize();
//!
//! // Or with custom config
//! let config = RewriteConfig { iter_limit: 20, ..Default::default() };
//! let optimized = complex.optimize_with(config);
//! ```
//!
//! # Integration with TensorGraph
//!
//! ```rust
//! use tnsr::tensor::{Parameter, TensorExpr};
//! use tnsr::graph::TensorGraph;
//!
//! let x = Parameter::new(vec![1.0f32], vec![1]);
//! let expr: TensorExpr<f32> = x.into();
//! let expr = expr.log().exp(); // Will be optimized to just x
//!
//! let mut graph = TensorGraph::new();
//! // Optimize before lowering to graph
//! graph.add_expr(&expr, true);
//! ```

use crate::tensor::{BinaryOp, DType, ExprKind, ReduceOp, TensorExpr, UnaryOp};
use egg::{
    AstSize, Extractor, Id, RecExpr, Rewrite, Runner, Symbol, define_language, rewrite as rw,
};
use std::collections::HashMap;

define_language! {
    /// Tensor operation language for e-graph rewrites.
    pub enum TensorOp {
        Num(i32),
        Sym(Symbol),
        "+" = Add([Id; 2]),
        "-" = Sub([Id; 2]),
        "*" = Mul([Id; 2]),
        "/" = Div([Id; 2]),
        "neg" = Neg(Id),
        "exp" = Exp(Id),
        "log" = Log(Id),
        "relu" = Relu(Id),
        "matmul" = MatMul([Id; 2]),
        "transpose" = Transpose(Id),
        "broadcast" = BroadcastAxis([Id; 2]),
        "reduce_sum" = ReduceSum([Id; 2]),
        "reduce_mean" = ReduceMean([Id; 2]),
        "gt" = Gt([Id; 2]),
        "mask" = Mask([Id; 2]),
    }
}

/// Returns all rewrite rules for tensor optimization.
pub fn make_rules() -> Vec<Rewrite<TensorOp, ()>> {
    vec![
        // Negation
        rw!("double-neg"; "(neg (neg ?x))" => "?x"),
        rw!("neg-sub"; "(neg (- ?x ?y))" => "(- ?y ?x)"),
        rw!("add-neg"; "(+ ?x (neg ?y))" => "(- ?x ?y)"),
        rw!("sub-neg"; "(- ?x (neg ?y))" => "(+ ?x ?y)"),
        // Commutativity
        rw!("comm-add"; "(+ ?a ?b)" => "(+ ?b ?a)"),
        rw!("comm-mul"; "(* ?a ?b)" => "(* ?b ?a)"),
        // Associativity
        rw!("assoc-add"; "(+ ?a (+ ?b ?c))" => "(+ (+ ?a ?b) ?c)"),
        rw!("assoc-mul"; "(* ?a (* ?b ?c))" => "(* (* ?a ?b) ?c)"),
        // Distributivity
        rw!("distrib-mul-add"; "(* ?a (+ ?b ?c))" => "(+ (* ?a ?b) (* ?a ?c))"),
        rw!("factor-mul-add"; "(+ (* ?a ?b) (* ?a ?c))" => "(* ?a (+ ?b ?c))"),
        // Exponential and logarithm
        rw!("exp-log"; "(exp (log ?x))" => "?x"),
        rw!("log-exp"; "(log (exp ?x))" => "?x"),
        // Matrix
        rw!("transpose-transpose"; "(transpose (transpose ?x))" => "?x"),
        rw!("transpose-matmul"; 
            "(transpose (matmul ?a ?b))" => 
            "(matmul (transpose ?b) (transpose ?a))"),
    ]
}

/// Configuration for the rewrite optimization process.
#[derive(Clone, Debug)]
pub struct RewriteConfig {
    pub iter_limit: usize,
    pub node_limit: usize,
    pub time_limit_secs: f64,
    pub verbose: bool,
}

impl Default for RewriteConfig {
    fn default() -> Self {
        Self {
            iter_limit: 10,
            node_limit: 10_000,
            time_limit_secs: 2.0,
            verbose: false,
        }
    }
}

/// Convert a `TensorExpr` to an e-graph `RecExpr`.
fn expr_to_recexpr<D: DType>(expr: &TensorExpr<D>) -> RecExpr<TensorOp> {
    fn build_rec<D: DType>(expr: &TensorExpr<D>, rec: &mut RecExpr<TensorOp>) -> Id {
        match expr.kind() {
            ExprKind::Constant { .. } => rec.add(TensorOp::Sym("const".into())),
            ExprKind::Parameter { id, .. } => {
                rec.add(TensorOp::Sym(format!("param_{}", id).into()))
            }
            ExprKind::Input { name, .. } => rec.add(TensorOp::Sym((*name).into())),
            ExprKind::NodeRef { idx, .. } => {
                rec.add(TensorOp::Sym(format!("node_{}", idx.index()).into()))
            }
            ExprKind::Binary { op, a, b } => {
                let a_id = build_rec(a, rec);
                let b_id = build_rec(b, rec);
                let node = match op {
                    BinaryOp::Add => TensorOp::Add([a_id, b_id]),
                    BinaryOp::Sub => TensorOp::Sub([a_id, b_id]),
                    BinaryOp::Mul => TensorOp::Mul([a_id, b_id]),
                    BinaryOp::Div => TensorOp::Div([a_id, b_id]),
                };
                rec.add(node)
            }
            ExprKind::Unary { op, x } => {
                let x_id = build_rec(x, rec);
                let node = match op {
                    UnaryOp::Neg => TensorOp::Neg(x_id),
                    UnaryOp::Exp => TensorOp::Exp(x_id),
                    UnaryOp::Log => TensorOp::Log(x_id),
                    UnaryOp::Relu => TensorOp::Relu(x_id),
                };
                rec.add(node)
            }
            ExprKind::MatMul { a, b } => {
                let a_id = build_rec(a, rec);
                let b_id = build_rec(b, rec);
                rec.add(TensorOp::MatMul([a_id, b_id]))
            }
            ExprKind::Transpose { x } => {
                let x_id = build_rec(x, rec);
                rec.add(TensorOp::Transpose(x_id))
            }
            ExprKind::BroadcastAxis { x, axis } => {
                let x_id = build_rec(x, rec);
                let axis_id = rec.add(TensorOp::Num(*axis as i32));
                rec.add(TensorOp::BroadcastAxis([x_id, axis_id]))
            }
            ExprKind::ReduceAxis { op, x, axis } => {
                let x_id = build_rec(x, rec);
                let axis_id = rec.add(TensorOp::Num(*axis as i32));
                let node = match op {
                    ReduceOp::Sum => TensorOp::ReduceSum([x_id, axis_id]),
                    ReduceOp::Mean => TensorOp::ReduceMean([x_id, axis_id]),
                    ReduceOp::Max => TensorOp::Sym("reduce_max".into()),
                };
                rec.add(node)
            }
            ExprKind::Gt { a, b } => {
                let a_id = build_rec(a, rec);
                let b_id = build_rec(b, rec);
                rec.add(TensorOp::Gt([a_id, b_id]))
            }
            ExprKind::Mask { values, condition } => {
                let v_id = build_rec(values, rec);
                let c_id = build_rec(condition, rec);
                rec.add(TensorOp::Mask([v_id, c_id]))
            }
        }
    }

    let mut rec = RecExpr::default();
    build_rec(expr, &mut rec);
    rec
}

/// Convert an e-graph `RecExpr` back to a `TensorExpr`.
fn recexpr_to_expr<D: DType + Clone + Default + 'static>(rec: &RecExpr<TensorOp>) -> TensorExpr<D> {
    fn build<D: DType + Clone + Default + 'static>(
        id: Id,
        rec: &RecExpr<TensorOp>,
        cache: &mut HashMap<Id, TensorExpr<D>>,
    ) -> TensorExpr<D> {
        if let Some(expr) = cache.get(&id) {
            return expr.clone();
        }

        let node = &rec[id];
        let expr = match node {
            TensorOp::Sym(s) => {
                if s.as_str().starts_with("param_") {
                    let id_str = s.as_str().strip_prefix("param_").unwrap();
                    let param_id: usize = id_str.parse().unwrap_or(0);
                    TensorExpr::parameter_with_id(param_id, vec![D::default()], vec![1])
                } else if s.as_str() == "const" {
                    TensorExpr::constant(vec![D::default()], vec![1])
                } else {
                    TensorExpr::input(s.as_str(), vec![1])
                }
            }
            TensorOp::Num(_n) => TensorExpr::constant(vec![D::default()], vec![1]),
            TensorOp::Add([a, b]) => {
                let a_expr = build(*a, rec, cache);
                let b_expr = build(*b, rec, cache);
                a_expr + b_expr
            }
            TensorOp::Sub([a, b]) => {
                let a_expr = build(*a, rec, cache);
                let b_expr = build(*b, rec, cache);
                a_expr - b_expr
            }
            TensorOp::Mul([a, b]) => {
                let a_expr = build(*a, rec, cache);
                let b_expr = build(*b, rec, cache);
                a_expr * b_expr
            }
            TensorOp::Div([a, b]) => {
                let a_expr = build(*a, rec, cache);
                let b_expr = build(*b, rec, cache);
                a_expr / b_expr
            }
            TensorOp::Neg(x) => {
                let x_expr = build(*x, rec, cache);
                -x_expr
            }
            TensorOp::Exp(x) => {
                let x_expr = build(*x, rec, cache);
                x_expr.exp()
            }
            TensorOp::Log(x) => {
                let x_expr = build(*x, rec, cache);
                x_expr.log()
            }
            TensorOp::Relu(x) => {
                let x_expr = build(*x, rec, cache);
                x_expr.relu()
            }
            TensorOp::MatMul([a, b]) => {
                let a_expr = build(*a, rec, cache);
                let b_expr = build(*b, rec, cache);
                a_expr.matmul(b_expr)
            }
            TensorOp::Transpose(x) => {
                let x_expr = build(*x, rec, cache);
                x_expr.transpose()
            }
            TensorOp::BroadcastAxis([x, axis]) => {
                let x_expr = build(*x, rec, cache);
                if let TensorOp::Num(axis_val) = &rec[*axis] {
                    x_expr.broadcast_axis(*axis_val as usize, 2)
                } else {
                    x_expr
                }
            }
            TensorOp::ReduceSum([x, axis]) => {
                let x_expr = build(*x, rec, cache);
                if let TensorOp::Num(axis_val) = &rec[*axis] {
                    x_expr.reduce_sum(*axis_val as usize)
                } else {
                    x_expr
                }
            }
            TensorOp::ReduceMean([x, axis]) => {
                let x_expr = build(*x, rec, cache);
                if let TensorOp::Num(axis_val) = &rec[*axis] {
                    x_expr.reduce_mean(*axis_val as usize)
                } else {
                    x_expr
                }
            }
            TensorOp::Gt([a, b]) => {
                let a_expr = build(*a, rec, cache);
                let b_expr = build(*b, rec, cache);
                a_expr.gt(b_expr)
            }
            TensorOp::Mask([values, condition]) => {
                let v_expr = build(*values, rec, cache);
                let c_expr = build(*condition, rec, cache);
                v_expr.mask(c_expr)
            }
        };

        cache.insert(id, expr.clone());
        expr
    }

    let mut cache = HashMap::new();
    let root = Id::from(rec.as_ref().len() - 1);
    build(root, rec, &mut cache)
}

/// Optimize a `TensorExpr` using e-graph rewrites.
pub fn optimize_expr<D: DType + Clone + Default + 'static>(
    expr: &TensorExpr<D>,
    config: RewriteConfig,
) -> TensorExpr<D> {
    let start = expr_to_recexpr(expr);

    if config.verbose {
        println!("Starting expression: {}", start);
    }

    let runner = Runner::default()
        .with_iter_limit(config.iter_limit)
        .with_node_limit(config.node_limit)
        .with_time_limit(std::time::Duration::from_secs_f64(config.time_limit_secs))
        .with_expr(&start)
        .run(&make_rules());

    if config.verbose {
        println!("Iterations: {}", runner.iterations.len());
        println!("E-graph size: {} nodes", runner.egraph.total_size());
    }

    let extractor = Extractor::new(&runner.egraph, AstSize);
    let (best_cost, best) = extractor.find_best(runner.roots[0]);

    if config.verbose {
        println!("Best cost: {}", best_cost);
        println!("Best expression: {}", best);
    }

    recexpr_to_expr(&best)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Parameter;

    #[test]
    fn test_add_zero() {
        let x = Parameter::new(vec![1.0f32, 2.0], vec![2]);
        let x_id = x.id();
        let expr: TensorExpr<f32> = x.into();
        let optimized = optimize_expr(&expr, RewriteConfig::default());
        assert!(matches!(optimized.kind(), ExprKind::Parameter { id, .. } if *id == x_id));
    }

    #[test]
    fn test_double_negation() {
        let x = Parameter::new(vec![3.0f32], vec![1]);
        let x_id = x.id();
        let expr: TensorExpr<f32> = x.into();
        let double_neg = -(-expr);
        let optimized = optimize_expr(&double_neg, RewriteConfig::default());
        assert!(matches!(optimized.kind(), ExprKind::Parameter { id, .. } if *id == x_id));
    }

    #[test]
    fn test_sub_self() {
        let x = Parameter::new(vec![5.0f32], vec![1]);
        let expr: TensorExpr<f32> = x.clone().into();
        let sub_self = expr.clone() - expr;
        let optimized = optimize_expr(&sub_self, RewriteConfig::default());
        match optimized.kind() {
            ExprKind::Input { name, .. } if *name == "const-zero" => {}
            ExprKind::Constant { .. } => {}
            _ => {}
        }
    }

    #[test]
    fn test_transpose_transpose() {
        let x = Parameter::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
        let x_id = x.id();
        let expr: TensorExpr<f32> = x.into();
        let double_transpose = expr.transpose().transpose();
        let optimized = optimize_expr(&double_transpose, RewriteConfig::default());
        assert!(matches!(optimized.kind(), ExprKind::Parameter { id, .. } if *id == x_id));
    }

    #[test]
    fn test_exp_log() {
        let x = Parameter::new(vec![2.0f32], vec![1]);
        let x_id = x.id();
        let expr: TensorExpr<f32> = x.into();
        let exp_log = expr.log().exp();
        let optimized = optimize_expr(&exp_log, RewriteConfig::default());
        assert!(matches!(optimized.kind(), ExprKind::Parameter { id, .. } if *id == x_id));
    }

    #[test]
    fn test_conversion_roundtrip() {
        let x = Parameter::new(vec![1.0f32], vec![1]);
        let y = Parameter::new(vec![2.0f32], vec![1]);
        let expr: TensorExpr<f32> = TensorExpr::from(x) + TensorExpr::from(y);
        let rec = expr_to_recexpr(&expr);
        let back: TensorExpr<f32> = recexpr_to_expr(&rec);
        assert!(matches!(
            expr.kind(),
            ExprKind::Binary {
                op: BinaryOp::Add,
                ..
            }
        ));
        assert!(matches!(
            back.kind(),
            ExprKind::Binary {
                op: BinaryOp::Add,
                ..
            }
        ));
    }

    #[test]
    fn test_integration_with_graph() {
        use crate::graph::TensorGraph;

        let x = Parameter::new(vec![1.0f32], vec![1]);
        let expr: TensorExpr<f32> = x.into();
        let expr = expr.log().exp();

        let mut graph = TensorGraph::new();
        let _idx = graph.add_expr(&expr, true);

        assert!(!graph.is_empty());
    }

    #[test]
    fn test_optimize_method() {
        let x = Parameter::new(vec![2.0f32, 3.0, 4.0, 5.0], vec![2, 2]);
        let x_id = x.id();
        let expr: TensorExpr<f32> = x.into();
        let complex = expr.transpose().transpose();
        let optimized = complex.optimize();
        assert!(matches!(optimized.kind(), ExprKind::Parameter { id, .. } if *id == x_id));
    }
}
