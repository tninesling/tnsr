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
//! `optimize`, `optimize_with`, and [`optimize_expr`] use conservative rules:
//! they simplify structural round trips without reassociating arithmetic or
//! cancelling exp/log operations. For supported expressions, callers may opt
//! into real-algebra rewrites explicitly:
//!
//! ```rust
//! use tnsr::graph::rewrite::{optimize_expr_with_numerics, RewriteConfig, RewriteNumerics};
//! use tnsr::tensor::TensorExpr;
//!
//! let x = TensorExpr::<f32>::input("x", vec![1]);
//! let expr = x.log().exp();
//! let optimized = optimize_expr_with_numerics(
//!     &expr,
//!     RewriteConfig::default(),
//!     RewriteNumerics::RealAlgebra,
//! );
//! ```
//!
//! Real-algebra rewrites can change rounding, signed zero, NaNs and infinities.
//! They are independent of backend matmul precision settings.
//!
//! # Integration with TensorGraph
//!
//! ```rust
//! use tnsr::tensor::{Parameter, TensorExpr};
//! use tnsr::graph::TensorGraph;
//!
//! let x = Parameter::new(vec![1.0f32], vec![1]);
//! let expr: TensorExpr<f32> = x.into();
//! let expr = expr.log().exp(); // Conservative rules retain both operations
//!
//! let mut graph = TensorGraph::new();
//! // Optimize before lowering to graph
//! graph.add_expr(&expr, true);
//! ```

mod islands;
mod layout_rules;
mod linear_rows;
mod reduction_rules;
mod segments;
mod tensor_analysis;

use crate::tensor::{BinaryOp, DType, ExprKind, ReduceOp, TensorExpr, UnaryOp};
use egg::{
    Analysis, Extractor, Id, RecExpr, Rewrite, Runner, Symbol, define_language, rewrite as rw,
};
use std::collections::HashMap;
use std::sync::Arc;

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
        "reshape" = Reshape(Box<[Id]>),
        "permute" = Permute(Box<[Id]>),
        "broadcast" = BroadcastAxis([Id; 3]),
        "reduce_sum" = ReduceSum([Id; 2]),
        "reduce_mean" = ReduceMean([Id; 2]),
        "gt" = Gt([Id; 2]),
        "mask" = Mask([Id; 2]),
    }
}

fn is_opaque<D: DType>(expr: &TensorExpr<D>) -> bool {
    let wide_axis = match expr.kind() {
        ExprKind::Permute { axes, .. } => axes.iter().any(|&axis| i32::try_from(axis).is_err()),
        ExprKind::BroadcastAxis { axis, .. } | ExprKind::ReduceAxis { axis, .. } => {
            i32::try_from(*axis).is_err()
        }
        _ => false,
    };
    // TensorOp metadata uses i32 literals. Avoid narrowing dimensions even
    // when only a descendant reshape or broadcast would serialize them.
    wide_axis
        || expr.shape().iter().any(|&dim| i32::try_from(dim).is_err())
        || matches!(
            expr.kind(),
            ExprKind::Linear { .. }
                | ExprKind::Cast { .. }
                | ExprKind::Index { .. }
                | ExprKind::Narrow { .. }
                | ExprKind::Concat { .. }
                | ExprKind::Embedding { .. }
                | ExprKind::EmbeddingBackward { .. }
                | ExprKind::IndexedCrossEntropy { .. }
                | ExprKind::IndexedCrossEntropyBackward { .. }
                | ExprKind::ReduceAxis {
                    op: ReduceOp::Max,
                    ..
                }
                | ExprKind::Conv2d { .. }
                | ExprKind::ConvTranspose2d { .. }
                | ExprKind::Conv2dBackwardWeight { .. }
                | ExprKind::MaxPool2d { .. }
                | ExprKind::MaxPool2dBackward { .. }
                | ExprKind::Flatten { .. }
        )
}

fn leaf_symbol<D: DType>(expr: &TensorExpr<D>) -> Option<Symbol> {
    let name = match expr.kind() {
        ExprKind::Constant { data } => format!("const_{:p}", Arc::as_ptr(data)),
        ExprKind::Parameter { id, .. } => format!("param_{id}"),
        ExprKind::Input { name } => format!("input_{name}"),
        ExprKind::NodeRef { idx } => format!("node_{}", idx.index()),
        _ if is_opaque(expr) => format!("opaque_{:p}", expr.kind()),
        _ => return None,
    };
    Some(name.into())
}

/// Returns conservative rewrite rules for supported tensor expressions.
pub fn make_rules() -> Vec<Rewrite<TensorOp, ()>> {
    make_rules_with_numerics(RewriteNumerics::PreserveOperations)
}

/// Numerical assumptions permitted during expression rewriting.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RewriteNumerics {
    /// Preserve arithmetic order, domains and rounding boundaries in supported
    /// expressions, while simplifying redundant layouts and double negation.
    #[default]
    PreserveOperations,
    /// Permit real-algebra identities, including reassociation and exp/log
    /// cancellation, matrix factoring, matrix-chain reassociation, and
    /// narrow/concat identities that change gradient scaffolding. These may
    /// change finite results, signed zero, NaNs and infinities; callers must
    /// establish suitability for their workload.
    RealAlgebra,
}

/// Returns shape-independent rewrite rules under explicit numerical assumptions.
/// The expression optimizer also uses shape analysis for layout composition,
/// guarded matrix identities and cost-based extraction.
pub fn make_rules_with_numerics(numerics: RewriteNumerics) -> Vec<Rewrite<TensorOp, ()>> {
    rules_with_numerics::<()>(numerics)
}

fn rules_with_numerics<A: Analysis<TensorOp> + 'static>(
    numerics: RewriteNumerics,
) -> Vec<Rewrite<TensorOp, A>> {
    let mut rules = vec![
        rw!("double-neg"; "(neg (neg ?x))" => "?x"),
        rw!("transpose-transpose"; "(transpose (transpose ?x))" => "?x"),
    ];
    if numerics == RewriteNumerics::PreserveOperations {
        return rules;
    }
    rules.extend(vec![
        // Negation
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
        rw!("transpose-matmul"; 
            "(transpose (matmul ?a ?b))" => 
            "(matmul (transpose ?b) (transpose ?a))"),
    ]);
    rules
}

/// Configuration for the rewrite optimization process.
#[derive(Clone, Debug)]
pub struct RewriteConfig {
    pub iter_limit: usize,
    pub node_limit: usize,
    pub time_limit_secs: f64,
}

impl Default for RewriteConfig {
    fn default() -> Self {
        Self {
            iter_limit: 10,
            node_limit: 10_000,
            time_limit_secs: 2.0,
        }
    }
}

/// Convert a `TensorExpr` to an e-graph `RecExpr`.
fn expr_to_recexpr<D: DType>(expr: &TensorExpr<D>) -> RecExpr<TensorOp> {
    fn build_rec<D: DType>(expr: &TensorExpr<D>, rec: &mut RecExpr<TensorOp>) -> Id {
        if let Some(symbol) = leaf_symbol(expr) {
            return rec.add(TensorOp::Sym(symbol));
        }
        match expr.kind() {
            ExprKind::Linear { .. } | ExprKind::Cast { .. } | ExprKind::Index { .. } => {
                unreachable!("index sources bypass expression rewriting")
            }
            ExprKind::Constant { .. }
            | ExprKind::Parameter { .. }
            | ExprKind::Input { .. }
            | ExprKind::NodeRef { .. } => rec.add(TensorOp::Sym(leaf_symbol(expr).unwrap())),
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
            ExprKind::Reshape { x } => {
                let mut children = Vec::with_capacity(expr.shape().len() + 1);
                children.push(build_rec(x, rec));
                children.extend(
                    expr.shape()
                        .iter()
                        .map(|&dim| rec.add(TensorOp::Num(dim as i32))),
                );
                rec.add(TensorOp::Reshape(children.into_boxed_slice()))
            }
            ExprKind::Permute { x, axes } => {
                let mut children = Vec::with_capacity(axes.len() + 1);
                children.push(build_rec(x, rec));
                children.extend(axes.iter().map(|&axis| rec.add(TensorOp::Num(axis as i32))));
                rec.add(TensorOp::Permute(children.into_boxed_slice()))
            }
            ExprKind::BroadcastAxis { x, axis } => {
                let x_id = build_rec(x, rec);
                let axis_id = rec.add(TensorOp::Num(*axis as i32));
                let size_id = rec.add(TensorOp::Num(expr.shape()[*axis] as i32));
                rec.add(TensorOp::BroadcastAxis([x_id, axis_id, size_id]))
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
            ExprKind::Conv2d { input, weight, .. } => {
                let _ = build_rec(input, rec);
                let _ = build_rec(weight, rec);
                rec.add(TensorOp::Sym("Conv2d".into()))
            }
            ExprKind::ConvTranspose2d {
                grad_output,
                weight,
                ..
            } => {
                let _ = build_rec(grad_output, rec);
                let _ = build_rec(weight, rec);
                rec.add(TensorOp::Sym("ConvTranspose2d".into()))
            }
            ExprKind::Conv2dBackwardWeight {
                input, grad_output, ..
            } => {
                let _ = build_rec(input, rec);
                let _ = build_rec(grad_output, rec);
                rec.add(TensorOp::Sym("Conv2dBackwardWeight".into()))
            }
            ExprKind::MaxPool2d { x, .. } => {
                let _ = build_rec(x, rec);
                rec.add(TensorOp::Sym("MaxPool2d".into()))
            }
            ExprKind::MaxPool2dBackward {
                input,
                output,
                grad_output,
                ..
            } => {
                let _ = build_rec(input, rec);
                let _ = build_rec(output, rec);
                let _ = build_rec(grad_output, rec);
                rec.add(TensorOp::Sym("MaxPool2dBackward".into()))
            }
            ExprKind::Flatten { x } => {
                let _ = build_rec(x, rec);
                rec.add(TensorOp::Sym("Flatten".into()))
            }
            ExprKind::Narrow { .. }
            | ExprKind::Concat { .. }
            | ExprKind::Embedding { .. }
            | ExprKind::EmbeddingBackward { .. }
            | ExprKind::IndexedCrossEntropy { .. }
            | ExprKind::IndexedCrossEntropyBackward { .. } => {
                unreachable!("indexed expressions bypass e-graph rewriting")
            }
        }
    }

    let mut rec = RecExpr::default();
    build_rec(expr, &mut rec);
    rec
}

/// Convert an e-graph `RecExpr` back to a `TensorExpr`.
fn recexpr_to_expr<D: DType + Clone + Default + 'static>(
    rec: &RecExpr<TensorOp>,
    original: &TensorExpr<D>,
) -> TensorExpr<D> {
    fn collect_leaves<D: DType>(expr: &TensorExpr<D>, leaves: &mut HashMap<Symbol, TensorExpr<D>>) {
        if let Some(symbol) = leaf_symbol(expr) {
            leaves.insert(symbol, expr.clone());
        } else {
            for child in expr.children() {
                collect_leaves(child, leaves);
            }
        }
    }

    fn build<D: DType + Clone + Default + 'static>(
        id: Id,
        rec: &RecExpr<TensorOp>,
        cache: &mut HashMap<Id, TensorExpr<D>>,
        leaves: &HashMap<Symbol, TensorExpr<D>>,
    ) -> TensorExpr<D> {
        if let Some(expr) = cache.get(&id) {
            return expr.clone();
        }

        let node = &rec[id];
        let expr = match node {
            TensorOp::Sym(s) => leaves
                .get(s)
                .unwrap_or_else(|| panic!("rewrite introduced unknown tensor leaf {s}"))
                .clone(),
            TensorOp::Num(_n) => TensorExpr::constant(vec![D::default()], vec![1]),
            TensorOp::Add([a, b]) => {
                let a_expr = build(*a, rec, cache, leaves);
                let b_expr = build(*b, rec, cache, leaves);
                a_expr + b_expr
            }
            TensorOp::Sub([a, b]) => {
                let a_expr = build(*a, rec, cache, leaves);
                let b_expr = build(*b, rec, cache, leaves);
                a_expr - b_expr
            }
            TensorOp::Mul([a, b]) => {
                let a_expr = build(*a, rec, cache, leaves);
                let b_expr = build(*b, rec, cache, leaves);
                a_expr * b_expr
            }
            TensorOp::Div([a, b]) => {
                let a_expr = build(*a, rec, cache, leaves);
                let b_expr = build(*b, rec, cache, leaves);
                a_expr / b_expr
            }
            TensorOp::Neg(x) => {
                let x_expr = build(*x, rec, cache, leaves);
                -x_expr
            }
            TensorOp::Exp(x) => {
                let x_expr = build(*x, rec, cache, leaves);
                x_expr.exp()
            }
            TensorOp::Log(x) => {
                let x_expr = build(*x, rec, cache, leaves);
                x_expr.log()
            }
            TensorOp::Relu(x) => {
                let x_expr = build(*x, rec, cache, leaves);
                x_expr.relu()
            }
            TensorOp::MatMul([a, b]) => {
                let a_expr = build(*a, rec, cache, leaves);
                let b_expr = build(*b, rec, cache, leaves);
                a_expr.matmul(b_expr)
            }
            TensorOp::Transpose(x) => {
                let x_expr = build(*x, rec, cache, leaves);
                x_expr.transpose()
            }
            TensorOp::Reshape(children) => {
                let x_expr = build(children[0], rec, cache, leaves);
                let shape = children[1..]
                    .iter()
                    .map(|id| match rec[*id] {
                        TensorOp::Num(dim) => dim as usize,
                        _ => panic!("reshape dimensions must be integers"),
                    })
                    .collect();
                x_expr.reshape(shape)
            }
            TensorOp::Permute(children) => {
                let x_expr = build(children[0], rec, cache, leaves);
                let axes = children[1..]
                    .iter()
                    .map(|id| match rec[*id] {
                        TensorOp::Num(axis) => axis as usize,
                        _ => panic!("permutation axes must be integers"),
                    })
                    .collect();
                x_expr.permute(axes)
            }
            TensorOp::BroadcastAxis([x, axis, size]) => {
                let x_expr = build(*x, rec, cache, leaves);
                match (&rec[*axis], &rec[*size]) {
                    (TensorOp::Num(axis), TensorOp::Num(size)) => {
                        x_expr.broadcast_axis(*axis as usize, *size as usize)
                    }
                    _ => panic!("broadcast axis and size must be integers"),
                }
            }
            TensorOp::ReduceSum([x, axis]) => {
                let x_expr = build(*x, rec, cache, leaves);
                if let TensorOp::Num(axis_val) = &rec[*axis] {
                    x_expr.reduce_sum(*axis_val as usize)
                } else {
                    x_expr
                }
            }
            TensorOp::ReduceMean([x, axis]) => {
                let x_expr = build(*x, rec, cache, leaves);
                if let TensorOp::Num(axis_val) = &rec[*axis] {
                    x_expr.reduce_mean(*axis_val as usize)
                } else {
                    x_expr
                }
            }
            TensorOp::Gt([a, b]) => {
                let a_expr = build(*a, rec, cache, leaves);
                let b_expr = build(*b, rec, cache, leaves);
                a_expr.gt(b_expr)
            }
            TensorOp::Mask([values, condition]) => {
                let v_expr = build(*values, rec, cache, leaves);
                let c_expr = build(*condition, rec, cache, leaves);
                v_expr.mask(c_expr)
            }
        };

        cache.insert(id, expr.clone());
        expr
    }

    let mut leaves = HashMap::new();
    collect_leaves(original, &mut leaves);
    let mut cache = HashMap::new();
    let root = Id::from(rec.as_ref().len() - 1);
    build(root, rec, &mut cache, &leaves)
}

/// Optimize a supported `TensorExpr` using conservative e-graph rewrites.
///
/// Arithmetic reassociation and inverse exp/log rules require explicit opt-in
/// through [`optimize_expr_with_numerics`]. This is also the default used by
/// `TensorExpr::optimize`, `TensorExpr::optimize_with`, and
/// `TensorGraph::add_expr(expr, true)`.
pub fn optimize_expr<D: DType + Clone + Default + 'static>(
    expr: &TensorExpr<D>,
    config: RewriteConfig,
) -> TensorExpr<D> {
    optimize_expr_with_numerics(expr, config, RewriteNumerics::PreserveOperations)
}

/// Optimize a supported `TensorExpr` under explicit numerical assumptions.
///
/// [`RewriteNumerics::RealAlgebra`] restores the algebraic rules that can alter
/// floating-point behavior, including rank-2 shared-matmul factoring and
/// cost-guided matrix-chain reassociation. It also collapses batched matrix rows
/// when a rank-2 RHS is shared across rows, including an optional shared bias vector. Apply
/// before differentiation to avoid batched parameter-gradient partials; this
/// changes gradient reduction grouping and may change backend matmul rounding.
/// Supported islands beneath unsupported operators are optimized independently.
/// Unsupported operations and wide metadata remain opaque leaves within each
/// island, so rewriting never moves arithmetic across a cast/index boundary.
/// Forward-exact narrow/concat identities require `RealAlgebra` because removing
/// their gradient scaffolding can change signed zero and nonfinite gradients. Runner limits
/// apply to each island, and shared opaque expressions are visited only once.
pub fn optimize_expr_with_numerics<D: DType + Clone + Default + 'static>(
    expr: &TensorExpr<D>,
    config: RewriteConfig,
    numerics: RewriteNumerics,
) -> TensorExpr<D> {
    islands::optimize(expr, &config, numerics)
}

fn optimize_supported<D: DType + Clone + Default + 'static>(
    expr: &TensorExpr<D>,
    config: &RewriteConfig,
    numerics: RewriteNumerics,
) -> TensorExpr<D> {
    // Preserve unsupported operations, wide metadata and observable cast
    // identities instead of rebuilding them from lossy placeholder symbols.
    if is_opaque(expr) {
        return if numerics == RewriteNumerics::RealAlgebra {
            linear_rows::rewrite(expr).unwrap_or_else(|| expr.clone())
        } else {
            expr.clone()
        };
    }

    let start = expr_to_recexpr(expr);

    tracing::debug!("Starting expression: {}", start);

    let analysis = tensor_analysis::TensorAnalysis::new(expr);
    let runner = Runner::<_, _, ()>::new(analysis)
        .with_iter_limit(config.iter_limit)
        .with_node_limit(config.node_limit)
        .with_time_limit(std::time::Duration::from_secs_f64(config.time_limit_secs))
        .with_expr(&start)
        .run(tensor_analysis::rules(numerics));

    tracing::debug!("Iterations: {}", runner.iterations.len());
    tracing::debug!("E-graph size: {} nodes", runner.egraph.total_size());

    let extractor = Extractor::new(&runner.egraph, tensor_analysis::TensorCost(&runner.egraph));
    let (best_cost, best) = extractor.find_best(runner.roots[0]);

    tracing::debug!("Best cost: {}", best_cost);
    tracing::debug!("Best expression: {}", best);

    let optimized = if best == start {
        expr.clone()
    } else {
        recexpr_to_expr(&best, expr)
    };
    if numerics == RewriteNumerics::RealAlgebra {
        linear_rows::rewrite(&optimized).unwrap_or(optimized)
    } else {
        optimized
    }
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
        let optimized = optimize_expr_with_numerics(
            &exp_log,
            RewriteConfig::default(),
            RewriteNumerics::RealAlgebra,
        );
        assert!(matches!(optimized.kind(), ExprKind::Parameter { id, .. } if *id == x_id));
    }

    #[test]
    fn test_conversion_roundtrip() {
        let x = Parameter::new(vec![1.0f32], vec![1]);
        let y = Parameter::new(vec![2.0f32], vec![1]);
        let expr: TensorExpr<f32> = TensorExpr::from(x) + TensorExpr::from(y);
        let rec = expr_to_recexpr(&expr);
        let back: TensorExpr<f32> = recexpr_to_expr(&rec, &expr);
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
    fn test_shape_operations_roundtrip() {
        let x = Parameter::new(vec![1.0f32; 6], vec![1, 6]);
        let expr: TensorExpr<f32> = TensorExpr::from(x)
            .broadcast_axis(0, 2)
            .reshape(vec![2, 3, 2])
            .permute(vec![1, 0, 2]);
        let optimized = optimize_expr(&expr, RewriteConfig::default());

        assert_eq!(optimized.shape(), &vec![3, 2, 2]);
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

#[cfg(test)]
mod numerical_contract_tests {
    use super::*;
    use crate::{Executor, SimpleExecutor, graph::TensorGraph};

    fn scalar(value: f32) -> TensorExpr<f32> {
        TensorExpr::constant(vec![value], vec![1])
    }

    fn execute(expr: TensorExpr<f32>) -> Vec<f32> {
        SimpleExecutor::new()
            .execute(&TensorGraph::from(expr), HashMap::new())
            .unwrap()
    }

    fn assert_preserved(expr: &TensorExpr<f32>) {
        let expected = execute(expr.clone());
        let actual = execute(optimize_expr(expr, RewriteConfig::default()));
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            if expected.is_nan() {
                assert!(actual.is_nan());
            } else {
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "{actual} vs {expected}"
                );
            }
        }
    }

    fn real_algebra(expr: &TensorExpr<f32>) -> TensorExpr<f32> {
        optimize_expr_with_numerics(expr, RewriteConfig::default(), RewriteNumerics::RealAlgebra)
    }

    #[test]
    fn conservative_preserves_exp_log_domain_overflow_and_underflow() {
        let values = vec![
            -1.0f32,
            -0.0,
            0.0,
            f32::MIN_POSITIVE,
            1.0,
            100.0,
            -110.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ];
        let x = TensorExpr::constant(values.clone(), vec![values.len()]);
        assert_preserved(&x.clone().log().exp());
        assert_preserved(&x.exp().log());
        assert!(execute(scalar(-1.0).log().exp())[0].is_nan());
        assert!(execute(scalar(100.0).exp().log())[0].is_infinite());
        assert_eq!(execute(scalar(-110.0).exp().log())[0], f32::NEG_INFINITY);
    }

    #[test]
    fn conservative_preserves_signed_zero_in_negated_subtraction() {
        let x = scalar(0.0);
        let expr = -(x.clone() - x);
        assert_preserved(&expr);
        assert_eq!(execute(expr.clone())[0].to_bits(), (-0.0f32).to_bits());
        assert_eq!(execute(real_algebra(&expr))[0].to_bits(), 0.0f32.to_bits());
    }

    #[test]
    fn conservative_preserves_nonfinite_products_and_finite_cancellation() {
        for value in [f32::MAX, f32::INFINITY] {
            let a = scalar(value);
            let expr = a.clone() * scalar(2.0) + a * scalar(-2.0);
            assert_preserved(&expr);
            assert!(execute(expr.clone())[0].is_nan());
            let optimized = execute(real_algebra(&expr))[0];
            if value.is_finite() {
                assert_eq!(optimized, 0.0);
            }
        }
        let a = scalar(1e10);
        let expr = a.clone() * scalar(1.0000001) + a * scalar(-1.0);
        assert_preserved(&expr);
        assert_eq!(execute(expr.clone())[0], 1024.0);
        assert_ne!(execute(real_algebra(&expr))[0], 1024.0);
    }

    #[test]
    fn algebraic_inverse_rules_require_explicit_opt_in() {
        let expr = scalar(-1.0).log().exp();
        assert!(execute(optimize_expr(&expr, RewriteConfig::default()))[0].is_nan());
        assert_eq!(execute(real_algebra(&expr)), vec![-1.0]);
    }

    #[test]
    fn conservative_still_eliminates_structural_round_trips() {
        let x = TensorExpr::constant(vec![-0.0f32], vec![1, 1]);
        let optimized = optimize_expr(
            &(-(-x.clone().transpose().transpose())),
            RewriteConfig::default(),
        );
        assert!(matches!(optimized.kind(), ExprKind::Constant { .. }));
        assert_eq!(execute(optimized)[0].to_bits(), (-0.0f32).to_bits());
    }

    #[test]
    fn public_optimization_entry_points_use_conservative_rules() {
        let expr = scalar(-1.0).log().exp();
        let config = RewriteConfig {
            iter_limit: 10,
            node_limit: 10_000,
            time_limit_secs: 2.0,
        };
        assert!(execute(expr.optimize())[0].is_nan());
        assert!(execute(expr.optimize_with(config))[0].is_nan());
        let mut graph = TensorGraph::new();
        graph.add_expr(&expr, true);
        let actual = SimpleExecutor::new()
            .execute(&graph, HashMap::new())
            .unwrap();
        assert!(actual[0].is_nan());
    }

    fn assert_unsupported_preserved(expr: &TensorExpr<f32>, expected: &[f32]) {
        assert_eq!(execute(expr.clone()), expected);
        for numerics in [
            RewriteNumerics::PreserveOperations,
            RewriteNumerics::RealAlgebra,
        ] {
            let optimized = optimize_expr_with_numerics(expr, RewriteConfig::default(), numerics);
            assert_eq!(optimized.shape(), expr.shape());
            assert_eq!(execute(optimized), expected);
        }
    }

    #[test]
    fn unsupported_max_reductions_keep_distinct_inputs_and_axes() {
        let a = TensorExpr::constant(vec![-1.0f32, 2.0, 3.0, -4.0], vec![2, 2]);
        let b = TensorExpr::constant(vec![10.0f32, 20.0, 30.0, 40.0], vec![2, 2]);
        let expression = a.reduce_max(0) + b.reduce_max(1);
        assert_eq!(expression.shape(), &[2, 2]);
        assert_unsupported_preserved(&expression, &[23.0, 22.0, 43.0, 42.0]);
    }

    #[test]
    fn unsupported_convolution_pooling_and_flatten_keep_shapes_and_values() {
        let input = TensorExpr::constant((1_u8..=9).map(f32::from).collect(), vec![1, 1, 3, 3]);
        let weight = TensorExpr::constant(vec![1.0f32, -1.0, 2.0, 0.0], vec![1, 1, 2, 2]);
        let convolution = input.clone().conv2d(weight, 1, 0);
        assert_unsupported_preserved(&convolution, &[7.0, 9.0, 13.0, 15.0]);
        let pooling = input.max_pool2d(2, 1);
        assert_unsupported_preserved(&pooling, &[5.0, 6.0, 8.0, 9.0]);
        let flattened = convolution.flatten();
        assert_eq!(flattened.shape(), &[1, 4]);
        assert_unsupported_preserved(&flattened, &[7.0, 9.0, 13.0, 15.0]);
    }

    #[test]
    fn unsupported_convolution_and_pooling_backward_keep_values() {
        let input = TensorExpr::constant((1_u8..=9).map(f32::from).collect(), vec![1, 1, 3, 3]);
        let weight = TensorExpr::constant(vec![1.0f32, -1.0, 2.0, 0.0], vec![1, 1, 2, 2]);
        let gradient = TensorExpr::constant(vec![1.0f32; 4], vec![1, 1, 2, 2]);
        let input_gradient =
            TensorExpr::conv_transpose_2d(gradient.clone(), weight, vec![1, 1, 3, 3], 1, 0);
        assert_unsupported_preserved(
            &input_gradient,
            &[1.0, 0.0, -1.0, 3.0, 2.0, -1.0, 2.0, 2.0, 0.0],
        );
        let weight_gradient = TensorExpr::conv2d_backward_weight(
            input.clone(),
            gradient.clone(),
            vec![1, 1, 2, 2],
            1,
            0,
        );
        assert_unsupported_preserved(&weight_gradient, &[12.0, 16.0, 24.0, 28.0]);
        let pooled = input.clone().max_pool2d(2, 1);
        let pool_gradient =
            TensorExpr::max_pool2d_backward(input, pooled, gradient, vec![1, 1, 3, 3], 2, 1);
        assert_unsupported_preserved(
            &pool_gradient,
            &[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        );
    }

    #[test]
    fn wide_shape_metadata_bypasses_i32_rewrite_encoding() {
        let wide = usize::try_from(i32::MAX).unwrap() + 1;
        let expression = TensorExpr::<f32>::input("wide", vec![wide]).reshape(vec![1, wide]);
        for numerics in [
            RewriteNumerics::PreserveOperations,
            RewriteNumerics::RealAlgebra,
        ] {
            let optimized =
                optimize_expr_with_numerics(&expression, RewriteConfig::default(), numerics);
            assert_eq!(optimized.shape(), &[1, wide]);
            assert!(matches!(optimized.kind(), ExprKind::Reshape { .. }));
        }
    }
}
