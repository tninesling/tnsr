//! Shape facts and an approximate work model for tensor equality saturation.
use super::{RewriteNumerics, TensorOp, leaf_symbol, rules_with_numerics};
use crate::tensor::{DType, TensorExpr, broadcast_output_shape};
use egg::{
    Analysis, CostFunction, DidMerge, EGraph, Id, Language, Rewrite, Subst, Symbol, rewrite as rw,
};
use std::collections::HashMap;
use std::sync::LazyLock;

type Graph = EGraph<TensorOp, TensorAnalysis>;

#[derive(Default)]
pub(super) struct TensorAnalysis {
    leaves: HashMap<Symbol, Vec<usize>>,
}

impl TensorAnalysis {
    pub(super) fn new<D: DType>(expr: &TensorExpr<D>) -> Self {
        fn collect<D: DType>(expr: &TensorExpr<D>, leaves: &mut HashMap<Symbol, Vec<usize>>) {
            if let Some(symbol) = leaf_symbol(expr) {
                leaves.insert(symbol, expr.shape().clone());
            } else {
                for child in expr.children() {
                    collect(child, leaves);
                }
            }
        }
        let mut analysis = Self::default();
        collect(expr, &mut analysis.leaves);
        analysis
    }
}

fn number(graph: &Graph, id: Id) -> Option<usize> {
    graph[id].nodes.iter().find_map(|node| match node {
        TensorOp::Num(n) => usize::try_from(*n).ok(),
        _ => None,
    })
}

fn numbers(graph: &Graph, ids: &[Id]) -> Option<Vec<usize>> {
    ids.iter().map(|&id| number(graph, id)).collect()
}

impl Analysis<TensorOp> for TensorAnalysis {
    type Data = Option<Vec<usize>>;

    fn make(graph: &mut Graph, node: &TensorOp) -> Self::Data {
        let shape = |id: Id| graph[id].data.clone();
        match node {
            TensorOp::Num(_) => None,
            TensorOp::Sym(s) => graph.analysis.leaves.get(s).cloned(),
            TensorOp::Neg(x) | TensorOp::Exp(x) | TensorOp::Log(x) | TensorOp::Relu(x) => shape(*x),
            TensorOp::Transpose(x) => {
                let mut result = shape(*x)?;
                let rank = result.len();
                if rank < 2 {
                    return None;
                }
                result.swap(rank - 2, rank - 1);
                Some(result)
            }
            TensorOp::Add([a, b])
            | TensorOp::Sub([a, b])
            | TensorOp::Mul([a, b])
            | TensorOp::Div([a, b])
            | TensorOp::Gt([a, b])
            | TensorOp::Mask([a, b]) => Some(broadcast_output_shape(&shape(*a)?, &shape(*b)?)),
            TensorOp::MatMul([a, b]) => {
                let a = shape(*a)?;
                let b = shape(*b)?;
                if a.len() < 2 || b.len() < 2 || a[a.len() - 1] != b[b.len() - 2] {
                    return None;
                }
                let mut result =
                    broadcast_output_shape(&a[..a.len() - 2].to_vec(), &b[..b.len() - 2].to_vec());
                result.extend([a[a.len() - 2], b[b.len() - 1]]);
                Some(result)
            }
            TensorOp::Reshape(ids) => numbers(graph, &ids[1..]),
            TensorOp::Permute(ids) => {
                let input = shape(ids[0])?;
                numbers(graph, &ids[1..])?
                    .iter()
                    .map(|&axis| input.get(axis).copied())
                    .collect()
            }
            TensorOp::BroadcastAxis([x, axis, size]) => {
                let mut result = shape(*x)?;
                *result.get_mut(number(graph, *axis)?)? = number(graph, *size)?;
                Some(result)
            }
            TensorOp::ReduceSum([x, axis]) | TensorOp::ReduceMean([x, axis]) => {
                let mut result = shape(*x)?;
                // Reductions keep dimensions in TensorExpr.
                *result.get_mut(number(graph, *axis)?)? = 1;
                Some(result)
            }
        }
    }

    fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> DidMerge {
        if to.is_none() && from.is_some() {
            *to = from;
            DidMerge(true, false)
        } else if to.is_some() && from.is_none() {
            DidMerge(false, true)
        } else {
            debug_assert!(
                from.is_none() || *to == from,
                "rewrite changed tensor shape"
            );
            DidMerge(false, false)
        }
    }

    fn modify(graph: &mut Graph, id: Id) {
        super::layout_rules::modify(graph, id);
        // Variadic layout metadata needs composition rather than fixed patterns.
        for node in graph[id].nodes.clone() {
            match node {
                TensorOp::Reshape(ids) => {
                    if graph[ids[0]].data == graph[id].data {
                        graph.union(id, ids[0]);
                    }
                    for inner in graph[ids[0]].nodes.clone() {
                        if let TensorOp::Reshape(inner) = inner {
                            let mut composed = ids.to_vec();
                            composed[0] = inner[0];
                            let next = graph.add(TensorOp::Reshape(composed.into_boxed_slice()));
                            graph.union(id, next);
                        }
                    }
                }
                TensorOp::Permute(ids) => {
                    let Some(axes) = numbers(graph, &ids[1..]) else {
                        continue;
                    };
                    if axes.iter().copied().eq(0..axes.len()) {
                        graph.union(id, ids[0]);
                    }
                    for inner in graph[ids[0]].nodes.clone() {
                        if let TensorOp::Permute(inner) = inner {
                            let mut composed = vec![inner[0]];
                            composed.extend(axes.iter().map(|&axis| inner[axis + 1]));
                            let next = graph.add(TensorOp::Permute(composed.into_boxed_slice()));
                            graph.union(id, next);
                        }
                    }
                }
                _ => {}
            }
        }
    }
}

pub(super) fn rules(numerics: RewriteNumerics) -> &'static [Rewrite<TensorOp, TensorAnalysis>] {
    static CONSERVATIVE: LazyLock<Vec<Rewrite<TensorOp, TensorAnalysis>>> =
        LazyLock::new(|| build_rules(RewriteNumerics::PreserveOperations));
    static ALGEBRAIC: LazyLock<Vec<Rewrite<TensorOp, TensorAnalysis>>> =
        LazyLock::new(|| build_rules(RewriteNumerics::RealAlgebra));
    match numerics {
        RewriteNumerics::PreserveOperations => &CONSERVATIVE,
        RewriteNumerics::RealAlgebra => &ALGEBRAIC,
    }
}

fn build_rules(numerics: RewriteNumerics) -> Vec<Rewrite<TensorOp, TensorAnalysis>> {
    let mut rules = rules_with_numerics(numerics);
    rules.extend(super::reduction_rules::rules(numerics));
    if numerics == RewriteNumerics::RealAlgebra {
        // Restrict new arithmetic identities to rank-2 matrices. In particular,
        // never move an implicitly broadcast add across a contraction.
        fn matrices(graph: &mut Graph, _: Id, subst: &Subst) -> bool {
            ["?a", "?b", "?c"].iter().all(|name| {
                let Ok(var) = name.parse() else {
                    return false;
                };
                graph[subst[var]]
                    .data
                    .as_ref()
                    .is_some_and(|shape| shape.len() == 2)
            })
        }
        fn same_shape(graph: &mut Graph, id: Id, subst: &Subst) -> bool {
            let (Ok(b), Ok(c)) = ("?b".parse(), "?c".parse()) else {
                return false;
            };
            matrices(graph, id, subst) && graph[subst[b]].data == graph[subst[c]].data
        }
        rules.extend([
            rw!("factor-matmul-left"; "(+ (matmul ?a ?b) (matmul ?a ?c))" => "(matmul ?a (+ ?b ?c))" if same_shape),
            rw!("factor-matmul-right"; "(+ (matmul ?b ?a) (matmul ?c ?a))" => "(matmul (+ ?b ?c) ?a)" if same_shape),
            rw!("associate-matmul-right"; "(matmul (matmul ?a ?b) ?c)" => "(matmul ?a (matmul ?b ?c))" if matrices),
            rw!("associate-matmul-left"; "(matmul ?a (matmul ?b ?c))" => "(matmul (matmul ?a ?b) ?c)" if matrices),
        ]);
    }
    rules
}

/// Approximate scalar work plus materialized output traffic. This is a bounded,
/// deterministic heuristic, not a hardware latency model; shared DAGs are still
/// charged as trees by egg's greedy extractor.
pub(super) struct TensorCost<'a>(pub &'a Graph);

impl CostFunction<TensorOp> for TensorCost<'_> {
    type Cost = u64;

    fn cost<C>(&mut self, node: &TensorOp, mut costs: C) -> u64
    where
        C: FnMut(Id) -> u64,
    {
        let elements = |id| {
            self.0[id].data.as_ref().map_or(1, |shape| {
                shape.iter().fold(1u64, |n, &d| {
                    n.saturating_mul(u64::try_from(d).unwrap_or(u64::MAX))
                })
            })
        };
        let own = match node {
            TensorOp::Num(_) | TensorOp::Sym(_) => 1,
            TensorOp::MatMul([a, b]) => {
                let a_shape = self.0[*a].data.as_deref().unwrap_or(&[]);
                let b_shape = self.0[*b].data.as_deref().unwrap_or(&[]);
                let n = b_shape.last().copied().unwrap_or(1);
                let a_batch = &a_shape[..a_shape.len().saturating_sub(2)];
                let b_batch = &b_shape[..b_shape.len().saturating_sub(2)];
                let batch = (0..a_batch.len().max(b_batch.len())).fold(1u64, |acc, i| {
                    let a = a_batch.iter().rev().nth(i).copied().unwrap_or(1);
                    let b = b_batch.iter().rev().nth(i).copied().unwrap_or(1);
                    acc.saturating_mul(
                        u64::try_from(if a == 1 { b } else { a }).unwrap_or(u64::MAX),
                    )
                });
                let m = a_shape.iter().rev().nth(1).copied().unwrap_or(1);
                let k = a_shape.last().copied().unwrap_or(1);
                [m, n, k]
                    .iter()
                    .fold(batch, |acc, &d| {
                        acc.saturating_mul(u64::try_from(d).unwrap_or(u64::MAX))
                    })
                    .saturating_mul(2)
                    .saturating_add(64)
            }
            TensorOp::Reshape(_) => 1, // row-major metadata operation
            TensorOp::ReduceSum([x, _]) | TensorOp::ReduceMean([x, _]) => {
                elements(*x).saturating_add(64)
            }
            TensorOp::BroadcastAxis([x, _, size]) => elements(*x)
                .saturating_mul(
                    u64::try_from(number(self.0, *size).unwrap_or(1)).unwrap_or(u64::MAX),
                )
                .saturating_add(64),
            _ => node
                .children()
                .iter()
                .map(|&id| elements(id))
                .max()
                .unwrap_or(1)
                .saturating_mul(3)
                .saturating_add(64),
        };
        node.fold(own, |sum, id| sum.saturating_add(costs(id)))
    }
}
