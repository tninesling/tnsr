//! Integer-only e-graph normalization shared by virtual maps and scheduled indices.
//!
//! Virtual maps use checked i64 arithmetic and Euclidean division. Scheduled
//! addresses use wrapping u64 arithmetic, matching PTX. Reassociation and quotient
//! reconstruction are restricted to the wrapping domain; erasing a checked
//! expression requires proof that its intermediate operations cannot overflow.
use std::collections::{BTreeSet, HashMap};

use anyhow::{Context, Result};
use egg::{
    Analysis, CostFunction, DidMerge, EGraph, Extractor, Id, Language, RecExpr, Runner, Symbol,
    define_language, rewrite,
};

use super::{Expr, IndexExpr};

define_language! {
    enum Integer {
        Num(i128),
        Leaf(Symbol),
        "+" = Add([Id; 2]),
        "-" = Sub([Id; 2]),
        "*" = Mul([Id; 2]),
        "/" = Div([Id; 2]),
        "%" = Mod([Id; 2]),
        "shr" = Shr([Id; 2]),
        "and" = And([Id; 2]),
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Semantics {
    Checked,
    Wrapping,
}

impl Semantics {
    fn limits(self) -> (i128, i128) {
        match self {
            Self::Checked => (i128::from(i64::MIN), i128::from(i64::MAX)),
            Self::Wrapping => (0, i128::from(u64::MAX)),
        }
    }
    fn constant(self, node: &Integer, a: i128, b: i128) -> Option<i128> {
        let value = match (self, node) {
            (Self::Wrapping, Integer::Add(_)) => i128::from((a as u64).wrapping_add(b as u64)),
            (Self::Wrapping, Integer::Sub(_)) => i128::from((a as u64).wrapping_sub(b as u64)),
            (Self::Wrapping, Integer::Mul(_)) => i128::from((a as u64).wrapping_mul(b as u64)),
            (_, Integer::Add(_)) => a.checked_add(b)?,
            (_, Integer::Sub(_)) => a.checked_sub(b)?,
            (_, Integer::Mul(_)) => a.checked_mul(b)?,
            (_, Integer::Div(_)) if b > 0 => a.div_euclid(b),
            (_, Integer::Mod(_)) if b > 0 => a.rem_euclid(b),
            (_, Integer::Shr(_)) if a >= 0 && (0..64).contains(&b) => a >> b,
            (_, Integer::And(_)) if a >= 0 && b >= 0 => a & b,
            _ => return None,
        };
        let (min, max) = self.limits();
        (min..=max).contains(&value).then_some(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Facts {
    constant: Option<i128>,
    bounds: (i128, i128),
    total: bool,
    dependencies: BTreeSet<Symbol>,
}

struct IntegerAnalysis {
    semantics: Semantics,
    leaf_bounds: HashMap<Symbol, (i128, i128)>,
}
impl Analysis<Integer> for IntegerAnalysis {
    type Data = Facts;
    fn make(graph: &mut EGraph<Integer, Self>, node: &Integer) -> Facts {
        let semantics = graph.analysis.semantics;
        let limits = semantics.limits();
        if let Integer::Num(value) = node {
            return Facts {
                constant: Some(*value),
                bounds: (*value, *value),
                total: true,
                dependencies: BTreeSet::new(),
            };
        }
        if let Integer::Leaf(symbol) = node {
            return Facts {
                constant: None,
                bounds: graph
                    .analysis
                    .leaf_bounds
                    .get(symbol)
                    .copied()
                    .unwrap_or(limits),
                total: true,
                dependencies: BTreeSet::from([*symbol]),
            };
        }
        let children = node.children();
        let a = &graph[children[0]].data;
        let b = &graph[children[1]].data;
        let constant = a
            .constant
            .zip(b.constant)
            .and_then(|(a, b)| semantics.constant(node, a, b));
        let exact_bounds = match node {
            Integer::Add(_) => a
                .bounds
                .0
                .checked_add(b.bounds.0)
                .zip(a.bounds.1.checked_add(b.bounds.1)),
            Integer::Sub(_) => a
                .bounds
                .0
                .checked_sub(b.bounds.1)
                .zip(a.bounds.1.checked_sub(b.bounds.0)),
            Integer::Mul(_) => {
                let products: Option<Vec<_>> = [a.bounds.0, a.bounds.1]
                    .into_iter()
                    .flat_map(|a| {
                        [b.bounds.0, b.bounds.1]
                            .into_iter()
                            .map(move |b| a.checked_mul(b))
                    })
                    .collect();
                products.map(|p| {
                    (
                        p.iter().copied().min().unwrap_or(0),
                        p.iter().copied().max().unwrap_or(0),
                    )
                })
            }
            Integer::Div(_) if b.constant.is_some_and(|b| b > 0) => b
                .constant
                .map(|b| (a.bounds.0.div_euclid(b), a.bounds.1.div_euclid(b))),
            Integer::Mod(_) if b.constant.is_some_and(|b| b > 0) => b.constant.map(|b| (0, b - 1)),
            Integer::Shr(_)
                if a.bounds.0 >= 0 && b.constant.is_some_and(|b| (0..64).contains(&b)) =>
            {
                b.constant.map(|b| (a.bounds.0 >> b, a.bounds.1 >> b))
            }
            Integer::And(_) if a.bounds.0 >= 0 && b.constant.is_some_and(|b| b >= 0) => {
                b.constant.map(|b| (0, a.bounds.1.min(b)))
            }
            _ => None,
        };
        let in_range = exact_bounds.is_some_and(|(min, max)| min >= limits.0 && max <= limits.1);
        Facts {
            constant,
            bounds: constant
                .map(|v| (v, v))
                .or(exact_bounds.filter(|_| in_range))
                .unwrap_or(limits),
            total: a.total && b.total && (semantics == Semantics::Wrapping || in_range),
            dependencies: a.dependencies.union(&b.dependencies).copied().collect(),
        }
    }
    fn merge(&mut self, to: &mut Facts, from: Facts) -> DidMerge {
        let before = to.clone();
        to.constant = to.constant.or(from.constant);
        to.bounds = (
            to.bounds.0.max(from.bounds.0),
            to.bounds.1.min(from.bounds.1),
        );
        to.total |= from.total;
        to.dependencies = to
            .dependencies
            .intersection(&from.dependencies)
            .copied()
            .collect();
        DidMerge(*to != before, *to != from)
    }
    fn modify(graph: &mut EGraph<Integer, Self>, id: Id) {
        if let Some(value) = graph[id].data.constant {
            let constant = graph.add(Integer::Num(value));
            graph.union(id, constant);
        }
        for node in graph[id].nodes.clone() {
            let children = node.children();
            if children.len() != 2 {
                continue;
            }
            let (a, b) = (graph.find(children[0]), graph.find(children[1]));
            let (ca, cb) = (graph[a].data.constant, graph[b].data.constant);
            let replacement = match &node {
                Integer::Add(_) if ca == Some(0) => Some(b),
                Integer::Add(_) | Integer::Sub(_) if cb == Some(0) => Some(a),
                Integer::Sub(_) if a == b && graph[a].data.total => {
                    Some(graph.add(Integer::Num(0)))
                }
                Integer::Mul(_) if ca == Some(1) => Some(b),
                Integer::Mul(_) if cb == Some(1) => Some(a),
                Integer::Mul(_)
                    if ca == Some(0) && graph[b].data.total
                        || cb == Some(0) && graph[a].data.total =>
                {
                    Some(graph.add(Integer::Num(0)))
                }
                Integer::Div(_) if cb == Some(1) => Some(a),
                Integer::Mod(_) if cb == Some(1) && graph[a].data.total => {
                    Some(graph.add(Integer::Num(0)))
                }
                Integer::Div(_) | Integer::Mod(_)
                    if cb.is_some_and(|d| {
                        d > 0 && graph[a].data.bounds.0 >= 0 && graph[a].data.bounds.1 < d
                    }) =>
                {
                    if matches!(node, Integer::Div(_)) && graph[a].data.total {
                        Some(graph.add(Integer::Num(0)))
                    } else if matches!(node, Integer::Mod(_)) {
                        Some(a)
                    } else {
                        None
                    }
                }
                Integer::Div(_) | Integer::Mod(_)
                    if graph[a].data.bounds.0 >= 0
                        && cb.is_some_and(|d| d > 1 && (d as u64).is_power_of_two()) =>
                {
                    let divisor = cb.unwrap_or(1) as u64;
                    let operand = graph.add(Integer::Num(if matches!(node, Integer::Div(_)) {
                        i128::from(divisor.trailing_zeros())
                    } else {
                        i128::from(divisor - 1)
                    }));
                    Some(graph.add(if matches!(node, Integer::Div(_)) {
                        Integer::Shr([a, operand])
                    } else {
                        Integer::And([a, operand])
                    }))
                }
                Integer::Mod(_) => {
                    graph[a]
                        .nodes
                        .clone()
                        .into_iter()
                        .find_map(|inner| match inner {
                            Integer::Mod([_, d]) if graph.find(d) == b => Some(a),
                            _ => None,
                        })
                }
                Integer::Div(_) => {
                    let mut result = None;
                    for inner in graph[a].nodes.clone() {
                        if let Integer::Div([x, d]) = inner
                            && let Some(product) = graph[d]
                                .data
                                .constant
                                .zip(cb)
                                .and_then(|(d, b)| d.checked_mul(b))
                                .filter(|p| *p > 0 && *p <= graph.analysis.semantics.limits().1)
                        {
                            let d = graph.add(Integer::Num(product));
                            result = Some(graph.add(Integer::Div([x, d])));
                        }
                    }
                    result
                }
                _ => None,
            };
            if let Some(replacement) = replacement {
                graph.union(id, replacement);
            }
        }
    }
}

struct AddressCost;
impl CostFunction<Integer> for AddressCost {
    type Cost = usize;
    fn cost<C: FnMut(Id) -> usize>(&mut self, node: &Integer, mut costs: C) -> usize {
        let own = match node {
            Integer::Div(_) | Integer::Mod(_) => 80,
            Integer::Mul(_) => 3,
            Integer::Leaf(_) | Integer::Num(_) => 0,
            _ => 1,
        };
        node.fold(own, |sum, id| sum.saturating_add(costs(id)))
    }
}

fn extract(
    input: RecExpr<Integer>,
    semantics: Semantics,
    leaf_bounds: HashMap<Symbol, (i128, i128)>,
) -> RecExpr<Integer> {
    let mut rules = Vec::new();
    if semantics == Semantics::Wrapping {
        rules.push(rewrite!("integer-add-commute"; "(+ ?a ?b)" => "(+ ?b ?a)"));
        rules.push(rewrite!("integer-mul-commute"; "(* ?a ?b)" => "(* ?b ?a)"));
        rules
            .push(rewrite!("integer-quotient-remainder"; "(+ (* (/ ?x ?d) ?d) (% ?x ?d))" => "?x"));
    }
    let runner = Runner::<Integer, IntegerAnalysis, ()>::new(IntegerAnalysis {
        semantics,
        leaf_bounds,
    })
    .with_iter_limit(4)
    .with_node_limit(2048)
    .with_expr(&input)
    .run(&rules);
    Extractor::new(&runner.egraph, AddressCost)
        .find_best(runner.roots[0])
        .1
}

pub(super) fn normalize_map(expression: IndexExpr) -> Result<IndexExpr> {
    normalize_map_in_domain(expression, &[])
}

/// Valid only within the supplied logical domain. This removes reshape wrapping
/// before substituting scheduled coordinates, whose edge loads are predicated.
pub(super) fn normalize_map_in_domain(expression: IndexExpr, shape: &[usize]) -> Result<IndexExpr> {
    fn encode(expr: &IndexExpr, out: &mut RecExpr<Integer>) -> Result<Id> {
        let node = match expr {
            IndexExpr::Const(v) => Integer::Num(i128::from(*v)),
            IndexExpr::IterDim(v) => Integer::Leaf(Symbol::from(format!("dim_{v}"))),
            IndexExpr::Symbol(v) => Integer::Leaf(Symbol::from(format!("symbol_{v}"))),
            IndexExpr::Add(a, b) => Integer::Add([encode(a, out)?, encode(b, out)?]),
            IndexExpr::Sub(a, b) => Integer::Sub([encode(a, out)?, encode(b, out)?]),
            IndexExpr::Mul(a, b) => Integer::Mul([encode(a, out)?, encode(b, out)?]),
            IndexExpr::FloorDiv(a, d) | IndexExpr::Mod(a, d) => {
                anyhow::ensure!(*d > 0, "index divisor/modulus must be positive");
                let ids = [encode(a, out)?, out.add(Integer::Num(i128::from(*d)))];
                if matches!(expr, IndexExpr::FloorDiv(..)) {
                    Integer::Div(ids)
                } else {
                    Integer::Mod(ids)
                }
            }
        };
        Ok(out.add(node))
    }
    fn decode(expr: &RecExpr<Integer>, id: Id) -> Result<IndexExpr> {
        let number = |id| match expr[id] {
            Integer::Num(v) => i64::try_from(v).context("normalized map constant exceeds i64"),
            _ => anyhow::bail!("map divisor must be constant"),
        };
        Ok(match &expr[id] {
            Integer::Num(v) => IndexExpr::Const(i64::try_from(*v)?),
            Integer::Leaf(v) => {
                let text = v.as_str();
                if let Some(v) = text.strip_prefix("dim_") {
                    IndexExpr::IterDim(v.parse()?)
                } else {
                    IndexExpr::Symbol(
                        text.strip_prefix("symbol_")
                            .context("unknown map leaf")?
                            .parse()?,
                    )
                }
            }
            Integer::Add([a, b]) => {
                IndexExpr::Add(Box::new(decode(expr, *a)?), Box::new(decode(expr, *b)?))
            }
            Integer::Sub([a, b]) => {
                IndexExpr::Sub(Box::new(decode(expr, *a)?), Box::new(decode(expr, *b)?))
            }
            Integer::Mul([a, b]) => {
                IndexExpr::Mul(Box::new(decode(expr, *a)?), Box::new(decode(expr, *b)?))
            }
            Integer::Div([a, d]) => IndexExpr::FloorDiv(Box::new(decode(expr, *a)?), number(*d)?),
            Integer::Mod([a, d]) => IndexExpr::Mod(Box::new(decode(expr, *a)?), number(*d)?),
            Integer::Shr([a, d]) => IndexExpr::FloorDiv(
                Box::new(decode(expr, *a)?),
                1_i64
                    .checked_shl(u32::try_from(number(*d)?)?)
                    .context("map shift divisor exceeds i64")?,
            ),
            Integer::And([a, d]) => IndexExpr::Mod(
                Box::new(decode(expr, *a)?),
                number(*d)?
                    .checked_add(1)
                    .context("map mask modulus exceeds i64")?,
            ),
        })
    }
    let mut input = RecExpr::default();
    encode(&expression, &mut input)?;
    let mut bounds = HashMap::new();
    for (dimension, &extent) in shape.iter().enumerate() {
        if extent > 0 {
            let upper =
                i64::try_from(extent - 1).context("logical domain exceeds signed index range")?;
            bounds.insert(
                Symbol::from(format!("dim_{dimension}")),
                (0, i128::from(upper)),
            );
        }
    }
    let result = extract(input, Semantics::Checked, bounds);
    decode(&result, Id::from(result.as_ref().len() - 1))
}

pub(super) fn normalize_address(expression: &Expr) -> Result<Expr> {
    fn encode(
        expr: &Expr,
        out: &mut RecExpr<Integer>,
        leaves: &mut HashMap<Symbol, Expr>,
    ) -> Result<Id> {
        let node = match expr {
            Expr::Const(v) => Integer::Num(i128::from(*v as u64)),
            Expr::Add(a, b) => Integer::Add([encode(a, out, leaves)?, encode(b, out, leaves)?]),
            Expr::Sub(a, b) => Integer::Sub([encode(a, out, leaves)?, encode(b, out, leaves)?]),
            Expr::Mul(a, b) => Integer::Mul([encode(a, out, leaves)?, encode(b, out, leaves)?]),
            Expr::FloorDiv(a, d) | Expr::Mod(a, d) => {
                anyhow::ensure!(*d > 0, "scheduled index divisor/modulus must be positive");
                let ids = [
                    encode(a, out, leaves)?,
                    out.add(Integer::Num(i128::try_from(*d)?)),
                ];
                if matches!(expr, Expr::FloorDiv(..)) {
                    Integer::Div(ids)
                } else {
                    Integer::Mod(ids)
                }
            }
            Expr::ShiftRight(a, d) | Expr::BitAnd(a, d) => {
                anyhow::ensure!(
                    !matches!(expr, Expr::ShiftRight(..)) || *d < 64,
                    "scheduled index shift exceeds u64 width"
                );
                let ids = [
                    encode(a, out, leaves)?,
                    out.add(Integer::Num(i128::from(*d))),
                ];
                if matches!(expr, Expr::ShiftRight(..)) {
                    Integer::Shr(ids)
                } else {
                    Integer::And(ids)
                }
            }
            leaf => {
                let symbol = Symbol::from(format!("{leaf:?}"));
                leaves.insert(symbol, leaf.clone());
                Integer::Leaf(symbol)
            }
        };
        Ok(out.add(node))
    }
    fn decode(expr: &RecExpr<Integer>, id: Id, leaves: &HashMap<Symbol, Expr>) -> Result<Expr> {
        let number = |id| match expr[id] {
            Integer::Num(v) => u64::try_from(v).context("normalized address constant exceeds u64"),
            _ => anyhow::bail!("address divisor must be constant"),
        };
        Ok(match &expr[id] {
            Integer::Num(v) => Expr::Const(u64::try_from(*v)? as i64),
            Integer::Leaf(v) => leaves.get(v).cloned().context("unknown address leaf")?,
            Integer::Add([a, b]) => {
                let (mut a, mut b) = (decode(expr, *a, leaves)?, decode(expr, *b, leaves)?);
                if b < a {
                    std::mem::swap(&mut a, &mut b);
                }
                Expr::Add(Box::new(a), Box::new(b))
            }
            Integer::Sub([a, b]) => Expr::Sub(
                Box::new(decode(expr, *a, leaves)?),
                Box::new(decode(expr, *b, leaves)?),
            ),
            Integer::Mul([a, b]) => {
                let (mut a, mut b) = (decode(expr, *a, leaves)?, decode(expr, *b, leaves)?);
                if b < a {
                    std::mem::swap(&mut a, &mut b);
                }
                Expr::Mul(Box::new(a), Box::new(b))
            }
            Integer::Div([a, d]) => Expr::FloorDiv(
                Box::new(decode(expr, *a, leaves)?),
                usize::try_from(number(*d)?)?,
            ),
            Integer::Mod([a, d]) => {
                // Export the quotient explicitly so a separate quotient use and
                // remainder can share one binding. For unsigned x and d > 0,
                // (x / d) * d <= x, so this representation cannot overflow.
                let value = decode(expr, *a, leaves)?;
                let divisor = number(*d)?;
                Expr::Sub(
                    Box::new(value.clone()),
                    Box::new(Expr::Mul(
                        Box::new(Expr::FloorDiv(Box::new(value), usize::try_from(divisor)?)),
                        Box::new(Expr::Const(divisor as i64)),
                    )),
                )
            }
            Integer::Shr([a, d]) => {
                Expr::ShiftRight(Box::new(decode(expr, *a, leaves)?), number(*d)?)
            }
            Integer::And([a, d]) => Expr::BitAnd(Box::new(decode(expr, *a, leaves)?), number(*d)?),
        })
    }
    let mut input = RecExpr::default();
    let mut leaves = HashMap::new();
    encode(expression, &mut input, &mut leaves)?;
    let result = extract(input, Semantics::Wrapping, HashMap::new());
    decode(&result, Id::from(result.as_ref().len() - 1), &leaves)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    fn evaluate(expr: &Expr, x: u64) -> u64 {
        match expr {
            Expr::Const(v) => *v as u64,
            Expr::Var(_) => x,
            Expr::Add(a, b) => evaluate(a, x).wrapping_add(evaluate(b, x)),
            Expr::Sub(a, b) => evaluate(a, x).wrapping_sub(evaluate(b, x)),
            Expr::Mul(a, b) => evaluate(a, x).wrapping_mul(evaluate(b, x)),
            Expr::FloorDiv(a, d) => evaluate(a, x) / *d as u64,
            Expr::Mod(a, d) => evaluate(a, x) % *d as u64,
            Expr::ShiftRight(a, d) => evaluate(a, x) >> d,
            Expr::BitAnd(a, d) => evaluate(a, x) & d,
            _ => unreachable!(),
        }
    }
    #[test]
    fn unsigned_rewrites_preserve_wrapping_boundary_values() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(61);
        for _ in 0..128 {
            let mut expr = Expr::Var("x".into());
            for _ in 0..4 {
                let literal = Box::new(Expr::Const(rng.random::<i64>()));
                let divisor = [1, 2, 3, 7, 16, usize::MAX][rng.random_range(0..6)];
                expr = match rng.random_range(0..5) {
                    0 => Expr::Add(Box::new(expr), literal),
                    1 => Expr::Sub(Box::new(expr), literal),
                    2 => Expr::Mul(Box::new(expr), literal),
                    3 => Expr::FloorDiv(Box::new(expr), divisor),
                    _ => Expr::Mod(Box::new(expr), divisor),
                };
            }
            let optimized = normalize_address(&expr).unwrap();
            for x in [0, 1, 2, 3, 17, u64::MAX / 2, u64::MAX, rng.random()] {
                assert_eq!(
                    evaluate(&expr, x),
                    evaluate(&optimized, x),
                    "{expr:?} -> {optimized:?}, x={x}"
                );
            }
        }
    }
    #[test]
    fn signed_maps_preserve_negative_values_and_overflow_errors() {
        let x = IndexExpr::IterDim(0);
        for divisor in [1, 2, 3, 8, i64::MAX] {
            for original in [
                IndexExpr::FloorDiv(Box::new(x.clone()), divisor),
                IndexExpr::Mod(Box::new(x.clone()), divisor),
            ] {
                let optimized = original.clone().normalize().unwrap();
                for value in [i64::MIN, -17, -1, 0, 1, 17, i64::MAX] {
                    assert_eq!(
                        original.evaluate(&[value], &[]).unwrap(),
                        optimized.evaluate(&[value], &[]).unwrap()
                    );
                }
            }
        }
        let overflowing = IndexExpr::Add(Box::new(x), Box::new(IndexExpr::Const(i64::MAX)));
        for original in [
            IndexExpr::Mul(Box::new(overflowing.clone()), Box::new(IndexExpr::Const(0))),
            IndexExpr::Sub(Box::new(overflowing.clone()), Box::new(overflowing.clone())),
            IndexExpr::Mod(Box::new(overflowing), 1),
        ] {
            let optimized = original.clone().normalize().unwrap();
            assert!(original.evaluate(&[1], &[]).is_err());
            assert!(
                optimized.evaluate(&[1], &[]).is_err(),
                "erased checked overflow: {optimized:?}"
            );
            assert_eq!(
                original.evaluate(&[-1], &[]).unwrap(),
                optimized.evaluate(&[-1], &[]).unwrap()
            );
        }
    }
    #[test]
    fn identity_folding_strength_reduction_and_quotient_reconstruction() {
        let x = Expr::Var("x".into());
        assert_eq!(
            normalize_address(&Expr::FloorDiv(Box::new(x.clone()), 1)).unwrap(),
            x
        );
        assert_eq!(
            normalize_address(&Expr::FloorDiv(Box::new(x.clone()), 8)).unwrap(),
            Expr::ShiftRight(Box::new(x.clone()), 3)
        );
        assert_eq!(
            normalize_address(&Expr::Mod(Box::new(x.clone()), 8)).unwrap(),
            Expr::BitAnd(Box::new(x.clone()), 7)
        );
        let reconstruct = Expr::Add(
            Box::new(Expr::FloorDiv(Box::new(x.clone()), 3) * 3_usize),
            Box::new(Expr::Mod(Box::new(x.clone()), 3)),
        );
        assert_eq!(normalize_address(&reconstruct).unwrap(), x);
        let nested = Expr::FloorDiv(Box::new(Expr::FloorDiv(Box::new(x.clone()), 3)), 5);
        assert_eq!(
            normalize_address(&nested).unwrap(),
            Expr::FloorDiv(Box::new(x), 15)
        );
    }
    #[test]
    fn logical_domain_bounds_remove_reshape_wraps_without_changing_unbounded_maps() {
        let quotient = IndexExpr::FloorDiv(
            Box::new(IndexExpr::Add(
                Box::new(IndexExpr::Mul(
                    Box::new(IndexExpr::IterDim(0)),
                    Box::new(IndexExpr::Const(5)),
                )),
                Box::new(IndexExpr::IterDim(1)),
            )),
            5,
        );
        let original = IndexExpr::Mod(Box::new(quotient.clone()), 3);
        let bounded = normalize_map_in_domain(original.clone(), &[3, 5]).unwrap();
        assert_eq!(bounded, quotient);
        for row in 0..3 {
            for col in 0..5 {
                assert_eq!(
                    original.evaluate(&[row, col], &[]).unwrap(),
                    bounded.evaluate(&[row, col], &[]).unwrap()
                );
            }
        }
        // Bounds are local to this use; general map normalization stays signed.
        let unbounded = normalize_map(original.clone()).unwrap();
        assert_eq!(
            unbounded.evaluate(&[-1, 0], &[]).unwrap(),
            original.evaluate(&[-1, 0], &[]).unwrap()
        );
    }

    #[test]
    fn invalid_divisors_are_rejected_before_dead_expression_folding() {
        let invalid = Expr::Mul(
            Box::new(Expr::FloorDiv(Box::new(Expr::Const(1)), 0)),
            Box::new(Expr::Const(0)),
        );
        assert!(normalize_address(&invalid).is_err());
        assert!(
            IndexExpr::Mod(Box::new(IndexExpr::Const(0)), 0)
                .normalize()
                .is_err()
        );
        assert!(
            IndexExpr::FloorDiv(Box::new(IndexExpr::Const(0)), -1)
                .normalize()
                .is_err()
        );
        assert!(normalize_address(&Expr::ShiftRight(Box::new(Expr::Const(0)), 64)).is_err());
    }
}
