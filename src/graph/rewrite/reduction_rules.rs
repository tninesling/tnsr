//! Shape-guarded reduction and broadcast identities.
use super::tensor_analysis::TensorAnalysis;
use super::{RewriteNumerics, TensorOp};
use egg::{EGraph, Id, Rewrite, Subst, rewrite as rw};

type Graph = EGraph<TensorOp, TensorAnalysis>;

fn variable(subst: &Subst, name: &str) -> Option<Id> {
    Some(subst[name.parse().ok()?])
}

fn number(graph: &Graph, id: Id) -> Option<usize> {
    graph[id].nodes.iter().find_map(|node| match node {
        TensorOp::Num(value) => usize::try_from(*value).ok(),
        _ => None,
    })
}

fn matrices(graph: &mut Graph, _: Id, subst: &Subst) -> bool {
    ["?a", "?b"].iter().all(|name| {
        variable(subst, name)
            .and_then(|id| graph[id].data.as_ref())
            .is_some_and(|shape| shape.len() == 2 && !shape.contains(&0))
    })
}

fn nonempty_broadcast(graph: &mut Graph, _: Id, subst: &Subst) -> bool {
    let Some(size) = variable(subst, "?size").and_then(|id| number(graph, id)) else {
        return false;
    };
    let Some(axis) = variable(subst, "?axis").and_then(|id| number(graph, id)) else {
        return false;
    };
    let Some(source) = variable(subst, "?x").and_then(|id| graph[id].data.as_ref()) else {
        return false;
    };
    size > 0 && source.get(axis) == Some(&1)
}

fn invariant_factor(graph: &mut Graph, _: Id, subst: &Subst) -> bool {
    let Some(axis) = variable(subst, "?axis").and_then(|id| number(graph, id)) else {
        return false;
    };
    let Some(x) = variable(subst, "?x").and_then(|id| graph[id].data.as_ref()) else {
        return false;
    };
    let Some(y) = variable(subst, "?y").and_then(|id| graph[id].data.as_ref()) else {
        return false;
    };
    // Equal rank makes the literal axis refer to the same dimension in both
    // inputs. Other dimensions already satisfy the original product's broadcast
    // contract. Exclude empty reductions rather than inventing their identities.
    x.len() == y.len() && x.get(axis).is_some_and(|&size| size > 0) && y.get(axis) == Some(&1)
}

pub(super) fn rules(numerics: RewriteNumerics) -> Vec<Rewrite<TensorOp, TensorAnalysis>> {
    // Each output element applies the identical unary operation to the identical
    // input value. This changes duplicated work, not arithmetic grouping.
    let mut rules = vec![
        rw!("neg-before-broadcast"; "(neg (broadcast ?x ?axis ?size))" => "(broadcast (neg ?x) ?axis ?size)"),
        rw!("exp-before-broadcast"; "(exp (broadcast ?x ?axis ?size))" => "(broadcast (exp ?x) ?axis ?size)"),
        rw!("log-before-broadcast"; "(log (broadcast ?x ?axis ?size))" => "(broadcast (log ?x) ?axis ?size)"),
        rw!("relu-before-broadcast"; "(relu (broadcast ?x ?axis ?size))" => "(broadcast (relu ?x) ?axis ?size)"),
    ];
    if numerics == RewriteNumerics::RealAlgebra {
        rules.extend([
            // This reverses materialization of a repeated row/column followed
            // by its mean. Preserve the executor's empty-axis contract by
            // requiring a positive extent.
            rw!("mean-broadcast-axis"; "(reduce_mean (broadcast ?x ?axis ?size) ?axis)" => "?x" if nonempty_broadcast),
            // Sum/mean are linear over the noncontracting matmul dimensions.
            // Restrict to rank 2 until batch-axis semantics are modeled here.
            rw!("sum-matmul-rows"; "(reduce_sum (matmul ?a ?b) 0)" => "(matmul (reduce_sum ?a 0) ?b)" if matrices),
            rw!("sum-matmul-cols"; "(reduce_sum (matmul ?a ?b) 1)" => "(matmul ?a (reduce_sum ?b 1))" if matrices),
            rw!("mean-matmul-rows"; "(reduce_mean (matmul ?a ?b) 0)" => "(matmul (reduce_mean ?a 0) ?b)" if matrices),
            rw!("mean-matmul-cols"; "(reduce_mean (matmul ?a ?b) 1)" => "(matmul ?a (reduce_mean ?b 1))" if matrices),
            rw!("sum-invariant-factor"; "(reduce_sum (* ?x ?y) ?axis)" => "(* (reduce_sum ?x ?axis) ?y)" if invariant_factor),
            rw!("mean-invariant-factor"; "(reduce_mean (* ?x ?y) ?axis)" => "(* (reduce_mean ?x ?axis) ?y)" if invariant_factor),
        ]);
    }
    rules
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::TensorExpr;
    use egg::{RecExpr, Runner};

    #[test]
    fn mean_does_not_cancel_broadcast_of_a_non_singleton_axis() {
        // Test the IR guard directly: the public TensorExpr constructor currently
        // rejects a non-singleton BroadcastAxis, but equality facts must remain
        // sound if another IR producer permits a same-size broadcast.
        let source = TensorExpr::<f32>::input("x", vec![4, 3]);
        let mut graph = Graph::new(TensorAnalysis::new(&source));
        let expr: RecExpr<TensorOp> = "(reduce_mean (broadcast input_x 0 4) 0)".parse().unwrap();
        let root = graph.add_expr(&expr);
        let source = graph.lookup(TensorOp::Sym("input_x".into())).unwrap();
        let runner = Runner::<_, _, ()>::default()
            .with_egraph(graph)
            .run(&rules(RewriteNumerics::RealAlgebra));
        assert_ne!(runner.egraph.find(root), runner.egraph.find(source));
        assert_eq!(runner.egraph[root].data.as_deref(), Some([1, 3].as_slice()));
    }
}
