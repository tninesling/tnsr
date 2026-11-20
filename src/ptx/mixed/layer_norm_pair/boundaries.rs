//! Evidence from the ORIGINAL selected F32 plan, never from tensor-source names
//! or a newly replanned candidate. The caller must supply the actual original
//! mixed-region NodeIndex -> plain-program mapping, physical input descriptors,
//! reduction policy and fusion scorer. A global graph output can still feed an
//! in-kernel consumer through a register: classify edges, not saved-node names.
use super::*;
use crate::ptx::{PtxExecutionPlan, PtxPlanAction};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Boundary {
    SameValue,
    SameKernel {
        step: usize,
        action: PtxPlanAction,
    },
    Stored {
        producer_step: usize,
        consumer_step: usize,
    },
    /// The pair spans multiple original mixed F32 programs. Merge evidence from
    /// those ORIGINAL programs; absence is not permission to guess a checkpoint.
    OutsideProgram,
}
#[derive(Clone, Debug)]
pub(super) struct Edge {
    pub label: &'static str,
    pub producer: NodeIndex,
    pub consumer: NodeIndex,
    pub boundary: Boundary,
}
fn physical_source(plan: &PtxExecutionPlan, mut n: NodeIndex) -> Result<NodeIndex> {
    for _ in 0..=plan.steps().len() {
        match plan.virtual_value(n) {
            Some(v) if v.source != n => n = v.source,
            _ => return Ok(n),
        }
    }
    anyhow::bail!("cycle in original LayerNorm plan virtual source")
}
/// Distinct arithmetic steps communicate through materialized F32 bindings.
/// Within a step, a simultaneously exported result is NOT a forced load/store
/// boundary for another operation in that same compiled function.
fn classify(plan: &PtxExecutionPlan, a: NodeIndex, b: NodeIndex) -> Result<Boundary> {
    let a = physical_source(plan, a)?;
    let b = physical_source(plan, b)?;
    if a == b {
        return Ok(Boundary::SameValue);
    }
    let owner = |n| {
        plan.steps()
            .iter()
            .enumerate()
            .find(|(_, s)| s.action != PtxPlanAction::VirtualView && s.members.contains(&n))
    };
    let Some((ai, producer)) = owner(a) else {
        anyhow::bail!("original LayerNorm producer has no physical step");
    };
    let Some((bi, consumer)) = owner(b) else {
        anyhow::bail!("original LayerNorm consumer has no physical step");
    };
    // A saved value appearing only as a region input says nothing about its
    // original producer edge. Only the program computing the consumer supplies
    // evidence; otherwise independent Upload owners invent a false checkpoint.
    if consumer.action == PtxPlanAction::Upload {
        return Ok(Boundary::OutsideProgram);
    }
    if ai == bi {
        return Ok(Boundary::SameKernel {
            step: ai,
            action: consumer.action,
        });
    }
    ensure!(
        producer.outputs.contains(&a),
        "cross-kernel LayerNorm value is not an original output"
    );
    Ok(Boundary::Stored {
        producer_step: ai,
        consumer_step: bi,
    })
}
fn node(pair: &LayerNormPair, id: usize) -> Option<NodeIndex> {
    pair.program.get(id).map(|op| op.node)
}
fn operation(pair: &LayerNormPair, n: NodeIndex) -> Option<&Operation> {
    pair.program.iter().find(|op| op.node == n)
}
fn arg(pair: &LayerNormPair, n: NodeIndex, i: usize) -> Option<NodeIndex> {
    node(pair, *operation(pair, n)?.inputs.get(i)?)
}
fn broadcast_source(pair: &LayerNormPair, mut n: NodeIndex) -> Option<NodeIndex> {
    while matches!(operation(pair, n)?.kind, Kind::Broadcast(_) | Kind::Reshape) {
        n = arg(pair, n, 0)?;
    }
    Some(n)
}
/// Query exact primitive edges. A future dispatcher can activate only after all
/// required edges are covered by original program evidence and PTX/SASS/device
/// checks validate the preserved opportunities. This function does not run or
/// change a plan, and does not assert any untested tail policy.
fn queries(pair: &LayerNormPair) -> Option<Vec<(&'static str, NodeIndex, NodeIndex)>> {
    let cp = ssa::checkpoint_candidates(pair)?;
    let cc2 = arg(pair, pair.backward, 0)?;
    let cc1 = arg(pair, cc2, 0)?;
    let ds = arg(pair, cp.first_square_vjp, 0)?;
    let dv = broadcast_source(pair, arg(pair, ds, 0)?)?;
    let q = arg(pair, dv, 0)?;
    let dm = broadcast_source(pair, arg(pair, cp.mean_term, 0)?)?;
    let negative = arg(pair, dm, 0)?;
    let c = arg(pair, pair.normalized, 0)?;
    let vb = arg(pair, pair.variance_plus_epsilon, 0)?;
    let variance = broadcast_source(pair, vb)?;
    let square = arg(pair, variance, 0)?;
    let affine = arg(pair, pair.forward, 0)?;
    let log = pair
        .program
        .iter()
        .find(|op| {
            op.kind == Kind::Unary(UnaryOp::Log)
                && arg(pair, op.node, 0) == Some(pair.variance_plus_epsilon)
        })?
        .node;
    Some(vec![
        ("mean -> center", pair.mean, c),
        ("center -> square", c, square),
        ("square -> variance Mean", square, variance),
        (
            "variance Mean -> epsilon Add",
            variance,
            pair.variance_plus_epsilon,
        ),
        ("epsilon Add -> Log", pair.variance_plus_epsilon, log),
        (
            "inverse std -> normalized",
            pair.inverse_std,
            pair.normalized,
        ),
        ("normalized -> affine Mul", pair.normalized, affine),
        ("affine Mul -> bias Add", affine, pair.forward),
        ("quotient -> variance-gradient Sum", q, dv),
        ("variance-gradient Sum -> dS scale", dv, ds),
        ("dS scale -> first square VJP", ds, cp.first_square_vjp),
        ("dS scale -> second square VJP", ds, cp.second_square_vjp),
        ("direct center VJP -> first Add", cp.direct, cc1),
        ("first square VJP -> first Add", cp.first_square_vjp, cc1),
        ("second square VJP -> second Add", cp.second_square_vjp, cc2),
        ("negative center VJP -> mean-gradient Sum", negative, dm),
        ("mean-gradient Sum -> mean scale", dm, cp.mean_term),
        ("mean scale -> inner dZ Add", cp.mean_term, pair.backward),
    ])
}
/// Prove the original four row collectives, including their lane count. A
/// consumer appearing only as a boundary Upload provides no tree evidence.
/// Returning None rejects a computed but incompatible original row schedule.
pub(super) fn row_trees_from_original_plan(
    plan: &PtxExecutionPlan,
    original_to_plain: &HashMap<NodeIndex, NodeIndex>,
    pair: &LayerNormPair,
) -> Result<Option<HashSet<NodeIndex>>> {
    let edges = queries(pair).context("LayerNorm ordered SSA anchors changed")?;
    let consumer = |label| edges.iter().find(|edge| edge.0 == label).map(|edge| edge.2);
    let anchors = [
        (pair.mean, ReduceOp::Mean),
        (
            consumer("square -> variance Mean").context("LN variance anchor missing")?,
            ReduceOp::Mean,
        ),
        (
            consumer("quotient -> variance-gradient Sum")
                .context("LN variance VJP anchor missing")?,
            ReduceOp::Sum,
        ),
        (
            consumer("negative center VJP -> mean-gradient Sum")
                .context("LN mean VJP anchor missing")?,
            ReduceOp::Sum,
        ),
    ];
    let mut proved = HashSet::new();
    for (original, expected_op) in anchors {
        let Some(&plain) = original_to_plain.get(&original) else {
            continue;
        };
        let plain = physical_source(plan, plain)?;
        let Some(step) = plan.steps().iter().find(|step| {
            !matches!(
                step.action,
                PtxPlanAction::Upload | PtxPlanAction::VirtualView
            ) && step.members.contains(&plain)
        }) else {
            continue;
        };
        let PtxPlanAction::ReductionRegion(id) = step.action else {
            return Ok(None);
        };
        let Some(region) = plan.reduction_regions().get(id) else {
            return Ok(None);
        };
        if region.schedule != (crate::tile::ReductionSchedule::Block { threads: 128 })
            || region.axis != 2
            || region.input_shape != pair.shape
            || region.op != expected_op
        {
            return Ok(None);
        }
        proved.insert(original);
    }
    Ok(Some(proved))
}
pub(super) fn from_original_plan(
    plan: &PtxExecutionPlan,
    original_to_plain: &HashMap<NodeIndex, NodeIndex>,
    pair: &LayerNormPair,
) -> Result<Vec<Edge>> {
    queries(pair)
        .context("LayerNorm ordered SSA anchors changed")?
        .into_iter()
        .map(|(label, producer, consumer)| {
            let boundary = match (
                original_to_plain.get(&producer),
                original_to_plain.get(&consumer),
            ) {
                (Some(&a), Some(&b)) => classify(plan, a, b)?,
                _ => Boundary::OutsideProgram,
            };
            Ok(Edge {
                label,
                producer,
                consumer,
                boundary,
            })
        })
        .collect()
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::ptx::PtxReductionMode;
    #[test]
    fn saved_output_is_not_a_barrier_for_an_in_kernel_consumer() {
        let shape = vec![2, 3];
        let u =
            TensorExpr::<f32>::input("a", shape.clone()) * TensorExpr::input("b", shape.clone());
        let p = TensorExpr::<f32>::input("c", shape.clone()) * TensorExpr::input("d", shape);
        let cc = u.clone() + p;
        // u escapes as a global region output AND feeds cc within that region.
        let graph: TensorGraph<f32> = (-TensorExpr::concat(&[cc, u], 0)).into();
        let plan = PtxExecutionPlan::build_with_reduction_mode(
            &graph,
            PtxReductionMode::DeterministicTree,
        )
        .unwrap();
        let u = graph
            .graph
            .node_indices()
            .find(|n| {
                matches!(
                    graph[*n],
                    TensorGraphNode::Binary {
                        op: BinaryOp::Mul,
                        ..
                    }
                ) && graph
                    .inputs(*n)
                    .iter()
                    .any(|i| matches!(graph[*i], TensorGraphNode::Input { name: "a", .. }))
            })
            .unwrap();
        let cc = graph
            .graph
            .node_indices()
            .find(|n| {
                matches!(
                    graph[*n],
                    TensorGraphNode::Binary {
                        op: BinaryOp::Add,
                        ..
                    }
                )
            })
            .unwrap();
        let concat = graph
            .graph
            .node_indices()
            .find(|n| matches!(graph[*n], TensorGraphNode::Concat { .. }))
            .unwrap();
        let neg = graph
            .graph
            .node_indices()
            .find(|n| {
                matches!(
                    graph[*n],
                    TensorGraphNode::Unary {
                        op: UnaryOp::Neg,
                        ..
                    }
                )
            })
            .unwrap();
        let inputs = graph
            .graph
            .node_indices()
            .filter(|n| matches!(graph[*n], TensorGraphNode::Input { .. }))
            .collect::<Vec<_>>();
        assert_eq!(
            classify(&plan, inputs[0], inputs[1]).unwrap(),
            Boundary::OutsideProgram
        );
        assert!(plan.steps().iter().any(|s| s.outputs.contains(&u)));
        assert!(matches!(
            classify(&plan, u, cc).unwrap(),
            Boundary::SameKernel { .. }
        ));
        assert!(matches!(
            classify(&plan, concat, neg).unwrap(),
            Boundary::Stored { .. }
        ));
    }
}

#[cfg(test)]
mod row_tree_tests {
    use super::*;
    use crate::ptx::PtxReductionMode;
    #[test]
    fn original_row_tree_requires_four_block128_anchors_at_both_widths() {
        for (batch, sequence, width) in
            [(1, 512, 512), (1, 512, 1024), (2, 128, 512), (2, 128, 1024)]
        {
            let shape = vec![batch, sequence, width];
            let x = Parameter::new(vec![0.; shape.iter().product()], shape.clone());
            let gamma = Parameter::new(vec![1.; width], vec![width]);
            let beta = Parameter::new(vec![0.; width], vec![width]);
            let expression = crate::nn::layer_norm(
                TensorExpr::from(x).cast(ValueDType::F32),
                2,
                gamma,
                beta,
                1e-5,
            );
            let mut graph = TensorGraph::<f32>::new();
            let y = graph.add_expr(&expression, false);
            let loss = graph.add_expr(
                &(TensorExpr::node_ref(y, shape.clone()) * TensorExpr::input("incoming", shape))
                    .reduce_axis_sum(2)
                    .reduce_axis_sum(1)
                    .reduce_axis_sum(0),
                false,
            );
            let graph = graph.with_gradients(loss);
            let pair = recognize(&graph).unwrap().remove(0);
            let mapped = graph.graph.node_indices().map(|n| (n, n)).collect();
            let plan = PtxExecutionPlan::build_with_reduction_mode(
                &graph,
                PtxReductionMode::DeterministicTree,
            )
            .unwrap();
            assert_eq!(
                row_trees_from_original_plan(&plan, &mapped, &pair)
                    .unwrap()
                    .unwrap()
                    .len(),
                4
            );
            let serial =
                PtxExecutionPlan::build_with_reduction_mode(&graph, PtxReductionMode::Strict)
                    .unwrap();
            assert!(
                row_trees_from_original_plan(&serial, &mapped, &pair)
                    .unwrap()
                    .is_none()
            );
        }
    }
}
