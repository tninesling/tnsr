//! Repair only fusion units participating in a quotient-graph cycle.
use super::*;

#[derive(Debug)]
pub(super) struct CycleMembers(pub HashSet<NodeIndex>);
impl Display for CycleMembers {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} graph nodes participate in cyclic fusion units",
            self.0.len()
        )
    }
}
impl Error for CycleMembers {}

pub(super) fn cycle_error(
    quotient: &Graph<usize, ()>,
    units: &[PlanUnit],
    regions: &[FusionRegion],
    reductions: &[ReductionRegion],
    matmuls: &[MatMulRegion],
) -> anyhow::Error {
    let mut members = HashSet::new();
    for component in petgraph::algo::kosaraju_scc(quotient)
        .into_iter()
        .filter(|scc| scc.len() > 1)
    {
        for node in component {
            match units[quotient[node]] {
                PlanUnit::Node(node) => {
                    members.insert(node);
                }
                PlanUnit::Region(id) => members.extend(regions[id].members.iter().copied()),
                PlanUnit::ReductionRegion(id) => {
                    members.extend(reductions[id].members.iter().copied())
                }
                PlanUnit::MatMulRegion(id) => members.extend(matmuls[id].members.iter().copied()),
            }
        }
    }
    anyhow::Error::new(CycleMembers(members)).context(FusionPlanError::ContractionCycle)
}

fn outputs<D: crate::tile::TileDType, G>(
    graph: &TensorGraph<D, G>,
    order: &[NodeIndex],
    members: &[NodeIndex],
    retained: &[crate::tile::RegionOutput],
) -> Vec<NodeIndex> {
    let set: HashSet<_> = members.iter().copied().collect();
    order
        .iter()
        .copied()
        .filter(|node| {
            set.contains(node)
                && (retained.iter().any(|output| output.node == *node)
                    || graph
                        .graph
                        .neighbors_directed(*node, Direction::Outgoing)
                        .next()
                        .is_none()
                    || graph
                        .graph
                        .neighbors_directed(*node, Direction::Outgoing)
                        .any(|consumer| !set.contains(&consumer)))
        })
        .collect()
}

/// Bisection strictly decreases fused members, so even adversarial partitions
/// converge to single-node units. Numerical operations and reduction policy stay
/// unchanged; independent regions are never discarded because another one cycles.
pub(super) fn refine<D: crate::tile::TileDType, G>(
    graph: &TensorGraph<D, G>,
    order: &[NodeIndex],
    conflicts: &HashSet<NodeIndex>,
    regions: &mut Vec<FusionRegion>,
    reductions: &mut [ReductionRegion],
    matmuls: &mut [MatMulRegion],
) -> Result<bool> {
    let touches = |members: &[NodeIndex]| members.iter().any(|node| conflicts.contains(node));
    let mut changed = false;
    let mut split = Vec::new();
    for region in regions.iter() {
        if region.members.len() > 1 && touches(&region.members) {
            let midpoint = region.members.len().div_ceil(2);
            for members in region.members.chunks(midpoint) {
                if members.len() > 1 {
                    split.push(FusionRegion::from_graph(
                        graph,
                        members.to_vec(),
                        outputs(graph, order, members, &region.outputs),
                    )?);
                }
            }
            changed = true;
        } else {
            split.push(region.clone());
        }
    }
    if changed {
        *regions = split;
        return Ok(true);
    }

    for region in matmuls {
        if !region.epilogue_operations.is_empty() && touches(&region.members) {
            let anchor = region.members[0];
            let keep = region.epilogue_operations.len() / 2;
            let epilogue: Vec<_> = region
                .epilogue_operations
                .iter()
                .take(keep)
                .map(|operation| operation.node)
                .collect();
            let members: Vec<_> = std::iter::once(anchor)
                .chain(epilogue.iter().copied())
                .collect();
            let schedule = region.schedule;
            *region = MatMulRegion::from_graph(
                graph,
                anchor,
                epilogue,
                outputs(graph, order, &members, &region.outputs),
            )?;
            region.schedule = schedule;
            changed = true;
        }
    }
    if changed {
        return Ok(true);
    }

    for region in reductions {
        if region.members.len() > 1 && touches(&region.members) {
            let anchor = region
                .members
                .iter()
                .copied()
                .find(|&node| matches!(graph[node], TensorGraphNode::ReduceAxis { .. }))
                .context("cyclic reduction has no anchor")?;
            let schedule = region.schedule;
            *region = ReductionRegion::from_graph(
                graph,
                Vec::new(),
                anchor,
                Vec::new(),
                Vec::new(),
                Vec::new(),
                vec![anchor],
            )?;
            region.schedule = schedule;
            changed = true;
        }
    }
    Ok(changed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{TensorExpr, UnaryOp};

    #[test]
    fn cycle_repair_preserves_an_independent_reduction_region() {
        let bad = TensorExpr::<f32>::input("bad", vec![4]).relu().exp().log();
        let good = (-TensorExpr::<f32>::input("good", vec![4]))
            .exp()
            .reduce_sum(0);
        let graph: TensorGraph<f32> = (bad + good).into();
        let order = graph.toposort();
        let relu = order
            .iter()
            .copied()
            .find(|&n| {
                matches!(
                    graph[n],
                    TensorGraphNode::Unary {
                        op: UnaryOp::Relu,
                        ..
                    }
                )
            })
            .unwrap();
        let log = order
            .iter()
            .copied()
            .find(|&n| {
                matches!(
                    graph[n],
                    TensorGraphNode::Unary {
                        op: UnaryOp::Log,
                        ..
                    }
                )
            })
            .unwrap();
        let anchor = order
            .iter()
            .copied()
            .find(|&n| matches!(graph[n], TensorGraphNode::ReduceAxis { .. }))
            .unwrap();
        let exp = graph.inputs(anchor)[0];
        let neg = graph.inputs(exp)[0];
        let mut regions =
            vec![FusionRegion::from_graph(&graph, vec![relu, log], vec![relu, log]).unwrap()];
        let safe = ReductionRegion::from_graph(
            &graph,
            vec![neg, exp],
            anchor,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            vec![anchor],
        )
        .unwrap();
        let mut reductions = vec![safe.clone()];
        let error = PtxExecutionPlan::build_with_regions(
            &graph,
            &order,
            regions.clone(),
            reductions.clone(),
            Vec::new(),
            order.last().copied(),
        )
        .unwrap_err();
        let conflicts = &error.downcast_ref::<CycleMembers>().unwrap().0;
        assert!(
            refine(
                &graph,
                &order,
                conflicts,
                &mut regions,
                &mut reductions,
                &mut []
            )
            .unwrap()
        );
        let plan = PtxExecutionPlan::build_with_regions(
            &graph,
            &order,
            regions,
            reductions,
            Vec::new(),
            order.last().copied(),
        )
        .unwrap();
        assert_eq!(plan.reduction_regions(), &[safe]);
    }

    #[test]
    fn refinement_preserves_an_explicit_matmul_precision_and_schedule() {
        let input = TensorExpr::constant(vec![0.01; 3 * 32 * 32], vec![3, 32, 32]);
        let weight = || TensorExpr::constant(vec![0.01; 32 * 32], vec![32, 32]);
        let hidden = input.matmul(weight()).exp();
        let graph: TensorGraph<f32> = (hidden.clone() + hidden.matmul(weight())).into();
        let order = graph.toposort();
        let (mut matmuls, _) = form_matmul_regions(&graph, &order, &HashSet::new(), true).unwrap();
        let schedule = MatMulSchedule::select(
            32,
            32,
            32,
            MatMulCapabilities {
                tf32: true,
                f16: false,
                bf16: false,
            },
            MatMulPrecision::StrictF32,
        );
        for region in &mut matmuls {
            region.schedule = schedule;
        }
        let plan = PtxExecutionPlan::build_refining_regions(
            &graph,
            &order,
            Vec::new(),
            Vec::new(),
            matmuls,
            order.last().copied(),
        )
        .unwrap();
        assert!(
            plan.matmul_regions()
                .iter()
                .all(|region| region.schedule == schedule)
        );
    }
}
