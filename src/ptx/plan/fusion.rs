use std::fmt::{self, Display, Write as _};

use super::*;
use crate::tile::{DType, FusionCost, FusionFeatures, FusionScorer, TileDType};

/// Preserve the previous greedy planner for controlled comparisons.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PtxFusionPolicy {
    #[default]
    CostAware,
    Greedy,
}

/// The graph partition represented by a scored alternative. Labels are kept
/// in `Display` so profiling consumers can match variants rather than strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusionAlternativeKind {
    KeepFusion,
    MaterializeProducers,
    SplitFullEpilogue,
    MaterializeProducersAndSplitFullEpilogue,
    SplitMatMulEpilogue,
    SplitPointwiseRegion,
}

impl Display for FusionAlternativeKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::KeepFusion => "keep fusion",
            Self::MaterializeProducers => "materialize producers",
            Self::SplitFullEpilogue => "split full epilogue",
            Self::MaterializeProducersAndSplitFullEpilogue => {
                "materialize producers and split full epilogue"
            }
            Self::SplitMatMulEpilogue => "split matmul epilogue",
            Self::SplitPointwiseRegion => "split pointwise region",
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FusionAlternative {
    pub kind: FusionAlternativeKind,
    pub features: Vec<FusionFeatures>,
    pub cost: FusionCost,
    pub selected: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FusionDecision {
    pub node: NodeIndex,
    pub alternatives: Vec<FusionAlternative>,
}

impl PtxExecutionPlan {
    pub fn fusion_decisions(&self) -> &[FusionDecision] {
        &self.fusion_decisions
    }

    /// Human-readable explanations; structured features remain available via
    /// `fusion_decisions` for profiling and future training data collection.
    pub fn describe_fusion(&self) -> String {
        let mut description = String::new();
        for decision in &self.fusion_decisions {
            for alternative in &decision.alternatives {
                let launches: usize = alternative
                    .features
                    .iter()
                    .map(|features| features.launches)
                    .fold(0usize, usize::saturating_add);
                let bytes: usize = alternative
                    .features
                    .iter()
                    .map(|features| features.intermediate_bytes)
                    .fold(0usize, usize::saturating_add);
                let repeated: usize = alternative
                    .features
                    .iter()
                    .map(|features| features.repeated_operations)
                    .fold(0usize, usize::saturating_add);
                let cost = alternative.cost;
                let _ = writeln!(
                    description,
                    "node={} {} {}: score={}ns (launch={}, traffic={}, compute={}, pressure={}); launches={}, intermediate_bytes={}, repeated_ops={}",
                    decision.node.index(),
                    if alternative.selected {
                        "selected"
                    } else {
                        "rejected"
                    },
                    alternative.kind,
                    cost.total_ns(),
                    cost.launch_ns,
                    cost.traffic_ns,
                    cost.compute_ns,
                    cost.pressure_ns,
                    launches,
                    bytes,
                    repeated
                );
            }
        }
        description
    }

    /// Rank a bounded set of legal region partitions. Existing schedules and
    /// arithmetic ordering are retained; this does not perform global search.
    pub fn select_fusion<D: TileDType, G>(
        &mut self,
        graph: &TensorGraph<D, G>,
        scorer: &dyn FusionScorer,
    ) -> Result<()> {
        // Splitting half regions introduces additional narrowing stores. Until
        // boundary rounding is represented in legality, retain their fusion.
        if D::TILE_DTYPE != DType::F32 {
            self.fusion_decisions.clear();
            return Ok(());
        }
        let order = graph.toposort();
        let mut decisions = Vec::new();
        // Anchors survive partitioning, so region indices can change safely.
        let anchors: Vec<_> = self
            .reduction_regions
            .iter()
            .filter_map(|region| {
                region
                    .members
                    .iter()
                    .copied()
                    .find(|&node| matches!(graph[node], TensorGraphNode::ReduceAxis { .. }))
            })
            .collect();
        for anchor in anchors {
            let Some(index) = self
                .reduction_regions
                .iter()
                .position(|region| region.members.contains(&anchor))
            else {
                continue;
            };
            let region = self.reduction_regions[index].clone();
            let producers: Vec<_> = region
                .producer_operations
                .iter()
                .map(|op| op.node)
                .collect();
            let reduced: Vec<_> = region
                .epilogue_operations
                .iter()
                .map(|op| op.node)
                .collect();
            let full: Vec<_> = region
                .full_epilogue_operations
                .iter()
                .map(|op| op.node)
                .collect();
            let mut candidates = Vec::new();
            // At most three alternatives to the existing region. Materializing
            // producers also lets neighboring pointwise operations fuse normally.
            for kind in [
                FusionAlternativeKind::MaterializeProducers,
                FusionAlternativeKind::SplitFullEpilogue,
                FusionAlternativeKind::MaterializeProducersAndSplitFullEpilogue,
            ] {
                let cut_producers = matches!(
                    kind,
                    FusionAlternativeKind::MaterializeProducers
                        | FusionAlternativeKind::MaterializeProducersAndSplitFullEpilogue
                );
                let cut_epilogue = matches!(
                    kind,
                    FusionAlternativeKind::SplitFullEpilogue
                        | FusionAlternativeKind::MaterializeProducersAndSplitFullEpilogue
                );
                if (cut_producers && producers.is_empty()) || (cut_epilogue && full.is_empty()) {
                    continue;
                }
                let kept_producers = if cut_producers {
                    Vec::new()
                } else {
                    producers.clone()
                };
                let kept_full = if cut_epilogue {
                    Vec::new()
                } else {
                    full.clone()
                };
                let bridges = if cut_epilogue {
                    Vec::new()
                } else {
                    region.broadcast_members.clone()
                };
                let members: Vec<_> = kept_producers
                    .iter()
                    .copied()
                    .chain(std::iter::once(anchor))
                    .chain(reduced.iter().copied())
                    .chain(bridges.iter().copied())
                    .chain(kept_full.iter().copied())
                    .collect();
                let mut alternative = ReductionRegion::from_graph(
                    graph,
                    kept_producers,
                    anchor,
                    reduced.clone(),
                    bridges,
                    kept_full,
                    outputs(graph, &members),
                )?;
                alternative.schedule = region.schedule;
                let mut reductions = self.reduction_regions.clone();
                reductions[index] = alternative;
                if let Some(plan) =
                    self.repartition(graph, &order, reductions, self.matmul_regions.clone())?
                {
                    candidates.push((kind, plan));
                }
            }
            decisions.push(self.choose(graph, scorer, anchor, candidates));
        }
        let anchors: Vec<_> = self
            .matmul_regions
            .iter()
            .map(|region| region.members[0])
            .collect();
        for anchor in anchors {
            let Some(index) = self
                .matmul_regions
                .iter()
                .position(|region| region.members[0] == anchor)
            else {
                continue;
            };
            let region = self.matmul_regions[index].clone();
            let mut candidates = Vec::new();
            if !region.epilogue_operations.is_empty() {
                // Retain broadcast-aware matmul dispatch even with no epilogue.
                let mut alternative =
                    MatMulRegion::from_graph(graph, anchor, Vec::new(), vec![anchor])?;
                alternative.schedule = region.schedule;
                let mut matmuls = self.matmul_regions.clone();
                matmuls[index] = alternative;
                if let Some(plan) =
                    self.repartition(graph, &order, self.reduction_regions.clone(), matmuls)?
                {
                    candidates.push((FusionAlternativeKind::SplitMatMulEpilogue, plan));
                }
            }
            decisions.push(self.choose(graph, scorer, anchor, candidates));
        }
        // Same-shape pointwise SSA shares producers already. Compare one balanced
        // cut to relieve live-value pressure without enumerating every partition.
        let components: Vec<_> = self
            .regions
            .iter()
            .map(|region| region.members.clone())
            .collect();
        for members in components {
            let Some(index) = self
                .regions
                .iter()
                .position(|region| region.members == members)
            else {
                continue;
            };
            let mut candidate = self.clone();
            candidate.regions.remove(index);
            for part in members.chunks(members.len().div_ceil(2)) {
                candidate.regions.push(FusionRegion::from_graph(
                    graph,
                    part.to_vec(),
                    outputs(graph, part),
                )?);
            }
            let plan = Self::build_with_regions(
                graph,
                &order,
                candidate.regions,
                self.reduction_regions.clone(),
                self.matmul_regions.clone(),
                self.graph_output,
            );
            let candidates = match plan {
                Ok(mut plan) => {
                    plan.matmul_schedules = self.matmul_schedules.clone();
                    vec![(FusionAlternativeKind::SplitPointwiseRegion, plan)]
                }
                Err(error)
                    if error.downcast_ref::<FusionPlanError>()
                        == Some(&FusionPlanError::ContractionCycle) =>
                {
                    Vec::new()
                }
                Err(error) => return Err(error),
            };
            decisions.push(self.choose(graph, scorer, members[0], candidates));
        }
        self.fusion_decisions = decisions;
        Ok(())
    }

    fn repartition<D: TileDType, G>(
        &self,
        graph: &TensorGraph<D, G>,
        order: &[NodeIndex],
        reductions: Vec<ReductionRegion>,
        matmuls: Vec<MatMulRegion>,
    ) -> Result<Option<Self>> {
        let claimed = reductions
            .iter()
            .flat_map(|region| &region.members)
            .chain(matmuls.iter().flat_map(|region| &region.members))
            .copied()
            .collect();
        let regions = form_pointwise_regions(graph, order, &claimed)?;
        match Self::build_with_regions(
            graph,
            order,
            regions,
            reductions,
            matmuls,
            self.graph_output,
        ) {
            Ok(mut plan) => {
                plan.matmul_schedules = self.matmul_schedules.clone();
                Ok(Some(plan))
            }
            Err(error)
                if error.downcast_ref::<FusionPlanError>()
                    == Some(&FusionPlanError::ContractionCycle) =>
            {
                Ok(None)
            }
            Err(error) => Err(error),
        }
    }

    fn choose<D: TileDType, G>(
        &mut self,
        graph: &TensorGraph<D, G>,
        scorer: &dyn FusionScorer,
        node: NodeIndex,
        candidates: Vec<(FusionAlternativeKind, Self)>,
    ) -> FusionDecision {
        let features = self.fusion_features(graph);
        let cost = scorer.score(&features);
        let mut alternatives = vec![FusionAlternative {
            kind: FusionAlternativeKind::KeepFusion,
            features,
            cost,
            selected: false,
        }];
        let mut selected = 0;
        let mut best = cost.total_ns();
        let mut winner = None;
        for (kind, plan) in candidates {
            let features = plan.fusion_features(graph);
            let cost = scorer.score(&features);
            if cost.total_ns() < best {
                selected = alternatives.len();
                best = cost.total_ns();
                winner = Some(plan);
            }
            alternatives.push(FusionAlternative {
                kind,
                features,
                cost,
                selected: false,
            });
        }
        alternatives[selected].selected = true;
        if let Some(plan) = winner {
            *self = plan;
        }
        FusionDecision { node, alternatives }
    }
}

pub(super) fn outputs<D: TileDType, G>(
    graph: &TensorGraph<D, G>,
    members: &[NodeIndex],
) -> Vec<NodeIndex> {
    let set: HashSet<_> = members.iter().copied().collect();
    members
        .iter()
        .copied()
        .filter(|&node| {
            let consumers: Vec<_> = graph
                .graph
                .neighbors_directed(node, Direction::Outgoing)
                .collect();
            consumers.is_empty() || consumers.iter().any(|consumer| !set.contains(consumer))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Parameter, TensorExpr};
    use crate::tile::AnalyticalFusionScorer;

    fn normalization(rows: usize, width: usize, expensive: bool) -> TensorGraph<f32> {
        let input = TensorExpr::input("x", vec![rows, width]);
        let producer = if expensive {
            input.exp().log().exp().log().exp().log().exp().log()
        } else {
            input.relu()
        };
        let sum = producer.clone().reduce_sum(1).broadcast(vec![rows, width]);
        (producer / sum).relu().into()
    }

    #[test]
    fn costly_recomputation_materializes_producer_but_retains_epilogue() {
        let graph = normalization(1024, 32, true);
        let mut plan = PtxExecutionPlan::build(&graph).unwrap();
        plan.select_fusion(
            &graph,
            &AnalyticalFusionScorer {
                multiprocessor_count: 76,
                ..AnalyticalFusionScorer::default()
            },
        )
        .unwrap();
        assert!(plan.reduction_regions()[0].producer_operations.is_empty());
        assert_eq!(
            plan.reduction_regions()[0].full_epilogue_operations.len(),
            2
        );
        assert_eq!(plan.regions()[0].operations.len(), 8);
        let decision = &plan.fusion_decisions()[0];
        assert_eq!(
            decision
                .alternatives
                .iter()
                .filter(|alternative| alternative.selected)
                .count(),
            1
        );
        assert_eq!(
            decision
                .alternatives
                .iter()
                .find(|alternative| alternative.selected)
                .unwrap()
                .kind,
            FusionAlternativeKind::MaterializeProducers
        );
        assert!(
            decision.alternatives[0]
                .features
                .iter()
                .any(|features| features.repeated_operations > 0)
        );
    }

    #[test]
    fn small_and_cheap_normalizations_keep_fusion() {
        for graph in [normalization(2, 32, true), normalization(1024, 32, false)] {
            let mut plan = PtxExecutionPlan::build(&graph).unwrap();
            plan.select_fusion(
                &graph,
                &AnalyticalFusionScorer {
                    multiprocessor_count: 76,
                    ..AnalyticalFusionScorer::default()
                },
            )
            .unwrap();
            assert!(!plan.reduction_regions()[0].producer_operations.is_empty());
            assert!(plan.fusion_decisions()[0].alternatives[0].selected);
        }
    }

    #[test]
    fn scorer_can_be_replaced_without_changing_candidate_generation() {
        struct AvoidRepetition;
        impl FusionScorer for AvoidRepetition {
            fn score(&self, kernels: &[FusionFeatures]) -> FusionCost {
                FusionCost {
                    compute_ns: kernels
                        .iter()
                        .map(|features| features.repeated_operations)
                        .sum(),
                    ..FusionCost::default()
                }
            }
        }
        let graph = normalization(2, 32, true);
        let mut plan = PtxExecutionPlan::build(&graph).unwrap();
        plan.select_fusion(&graph, &AvoidRepetition).unwrap();
        assert!(plan.reduction_regions()[0].producer_operations.is_empty());
    }

    #[test]
    fn target_resource_pressure_can_split_a_matmul_epilogue() {
        let shape = vec![512, 512];
        let mut expression = TensorExpr::input("lhs", vec![512, 4096])
            .matmul(TensorExpr::input("rhs", vec![4096, 512]));
        for i in 0..12 {
            expression = expression
                + TensorExpr::from(Parameter::new(vec![i as f32; 512 * 512], shape.clone()));
        }
        let graph: TensorGraph<f32> = expression.relu().into();
        let mut plan = PtxExecutionPlan::build(&graph).unwrap();
        plan.select_fusion(
            &graph,
            &AnalyticalFusionScorer {
                registers_per_sm: 60_000,
                ..AnalyticalFusionScorer::default()
            },
        )
        .unwrap();
        assert!(
            plan.matmul_regions()[0].epilogue_operations.is_empty(),
            "resource-heavy fusion should split: {:?}",
            plan.fusion_decisions()
        );
        // Synthetic resource budgets exercise the pressure term separately from
        // the hardware-dependent end-to-end benchmarks.
        assert!(
            plan.fusion_decisions()
                .iter()
                .flat_map(|decision| &decision.alternatives)
                .any(|alternative| alternative.cost.pressure_ns > 0)
        );
    }

    #[test]
    fn empty_dimensions_do_not_overflow_or_create_repetition() {
        let graph = normalization(0, 32, true);
        let mut plan = PtxExecutionPlan::build(&graph).unwrap();
        plan.select_fusion(
            &graph,
            &AnalyticalFusionScorer {
                multiprocessor_count: 76,
                ..AnalyticalFusionScorer::default()
            },
        )
        .unwrap();
        assert!(
            plan.fusion_decisions()
                .iter()
                .flat_map(|decision| &decision.alternatives)
                .flat_map(|alternative| &alternative.features)
                .all(|features| features.repeated_operations == 0)
        );
    }
}
