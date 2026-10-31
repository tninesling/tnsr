use std::fmt::{self, Display, Write as _};
use std::sync::Arc;

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
    /// Change ownership of independent serial fibers, retaining reduction order.
    SerialReductionThreads {
        threads: u32,
    },
    MaterializeProducersWithSerialThreads {
        threads: u32,
    },
    SplitFullEpilogueWithSerialThreads {
        threads: u32,
    },
    MaterializeProducersAndSplitFullEpilogueWithSerialThreads {
        threads: u32,
    },
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
            Self::SerialReductionThreads { threads } => {
                return write!(formatter, "serial reduction {threads} threads/block");
            }
            Self::MaterializeProducersWithSerialThreads { threads } => {
                return write!(formatter, "materialize producers; serial reduction {threads} threads/block");
            }
            Self::SplitFullEpilogueWithSerialThreads { threads } => {
                return write!(formatter, "split full epilogue; serial reduction {threads} threads/block");
            }
            Self::MaterializeProducersAndSplitFullEpilogueWithSerialThreads { threads } => {
                return write!(formatter, "materialize producers and split full epilogue; serial reduction {threads} threads/block");
            }
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FusionAlternative {
    pub kind: FusionAlternativeKind,
    /// Complete ordered kernel features, shared across alternatives/decisions.
    pub features: Vec<Arc<FusionFeatures>>,
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

    /// Rank bounded legal region partitions and serial launch geometries.
    /// Reduction arithmetic ordering is retained; this is local plan search.
    pub fn select_fusion<D: TileDType, G>(
        &mut self,
        graph: &TensorGraph<D, G>,
        scorer: &dyn FusionScorer,
    ) -> Result<()> {
        self.select_fusion_with_serial_threads(graph, scorer, None)
    }

    /// Force a legal serial launch geometry without altering arithmetic trees.
    /// Cooperative reduction schedules are unchanged.
    pub fn set_serial_reduction_block_threads(&mut self, threads: u32) -> Result<()> {
        ReductionRegion::validate_serial_block_threads(threads)?;
        for region in &mut self.reduction_regions {
            if region.schedule == ReductionSchedule::Serial {
                region.serial_block_threads = threads;
            }
        }
        Ok(())
    }

    pub(crate) fn select_fusion_with_serial_threads<D: TileDType, G>(
        &mut self,
        graph: &TensorGraph<D, G>,
        scorer: &dyn FusionScorer,
        serial_threads: Option<u32>,
    ) -> Result<()> {
        if let Some(threads) = serial_threads {
            self.set_serial_reduction_block_threads(threads)?;
        }
        // Splitting half regions introduces additional narrowing stores. Until
        // boundary rounding is represented in legality, retain their fusion.
        if D::TILE_DTYPE != DType::F32 {
            self.fusion_decisions.clear();
            return Ok(());
        }
        self.fusion_decisions.clear();
        let order = graph.toposort();
        let mut decisions = Vec::new();
        // A large backward plan has thousands of mostly unchanged kernels in
        // hundreds of alternatives. Retain every feature without duplicating
        // its shape/allocation for every local decision.
        let mut feature_pool = HashMap::new();
        // Legacy standalone strict reductions are also candidates for the
        // serial region schedule. Keep the original kernel available, and
        // exclude empty axes because legacy and region empty-mean contracts
        // differ. No cooperative arithmetic tree is introduced here.
        let standalone: Vec<_> = self
            .steps
            .iter()
            .filter_map(|step| {
                if step.action != PtxPlanAction::Kernel {
                    return None;
                }
                let TensorGraphNode::ReduceAxis { op, axis, .. } = graph[step.node] else {
                    return None;
                };
                if !matches!(
                    op,
                    crate::tensor::ReduceOp::Sum | crate::tensor::ReduceOp::Mean
                ) {
                    return None;
                }
                let input = graph.inputs(step.node).first().copied()?;
                let shape = graph[input].shape();
                (!shape.contains(&0) && shape.get(axis).is_some_and(|&d| d > 0))
                    .then_some(step.node)
            })
            .collect();
        for anchor in standalone.iter().copied() {
            let region = ReductionRegion::from_graph(
                graph,
                Vec::new(),
                anchor,
                Vec::new(),
                Vec::new(),
                Vec::new(),
                vec![anchor],
            )?;
            let mut reductions = self.reduction_regions.clone();
            reductions.push(region);
            let mut candidates = Vec::new();
            if let Some(mut plan) =
                self.repartition(graph, &order, reductions, self.matmul_regions.clone())?
            {
                if let Some(threads) = serial_threads {
                    plan.set_serial_reduction_block_threads(threads)?;
                }
                let threads = plan
                    .reduction_regions
                    .iter()
                    .find(|r| r.members.contains(&anchor))
                    .map_or(1024, |r| r.serial_block_threads);
                candidates.push((
                    FusionAlternativeKind::SerialReductionThreads { threads },
                    plan,
                ));
            }
            decisions.push(self.choose(
                graph,
                scorer,
                anchor,
                candidates,
                serial_threads.is_none(),
                &mut feature_pool,
            ));
        }
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
            .filter(|anchor| !standalone.contains(anchor))
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
                alternative.serial_block_threads = region.serial_block_threads;
                let mut reductions = self.reduction_regions.clone();
                reductions[index] = alternative;
                if let Some(plan) =
                    self.repartition(graph, &order, reductions, self.matmul_regions.clone())?
                {
                    candidates.push((kind, plan));
                }
            }
            decisions.push(self.choose(
                graph,
                scorer,
                anchor,
                candidates,
                serial_threads.is_none(),
                &mut feature_pool,
            ));
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
            decisions.push(self.choose(
                graph,
                scorer,
                anchor,
                candidates,
                serial_threads.is_none(),
                &mut feature_pool,
            ));
        }
        // Same-shape pointwise SSA shares producers already. Compare one balanced
        // cut to relieve live-value pressure without enumerating every partition.
        let components: Vec<_> = self
            .regions
            .iter()
            .filter(|region| region.members.len() > 1)
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
            decisions.push(self.choose(
                graph,
                scorer,
                members[0],
                candidates,
                false,
                &mut feature_pool,
            ));
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
        let proposed: HashSet<_> = reductions
            .iter()
            .flat_map(|region| &region.members)
            .chain(matmuls.iter().flat_map(|region| &region.members))
            .copied()
            .collect();
        let freed: HashSet<_> = self
            .reduction_regions
            .iter()
            .flat_map(|region| &region.members)
            .chain(
                self.matmul_regions
                    .iter()
                    .flat_map(|region| &region.members),
            )
            .filter(|node| !proposed.contains(node))
            .copied()
            .collect();
        // Keep already-legal pointwise partitions. Only materialized members
        // participate in new groups; regrouping unrelated backward components
        // reintroduces cycles and makes each local cost query a global search.
        let claimed: HashSet<_> = order
            .iter()
            .filter(|node| !freed.contains(node))
            .copied()
            .collect();
        let mut regions = self.regions.clone();
        regions.extend(form_pointwise_regions(graph, order, &claimed)?);
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
        mut candidates: Vec<(FusionAlternativeKind, Self)>,
        serial_geometry: bool,
        feature_pool: &mut HashMap<FusionFeatures, Arc<FusionFeatures>>,
    ) -> FusionDecision {
        let mut alternatives = Vec::new();
        let mut best = usize::MAX;
        let mut selected = 0;
        // Geometry alternatives only modify one feature and one region field.
        // Keep complete ordered inputs for arbitrary, non-additive scorers, but
        // neither clone plans nor recompute unrelated kernels for these variants.
        let mut winner = None;
        for (candidate, kind, plan) in
            std::iter::once((None, FusionAlternativeKind::KeepFusion, &*self)).chain(
                candidates
                    .iter()
                    .enumerate()
                    .map(|(index, (kind, plan))| (Some(index), *kind, plan)),
            )
        {
            let mut features = plan.fusion_features(graph);
            let shared = share_features(features.clone(), feature_pool);
            // The original keep-fusion entry precedes all geometry alternatives;
            // each partition's geometry alternatives precede its unmodified plan.
            if candidate.is_none() {
                let cost = scorer.score(&features);
                best = cost.total_ns();
                alternatives.push(FusionAlternative {
                    kind,
                    features: shared.clone(),
                    cost,
                    selected: false,
                });
            }
            if serial_geometry
                && let Some((region_index, region)) = plan
                    .reduction_regions
                    .iter()
                    .enumerate()
                    .find(|(_, region)| region.members.contains(&node))
                && region.schedule == ReductionSchedule::Serial
            {
                let fibers = region
                    .output_shape
                    .iter()
                    .copied()
                    .fold(1usize, usize::saturating_mul);
                if fibers > 0 {
                    let feature_index = plan
                        .steps
                        .iter()
                        .filter(|step| {
                            !matches!(
                                step.action,
                                PtxPlanAction::Upload | PtxPlanAction::VirtualView
                            )
                        })
                        .position(|step| {
                            step.action == PtxPlanAction::ReductionRegion(region_index)
                        });
                    if let Some(feature_index) = feature_index {
                        let original = features[feature_index].clone();
                        let largest = fibers.clamp(32, 1024).next_power_of_two();
                        for threads in [32u32, 64, 128, 256, 512, 1024] {
                            if threads == region.serial_block_threads || threads as usize > largest
                            {
                                continue;
                            }
                            let Some(kind) = serial_geometry_kind(kind, threads) else {
                                continue;
                            };
                            super::fusion_features::set_serial_geometry_features(
                                &mut features[feature_index],
                                region,
                                threads as usize,
                            );
                            let cost = scorer.score(&features);
                            if cost.total_ns() < best {
                                best = cost.total_ns();
                                selected = alternatives.len();
                                winner = Some((candidate, Some((region_index, threads))));
                            }
                            let mut variant_shared = shared.clone();
                            variant_shared[feature_index] =
                                share_feature(features[feature_index].clone(), feature_pool);
                            alternatives.push(FusionAlternative {
                                kind,
                                features: variant_shared,
                                cost,
                                selected: false,
                            });
                        }
                        features[feature_index] = original;
                    }
                }
            }
            if candidate.is_some() {
                let cost = scorer.score(&features);
                if cost.total_ns() < best {
                    best = cost.total_ns();
                    selected = alternatives.len();
                    winner = Some((candidate, None));
                }
                alternatives.push(FusionAlternative {
                    kind,
                    features: shared,
                    cost,
                    selected: false,
                });
            }
        }
        alternatives[selected].selected = true;
        if let Some((candidate, geometry)) = winner {
            if let Some(index) = candidate {
                *self = candidates.swap_remove(index).1;
            }
            if let Some((index, threads)) = geometry {
                self.reduction_regions[index].serial_block_threads = threads;
            }
        }
        FusionDecision { node, alternatives }
    }
}

fn serial_geometry_kind(
    kind: FusionAlternativeKind,
    threads: u32,
) -> Option<FusionAlternativeKind> {
    Some(match kind {
        FusionAlternativeKind::KeepFusion
        | FusionAlternativeKind::SerialReductionThreads { .. } => {
            FusionAlternativeKind::SerialReductionThreads { threads }
        }
        FusionAlternativeKind::MaterializeProducers => {
            FusionAlternativeKind::MaterializeProducersWithSerialThreads { threads }
        }
        FusionAlternativeKind::SplitFullEpilogue => {
            FusionAlternativeKind::SplitFullEpilogueWithSerialThreads { threads }
        }
        FusionAlternativeKind::MaterializeProducersAndSplitFullEpilogue => {
            FusionAlternativeKind::MaterializeProducersAndSplitFullEpilogueWithSerialThreads {
                threads,
            }
        }
        _ => return None,
    })
}

fn share_feature(
    feature: FusionFeatures,
    pool: &mut HashMap<FusionFeatures, Arc<FusionFeatures>>,
) -> Arc<FusionFeatures> {
    if let Some(shared) = pool.get(&feature) {
        return shared.clone();
    }
    let shared = Arc::new(feature.clone());
    pool.insert(feature, shared.clone());
    shared
}

fn share_features(
    features: Vec<FusionFeatures>,
    pool: &mut HashMap<FusionFeatures, Arc<FusionFeatures>>,
) -> Vec<Arc<FusionFeatures>> {
    features
        .into_iter()
        .map(|feature| share_feature(feature, pool))
        .collect()
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
        let selected = decision
            .alternatives
            .iter()
            .find(|alternative| alternative.selected)
            .unwrap();
        assert!(matches!(
            selected.kind,
            FusionAlternativeKind::MaterializeProducers
                | FusionAlternativeKind::MaterializeProducersWithSerialThreads { .. }
        ));
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
            let selected = plan.fusion_decisions()[0]
                .alternatives
                .iter()
                .find(|a| a.selected)
                .unwrap();
            assert!(matches!(
                selected.kind,
                FusionAlternativeKind::KeepFusion
                    | FusionAlternativeKind::SerialReductionThreads { .. }
            ));
        }
    }

    #[test]
    fn shared_diagnostics_retain_complete_cost_inputs() {
        let normalize = |name| {
            let input = TensorExpr::<f32>::input(name, vec![1024, 32]).relu();
            input.clone() / input.reduce_sum(1).broadcast(vec![1024, 32])
        };
        let graph: TensorGraph<f32> = [
            "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p",
        ]
        .into_iter()
        .map(normalize)
        .reduce(|left, right| left + right)
        .unwrap()
        .into();
        let mut plan = PtxExecutionPlan::build(&graph).unwrap();
        let scorer = AnalyticalFusionScorer {
            multiprocessor_count: 76,
            ..Default::default()
        };
        plan.select_fusion(&graph, &scorer).unwrap();
        let mut pointers = HashSet::new();
        let mut total = 0;
        for alternative in plan.fusion_decisions().iter().flat_map(|d| &d.alternatives) {
            let restored: Vec<_> = alternative
                .features
                .iter()
                .map(|f| f.as_ref().clone())
                .collect();
            assert_eq!(scorer.score(&restored), alternative.cost);
            for feature in &alternative.features {
                pointers.insert(Arc::as_ptr(feature));
                total += 1;
            }
        }
        assert!(
            total > pointers.len() * 2,
            "common kernels should share diagnostic storage: {total} entries, {} allocations",
            pointers.len()
        );
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
    // Reference expansion deliberately retains the old expensive implementation:
    // clone a complete plan for every geometry and recompute all its features.
    fn choose_full_plan_reference(
        plan: &mut PtxExecutionPlan,
        graph: &TensorGraph<f32>,
        scorer: &dyn FusionScorer,
        node: NodeIndex,
        candidates: Vec<(FusionAlternativeKind, PtxExecutionPlan)>,
    ) -> FusionDecision {
        fn expand(
            plan: &PtxExecutionPlan,
            node: NodeIndex,
            kind: FusionAlternativeKind,
        ) -> Vec<(FusionAlternativeKind, PtxExecutionPlan)> {
            let Some((index, region)) = plan
                .reduction_regions
                .iter()
                .enumerate()
                .find(|(_, region)| region.members.contains(&node))
            else {
                return Vec::new();
            };
            if region.schedule != ReductionSchedule::Serial {
                return Vec::new();
            }
            let fibers = region
                .output_shape
                .iter()
                .copied()
                .fold(1usize, usize::saturating_mul);
            if fibers == 0 {
                return Vec::new();
            }
            let largest = fibers.clamp(32, 1024).next_power_of_two();
            [32u32, 64, 128, 256, 512, 1024]
                .into_iter()
                .filter_map(|threads| {
                    if threads == region.serial_block_threads || threads as usize > largest {
                        return None;
                    }
                    let kind = serial_geometry_kind(kind, threads)?;
                    let mut variant = plan.clone();
                    variant.reduction_regions[index].serial_block_threads = threads;
                    Some((kind, variant))
                })
                .collect()
        }
        let mut expanded = expand(plan, node, FusionAlternativeKind::KeepFusion);
        for (kind, candidate) in candidates {
            expanded.extend(expand(&candidate, node, kind));
            expanded.push((kind, candidate));
        }
        plan.choose(graph, scorer, node, expanded, false, &mut HashMap::new())
    }

    fn producer_cut(
        plan: &PtxExecutionPlan,
        graph: &TensorGraph<f32>,
        index: usize,
    ) -> (NodeIndex, Vec<(FusionAlternativeKind, PtxExecutionPlan)>) {
        let region = &plan.reduction_regions[index];
        let anchor = region
            .members
            .iter()
            .copied()
            .find(|&node| matches!(graph[node], TensorGraphNode::ReduceAxis { .. }))
            .unwrap();
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
        let members: Vec<_> = std::iter::once(anchor)
            .chain(reduced.iter().copied())
            .chain(region.broadcast_members.iter().copied())
            .chain(full.iter().copied())
            .collect();
        let mut cut = ReductionRegion::from_graph(
            graph,
            Vec::new(),
            anchor,
            reduced,
            region.broadcast_members.clone(),
            full,
            outputs(graph, &members),
        )
        .unwrap();
        cut.schedule = region.schedule;
        cut.serial_block_threads = region.serial_block_threads;
        let mut reductions = plan.reduction_regions.clone();
        reductions[index] = cut;
        let candidate = plan
            .repartition(
                graph,
                &graph.toposort(),
                reductions,
                plan.matmul_regions.clone(),
            )
            .unwrap()
            .unwrap();
        (
            anchor,
            vec![(FusionAlternativeKind::MaterializeProducers, candidate)],
        )
    }

    fn many_normalizations(count: usize) -> TensorGraph<f32> {
        (0..count)
            .map(|_| {
                let input = TensorExpr::<f32>::input("input", vec![128, 192]).relu();
                input.clone() / input.reduce_sum(1).broadcast(vec![128, 192])
            })
            .reduce(|left, right| left + right)
            .unwrap()
            .into()
    }

    #[test]
    fn serial_geometry_overlays_match_full_plan_expansion() {
        struct InteractingScorer;
        impl FusionScorer for InteractingScorer {
            fn score(&self, kernels: &[FusionFeatures]) -> FusionCost {
                // Non-additive and order sensitive, so scoring only changed
                // kernels would not preserve this scorer's behavior.
                let interaction = kernels
                    .windows(2)
                    .enumerate()
                    .map(|(index, pair)| {
                        (index + 1).saturating_mul(
                            pair[0]
                                .threads_per_block
                                .abs_diff(pair[1].threads_per_block),
                        )
                    })
                    .fold(0usize, usize::saturating_add);
                FusionCost {
                    compute_ns: interaction,
                    ..Default::default()
                }
            }
        }
        for graph in [
            normalization(2, 32, false),
            normalization(1024, 32, true),
            normalization(0, 32, false),
            many_normalizations(4),
        ] {
            let baseline = PtxExecutionPlan::build(&graph).unwrap();
            let (anchor, candidates) = producer_cut(&baseline, &graph, 0);
            for scorer in [
                &AnalyticalFusionScorer::default() as &dyn FusionScorer,
                &InteractingScorer,
            ] {
                let mut actual = baseline.clone();
                let mut expected = baseline.clone();
                let decision = actual.choose(
                    &graph,
                    scorer,
                    anchor,
                    candidates.clone(),
                    true,
                    &mut HashMap::new(),
                );
                let reference = choose_full_plan_reference(
                    &mut expected,
                    &graph,
                    scorer,
                    anchor,
                    candidates.clone(),
                );
                assert_eq!(decision, reference);
                assert_eq!(actual, expected);
                assert_eq!(
                    actual.kernel_features(&graph),
                    expected.kernel_features(&graph)
                );
                let ptx = |plan: &PtxExecutionPlan| {
                    let mut tile = crate::tile::TileGraph::from_graph(&TensorGraph::<f32>::new());
                    tile.add_fusion_regions(&plan.regions).unwrap();
                    tile.add_reduction_regions(&plan.reduction_regions).unwrap();
                    tile.add_matmul_regions(&plan.matmul_regions).unwrap();
                    crate::ptx::PtxGraph::from(tile).to_ptx()
                };
                assert_eq!(ptx(&actual), ptx(&expected));
            }
        }
    }

    #[test]
    fn standalone_promotion_geometry_matches_full_plan_expansion() {
        let graph: TensorGraph<f32> = TensorExpr::input("x", vec![128, 192]).reduce_sum(0).into();
        let baseline = PtxExecutionPlan::build(&graph).unwrap();
        let anchor = baseline.graph_output.unwrap();
        assert!(baseline.reduction_regions.is_empty());
        let region = ReductionRegion::from_graph(
            &graph,
            Vec::new(),
            anchor,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            vec![anchor],
        )
        .unwrap();
        let promoted = baseline
            .repartition(&graph, &graph.toposort(), vec![region], Vec::new())
            .unwrap()
            .unwrap();
        let candidates = vec![(
            FusionAlternativeKind::SerialReductionThreads { threads: 1024 },
            promoted,
        )];
        let scorer = AnalyticalFusionScorer::default();
        let mut actual = baseline.clone();
        let mut expected = baseline;
        let decision = actual.choose(
            &graph,
            &scorer,
            anchor,
            candidates.clone(),
            true,
            &mut HashMap::new(),
        );
        let reference =
            choose_full_plan_reference(&mut expected, &graph, &scorer, anchor, candidates);
        assert_eq!(decision, reference);
        assert_eq!(actual, expected);
    }

    #[test]
    #[ignore = "device-free fusion planning microbenchmark; run with --ignored --nocapture"]
    fn serial_geometry_overlay_planning_benchmark() {
        let graph = many_normalizations(64);
        let baseline = PtxExecutionPlan::build(&graph).unwrap();
        let (anchor, candidates) = producer_cut(&baseline, &graph, 0);
        let scorer = AnalyticalFusionScorer::default();
        let repetitions = 20;
        let mut reference = std::time::Duration::ZERO;
        let mut overlays = std::time::Duration::ZERO;
        // Alternate order and discard one warm pair to reduce allocator/cache
        // and host-load drift; this measures planning only, never CUDA/JIT.
        for iteration in 0..=repetitions {
            let order = if iteration % 2 == 0 {
                [true, false]
            } else {
                [false, true]
            };
            for full_plan in order {
                let start = std::time::Instant::now();
                let mut plan = baseline.clone();
                let decision = if full_plan {
                    choose_full_plan_reference(
                        &mut plan,
                        &graph,
                        &scorer,
                        anchor,
                        candidates.clone(),
                    )
                } else {
                    plan.choose(
                        &graph,
                        &scorer,
                        anchor,
                        candidates.clone(),
                        true,
                        &mut HashMap::new(),
                    )
                };
                std::hint::black_box(decision);
                if iteration > 0 {
                    if full_plan {
                        reference += start.elapsed();
                    } else {
                        overlays += start.elapsed();
                    }
                }
            }
        }
        eprintln!(
            "geometry planning {repetitions} iterations / {} physical steps: full-plan={reference:?}, overlays={overlays:?}, speedup={:.2}x",
            baseline.steps.len(),
            reference.as_secs_f64() / overlays.as_secs_f64()
        );
    }
}
