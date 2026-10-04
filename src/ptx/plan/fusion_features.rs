use super::fusion::outputs;
use super::*;
use crate::tensor::{BinaryOp, UnaryOp};
use crate::tile::{FusionFeatures, MatMulPlan, ReductionInputDomain, TileDType};

impl PtxExecutionPlan {
    pub(super) fn fusion_features<D: TileDType, G>(
        &self,
        graph: &TensorGraph<D, G>,
    ) -> Vec<FusionFeatures> {
        let bytes = D::TILE_DTYPE.size_bytes();
        self.steps
            .iter()
            .filter_map(|step| {
                if matches!(
                    step.action,
                    PtxPlanAction::Upload | PtxPlanAction::VirtualView
                ) {
                    return None;
                }
                let mut features = FusionFeatures {
                    storage_dtype: Some(D::TILE_DTYPE),
                    logical_shape: step.shape.clone(),
                    launches: 1,
                    threads_per_block: 256,
                    ..FusionFeatures::default()
                };
                let elements = count(&step.shape);
                features.active_threads = elements;
                features.estimated_registers_per_thread =
                    16 + live_values(graph, &step.members) * 2;
                features.traffic_bytes = step
                    .inputs
                    .iter()
                    .map(|&node| count(graph[node].shape()).saturating_mul(bytes))
                    .chain(
                        step.output_shapes
                            .iter()
                            .map(|shape| count(shape).saturating_mul(bytes)),
                    )
                    .fold(0usize, usize::saturating_add);
                features.intermediate_bytes = step
                    .outputs
                    .iter()
                    .zip(&step.output_shapes)
                    .filter(|(node, _)| {
                        graph
                            .graph
                            .neighbors_directed(**node, Direction::Outgoing)
                            .next()
                            .is_some()
                    })
                    .map(|(_, shape)| count(shape).saturating_mul(bytes))
                    .fold(0usize, usize::saturating_add);
                for &member in &step.members {
                    features.weighted_operations = features.weighted_operations.saturating_add(
                        count(graph[member].shape())
                            .saturating_mul(operation_weight(&graph[member])),
                    );
                }
                // Coordinate work scales with the iteration domain, not merely
                // unique broadcast storage. Virtual maps are normalized by tile IR.
                features.address_operations = elements
                    .saturating_mul(step.inputs.len())
                    .saturating_mul(step.shape.len().max(1));
                match step.action {
                    PtxPlanAction::ReductionRegion(index) => {
                        let region = &self.reduction_regions[index];
                        let full = count(&region.input_shape);
                        let reduced = count(&region.output_shape);
                        let full_inputs = region
                            .inputs
                            .iter()
                            .filter(|input| input.domain == ReductionInputDomain::Full)
                            .count();
                        let reduced_inputs = region.inputs.len() - full_inputs;
                        let visits = if region.full_epilogue_operations.is_empty() {
                            1
                        } else {
                            2
                        };
                        features.address_operations = full
                            .saturating_mul(full_inputs)
                            .saturating_mul(visits)
                            .saturating_add(reduced.saturating_mul(reduced_inputs))
                            .saturating_mul(region.input_shape.len().max(1));
                        features.weighted_operations = features
                            .weighted_operations
                            .saturating_add(full.saturating_sub(reduced));
                        let threads = match region.schedule {
                            ReductionSchedule::Serial => 1,
                            ReductionSchedule::Subgroup { width } => width as usize,
                            ReductionSchedule::Block { threads } => threads as usize,
                        };
                        features.threads_per_block = if threads == 1 { 256 } else { threads };
                        features.active_threads = reduced.saturating_mul(threads);
                        features.estimated_registers_per_thread += 16;
                        if threads > 32 {
                            features.shared_bytes_per_block = threads.saturating_mul(4);
                        }
                        if !region.full_epilogue_operations.is_empty() {
                            features.repeated_operations = region
                                .producer_operations
                                .iter()
                                .map(|op| full.saturating_mul(operation_weight(&graph[op.node])))
                                .fold(0usize, usize::saturating_add);
                            features.weighted_operations = features
                                .weighted_operations
                                .saturating_add(features.repeated_operations);
                            let reload = region
                                .inputs
                                .iter()
                                .filter(|input| input.domain == ReductionInputDomain::Full)
                                .count()
                                .saturating_mul(full)
                                .saturating_mul(bytes);
                            features.traffic_bytes = features.traffic_bytes.saturating_add(reload);
                        }
                    }
                    PtxPlanAction::MatMulRegion(index) => {
                        let region = &self.matmul_regions[index];
                        matmul_features(&mut features, region.schedule, elements, region.k);
                    }
                    PtxPlanAction::Kernel
                        if matches!(graph[step.node], TensorGraphNode::MatMul { .. }) =>
                    {
                        if let Some(&schedule) = self.matmul_schedules.get(&step.node) {
                            features.launches = count(&step.shape[..step.shape.len() - 2]);
                            matmul_features(
                                &mut features,
                                schedule,
                                elements,
                                schedule.logical_shape.k,
                            );
                        }
                    }
                    _ => {}
                }
                if elements == 0 {
                    features.launches = 0;
                }
                Some(features)
            })
            .collect()
    }
}

fn count(shape: &[usize]) -> usize {
    shape.iter().copied().fold(1, usize::saturating_mul)
}

fn operation_weight<D: TileDType>(node: &TensorGraphNode<D>) -> usize {
    match node {
        TensorGraphNode::Unary {
            op: UnaryOp::Exp | UnaryOp::Log,
            ..
        }
        | TensorGraphNode::Binary {
            op: BinaryOp::Div, ..
        } => 32,
        TensorGraphNode::Unary { .. }
        | TensorGraphNode::Binary { .. }
        | TensorGraphNode::Gt { .. }
        | TensorGraphNode::Mask { .. } => 1,
        TensorGraphNode::ReduceAxis { .. } => 1,
        _ => 0,
    }
}

fn matmul_features(
    features: &mut FusionFeatures,
    schedule: MatMulSchedule,
    elements: usize,
    k: usize,
) {
    let tile = schedule.block_tile;
    features.threads_per_block = schedule.thread_count();
    let shape = &features.logical_shape;
    let batch = count(&shape[..shape.len() - 2]);
    features.active_threads = batch
        .saturating_mul(shape[shape.len() - 2].div_ceil(tile.m))
        .saturating_mul(shape[shape.len() - 1].div_ceil(tile.n))
        .saturating_mul(schedule.thread_count());
    features.estimated_registers_per_thread += 48;
    features.shared_bytes_per_block = schedule.pipeline_stages.saturating_mul(
        schedule
            .operand_layouts
            .0
            .allocation_bytes(tile.m, schedule.operand_dtype)
            .saturating_add(
                schedule
                    .operand_layouts
                    .1
                    .allocation_bytes(tile.k, schedule.operand_dtype),
            ),
    );
    if schedule.plan != MatMulPlan::ScalarF32 {
        features.shared_bytes_per_block = features
            .shared_bytes_per_block
            .saturating_add(tile.m.saturating_mul(tile.n).saturating_mul(4));
    }
    let work = elements.saturating_mul(k).saturating_mul(2);
    features.weighted_operations =
        features
            .weighted_operations
            .saturating_add(if schedule.plan == MatMulPlan::ScalarF32 {
                work
            } else {
                work / 16
            });
}

fn live_values<D: TileDType, G>(graph: &TensorGraph<D, G>, members: &[NodeIndex]) -> usize {
    let mut last_use = HashMap::new();
    for (position, &node) in members.iter().enumerate() {
        for input in graph.inputs(node) {
            last_use.insert(input, position);
        }
    }
    for node in outputs(graph, members) {
        last_use.insert(node, members.len());
    }
    let member_set: HashSet<_> = members.iter().copied().collect();
    let mut live: HashSet<_> = last_use
        .keys()
        .filter(|node| !member_set.contains(node))
        .copied()
        .collect();
    let mut peak = live.len();
    for (position, &node) in members.iter().enumerate() {
        live.insert(node);
        peak = peak.max(live.len());
        live.retain(|node| last_use.get(node).is_some_and(|&last| last > position));
    }
    peak
}
