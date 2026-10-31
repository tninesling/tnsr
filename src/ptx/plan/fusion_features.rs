use super::fusion::outputs;
use super::*;
use crate::tensor::{BinaryOp, UnaryOp};
use crate::tile::{FusionFeatures, MatMulPlan, ReductionInputDomain, TileDType};

impl PtxExecutionPlan {
    /// Compiler-visible estimates for each physical step in this plan, in
    /// execution order, without uploading inputs or executing any CUDA work.
    /// Upload and virtual-only steps are omitted. Estimates describe the
    /// current partition and schedules, so query again after replanning.
    ///
    /// Register, traffic, working-set and index-operation counts are planning
    /// estimates rather than profiler measurements. This API is also available
    /// when fusion selection retained no alternative diagnostics.
    pub fn kernel_features<D: TileDType, G>(
        &self,
        graph: &TensorGraph<D, G>,
    ) -> Vec<FusionFeatures> {
        self.fusion_features(graph)
    }

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
                // Scalar SSA regions use the 256-thread pointwise helper.
                // Standalone elementwise and materialized-view dispatch still
                // use cudarc's 1,024-thread for_num_elems helper.
                if step.materialized_view.is_some()
                    || (step.action == PtxPlanAction::Kernel
                        && standalone_elementwise(&graph[step.node]))
                {
                    features.threads_per_block = 1024;
                }
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
                        features.reduction_extent = region.input_shape[region.axis];
                        features.reduction_stride = count(&region.input_shape[region.axis + 1..]);
                        for input in &region.inputs {
                            input_features(&mut features, &input.tensor);
                        }
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
                        features.full_domain_passes = visits;
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
                        features.reduction_lanes = threads;
                        features.threads_per_block = match region.schedule {
                            ReductionSchedule::Serial => region.serial_block_threads as usize,
                            ReductionSchedule::Subgroup { width } => {
                                width as usize * region.schedule.fibers_per_block() as usize
                            }
                            ReductionSchedule::Block { threads } => threads as usize,
                        };
                        features.active_threads = reduced.saturating_mul(threads);
                        if threads == 1 {
                            let block_threads = features.threads_per_block;
                            set_serial_geometry_features(&mut features, region, block_threads);
                        }
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
                    PtxPlanAction::PointwiseRegion(index) => {
                        for input in &self.regions[index].inputs {
                            input_features(&mut features, &input.tensor);
                        }
                    }
                    PtxPlanAction::OnlineRegion(index) => {
                        let region = &self.online_regions[index];
                        for input in &region.inputs {
                            input_features(&mut features, &input.tensor);
                        }
                        if let Some(threads) = region.block_threads() {
                            features.threads_per_block = threads as usize;
                            let width = step.shape.last().copied().unwrap_or(1);
                            features.active_threads = elements
                                .checked_div(width)
                                .unwrap_or(0)
                                .saturating_mul(threads as usize);
                        }
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
                    PtxPlanAction::Kernel
                        if matches!(graph[step.node], TensorGraphNode::ReduceAxis { .. }) =>
                    {
                        // A bare reduction is dispatched through the legacy
                        // strict kernel, not through a ReductionRegion. It still
                        // visits every input element in one thread per output.
                        if let TensorGraphNode::ReduceAxis { axis, .. } = &graph[step.node]
                            && let Some(&input) = graph.inputs(step.node).first()
                        {
                            let input_shape = graph[input].shape();
                            let full = count(input_shape);
                            features.threads_per_block = 1024;
                            features.reduction_extent = input_shape[*axis];
                            features.reduction_stride = count(&input_shape[*axis + 1..]);
                            features.reduction_lanes = 1;
                            features.full_domain_passes = 1;
                            features.weighted_operations = full;
                            features.address_operations =
                                full.saturating_mul(input_shape.len().max(1));
                            features.traffic_bytes = full.saturating_mul(bytes).saturating_add(
                                step.output_shapes
                                    .iter()
                                    .map(|shape| count(shape).saturating_mul(bytes))
                                    .fold(0usize, usize::saturating_add),
                            );
                            features.serial_working_set_bytes = full
                                .min(
                                    features
                                        .reduction_extent
                                        .saturating_mul(features.threads_per_block),
                                )
                                .saturating_mul(bytes);
                            features.estimated_registers_per_thread += 16;
                        }
                    }
                    _ => {
                        if let Some(view) = &step.materialized_view {
                            input_features(&mut features, view);
                        }
                    }
                }
                if elements == 0 {
                    features.launches = 0;
                }
                Some(features)
            })
            .collect()
    }
}

/// Only these fields depend on independent serial-fiber block ownership.
/// Keep this shared with local fusion-search geometry overlays.
pub(super) fn set_serial_geometry_features(
    features: &mut FusionFeatures,
    region: &ReductionRegion,
    threads: usize,
) {
    features.threads_per_block = threads;
    let row_elements = region.input_shape[region.axis].saturating_mul(threads);
    let bytes = features.storage_dtype.map_or(0, |dtype| dtype.size_bytes());
    features.serial_working_set_bytes = region
        .inputs
        .iter()
        .filter(|input| input.domain == ReductionInputDomain::Full)
        .map(|input| {
            count(&input.source_shape)
                .min(row_elements)
                .saturating_mul(bytes)
        })
        .fold(0usize, usize::saturating_add);
}

fn standalone_elementwise<D: TileDType>(node: &TensorGraphNode<D>) -> bool {
    match node {
        TensorGraphNode::Unary { .. }
        | TensorGraphNode::Binary { .. }
        | TensorGraphNode::Gt { .. }
        | TensorGraphNode::Mask { .. } => true,
        #[cfg(feature = "fusion")]
        TensorGraphNode::FusedUnary { .. } => true,
        _ => false,
    }
}

fn input_features(features: &mut FusionFeatures, tensor: &VirtualTensor) {
    features.non_identity_inputs +=
        usize::from(!tensor.access.is_identity() || tensor.predicate.is_some());
    features.input_index_operations = features
        .input_index_operations
        .saturating_add(tensor.access.operation_count());
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
            .allocation_bytes(tile.m, tile.k, schedule.operand_dtype)
            .saturating_add(schedule.operand_layouts.1.allocation_bytes(
                tile.k,
                tile.n,
                schedule.operand_dtype,
            )),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::TensorExpr;
    use crate::tile::DType;

    #[test]
    fn strict_reduction_features_follow_selected_serial_geometry_without_execution() {
        let graph: TensorGraph<f32> = TensorExpr::input("x", vec![192, 256])
            .relu()
            .reduce_sum(0)
            .into();
        let mut plan = PtxExecutionPlan::build(&graph).unwrap();
        assert!(plan.fusion_decisions().is_empty());
        let baseline = plan.kernel_features(&graph);
        let reduction = baseline.iter().find(|f| f.reduction_lanes == 1).unwrap();
        assert_eq!(reduction.reduction_extent, 192);
        assert_eq!(reduction.reduction_stride, 256);
        assert_eq!(reduction.threads_per_block, 1024);
        assert_eq!(reduction.active_threads, 256);
        assert_eq!(reduction.full_domain_passes, 1);
        assert_eq!(reduction.serial_working_set_bytes, 192 * 256 * 4);

        plan.reduction_regions[0].serial_block_threads = 32;
        let small = plan.kernel_features(&graph);
        let reduction = small.iter().find(|f| f.reduction_lanes == 1).unwrap();
        assert_eq!(reduction.threads_per_block, 32);
        assert_eq!(reduction.active_threads, 256);
        assert_eq!(reduction.serial_working_set_bytes, 192 * 32 * 4);
    }

    #[test]
    fn bare_reduction_reports_full_input_work_and_its_own_dispatch_geometry() {
        let input = TensorExpr::<f32>::input("x", vec![192, 256]);
        let bare: TensorGraph<f32> = input.clone().reduce_sum(0).into();
        let fused: TensorGraph<f32> = input.relu().reduce_sum(0).into();
        let bare_plan = PtxExecutionPlan::build(&bare).unwrap();
        assert!(bare_plan.reduction_regions().is_empty());
        assert!(bare_plan.steps().iter().any(|step| {
            step.action == PtxPlanAction::Kernel
                && matches!(bare[step.node], TensorGraphNode::ReduceAxis { .. })
        }));
        let mut fused_plan = PtxExecutionPlan::build(&fused).unwrap();
        fused_plan.reduction_regions[0].serial_block_threads = 32;
        let bare = bare_plan.kernel_features(&bare);
        let fused = fused_plan.kernel_features(&fused);
        let bare = bare.iter().find(|f| f.reduction_lanes == 1).unwrap();
        let fused = fused.iter().find(|f| f.reduction_lanes == 1).unwrap();
        assert_eq!(bare.reduction_extent, fused.reduction_extent);
        assert_eq!(bare.reduction_stride, fused.reduction_stride);
        assert_eq!(bare.active_threads, fused.active_threads);
        assert_eq!(bare.full_domain_passes, 1);
        assert_eq!(bare.threads_per_block, 1024);
        assert_eq!(fused.threads_per_block, 32);
        assert_eq!(bare.weighted_operations, 192 * 256);
        assert_eq!(bare.address_operations, 192 * 256 * 2);
        assert_eq!(bare.traffic_bytes, (192 * 256 + 256) * 4);
        assert_eq!(bare.serial_working_set_bytes, 192 * 256 * 4);
    }

    #[test]
    fn virtual_input_maps_are_visible_before_lowering_or_execution() {
        let graph: TensorGraph<f32> = TensorExpr::input("x", vec![8, 16])
            .transpose()
            .relu()
            .exp()
            .into();
        let plan = PtxExecutionPlan::build(&graph).unwrap();
        let features = plan.kernel_features(&graph);
        assert!(features.iter().any(|f| f.non_identity_inputs > 0));
        assert!(features.iter().all(|f| f.reduction_lanes == 0));
        assert!(features.iter().all(|f| f.threads_per_block == 256));
    }

    #[test]
    fn standalone_and_fused_elementwise_geometry_match_their_dispatch_helpers() {
        let input = TensorExpr::<f32>::input("x", vec![8, 16]);
        let standalone: TensorGraph<f32> = input.clone().relu().into();
        let fused: TensorGraph<f32> = input.relu().exp().into();
        let standalone = PtxExecutionPlan::build(&standalone)
            .unwrap()
            .kernel_features(&standalone);
        let fused = PtxExecutionPlan::build(&fused)
            .unwrap()
            .kernel_features(&fused);
        assert_eq!(standalone.len(), 1);
        assert_eq!(fused.len(), 1);
        assert_eq!(standalone[0].threads_per_block, 1024);
        assert_eq!(fused[0].threads_per_block, 256);
    }

    #[test]
    fn recomputed_full_epilogue_reports_two_domain_visits() {
        let x = TensorExpr::<f32>::input("x", vec![8, 16]).exp();
        let graph: TensorGraph<f32> = (x.clone() / x.reduce_sum(1).broadcast(vec![8, 16])).into();
        let plan = PtxExecutionPlan::build(&graph).unwrap();
        assert!(plan.kernel_features(&graph).iter().any(|f| {
            f.reduction_extent == 16 && f.full_domain_passes == 2 && f.repeated_operations > 0
        }));
    }

    #[test]
    fn half_storage_plan_features_do_not_require_fusion_decisions() {
        let graph: TensorGraph<half::bf16> =
            TensorExpr::input("x", vec![8, 16]).relu().exp().into();
        let plan = PtxExecutionPlan::build(&graph).unwrap();
        assert!(plan.fusion_decisions().is_empty());
        let features = plan.kernel_features(&graph);
        assert!(!features.is_empty());
        assert!(
            features
                .iter()
                .all(|f| f.storage_dtype == Some(DType::BF16))
        );
    }
}
