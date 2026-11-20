use super::fusion::outputs;
use super::*;
use crate::tensor::{BinaryOp, UnaryOp};
use crate::tile::{FusionFeatures, MatMulPlan, ReductionInputDomain, TileDType};

/// Dispatch path whose launch geometry the diagnostics describe. This does
/// not select execution behavior or change fusion profitability decisions.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PtxExecutionFlavor {
    #[default]
    Homogeneous,
    FixedBinding,
}

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

    /// Features for a particular dispatch path. Fixed-binding pointwise and
    /// materialized-view steps use 256 threads, while legacy homogeneous
    /// standalone dispatch uses 1,024. Cooperative reduction resources remain
    /// estimates: launch-time function limits can reduce their packed size.
    /// This reports a plan; it does not validate fixed-binding support for it.
    pub fn kernel_features_for_execution<D: TileDType, G>(
        &self,
        graph: &TensorGraph<D, G>,
        flavor: PtxExecutionFlavor,
    ) -> Vec<FusionFeatures> {
        self.fusion_features_for_execution(graph, flavor)
    }

    pub(super) fn fusion_features<D: TileDType, G>(
        &self,
        graph: &TensorGraph<D, G>,
    ) -> Vec<FusionFeatures> {
        self.fusion_features_for_execution(graph, PtxExecutionFlavor::Homogeneous)
    }

    fn fusion_features_for_execution<D: TileDType, G>(
        &self,
        graph: &TensorGraph<D, G>,
        flavor: PtxExecutionFlavor,
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
                if flavor == PtxExecutionFlavor::Homogeneous
                    && (step.materialized_view.is_some()
                        || (step.action == PtxPlanAction::Kernel
                            && standalone_elementwise(&graph[step.node])))
                {
                    features.threads_per_block = 1024;
                }
                let elements = count(&step.shape);
                features.active_threads = elements;
                features.estimated_registers_per_thread =
                    16 + live_values(graph, &step.members) * 2;
                features.unique_input_bytes = step
                    .inputs
                    .iter()
                    .copied()
                    .collect::<HashSet<_>>()
                    .iter()
                    .map(|&node| count(graph[node].shape()).saturating_mul(bytes))
                    .fold(0usize, usize::saturating_add);
                features.output_bytes = step
                    .output_shapes
                    .iter()
                    .map(|shape| count(shape).saturating_mul(bytes))
                    .fold(0usize, usize::saturating_add);
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
                        if let Some(tile) = region.column_tile() {
                            features.threads_per_block = tile.threads as usize;
                            features.shared_bytes_per_block = tile.shared_bytes;
                            features.active_threads =
                                (tile.grid_x as usize).saturating_mul(tile.threads as usize);
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
                            features.threads_per_block = match flavor {
                                PtxExecutionFlavor::Homogeneous => 1024,
                                PtxExecutionFlavor::FixedBinding => 256,
                            };
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
    let m = shape[shape.len() - 2];
    let n = shape[shape.len() - 1];
    let row_tiles = m.div_ceil(tile.m);
    let col_tiles = n.div_ceil(tile.n);
    // Each row of A is staged once per output-column tile; each column of B
    // once per output-row tile. Independent source predicates may remove
    // loads. Broadcast batches still execute separate CTAs, even when the
    // backing tensor was counted only once in unique_input_bytes.
    features.matmul_global_load_bytes = Some(
        col_tiles
            .saturating_mul(m)
            .saturating_mul(k)
            .saturating_add(row_tiles.saturating_mul(k).saturating_mul(n))
            .saturating_mul(batch)
            .saturating_mul(schedule.storage_dtype.size_bytes()),
    );
    features.matmul_shared_operand_bytes = schedule.register_thread_tile().map(|(rows, cols)| {
        row_tiles
            .saturating_mul(col_tiles)
            .saturating_mul(batch)
            .saturating_mul(schedule.thread_count())
            .saturating_mul(k.div_ceil(tile.k))
            .saturating_mul(tile.k)
            .saturating_mul(rows.saturating_add(cols))
            .saturating_mul(4)
    });
    features.active_threads = batch
        .saturating_mul(shape[shape.len() - 2].div_ceil(tile.m))
        .saturating_mul(shape[shape.len() - 1].div_ceil(tile.n))
        .saturating_mul(schedule.thread_count());
    features.estimated_registers_per_thread += 48;
    features.shared_bytes_per_block = schedule.shared_mem_bytes();
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
    use crate::tile::{DType, FusionScorer};

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

    fn enumerated_global_operand_bytes(schedule: MatMulSchedule, batch: usize) -> usize {
        let shape = schedule.logical_shape;
        let tile = schedule.block_tile;
        let mut loads = 0usize;
        for _ in 0..batch {
            for row in (0..shape.m).step_by(tile.m) {
                for col in (0..shape.n).step_by(tile.n) {
                    for reduction in (0..shape.k).step_by(tile.k) {
                        for i in 0..tile.m {
                            for k in 0..tile.k {
                                loads += usize::from(row + i < shape.m && reduction + k < shape.k);
                            }
                        }
                        for k in 0..tile.k {
                            for j in 0..tile.n {
                                loads += usize::from(reduction + k < shape.k && col + j < shape.n);
                            }
                        }
                    }
                }
            }
        }
        loads * schedule.storage_dtype.size_bytes()
    }

    #[test]
    fn matmul_requests_match_predicated_tile_enumeration_for_tails_and_empty_domains() {
        let capabilities = MatMulCapabilities {
            tf32: false,
            f16: false,
            bf16: false,
        };
        let resources = MatMulResources::default();
        let baseline = MatMulSchedule::select_with_resources(
            4096,
            683,
            256,
            DType::F32,
            capabilities,
            MatMulPrecision::StrictF32,
            resources,
        );
        let outer = baseline.with_experimental_register_tile(resources);
        assert_eq!(baseline.register_thread_tile(), Some((4, 1)));
        assert_eq!(outer.register_thread_tile(), Some((4, 4)));
        for mut schedule in [baseline, outer] {
            for (m, n, k, batch) in [
                (33, 67, 35, 2),
                (64, 64, 32, 1),
                (0, 67, 35, 2),
                (33, 0, 35, 2),
                (33, 67, 0, 2),
                (33, 67, 35, 0),
            ] {
                schedule.logical_shape = crate::tile::MatMulTile { m, n, k };
                let mut features = FusionFeatures {
                    logical_shape: vec![batch, m, n],
                    ..FusionFeatures::default()
                };
                matmul_features(&mut features, schedule, batch * m * n, k);
                assert_eq!(
                    features.matmul_global_load_bytes,
                    Some(enumerated_global_operand_bytes(schedule, batch))
                );

                // Enumerate each thread's actual unique row/column operands.
                // The K loop executes the complete padded tile even for tails.
                let (thread_rows, thread_cols) = schedule.register_thread_tile().unwrap();
                let tile = schedule.block_tile;
                let mut per_k = 0usize;
                for thread in 0..schedule.thread_count() {
                    let lane_row = thread / (tile.n / thread_cols);
                    let lane_col = thread % (tile.n / thread_cols);
                    let mut a_rows = HashSet::new();
                    let mut b_cols = HashSet::new();
                    for slot in 0..thread_rows * thread_cols {
                        a_rows.insert(lane_row + (slot / thread_cols) * (tile.m / thread_rows));
                        b_cols.insert(lane_col + (slot % thread_cols) * (tile.n / thread_cols));
                    }
                    per_k += a_rows.len() + b_cols.len();
                }
                let mut enumerated_shared = 0usize;
                for _ in 0..batch {
                    for _ in (0..m).step_by(tile.m) {
                        for _ in (0..n).step_by(tile.n) {
                            for _ in (0..k).step_by(tile.k) {
                                enumerated_shared += per_k * tile.k * 4;
                            }
                        }
                    }
                }
                assert_eq!(
                    features.matmul_shared_operand_bytes,
                    Some(enumerated_shared)
                );
            }
        }
    }

    #[test]
    fn batched_transposed_broadcast_gemm_distinguishes_storage_and_tile_requests() {
        let graph: TensorGraph<f32> = TensorExpr::input("a", vec![2, 33, 35])
            .matmul(TensorExpr::input("b", vec![67, 35]).transpose())
            .into();
        let plan = PtxExecutionPlan::build(&graph).unwrap();
        let features = plan.kernel_features(&graph);
        let gemm = features
            .iter()
            .find(|f| f.matmul_global_load_bytes.is_some())
            .unwrap();
        assert_eq!(gemm.unique_input_bytes, (2 * 33 * 35 + 67 * 35) * 4);
        assert_eq!(gemm.output_bytes, 2 * 33 * 67 * 4);
        let schedule = plan.matmul_regions()[0].schedule;
        assert_eq!(
            gemm.matmul_global_load_bytes,
            Some(enumerated_global_operand_bytes(schedule, 2))
        );
        assert!(gemm.matmul_global_load_bytes.unwrap() > gemm.unique_input_bytes);
    }

    #[test]
    fn explicit_execution_flavor_changes_diagnostics_without_changing_default_score() {
        let graph: TensorGraph<f32> = TensorExpr::input("x", vec![8, 16]).relu().into();
        let plan = PtxExecutionPlan::build(&graph).unwrap();
        let legacy = plan.kernel_features(&graph);
        let explicit = plan.kernel_features_for_execution(&graph, PtxExecutionFlavor::Homogeneous);
        assert_eq!(legacy, explicit);
        let fixed = plan.kernel_features_for_execution(&graph, PtxExecutionFlavor::FixedBinding);
        assert_eq!(legacy[0].threads_per_block, 1024);
        assert_eq!(fixed[0].threads_per_block, 256);
        assert_eq!(legacy[0].traffic_bytes, fixed[0].traffic_bytes);
        assert_eq!(plan.kernel_features(&graph), legacy);
        assert!(legacy[0].matmul_global_load_bytes.is_none());
        assert!(legacy[0].matmul_shared_operand_bytes.is_none());
    }

    #[test]
    fn diagnostic_traffic_does_not_recalibrate_the_analytical_scorer() {
        let scorer = crate::tile::AnalyticalFusionScorer::default();
        let baseline = FusionFeatures {
            traffic_bytes: 1024,
            threads_per_block: 256,
            active_threads: 256,
            ..FusionFeatures::default()
        };
        let mut diagnostic = baseline.clone();
        diagnostic.unique_input_bytes = 128;
        diagnostic.output_bytes = 256;
        diagnostic.matmul_global_load_bytes = Some(usize::MAX);
        diagnostic.matmul_shared_operand_bytes = Some(usize::MAX);
        assert_eq!(scorer.score(&[baseline]), scorer.score(&[diagnostic]));
    }

    #[test]
    fn tensor_core_shared_operand_requests_remain_explicitly_unmodeled() {
        let schedule = MatMulSchedule::select_for_dtype(
            64,
            64,
            64,
            DType::BF16,
            MatMulCapabilities {
                tf32: true,
                f16: true,
                bf16: true,
            },
            MatMulPrecision::NativeHalf,
        );
        assert_eq!(schedule.plan, MatMulPlan::TensorCoreBF16);
        let mut features = FusionFeatures {
            logical_shape: vec![64, 64],
            ..FusionFeatures::default()
        };
        matmul_features(&mut features, schedule, 64 * 64, 64);
        assert!(features.matmul_global_load_bytes.is_some());
        assert!(features.matmul_shared_operand_bytes.is_none());
        assert_eq!(features.shared_bytes_per_block, schedule.shared_mem_bytes());
    }
}
