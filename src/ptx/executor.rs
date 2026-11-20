mod indices;
mod plain_bf16_output;
mod replay;
mod square128_streamed;
use indices::{IndexArgument, IndexBuffers};
pub use replay::PtxReplayMetrics;
use replay::{IndexRequirement, ReplayState};

use super::graph::PtxGraph;
use super::plan::{PtxExecutionPlan, PtxFusionPolicy, PtxPlanAction, PtxReductionMode};
use super::readback::materialize_host_view;
use super::target::PtxTarget;
use super::types::{CudaDType, F32};
use crate::Executor;
use crate::alloc::{AllocStats, CudaBufferPool};
use crate::graph::{TensorGraph, TensorGraphNode, WithGrad};
use crate::tile::{AnalyticalFusionScorer, MatMulPipeline, MatMulSharedLayout, TileGraph};
use anyhow::{Context as _, Result};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use num_traits::ToPrimitive;
use petgraph::visit::EdgeRef;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

type RegionValues<T> = Vec<(petgraph::graph::NodeIndex, Arc<CudaSlice<T>>)>;

// This versions the in-memory compilation-key schema, not the crate release.
// Bump it whenever signature encoding or generated-code-affecting inputs change.
const PTX_GRAPH_SIGNATURE_MAGIC: &[u8] = b"tnsr-ptx-graph";
const PTX_GRAPH_SIGNATURE_VERSION: u8 = 50;

/// Measurements from the most recent successful PTX compilation.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PtxCompileMetrics {
    pub graph_nodes: usize,
    /// Kernel-producing plan steps; identical steps may share entry points.
    pub generated_kernels: usize,
    pub unique_kernels: usize,
    pub module_load_time: Duration,
    pub ptx_source_bytes: usize,
    pub compile_time: Duration,
    pub index_optimization_time: Duration,
    pub fusion_selection_time: Duration,
}

/// Measurements from the most recent successful graph execution.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PtxExecutionMetrics {
    pub kernel_launches: usize,
    pub materialized_values: usize,
    pub materialized_bytes: usize,
    pub intermediate_materialized_bytes: usize,
    pub device_copies: usize,
    /// Host graph launches; `kernel_launches` still counts physical GPU kernels.
    pub graph_launches: usize,
}

impl PtxExecutionMetrics {
    fn add_work(&mut self, other: &Self) {
        self.kernel_launches += other.kernel_launches;
        self.materialized_values += other.materialized_values;
        self.materialized_bytes += other.materialized_bytes;
        self.intermediate_materialized_bytes += other.intermediate_materialized_bytes;
        self.device_copies += other.device_copies;
        self.graph_launches += other.graph_launches;
    }
}

/// Unmeasured kernel alternatives; each option participates in compiled identity.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct ExperimentalKernelOptions {
    indexed_loss_cooperative_ordered_backward: bool,
    indexed_loss_cooperative_ordered_forward: bool,
    shared_memory_reuse: bool,
    online_weight_sharing: bool,
    narrow_bf16_tile: bool,
    bf16_eight_compute_warps: bool,
    ordinary_bf16_four_warp_block: bool,
    plain_bf16_output: bool,
    medium_bf16_tile_policy: crate::tile::MediumBf16TilePolicy,
    column_reduction_policy: crate::tile::ColumnReductionPolicy,
    large_bf16_tile_policy: crate::tile::LargeBf16TilePolicy,
    batched_attention_bf16_tile_policy: crate::tile::BatchedAttentionBf16TilePolicy,
}

/// PTX executor that compiles TensorGraph → TileGraph → PtxGraph → PTX string
/// and executes kernels via cudarc
pub struct PtxExecutor<D: CudaDType = F32> {
    device: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    stream_gate: Arc<Mutex<()>>,
    // Destroy graphs before any referenced bindings, modules or workspace.
    replay: Option<ReplayState<D>>,
    // These private allocations outlive every captured graph that references them.
    exact_indices: IndexBuffers,
    index_plan: crate::graph::indices::IndexPlan,
    replay_enabled: bool,
    replay_blocked: bool,
    replay_metrics: PtxReplayMetrics,
    capturing: Cell<bool>,
    allocation_frozen: Cell<bool>,
    replay_frame_active: bool,
    capture_indices: RefCell<Vec<IndexRequirement>>,
    pending_replay_work: PtxExecutionMetrics,
    bindings: HashMap<petgraph::graph::NodeIndex, Arc<CudaSlice<D::HostType>>>,
    sources: super::sources::SourceCache<D>,
    module: Option<Arc<CudaModule>>,
    functions: RefCell<HashMap<String, Arc<CudaFunction>>>,
    values: HashMap<petgraph::graph::NodeIndex, Arc<CudaSlice<D::HostType>>>,
    pool: RefCell<CudaBufferPool<D::HostType>>,
    stats: AllocStats,
    /// Stores the PtxGraph to access kernel names during execution
    ptx_graph: Option<Box<PtxGraph>>,
    execution_plan: Option<Arc<PtxExecutionPlan>>,
    compilation_signature: Option<Vec<u8>>,
    compilation_count: usize,
    square128_streamed_fallbacks: usize,
    compile_metrics: PtxCompileMetrics,
    execution_metrics: PtxExecutionMetrics,
    kernel_launches: Cell<usize>,
    reduction_mode: PtxReductionMode,
    fusion_policy: PtxFusionPolicy,
    serial_reduction_block_threads: Option<u32>,
    target: PtxTarget,
    matmul_precision: crate::tile::MatMulPrecision,
    matmul_shared_layout: MatMulSharedLayout,
    matmul_pipeline: MatMulPipeline,
    matmul_grid_group_m: usize,
    experimental_register_tiles: bool,
    experimental_kernel_options: ExperimentalKernelOptions,
}

impl<D: CudaDType> Default for PtxExecutor<D> {
    fn default() -> Self {
        Self::new_for_dtype()
    }
}

impl PtxExecutor<F32> {
    /// Create a new PTX executor, panicking if CUDA initialization fails.
    ///
    /// For fallible initialization, use [`PtxExecutor::try_new`].
    pub fn new() -> Self {
        Self::new_for_dtype()
    }

    /// Create a PTX executor with an explicit reduction ordering policy.
    pub fn new_with_reduction_mode(reduction_mode: PtxReductionMode) -> Self {
        Self::try_new_with_reduction_mode(reduction_mode)
            .expect("Failed to initialize PTX executor")
    }

    /// Try to create a new PTX executor, returning an error if initialization fails.
    pub fn try_new() -> Result<Self> {
        Self::try_new_for_dtype()
    }
}

impl<D: CudaDType> PtxExecutor<D> {
    /// Create an executor for a specific storage dtype.
    pub fn new_for_dtype() -> Self {
        Self::try_new_for_dtype().expect("Failed to initialize PTX executor")
    }

    /// Try to create an executor for a specific storage dtype.
    pub fn try_new_for_dtype() -> Result<Self> {
        Self::try_new_with_reduction_mode(PtxReductionMode::Strict)
    }

    /// Initialize this storage dtype with an explicit reduction policy.
    pub fn try_new_with_reduction_mode(reduction_mode: PtxReductionMode) -> Result<Self> {
        let device = CudaContext::new(0).context("Failed to initialize CUDA device 0")?;
        Self::with_context(device, reduction_mode)
    }

    pub(crate) fn with_context(
        device: Arc<CudaContext>,
        reduction_mode: PtxReductionMode,
    ) -> Result<Self> {
        Self::with_stream(
            device.default_stream(),
            reduction_mode,
            Arc::new(Mutex::new(())),
        )
    }

    pub(crate) fn with_stream(
        stream: Arc<CudaStream>,
        reduction_mode: PtxReductionMode,
        stream_gate: Arc<Mutex<()>>,
    ) -> Result<Self> {
        let device = stream.context().clone();
        let target = PtxTarget::from_context(&device)?;
        anyhow::ensure!(
            <D::HostType as crate::tile::TileDType>::TILE_DTYPE != crate::tile::DType::BF16
                || target.compute_capability.0 >= 8,
            "Native BF16 PTX storage requires SM80 or newer; use the CPU backend on older GPUs"
        );
        Ok(PtxExecutor {
            device,
            stream,
            stream_gate,
            replay: None,
            exact_indices: IndexBuffers::default(),
            index_plan: crate::graph::indices::IndexPlan::default(),
            replay_enabled: false,
            replay_blocked: false,
            replay_metrics: PtxReplayMetrics::default(),
            capturing: Cell::new(false),
            allocation_frozen: Cell::new(false),
            replay_frame_active: false,
            capture_indices: RefCell::new(Vec::new()),
            pending_replay_work: PtxExecutionMetrics::default(),
            bindings: HashMap::new(),
            sources: super::sources::SourceCache::default(),
            module: None,
            functions: RefCell::new(HashMap::new()),
            values: HashMap::new(),
            pool: RefCell::new(CudaBufferPool::default()),
            stats: AllocStats::default(),
            ptx_graph: None,
            execution_plan: None,
            compilation_signature: None,
            compilation_count: 0,
            square128_streamed_fallbacks: 0,
            compile_metrics: PtxCompileMetrics::default(),
            execution_metrics: PtxExecutionMetrics::default(),
            kernel_launches: Cell::new(0),
            reduction_mode,
            fusion_policy: PtxFusionPolicy::CostAware,
            serial_reduction_block_threads: None,
            target,
            matmul_precision: crate::tile::MatMulPrecision::AllowTf32,
            matmul_shared_layout: MatMulSharedLayout::Auto,
            matmul_pipeline: MatMulPipeline::Auto,
            matmul_grid_group_m: 1,
            experimental_register_tiles: false,
            experimental_kernel_options: ExperimentalKernelOptions::default(),
        })
    }

    /// Select matmul arithmetic for subsequent compilations. Changing the policy
    /// invalidates the compilation signature so execute recompiles cached graphs.
    pub fn set_matmul_precision(&mut self, precision: crate::tile::MatMulPrecision) {
        let gate = self.stream_gate.clone();
        let _guard = lock_stream(&gate);
        if self.matmul_precision != precision {
            self.invalidate_replay();
            self.matmul_precision = precision;
            self.compilation_signature = None;
        }
    }

    /// Choose shared operand layouts for subsequent compilations.
    pub fn set_matmul_shared_layout(&mut self, layout: MatMulSharedLayout) {
        let gate = self.stream_gate.clone();
        let _guard = lock_stream(&gate);
        if self.matmul_shared_layout != layout {
            self.invalidate_replay();
            self.matmul_shared_layout = layout;
            self.compilation_signature = None;
        }
    }

    /// Select synchronous staging or a bounded asynchronous pipeline.
    pub fn set_matmul_pipeline(&mut self, pipeline: MatMulPipeline) {
        let gate = self.stream_gate.clone();
        let _guard = lock_stream(&gate);
        if self.matmul_pipeline != pipeline {
            self.invalidate_replay();
            self.matmul_pipeline = pipeline;
            self.compilation_signature = None;
        }
    }

    /// Enable a larger scalar register tile for controlled experiments.
    pub fn set_experimental_register_tiles(&mut self, enabled: bool) {
        let gate = self.stream_gate.clone();
        let _guard = lock_stream(&gate);
        if self.experimental_register_tiles != enabled {
            self.invalidate_replay();
            self.experimental_register_tiles = enabled;
            self.compilation_signature = None;
        }
    }

    /// Experimentally reuse dead operand shared storage for the GEMM epilogue.
    /// Disabled by default pending device correctness and performance qualification.
    pub fn set_experimental_shared_memory_reuse(&mut self, enabled: bool) {
        let gate = self.stream_gate.clone();
        let _guard = lock_stream(&gate);
        if self.experimental_kernel_options.shared_memory_reuse != enabled {
            self.invalidate_replay();
            self.experimental_kernel_options.shared_memory_reuse = enabled;
            self.compilation_signature = None;
        }
    }

    /// Opt into a synchronous 32x32x32 tile for eligible narrow BF16 GEMMs.
    pub fn set_experimental_narrow_bf16_tile(&mut self, enabled: bool) {
        let gate = self.stream_gate.clone();
        let _guard = lock_stream(&gate);
        if self.experimental_kernel_options.narrow_bf16_tile != enabled {
            self.invalidate_replay();
            self.experimental_kernel_options.narrow_bf16_tile = enabled;
            self.compilation_signature = None;
        }
    }

    /// Select bounded rank-two BF16 reuse tiles before ordinary layout/pipeline policy.
    pub fn set_experimental_medium_bf16_tile_policy(
        &mut self,
        policy: crate::tile::MediumBf16TilePolicy,
    ) {
        let gate = self.stream_gate.clone();
        let _guard = lock_stream(&gate);
        if self.experimental_kernel_options.medium_bf16_tile_policy != policy {
            self.invalidate_replay();
            self.experimental_kernel_options.medium_bf16_tile_policy = policy;
            self.compilation_signature = None;
        }
    }

    /// Select ordinary BF16 rectangles only with explicit physical scratch reuse.
    pub fn set_experimental_large_bf16_tile_policy(
        &mut self,
        policy: crate::tile::LargeBf16TilePolicy,
    ) {
        let gate = self.stream_gate.clone();
        let _guard = lock_stream(&gate);
        if self.experimental_kernel_options.large_bf16_tile_policy != policy {
            self.invalidate_replay();
            self.experimental_kernel_options.large_bf16_tile_policy = policy;
            self.compilation_signature = None;
        }
    }

    /// Select exact S512/D64 batched BF16 rectangles after complete view/copy
    /// and explicit two-stage resource proofs. Defaults to Off.
    pub fn set_experimental_batched_attention_bf16_tile_policy(
        &mut self,
        policy: crate::tile::BatchedAttentionBf16TilePolicy,
    ) {
        let gate = self.stream_gate.clone();
        let _guard = lock_stream(&gate);
        if self
            .experimental_kernel_options
            .batched_attention_bf16_tile_policy
            != policy
        {
            self.invalidate_replay();
            self.experimental_kernel_options
                .batched_attention_bf16_tile_policy = policy;
            self.compilation_signature = None;
        }
    }

    /// Tile independent columns without changing the Block128 reduction tree.
    pub fn set_experimental_column_reduction_policy(
        &mut self,
        policy: crate::tile::ColumnReductionPolicy,
    ) {
        let gate = self.stream_gate.clone();
        let _guard = lock_stream(&gate);
        if self.experimental_kernel_options.column_reduction_policy != policy {
            self.invalidate_replay();
            self.experimental_kernel_options.column_reduction_policy = policy;
            self.compilation_signature = None;
        }
    }

    /// Repartition eligible BF16 64x64x32 tiles across all eight compute warps.
    pub fn set_experimental_bf16_eight_compute_warps(&mut self, enabled: bool) {
        let gate = self.stream_gate.clone();
        let _guard = lock_stream(&gate);
        if self.experimental_kernel_options.bf16_eight_compute_warps != enabled {
            self.invalidate_replay();
            self.experimental_kernel_options.bf16_eight_compute_warps = enabled;
            self.compilation_signature = None;
        }
    }

    /// Remove scalar output bounds only after an exact full-MN BF16 proof.
    /// Storage rounding, input staging, arithmetic and launch geometry stay unchanged.
    pub fn set_experimental_plain_bf16_output(&mut self, enabled: bool) {
        let gate = self.stream_gate.clone();
        let _guard = lock_stream(&gate);
        if self.experimental_kernel_options.plain_bf16_output != enabled {
            self.invalidate_replay();
            self.experimental_kernel_options.plain_bf16_output = enabled;
            self.compilation_signature = None;
        }
    }
    pub(crate) fn plain_bf16_output_counts(&self) -> (usize, usize) {
        self.ptx_graph
            .as_ref()
            .map(|g| g.plain_bf16_output_counts())
            .unwrap_or_default()
    }
    pub(crate) fn plain_bf16_output_selected(&self, id: usize) -> bool {
        self.ptx_graph
            .as_ref()
            .is_some_and(|g| g.plain_bf16_output_selected(id))
    }
    /// Keep an eligible ordinary BF16 64x64 tile on four declared block warps.
    /// The increasing-K16 ownership and fragment ILP remain unchanged.
    pub fn set_experimental_ordinary_bf16_four_warp_block(&mut self, enabled: bool) {
        let gate = self.stream_gate.clone();
        let _guard = lock_stream(&gate);
        if self
            .experimental_kernel_options
            .ordinary_bf16_four_warp_block
            != enabled
        {
            self.invalidate_replay();
            self.experimental_kernel_options
                .ordinary_bf16_four_warp_block = enabled;
            self.compilation_signature = None;
        }
    }

    /// Cooperatively compute V4096 backward CE maxima/exponentials while
    /// retaining the original increasing-column F32 sum and gradient arithmetic.
    pub fn set_experimental_indexed_loss_cooperative_ordered_backward(&mut self, enabled: bool) {
        let gate = self.stream_gate.clone();
        let _guard = lock_stream(&gate);
        if self
            .experimental_kernel_options
            .indexed_loss_cooperative_ordered_backward
            != enabled
        {
            self.invalidate_replay();
            self.experimental_kernel_options
                .indexed_loss_cooperative_ordered_backward = enabled;
            self.compilation_signature = None;
        }
    }

    pub(crate) fn indexed_loss_cooperative_ordered_backward_selected(
        &self,
        node: crate::graph::NodeIndex,
    ) -> bool {
        self.ptx_graph.as_ref().is_some_and(|g| {
            g.indexed_loss_cooperative_ordered_backward_nodes()
                .contains(&node)
        })
    }

    pub(crate) fn indexed_loss_cooperative_ordered_backward_programs(&self) -> usize {
        self.ptx_graph.as_ref().map_or(0, |g| {
            g.indexed_loss_cooperative_ordered_backward_programs()
        })
    }

    /// Cooperatively compute V4096 forward CE maxima/exponentials while
    /// retaining the original increasing-column F32 sum and gradient arithmetic.
    pub fn set_experimental_indexed_loss_cooperative_ordered_forward(&mut self, enabled: bool) {
        let gate = self.stream_gate.clone();
        let _guard = lock_stream(&gate);
        if self
            .experimental_kernel_options
            .indexed_loss_cooperative_ordered_forward
            != enabled
        {
            self.invalidate_replay();
            self.experimental_kernel_options
                .indexed_loss_cooperative_ordered_forward = enabled;
            self.compilation_signature = None;
        }
    }

    pub(crate) fn indexed_loss_cooperative_ordered_forward_selected(
        &self,
        node: crate::graph::NodeIndex,
    ) -> bool {
        self.ptx_graph.as_ref().is_some_and(|g| {
            g.indexed_loss_cooperative_ordered_forward_nodes()
                .contains(&node)
        })
    }

    pub(crate) fn indexed_loss_cooperative_ordered_forward_programs(&self) -> usize {
        self.ptx_graph
            .as_ref()
            .map_or(0, |g| g.indexed_loss_cooperative_ordered_forward_programs())
    }

    /// Experimentally share exponential weights between Online output lanes.
    /// Applies only to recognized cooperative online regions, not cached decoding.
    pub fn set_experimental_online_weight_sharing(&mut self, enabled: bool) {
        let gate = self.stream_gate.clone();
        let _guard = lock_stream(&gate);
        if self.experimental_kernel_options.online_weight_sharing != enabled {
            self.invalidate_replay();
            self.experimental_kernel_options.online_weight_sharing = enabled;
            self.compilation_signature = None;
        }
    }

    /// Group adjacent GEMM row tiles to improve operand reuse in cache.
    /// One keeps the original grid traversal; arithmetic within each tile is unchanged.
    pub fn set_matmul_grid_group_m(&mut self, group: usize) -> Result<()> {
        crate::tile::MatMulSchedule::validate_grid_group_m(group)?;
        let gate = self.stream_gate.clone();
        let _guard = lock_stream(&gate);
        if self.matmul_grid_group_m != group {
            self.invalidate_replay();
            self.matmul_grid_group_m = group;
            self.compilation_signature = None;
        }
        Ok(())
    }

    /// Use cost-aware selection or the previous greedy fusion for ablations.
    pub fn set_fusion_policy(&mut self, policy: PtxFusionPolicy) {
        let gate = self.stream_gate.clone();
        let _guard = lock_stream(&gate);
        if self.fusion_policy != policy {
            self.invalidate_replay();
            self.fusion_policy = policy;
            self.compilation_signature = None;
        }
    }

    /// Override the serial reduction-region block size for controlled profiling.
    /// `None` restores planner selection. Each thread still visits its own
    /// output fiber in increasing reduction-index order under Strict policy.
    pub fn set_serial_reduction_block_threads(&mut self, threads: Option<u32>) -> Result<()> {
        if let Some(threads) = threads {
            crate::tile::ReductionRegion::validate_serial_block_threads(threads)?;
        }
        let gate = self.stream_gate.clone();
        let _guard = lock_stream(&gate);
        if self.serial_reduction_block_threads != threads {
            self.invalidate_replay();
            self.serial_reduction_block_threads = threads;
            self.compilation_signature = None;
        }
        Ok(())
    }

    fn take_buffer(&self, len: usize) -> Result<CudaSlice<D::HostType>> {
        if self.allocation_frozen.get() {
            return self.pool.borrow_mut().take_cached(len);
        }
        self.pool.borrow_mut().take(&self.stream.clone(), len)
    }

    fn validate_indices(
        &self,
        indices: &Arc<CudaSlice<D::HostType>>,
        upper: usize,
        operation: &str,
    ) -> Result<()> {
        if let Some(result) = self.sources.validate_input(indices, upper, operation) {
            if self.capturing.get() {
                self.capture_indices.borrow_mut().push(IndexRequirement {
                    key: Arc::as_ptr(indices) as usize,
                    upper,
                    operation: operation.to_owned(),
                });
            }
            return result;
        }
        anyhow::ensure!(
            !self.capturing.get(),
            "computed indices require uncaptured validation"
        );
        let mut host_indices = vec![D::HostType::default(); indices.len()];
        if !host_indices.is_empty() {
            self.stream
                .clone()
                .memcpy_dtoh(indices.as_ref(), &mut host_indices)
                .with_context(|| format!("Failed to validate PTX {operation} indices"))?;
        }
        for (position, value) in host_indices.iter().enumerate() {
            let value = value
                .to_f32()
                .context("Index cannot be represented as f32")?;
            anyhow::ensure!(
                value.is_finite(),
                "{operation} index at position {position} must be finite, got {value}"
            );
            anyhow::ensure!(
                value >= 0.0,
                "{operation} index at position {position} must be non-negative, got {value}"
            );
            anyhow::ensure!(
                value.fract() == 0.0,
                "{operation} index at position {position} must be an integer, got {value}"
            );
            let index = value as usize;
            anyhow::ensure!(
                index as f32 == value,
                "{operation} index at position {position} cannot be represented as usize: {value}"
            );
            anyhow::ensure!(
                index < upper,
                "{operation} index at position {position} is out of range: {index} >= {upper}"
            );
        }
        Ok(())
    }

    /// Memory allocation statistics gathered during execution.
    pub fn stats(&self) -> &AllocStats {
        &self.stats
    }

    /// Number of PTX modules successfully compiled and loaded by this executor.
    pub fn compilation_count(&self) -> usize {
        self.compilation_count
    }

    pub fn compile_metrics(&self) -> &PtxCompileMetrics {
        &self.compile_metrics
    }

    pub fn execution_metrics(&self) -> &PtxExecutionMetrics {
        &self.execution_metrics
    }

    pub(crate) fn execution_stream(&self) -> Arc<CudaStream> {
        self.stream.clone()
    }

    pub(crate) fn large_square128_streamed_counts(&self) -> (usize, usize) {
        (
            self.execution_plan
                .as_ref()
                .map_or(0, |p| p.large_square128_streamed_count()),
            self.square128_streamed_fallbacks,
        )
    }

    pub fn execution_plan(&self) -> Option<&PtxExecutionPlan> {
        self.execution_plan.as_deref()
    }

    /// Return the generated PTX module for diagnostics after compilation.
    pub fn module_source(&self) -> Option<String> {
        self.ptx_graph.as_ref().map(|graph| graph.module_source())
    }

    pub(crate) fn planned_kernel_function(
        &self,
        step: &super::PtxPlanStep,
    ) -> Result<CudaFunction> {
        let graph = self.ptx_graph.as_ref().context("no compiled PTX graph")?;
        let name = match step.action {
            super::PtxPlanAction::Kernel => graph.kernel_name(step.node),
            super::PtxPlanAction::PointwiseRegion(id) => graph.region_kernel_name(id),
            super::PtxPlanAction::ReductionRegion(id) => graph.reduction_region_kernel_name(id),
            super::PtxPlanAction::MatMulRegion(id) => graph.matmul_region_kernel_name(id),
            _ => None,
        }
        .context("unsupported prepared linear kernel")?;
        Ok(self
            .module
            .as_ref()
            .context("no compiled module")?
            .load_function(name)?)
    }

    pub(crate) fn conv_weight_tree_launch_config(
        &self,
        node: petgraph::graph::NodeIndex,
        len: usize,
    ) -> Result<Option<LaunchConfig>> {
        let threads = self
            .ptx_graph
            .as_ref()
            .context("missing compiled PTX graph")?
            .conv_weight_threads(node);
        if threads == 1 {
            return Ok(None);
        }
        let len = u32::try_from(len)
            .context("Conv2dBackwardWeight output is too large for a CUDA launch")?;
        Ok(Some(LaunchConfig {
            grid_dim: (len, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        }))
    }

    pub(crate) fn indexed_loss_row_launch_config(
        &self,
        node: petgraph::graph::NodeIndex,
        len: usize,
    ) -> Result<Option<LaunchConfig>> {
        self.ptx_graph
            .as_ref()
            .context("missing compiled PTX graph")?
            .indexed_loss_row_launch(node, len)
    }

    /// Describe the current physical PTX execution plan.
    pub fn describe_plan<G>(&self, graph: &TensorGraph<D::HostType, G>) -> Result<String> {
        let supplied_signature = graph_compilation_signature(
            graph,
            self.reduction_mode,
            self.target,
            self.matmul_precision,
            self.matmul_shared_layout,
            self.matmul_pipeline,
            (
                self.fusion_policy,
                self.serial_reduction_block_threads,
                self.matmul_grid_group_m,
                self.experimental_register_tiles,
                self.experimental_kernel_options,
            ),
        );
        let compiled_signature = self
            .compilation_signature
            .as_deref()
            .context("No compiled graph signature available. Call compile_owned() first.")?;
        anyhow::ensure!(
            compiled_signature == supplied_signature.as_slice(),
            "Supplied graph structure does not match the compiled PTX module"
        );
        let ptx_graph = self
            .ptx_graph
            .as_ref()
            .context("No PTX graph available. Call compile_owned() first.")?;
        let mut description = format!(
            "PTX plan: {} graph nodes, {} physical kernels\n",
            graph.graph.node_count(),
            self.compile_metrics.generated_kernels
        );
        let plan = self
            .execution_plan
            .as_ref()
            .context("No PTX execution plan available. Call compile_owned() first.")?;
        for (step, plan_step) in plan.steps().iter().enumerate() {
            let node_index = plan_step.node;
            let action = match plan_step.action {
                PtxPlanAction::Upload => "upload".to_string(),
                PtxPlanAction::DeviceCopy => "device-copy".to_string(),
                PtxPlanAction::VirtualView => "virtual-view".to_string(),
                PtxPlanAction::Kernel => {
                    let name = ptx_graph
                        .kernel_name(node_index)
                        .context("Missing compiled kernel in unfused plan")?;
                    if matches!(
                        graph[node_index],
                        TensorGraphNode::Conv2dBackwardWeight { .. }
                    ) {
                        format!(
                            "kernel {name} conv_weight_threads={}",
                            ptx_graph.conv_weight_threads(node_index)
                        )
                    } else {
                        format!("kernel {name}")
                    }
                }
                PtxPlanAction::PointwiseRegion(region_id) => format!(
                    "kernel {}",
                    ptx_graph
                        .region_kernel_name(region_id)
                        .context("Missing compiled pointwise region kernel")?
                ),
                PtxPlanAction::ReductionRegion(region_id) => {
                    let region = plan
                        .reduction_regions()
                        .get(region_id)
                        .context("Missing reduction region schedule")?;
                    let schedule = region.schedule;
                    let geometry = if let Some(tile) = region.column_tile() {
                        format!(
                            " column_tile={} column_threads={}",
                            tile.columns_per_block, tile.threads
                        )
                    } else if schedule == crate::tile::ReductionSchedule::Serial {
                        format!(" serial_block_threads={}", region.serial_block_threads)
                    } else {
                        String::new()
                    };
                    format!(
                        "kernel {} schedule={schedule:?}{geometry}",
                        ptx_graph
                            .reduction_region_kernel_name(region_id)
                            .context("Missing compiled reduction region kernel")?
                    )
                }
                PtxPlanAction::OnlineRegion(id) => format!(
                    "kernel {} schedule=Online",
                    ptx_graph
                        .region_kernel_name(plan.regions().len() + id)
                        .context("Missing online region kernel")?
                ),
                PtxPlanAction::MatMulRegion(region_id) => format!(
                    "kernel {} schedule={:?}",
                    ptx_graph
                        .matmul_region_kernel_name(region_id)
                        .context("Missing compiled matmul region kernel")?,
                    plan.matmul_regions()[region_id].schedule,
                ),
            };
            writeln!(
                description,
                "{step:04}: node {:04} {:<32} shape={:?} outputs={:?} action={action}",
                node_index.index(),
                plan_step.operation,
                plan_step.shape,
                plan_step.outputs
            )?;
        }
        Ok(description)
    }

    fn record_kernel_launch(&self) {
        self.kernel_launches.set(self.kernel_launches.get() + 1);
    }

    fn kernel_function(&self, module: &Arc<CudaModule>, name: &str) -> Result<Arc<CudaFunction>> {
        let mut functions = self.functions.borrow_mut();
        if let Some(function) = functions.get(name) {
            return Ok(function.clone());
        }
        let function = Arc::new(module.load_function(name)?);
        functions.insert(name.to_owned(), function.clone());
        Ok(function)
    }

    fn recycle_value(&self, value: Arc<CudaSlice<D::HostType>>) -> usize {
        let bytes = value.len() * std::mem::size_of::<D::HostType>();
        if let Ok(value) = Arc::try_unwrap(value) {
            self.pool.borrow_mut().give(value);
            bytes
        } else {
            0
        }
    }

    /// Reset allocation counters without discarding pooled device buffers.
    pub fn reset_stats(&mut self) {
        self.stats.reset();
        self.pool.get_mut().reset_stats();
        self.exact_indices.allocated_bytes = 0;
        self.exact_indices.allocations = 0;
    }

    /// Discard cached device buffers to release VRAM held by the executor.
    pub fn clear_pool(&mut self) {
        let gate = self.stream_gate.clone();
        let _guard = lock_stream(&gate);
        self.invalidate_replay();
        self.exact_indices.buffers.clear();
        self.pool.get_mut().clear();
    }

    /// Release pinned gradient buffers before the next execution.
    pub fn release_gradients(&mut self, graph: &TensorGraph<D::HostType, WithGrad>) {
        let gate = self.stream_gate.clone();
        let _guard = lock_stream(&gate);
        self.invalidate_replay();
        for grad_node in graph.gradient_metadata().param_to_grad.values() {
            if let Some(buf) = self.values.remove(grad_node) {
                self.recycle_value(buf);
            }
        }
    }

    /// Compile an owned TensorGraph to PTX, sharing its tensor buffers during lowering.
    pub fn compile_owned<G>(&mut self, graph: TensorGraph<D::HostType, G>) -> Result<()> {
        self.compile(&graph)
    }

    /// Compile a borrowed graph without cloning tensor data.
    pub fn compile<G>(&mut self, graph: &TensorGraph<D::HostType, G>) -> Result<()> {
        let gate = self.stream_gate.clone();
        let _guard = lock_stream(&gate);
        self.compile_impl(graph)
    }

    /// Compile fixed bindings whose already-rounded source storage differs from
    /// the f32 arithmetic graph. This internal module is never executed through
    /// the homogeneous executor's upload/readback path.
    pub(crate) fn compile_bound_storage<G>(
        &mut self,
        graph: &TensorGraph<D::HostType, G>,
        storage: &HashMap<crate::graph::NodeIndex, crate::tile::DType>,
    ) -> Result<()> {
        let gate = self.stream_gate.clone();
        let _guard = lock_stream(&gate);
        self.compile_impl_with_storage(graph, storage)
    }

    fn compile_impl<G>(&mut self, graph: &TensorGraph<D::HostType, G>) -> Result<()> {
        self.compile_impl_with_storage(graph, &HashMap::new())
    }

    fn compile_impl_with_storage<G>(
        &mut self,
        graph: &TensorGraph<D::HostType, G>,
        storage: &HashMap<crate::graph::NodeIndex, crate::tile::DType>,
    ) -> Result<()> {
        anyhow::ensure!(
            !graph.graph.node_weights().any(|n| matches!(
                n,
                TensorGraphNode::Cast { .. } | TensorGraphNode::Linear { .. }
            )),
            "mixed casts require MixedExecutor"
        );
        let index_plan = crate::graph::indices::IndexPlan::new(graph)?;
        self.invalidate_replay();
        self.exact_indices.buffers.clear();
        let started = Instant::now();
        let graph_nodes = graph.graph.node_count();
        let mut signature = graph_compilation_signature(
            graph,
            self.reduction_mode,
            self.target,
            self.matmul_precision,
            self.matmul_shared_layout,
            self.matmul_pipeline,
            (
                self.fusion_policy,
                self.serial_reduction_block_threads,
                self.matmul_grid_group_m,
                self.experimental_register_tiles,
                self.experimental_kernel_options,
            ),
        );
        if !storage.is_empty() {
            signature.extend_from_slice(b"bound-storage");
            let mut entries: Vec<_> = storage.iter().collect();
            entries.sort_by_key(|(node, _)| node.index());
            for (node, dtype) in entries {
                signature_usize(&mut signature, node.index());
                signature_usize(&mut signature, *dtype as usize);
            }
        }
        let mut execution_plan =
            PtxExecutionPlan::build_with_reduction_mode(graph, self.reduction_mode)?;
        execution_plan.schedule_matmuls_with_tile_policies(
            self.target.matmul_capabilities(),
            self.matmul_precision,
            self.target.matmul_resources,
            self.matmul_shared_layout,
            self.matmul_pipeline,
            self.target.supports_async_copy(),
            self.experimental_register_tiles,
            self.experimental_kernel_options.medium_bf16_tile_policy,
            self.experimental_kernel_options.large_bf16_tile_policy,
            self.experimental_kernel_options.shared_memory_reuse,
        )?;
        let fusion_started = Instant::now();
        if self.fusion_policy == PtxFusionPolicy::CostAware {
            execution_plan.select_fusion_with_serial_threads(
                graph,
                &AnalyticalFusionScorer {
                    multiprocessor_count: self.target.multiprocessor_count,
                    registers_per_sm: self.target.matmul_resources.registers_per_sm,
                    shared_bytes_per_sm: self.target.matmul_resources.shared_bytes_per_sm,
                    threads_per_sm: self.target.matmul_resources.threads_per_sm,
                    ..AnalyticalFusionScorer::default()
                },
                self.serial_reduction_block_threads,
            )?;
        } else if let Some(threads) = self.serial_reduction_block_threads {
            execution_plan.set_serial_reduction_block_threads(threads)?;
        }
        if self.reduction_mode == PtxReductionMode::Online {
            execution_plan.fuse_online(graph)?;
        }
        // Candidate repartitioning can change virtual operand bindings. Recompute
        // copy/layout eligibility with the final plan and the selected precision.
        execution_plan.schedule_matmuls_with_tile_policies(
            self.target.matmul_capabilities(),
            self.matmul_precision,
            self.target.matmul_resources,
            self.matmul_shared_layout,
            self.matmul_pipeline,
            self.target.supports_async_copy(),
            self.experimental_register_tiles,
            self.experimental_kernel_options.medium_bf16_tile_policy,
            self.experimental_kernel_options.large_bf16_tile_policy,
            self.experimental_kernel_options.shared_memory_reuse,
        )?;
        execution_plan.configure_batched_attention_tiles(
            self.experimental_kernel_options
                .batched_attention_bf16_tile_policy,
            self.matmul_shared_layout,
            self.matmul_pipeline,
            self.target.supports_async_copy(),
            self.target.matmul_resources,
            self.experimental_kernel_options.shared_memory_reuse,
        );
        execution_plan.set_matmul_grid_group_m(self.matmul_grid_group_m)?;
        execution_plan.set_matmul_narrow_bf16_tile(
            self.experimental_kernel_options.narrow_bf16_tile,
            self.target.matmul_resources,
        );
        execution_plan.set_bf16_eight_compute_warps(
            self.experimental_kernel_options.bf16_eight_compute_warps,
        );
        execution_plan
            .set_matmul_shared_memory_reuse(self.experimental_kernel_options.shared_memory_reuse);
        execution_plan.configure_ordinary_bf16_four_warp_block(
            self.experimental_kernel_options
                .ordinary_bf16_four_warp_block,
            self.target.matmul_resources,
        );
        execution_plan
            .set_online_weight_sharing(self.experimental_kernel_options.online_weight_sharing);
        let fusion_selection_time = fusion_started.elapsed();
        let square_baseline = if self.experimental_kernel_options.large_bf16_tile_policy
            == crate::tile::LargeBf16TilePolicy::Square128Streamed
        {
            let baseline = execution_plan.clone();
            execution_plan.configure_square128_streamed(
                self.matmul_shared_layout,
                self.matmul_pipeline,
                self.target.supports_async_copy(),
                self.target.matmul_resources,
            );
            (execution_plan.large_square128_streamed_count() > 0).then_some(baseline)
        } else {
            None
        };
        let square_attempts = execution_plan.large_square128_streamed_count();
        let mut index_optimization_time = Duration::ZERO;
        let mut module_load_time = Duration::ZERO;
        let mut compile = |plan: &mut PtxExecutionPlan| -> Result<_> {
            let (tiles, index_time) = self.lower_planned_tiles(graph, storage, plan)?;
            index_optimization_time += index_time;
            let began = Instant::now();
            let result = self.compile_tile_variant_with_plan(tiles, plan);
            module_load_time += began.elapsed();
            result
        };
        let candidate = compile(&mut execution_plan);
        let (ptx_graph, cuda_module, square_fallbacks) = if let Some(mut baseline) = square_baseline
        {
            match candidate {
                Ok((graph, module))
                    if self
                        .square128_resources_fit(&execution_plan, &graph, &module)
                        .unwrap_or(false) =>
                {
                    (graph, module, 0)
                }
                _ => {
                    let (graph, module) = compile(&mut baseline)
                        .context("Square128 complete baseline fallback failed")?;
                    execution_plan = baseline;
                    (graph, module, square_attempts)
                }
            }
        } else {
            let (graph, module) = candidate?;
            (graph, module, 0)
        };
        let generated_kernels = execution_plan
            .steps()
            .iter()
            .filter(|step| {
                matches!(
                    step.action,
                    PtxPlanAction::Kernel
                        | PtxPlanAction::PointwiseRegion(_)
                        | PtxPlanAction::ReductionRegion(_)
                        | PtxPlanAction::MatMulRegion(_)
                        | PtxPlanAction::OnlineRegion(_)
                )
            })
            .count();
        let ptx_source_bytes = ptx_graph.module_source().len();
        execution_plan.retain_column_reductions(|id, region| {
            let name = ptx_graph
                .reduction_region_kernel_name(id)
                .context("Column reduction function absent")?;
            let function = cuda_module
                .load_function(name)
                .context("Column reduction function load failed")?;
            super::target::column_reduction_fits(region, &function)
        })?;
        let unique_kernels = ptx_graph.unique_kernel_count();
        self.square128_streamed_fallbacks = square_fallbacks;
        self.module = Some(cuda_module);
        self.functions.get_mut().clear();
        self.ptx_graph = Some(Box::new(ptx_graph));
        self.execution_plan = Some(Arc::new(execution_plan));
        self.compilation_signature = Some(signature);
        self.index_plan = index_plan;
        self.compilation_count += 1;
        self.compile_metrics = PtxCompileMetrics {
            graph_nodes,
            generated_kernels,
            unique_kernels,
            module_load_time,
            ptx_source_bytes,
            compile_time: started.elapsed(),
            index_optimization_time,
            fusion_selection_time,
        };

        Ok(())
    }

    /// Get the value of a specific node from the executor's cache after execution
    pub fn get_value(&self, node_idx: petgraph::graph::NodeIndex) -> Option<Vec<D::HostType>> {
        let gate = self.stream_gate.clone();
        let _guard = lock_stream(&gate);
        self.values.get(&node_idx).and_then(|cuda_slice| {
            let readback_started = Instant::now();
            let mut storage = vec![D::HostType::default(); cuda_slice.len()];
            self.stream.clone()
                .memcpy_dtoh(cuda_slice.as_ref(), &mut storage)
                .ok()?;
            let plan = self.execution_plan.as_ref()?;
            let view = plan.virtual_value(node_idx)?;
            let source_shape = plan.value_shape(view.source)?;
            let copy_seconds = readback_started.elapsed().as_secs_f64();
            let materialize_started = Instant::now();
            let output = materialize_host_view(storage, view, source_shape).ok();
            if std::env::var("TNSR_PROFILE_READBACK").as_deref() == Ok("1") {
                eprintln!("{}", serde_json::json!({"node":node_idx.index(),"source_shape":source_shape,
                    "output_shape":view.shape,"storage_elements":cuda_slice.len(),
                    "copy_seconds":copy_seconds,"view_seconds":materialize_started.elapsed().as_secs_f64()}));
            }
            output
        })
    }

    fn execute_matmul(
        &self,
        node: crate::graph::NodeIndex,
        kernel_name: &str,
        a: &CudaSlice<D::HostType>,
        b: &CudaSlice<D::HostType>,
        shapes: (&[usize], &[usize], &[usize]),
    ) -> Result<CudaSlice<D::HostType>> {
        let (a_shape, b_shape, output_shape) = shapes;
        anyhow::ensure!(
            a_shape.len() >= 2 && b_shape.len() >= 2 && output_shape.len() >= 2,
            "MatMul requires rank >= 2, got {a_shape:?} and {b_shape:?}"
        );
        let m = a_shape[a_shape.len() - 2];
        let k = a_shape[a_shape.len() - 1];
        let b_k = b_shape[b_shape.len() - 2];
        let n = b_shape[b_shape.len() - 1];
        anyhow::ensure!(k == b_k, "MatMul K mismatch: {k} != {b_k}");
        anyhow::ensure!(
            output_shape[output_shape.len() - 2..] == [m, n],
            "MatMul output shape {output_shape:?} does not end in [{m}, {n}]"
        );

        let output_batch_shape = &output_shape[..output_shape.len() - 2];
        validate_batch_broadcast(a_shape, output_batch_shape)?;
        validate_batch_broadcast(b_shape, output_batch_shape)?;
        let output_batches: usize = output_batch_shape.iter().product();
        let output_len: usize = output_shape.iter().product();
        let mut output = self.take_buffer(output_len)?;
        if output_batches == 0 || m == 0 || n == 0 {
            return Ok(output);
        }

        let schedule = self
            .execution_plan
            .as_ref()
            .and_then(|plan| plan.matmul_schedules().get(&node))
            .context("Matmul schedule is unavailable")?;
        let stream = self.stream.clone();
        if k == 0 {
            stream
                .memset_zeros(&mut output)
                .context("Failed to zero MatMul output with K=0")?;
            return Ok(output);
        }
        let module = self.module.as_ref().context("PTX module is not compiled")?;
        let function = self.kernel_function(module, kernel_name)?;
        let config = LaunchConfig {
            grid_dim: (
                u32::try_from(n.div_ceil(schedule.block_tile.n))
                    .context("Matmul grid width exceeds u32")?,
                u32::try_from(m.div_ceil(schedule.block_tile.m))
                    .context("Matmul grid height exceeds u32")?,
                1,
            ),
            block_dim: schedule.block_threads,
            shared_mem_bytes: 0,
        };

        for batch in 0..output_batches {
            let a_batch = batch_matrix_offset(batch, output_batch_shape, a_shape);
            let b_batch = batch_matrix_offset(batch, output_batch_shape, b_shape);
            let a_start = a_batch * m * k;
            let b_start = b_batch * k * n;
            let output_start = batch * m * n;
            let a_view = a.slice(a_start..a_start + m * k);
            let b_view = b.slice(b_start..b_start + k * n);
            let mut output_view = output.slice_mut(output_start..output_start + m * n);
            let mut launcher = stream.launch_builder(&function);
            launcher.arg(&a_view);
            launcher.arg(&b_view);
            launcher.arg(&mut output_view);
            self.record_kernel_launch();
            unsafe { launcher.launch(config) }
                .with_context(|| format!("CUDA {kernel_name} kernel launch failed"))?;
        }
        Ok(output)
    }

    fn execute_matmul_region(
        &self,
        module: &Arc<CudaModule>,
        ptx_graph: &PtxGraph,
        region_id: usize,
        step: &super::plan::PtxPlanStep,
    ) -> Result<RegionValues<D::HostType>> {
        let kernel_name = ptx_graph
            .matmul_region_kernel_name(region_id)
            .context("Missing compiled matmul region kernel")?;
        let region = self
            .execution_plan
            .as_ref()
            .and_then(|plan| plan.matmul_regions().get(region_id))
            .context("Matmul region metadata is unavailable")?;
        anyhow::ensure!(
            step.outputs
                == region
                    .outputs
                    .iter()
                    .map(|output| output.node)
                    .collect::<Vec<_>>(),
            "Matmul region output bindings do not match the execution plan"
        );
        let input_values: Vec<_> = step
            .inputs
            .iter()
            .map(|input| {
                self.values.get(input).cloned().with_context(|| {
                    format!("Matmul region input node {} is unavailable", input.index())
                })
            })
            .collect::<Result<_>>()?;
        anyhow::ensure!(
            input_values.len() == region.inputs.len(),
            "Matmul region input binding count mismatch"
        );
        for (input, descriptor) in input_values.iter().zip(&region.inputs) {
            let expected =
                checked_element_count(&descriptor.source_shape, "Matmul region input storage")?;
            anyhow::ensure!(
                input.len() == expected,
                "Matmul region input length {} does not match storage length {expected}",
                input.len()
            );
        }
        let output_len = checked_element_count(&region.output_shape, "Matmul region output")?;
        let mut outputs = Vec::with_capacity(step.outputs.len());
        for _ in &step.outputs {
            outputs.push(self.take_buffer(output_len)?);
        }

        let function = self.kernel_function(module, kernel_name)?;
        let stream = self.stream.clone();
        let mut launcher = stream.launch_builder(&function);
        for input in &input_values {
            launcher.arg(input.as_ref());
        }
        for output in &mut outputs {
            launcher.arg(output);
        }
        self.record_kernel_launch();
        let config = LaunchConfig {
            grid_dim: (
                u32::try_from(region.n.div_ceil(region.schedule.block_tile.n))
                    .context("Matmul region grid width exceeds u32")?,
                u32::try_from(region.m.div_ceil(region.schedule.block_tile.m))
                    .context("Matmul region grid height exceeds u32")?,
                u32::try_from(checked_element_count(
                    &region.batch_shape,
                    "Matmul batch count",
                )?)
                .context("Matmul batch grid exceeds u32")?,
            ),
            block_dim: region.schedule.block_threads,
            shared_mem_bytes: 0,
        };
        unsafe { launcher.launch(config) }
            .with_context(|| format!("CUDA {kernel_name} kernel launch failed"))?;

        Ok(step
            .outputs
            .iter()
            .copied()
            .zip(outputs.into_iter().map(Arc::new))
            .collect())
    }

    fn execute_scalar_region(
        &self,
        module: &Arc<CudaModule>,
        ptx_graph: &PtxGraph,
        step: &super::plan::PtxPlanStep,
    ) -> Result<RegionValues<D::HostType>> {
        let plan = self
            .execution_plan
            .as_ref()
            .context("Scalar region plan unavailable")?;
        let (kernel_id, inputs, threads) = match step.action {
            PtxPlanAction::PointwiseRegion(id) => (id, &plan.regions()[id].inputs, None),
            PtxPlanAction::OnlineRegion(id) => {
                let region = &plan.online_regions()[id];
                (
                    plan.regions().len() + id,
                    &region.inputs,
                    region.block_threads(),
                )
            }
            _ => anyhow::bail!("Expected a scalar region step"),
        };
        let kernel_name = ptx_graph
            .region_kernel_name(kernel_id)
            .context("Scalar region kernel unavailable")?;
        let input_values: Vec<_> = step
            .inputs
            .iter()
            .map(|input| {
                self.values.get(input).cloned().with_context(|| {
                    format!("Scalar region input node {} is unavailable", input.index())
                })
            })
            .collect::<Result<_>>()?;
        let len = checked_element_count(&step.shape, "Scalar region")?;
        anyhow::ensure!(
            inputs.len() == input_values.len(),
            "Scalar region input binding count mismatch"
        );
        for (input, descriptor) in input_values.iter().zip(inputs) {
            let expected =
                checked_element_count(&descriptor.source_shape, "Scalar region input storage")?;
            anyhow::ensure!(
                input.len() == expected,
                "Scalar region input length {} does not match storage length {expected}",
                input.len(),
            );
        }
        let mut outputs = Vec::with_capacity(step.outputs.len());
        for _ in &step.outputs {
            outputs.push(self.take_buffer(len)?);
        }
        if len != 0 {
            let config = if let Some(threads) = threads {
                LaunchConfig {
                    grid_dim: (
                        u32::try_from(len / step.shape[step.shape.len() - 1])
                            .context("Scalar grid too large")?,
                        1,
                        1,
                    ),
                    block_dim: (threads, 1, 1),
                    shared_mem_bytes: 0,
                }
            } else {
                crate::ptx::target::pointwise_launch_config(
                    u32::try_from(len).context("Scalar output too large")?,
                )
            };
            let function = self.kernel_function(module, kernel_name)?;
            let stream = self.stream.clone();
            let mut launcher = stream.launch_builder(&function);
            for input in &input_values {
                launcher.arg(input.as_ref());
            }
            for output in &mut outputs {
                launcher.arg(output);
            }
            self.record_kernel_launch();
            // SAFETY: Storage lengths and output extent are checked; generated
            // kernels guard output lanes and traverse valid input coordinates.
            unsafe { launcher.launch(config) }
                .with_context(|| format!("CUDA {kernel_name} kernel launch failed"))?;
        }
        Ok(step
            .outputs
            .iter()
            .copied()
            .zip(outputs.into_iter().map(Arc::new))
            .collect())
    }

    fn execute_reduction_region(
        &self,
        module: &Arc<CudaModule>,
        ptx_graph: &PtxGraph,
        region_id: usize,
        step: &super::plan::PtxPlanStep,
    ) -> Result<RegionValues<D::HostType>> {
        let kernel_name = ptx_graph
            .reduction_region_kernel_name(region_id)
            .context("Missing compiled reduction region kernel")?;
        let region = self
            .execution_plan
            .as_ref()
            .and_then(|plan| plan.reduction_regions().get(region_id))
            .context("Reduction region metadata is unavailable")?;
        anyhow::ensure!(
            step.outputs
                == region
                    .outputs
                    .iter()
                    .map(|output| output.node)
                    .collect::<Vec<_>>(),
            "Reduction region output bindings do not match the execution plan"
        );
        anyhow::ensure!(
            step.output_shapes == region.output_shapes,
            "Reduction region output shapes do not match the execution plan"
        );
        anyhow::ensure!(
            step.shape == region.output_shape,
            "Reduction region launch shape does not match the execution plan"
        );
        let input_values: Vec<_> = step
            .inputs
            .iter()
            .map(|input| {
                self.values.get(input).cloned().with_context(|| {
                    format!(
                        "Reduction region input node {} is unavailable",
                        input.index()
                    )
                })
            })
            .collect::<Result<_>>()?;
        anyhow::ensure!(
            region.inputs.len() == input_values.len(),
            "Reduction region input binding count mismatch"
        );
        for (input, descriptor) in input_values.iter().zip(&region.inputs) {
            let expected =
                checked_element_count(&descriptor.source_shape, "Reduction region input storage")?;
            anyhow::ensure!(
                input.len() == expected,
                "Reduction region input length {} does not match storage length {expected}",
                input.len(),
            );
        }
        let len = checked_element_count(&step.shape, "Reduction region")?;
        anyhow::ensure!(
            step.output_shapes.len() == step.outputs.len(),
            "Reduction region output shape binding count mismatch"
        );
        let mut outputs = Vec::with_capacity(step.outputs.len());
        for shape in &step.output_shapes {
            outputs
                .push(self.take_buffer(checked_element_count(shape, "Reduction region output")?)?);
        }
        if len != 0 {
            let launch_len = u32::try_from(len)
                .context("Reduction region output is too large for a CUDA launch")?;
            let function = self.kernel_function(module, kernel_name)?;
            let stream = self.stream.clone();
            let mut launcher = stream.launch_builder(&function);
            for input in &input_values {
                launcher.arg(input.as_ref());
            }
            for output in &mut outputs {
                launcher.arg(output);
            }
            self.record_kernel_launch();
            let config =
                super::target::region_reduction_launch_config(region, launch_len, &function)?;
            unsafe { launcher.launch(config) }
                .with_context(|| format!("CUDA {kernel_name} kernel launch failed"))?;
        }
        Ok(step
            .outputs
            .iter()
            .copied()
            .zip(outputs.into_iter().map(Arc::new))
            .collect())
    }

    /// Execute a compiled graph with the given inputs
    pub fn execute_compiled<G>(
        &mut self,
        graph: &TensorGraph<D::HostType, G>,
        inputs: HashMap<String, Vec<D::HostType>>,
    ) -> Result<Vec<D::HostType>> {
        let gate = self.stream_gate.clone();
        let _guard = lock_stream(&gate);
        self.execute_compiled_with_indices_locked(graph, inputs, HashMap::new())
    }

    /// Run a previously compiled graph with exact integer feeds.
    pub fn execute_compiled_with_indices<G>(
        &mut self,
        graph: &TensorGraph<D::HostType, G>,
        inputs: HashMap<String, Vec<D::HostType>>,
        indices: crate::tensor::IndexInputs,
    ) -> Result<Vec<D::HostType>> {
        let gate = self.stream_gate.clone();
        let _guard = lock_stream(&gate);
        self.execute_compiled_with_indices_locked(graph, inputs, indices)
    }

    fn execute_compiled_with_indices_locked<G>(
        &mut self,
        graph: &TensorGraph<D::HostType, G>,
        inputs: HashMap<String, Vec<D::HostType>>,
        indices: crate::tensor::IndexInputs,
    ) -> Result<Vec<D::HostType>> {
        self.invalidate_replay();
        let sources = self.index_plan.prepare(graph, indices)?;
        let copies = self.exact_indices.stage(&self.stream, sources)?;
        let result = self.execute_compiled_impl(graph, inputs, true);
        self.execution_metrics.device_copies += copies;
        result
    }

    fn execute_compiled_impl<G>(
        &mut self,
        graph: &TensorGraph<D::HostType, G>,
        inputs: HashMap<String, Vec<D::HostType>>,
        read_output: bool,
    ) -> Result<Vec<D::HostType>> {
        self.execution_metrics = PtxExecutionMetrics::default();
        self.kernel_launches.set(0);
        let supplied_signature = graph_compilation_signature(
            graph,
            self.reduction_mode,
            self.target,
            self.matmul_precision,
            self.matmul_shared_layout,
            self.matmul_pipeline,
            (
                self.fusion_policy,
                self.serial_reduction_block_threads,
                self.matmul_grid_group_m,
                self.experimental_register_tiles,
                self.experimental_kernel_options,
            ),
        );
        let compiled_signature = self
            .compilation_signature
            .as_deref()
            .context("No compiled graph signature available. Call compile_owned() first.")?;
        anyhow::ensure!(
            compiled_signature == supplied_signature,
            "Supplied graph structure does not match the compiled PTX module; compile the graph before execution"
        );
        let module = self
            .module
            .as_ref()
            .context("No compiled module available. Call compile() first.")?;

        let ptx_graph = self
            .ptx_graph
            .as_ref()
            .context("No PTX graph available. Call compile() first.")?;

        let execution_plan = self
            .execution_plan
            .as_ref()
            .context("No PTX execution plan available. Call compile_owned() first.")?
            .clone();
        let plan_steps = execution_plan.steps();
        let graph_output = execution_plan.graph_output().context("Graph is empty")?;
        if !self.replay_frame_active {
            self.sources.prune();
        }
        let old_values: Vec<_> = self.values.drain().map(|(_, value)| value).collect();
        for value in old_values {
            self.recycle_value(value);
        }
        let mut live_bytes = self.exact_indices.bytes();

        for plan_step in plan_steps {
            let node_idx = &plan_step.node;
            let planned_inputs = &plan_step.inputs;
            let node = &graph[*node_idx];
            if let PtxPlanAction::MatMulRegion(region_id) = plan_step.action {
                let region_outputs =
                    self.execute_matmul_region(module, ptx_graph, region_id, plan_step)?;
                for (output_node, output) in region_outputs {
                    let bytes = output.len() * std::mem::size_of::<D::HostType>();
                    self.execution_metrics.materialized_values += 1;
                    self.execution_metrics.materialized_bytes += bytes;
                    if output_node != graph_output {
                        self.execution_metrics.intermediate_materialized_bytes += bytes;
                    }
                    self.values.insert(output_node, output);
                    live_bytes += bytes;
                }
                self.stats.record_live(live_bytes);
                for &dead in &plan_step.release_after {
                    if let Some(value) = self.values.remove(&dead) {
                        live_bytes -= self.recycle_value(value);
                    }
                }
                continue;
            }
            if matches!(
                plan_step.action,
                PtxPlanAction::PointwiseRegion(_) | PtxPlanAction::OnlineRegion(_)
            ) {
                let region_outputs = self.execute_scalar_region(module, ptx_graph, plan_step)?;
                for (output_node, output) in region_outputs {
                    let bytes = output.len() * std::mem::size_of::<D::HostType>();
                    self.execution_metrics.materialized_values += 1;
                    self.execution_metrics.materialized_bytes += bytes;
                    if output_node != graph_output {
                        self.execution_metrics.intermediate_materialized_bytes += bytes;
                    }
                    self.values.insert(output_node, output);
                    live_bytes += bytes;
                }
                self.stats.record_live(live_bytes);
                for &dead in &plan_step.release_after {
                    if let Some(value) = self.values.remove(&dead) {
                        live_bytes -= self.recycle_value(value);
                    }
                }
                continue;
            }
            if let PtxPlanAction::ReductionRegion(region_id) = plan_step.action {
                let region_outputs =
                    self.execute_reduction_region(module, ptx_graph, region_id, plan_step)?;
                for (output_node, output) in region_outputs {
                    let bytes = output.len() * std::mem::size_of::<D::HostType>();
                    self.execution_metrics.materialized_values += 1;
                    self.execution_metrics.materialized_bytes += bytes;
                    if output_node != graph_output {
                        self.execution_metrics.intermediate_materialized_bytes += bytes;
                    }
                    self.values.insert(output_node, output);
                    live_bytes += bytes;
                }
                self.stats.record_live(live_bytes);
                for &dead in &plan_step.release_after {
                    if let Some(value) = self.values.remove(&dead) {
                        live_bytes -= self.recycle_value(value);
                    }
                }
                continue;
            }
            if plan_step.action == PtxPlanAction::VirtualView {
                let input = planned_inputs
                    .first()
                    .context("virtual view plan step has no input")?;
                let result = Arc::clone(
                    self.values
                        .get(input)
                        .context("virtual view input value is unavailable")?,
                );
                self.values.insert(*node_idx, result);
                self.stats.record_live(live_bytes);
                for &dead in &plan_step.release_after {
                    if let Some(value) = self.values.remove(&dead) {
                        live_bytes -= self.recycle_value(value);
                    }
                }
                continue;
            }
            let binding = if let Some(value) = self.bindings.get(node_idx) {
                Some(super::sources::SourceBinding {
                    value: value.clone(),
                    uploaded: false,
                })
            } else {
                self.sources
                    .get(node, &self.stream, |len| self.take_buffer(len))?
            };
            if let Some(binding) = binding {
                self.execution_metrics.device_copies += usize::from(binding.uploaded);
                let bytes = binding.value.len() * std::mem::size_of::<D::HostType>();
                self.execution_metrics.materialized_values += 1;
                self.execution_metrics.materialized_bytes += bytes;
                self.values.insert(*node_idx, binding.value);
                live_bytes += bytes;
                self.stats.record_live(live_bytes);
                for &dead in &plan_step.release_after {
                    if let Some(value) = self.values.remove(&dead) {
                        live_bytes -= self.recycle_value(value);
                    }
                }
                continue;
            }
            let result = match node {
                TensorGraphNode::Linear { .. } | TensorGraphNode::Cast { .. } => {
                    anyhow::bail!("mixed casts require MixedExecutor")
                }
                _ if plan_step.materialized_view.is_some() => {
                    let kernel_name = ptx_graph
                        .kernel_name(*node_idx)
                        .context("Missing materialized view kernel")?;
                    let input = self
                        .values
                        .get(&planned_inputs[0])
                        .context("Missing source storage for materialized view")?;
                    let len = checked_element_count(node.shape(), "Materialized view")?;
                    let mut output = self.take_buffer(len)?;
                    if len != 0 {
                        let launch_len = u32::try_from(len)
                            .context("Materialized view is too large for a CUDA launch")?;
                        let function = self.kernel_function(module, kernel_name)?;
                        let stream = self.stream.clone();
                        let mut launcher = stream.launch_builder(&function);
                        launcher.arg(input.as_ref());
                        launcher.arg(&mut output);
                        self.record_kernel_launch();
                        // SAFETY: the composed map addresses validated source storage;
                        // the generated kernel bounds-checks the logical output domain.
                        unsafe { launcher.launch(LaunchConfig::for_num_elems(launch_len)) }
                            .context("CUDA materialized view kernel launch failed")?;
                    }
                    output
                }

                TensorGraphNode::IndexSource { .. } => {
                    let value = self
                        .exact_indices
                        .buffers
                        .get(node_idx)
                        .context("integer source was not staged")?;
                    let bytes = value.values.len() * std::mem::size_of::<u32>();
                    self.execution_metrics.materialized_values += 1;
                    self.execution_metrics.materialized_bytes += bytes;
                    continue;
                }
                TensorGraphNode::Constant { data, .. } => {
                    let stream = self.stream.clone();
                    let mut device_data = self.take_buffer(data.len())?;
                    if !data.is_empty() {
                        self.execution_metrics.device_copies += 1;
                        stream
                            .memcpy_htod(data.as_slice(), &mut device_data)
                            .context("Failed to copy constant to CUDA device")?;
                    }
                    device_data
                }
                TensorGraphNode::Input { name, shape } => {
                    let val = inputs
                        .get::<str>(name)
                        .with_context(|| format!("Input '{}' not found", name))?;
                    let expected_len = checked_element_count(shape, &format!("Input '{name}'"))?;
                    anyhow::ensure!(
                        val.len() == expected_len,
                        "Input '{name}' has {} values but declared shape {shape:?} requires {expected_len}",
                        val.len()
                    );
                    let stream = self.stream.clone();
                    let mut device_data = self.take_buffer(val.len())?;
                    if !val.is_empty() {
                        self.execution_metrics.device_copies += 1;
                        stream
                            .memcpy_htod(val.as_slice(), &mut device_data)
                            .context("Failed to copy input to CUDA device")?;
                    }
                    device_data
                }
                TensorGraphNode::Parameter { data, .. } => {
                    use anyhow::anyhow;
                    let v = data
                        .lock()
                        .map_err(|e| anyhow!("Failed to lock parameter data: {}", e))?;
                    let stream = self.stream.clone();
                    let mut device_data = self.take_buffer(v.len())?;
                    if !v.is_empty() {
                        self.execution_metrics.device_copies += 1;
                        stream
                            .memcpy_htod(v.as_slice(), &mut device_data)
                            .context("Failed to copy parameter to CUDA device")?;
                    }
                    device_data
                }
                TensorGraphNode::Unary { .. } => {
                    let kernel_name = ptx_graph
                        .kernel_name(*node_idx)
                        .context("Missing compiled unary kernel")?;

                    let ins = planned_inputs;
                    let input = self
                        .values
                        .get(&ins[0])
                        .context("Missing input value for unary operation")?;

                    let len = input.len();
                    let stream = self.stream.clone();
                    let mut out = self.take_buffer(len)?;
                    if len != 0 {
                        let f = self.kernel_function(module, kernel_name)?;
                        let cfg = LaunchConfig::for_num_elems(len as u32);
                        let mut launcher = stream.launch_builder(&f);
                        launcher.arg(input.as_ref());
                        launcher.arg(&mut out);
                        self.record_kernel_launch();
                        unsafe { launcher.launch(cfg) }.with_context(|| {
                            format!("CUDA {} kernel launch failed", kernel_name)
                        })?;
                    }

                    out
                }
                #[cfg(feature = "fusion")]
                TensorGraphNode::FusedUnary { .. } => {
                    let kernel_name = ptx_graph
                        .kernel_name(*node_idx)
                        .context("Missing compiled fused unary kernel")?;

                    let ins = planned_inputs;
                    let input = self
                        .values
                        .get(&ins[0])
                        .context("Missing input value for fused unary operation")?;

                    let len = input.len();
                    let stream = self.stream.clone();
                    let mut out = self.take_buffer(len)?;
                    if len != 0 {
                        let f = self.kernel_function(module, kernel_name)?;
                        let cfg = LaunchConfig::for_num_elems(len as u32);
                        let mut launcher = stream.launch_builder(&f);
                        launcher.arg(input.as_ref());
                        launcher.arg(&mut out);
                        self.record_kernel_launch();
                        unsafe { launcher.launch(cfg) }.with_context(|| {
                            format!("CUDA {} kernel launch failed", kernel_name)
                        })?;
                    }

                    out
                }
                TensorGraphNode::Binary { .. } => {
                    let kernel_name = ptx_graph
                        .kernel_name(*node_idx)
                        .context("Missing compiled binary kernel")?;

                    let ins = planned_inputs;
                    let lhs = self
                        .values
                        .get(&ins[0])
                        .context("Missing left operand for binary operation")?;
                    let rhs = self
                        .values
                        .get(&ins[1])
                        .context("Missing right operand for binary operation")?;

                    let len = lhs.len();
                    anyhow::ensure!(rhs.len() == len, "Binary operand length mismatch");
                    let stream = self.stream.clone();
                    let mut out = self.take_buffer(len)?;
                    if len != 0 {
                        let f = self.kernel_function(module, kernel_name)?;
                        let cfg = LaunchConfig::for_num_elems(len as u32);
                        let mut launcher = stream.launch_builder(&f);
                        launcher.arg(lhs.as_ref());
                        launcher.arg(rhs.as_ref());
                        launcher.arg(&mut out);
                        self.record_kernel_launch();
                        unsafe { launcher.launch(cfg) }.with_context(|| {
                            format!("CUDA {} kernel launch failed", kernel_name)
                        })?;
                    }

                    out
                }
                TensorGraphNode::Gt { .. } => {
                    let kernel_name = ptx_graph
                        .kernel_name(*node_idx)
                        .context("Missing compiled Gt kernel")?;

                    let ins = planned_inputs;
                    let lhs = self
                        .values
                        .get(&ins[0])
                        .context("Missing left operand for Gt")?;
                    let rhs = self
                        .values
                        .get(&ins[1])
                        .context("Missing right operand for Gt")?;

                    let len = lhs.len();
                    anyhow::ensure!(rhs.len() == len, "Gt operand length mismatch");
                    let stream = self.stream.clone();
                    let mut out = self.take_buffer(len)?;
                    if len != 0 {
                        let f = self.kernel_function(module, kernel_name)?;
                        let cfg = LaunchConfig::for_num_elems(len as u32);
                        let mut launcher = stream.launch_builder(&f);
                        launcher.arg(lhs.as_ref());
                        launcher.arg(rhs.as_ref());
                        launcher.arg(&mut out);
                        self.record_kernel_launch();
                        unsafe { launcher.launch(cfg) }.with_context(|| {
                            format!("CUDA {} kernel launch failed", kernel_name)
                        })?;
                    }

                    out
                }
                TensorGraphNode::Narrow { shape, .. } | TensorGraphNode::Concat { shape, .. } => {
                    let len = shape.iter().product();
                    let mut output = self.take_buffer(len)?;
                    if len > 0 {
                        let kernel_name = ptx_graph
                            .kernel_name(*node_idx)
                            .context("Missing segment kernel")?;
                        let function = self.kernel_function(module, kernel_name)?;
                        let stream = self.stream.clone();
                        let mut launcher = stream.launch_builder(&function);
                        for input in planned_inputs {
                            launcher.arg(
                                self.values
                                    .get(input)
                                    .context("Segment input missing")?
                                    .as_ref(),
                            );
                        }
                        launcher.arg(&mut output);
                        let extent = u32::try_from(len)
                            .context("Segment output exceeds CUDA launch range")?;
                        self.record_kernel_launch();
                        // SAFETY: generated kernels bound each output and guard input
                        // loads by segment membership; buffers match the graph shapes.
                        unsafe { launcher.launch(LaunchConfig::for_num_elems(extent)) }
                            .context("Segment kernel launch failed")?;
                    }
                    output
                }
                TensorGraphNode::Mask { .. } => {
                    let kernel_name = ptx_graph
                        .kernel_name(*node_idx)
                        .context("Missing compiled Mask kernel")?;

                    let ins = planned_inputs;
                    let values = self
                        .values
                        .get(&ins[0])
                        .context("Missing values for Mask")?;
                    let condition = self
                        .values
                        .get(&ins[1])
                        .context("Missing condition for Mask")?;

                    let len = values.len();
                    anyhow::ensure!(condition.len() == len, "Mask operand length mismatch");
                    let stream = self.stream.clone();
                    let mut out = self.take_buffer(len)?;
                    if len != 0 {
                        let f = self.kernel_function(module, kernel_name)?;
                        let cfg = LaunchConfig::for_num_elems(len as u32);
                        let mut launcher = stream.launch_builder(&f);
                        launcher.arg(values.as_ref());
                        launcher.arg(condition.as_ref());
                        launcher.arg(&mut out);
                        self.record_kernel_launch();
                        unsafe { launcher.launch(cfg) }
                            .context("CUDA mask kernel launch failed")?;
                    }

                    out
                }
                TensorGraphNode::MatMul { .. } => {
                    let kernel_name = ptx_graph
                        .kernel_name(*node_idx)
                        .context("Missing compiled MatMul kernel")?;

                    let ins = planned_inputs;
                    let a = self
                        .values
                        .get(&ins[0])
                        .context("Missing A matrix for MatMul")?;
                    let b = self
                        .values
                        .get(&ins[1])
                        .context("Missing B matrix for MatMul")?;

                    let a_shape = graph.graph[ins[0]].shape();
                    let b_shape = graph.graph[ins[1]].shape();
                    self.execute_matmul(
                        *node_idx,
                        kernel_name,
                        a,
                        b,
                        (a_shape, b_shape, graph.graph[*node_idx].shape()),
                    )?
                }
                TensorGraphNode::Embedding { .. } => {
                    let kernel_name = ptx_graph
                        .kernel_name(*node_idx)
                        .context("Missing compiled Embedding kernel")?;
                    let ins = planned_inputs;
                    anyhow::ensure!(ins.len() == 2, "Embedding requires 2 inputs");
                    let weight = self
                        .values
                        .get(&ins[0])
                        .context("Missing weight for Embedding")?;
                    let indices = self.index_argument(ins[1])?;
                    let weight_shape = graph.graph[ins[0]].shape();
                    let indices_shape = graph.graph[ins[1]].shape();
                    anyhow::ensure!(
                        weight_shape.len() == 2,
                        "Embedding weight must be [V, C], got {weight_shape:?}"
                    );
                    let index_count: usize = indices_shape.iter().product();
                    let output_len: usize = node.shape().iter().product();
                    anyhow::ensure!(
                        indices.len() == index_count,
                        "Embedding indices length mismatch"
                    );
                    anyhow::ensure!(
                        weight.len() == weight_shape.iter().product::<usize>(),
                        "Embedding weight length mismatch"
                    );
                    self.validate_index_argument(&indices, weight_shape[0], "Embedding")?;
                    let mut output = self.take_buffer(output_len)?;
                    if output_len != 0 {
                        let launch_len = u32::try_from(output_len)
                            .context("Embedding output is too large for a CUDA launch")?;
                        let function = self.kernel_function(module, kernel_name)?;
                        let stream = self.stream.clone();
                        let mut launcher = stream.launch_builder(&function);
                        launcher.arg(weight.as_ref());
                        match &indices {
                            IndexArgument::Exact { values, .. } => {
                                launcher.arg(*values);
                            }
                            IndexArgument::Floating(values) => {
                                launcher.arg(values.as_ref());
                            }
                        }
                        launcher.arg(&mut output);
                        self.record_kernel_launch();
                        unsafe { launcher.launch(LaunchConfig::for_num_elems(launch_len)) }
                            .context("PTX embedding kernel launch failed")?;
                    }
                    output
                }
                TensorGraphNode::EmbeddingBackward { .. } => {
                    let kernel_name = ptx_graph
                        .kernel_name(*node_idx)
                        .context("Missing compiled EmbeddingBackward kernel")?;
                    let ins = planned_inputs;
                    anyhow::ensure!(ins.len() == 2, "EmbeddingBackward requires 2 inputs");
                    let indices = self.index_argument(ins[0])?;
                    let grad_output = self
                        .values
                        .get(&ins[1])
                        .context("Missing grad_output for EmbeddingBackward")?;
                    let output_shape = node.shape();
                    anyhow::ensure!(
                        output_shape.len() == 2,
                        "EmbeddingBackward output must be [V, C], got {output_shape:?}"
                    );
                    let index_count: usize = graph.graph[ins[0]].shape().iter().product();
                    let grad_output_len = index_count * output_shape[1];
                    anyhow::ensure!(
                        indices.len() == index_count,
                        "EmbeddingBackward indices length mismatch"
                    );
                    anyhow::ensure!(
                        grad_output.len() == grad_output_len,
                        "EmbeddingBackward grad_output length mismatch"
                    );
                    self.validate_index_argument(&indices, output_shape[0], "EmbeddingBackward")?;
                    let output_len: usize = output_shape.iter().product();
                    let stream = self.stream.clone();
                    let mut output = self.take_buffer(output_len)?;
                    if output_len != 0 {
                        stream
                            .memset_zeros(&mut output)
                            .context("Failed to zero PTX embedding gradient output")?;
                    }
                    if grad_output_len != 0 {
                        let launch_elements = if <D::HostType as crate::tile::TileDType>::TILE_DTYPE
                            == crate::tile::DType::F32
                        {
                            grad_output_len
                        } else {
                            output_len
                        };
                        let launch_len = u32::try_from(launch_elements).context(
                            "EmbeddingBackward grad_output is too large for a CUDA launch",
                        )?;
                        let function = self.kernel_function(module, kernel_name)?;
                        let mut launcher = stream.launch_builder(&function);
                        match &indices {
                            IndexArgument::Exact { values, .. } => {
                                launcher.arg(*values);
                            }
                            IndexArgument::Floating(values) => {
                                launcher.arg(values.as_ref());
                            }
                        }
                        launcher.arg(grad_output.as_ref());
                        launcher.arg(&mut output);
                        self.record_kernel_launch();
                        unsafe { launcher.launch(LaunchConfig::for_num_elems(launch_len)) }
                            .context("PTX embedding_backward kernel launch failed")?;
                    }
                    output
                }
                TensorGraphNode::IndexedCrossEntropy { .. } => {
                    let kernel_name = ptx_graph
                        .kernel_name(*node_idx)
                        .context("Missing compiled IndexedCrossEntropy kernel")?;
                    let ins = planned_inputs;
                    anyhow::ensure!(ins.len() == 2, "IndexedCrossEntropy requires 2 inputs");
                    let logits = self
                        .values
                        .get(&ins[0])
                        .context("Missing logits for IndexedCrossEntropy")?;
                    let targets = self.index_argument(ins[1])?;
                    let logits_shape = graph.graph[ins[0]].shape();
                    let vocabulary = *logits_shape
                        .last()
                        .context("IndexedCrossEntropy logits must have rank >= 1")?;
                    anyhow::ensure!(
                        vocabulary > 0,
                        "IndexedCrossEntropy vocabulary must be nonzero"
                    );
                    let row_count: usize = node.shape().iter().product();
                    anyhow::ensure!(
                        targets.len() == row_count,
                        "Cross entropy targets length mismatch"
                    );
                    anyhow::ensure!(
                        logits.len() == row_count * vocabulary,
                        "Cross entropy logits length mismatch"
                    );
                    self.validate_index_argument(&targets, vocabulary, "IndexedCrossEntropy")?;
                    let mut output = self.take_buffer(row_count)?;
                    if row_count != 0 {
                        let launch_len = u32::try_from(row_count)
                            .context("IndexedCrossEntropy output is too large for a CUDA launch")?;
                        let function = self.kernel_function(module, kernel_name)?;
                        let stream = self.stream.clone();
                        let mut launcher = stream.launch_builder(&function);
                        launcher.arg(logits.as_ref());
                        match &targets {
                            IndexArgument::Exact { values, .. } => {
                                launcher.arg(*values);
                            }
                            IndexArgument::Floating(values) => {
                                launcher.arg(values.as_ref());
                            }
                        }
                        launcher.arg(&mut output);
                        self.record_kernel_launch();
                        unsafe {
                            launcher.launch(
                                self.indexed_loss_row_launch_config(*node_idx, row_count)?
                                    .unwrap_or(LaunchConfig::for_num_elems(launch_len)),
                            )
                        }
                        .context("PTX indexed_cross_entropy kernel launch failed")?;
                    }
                    output
                }
                TensorGraphNode::IndexedCrossEntropyBackward { .. } => {
                    let kernel_name = ptx_graph
                        .kernel_name(*node_idx)
                        .context("Missing compiled IndexedCrossEntropyBackward kernel")?;
                    let ins = planned_inputs;
                    anyhow::ensure!(
                        ins.len() == 3,
                        "IndexedCrossEntropyBackward requires 3 inputs"
                    );
                    let logits = self
                        .values
                        .get(&ins[0])
                        .context("Missing logits for IndexedCrossEntropyBackward")?;
                    let targets = self.index_argument(ins[1])?;
                    let grad_output = self
                        .values
                        .get(&ins[2])
                        .context("Missing grad_output for IndexedCrossEntropyBackward")?;
                    let output_shape = node.shape();
                    let vocabulary = *output_shape
                        .last()
                        .context("IndexedCrossEntropyBackward logits must have rank >= 1")?;
                    anyhow::ensure!(
                        vocabulary > 0,
                        "IndexedCrossEntropyBackward vocabulary must be nonzero"
                    );
                    let row_count = output_shape.iter().product::<usize>() / vocabulary;
                    anyhow::ensure!(
                        logits.len() == row_count * vocabulary,
                        "Cross entropy logits length mismatch"
                    );
                    anyhow::ensure!(
                        targets.len() == row_count,
                        "Cross entropy targets length mismatch"
                    );
                    anyhow::ensure!(
                        grad_output.len() == row_count,
                        "Cross entropy grad_output length mismatch"
                    );
                    self.validate_index_argument(
                        &targets,
                        vocabulary,
                        "IndexedCrossEntropyBackward",
                    )?;
                    let output_len = row_count * vocabulary;
                    let mut output = self.take_buffer(output_len)?;
                    if output_len != 0 {
                        let launch_len = u32::try_from(output_len).context(
                            "IndexedCrossEntropyBackward output is too large for a CUDA launch",
                        )?;
                        let function = self.kernel_function(module, kernel_name)?;
                        let stream = self.stream.clone();
                        let mut launcher = stream.launch_builder(&function);
                        launcher.arg(logits.as_ref());
                        match &targets {
                            IndexArgument::Exact { values, .. } => {
                                launcher.arg(*values);
                            }
                            IndexArgument::Floating(values) => {
                                launcher.arg(values.as_ref());
                            }
                        }
                        launcher.arg(grad_output.as_ref());
                        launcher.arg(&mut output);
                        self.record_kernel_launch();
                        unsafe {
                            launcher.launch(
                                self.indexed_loss_row_launch_config(*node_idx, output_len)?
                                    .unwrap_or(LaunchConfig::for_num_elems(launch_len)),
                            )
                        }
                        .context("PTX indexed_cross_entropy_backward kernel launch failed")?;
                    }
                    output
                }
                TensorGraphNode::BroadcastAxis { .. } => {
                    let kernel_name = ptx_graph
                        .kernel_name(*node_idx)
                        .context("Missing compiled BroadcastAxis kernel")?;

                    let ins = planned_inputs;
                    let input = self
                        .values
                        .get(&ins[0])
                        .context("Missing input for BroadcastAxis")?;

                    // Get output shape
                    let out_shape = graph.graph[*node_idx].shape();
                    let out_len: usize = out_shape.iter().product();

                    let stream = self.stream.clone();
                    let mut out = self.take_buffer(out_len)?;
                    if out_len != 0 {
                        let f = self.kernel_function(module, kernel_name)?;
                        let cfg = LaunchConfig::for_num_elems(out_len as u32);
                        let mut launcher = stream.launch_builder(&f);
                        launcher.arg(input.as_ref());
                        launcher.arg(&mut out);
                        self.record_kernel_launch();
                        unsafe { launcher.launch(cfg) }.with_context(|| {
                            format!("CUDA {} kernel launch failed", kernel_name)
                        })?;
                    }

                    out
                }
                TensorGraphNode::ReduceAxis { .. } => {
                    let kernel_name = ptx_graph
                        .kernel_name(*node_idx)
                        .context("Missing compiled ReduceAxis kernel")?;

                    let ins = planned_inputs;
                    let input = self
                        .values
                        .get(&ins[0])
                        .context("Missing input for ReduceAxis")?;

                    // Get output shape
                    let out_shape = graph.graph[*node_idx].shape();
                    let out_len: usize = out_shape.iter().product();

                    let stream = self.stream.clone();
                    let mut out = self.take_buffer(out_len)?;
                    if out_len != 0 {
                        let f = self.kernel_function(module, kernel_name)?;
                        let cfg = LaunchConfig::for_num_elems(out_len as u32);
                        let mut launcher = stream.launch_builder(&f);
                        launcher.arg(input.as_ref());
                        launcher.arg(&mut out);
                        self.record_kernel_launch();
                        unsafe { launcher.launch(cfg) }.with_context(|| {
                            format!("CUDA {} kernel launch failed", kernel_name)
                        })?;
                    }

                    out
                }
                TensorGraphNode::Transpose { .. } | TensorGraphNode::Permute { .. } => {
                    let kernel_name = ptx_graph
                        .kernel_name(*node_idx)
                        .context("Missing compiled permutation kernel")?;

                    let ins = planned_inputs;
                    let input = self
                        .values
                        .get(&ins[0])
                        .context("Missing input for permutation")?;

                    let len = input.len();
                    let stream = self.stream.clone();
                    let mut out = self.take_buffer(len)?;
                    if len != 0 {
                        let f = self.kernel_function(module, kernel_name)?;
                        let cfg = LaunchConfig::for_num_elems(len as u32);
                        let mut launcher = stream.launch_builder(&f);
                        launcher.arg(input.as_ref());
                        launcher.arg(&mut out);
                        self.record_kernel_launch();
                        unsafe { launcher.launch(cfg) }.with_context(|| {
                            format!("CUDA {} kernel launch failed", kernel_name)
                        })?;
                    }

                    out
                }
                TensorGraphNode::Reshape { .. } | TensorGraphNode::Flatten { .. } => {
                    let input_index = planned_inputs[0];
                    let input = self
                        .values
                        .get(&input_index)
                        .context("Missing input for contiguous view")?;
                    let output_len: usize = node.shape().iter().product();
                    anyhow::ensure!(
                        output_len == input.len(),
                        "Contiguous view changes element count from {} to {output_len}",
                        input.len()
                    );
                    let mut output = self.take_buffer(output_len)?;
                    if output_len != 0 {
                        self.execution_metrics.device_copies += 1;
                        self.stream
                            .clone()
                            .memcpy_dtod(input.as_ref(), &mut output)
                            .context("Failed to copy contiguous view")?;
                    }
                    output
                }
                TensorGraphNode::Conv2d { .. } => {
                    let kernel_name = ptx_graph
                        .kernel_name(*node_idx)
                        .context("Missing compiled Conv2d kernel")?;
                    let inputs = planned_inputs;
                    let input = self
                        .values
                        .get(&inputs[0])
                        .context("Missing input for Conv2d")?;
                    let weight = self
                        .values
                        .get(&inputs[1])
                        .context("Missing weight for Conv2d")?;
                    let output_len: usize = node.shape().iter().product();
                    let mut output = self.take_buffer(output_len)?;
                    if output_len != 0 {
                        let function = self.kernel_function(module, kernel_name)?;
                        let stream = self.stream.clone();
                        let mut launcher = stream.launch_builder(&function);
                        launcher
                            .arg(input.as_ref())
                            .arg(weight.as_ref())
                            .arg(&mut output);
                        self.record_kernel_launch();
                        unsafe {
                            launcher.launch(LaunchConfig::for_num_elems(
                                u32::try_from(output_len)
                                    .context("Conv2d output is too large for a CUDA launch")?,
                            ))
                        }
                        .context("PTX Conv2d kernel launch failed")?;
                    }
                    output
                }
                TensorGraphNode::ConvTranspose2d { .. } => {
                    let kernel_name = ptx_graph
                        .kernel_name(*node_idx)
                        .context("Missing compiled ConvTranspose2d kernel")?;
                    let inputs = planned_inputs;
                    let grad_output = self
                        .values
                        .get(&inputs[0])
                        .context("Missing grad_output for ConvTranspose2d")?;
                    let weight = self
                        .values
                        .get(&inputs[1])
                        .context("Missing weight for ConvTranspose2d")?;
                    let output_len: usize = node.shape().iter().product();
                    let mut output = self.take_buffer(output_len)?;
                    if output_len != 0 {
                        let function = self.kernel_function(module, kernel_name)?;
                        let stream = self.stream.clone();
                        let mut launcher = stream.launch_builder(&function);
                        launcher
                            .arg(grad_output.as_ref())
                            .arg(weight.as_ref())
                            .arg(&mut output);
                        self.record_kernel_launch();
                        unsafe {
                            launcher.launch(LaunchConfig::for_num_elems(
                                u32::try_from(output_len).context(
                                    "ConvTranspose2d output is too large for a CUDA launch",
                                )?,
                            ))
                        }
                        .context("PTX ConvTranspose2d kernel launch failed")?;
                    }
                    output
                }
                TensorGraphNode::Conv2dBackwardWeight { .. } => {
                    let kernel_name = ptx_graph
                        .kernel_name(*node_idx)
                        .context("Missing compiled Conv2dBackwardWeight kernel")?;
                    let inputs = planned_inputs;
                    let input = self
                        .values
                        .get(&inputs[0])
                        .context("Missing input for Conv2dBackwardWeight")?;
                    let grad_output = self
                        .values
                        .get(&inputs[1])
                        .context("Missing grad_output for Conv2dBackwardWeight")?;
                    let output_len: usize = node.shape().iter().product();
                    let mut output = self.take_buffer(output_len)?;
                    if output_len != 0 {
                        let function = self.kernel_function(module, kernel_name)?;
                        let stream = self.stream.clone();
                        let mut launcher = stream.launch_builder(&function);
                        launcher
                            .arg(input.as_ref())
                            .arg(grad_output.as_ref())
                            .arg(&mut output);
                        self.record_kernel_launch();
                        unsafe {
                            launcher.launch(self.conv_weight_tree_launch_config(*node_idx, output_len)?.unwrap_or(LaunchConfig::for_num_elems(u32::try_from(output_len).context("Conv2dBackwardWeight output is too large for a CUDA launch")?)))
                        }
                        .context("PTX Conv2dBackwardWeight kernel launch failed")?;
                    }
                    output
                }
                TensorGraphNode::MaxPool2d { .. } => {
                    let kernel_name = ptx_graph
                        .kernel_name(*node_idx)
                        .context("Missing compiled MaxPool2d kernel")?;
                    let input_index = planned_inputs[0];
                    let input = self
                        .values
                        .get(&input_index)
                        .context("Missing input for MaxPool2d")?;
                    let output_len: usize = node.shape().iter().product();
                    let mut output = self.take_buffer(output_len)?;
                    if output_len != 0 {
                        let function = self.kernel_function(module, kernel_name)?;
                        let stream = self.stream.clone();
                        let mut launcher = stream.launch_builder(&function);
                        launcher.arg(input.as_ref()).arg(&mut output);
                        self.record_kernel_launch();
                        unsafe {
                            launcher.launch(LaunchConfig::for_num_elems(
                                u32::try_from(output_len)
                                    .context("MaxPool2d output is too large for a CUDA launch")?,
                            ))
                        }
                        .context("PTX MaxPool2d kernel launch failed")?;
                    }
                    output
                }
                TensorGraphNode::MaxPool2dBackward { .. } => {
                    let kernel_name = ptx_graph
                        .kernel_name(*node_idx)
                        .context("Missing compiled MaxPool2dBackward kernel")?;
                    let inputs = planned_inputs;
                    let input = self
                        .values
                        .get(&inputs[0])
                        .context("Missing input for MaxPool2dBackward")?;
                    let pooled = self
                        .values
                        .get(&inputs[1])
                        .context("Missing pooled output for MaxPool2dBackward")?;
                    let grad_output = self
                        .values
                        .get(&inputs[2])
                        .context("Missing grad_output for MaxPool2dBackward")?;
                    let output_len: usize = node.shape().iter().product();
                    let mut output = self.take_buffer(output_len)?;
                    if output_len != 0 {
                        let function = self.kernel_function(module, kernel_name)?;
                        let stream = self.stream.clone();
                        let mut launcher = stream.launch_builder(&function);
                        launcher
                            .arg(input.as_ref())
                            .arg(pooled.as_ref())
                            .arg(grad_output.as_ref())
                            .arg(&mut output);
                        self.record_kernel_launch();
                        unsafe {
                            launcher.launch(LaunchConfig::for_num_elems(
                                u32::try_from(output_len).context(
                                    "MaxPool2dBackward output is too large for a CUDA launch",
                                )?,
                            ))
                        }
                        .context("PTX MaxPool2dBackward kernel launch failed")?;
                    }
                    output
                }
            };
            let result = Arc::new(result);
            if let TensorGraphNode::Input { name, .. } = node
                && let Some(host) = inputs.get::<str>(name)
            {
                self.sources.record_input(&result, host);
            }

            let bytes = result.len() * std::mem::size_of::<D::HostType>();
            self.execution_metrics.materialized_values += 1;
            self.execution_metrics.materialized_bytes += bytes;
            let is_source = matches!(
                node,
                TensorGraphNode::Constant { .. }
                    | TensorGraphNode::Input { .. }
                    | TensorGraphNode::Parameter { .. }
            );
            if !is_source && *node_idx != graph_output {
                self.execution_metrics.intermediate_materialized_bytes += bytes;
            }
            self.values.insert(*node_idx, result);
            live_bytes += bytes;
            self.stats.record_live(live_bytes);

            for &dead in &plan_step.release_after {
                if let Some(value) = self.values.remove(&dead) {
                    live_bytes -= self.recycle_value(value);
                }
            }
        }

        self.stats = self.stats.with_pool_stats(self.pool.get_mut().stats());
        self.stats.bytes_allocated += self.exact_indices.allocated_bytes;
        self.stats.buffers_allocated += self.exact_indices.allocations;
        self.execution_metrics.kernel_launches = self.kernel_launches.get();

        if !read_output {
            return Ok(Vec::new());
        }
        let last_node_idx = &graph_output;
        let out_device = self
            .values
            .get(last_node_idx)
            .context("Output value not found after execution")?;
        let mut storage_host = vec![D::HostType::default(); out_device.len()];
        if !storage_host.is_empty() {
            self.execution_metrics.device_copies += 1;
            self.stream
                .clone()
                .memcpy_dtoh(out_device.as_ref(), &mut storage_host)
                .context("Failed to copy output from CUDA device to host")?;
        }
        let output_view = self
            .execution_plan
            .as_ref()
            .and_then(|plan| plan.virtual_value(graph_output))
            .context("Graph output has no virtual value")?;
        let source_shape = self
            .execution_plan
            .as_ref()
            .and_then(|plan| plan.value_shape(output_view.source))
            .context("Virtual output source is absent from the execution plan")?;
        materialize_host_view(storage_host, output_view, source_shape)
    }

    pub(crate) fn context(&self) -> Arc<CudaContext> {
        self.device.clone()
    }

    pub(crate) fn device_value(
        &self,
        node: petgraph::graph::NodeIndex,
    ) -> Result<super::resident::DeviceValue<D>> {
        let plan = self.execution_plan.as_ref().context("no compiled plan")?;
        let view = plan
            .virtual_value(node)
            .context("device value has no layout")?
            .clone();
        let source_shape = plan
            .value_shape(view.source)
            .context("device value has no source shape")?
            .clone();
        let storage = self
            .values
            .get(&node)
            .context("device value was not retained")?
            .clone();
        Ok(super::resident::DeviceValue {
            storage,
            view,
            source_shape,
        })
    }

    pub(crate) fn run_device<G>(
        &mut self,
        graph: &TensorGraph<D::HostType, G>,
        inputs: HashMap<String, Vec<D::HostType>>,
        bindings: HashMap<petgraph::graph::NodeIndex, Arc<CudaSlice<D::HostType>>>,
    ) -> Result<()> {
        self.run_device_with_indices(graph, inputs, HashMap::new(), bindings)
    }

    pub(crate) fn run_device_with_indices<G>(
        &mut self,
        graph: &TensorGraph<D::HostType, G>,
        inputs: HashMap<String, Vec<D::HostType>>,
        indices: crate::tensor::IndexInputs,
        bindings: HashMap<petgraph::graph::NodeIndex, Arc<CudaSlice<D::HostType>>>,
    ) -> Result<()> {
        let gate = self.stream_gate.clone();
        let _guard = lock_stream(&gate);
        self.pending_replay_work = PtxExecutionMetrics::default();
        for (node, value) in &bindings {
            let source = graph
                .graph
                .node_weight(*node)
                .context("device binding node is absent")?;
            anyhow::ensure!(
                matches!(
                    source,
                    TensorGraphNode::Parameter { .. } | TensorGraphNode::Input { .. }
                ),
                "device bindings require source nodes"
            );
            anyhow::ensure!(
                Arc::ptr_eq(value.context(), &self.device),
                "device binding context mismatch"
            );
            anyhow::ensure!(
                value.len() == checked_element_count(source.shape(), "device binding")?,
                "device binding length mismatch"
            );
        }
        let signature = graph_compilation_signature(
            graph,
            self.reduction_mode,
            self.target,
            self.matmul_precision,
            self.matmul_shared_layout,
            self.matmul_pipeline,
            (
                self.fusion_policy,
                self.serial_reduction_block_threads,
                self.matmul_grid_group_m,
                self.experimental_register_tiles,
                self.experimental_kernel_options,
            ),
        );
        if self.module.is_none() || self.compilation_signature.as_deref() != Some(&signature) {
            self.compile_impl(graph)?;
        }
        let index_sources = self.index_plan.prepare(graph, indices)?;
        let index_copies = self.exact_indices.stage(&self.stream, index_sources)?;
        if self.replay_enabled
            && !self.replay_blocked
            && self.run_replay(graph, &inputs, &bindings)?
        {
            self.execution_metrics.device_copies += index_copies;
            if self.replay_metrics.active {
                self.replay_metrics.stable_binding_bytes += self.exact_indices.bytes();
            }
            return Ok(());
        }
        self.bindings = bindings;
        let result = self.execute_compiled_impl(graph, inputs, false);
        self.bindings.clear();
        self.execution_metrics
            .add_work(&std::mem::take(&mut self.pending_replay_work));
        self.kernel_launches
            .set(self.execution_metrics.kernel_launches);
        self.execution_metrics.device_copies += index_copies;
        result.map(|_| ())
    }

    /// Compile and execute, reusing the module for the same graph structure.
    pub fn compile_and_execute<G>(
        &mut self,
        graph: &TensorGraph<D::HostType, G>,
        inputs: HashMap<String, Vec<D::HostType>>,
    ) -> Result<Vec<D::HostType>> {
        self.compile_and_execute_with_indices(graph, inputs, HashMap::new())
    }

    /// Compile and execute with physically separate u32 index sources.
    pub fn compile_and_execute_with_indices<G>(
        &mut self,
        graph: &TensorGraph<D::HostType, G>,
        inputs: HashMap<String, Vec<D::HostType>>,
        indices: crate::tensor::IndexInputs,
    ) -> Result<Vec<D::HostType>> {
        let signature = graph_compilation_signature(
            graph,
            self.reduction_mode,
            self.target,
            self.matmul_precision,
            self.matmul_shared_layout,
            self.matmul_pipeline,
            (
                self.fusion_policy,
                self.serial_reduction_block_threads,
                self.matmul_grid_group_m,
                self.experimental_register_tiles,
                self.experimental_kernel_options,
            ),
        );
        if self.module.is_none() || self.compilation_signature.as_deref() != Some(&signature) {
            self.compile(graph)?;
        }
        self.execute_compiled_with_indices(graph, inputs, indices)
    }
}

fn validate_batch_broadcast(input_shape: &[usize], output_batch_shape: &[usize]) -> Result<()> {
    let input_batch_shape = &input_shape[..input_shape.len() - 2];
    anyhow::ensure!(
        input_batch_shape.len() <= output_batch_shape.len(),
        "MatMul input batch shape {input_batch_shape:?} has higher rank than output batch shape {output_batch_shape:?}"
    );
    let leading = output_batch_shape.len() - input_batch_shape.len();
    for (dimension, &input_size) in input_batch_shape.iter().enumerate() {
        let output_size = output_batch_shape[leading + dimension];
        anyhow::ensure!(
            input_size == 1 || input_size == output_size,
            "Cannot broadcast MatMul batch dimension {input_size} to {output_size}"
        );
    }
    Ok(())
}

fn checked_element_count(shape: &[usize], description: &str) -> Result<usize> {
    shape.iter().try_fold(1usize, |count, &dimension| {
        count
            .checked_mul(dimension)
            .with_context(|| format!("{description} shape {shape:?} overflows usize"))
    })
}

fn signature_usize(signature: &mut Vec<u8>, value: usize) {
    signature.extend_from_slice(&(value as u64).to_le_bytes());
}

fn signature_slice(signature: &mut Vec<u8>, values: &[usize]) {
    signature_usize(signature, values.len());
    for &value in values {
        signature_usize(signature, value);
    }
}

fn signature_str(signature: &mut Vec<u8>, value: &str) {
    signature_usize(signature, value.len());
    signature.extend_from_slice(value.as_bytes());
}

fn graph_compilation_signature<D: crate::tile::TileDType, G>(
    graph: &TensorGraph<D, G>,
    reduction_mode: PtxReductionMode,
    target: PtxTarget,
    matmul_precision: crate::tile::MatMulPrecision,
    matmul_shared_layout: MatMulSharedLayout,
    matmul_pipeline: MatMulPipeline,
    fusion_options: (
        PtxFusionPolicy,
        Option<u32>,
        usize,
        bool,
        ExperimentalKernelOptions,
    ),
) -> Vec<u8> {
    let (
        fusion_policy,
        serial_reduction_block_threads,
        matmul_grid_group_m,
        experimental_register_tiles,
        experimental_kernel_options,
    ) = fusion_options;
    let mut signature = Vec::new();
    signature.extend_from_slice(PTX_GRAPH_SIGNATURE_MAGIC);
    signature.push(PTX_GRAPH_SIGNATURE_VERSION);
    signature.push(match D::TILE_DTYPE {
        crate::tile::DType::F32 => 0,
        crate::tile::DType::F16 => 1,
        crate::tile::DType::BF16 => 2,
        _ => unreachable!("unsupported host storage dtype"),
    });
    signature.push(match reduction_mode {
        PtxReductionMode::Strict => 0,
        PtxReductionMode::DeterministicTree => 1,
        PtxReductionMode::Online => 2,
    });
    signature.push(match matmul_precision {
        crate::tile::MatMulPrecision::StrictF32 => 0,
        crate::tile::MatMulPrecision::AllowTf32 => 1,
        crate::tile::MatMulPrecision::Tf32x3 => 2,
        crate::tile::MatMulPrecision::NativeHalf => 3,
    });
    signature.push(match matmul_shared_layout {
        MatMulSharedLayout::Auto => 0,
        MatMulSharedLayout::Contiguous => 1,
        MatMulSharedLayout::Padded => 2,
        MatMulSharedLayout::Swizzled => 3,
    });
    signature.push(match matmul_pipeline {
        MatMulPipeline::Auto => 0,
        MatMulPipeline::Synchronous => 1,
        MatMulPipeline::DoubleBuffered => 2,
    });
    signature.push(match fusion_policy {
        PtxFusionPolicy::CostAware => 0,
        PtxFusionPolicy::Greedy => 1,
    });
    signature.extend_from_slice(&serial_reduction_block_threads.unwrap_or(0).to_le_bytes());
    signature.extend_from_slice(&matmul_grid_group_m.to_le_bytes());
    signature.push(u8::from(experimental_register_tiles));
    signature.push(u8::from(experimental_kernel_options.plain_bf16_output));
    signature.push(u8::from(experimental_kernel_options.shared_memory_reuse));
    signature.push(u8::from(experimental_kernel_options.online_weight_sharing));
    signature.push(u8::from(experimental_kernel_options.narrow_bf16_tile));
    signature.push(u8::from(
        experimental_kernel_options.bf16_eight_compute_warps,
    ));
    signature.push(match experimental_kernel_options.medium_bf16_tile_policy {
        crate::tile::MediumBf16TilePolicy::Off => 0,
        crate::tile::MediumBf16TilePolicy::SquareOnly => 1,
        crate::tile::MediumBf16TilePolicy::SquareThenTall => 2,
    });
    signature.push(match experimental_kernel_options.column_reduction_policy {
        crate::tile::ColumnReductionPolicy::Off => 0,
        crate::tile::ColumnReductionPolicy::Four => 1,
        crate::tile::ColumnReductionPolicy::Eight => 2,
    });
    signature.push(match experimental_kernel_options.large_bf16_tile_policy {
        crate::tile::LargeBf16TilePolicy::Off => 0,
        crate::tile::LargeBf16TilePolicy::Tall => 1,
        crate::tile::LargeBf16TilePolicy::Wide => 2,
        crate::tile::LargeBf16TilePolicy::TallThenWide => 3,
        crate::tile::LargeBf16TilePolicy::Square128Streamed => 4,
    });
    signature.push(
        match experimental_kernel_options.batched_attention_bf16_tile_policy {
            crate::tile::BatchedAttentionBf16TilePolicy::Off => 0,
            crate::tile::BatchedAttentionBf16TilePolicy::TallScores => 1,
            crate::tile::BatchedAttentionBf16TilePolicy::WideScores => 2,
            crate::tile::BatchedAttentionBf16TilePolicy::TallScoresByGrid => 3,
        },
    );
    signature.push(u8::from(
        experimental_kernel_options.ordinary_bf16_four_warp_block,
    ));
    signature.push(u8::from(
        experimental_kernel_options.indexed_loss_cooperative_ordered_backward,
    ));
    signature.push(u8::from(
        experimental_kernel_options.indexed_loss_cooperative_ordered_forward,
    ));
    signature.extend_from_slice(&target.compute_capability.0.to_le_bytes());
    signature.extend_from_slice(&target.compute_capability.1.to_le_bytes());
    signature_usize(&mut signature, target.multiprocessor_count);
    for budget in [
        target.matmul_resources.shared_bytes_per_block,
        target.matmul_resources.shared_bytes_per_sm,
        target.matmul_resources.registers_per_sm,
        target.matmul_resources.threads_per_sm,
    ] {
        signature_usize(&mut signature, budget);
    }
    signature_usize(&mut signature, graph.graph.node_count());
    for node_index in graph.graph.node_indices() {
        let node = &graph.graph[node_index];
        signature_usize(&mut signature, node_index.index());
        signature_str(&mut signature, node.name());
        match node {
            TensorGraphNode::Linear { .. } => signature.push(34),
            TensorGraphNode::Cast { dtype, .. } => {
                signature.push(33);
                signature.push(match dtype {
                    crate::tensor::ValueDType::F32 => 0,
                    crate::tensor::ValueDType::BF16 => 1,
                });
            }
            TensorGraphNode::IndexSource { source } => {
                signature.push(32);
                match source.source() {
                    crate::tensor::indices::IndexSource::Input(name) => {
                        signature.push(0);
                        signature_str(&mut signature, name);
                    }
                    crate::tensor::indices::IndexSource::Constant(_) => signature.push(1),
                }
            }
            TensorGraphNode::Constant { .. } => signature.push(0),
            TensorGraphNode::Input { name, .. } => {
                signature.push(1);
                signature_str(&mut signature, name);
            }
            TensorGraphNode::Parameter { .. } => signature.push(2),
            TensorGraphNode::Unary { op, .. } => {
                signature.push(3);
                signature.push(match op {
                    crate::tensor::UnaryOp::Neg => 0,
                    crate::tensor::UnaryOp::Exp => 1,
                    crate::tensor::UnaryOp::Log => 2,
                    crate::tensor::UnaryOp::Relu => 3,
                });
            }
            #[cfg(feature = "fusion")]
            TensorGraphNode::FusedUnary { ops, .. } => {
                signature.push(4);
                signature_usize(&mut signature, ops.len());
                for op in ops {
                    signature.push(match op {
                        crate::tensor::UnaryOp::Neg => 0,
                        crate::tensor::UnaryOp::Exp => 1,
                        crate::tensor::UnaryOp::Log => 2,
                        crate::tensor::UnaryOp::Relu => 3,
                    });
                }
            }
            TensorGraphNode::Binary { op, .. } => {
                signature.push(5);
                signature.push(match op {
                    crate::tensor::BinaryOp::Add => 0,
                    crate::tensor::BinaryOp::Sub => 1,
                    crate::tensor::BinaryOp::Mul => 2,
                    crate::tensor::BinaryOp::Div => 3,
                });
            }
            TensorGraphNode::MatMul { .. } => signature.push(6),
            TensorGraphNode::Embedding { .. } => signature.push(7),
            TensorGraphNode::EmbeddingBackward { .. } => signature.push(8),
            TensorGraphNode::IndexedCrossEntropy { .. } => signature.push(9),
            TensorGraphNode::IndexedCrossEntropyBackward { .. } => signature.push(10),
            TensorGraphNode::Transpose { .. } => signature.push(11),
            TensorGraphNode::Reshape { .. } => signature.push(12),
            TensorGraphNode::Permute { axes, .. } => {
                signature.push(13);
                signature_slice(&mut signature, axes);
            }
            TensorGraphNode::BroadcastAxis { axis, .. } => {
                signature.push(14);
                signature_usize(&mut signature, *axis);
            }
            TensorGraphNode::ReduceAxis { op, axis, .. } => {
                signature.push(15);
                signature.push(match op {
                    crate::tensor::ReduceOp::Sum => 0,
                    crate::tensor::ReduceOp::Max => 1,
                    crate::tensor::ReduceOp::Mean => 2,
                });
                signature_usize(&mut signature, *axis);
            }
            TensorGraphNode::Gt { .. } => signature.push(16),
            TensorGraphNode::Narrow { axis, start, .. } => {
                signature.push(30);
                signature_usize(&mut signature, *axis);
                signature_usize(&mut signature, *start);
            }
            TensorGraphNode::Concat { axis, .. } => {
                signature.push(31);
                signature_usize(&mut signature, *axis);
            }
            TensorGraphNode::Mask { .. } => signature.push(17),
            TensorGraphNode::Conv2d {
                stride, padding, ..
            } => {
                signature.push(18);
                signature_usize(&mut signature, *stride);
                signature_usize(&mut signature, *padding);
            }
            TensorGraphNode::ConvTranspose2d {
                stride, padding, ..
            } => {
                signature.push(19);
                signature_usize(&mut signature, *stride);
                signature_usize(&mut signature, *padding);
            }
            TensorGraphNode::Conv2dBackwardWeight {
                stride, padding, ..
            } => {
                signature.push(20);
                signature_usize(&mut signature, *stride);
                signature_usize(&mut signature, *padding);
            }
            TensorGraphNode::MaxPool2d {
                kernel_size,
                stride,
                ..
            } => {
                signature.push(21);
                signature_usize(&mut signature, *kernel_size);
                signature_usize(&mut signature, *stride);
            }
            TensorGraphNode::MaxPool2dBackward {
                kernel_size,
                stride,
                ..
            } => {
                signature.push(22);
                signature_usize(&mut signature, *kernel_size);
                signature_usize(&mut signature, *stride);
            }
            TensorGraphNode::Flatten { .. } => signature.push(23),
        }
        signature_slice(&mut signature, node.shape());
    }

    let mut edges: Vec<(usize, usize, usize)> = graph
        .graph
        .edge_references()
        .map(|edge| (edge.source().index(), edge.target().index(), *edge.weight()))
        .collect();
    edges.sort_unstable();
    signature_usize(&mut signature, edges.len());
    for (source, target, weight) in edges {
        signature_usize(&mut signature, source);
        signature_usize(&mut signature, target);
        signature_usize(&mut signature, weight);
    }
    signature
}

fn batch_matrix_offset(
    output_batch: usize,
    output_batch_shape: &[usize],
    input_shape: &[usize],
) -> usize {
    let input_batch_shape = &input_shape[..input_shape.len() - 2];
    let leading = output_batch_shape.len() - input_batch_shape.len();
    let mut remaining = output_batch;
    let mut output_coordinates = vec![0; output_batch_shape.len()];
    for dimension in (0..output_batch_shape.len()).rev() {
        output_coordinates[dimension] = remaining % output_batch_shape[dimension];
        remaining /= output_batch_shape[dimension];
    }
    input_batch_shape
        .iter()
        .enumerate()
        .fold(0, |offset, (dimension, &size)| {
            let coordinate = if size == 1 {
                0
            } else {
                output_coordinates[leading + dimension]
            };
            offset * size + coordinate
        })
}

impl<D: CudaDType> Executor<D::HostType> for PtxExecutor<D> {
    fn execute<G>(
        &mut self,
        graph: &TensorGraph<D::HostType, G>,
        inputs: HashMap<String, Vec<D::HostType>>,
    ) -> Result<Vec<D::HostType>>
    where
        TensorGraph<D::HostType, G>: Clone,
    {
        let signature = graph_compilation_signature(
            graph,
            self.reduction_mode,
            self.target,
            self.matmul_precision,
            self.matmul_shared_layout,
            self.matmul_pipeline,
            (
                self.fusion_policy,
                self.serial_reduction_block_threads,
                self.matmul_grid_group_m,
                self.experimental_register_tiles,
                self.experimental_kernel_options,
            ),
        );
        let needs_compilation = self.module.is_none()
            || self.ptx_graph.is_none()
            || self.compilation_signature.as_deref() != Some(signature.as_slice());
        if needs_compilation {
            self.compile_owned(graph.clone())?;
        }
        self.execute_compiled(graph, inputs)
    }

    fn execute_with_indices<G>(
        &mut self,
        graph: &TensorGraph<D::HostType, G>,
        inputs: HashMap<String, Vec<D::HostType>>,
        indices: crate::tensor::IndexInputs,
    ) -> Result<Vec<D::HostType>>
    where
        TensorGraph<D::HostType, G>: Clone,
        TensorGraph<f32, G>: Clone,
    {
        let signature = graph_compilation_signature(
            graph,
            self.reduction_mode,
            self.target,
            self.matmul_precision,
            self.matmul_shared_layout,
            self.matmul_pipeline,
            (
                self.fusion_policy,
                self.serial_reduction_block_threads,
                self.matmul_grid_group_m,
                self.experimental_register_tiles,
                self.experimental_kernel_options,
            ),
        );
        if self.module.is_none()
            || self.compilation_signature.as_deref() != Some(signature.as_slice())
        {
            self.compile(graph)?;
        }
        self.execute_compiled_with_indices(graph, inputs, indices)
    }

    fn get_gradients(
        &self,
        graph: &TensorGraph<D::HostType, WithGrad>,
    ) -> HashMap<usize, Vec<D::HostType>> {
        let mut result = HashMap::new();

        // Iterate through all parameters in the gradient metadata
        for (param_id, grad_node_idx) in &graph.gradient_metadata().param_to_grad {
            // Look up the gradient value from our computed values
            if let Some(grad_value) = self.get_value(*grad_node_idx) {
                result.insert(*param_id, grad_value);
            }
        }

        result
    }
}

fn lock_stream(gate: &Mutex<()>) -> MutexGuard<'_, ()> {
    match gate.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

impl<D: CudaDType> Drop for PtxExecutor<D> {
    fn drop(&mut self) {
        let gate = self.stream_gate.clone();
        let _guard = lock_stream(&gate);
        self.replay = None;
        self.exact_indices.buffers.clear();
        self.bindings.clear();
        self.values.clear();
        self.sources.clear();
        self.pool.get_mut().clear();
        self.functions.get_mut().clear();
        self.module = None;
    }
}

#[cfg(test)]
mod experiment_signature_tests {
    use super::*;
    use crate::tensor::TensorExpr;

    #[test]
    fn online_weight_sharing_codegen_is_available_without_a_device() {
        let graph: TensorGraph<f32> =
            crate::nn::softmax(TensorExpr::input("scores", vec![2, 33]), 1)
                .matmul(TensorExpr::input("values", vec![33, 64]))
                .into();
        let mut plan =
            PtxExecutionPlan::build_with_reduction_mode(&graph, PtxReductionMode::Online).unwrap();
        plan.fuse_online(&graph).unwrap();
        assert_eq!(plan.online_regions().len(), 1);
        let mut barriers = Vec::new();
        for enabled in [false, true] {
            plan.set_online_weight_sharing(enabled);
            let ir = plan.online_regions()[0].lower_to_tile_ir(0).unwrap();
            ir.validate_layouts().unwrap();
            assert_eq!(ir.shared_mem_bytes, 33 * 4);
            let mut tiles = TileGraph::from(graph.clone());
            tiles.graph.clear();
            tiles.graph.add_node(ir);
            let ptx = PtxGraph::from(tiles).to_ptx();
            barriers.push(ptx.matches("bar.sync").count());
            if let Some(directory) = std::env::var_os("TNSR_PAPER_PTX_DIR") {
                let directory = std::path::PathBuf::from(directory);
                std::fs::create_dir_all(&directory).unwrap();
                let mode = if enabled { "shared" } else { "baseline" };
                std::fs::write(directory.join(format!("online-{mode}.ptx")), ptx).unwrap();
            }
        }
        assert_eq!(barriers[1], barriers[0] + 1);
    }

    #[test]
    fn narrow_bf16_override_codegen_without_device() -> anyhow::Result<()> {
        let target = PtxTarget::sm80();
        for m in [128, 129, 512] {
            let graph: TensorGraph<half::bf16> = TensorExpr::input("a", vec![2, m, 64])
                .matmul(TensorExpr::input("b", vec![64, 64]))
                .into();
            let mut plan = PtxExecutionPlan::build(&graph)?;
            plan.schedule_matmuls(
                target.matmul_capabilities(),
                crate::tile::MatMulPrecision::NativeHalf,
                target.matmul_resources,
                MatMulSharedLayout::Auto,
                MatMulPipeline::Auto,
                true,
                false,
            )?;
            plan.set_matmul_narrow_bf16_tile(false, target.matmul_resources);
            assert!(
                plan.matmul_regions()
                    .iter()
                    .all(|r| r.schedule.block_tile.m == 16)
            );
            plan.set_matmul_narrow_bf16_tile(true, target.matmul_resources);
            assert!(!plan.matmul_regions().is_empty());
            for region in plan.matmul_regions() {
                let schedule = region.schedule;
                assert_eq!(
                    schedule.block_tile,
                    crate::tile::MatMulTile {
                        m: 32,
                        n: 32,
                        k: 32
                    }
                );
                assert_eq!(schedule.warp_topology, (2, 2));
                assert_eq!(schedule.block_threads, (32, 8, 1));
                assert_eq!(schedule.thread_count(), 256);
                let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
                tiles.add_matmul_regions(std::slice::from_ref(region))?;
                tiles.optimize_indices()?;
                for ir in tiles.graph.node_weights() {
                    ir.validate_layouts()?;
                }
                let ptx = PtxGraph::from(tiles).with_target(target).module_source();
                assert!(ptx.contains(".shared .align 32 .b8 shared[8192];"));
                assert_eq!(ptx.matches("wmma.mma.sync").count(), 2);
                assert!(!ptx.contains("cp.async"));
            }
            plan.set_matmul_shared_memory_reuse(true);
            assert!(
                plan.matmul_regions()
                    .iter()
                    .all(|r| r.schedule.shared_mem_bytes() == 4096)
            );
        }
        Ok(())
    }

    #[test]
    fn narrow_bf16_tile_preserves_forward_backward_and_replay() -> anyhow::Result<()> {
        use crate::ptx::BF16;
        use crate::tensor::Parameter;
        use half::bf16;
        if let Err(error) = CudaContext::new(0) {
            assert!(
                std::env::var("TNSR_REQUIRE_CUDA").as_deref() != Ok("1"),
                "{error:#}"
            );
            return Ok(());
        }
        let bits = |values: Vec<bf16>| values.into_iter().map(bf16::to_bits).collect::<Vec<_>>();
        for m in [128, 129] {
            let a = Parameter::new(vec![bf16::from_f32(0.125); m * 64], vec![m, 64]);
            let b = Parameter::new(
                (0..4096)
                    .map(|i| bf16::from_f32((i % 13) as f32 / 64. - 0.0625))
                    .collect(),
                vec![64, 64],
            );
            let output = TensorExpr::from(a.clone()).matmul(TensorExpr::from(b.clone()));
            let mut graph = TensorGraph::new();
            let forward = graph.add_expr(&output, false);
            let loss = graph.add_expr(&output.mean_all(), false);
            let graph = graph.with_gradients(loss);
            for replay in [false, true] {
                // Production resident replay owns a private capture-capable
                // stream; the default executor uses the legacy default stream.
                let make_executor = || {
                    let context = CudaContext::new(0)?;
                    // SAFETY: as in ResidentAdam replay, every allocation,
                    // upload, kernel and readback for this executor uses its
                    // sole private ordered stream. No buffers cross executors;
                    // executor Drop synchronizes before releasing storage.
                    // Disable cudarc events before any buffers are allocated,
                    // since event dependencies are not capture-compatible.
                    unsafe { context.disable_event_tracking() };
                    PtxExecutor::<BF16>::with_stream(
                        context.new_stream()?,
                        PtxReductionMode::Strict,
                        Arc::new(Mutex::new(())),
                    )
                };
                let mut baseline = make_executor()?;
                let mut candidate = make_executor()?;
                for executor in [&mut baseline, &mut candidate] {
                    executor.set_matmul_precision(crate::tile::MatMulPrecision::NativeHalf);
                    executor.enable_graph_replay(replay);
                }
                candidate.set_experimental_narrow_bf16_tile(true);
                for phase in 0..3 {
                    a.set_values(
                        (0..m * 64)
                            .map(|i| bf16::from_f32(((i + phase) % 17) as f32 / 32. - 0.125))
                            .collect(),
                    )?;
                    // Replay requires device bindings for every parameter.
                    // Fresh per-executor allocations also exercise stable
                    // replay-source refresh when parameter contents change.
                    for executor in [&mut baseline, &mut candidate] {
                        let mut bindings = HashMap::new();
                        for node in graph.graph.node_indices() {
                            if let TensorGraphNode::Parameter { data, .. } = &graph[node] {
                                let values = data
                                    .lock()
                                    .map_err(|e| anyhow::anyhow!("test parameter lock: {e}"))?;
                                let mut device_values =
                                    executor.stream.alloc_zeros(values.len())?;
                                executor
                                    .stream
                                    .memcpy_htod(values.as_slice(), &mut device_values)?;
                                bindings.insert(node, Arc::new(device_values));
                            }
                        }
                        executor.run_device(&graph, HashMap::new(), bindings)?;
                    }
                    let plan = candidate.execution_plan().unwrap();
                    assert!(
                        plan.matmul_regions()
                            .iter()
                            .any(|r| r.schedule.logical_shape.m == m
                                && r.schedule.logical_shape.n == 64
                                && r.schedule.block_tile.m == 32)
                    );
                    for node in [forward, loss] {
                        let old = baseline.get_value(node).unwrap();
                        assert!(old.iter().all(|v| v.is_finite()));
                        assert_eq!(bits(old), bits(candidate.get_value(node).unwrap()));
                    }
                    let old = baseline.get_gradients(&graph);
                    let new = candidate.get_gradients(&graph);
                    for id in [a.id(), b.id()] {
                        assert!(old[&id].iter().all(|v| v.is_finite()));
                        assert_eq!(bits(old[&id].clone()), bits(new[&id].clone()));
                    }
                }
                if replay {
                    assert!(
                        candidate.replay_metrics().graph_launches > 0,
                        "candidate replay metrics: {:?}",
                        candidate.replay_metrics()
                    );
                    assert_eq!(
                        candidate.replay_metrics().fallbacks,
                        0,
                        "candidate replay metrics: {:?}",
                        candidate.replay_metrics()
                    );
                    assert!(
                        baseline.replay_metrics().graph_launches > 0,
                        "baseline replay metrics: {:?}",
                        baseline.replay_metrics()
                    );
                    assert_eq!(
                        baseline.replay_metrics().fallbacks,
                        0,
                        "baseline replay metrics: {:?}",
                        baseline.replay_metrics()
                    );
                }
                candidate.set_experimental_narrow_bf16_tile(false);
                let error = candidate
                    .execute_compiled(&graph, HashMap::new())
                    .unwrap_err();
                assert!(error.to_string().contains("No compiled graph signature"));
                candidate.execute(&graph, HashMap::new())?;
                assert_eq!(
                    bits(baseline.get_value(loss).unwrap()),
                    bits(candidate.get_value(loss).unwrap())
                );
            }
        }
        Ok(())
    }

    #[test]
    fn bf16_eight_compute_warps_codegen_preserves_geometry_and_memory() -> anyhow::Result<()> {
        let mut target = PtxTarget::sm80();
        target.matmul_resources.shared_bytes_per_sm = 100 * 1024;
        for (m, n, k, transpose_b) in [
            (2048, 256, 64, false),
            (4096, 1408, 512, false),
            (4096, 1408, 512, true),
            (4096, 512, 1408, true),
            (2049, 257, 65, true),
        ] {
            let b = TensorExpr::input("b", if transpose_b { vec![n, k] } else { vec![k, n] });
            let b = if transpose_b { b.transpose() } else { b };
            let graph: TensorGraph<half::bf16> =
                TensorExpr::input("a", vec![m, k]).matmul(b).into();
            let mut plan = PtxExecutionPlan::build(&graph)?;
            plan.schedule_matmuls(
                target.matmul_capabilities(),
                crate::tile::MatMulPrecision::NativeHalf,
                target.matmul_resources,
                MatMulSharedLayout::Auto,
                MatMulPipeline::Auto,
                true,
                false,
            )?;
            let old = plan.matmul_regions().to_vec();
            plan.set_bf16_eight_compute_warps(false);
            assert_eq!(plan.matmul_regions()[0].schedule, old[0].schedule);
            plan.set_bf16_eight_compute_warps(true);
            for (before, after) in old.iter().zip(plan.matmul_regions()) {
                let mut expected = before.schedule;
                expected.warp_topology = (2, 4);
                expected.warp_fragments = (2, 1);
                assert_eq!(after.schedule, expected);
                let emit = |r: &crate::tile::MatMulRegion| -> anyhow::Result<String> {
                    let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
                    tiles.add_matmul_regions(std::slice::from_ref(r))?;
                    tiles.optimize_indices()?;
                    for ir in tiles.graph.node_weights() {
                        ir.validate_layouts()?;
                    }
                    Ok(PtxGraph::from(tiles).with_target(target).module_source())
                };
                let a = emit(before)?;
                let b = emit(after)?;
                for (opcode, old_count, new_count) in [
                    ("wmma.load.a", 4, 4),
                    ("wmma.load.b", 4, 2),
                    ("wmma.mma", 8, 4),
                    ("wmma.store.d", 4, 2),
                ] {
                    assert_eq!(a.matches(opcode).count(), old_count);
                    assert_eq!(b.matches(opcode).count(), new_count);
                }
                for opcode in ["cp.async", "bar.sync"] {
                    assert_eq!(a.matches(opcode).count(), b.matches(opcode).count());
                }
                let shared = format!(
                    ".shared .align 32 .b8 shared[{}];",
                    before.schedule.shared_mem_bytes()
                );
                assert!(a.contains(&shared) && b.contains(&shared));
            }
        }
        Ok(())
    }

    #[test]
    fn bf16_eight_compute_warps_toggle_invalidates_compilation_and_matches_bits()
    -> anyhow::Result<()> {
        use crate::ptx::BF16;
        use half::bf16;
        if let Err(error) = CudaContext::new(0) {
            assert!(
                std::env::var("TNSR_REQUIRE_CUDA").as_deref() != Ok("1"),
                "{error:#}"
            );
            return Ok(());
        }
        let graph: TensorGraph<bf16> = TensorExpr::input("a", vec![2048, 64])
            .matmul(TensorExpr::input("b", vec![64, 256]))
            .into();
        let mut executor = PtxExecutor::<BF16>::try_new_for_dtype()?;
        executor.set_matmul_precision(crate::tile::MatMulPrecision::NativeHalf);
        let mut expected = vec![];
        for (iteration, enabled) in [false, true, false].into_iter().enumerate() {
            executor.set_experimental_bf16_eight_compute_warps(enabled);
            if iteration > 0 {
                let error = executor
                    .execute_compiled(&graph, HashMap::new())
                    .unwrap_err();
                assert!(error.to_string().contains("No compiled graph signature"));
            }
            executor.compile(&graph)?;
            let schedule = executor.execution_plan().unwrap().matmul_regions()[0].schedule;
            assert_eq!(
                schedule.warp_topology,
                if enabled { (2, 4) } else { (2, 2) }
            );
            assert_eq!(
                schedule.warp_fragments,
                if enabled { (2, 1) } else { (2, 2) }
            );
            let mut actual = vec![];
            for phase in 0..2 {
                let feed = HashMap::from([
                    (
                        "a".into(),
                        (0..2048 * 64)
                            .map(|i| bf16::from_f32(((i + phase) % 17) as f32 / 32. - 0.125))
                            .collect(),
                    ),
                    (
                        "b".into(),
                        (0..64 * 256)
                            .map(|i| bf16::from_f32((i % 13) as f32 / 64. - 0.0625))
                            .collect(),
                    ),
                ]);
                let values = executor.execute_compiled(&graph, feed)?;
                assert!(values.iter().all(|v| v.is_finite()));
                actual.push(values.into_iter().map(bf16::to_bits).collect::<Vec<_>>());
            }
            if iteration == 0 {
                expected = actual;
            } else {
                assert_eq!(actual, expected);
            }
        }
        Ok(())
    }

    #[test]
    fn medium_bf16_plan_counts_exclude_legacy_and_batch_fallback() -> anyhow::Result<()> {
        use crate::tile::MediumBf16TilePolicy as Policy;
        let target = PtxTarget::sm80();
        for (m, n, batch, policy, expected) in [
            (128, 4096, false, Policy::SquareOnly, (1, 0)),
            (128, 2048, false, Policy::SquareThenTall, (0, 1)),
            (128, 2048, false, Policy::SquareOnly, (0, 0)),
            (128, 4096, true, Policy::SquareThenTall, (0, 0)),
            (4096, 1408, false, Policy::SquareThenTall, (0, 0)),
        ] {
            let a_shape = if batch { vec![2, m, 128] } else { vec![m, 128] };
            let graph: TensorGraph<half::bf16> = TensorExpr::input("a", a_shape)
                .matmul(TensorExpr::input("b", vec![128, n]))
                .into();
            let mut plan = PtxExecutionPlan::build(&graph)?;
            plan.schedule_matmuls_with_medium_policy(
                target.matmul_capabilities(),
                crate::tile::MatMulPrecision::NativeHalf,
                target.matmul_resources,
                MatMulSharedLayout::Auto,
                MatMulPipeline::Auto,
                true,
                false,
                policy,
            )?;
            assert_eq!(plan.medium_bf16_tile_counts(), expected);
            for r in plan.matmul_regions() {
                let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
                tiles.add_matmul_regions(std::slice::from_ref(r))?;
                tiles.optimize_indices()?;
                for ir in tiles.graph.node_weights() {
                    ir.validate_layouts()?;
                }
            }
            plan.schedule_matmuls(
                target.matmul_capabilities(),
                crate::tile::MatMulPrecision::NativeHalf,
                target.matmul_resources,
                MatMulSharedLayout::Auto,
                MatMulPipeline::Auto,
                true,
                false,
            )?;
            assert_eq!(plan.medium_bf16_tile_counts(), (0, 0));
        }
        Ok(())
    }

    #[test]
    fn medium_bf16_policy_toggle_invalidates_identity_and_matches_bits() -> anyhow::Result<()> {
        use crate::{ptx::BF16, tile::MediumBf16TilePolicy as Policy};
        use half::bf16;
        if let Err(error) = CudaContext::new(0) {
            assert!(
                std::env::var("TNSR_REQUIRE_CUDA").as_deref() != Ok("1"),
                "{error:#}"
            );
            return Ok(());
        }
        let graph: TensorGraph<bf16> = TensorExpr::input("a", vec![128, 128])
            .matmul(TensorExpr::input("b", vec![128, 2048]))
            .into();
        let mut executor = PtxExecutor::<BF16>::try_new_for_dtype()?;
        executor.set_matmul_precision(crate::tile::MatMulPrecision::NativeHalf);
        let mut expected = vec![];
        for (i, policy) in [
            Policy::Off,
            Policy::SquareOnly,
            Policy::SquareThenTall,
            Policy::Off,
        ]
        .into_iter()
        .enumerate()
        {
            executor.set_experimental_medium_bf16_tile_policy(policy);
            if i > 0 {
                let error = executor
                    .execute_compiled(&graph, HashMap::new())
                    .unwrap_err();
                assert!(error.to_string().contains("No compiled graph signature"));
            }
            executor.compile(&graph)?;
            assert_eq!(
                executor.execution_plan().unwrap().medium_bf16_tile_counts(),
                if policy == Policy::SquareThenTall {
                    (0, 1)
                } else {
                    (0, 0)
                }
            );
            let mut actual = vec![];
            for phase in 0..2 {
                let feeds = HashMap::from([
                    (
                        "a".into(),
                        (0..128 * 128)
                            .map(|j| bf16::from_f32(((j + phase) % 17) as f32 / 32. - 0.125))
                            .collect(),
                    ),
                    (
                        "b".into(),
                        (0..128 * 2048)
                            .map(|j| bf16::from_f32((j % 13) as f32 / 64. - 0.0625))
                            .collect(),
                    ),
                ]);
                let output = executor.execute_compiled(&graph, feeds)?;
                assert!(output.iter().all(|v| v.is_finite()));
                actual.push(output.into_iter().map(bf16::to_bits).collect::<Vec<_>>());
            }
            if i == 0 {
                expected = actual;
            } else {
                assert_eq!(actual, expected);
            }
        }
        Ok(())
    }

    #[test]
    fn column_policy_toggle_invalidates_cache_and_preserves_bf16_reduction_bits()
    -> anyhow::Result<()> {
        use crate::{ptx::BF16, tile::ColumnReductionPolicy as Policy};
        use half::bf16;
        if let Err(error) = CudaContext::new(0) {
            assert!(
                std::env::var("TNSR_REQUIRE_CUDA").as_deref() != Ok("1"),
                "{error:#}"
            );
            return Ok(());
        }
        let graph: TensorGraph<bf16> = TensorExpr::input("x", vec![513, 9]).reduce_sum(0).into();
        let mut executor =
            PtxExecutor::<BF16>::try_new_with_reduction_mode(PtxReductionMode::DeterministicTree)?;
        let mut expected = Vec::new();
        for (iteration, policy) in [Policy::Off, Policy::Four, Policy::Eight, Policy::Off]
            .into_iter()
            .enumerate()
        {
            executor.set_experimental_column_reduction_policy(policy);
            if iteration > 0 {
                let error = executor
                    .execute_compiled(&graph, HashMap::new())
                    .unwrap_err();
                assert!(error.to_string().contains("No compiled graph signature"));
            }
            executor.compile(&graph)?;
            assert_eq!(
                executor
                    .execution_plan()
                    .expect("compiled plan")
                    .column_reduction_counts(),
                match policy {
                    Policy::Off => (0, 0),
                    Policy::Four => (1, 0),
                    Policy::Eight => (0, 1),
                }
            );
            let mut observations = Vec::new();
            for phase in 0..3 {
                let values: Vec<_> = (0..513 * 9)
                    .map(|i| bf16::from_f32(((i + phase) % 31) as f32 / 16. - 0.75))
                    .collect();
                let actual =
                    executor.execute_compiled(&graph, HashMap::from([("x".into(), values)]))?;
                observations.push(actual.into_iter().map(bf16::to_bits).collect::<Vec<_>>());
            }
            if iteration == 0 {
                expected = observations;
            } else {
                assert_eq!(observations, expected);
            }
            for value in [-0.0f32, f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
                let actual = executor.execute_compiled(
                    &graph,
                    HashMap::from([("x".into(), vec![bf16::from_f32(value); 513 * 9])]),
                )?;
                for actual in actual {
                    if value.is_nan() {
                        assert!(actual.is_nan());
                    } else if value.is_infinite() {
                        assert_eq!(actual.to_f32(), value);
                    } else {
                        assert_eq!(actual.to_bits(), bf16::from_f32(0.0).to_bits());
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn large_bf16_policy_cache_toggle_exact_output_and_nonfinite_classes() -> anyhow::Result<()> {
        use crate::{ptx::BF16, tile::LargeBf16TilePolicy as Policy};
        use half::bf16;
        if let Err(error) = CudaContext::new(0) {
            assert!(
                std::env::var("TNSR_REQUIRE_CUDA").as_deref() != Ok("1"),
                "{error:#}"
            );
            return Ok(());
        }
        let graph: TensorGraph<bf16> = TensorExpr::input("a", vec![2048, 128])
            .matmul(TensorExpr::input("b", vec![128, 512]))
            .into();
        let mut executor = PtxExecutor::<BF16>::try_new_for_dtype()?;
        executor.set_matmul_precision(crate::tile::MatMulPrecision::NativeHalf);
        executor.set_experimental_shared_memory_reuse(true);
        let mut expected = Vec::new();
        for (iteration, policy) in [
            Policy::Off,
            Policy::Tall,
            Policy::Wide,
            Policy::TallThenWide,
            Policy::Off,
        ]
        .into_iter()
        .enumerate()
        {
            executor.set_experimental_large_bf16_tile_policy(policy);
            if iteration > 0 {
                assert!(
                    executor
                        .execute_compiled(&graph, HashMap::new())
                        .unwrap_err()
                        .to_string()
                        .contains("No compiled graph signature")
                );
            }
            executor.compile(&graph)?;
            assert_eq!(
                executor.execution_plan().unwrap().large_bf16_tile_counts(),
                match policy {
                    Policy::Off => (0, 0),
                    Policy::Wide => (0, 1),
                    _ => (1, 0),
                }
            );
            let mut phases = Vec::new();
            for phase in 0..2 {
                let feeds = HashMap::from([
                    (
                        "a".into(),
                        (0..2048 * 128)
                            .map(|i| bf16::from_f32(((i + phase) % 17) as f32 / 64. - 0.125))
                            .collect(),
                    ),
                    (
                        "b".into(),
                        (0..128 * 512)
                            .map(|i| bf16::from_f32((i % 13) as f32 / 32. - 0.0625))
                            .collect(),
                    ),
                ]);
                let actual = executor.execute_compiled(&graph, feeds)?;
                assert!(actual.iter().all(|v| v.is_finite()));
                phases.push(actual.into_iter().map(bf16::to_bits).collect::<Vec<_>>());
            }
            if iteration == 0 {
                expected = phases;
            } else {
                assert_eq!(phases, expected);
            }
            for (left, right, expected_nan) in [
                (1.0f32, f32::INFINITY, false),
                (1., f32::NEG_INFINITY, false),
                (0., f32::INFINITY, true),
                (-0., 1., false),
            ] {
                let actual = executor.execute_compiled(
                    &graph,
                    HashMap::from([
                        ("a".into(), vec![bf16::from_f32(left); 2048 * 128]),
                        ("b".into(), vec![bf16::from_f32(right); 128 * 512]),
                    ]),
                )?;
                for value in actual {
                    if expected_nan {
                        assert!(value.is_nan());
                    } else if right.is_infinite() {
                        assert_eq!(value.to_f32(), right);
                    } else {
                        assert_eq!(value.to_bits(), bf16::from_f32(0.0).to_bits());
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn batched_attention_policies_have_distinct_compilation_identity() {
        let graph: TensorGraph<f32> = TensorExpr::input("a", vec![8, 8, 512, 64])
            .matmul(TensorExpr::input("b", vec![8, 8, 64, 512]))
            .into();
        let mut seen = std::collections::HashSet::new();
        for policy in [
            crate::tile::BatchedAttentionBf16TilePolicy::Off,
            crate::tile::BatchedAttentionBf16TilePolicy::TallScores,
            crate::tile::BatchedAttentionBf16TilePolicy::WideScores,
            crate::tile::BatchedAttentionBf16TilePolicy::TallScoresByGrid,
        ] {
            let signature = graph_compilation_signature(
                &graph,
                PtxReductionMode::DeterministicTree,
                PtxTarget::sm80(),
                crate::tile::MatMulPrecision::NativeHalf,
                MatMulSharedLayout::Auto,
                MatMulPipeline::Auto,
                (
                    PtxFusionPolicy::CostAware,
                    None,
                    1,
                    false,
                    ExperimentalKernelOptions {
                        batched_attention_bf16_tile_policy: policy,
                        ..ExperimentalKernelOptions::default()
                    },
                ),
            );
            assert!(seen.insert(signature));
        }
        assert_eq!(seen.len(), 4);
        assert_eq!(
            ExperimentalKernelOptions::default().batched_attention_bf16_tile_policy,
            crate::tile::BatchedAttentionBf16TilePolicy::Off
        );
    }

    #[test]
    fn batched_attention_policy_cache_toggle_uses_actual_regions_and_changed_feeds()
    -> anyhow::Result<()> {
        use crate::{
            ptx::BF16,
            tile::{BatchedAttentionBf16TilePolicy as Policy, BatchedAttentionTileCounts},
        };
        use half::bf16;
        if let Err(error) = CudaContext::new(0) {
            assert_ne!(
                std::env::var("TNSR_REQUIRE_CUDA").as_deref(),
                Ok("1"),
                "{error:#}"
            );
            return Ok(());
        }
        let graph: TensorGraph<bf16> = TensorExpr::input("a", vec![8, 8, 512, 64])
            .matmul(TensorExpr::input("b", vec![8, 8, 64, 512]))
            .into();
        let mut executor = PtxExecutor::<BF16>::try_new_for_dtype()?;
        executor.set_matmul_precision(crate::tile::MatMulPrecision::NativeHalf);
        executor.set_experimental_shared_memory_reuse(true);
        let mut expected: Vec<Vec<u16>> = Vec::new();
        for (iteration, policy) in [
            Policy::Off,
            Policy::TallScores,
            Policy::WideScores,
            Policy::TallScoresByGrid,
            Policy::Off,
        ]
        .into_iter()
        .enumerate()
        {
            executor.set_experimental_batched_attention_bf16_tile_policy(policy);
            if iteration > 0 {
                assert!(
                    executor
                        .execute_compiled(&graph, HashMap::new())
                        .unwrap_err()
                        .to_string()
                        .contains("No compiled graph signature")
                );
            }
            executor.compile(&graph)?;
            let counts = executor
                .execution_plan()
                .unwrap()
                .batched_attention_bf16_tile_counts();
            assert_eq!(
                counts,
                match policy {
                    Policy::Off => BatchedAttentionTileCounts::default(),
                    Policy::TallScores | Policy::TallScoresByGrid => BatchedAttentionTileCounts {
                        scores: 1,
                        tall: 1,
                        ..Default::default()
                    },
                    Policy::WideScores => BatchedAttentionTileCounts {
                        scores: 1,
                        wide: 1,
                        ..Default::default()
                    },
                }
            );
            if let Ok(directory) = std::env::var("TNSR_BATCHED_ATTENTION_AUDIT") {
                let directory = std::path::Path::new(&directory);
                std::fs::create_dir_all(directory)?;
                std::fs::write(
                    directory.join(format!("actual-{iteration}-{policy:?}.ptx")),
                    executor.module_source().unwrap(),
                )?;
            }
            let mut phases = Vec::new();
            for phase in 0..2 {
                let feeds = HashMap::from([
                    (
                        "a".into(),
                        (0..64 * 512 * 64)
                            .map(|i| bf16::from_f32(((i + phase * 13) % 17) as f32 / 64. - 0.125))
                            .collect(),
                    ),
                    (
                        "b".into(),
                        (0..64 * 64 * 512)
                            .map(|i| bf16::from_f32(((i + phase * 7) % 13) as f32 / 32. - 0.0625))
                            .collect(),
                    ),
                ]);
                let values = executor.execute_compiled(&graph, feeds)?;
                assert!(values.iter().all(|v| v.is_finite()));
                let bits: Vec<_> = values.into_iter().map(bf16::to_bits).collect();
                if iteration > 0 {
                    let reference = expected.get(phase).expect("baseline phase recorded");
                    assert_eq!(bits.len(), reference.len());
                    for (i, (&actual, &old)) in bits.iter().zip(reference).enumerate() {
                        assert_eq!(actual, old, "policy{policy:?} phase{phase} index{i}");
                    }
                }
                phases.push(bits);
            }
            if iteration == 0 {
                expected = phases;
            }
            for (left, right, nan) in [
                (1.0f32, f32::INFINITY, false),
                (1., f32::NEG_INFINITY, false),
                (0., f32::INFINITY, true),
                (-0., 1., false),
            ] {
                let output = executor.execute_compiled(
                    &graph,
                    HashMap::from([
                        ("a".into(), vec![bf16::from_f32(left); 64 * 512 * 64]),
                        ("b".into(), vec![bf16::from_f32(right); 64 * 64 * 512]),
                    ]),
                )?;
                for (i, value) in output.into_iter().enumerate() {
                    if nan {
                        assert!(value.is_nan(), "{policy:?} index{i}");
                    } else if right.is_infinite() {
                        assert_eq!(value.to_f32(), right, "{policy:?} index{i}");
                    } else {
                        assert_eq!(value.to_bits(), 0, "{policy:?} signedzero index{i}");
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn ordinary_four_warp_plan_admission_and_toggle_without_device() -> anyhow::Result<()> {
        let resources = crate::tile::MatMulResources {
            shared_bytes_per_sm: 100 * 1024,
            threads_per_sm: 1536,
            ..crate::tile::MatMulResources::default()
        };
        for (m, n, k, expected) in [
            (2048, 256, 64, 1),
            (2049, 256, 64, 0),
            (2048, 255, 64, 0),
            (2048, 256, 65, 0),
            (512, 512, 64, 0),
        ] {
            let graph: TensorGraph<half::bf16> = TensorExpr::input("a", vec![m, k])
                .matmul(TensorExpr::input("b", vec![k, n]))
                .into();
            let mut plan = PtxExecutionPlan::build(&graph)?;
            plan.schedule_matmuls(
                PtxTarget::sm80().matmul_capabilities(),
                crate::tile::MatMulPrecision::NativeHalf,
                resources,
                MatMulSharedLayout::Auto,
                MatMulPipeline::Auto,
                true,
                false,
            )?;
            plan.set_matmul_shared_memory_reuse(true);
            let baseline = plan.clone();
            plan.configure_ordinary_bf16_four_warp_block(false, resources);
            assert_eq!(plan, baseline);
            plan.configure_ordinary_bf16_four_warp_block(true, resources);
            assert_eq!(plan.ordinary_bf16_four_warp_count(), expected);
            if expected == 0 {
                assert_eq!(plan, baseline, "whole-kernel guard fallback");
            }
            plan.configure_ordinary_bf16_four_warp_block(true, resources);
            assert_eq!(plan.ordinary_bf16_four_warp_count(), expected);
            plan.configure_ordinary_bf16_four_warp_block(false, resources);
            assert_eq!(plan, baseline, "complete baseline restored");
        }
        Ok(())
    }

    #[test]
    fn ordinary_four_warp_cache_feeds_all_orientations_and_nonfinite_bits() -> anyhow::Result<()> {
        use crate::{ptx::BF16, tensor::TensorExpr};
        use half::bf16;
        if let Err(e) = CudaContext::new(0) {
            assert_ne!(
                std::env::var("TNSR_REQUIRE_CUDA").as_deref(),
                Ok("1"),
                "{e:#}"
            );
            return Ok(());
        }
        for (left, right) in [(false, false), (false, true), (true, false), (true, true)] {
            let a = TensorExpr::input("a", if left { vec![64, 2048] } else { vec![2048, 64] });
            let b = TensorExpr::input("b", if right { vec![256, 64] } else { vec![64, 256] });
            let graph: TensorGraph<bf16> = (if left { a.transpose() } else { a })
                .matmul(if right { b.transpose() } else { b })
                .into();
            let mut executor = PtxExecutor::<BF16>::try_new_for_dtype()?;
            executor.set_matmul_precision(crate::tile::MatMulPrecision::NativeHalf);
            executor.set_experimental_shared_memory_reuse(true);
            let mut expected: Vec<Vec<u16>> = Vec::new();
            let mut baseline_source = String::new();
            for (iteration, enabled) in [false, true, false].into_iter().enumerate() {
                executor.set_experimental_ordinary_bf16_four_warp_block(enabled);
                if iteration > 0 {
                    assert!(
                        executor
                            .execute_compiled(&graph, HashMap::new())
                            .unwrap_err()
                            .to_string()
                            .contains("No compiled graph signature")
                    );
                }
                executor.compile(&graph)?;
                if iteration == 0 {
                    baseline_source = executor.module_source().unwrap().to_string();
                }
                if iteration == 2 {
                    assert_eq!(
                        executor.module_source().unwrap(),
                        baseline_source.as_str(),
                        "default-Off whole module source restored"
                    );
                }
                assert_eq!(
                    executor
                        .execution_plan()
                        .unwrap()
                        .ordinary_bf16_four_warp_count(),
                    usize::from(enabled)
                );
                if let Ok(path) = std::env::var("TNSR_ORDINARY_FOUR_WARP_AUDIT") {
                    let path = std::path::Path::new(&path);
                    std::fs::create_dir_all(path)?;
                    std::fs::write(
                        path.join(format!("matrix-{left}-{right}-{iteration}-{enabled}.ptx")),
                        executor.module_source().unwrap(),
                    )?;
                }
                for revision in 0..3 {
                    let feed = |len: usize, operand: usize| {
                        (0..len)
                            .map(|i| {
                                if revision == 2 && i % 4093 == 0 {
                                    return bf16::from_bits(
                                        [
                                            0, 0x8000, 1, 0x8001, 0x7f7f, 0xff7f, 0x7f80, 0xff80,
                                            0x7fc1, 0x7f81,
                                        ][(i / 4093 + operand) % 10],
                                    );
                                }
                                bf16::from_f32(if revision == 0 {
                                    ((i * 7 + operand * 3) % 17) as f32 / 32. - 0.25
                                } else {
                                    ((i * 131 + operand * 19) % 257) as f32 / 97. - 1.23
                                })
                            })
                            .collect::<Vec<_>>()
                    };
                    let out = executor.execute_compiled(
                        &graph,
                        HashMap::from([
                            ("a".into(), feed(2048 * 64, 0)),
                            ("b".into(), feed(64 * 256, 1)),
                        ]),
                    )?;
                    let bits = out.into_iter().map(bf16::to_bits).collect::<Vec<_>>();
                    if iteration == 0 {
                        expected.push(bits);
                    } else {
                        for (i, (&x, &y)) in bits.iter().zip(&expected[revision]).enumerate() {
                            assert!(
                                (bf16::from_bits(x).is_nan() && bf16::from_bits(y).is_nan())
                                    || x == y,
                                "orientation{left}/{right} revision{revision} index{i}: {x:04x}/{y:04x}"
                            );
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn kernel_experiments_have_distinct_compilation_signatures() {
        let graph: TensorGraph<f32> = TensorExpr::input("a", vec![32, 64])
            .matmul(TensorExpr::input("b", vec![64, 32]))
            .into();
        let signature = |shared_memory_reuse,
                         online_weight_sharing,
                         narrow_bf16_tile,
                         bf16_eight_compute_warps,
                         medium_bf16_tile_policy,
                         column_reduction_policy,
                         large_bf16_tile_policy,
                         batched_attention_bf16_tile_policy,
                         ordinary_bf16_four_warp_block,
                         indexed_loss_cooperative_ordered_backward,
                         indexed_loss_cooperative_ordered_forward,
                         plain_bf16_output| {
            graph_compilation_signature(
                &graph,
                PtxReductionMode::Online,
                PtxTarget::sm80(),
                crate::tile::MatMulPrecision::NativeHalf,
                MatMulSharedLayout::Auto,
                MatMulPipeline::Auto,
                (
                    PtxFusionPolicy::CostAware,
                    None,
                    1,
                    false,
                    ExperimentalKernelOptions {
                        shared_memory_reuse,
                        online_weight_sharing,
                        narrow_bf16_tile,
                        bf16_eight_compute_warps,
                        medium_bf16_tile_policy,
                        column_reduction_policy,
                        large_bf16_tile_policy,
                        batched_attention_bf16_tile_policy,
                        ordinary_bf16_four_warp_block,
                        indexed_loss_cooperative_ordered_backward,
                        indexed_loss_cooperative_ordered_forward,
                        plain_bf16_output,
                    },
                ),
            )
        };
        let mut seen = std::collections::HashSet::new();
        for reuse in [false, true] {
            for weights in [false, true] {
                for narrow in [false, true] {
                    for eight in [false, true] {
                        for policy in [
                            crate::tile::MediumBf16TilePolicy::Off,
                            crate::tile::MediumBf16TilePolicy::SquareOnly,
                            crate::tile::MediumBf16TilePolicy::SquareThenTall,
                        ] {
                            for columns in [
                                crate::tile::ColumnReductionPolicy::Off,
                                crate::tile::ColumnReductionPolicy::Four,
                                crate::tile::ColumnReductionPolicy::Eight,
                            ] {
                                for large in [
                                    crate::tile::LargeBf16TilePolicy::Off,
                                    crate::tile::LargeBf16TilePolicy::Tall,
                                    crate::tile::LargeBf16TilePolicy::Wide,
                                    crate::tile::LargeBf16TilePolicy::TallThenWide,
                                    crate::tile::LargeBf16TilePolicy::Square128Streamed,
                                ] {
                                    for batched in [
                                        crate::tile::BatchedAttentionBf16TilePolicy::Off,
                                        crate::tile::BatchedAttentionBf16TilePolicy::TallScores,
                                        crate::tile::BatchedAttentionBf16TilePolicy::WideScores,
                                        crate::tile::BatchedAttentionBf16TilePolicy::TallScoresByGrid,
                                    ] {
                                        for four in [false, true] {
                                            for cooperative in [false,true] {
                                                for forward in [false,true] {for plain in [false,true] {assert!(seen.insert(signature(reuse, weights, narrow, eight, policy, columns, large, batched, four, cooperative, forward, plain)));}}
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(seen.len(), 46080);
        assert_eq!(
            ExperimentalKernelOptions::default(),
            ExperimentalKernelOptions {
                indexed_loss_cooperative_ordered_backward: false,
                indexed_loss_cooperative_ordered_forward: false,
                shared_memory_reuse: false,
                online_weight_sharing: false,
                narrow_bf16_tile: false,
                bf16_eight_compute_warps: false,
                ordinary_bf16_four_warp_block: false,
                plain_bf16_output: false,
                medium_bf16_tile_policy: crate::tile::MediumBf16TilePolicy::Off,
                column_reduction_policy: crate::tile::ColumnReductionPolicy::Off,
                large_bf16_tile_policy: crate::tile::LargeBf16TilePolicy::Off,
                batched_attention_bf16_tile_policy:
                    crate::tile::BatchedAttentionBf16TilePolicy::Off,
            }
        );
    }
}
