//! Fixed-shape execution of explicit f32/BF16 graphs, including graph autograd.
//!
//! The default planner retains each arithmetic storage rounding boundary. The
//! explicit online-pair numerical screen changes attention QK/P checkpoints.
//! View chains are
//! virtual and casts materialize directly into the destination dtype. This
//! planner fuses f32 regions while preserving every BF16 arithmetic boundary.
use super::{
    BF16, CudaDType, F32, PtxGraph, PtxTarget,
    linear::{
        Binding, Buffer, Captured,
        program::{Kernel, Program, ProgramKernelOptions, binding},
    },
};
mod attention_softmax;
mod attention_softmax_backward;
mod attention_softmax_warp;
mod attention_training_forward;
mod attention_training_online;
mod borrowed_gradient;
mod direct;
mod shadows;
pub(crate) use borrowed_gradient::CompletedRunStamp;
pub(crate) use shadows::{
    RetiredShadowBindings, ShadowBinding, ShadowDeclaration, shadow_declarations,
};
mod layer_norm_pair;
mod memory;
mod nn128x64_ldmatrix;
#[cfg(test)]
mod non_nn_calibration;
mod nt128_bias;
#[cfg(test)]
mod plain_bf16_output_tests;
mod planner;
mod qkv_grad_pack;
#[cfg(test)]
mod raw_gradient_census;
#[cfg(test)]
pub(crate) mod shadow_cast_census;
mod split_k;
mod split_k_small_grid;
mod swiglu_pair;
#[cfg(test)]
pub(crate) mod test_decoder;
mod transpose_output;

use crate::{
    graph::{NodeIndex, TensorGraph, TensorGraphNode, WithGrad, mixed::value_dtypes},
    tensor::ValueDType,
    tile::{DType, MatMulPrecision, TileDType, TileGraph, TileIRBuilder, VirtualTensor},
};
use anyhow::{Context, Result, ensure};
use cudarc::{
    driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, LaunchConfig, sys},
    nvrtc::Ptx,
};
use std::{
    any::Any,
    cell::RefCell,
    collections::HashMap,
    sync::{Arc, Mutex, Weak},
};

#[derive(Clone)]
struct Value {
    binding: Binding,
    view: VirtualTensor,
    source_shape: Vec<usize>,
    chain: Vec<TensorGraphNode<f32>>,
}

/// Opt-in mixed execution experiments. Existing constructors retain defaults.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MixedOptions {
    /// Cooperative CE backward statistics with the original ordered F32 sum.
    pub indexed_loss_cooperative_ordered_backward: bool,
    pub indexed_loss_cooperative_ordered_forward: bool,
    /// Exact four-checkpoint S512/D64 attention union; defaults off.
    pub attention_training_forward_fusion: bool,
    /// Numerical screen of a complete online attention pair. Changes QK/P
    /// checkpoints; retains all original quadratic allocation owners. Off by default.
    pub online_attention_pair_numerical_screen: bool,
    /// Exact closed packed epilogues for qualified online S512/D64/H8 scopes.
    /// Retains every old raw-gradient allocation; disabled by default.
    pub online_attention_packed_backward: bool,
    /// Carry online forward statistics in registers; complete Global pair fallback.
    /// Requires online_attention_pair_numerical_screen; disabled by default.
    pub online_attention_register_statistics: bool,
    #[cfg(test)]
    pub force_online_attention_register_statistics_rejection: bool,
    /// Exact closed F32 W512 LayerNorm pair, preflighted against the original plan.
    pub layer_norm_pair_fusion: bool,
    /// Reuse a 64x64 tile for proven aligned four-part BF16 Split-K partials.
    pub split_k_bf16_tile_policy: crate::tile::SplitKBf16TilePolicy,
    /// Constructor-local forced late rejection for complete-fallback tests.
    #[cfg(test)]
    pub(crate) split_k_rectangle_test_resource_rejection: bool,
    /// Numerically experimental four-part BF16 Split-K for small rank-two grids.
    /// F32 partials retain exactly one final ordered-sum BF16 rounding boundary.
    pub small_grid_split_k_bf16: bool,
    /// Recompute the closed mixed-storage stable SiLU primitive VJP, retaining
    /// every original BF16 rounding boundary. Disabled by default.
    pub swiglu_pair_fusion: bool,
    /// Pack closed BF16 Q/K/V Narrow gradients without padded global temporaries.
    /// Preserves the original pad casts and ordered BF16 gradient joins.
    pub qkv_grad_pack_fusion: bool,
    /// Bind resident f32 masters directly, retaining active immutable versions
    /// and caching at most two captured address variants. Disabled by default.
    pub direct_master_bindings: bool,
    /// Fuse a closed S512 compact attention VJP through its BF16 score gradient.
    pub attention_softmax_backward_fusion: bool,
    /// Fuse closed causal S512/D64 f32 normalization and its BF16 cast.
    /// Retains f32 probabilities for compact-softmax backward consumers.
    pub attention_softmax_fusion: bool,
    /// Narrow full physical storage contiguously, retaining bijective views.
    /// This changes GEMM operand layout eligibility and needs workload timing.
    pub physical_order_casts: bool,
    /// Compute closed rank-two BF16 MatMul outputs in transposed physical
    /// storage, retaining their original logical views and rounding boundaries.
    pub transposed_bf16_outputs: bool,
    /// Reuse dead GEMM operand shared storage for its epilogue in ordinary
    /// BF16 programs. Disabled by default; does not change schedule selection.
    pub shared_memory_reuse: bool,
    /// Use the bounded synchronous 32x32 tile for eligible N64 BF16 GEMMs.
    pub narrow_bf16_tile: bool,
    /// Repartition eligible ordinary BF16 64x64x32 tiles across eight warps.
    pub bf16_eight_compute_warps: bool,
    /// Keep ordinary BF16 64x64 compute ownership on four declared block warps.
    pub ordinary_bf16_four_warp_block: bool,
    /// Proven full-MN scalar BF16 output address simplification; defaults off.
    pub plain_bf16_output: bool,
    /// Qualified eight-warp 128x128 BF16 Linear with one bias Add, defaults off.
    pub fused_nt128_bias: bool,
    /// Exact two-shape coalesced ldmatrix NT128 replacement after final reuse.
    /// Requires fused_nt128_bias; every rejection retains the complete WMMA kernel.
    pub fused_nt128_bias_ldmatrix: bool,
    /// Exact B32 FFN dX collective-load128x64 plain BF16 kernel; defaults off.
    pub nn128x64_ldmatrix: bool,
    /// Alias one closed BF16 contiguous reshape before a bias Sum. Defaults off.
    pub contiguous_bias_reshape_aliases: bool,
    /// Bounded rank-two BF16 reuse tiles, before ordinary layout/pipeline policy.
    pub medium_bf16_tile_policy: crate::tile::MediumBf16TilePolicy,
    /// Larger ordinary BF16 rectangles; requires explicit shared arena reuse.
    pub large_bf16_tile_policy: crate::tile::LargeBf16TilePolicy,
    /// Exact S512/D64 batched attention rectangles; requires shared arena reuse.
    pub batched_attention_bf16_tile_policy: crate::tile::BatchedAttentionBf16TilePolicy,
    /// Group independent columns of eligible Block128 reductions.
    pub column_reduction_policy: crate::tile::ColumnReductionPolicy,
    /// Compact completed resident execution prefixes during construction to
    /// avoid retaining the full unpooled payload before final lifetime reuse.
    pub incremental_allocation_reuse: bool,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct MixedInfo {
    /// Actually installed complete three-kernel numerical-screen pairs.
    pub online_attention_pair_numerical_screen_pairs: usize,
    pub online_attention_register_statistics_pairs: usize,
    pub online_attention_global_statistics_pairs: usize,
    pub online_attention_register_statistics_fallback_pairs: usize,
    pub online_attention_register_statistics_rejections: Vec<String>,
    pub online_attention_packed_backward_regions: usize,
    pub online_attention_packed_backward_fallback_regions: usize,
    pub online_attention_packed_backward_pack_kernels_removed: usize,
    pub online_attention_packed_backward_rejections: Vec<String>,
    pub online_attention_pair_numerical_screen_async16_pairs: usize,
    pub online_attention_pair_numerical_screen_scalar_pairs: usize,
    pub online_attention_pair_numerical_screen_fallback_pairs: usize,
    pub online_attention_pair_numerical_screen_workspace_bytes: usize,
    /// Original output node IDs and strict admission reasons, not requested flags.
    pub online_attention_pair_numerical_screen_rejections: Vec<String>,
    pub attention_training_forward_regions: usize,
    pub attention_training_forward_fallback_regions: usize,
    pub attention_training_forward_materializations: usize,
    pub indexed_loss_cooperative_ordered_backward_programs: usize,
    pub indexed_loss_cooperative_ordered_forward_programs: usize,
    pub contiguous_bias_reshape_alias_steps: usize,
    pub contiguous_bias_reshape_fallback_steps: usize,
    pub contiguous_bias_reshape_avoided_bytes: usize,
    pub fused_nt128_bias_programs: usize,
    pub fused_nt128_bias_fallback_programs: usize,
    pub fused_nt128_bias_rejections: Vec<String>,
    /// Actually installed ldmatrix kernels, included in fused_nt128_bias_programs.
    pub fused_nt128_bias_ldmatrix_programs: usize,
    pub fused_nt128_bias_ldmatrix_fallback_programs: usize,
    pub fused_nt128_bias_ldmatrix_rejections: Vec<String>,
    pub fused_nt128_bias_ldmatrix_selections: Vec<String>,
    pub nn128x64_ldmatrix_programs: usize,
    pub nn128x64_ldmatrix_fallback_programs: usize,
    pub nn128x64_ldmatrix_rejections: Vec<String>,
    pub nn128x64_ldmatrix_selections: Vec<String>,
    pub plain_bf16_output_programs: usize,
    pub plain_bf16_output_fallback_programs: usize,
    /// Actually selected 64x64 Split-K partial kernels, excluding the reducer.
    pub split_k_square64_programs: usize,
    /// Requested Split-K programs retaining the complete baseline, including
    /// existing 64x64 and other out-of-scope schedule families.
    pub split_k_square64_fallback_programs: usize,
    /// Actually installed experimental rectangle partials, excluding reducers.
    pub split_k_rectangle128x64_programs: usize,
    pub split_k_rectangle128x64_fallback_programs: usize,
    pub split_k_rectangle128x64_rejections: Vec<String>,
    pub split_k_rectangle128x64_selections: Vec<String>,
    /// Installed small-grid split MatMuls, each with one separate reducer.
    pub small_grid_split_k_programs: usize,
    /// Shape-eligible small-grid requests retaining the complete ordinary path.
    pub small_grid_split_k_fallback_programs: usize,
    pub layer_norm_pair_forward_regions: usize,
    pub layer_norm_pair_backward_regions: usize,
    pub layer_norm_pair_fallback_regions: usize,
    pub swiglu_pair_forward_regions: usize,
    pub swiglu_pair_backward_regions: usize,
    pub qkv_grad_pack_regions: usize,
    /// Unique physical resident masters, excluded from the owned f32 payload.
    pub direct_master_parameters: usize,
    pub direct_master_bytes: usize,
    pub direct_capture_hits: usize,
    /// Includes the initial preparation capture, then newly captured address tuples.
    pub direct_capture_misses: usize,
    pub direct_capture_variants: usize,
    pub attention_softmax_backward_regions: usize,
    pub attention_softmax_regions: usize,
    pub physical_order_cast_materializations: usize,
    pub bf16_master_shadow_cast_aliases: usize,
    pub bf16_master_shadow_alias_bytes: usize,
    pub transposed_bf16_output_programs: usize,
    pub bf16_eight_compute_warp_programs: usize,
    /// Actually installed ordinary 128-thread MatMul kernels across programs.
    pub ordinary_bf16_four_warp_programs: usize,
    pub medium_bf16_square_programs: usize,
    pub medium_bf16_tall_programs: usize,
    pub large_bf16_tall_programs: usize,
    pub large_bf16_wide_programs: usize,
    pub large_bf16_square128_streamed_programs: usize,
    pub large_bf16_square128_streamed_fallback_programs: usize,
    pub batched_attention_score_programs: usize,
    pub batched_attention_query_width_programs: usize,
    pub batched_attention_width_query_programs: usize,
    pub batched_attention_tall_programs: usize,
    pub batched_attention_wide_programs: usize,
    pub column_reduction_four_regions: usize,
    pub column_reduction_eight_regions: usize,
    /// Private temporary allocations replaced by nonoverlapping live ranges.
    pub reused_allocations: usize,
    /// Retained tensor payload before resident lifetime reuse (not peak process memory).
    pub unpooled_bytes: usize,
    pub exact_index_bytes: usize,
    pub immutable_source_bytes: usize,
    pub kernels: usize,
    pub f32_bytes: usize,
    pub bf16_bytes: usize,
    pub replay: bool,
    /// Programs whose emitted PTX contains native BF16 MMA instructions.
    /// Active ordinary typed Program<D> islands; custom attention regions are
    /// counted separately and may each contain two logical MMA operations.
    pub bf16_tensor_core_programs: usize,
}

/// Owns all typed allocations and compiled functions for a single graph.
/// Uploads are explicit and excluded from `run` and `time_runs`.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct MixedTransfers {
    pub f32_uploads: usize,
    pub f32_bytes: usize,
    pub index_uploads: usize,
    pub index_bytes: usize,
    pub parameter_copies: usize,
    pub parameter_bytes: usize,
    pub upload_seconds: f64,
    pub parameter_refresh_seconds: f64,
}

struct Readback {
    buffer: CudaSlice<f32>,
    kernel: Kernel,
}

pub(crate) struct ResidentBindings {
    pub stream: Arc<CudaStream>,
    pub gate: Arc<Mutex<()>>,
    pub parameters: HashMap<NodeIndex, Arc<CudaSlice<f32>>>,
}

#[derive(Clone, Copy)]
enum SourceStorage {
    Owned(usize),
    External,
}

pub struct MixedExecutor {
    freshness: borrowed_gradient::Freshness,
    borrowed_prefixes: borrowed_gradient::PrefixCache,
    #[cfg(test)]
    shadow_cast_observations: Vec<shadow_cast_census::Observation>,
    indexed_loss_cooperative_ordered_backward: bool,
    indexed_loss_cooperative_ordered_forward: bool,
    layer_norm_preparation: Option<layer_norm_pair::rewrite::Preparation>,
    attention_training_preparation: Option<attention_training_forward::rewrite::Preparation>,
    online_attention_pair_preparation:
        Option<attention_training_online::runtime_screen::Preparation>,
    captured: Option<Captured>,
    stream: Arc<CudaStream>,
    buffers: Vec<Buffer>,
    programs: Vec<Box<dyn Any>>,
    kernels: Vec<Kernel>,
    values: HashMap<NodeIndex, Value>,
    sources: HashMap<NodeIndex, SourceStorage>,
    direct: Option<direct::DirectBindings>,
    source_nodes: HashMap<NodeIndex, TensorGraphNode<f32>>,
    bound_parameters: HashMap<NodeIndex, Weak<CudaSlice<f32>>>,
    gate: Arc<Mutex<()>>,
    index_graph: TensorGraph<f32>,
    index_plan: crate::graph::indices::IndexPlan,
    index_buffers: HashMap<NodeIndex, CudaSlice<u32>>,
    readbacks: RefCell<HashMap<NodeIndex, Readback>>,
    uploaded: bool,
    constants_initialized: bool,
    transfers: MixedTransfers,
    has_run: bool,
    info: MixedInfo,
    column_reduction_policy: crate::tile::ColumnReductionPolicy,
    split_k_bf16_tile_policy: crate::tile::SplitKBf16TilePolicy,
    #[cfg(test)]
    split_k_rectangle_test_resource_rejection: bool,
    split_k_shared_memory_reuse: bool,
    small_grid_split_k_bf16: bool,
    fused_nt128_bias: bool,
    fused_nt128_bias_ldmatrix: bool,
    nn128x64_ldmatrix: bool,
    nn128x64_pending: Vec<nn128x64_ldmatrix::Pending>,
    nn128x64_sources: Vec<(NodeIndex, String, String)>,
    nt128_ldmatrix_pending: Vec<nt128_bias::ldmatrix::Pending>,
    /// Actual late-selected source; retained Programs describe the WMMA fallback.
    nt128_ldmatrix_sources: Vec<(NodeIndex, String, String)>,
}
impl Drop for MixedExecutor {
    fn drop(&mut self) {
        let gate = self.gate.clone();
        let _guard = match gate.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        let _ = self.stream.synchronize();
        // Keep graph destruction, asynchronous frees and module release outside
        // every other owner's capture on the shared resident stream.
        self.captured = None;
        self.direct = None;
        self.readbacks.get_mut().clear();
        self.programs.clear();
        self.buffers.clear();
        self.index_buffers.clear();
        self.kernels.clear();
        self.source_nodes.clear();
        self.index_graph.graph.clear();
        let _ = self.stream.synchronize();
    }
}
impl MixedExecutor {
    /// Homogeneous comparator using the existing whole-graph fusion planner.
    /// Only materialized nodes are readable; intermediate fused values may be absent.
    pub fn new_fused_f32<G>(
        graph: &TensorGraph<f32, G>,
        precision: MatMulPrecision,
        replay: bool,
    ) -> Result<Self> {
        ensure!(
            value_dtypes(graph).is_empty(),
            "fused f32 comparator cannot contain casts"
        );
        Self::prepare(
            graph,
            precision,
            replay,
            true,
            false,
            None,
            MixedOptions::default(),
        )
    }

    pub fn new<G>(
        graph: &TensorGraph<f32, G>,
        precision: MatMulPrecision,
        replay: bool,
    ) -> Result<Self> {
        Self::new_with_options(graph, precision, replay, MixedOptions::default())
    }

    /// Compile with explicitly selected experiments. Most retain arithmetic
    /// storage boundaries; the online-pair numerical screen changes attention
    /// checkpoints and is separately qualified against complete training gates.
    pub fn new_with_options<G>(
        graph: &TensorGraph<f32, G>,
        precision: MatMulPrecision,
        replay: bool,
        options: MixedOptions,
    ) -> Result<Self> {
        Self::prepare(graph, precision, replay, false, true, None, options)
    }

    /// Diagnostic comparator retaining one arithmetic operation per program.
    pub fn new_unfused<G>(
        graph: &TensorGraph<f32, G>,
        precision: MatMulPrecision,
        replay: bool,
    ) -> Result<Self> {
        Self::prepare(
            graph,
            precision,
            replay,
            false,
            false,
            None,
            MixedOptions::default(),
        )
    }

    fn prepare<G>(
        graph: &TensorGraph<f32, G>,
        precision: MatMulPrecision,
        replay: bool,
        fused_f32: bool,
        fuse_f32_regions: bool,
        resident: Option<ResidentBindings>,
        options: MixedOptions,
    ) -> Result<Self> {
        Self::prepare_with_shadows(
            graph,
            precision,
            replay,
            fused_f32,
            fuse_f32_regions,
            resident,
            options,
            HashMap::new(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_with_shadows<G>(
        graph: &TensorGraph<f32, G>,
        precision: MatMulPrecision,
        replay: bool,
        fused_f32: bool,
        fuse_f32_regions: bool,
        resident: Option<ResidentBindings>,
        options: MixedOptions,
        shadow_bindings: HashMap<NodeIndex, ShadowBinding>,
    ) -> Result<Self> {
        ensure!(!graph.is_empty(), "mixed graph is empty");
        ensure!(
            !(options.online_attention_pair_numerical_screen
                && options.attention_training_forward_fusion),
            "online numerical screen and checkpoint-preserving attention forward fusion are mutually exclusive"
        );
        ensure!(
            !options.online_attention_packed_backward
                || (options.online_attention_pair_numerical_screen && options.qkv_grad_pack_fusion),
            "online_attention_packed_backward requires online attention pair and QKV gradient pack"
        );
        let reuse_intermediates = resident.is_some();
        ensure!(
            !options.direct_master_bindings || reuse_intermediates,
            "direct master bindings require a resident owner"
        );
        ensure!(
            !options.incremental_allocation_reuse || reuse_intermediates,
            "incremental allocation reuse requires resident bindings"
        );
        let (stream, gate, bound_parameters) = match resident {
            Some(bindings) => (bindings.stream, bindings.gate, bindings.parameters),
            None => (
                CudaContext::new(0)?.new_stream()?,
                Arc::new(Mutex::new(())),
                HashMap::new(),
            ),
        };
        ensure!(
            shadow_bindings.is_empty()
                || (options.direct_master_bindings && options.physical_order_casts),
            "BF16 shadows require direct masters and physical-order casts"
        );
        shadows::validate_declarations(graph, &stream, &bound_parameters, &shadow_bindings)?;
        let mut index_graph = TensorGraph::new();
        index_graph.graph = graph.graph.clone();
        let index_plan = crate::graph::indices::IndexPlan::new(graph)?;
        let dtypes = value_dtypes(graph);
        let layer_norm_preparation =
            if options.layer_norm_pair_fusion && fuse_f32_regions && !fused_f32 {
                Some(layer_norm_pair::rewrite::Preparation::new(graph)?)
            } else {
                None
            };
        let attention_training_preparation = options.attention_training_forward_fusion.then(|| {
            attention_training_forward::rewrite::Preparation::new(
                graph,
                options.attention_softmax_fusion && fuse_f32_regions && !fused_f32,
            )
        });
        ensure!(
            !options.fused_nt128_bias_ldmatrix || options.fused_nt128_bias,
            "fused_nt128_bias_ldmatrix requires fused_nt128_bias"
        );
        ensure!(
            !options.online_attention_register_statistics
                || options.online_attention_pair_numerical_screen,
            "online_attention_register_statistics requires online attention pair"
        );
        let online_attention_pair_preparation =
            options.online_attention_pair_numerical_screen.then(|| {
                attention_training_online::runtime_screen::Preparation::with_statistics(
                    graph,
                    options.online_attention_packed_backward,
                    options.online_attention_register_statistics,
                    #[cfg(test)]
                    options.force_online_attention_register_statistics_rejection,
                )
            });
        let mut result = Self {
            freshness: borrowed_gradient::Freshness::default(),
            borrowed_prefixes: borrowed_gradient::PrefixCache::default(),
            #[cfg(test)]
            shadow_cast_observations: Vec::new(),
            indexed_loss_cooperative_ordered_backward: options
                .indexed_loss_cooperative_ordered_backward,
            indexed_loss_cooperative_ordered_forward: options
                .indexed_loss_cooperative_ordered_forward,
            layer_norm_preparation,
            attention_training_preparation,
            online_attention_pair_preparation,
            captured: None,
            stream,
            buffers: Vec::new(),
            programs: Vec::new(),
            kernels: Vec::new(),
            values: HashMap::new(),
            sources: HashMap::new(),
            direct: None,
            source_nodes: HashMap::new(),
            bound_parameters: bound_parameters.keys().map(|n| (*n, Weak::new())).collect(),
            gate,
            index_graph,
            index_plan,
            index_buffers: HashMap::new(),
            readbacks: RefCell::new(HashMap::new()),
            uploaded: false,
            constants_initialized: false,
            transfers: MixedTransfers::default(),
            has_run: false,
            column_reduction_policy: options.column_reduction_policy,
            split_k_bf16_tile_policy: options.split_k_bf16_tile_policy,
            #[cfg(test)]
            split_k_rectangle_test_resource_rejection: options
                .split_k_rectangle_test_resource_rejection,
            split_k_shared_memory_reuse: options.shared_memory_reuse,
            small_grid_split_k_bf16: options.small_grid_split_k_bf16,
            fused_nt128_bias: options.fused_nt128_bias,
            fused_nt128_bias_ldmatrix: options.fused_nt128_bias_ldmatrix,
            nn128x64_ldmatrix: options.nn128x64_ldmatrix,
            nn128x64_pending: Vec::new(),
            nn128x64_sources: Vec::new(),
            nt128_ldmatrix_pending: Vec::new(),
            nt128_ldmatrix_sources: Vec::new(),
            info: MixedInfo {
                online_attention_pair_numerical_screen_pairs: 0,
                online_attention_register_statistics_pairs: 0,
                online_attention_global_statistics_pairs: 0,
                online_attention_register_statistics_fallback_pairs: 0,
                online_attention_register_statistics_rejections: Vec::new(),
                online_attention_packed_backward_regions: 0,
                online_attention_packed_backward_fallback_regions: 0,
                online_attention_packed_backward_pack_kernels_removed: 0,
                online_attention_packed_backward_rejections: Vec::new(),
                online_attention_pair_numerical_screen_async16_pairs: 0,
                online_attention_pair_numerical_screen_scalar_pairs: 0,
                online_attention_pair_numerical_screen_fallback_pairs: 0,
                online_attention_pair_numerical_screen_workspace_bytes: 0,
                online_attention_pair_numerical_screen_rejections: Vec::new(),
                attention_training_forward_regions: 0,
                attention_training_forward_fallback_regions: 0,
                attention_training_forward_materializations: 0,
                indexed_loss_cooperative_ordered_backward_programs: 0,
                indexed_loss_cooperative_ordered_forward_programs: 0,
                contiguous_bias_reshape_alias_steps: 0,
                contiguous_bias_reshape_fallback_steps: 0,
                contiguous_bias_reshape_avoided_bytes: 0,
                fused_nt128_bias_programs: 0,
                fused_nt128_bias_fallback_programs: 0,
                fused_nt128_bias_rejections: Vec::new(),
                fused_nt128_bias_ldmatrix_programs: 0,
                fused_nt128_bias_ldmatrix_fallback_programs: 0,
                fused_nt128_bias_ldmatrix_rejections: Vec::new(),
                fused_nt128_bias_ldmatrix_selections: Vec::new(),
                nn128x64_ldmatrix_programs: 0,
                nn128x64_ldmatrix_fallback_programs: 0,
                nn128x64_ldmatrix_rejections: Vec::new(),
                nn128x64_ldmatrix_selections: Vec::new(),
                plain_bf16_output_programs: 0,
                plain_bf16_output_fallback_programs: 0,
                split_k_square64_programs: 0,
                split_k_square64_fallback_programs: 0,
                split_k_rectangle128x64_programs: 0,
                split_k_rectangle128x64_fallback_programs: 0,
                split_k_rectangle128x64_rejections: Vec::new(),
                split_k_rectangle128x64_selections: Vec::new(),
                small_grid_split_k_programs: 0,
                small_grid_split_k_fallback_programs: 0,
                layer_norm_pair_forward_regions: 0,
                layer_norm_pair_backward_regions: 0,
                layer_norm_pair_fallback_regions: 0,
                swiglu_pair_forward_regions: 0,
                swiglu_pair_backward_regions: 0,
                qkv_grad_pack_regions: 0,
                direct_master_parameters: 0,
                direct_master_bytes: 0,
                direct_capture_hits: 0,
                direct_capture_misses: 0,
                direct_capture_variants: 0,
                attention_softmax_backward_regions: 0,
                attention_softmax_regions: 0,
                physical_order_cast_materializations: 0,
                bf16_master_shadow_cast_aliases: 0,
                bf16_master_shadow_alias_bytes: 0,
                transposed_bf16_output_programs: 0,
                bf16_eight_compute_warp_programs: 0,
                ordinary_bf16_four_warp_programs: 0,
                medium_bf16_square_programs: 0,
                medium_bf16_tall_programs: 0,
                large_bf16_tall_programs: 0,
                large_bf16_wide_programs: 0,
                large_bf16_square128_streamed_programs: 0,
                large_bf16_square128_streamed_fallback_programs: 0,
                batched_attention_score_programs: 0,
                batched_attention_query_width_programs: 0,
                batched_attention_width_query_programs: 0,
                batched_attention_tall_programs: 0,
                batched_attention_wide_programs: 0,
                column_reduction_four_regions: 0,
                column_reduction_eight_regions: 0,
                reused_allocations: 0,
                unpooled_bytes: 0,
                exact_index_bytes: 0,
                immutable_source_bytes: 0,
                kernels: 0,
                f32_bytes: 0,
                bf16_bytes: 0,
                replay,
                bf16_tensor_core_programs: 0,
            },
        };
        // Declare the guard after the owner so errors unlock before its Drop.
        // Allocations and uploads also belong outside another owner's capture.
        let preparation_gate = result.gate.clone();
        let _preparation_guard = preparation_gate
            .lock()
            .map_err(|_| anyhow::anyhow!("mixed stream lock poisoned"))?;
        let mut f32_region = Vec::new();
        let topo = graph.toposort();
        // Recognize before allocating any normalization interiors. The retained
        // f32 P is emitted at its original node, together with the later BF16
        // cast, so either forward or backward topological fanout is safe.
        let attention_regions =
            if options.attention_softmax_fusion && fuse_f32_regions && !fused_f32 {
                attention_softmax::recognize(graph)
            } else {
                Vec::new()
            };
        let attention_backward_regions =
            if options.attention_softmax_backward_fusion && fuse_f32_regions && !fused_f32 {
                attention_softmax_backward::recognize(graph)
            } else {
                Vec::new()
            };
        let qkv_pack_regions = if options.qkv_grad_pack_fusion && fuse_f32_regions && !fused_f32 {
            qkv_grad_pack::recognize(graph)
        } else {
            Vec::new()
        };
        let swiglu_regions = if options.swiglu_pair_fusion && fuse_f32_regions && !fused_f32 {
            swiglu_pair::recognize(graph)?
                .into_iter()
                .filter(|r| swiglu_pair::planned_contiguous(r, graph, &dtypes, options))
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let swiglu_forward_at: HashMap<_, _> =
            swiglu_regions.iter().map(|r| (r.forward_emit, r)).collect();
        let swiglu_backward_at: HashMap<_, _> = swiglu_regions
            .iter()
            .map(|r| (r.backward_emit, r))
            .collect();
        let qkv_pack_at: HashMap<_, _> = qkv_pack_regions.iter().map(|r| (r.output, r)).collect();
        let attention_backward_at: HashMap<_, _> = attention_backward_regions
            .iter()
            .map(|r| (r.output, r))
            .collect();
        let attention_at: HashMap<_, _> = attention_regions.iter().map(|r| (r.output, r)).collect();
        let mut attention_skipped: std::collections::HashSet<_> = attention_regions
            .iter()
            .flat_map(|r| r.members.iter().copied().filter(move |n| *n != r.output))
            .chain(
                attention_backward_regions
                    .iter()
                    .flat_map(|r| r.members.iter().copied().filter(move |n| *n != r.output)),
            )
            .chain(
                qkv_pack_regions
                    .iter()
                    .flat_map(|r| r.members.iter().copied().filter(move |n| *n != r.output)),
            )
            .collect();
        attention_skipped.extend(swiglu_regions.iter().flat_map(|r| {
            r.members
                .iter()
                .copied()
                .filter(move |n| *n != r.forward_emit && *n != r.backward_emit)
        }));
        let mut last_consumer = HashMap::new();
        if options.incremental_allocation_reuse {
            for (ordinal, &node) in topo.iter().enumerate() {
                for input in graph.inputs(node) {
                    last_consumer.insert(input, ordinal);
                }
            }
        }
        // The original promotion is now skipped, but QK must remain live until
        // the fused probability node. This also keeps its binding available
        // across incremental compaction before this kernel is constructed.
        if options.incremental_allocation_reuse {
            let ordinals: HashMap<_, _> = topo.iter().enumerate().map(|(i, &n)| (n, i)).collect();
            for region in &attention_regions {
                for input in [region.qk, region.mask] {
                    last_consumer
                        .entry(input)
                        .and_modify(|last| *last = (*last).max(ordinals[&region.output]))
                        .or_insert(ordinals[&region.output]);
                }
            }
            for region in &attention_backward_regions {
                for input in [region.probability, region.incoming] {
                    last_consumer
                        .entry(input)
                        .and_modify(|last| *last = (*last).max(ordinals[&region.output]))
                        .or_insert(ordinals[&region.output]);
                }
            }
        }
        if options.incremental_allocation_reuse {
            let ordinals: HashMap<_, _> = topo.iter().enumerate().map(|(i, &n)| (n, i)).collect();
            for region in &qkv_pack_regions {
                for input in &region.inputs {
                    last_consumer
                        .entry(input.source)
                        .and_modify(|last| *last = (*last).max(ordinals[&region.output]))
                        .or_insert(ordinals[&region.output]);
                }
            }
        }
        if options.incremental_allocation_reuse {
            let ordinals: HashMap<_, _> = topo.iter().enumerate().map(|(i, &n)| (n, i)).collect();
            for region in &swiglu_regions {
                for &input in region.inputs.iter().chain(region.storage_roots.iter()) {
                    last_consumer
                        .entry(input)
                        .and_modify(|last| *last = (*last).max(ordinals[&region.backward_emit]))
                        .or_insert(ordinals[&region.backward_emit]);
                }
            }
        }
        if let Some(preparation) = &result.layer_norm_preparation {
            let positions: HashMap<_, _> = topo.iter().enumerate().map(|(i, &n)| (n, i)).collect();
            preparation.extend_last_consumers(&positions, &mut last_consumer)?;
        }
        if let Some(preparation) = &result.attention_training_preparation {
            let positions: HashMap<_, _> = topo.iter().enumerate().map(|(i, &n)| (n, i)).collect();
            preparation.extend_last_consumers(&positions, &mut last_consumer)?;
        }
        if let Some(preparation) = &result.online_attention_pair_preparation {
            preparation.extend_last_consumers(&mut last_consumer);
        }
        let mut previously_retained_bytes = 0;
        let mut direct_master_addresses = std::collections::HashSet::new();
        for (ordinal, node) in topo.into_iter().enumerate() {
            if attention_skipped.contains(&node) {
                continue;
            }
            if let Some(region) = swiglu_forward_at.get(&node) {
                result.f32_region(graph, &f32_region, precision)?;
                f32_region.clear();
                result.swiglu_pair(region, false)?;
                continue;
            }
            if let Some(region) = swiglu_backward_at.get(&node) {
                result.f32_region(graph, &f32_region, precision)?;
                f32_region.clear();
                result.swiglu_pair(region, true)?;
                continue;
            }
            if let Some(region) = qkv_pack_at.get(&node) {
                result.f32_region(graph, &f32_region, precision)?;
                f32_region.clear();
                let first = result.kernels.len();
                result.qkv_grad_pack(region)?;
                if let Some(preparation) = &mut result.online_attention_pair_preparation {
                    let output = result
                        .values
                        .get(&region.output)
                        .context("QKV pack output absent after installation")?;
                    preparation.record_emitted(
                        region.output,
                        &result.kernels[first..],
                        output,
                        &result.values,
                    );
                }
                continue;
            }
            if let Some(region) = attention_backward_at.get(&node) {
                result.f32_region(graph, &f32_region, precision)?;
                f32_region.clear();
                result.attention_softmax_backward(region)?;
                continue;
            }
            if let Some(region) = attention_at.get(&node) {
                result.f32_region(graph, &f32_region, precision)?;
                f32_region.clear();
                result.attention_softmax(region)?;
                continue;
            }
            if fused_f32
                && !matches!(
                    graph[node],
                    TensorGraphNode::Input { .. }
                        | TensorGraphNode::Parameter { .. }
                        | TensorGraphNode::Constant { .. }
                )
            {
                continue;
            }
            if let TensorGraphNode::IndexSource { .. } = &graph[node] {
                let shape = graph[node].shape().clone();
                let len = shape
                    .iter()
                    .try_fold(1_usize, |n, d| n.checked_mul(*d))
                    .context("integer extent overflow")?;
                let buffer = result.stream.alloc_zeros::<u32>(len)?;
                let binding = Binding {
                    pointer: buffer.device_ptr(&result.stream).0,
                    len,
                    dtype: DType::U32,
                };
                result.values.insert(
                    node,
                    Value {
                        binding,
                        view: VirtualTensor::identity(node, shape.clone(), DType::U32),
                        source_shape: shape,
                        chain: Vec::new(),
                    },
                );
                result.info.exact_index_bytes += len * 4;
                result.index_buffers.insert(node, buffer);
                continue;
            }
            let dtype = match dtypes.get(&node).copied().unwrap_or(ValueDType::F32) {
                ValueDType::F32 => DType::F32,
                ValueDType::BF16 => DType::BF16,
            };
            let shape = graph[node].shape().clone();
            let len = shape
                .iter()
                .try_fold(1_usize, |n, d| n.checked_mul(*d))
                .context("mixed extent overflow")?;
            ensure!(
                len > 0 && u32::try_from(len).is_ok(),
                "mixed extent must be positive and fit a CUDA launch"
            );
            let inputs = graph.inputs(node);
            let source = matches!(
                graph[node],
                TensorGraphNode::Input { .. }
                    | TensorGraphNode::Constant { .. }
                    | TensorGraphNode::Parameter { .. }
            );
            let promotion = matches!(
                graph[node],
                TensorGraphNode::Cast {
                    dtype: ValueDType::F32,
                    ..
                }
            ) && dtypes.get(&inputs[0]) == Some(&ValueDType::BF16);
            if fuse_f32_regions && !source && !promotion && dtype == DType::F32 {
                f32_region.push(node);
                continue;
            }
            if fuse_f32_regions && !source && !promotion {
                result.f32_region(graph, &f32_region, precision)?;
                f32_region.clear();
            }
            match &graph[node] {
                TensorGraphNode::Input { .. }
                | TensorGraphNode::Parameter { .. }
                | TensorGraphNode::Constant { .. } => {
                    let (storage, value) = if options.direct_master_bindings
                        && let Some(parameter) = bound_parameters.get(&node)
                    {
                        ensure!(
                            matches!(graph[node], TensorGraphNode::Parameter { .. })
                                && parameter.len() == len
                                && Arc::ptr_eq(parameter.context(), result.stream.context()),
                            "invalid direct resident parameter source"
                        );
                        let binding = Binding {
                            pointer: parameter.device_ptr(&result.stream).0,
                            len,
                            dtype: DType::F32,
                        };
                        if direct_master_addresses.insert(binding.pointer) {
                            result.info.direct_master_parameters += 1;
                            result.info.direct_master_bytes += len * 4;
                        }
                        (
                            SourceStorage::External,
                            Value {
                                binding,
                                view: VirtualTensor::identity(node, shape.clone(), DType::F32),
                                source_shape: shape,
                                chain: Vec::new(),
                            },
                        )
                    } else {
                        let (index, value) = result.alloc(node, shape, DType::F32)?;
                        (SourceStorage::Owned(index), value)
                    };
                    result.sources.insert(node, storage);
                    result.source_nodes.insert(node, graph[node].clone());
                    result.values.insert(node, value);
                }
                TensorGraphNode::Cast { .. } => {
                    let source = result
                        .values
                        .get(&inputs[0])
                        .context("mixed cast source missing")?
                        .clone();
                    if let Some(shadow) = shadow_bindings.get(&node) {
                        ensure!(dtype == DType::BF16, "shadow target is not BF16");
                        let alias = shadows::cast_alias(
                            graph,
                            node,
                            &source,
                            shadow,
                            &result.stream,
                            &bound_parameters,
                        )?;
                        result.info.bf16_master_shadow_cast_aliases += 1;
                        result.info.bf16_master_shadow_alias_bytes += alias.binding.len * 2;
                        result.values.insert(node, alias);
                    } else if source.binding.dtype == dtype
                        || (dtype == DType::F32 && source.binding.dtype == DType::BF16)
                    {
                        result.values.insert(node, source);
                    } else if let Some(physical) = options
                        .physical_order_casts
                        .then(|| physical_order_cast_source(&source, dtype))
                        .flatten()
                    {
                        let (_, target) = result.alloc(node, source.source_shape.clone(), dtype)?;
                        let mut kernel = materialize(&result.stream, &physical, target.binding)?;
                        kernel
                            .label
                            .push_str(&format!(" physical-order logical={:?}", source.view.shape));
                        #[cfg(test)]
                        result
                            .shadow_cast_observations
                            .push(shadow_cast_census::observe(
                                graph,
                                node,
                                &source,
                                &physical_order_cast_result(&source, node, target.binding),
                                result.kernels.len(),
                                &kernel.label,
                            )?);
                        result.kernels.push(kernel);
                        result.info.physical_order_cast_materializations += 1;
                        result.values.insert(
                            node,
                            physical_order_cast_result(&source, node, target.binding),
                        );
                    } else {
                        let (_, target) = result.alloc(node, shape, dtype)?;
                        result
                            .kernels
                            .push(materialize(&result.stream, &source, target.binding)?);
                        result.values.insert(node, target);
                    }
                }
                n if is_view(n) => {
                    let mut value = result
                        .values
                        .get(&inputs[0])
                        .context("mixed view source missing")?
                        .clone();
                    value.view = match n {
                        TensorGraphNode::Reshape { .. } | TensorGraphNode::Flatten { .. } => {
                            value.view.reshape(shape)?
                        }
                        TensorGraphNode::Permute { axes, .. } => value.view.permute(axes)?,
                        TensorGraphNode::Transpose { .. } => {
                            let mut axes: Vec<usize> = (0..shape.len()).collect();
                            ensure!(axes.len() >= 2, "mixed transpose requires rank >= 2");
                            let end = axes.len();
                            axes.swap(end - 2, end - 1);
                            value.view.permute(&axes)?
                        }
                        TensorGraphNode::BroadcastAxis { axis, .. } => {
                            value.view.broadcast_axis(*axis, shape[*axis])?
                        }
                        _ => unreachable!(),
                    };
                    value.chain.push(n.clone());
                    result.values.insert(node, value);
                }
                n => {
                    let sources = inputs
                        .iter()
                        .map(|input| {
                            result
                                .values
                                .get(input)
                                .cloned()
                                .context("mixed operand missing")
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let mut operands = Vec::new();
                    for source in sources {
                        if source.binding.dtype == dtype || source.binding.dtype == DType::U32 {
                            operands.push(source);
                        } else {
                            let (_, target) =
                                result.alloc(node, source.view.shape.clone(), dtype)?;
                            result.kernels.push(materialize(
                                &result.stream,
                                &source,
                                target.binding,
                            )?);
                            operands.push(target);
                        }
                    }
                    let output = if dtype == DType::BF16 {
                        let kernel_options = ProgramKernelOptions {
                            indexed_loss_cooperative_ordered_backward: options
                                .indexed_loss_cooperative_ordered_backward,
                            indexed_loss_cooperative_ordered_forward: options
                                .indexed_loss_cooperative_ordered_forward,
                            shared_memory_reuse: options.shared_memory_reuse,
                            narrow_bf16_tile: options.narrow_bf16_tile,
                            bf16_eight_compute_warps: options.bf16_eight_compute_warps,
                            ordinary_bf16_four_warp_block: options.ordinary_bf16_four_warp_block,
                            plain_bf16_output: options.plain_bf16_output,
                            contiguous_bias_reshape_aliases: options
                                .contiguous_bias_reshape_aliases,
                            medium_bf16_tile_policy: options.medium_bf16_tile_policy,
                            large_bf16_tile_policy: options.large_bf16_tile_policy,
                            batched_attention_bf16_tile_policy: options
                                .batched_attention_bf16_tile_policy,
                            column_reduction_policy: options.column_reduction_policy,
                        };
                        if options.transposed_bf16_outputs
                            && precision != MatMulPrecision::StrictF32
                            && transpose_output::eligible(graph, node, &dtypes)
                            && PtxTarget::from_context(result.stream.context())?
                                .matmul_capabilities()
                                .bf16
                        {
                            result.transposed_bf16_output(
                                node,
                                n,
                                &operands,
                                precision,
                                kernel_options,
                            )?
                        } else {
                            result.operation::<BF16>(
                                node,
                                n,
                                &operands,
                                precision,
                                kernel_options,
                            )?
                        }
                    } else {
                        result.operation::<F32>(
                            node,
                            n,
                            &operands,
                            precision,
                            ProgramKernelOptions {
                                indexed_loss_cooperative_ordered_backward: options
                                    .indexed_loss_cooperative_ordered_backward,
                                indexed_loss_cooperative_ordered_forward: options
                                    .indexed_loss_cooperative_ordered_forward,
                                column_reduction_policy: options.column_reduction_policy,
                                ..ProgramKernelOptions::default()
                            },
                        )?
                    };
                    result.values.insert(node, output);
                }
            }
            // A visited F32 consumer may still be queued rather than emitted.
            // Only compact at completed boundaries with no pending region, and
            // after local operand/packed temporaries have entered emitted kernels.
            if options.incremental_allocation_reuse && f32_region.is_empty() {
                let future = last_consumer
                    .iter()
                    .filter_map(|(&node, &last)| (last > ordinal).then_some(node))
                    .collect();
                previously_retained_bytes =
                    result.reuse_intermediates(graph, &future, previously_retained_bytes)?;
            }
        }
        result.f32_region(graph, &f32_region, precision)?;
        if fused_f32 {
            let mut plain = TensorGraph::new();
            plain.graph = graph.graph.clone();
            let bindings = result
                .sources
                .keys()
                .map(|node| (*node, result.values[node].binding))
                .collect();
            let program = Program::<F32>::new_bound_with_kernel_options(
                &result.stream,
                plain,
                bindings,
                None,
                precision,
                ProgramKernelOptions {
                    indexed_loss_cooperative_ordered_backward: options
                        .indexed_loss_cooperative_ordered_backward,
                    indexed_loss_cooperative_ordered_forward: options
                        .indexed_loss_cooperative_ordered_forward,
                    column_reduction_policy: options.column_reduction_policy,
                    ..ProgramKernelOptions::default()
                },
            )?;
            result
                .info
                .indexed_loss_cooperative_ordered_backward_programs +=
                program.indexed_loss_cooperative_ordered_backward_programs;
            result.info.column_reduction_four_regions += program.column_reduction_counts.0;
            result
                .info
                .indexed_loss_cooperative_ordered_forward_programs +=
                program.indexed_loss_cooperative_ordered_forward_programs;
            result.info.column_reduction_eight_regions += program.column_reduction_counts.1;
            result.info.f32_bytes += program.bytes();
            result.kernels.extend(program.kernels.clone());
            for (&node, &binding) in &program.values {
                if let Some(view) = program.views.get(&node) {
                    result.values.insert(
                        node,
                        Value {
                            binding: *program
                                .values
                                .get(&view.source)
                                .context("fused view source missing")?,
                            view: view.clone(),
                            source_shape: graph[view.source].shape().clone(),
                            chain: Vec::new(),
                        },
                    );
                } else {
                    result.values.insert(
                        node,
                        Value {
                            binding,
                            view: VirtualTensor::identity(
                                node,
                                graph[node].shape().clone(),
                                DType::F32,
                            ),
                            source_shape: graph[node].shape().clone(),
                            chain: Vec::new(),
                        },
                    );
                }
            }
            result.programs.push(Box::new(program));
        }
        result.info.immutable_source_bytes = result
            .source_nodes
            .values()
            .filter_map(|n| {
                if let TensorGraphNode::Constant { data, .. } = n {
                    Some(data.len() * 4)
                } else {
                    None
                }
            })
            .sum();
        if let Some(preparation) = &mut result.attention_training_preparation {
            preparation.capture(&result.values, &result.kernels);
        }
        if let Some(preparation) = &mut result.online_attention_pair_preparation {
            preparation.capture_before_reuse(&result.values, &result.kernels);
        }
        if reuse_intermediates {
            result.reuse_intermediates(graph, &Default::default(), previously_retained_bytes)?;
        } else {
            result.info.unpooled_bytes = result.info.f32_bytes + result.info.bf16_bytes;
        }
        if let Some(preparation) = result.layer_norm_preparation.take() {
            preparation.apply(&mut result)?;
            if reuse_intermediates {
                let retained = result.info.f32_bytes + result.info.bf16_bytes;
                result.reuse_intermediates(graph, &Default::default(), retained)?;
            }
        }
        if let Some(preparation) = result.attention_training_preparation.take() {
            let accepted = preparation.apply(&mut result)?;
            if reuse_intermediates && accepted > 0 {
                let retained = result.info.f32_bytes + result.info.bf16_bytes;
                result.reuse_intermediates(graph, &Default::default(), retained)?;
            }
        }
        // Terminal numerical rewrite: keep every old Program/allocation owner,
        // and do not compact after installing the new reads and gradient writes.
        if let Some(preparation) = result.online_attention_pair_preparation.take() {
            let report = preparation.apply(&mut result)?;
            result.info.online_attention_pair_numerical_screen_pairs = report.selected_pairs;
            result.info.online_attention_register_statistics_pairs =
                report.register_statistics_pairs;
            result.info.online_attention_global_statistics_pairs = report.global_statistics_pairs;
            result
                .info
                .online_attention_register_statistics_fallback_pairs =
                report.register_statistics_fallbacks.len();
            result.info.online_attention_register_statistics_rejections = report
                .register_statistics_fallbacks
                .iter()
                .map(|f| format!("output {}: {}", f.output.index(), f.reason))
                .collect();
            result.info.online_attention_packed_backward_regions = report.packed_outputs.len();
            result
                .info
                .online_attention_packed_backward_fallback_regions = report.packed_fallbacks.len();
            result
                .info
                .online_attention_packed_backward_pack_kernels_removed =
                report.packed_pack_kernels_removed;
            result.info.online_attention_packed_backward_rejections = report
                .packed_fallbacks
                .iter()
                .map(|r| format!("output {}: {}", r.output.index(), r.reason))
                .collect();
            result
                .info
                .online_attention_pair_numerical_screen_async16_pairs = report.async16_pairs;
            result
                .info
                .online_attention_pair_numerical_screen_scalar_pairs = report.scalar_pairs;
            result
                .info
                .online_attention_pair_numerical_screen_fallback_pairs = report.fallbacks.len();
            result
                .info
                .online_attention_pair_numerical_screen_workspace_bytes = report.workspace_bytes;
            result
                .info
                .online_attention_pair_numerical_screen_rejections = report
                .fallbacks
                .iter()
                .map(|r| format!("output {}: {}", r.output.index(), r.reason))
                .collect();
        }
        // Exact footprint replacement only, after the last allocator/online pass
        // and before direct capture keys or constructor warmup can observe it.
        nt128_bias::ldmatrix::apply(&mut result);
        nn128x64_ldmatrix::apply(&mut result);
        result.info.kernels = result.kernels.len();
        ensure!(
            result.info.bf16_master_shadow_cast_aliases == shadow_bindings.len(),
            "not every declared BF16 shadow cast was imported"
        );
        if options.direct_master_bindings {
            let private = if shadow_bindings.is_empty() {
                Vec::new()
            } else {
                shadows::private_storage(&result)
            };
            result.direct = Some(direct::DirectBindings::new_with_shadows(
                &result.stream,
                &bound_parameters,
                &result.kernels,
                &result.values,
                replay,
                &shadow_bindings,
                &private,
            )?);
        }
        result.upload_impl(HashMap::new(), HashMap::new(), true)?;
        result.refresh_parameters_impl_with_shadows(&bound_parameters, &shadow_bindings, true)?;
        // Constructors may have required inputs. Zero-initialized sources are
        // safe for warmup; no user-visible output is available until upload/run.
        for k in &result.kernels {
            k.run(&result.stream)?;
        }
        result.synchronize()?;
        if replay && !result.kernels.is_empty() {
            result
                .stream
                .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
            let launched = result
                .kernels
                .iter()
                .try_for_each(|k| k.run(&result.stream));
            let ended = result.stream.end_capture(
                sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            );
            launched?;
            result.captured = Some(Captured {
                graph: Some(ended?.context("empty mixed capture")?),
                stream: result.stream.clone(),
            });
        } else {
            result.info.replay = false;
        }
        result.update_direct_info();
        result.uploaded = false;
        result.freshness.dirty();
        Ok(result)
    }

    fn alloc(
        &mut self,
        node: NodeIndex,
        shape: Vec<usize>,
        dtype: DType,
    ) -> Result<(usize, Value)> {
        let len = shape.iter().product::<usize>();
        let buffer = match dtype {
            DType::F32 => {
                self.info.f32_bytes += len * 4;
                Buffer::F32(self.stream.alloc_zeros(len)?)
            }
            DType::BF16 => {
                self.info.bf16_bytes += len * 2;
                Buffer::BF16(self.stream.alloc_zeros(len)?)
            }
            _ => anyhow::bail!("unsupported mixed storage"),
        };
        let value = Value {
            binding: binding(&buffer, &self.stream),
            view: VirtualTensor::identity(node, shape.clone(), dtype),
            source_shape: shape,
            chain: Vec::new(),
        };
        let index = self.buffers.len();
        self.buffers.push(buffer);
        Ok((index, value))
    }

    fn operation<D: CudaDType>(
        &mut self,
        node: NodeIndex,
        op: &TensorGraphNode<f32>,
        operands: &[Value],
        precision: MatMulPrecision,
        kernel_options: ProgramKernelOptions,
    ) -> Result<Value> {
        #[cfg(test)]
        attention_training_forward::binding_observer::record(node, operands);
        let kernel_start = self.kernels.len();
        if D::HostType::TILE_DTYPE == DType::BF16
            && let Some(output) = self.split_bf16_matmul(node, op, operands, precision)?
        {
            if let Some(preparation) = &mut self.online_attention_pair_preparation {
                preparation.record_emitted(
                    node,
                    &self.kernels[kernel_start..],
                    &output,
                    &self.values,
                );
            }
            return Ok(output);
        }
        if D::HostType::TILE_DTYPE == DType::BF16
            && self.small_grid_split_k_bf16
            && let Some(output) =
                self.small_grid_split_bf16_matmul(node, op, operands, precision)?
        {
            if let Some(preparation) = &mut self.online_attention_pair_preparation {
                preparation.record_emitted(
                    node,
                    &self.kernels[kernel_start..],
                    &output,
                    &self.values,
                );
            }
            return Ok(output);
        }
        let mut graph = TensorGraph::<D::HostType>::new();
        let mut inputs = HashMap::new();
        let mut ends = Vec::new();
        for source in operands {
            let n = if source.binding.dtype == DType::U32 {
                let TensorGraphNode::IndexSource { source: index } =
                    &self.index_graph[source.view.source]
                else {
                    anyhow::bail!("integer binding source missing")
                };
                TensorGraphNode::IndexSource {
                    source: index.clone(),
                }
            } else {
                TensorGraphNode::Input {
                    name: "boundary",
                    shape: source.source_shape.clone(),
                }
            };
            let mut end = graph.graph.add_node(n);
            inputs.insert(end, source.binding);
            for view in &source.chain {
                let next = graph.graph.add_node(typed_node(view)?);
                graph.graph.add_edge(end, next, 0);
                end = next;
            }
            ends.push(end);
        }
        let root = if matches!(op, TensorGraphNode::Linear { .. }) {
            let a = crate::tensor::TensorExpr::node_ref(ends[0], operands[0].view.shape.clone());
            let b = crate::tensor::TensorExpr::node_ref(ends[1], operands[1].view.shape.clone());
            let bias = crate::tensor::TensorExpr::node_ref(ends[2], operands[2].view.shape.clone());
            (a.matmul(b)
                + bias
                    .reshape(vec![1, op.shape()[1]])
                    .broadcast(op.shape().clone()))
            .lower_to_graph(&mut graph)
        } else {
            let root = graph.graph.add_node(typed_node(op)?);
            for (slot, end) in ends.into_iter().enumerate() {
                graph.graph.add_edge(end, root, slot);
            }
            root
        };
        let _ = root;
        let nt128_request = self.fused_nt128_bias
            && D::HostType::TILE_DTYPE == DType::BF16
            && matches!(op, TensorGraphNode::Linear { .. })
            && nt128_bias::shape_candidate(op.shape(), &operands[0].view.shape);
        let nn128x64_request = self.nn128x64_ldmatrix
            && D::HostType::TILE_DTYPE == DType::BF16
            && nn128x64_ldmatrix::requested(op, &operands[0].view.shape);
        let mut program = if nt128_request || nn128x64_request {
            Program::<D>::new_bound_with_plan_metadata(
                &self.stream,
                graph,
                inputs,
                None,
                precision,
                kernel_options,
            )?
        } else {
            Program::<D>::new_bound_with_kernel_options(
                &self.stream,
                graph,
                inputs,
                None,
                precision,
                kernel_options,
            )?
        };
        let ldmatrix_region = if nt128_request && self.fused_nt128_bias_ldmatrix {
            nt128_bias::ldmatrix::region(&program)
        } else {
            None
        };
        if nt128_request {
            match nt128_bias::replace(node, &mut program, &self.stream) {
                nt128_bias::Outcome::Selected => {
                    self.info.fused_nt128_bias_programs += 1;
                    if let Some(region) = ldmatrix_region
                        && let Some(pending) = nt128_bias::ldmatrix::pending(node, region, &program)
                    {
                        self.nt128_ldmatrix_pending.push(pending);
                    }
                }
                nt128_bias::Outcome::Retained(reason) => {
                    self.info.fused_nt128_bias_fallback_programs += 1;
                    self.info
                        .fused_nt128_bias_rejections
                        .push(format!("node {}: {reason}", node.index()));
                }
            }
        }
        if nn128x64_request {
            match nn128x64_ldmatrix::pending(node, &program) {
                Ok(pending) => self.nn128x64_pending.push(pending),
                Err(reason) => {
                    self.info.nn128x64_ldmatrix_fallback_programs += 1;
                    self.info
                        .nn128x64_ldmatrix_rejections
                        .push(format!("node {}: {reason:#}", node.index()));
                }
            }
        }
        ensure!(
            !matches!(op, TensorGraphNode::Linear { .. }) || program.root_fused_matmul,
            "linear bias must fuse before BF16 output rounding"
        );
        for kernel in &mut program.kernels {
            kernel.owner = Some(node);
            kernel.label = format!("node={} {} {}", node.index(), op.name(), kernel.label);
        }
        self.info.contiguous_bias_reshape_alias_steps +=
            program.contiguous_bias_reshape_alias_steps;
        self.info.contiguous_bias_reshape_fallback_steps +=
            program.contiguous_bias_reshape_fallback_steps;
        self.info.contiguous_bias_reshape_avoided_bytes +=
            program.contiguous_bias_reshape_avoided_bytes;
        self.info.indexed_loss_cooperative_ordered_backward_programs +=
            program.indexed_loss_cooperative_ordered_backward_programs;
        self.info.indexed_loss_cooperative_ordered_forward_programs +=
            program.indexed_loss_cooperative_ordered_forward_programs;
        self.info.bf16_tensor_core_programs +=
            usize::from(program.ptx.contains("f32.bf16.bf16.f32"));
        self.info.bf16_eight_compute_warp_programs += usize::from(program.bf16_eight_compute_warps);
        self.info.ordinary_bf16_four_warp_programs += program.ordinary_bf16_four_warp_count;
        self.info.plain_bf16_output_programs += program.plain_bf16_output_counts.0;
        self.info.plain_bf16_output_fallback_programs += program.plain_bf16_output_counts.1;
        self.info.medium_bf16_square_programs += usize::from(program.medium_bf16_tile_counts.0 > 0);
        self.info.medium_bf16_tall_programs += usize::from(program.medium_bf16_tile_counts.1 > 0);
        self.info.large_bf16_tall_programs += usize::from(program.large_bf16_tile_counts.0 > 0);
        self.info.large_bf16_wide_programs += usize::from(program.large_bf16_tile_counts.1 > 0);
        self.info.large_bf16_square128_streamed_programs +=
            program.large_square128_streamed_counts.0;
        self.info.large_bf16_square128_streamed_fallback_programs +=
            program.large_square128_streamed_counts.1;
        self.info.batched_attention_score_programs +=
            program.batched_attention_bf16_tile_counts.scores;
        self.info.batched_attention_query_width_programs +=
            program.batched_attention_bf16_tile_counts.query_width;
        self.info.batched_attention_width_query_programs +=
            program.batched_attention_bf16_tile_counts.width_query;
        self.info.batched_attention_tall_programs +=
            program.batched_attention_bf16_tile_counts.tall;
        self.info.batched_attention_wide_programs +=
            program.batched_attention_bf16_tile_counts.wide;
        self.info.column_reduction_four_regions += program.column_reduction_counts.0;
        self.info.column_reduction_eight_regions += program.column_reduction_counts.1;
        match D::HostType::TILE_DTYPE {
            DType::F32 => self.info.f32_bytes += program.bytes(),
            DType::BF16 => self.info.bf16_bytes += program.bytes(),
            _ => anyhow::bail!("unsupported mixed program dtype"),
        }
        let value = Value {
            binding: program.output,
            view: VirtualTensor::identity(node, op.shape().clone(), D::HostType::TILE_DTYPE),
            source_shape: op.shape().clone(),
            chain: Vec::new(),
        };
        if let Some(preparation) = &mut self.attention_training_preparation {
            preparation.record_operation(node, &program, &value, &self.values);
        }
        if let Some(preparation) = &mut self.online_attention_pair_preparation {
            preparation.record_operation(node, &program, &value, &self.values);
        }
        self.kernels.extend(program.kernels.clone());
        self.programs.push(Box::new(program));
        Ok(value)
    }

    pub fn upload_with_indices(
        &mut self,
        inputs: HashMap<String, Vec<f32>>,
        indices: crate::tensor::IndexInputs,
    ) -> Result<()> {
        self.upload_impl(inputs, indices, false)
    }

    pub(crate) fn for_resident<G>(
        graph: &TensorGraph<f32, G>,
        precision: MatMulPrecision,
        replay: bool,
        bindings: ResidentBindings,
    ) -> Result<Self> {
        Self::for_resident_with_options(graph, precision, replay, bindings, MixedOptions::default())
    }

    pub(crate) fn for_resident_with_options<G>(
        graph: &TensorGraph<f32, G>,
        precision: MatMulPrecision,
        replay: bool,
        bindings: ResidentBindings,
        options: MixedOptions,
    ) -> Result<Self> {
        Self::prepare(
            graph,
            precision,
            replay,
            false,
            true,
            Some(bindings),
            options,
        )
    }
    pub(crate) fn execution_stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
    pub(crate) fn for_resident_with_shadows<G>(
        graph: &TensorGraph<f32, G>,
        precision: MatMulPrecision,
        replay: bool,
        bindings: ResidentBindings,
        options: MixedOptions,
        shadows: HashMap<NodeIndex, ShadowBinding>,
    ) -> Result<Self> {
        Self::prepare_with_shadows(
            graph,
            precision,
            replay,
            false,
            true,
            Some(bindings),
            options,
            shadows,
        )
    }
    pub(crate) fn device_binding(
        &self,
        node: NodeIndex,
    ) -> Result<(Binding, VirtualTensor, Vec<usize>)> {
        ensure!(self.has_run, "mixed graph has not run");
        let value = self
            .values
            .get(&node)
            .context("mixed gradient not retained")?;
        ensure!(
            matches!(value.binding.dtype, DType::F32 | DType::BF16)
                && value.binding.dtype == value.view.dtype,
            "resident gradient storage/view mismatch"
        );
        Ok((
            value.binding,
            value.view.clone(),
            value.source_shape.clone(),
        ))
    }
    pub(crate) fn refresh_parameters(
        &mut self,
        parameters: &HashMap<NodeIndex, Arc<CudaSlice<f32>>>,
    ) -> Result<direct::MasterLeases> {
        ensure!(
            !self
                .direct
                .as_ref()
                .is_some_and(direct::DirectBindings::has_shadows),
            "shadow-bound executor requires paired parameter refresh"
        );
        Ok(self
            .refresh_parameters_impl_with_shadows(parameters, &HashMap::new(), false)?
            .masters)
    }
    pub(crate) fn refresh_parameters_with_shadows(
        &mut self,
        parameters: &HashMap<NodeIndex, Arc<CudaSlice<f32>>>,
        shadows: &HashMap<NodeIndex, ShadowBinding>,
    ) -> Result<RetiredShadowBindings> {
        self.refresh_parameters_impl_with_shadows(parameters, shadows, false)
    }
    // `preparing` is used only by prepare while holding the stream gate.
    fn refresh_parameters_impl_with_shadows(
        &mut self,
        parameters: &HashMap<NodeIndex, Arc<CudaSlice<f32>>>,
        shadows: &HashMap<NodeIndex, ShadowBinding>,
        preparing: bool,
    ) -> Result<RetiredShadowBindings> {
        self.freshness.dirty();
        let began = std::time::Instant::now();
        ensure!(
            parameters.len() == self.bound_parameters.len(),
            "resident parameter set changed"
        );
        for (&node, source) in parameters {
            ensure!(
                self.bound_parameters.contains_key(&node) && self.sources.contains_key(&node),
                "unexpected resident parameter"
            );
            ensure!(
                Arc::ptr_eq(source.context(), self.stream.context())
                    && source.len() == self.values[&node].binding.len,
                "resident parameter context/extent mismatch"
            );
        }
        let gate = self.gate.clone();
        let _guard = if preparing {
            None
        } else {
            Some(
                gate.lock()
                    .map_err(|_| anyhow::anyhow!("mixed stream lock poisoned"))?,
            )
        };
        if let Some(direct) = &mut self.direct {
            let retired = direct.refresh_with_shadows(
                &self.stream,
                parameters,
                direct::DirectFrame {
                    kernels: &mut self.kernels,
                    values: &mut self.values,
                    captured: &mut self.captured,
                    readbacks: self.readbacks.get_mut(),
                },
                shadows,
            )?;
            for (&node, source) in parameters {
                self.bound_parameters.insert(node, Arc::downgrade(source));
            }
            self.update_direct_info();
            self.transfers.parameter_refresh_seconds = began.elapsed().as_secs_f64();
            return Ok(retired);
        }
        for (&node, source) in parameters {
            let previous = self
                .bound_parameters
                .get_mut(&node)
                .context("resident source missing")?;
            let identity = Arc::downgrade(source);
            if !Weak::ptr_eq(previous, &identity) {
                let SourceStorage::Owned(index) = self.sources[&node] else {
                    anyhow::bail!("resident source ownership changed");
                };
                let Buffer::F32(target) = &mut self.buffers[index] else {
                    anyhow::bail!("resident master not f32")
                };
                self.stream.memcpy_dtod(source.as_ref(), target)?;
                self.transfers.parameter_copies += 1;
                self.transfers.parameter_bytes += source.len() * 4;
                *previous = identity;
            }
        }
        self.transfers.parameter_refresh_seconds = began.elapsed().as_secs_f64();
        ensure!(
            shadows.is_empty(),
            "BF16 shadows require direct parameter bindings"
        );
        Ok(RetiredShadowBindings {
            masters: HashMap::new(),
            shadows: HashMap::new(),
        })
    }

    fn update_direct_info(&mut self) {
        if let Some(direct) = &self.direct {
            self.info.direct_capture_hits = direct.capture_hits;
            self.info.direct_capture_misses = direct.capture_misses;
            self.info.direct_capture_variants = direct.variants(self.captured.is_some());
        }
    }

    /// Validate every input and parameter before replacing any device source.
    pub fn upload(&mut self, inputs: HashMap<String, Vec<f32>>) -> Result<()> {
        self.upload_impl(inputs, HashMap::new(), false)
    }
    fn upload_impl(
        &mut self,
        inputs: HashMap<String, Vec<f32>>,
        mut indices: crate::tensor::IndexInputs,
        allow_missing_inputs: bool,
    ) -> Result<()> {
        self.freshness.dirty();
        let began = std::time::Instant::now();
        if allow_missing_inputs {
            for n in self.index_graph.graph.node_weights() {
                if let TensorGraphNode::IndexSource { source } = n
                    && let crate::tensor::indices::IndexSource::Input(name) = source.source()
                {
                    indices.insert((*name).into(), vec![0; source.shape().iter().product()]);
                }
            }
        }
        let index_values = self.index_plan.prepare(&self.index_graph, indices)?;
        let mut copies = Vec::new();
        for (&node, &storage) in &self.sources {
            if self.constants_initialized
                && matches!(self.source_nodes[&node], TensorGraphNode::Constant { .. })
            {
                continue;
            }
            if self.bound_parameters.contains_key(&node) {
                continue;
            }
            let SourceStorage::Owned(index) = storage else {
                anyhow::bail!("external source is not a registered resident master");
            };
            let data = match &self.source_nodes[&node] {
                TensorGraphNode::Input { name, .. } => match inputs.get(*name) {
                    Some(data) => data.clone(),
                    None if allow_missing_inputs => continue,
                    None => anyhow::bail!("mixed input {name} missing"),
                },
                TensorGraphNode::Constant { data, .. } => data.as_ref().clone(),
                TensorGraphNode::Parameter { data, .. } => data.lock()?.to_vec(),
                _ => anyhow::bail!("mixed source graph changed"),
            };
            ensure!(
                data.len() == self.values[&node].binding.len,
                "mixed source length mismatch"
            );
            copies.push((index, data));
        }
        let gate = self.gate.clone();
        // Missing feeds are permitted only during gated construction warmup.
        let _guard = if allow_missing_inputs {
            None
        } else {
            Some(
                gate.lock()
                    .map_err(|_| anyhow::anyhow!("mixed stream lock poisoned"))?,
            )
        };
        self.synchronize()?;
        self.uploaded = false;
        self.has_run = false;
        self.transfers = MixedTransfers::default();
        for (index, data) in copies {
            if let Buffer::F32(buffer) = &mut self.buffers[index] {
                self.stream.memcpy_htod(&data, buffer)?;
                self.transfers.f32_uploads += 1;
                self.transfers.f32_bytes += data.len() * 4;
            }
        }
        for (node, data) in index_values {
            self.transfers.index_uploads += 1;
            self.transfers.index_bytes += data.len() * 4;
            self.stream.memcpy_htod(
                data.as_slice(),
                self.index_buffers
                    .get_mut(&node)
                    .context("integer allocation missing")?,
            )?;
        }
        self.uploaded = true;
        self.freshness.uploaded();
        self.constants_initialized = true;
        self.transfers.upload_seconds = began.elapsed().as_secs_f64();
        Ok(())
    }
    pub fn run(&mut self) -> Result<()> {
        self.freshness.dirty();
        let gate = self.gate.clone();
        let _guard = gate
            .lock()
            .map_err(|_| anyhow::anyhow!("mixed stream lock poisoned"))?;
        ensure!(self.uploaded, "mixed inputs have not been uploaded");
        if let Some(captured) = &self.captured {
            captured
                .graph
                .as_ref()
                .context("mixed capture missing")?
                .launch()?;
        } else {
            for k in &self.kernels {
                k.run(&self.stream)?;
            }
        }
        self.has_run = true;
        self.freshness.ran();
        Ok(())
    }
    pub fn read(&self, node: NodeIndex) -> Result<Vec<f32>> {
        let gate = self.gate.clone();
        let _guard = gate
            .lock()
            .map_err(|_| anyhow::anyhow!("mixed stream lock poisoned"))?;
        ensure!(self.has_run, "mixed graph has not run");
        let value = self
            .values
            .get(&node)
            .with_context(|| format!("mixed read node {} missing", node.index()))?;
        let mut readbacks = self.readbacks.borrow_mut();
        if let std::collections::hash_map::Entry::Vacant(entry) = readbacks.entry(node) {
            let output = self
                .stream
                .alloc_zeros::<f32>(value.view.shape.iter().product())?;
            let target = Binding {
                pointer: output.device_ptr(&self.stream).0,
                len: output.len(),
                dtype: DType::F32,
            };
            let kernel = materialize(&self.stream, value, target)?;
            entry.insert(Readback {
                buffer: output,
                kernel,
            });
        }
        let readback = readbacks
            .get(&node)
            .context("mixed readback cache missing")?;
        readback.kernel.run(&self.stream)?;
        let mut host = vec![0.; readback.buffer.len()];
        self.stream.memcpy_dtoh(&readback.buffer, &mut host)?;
        self.synchronize()?;
        Ok(host)
    }
    pub fn gradients(
        &self,
        graph: &TensorGraph<f32, WithGrad>,
    ) -> Result<HashMap<usize, Vec<f32>>> {
        graph
            .gradient_metadata()
            .param_to_grad
            .iter()
            .map(|(&id, &node)| Ok((id, self.read(node)?)))
            .collect()
    }
    pub fn synchronize(&self) -> Result<()> {
        self.stream.synchronize().map_err(Into::into)
    }
    /// Ordered captured kernel labels for matching a CUDA graph node trace.
    pub fn kernel_labels(&self) -> Vec<&str> {
        self.kernels
            .iter()
            .map(|kernel| kernel.label.as_str())
            .collect()
    }

    pub fn transfers(&self) -> &MixedTransfers {
        &self.transfers
    }
    pub fn info(&self) -> &MixedInfo {
        &self.info
    }
    pub fn time_runs(&mut self, repeats: usize) -> Result<f64> {
        self.freshness.dirty();
        ensure!(repeats > 0, "mixed timing needs positive repeats");
        let begin = self
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        for _ in 0..repeats {
            self.run()?;
        }
        let end = self
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        Ok(f64::from(begin.elapsed_ms(&end)?) * 1000. / repeats as f64)
    }
}

fn is_view(n: &TensorGraphNode<f32>) -> bool {
    matches!(
        n,
        TensorGraphNode::Reshape { .. }
            | TensorGraphNode::Flatten { .. }
            | TensorGraphNode::Transpose { .. }
            | TensorGraphNode::Permute { .. }
            | TensorGraphNode::BroadcastAxis { .. }
    )
}

fn typed_node<D>(n: &TensorGraphNode<f32>) -> Result<TensorGraphNode<D>> {
    Ok(match n {
        TensorGraphNode::Embedding { shape } => TensorGraphNode::Embedding {
            shape: shape.clone(),
        },
        TensorGraphNode::EmbeddingBackward { shape } => TensorGraphNode::EmbeddingBackward {
            shape: shape.clone(),
        },
        TensorGraphNode::IndexedCrossEntropy { shape } => TensorGraphNode::IndexedCrossEntropy {
            shape: shape.clone(),
        },
        TensorGraphNode::IndexedCrossEntropyBackward { shape } => {
            TensorGraphNode::IndexedCrossEntropyBackward {
                shape: shape.clone(),
            }
        }
        TensorGraphNode::Concat { axis, shape } => TensorGraphNode::Concat {
            axis: *axis,
            shape: shape.clone(),
        },
        TensorGraphNode::Unary { op, shape } => TensorGraphNode::Unary {
            op: *op,
            shape: shape.clone(),
        },
        TensorGraphNode::Binary { op, shape } => TensorGraphNode::Binary {
            op: *op,
            shape: shape.clone(),
        },
        TensorGraphNode::MatMul { shape } => TensorGraphNode::MatMul {
            shape: shape.clone(),
        },
        TensorGraphNode::ReduceAxis { op, axis, shape } => TensorGraphNode::ReduceAxis {
            op: *op,
            axis: *axis,
            shape: shape.clone(),
        },
        TensorGraphNode::Reshape { shape } => TensorGraphNode::Reshape {
            shape: shape.clone(),
        },
        TensorGraphNode::Flatten { shape } => TensorGraphNode::Flatten {
            shape: shape.clone(),
        },
        TensorGraphNode::Transpose { shape } => TensorGraphNode::Transpose {
            shape: shape.clone(),
        },
        TensorGraphNode::Permute { axes, shape } => TensorGraphNode::Permute {
            axes: axes.clone(),
            shape: shape.clone(),
        },
        TensorGraphNode::BroadcastAxis { axis, shape } => TensorGraphNode::BroadcastAxis {
            axis: *axis,
            shape: shape.clone(),
        },
        TensorGraphNode::Gt { shape } => TensorGraphNode::Gt {
            shape: shape.clone(),
        },
        TensorGraphNode::Mask { shape } => TensorGraphNode::Mask {
            shape: shape.clone(),
        },
        TensorGraphNode::Narrow { axis, start, shape } => TensorGraphNode::Narrow {
            axis: *axis,
            start: *start,
            shape: shape.clone(),
        },
        _ => anyhow::bail!("mixed operator {} is not supported yet", n.name()),
    })
}

/// A cast commutes with a complete bijective storage view, but not arbitrary
/// masking/broadcast/gather maps. Rebuild the recorded chain and compare its
/// complete normalized map before replacing a gather with a contiguous cast.
fn physical_order_cast_source(source: &Value, output_dtype: DType) -> Option<Value> {
    if source.binding.dtype != DType::F32
        || source.view.dtype != DType::F32
        || output_dtype != DType::BF16
        || source.binding.len == 0
        || source.view.predicate.is_some()
        || source.view.out_of_bounds.is_some()
    {
        return None;
    }
    let extent = |shape: &[usize]| shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d));
    if extent(&source.source_shape) != Some(source.binding.len)
        || extent(&source.view.shape) != Some(source.binding.len)
    {
        return None;
    }
    let identity = VirtualTensor::identity(
        source.view.source,
        source.source_shape.clone(),
        source.binding.dtype,
    );
    let mut expected = identity.clone();
    let mut reordered = false;
    for operation in &source.chain {
        expected = match operation {
            TensorGraphNode::Reshape { shape } | TensorGraphNode::Flatten { shape } => {
                expected.reshape(shape.clone()).ok()?
            }
            TensorGraphNode::Permute { axes, .. } => {
                reordered |= axes.iter().enumerate().any(|(i, &axis)| i != axis);
                expected.permute(axes).ok()?
            }
            TensorGraphNode::Transpose { .. } => {
                let mut axes: Vec<_> = (0..expected.shape.len()).collect();
                if axes.len() < 2 {
                    return None;
                }
                let end = axes.len();
                axes.swap(end - 2, end - 1);
                reordered = true;
                expected.permute(&axes).ok()?
            }
            _ => return None,
        };
        if expected.shape != *operation.shape() {
            return None;
        }
    }
    if !reordered
        || expected.shape != source.view.shape
        || expected.access.normalize_in_domain(&expected.shape).ok()?
            != source
                .view
                .access
                .clone()
                .normalize_in_domain(&source.view.shape)
                .ok()?
    {
        return None;
    }
    Some(Value {
        binding: source.binding,
        view: identity,
        source_shape: source.source_shape.clone(),
        chain: Vec::new(),
    })
}

/// The newly allocated BF16 buffer holds the original physical order; every
/// subsequent consumer sees the original logical view and recorded chain.
fn physical_order_cast_result(source: &Value, node: NodeIndex, target: Binding) -> Value {
    let mut view = source.view.clone();
    view.source = node;
    view.dtype = target.dtype;
    Value {
        binding: target,
        view,
        source_shape: source.source_shape.clone(),
        chain: source.chain.clone(),
    }
}

fn materialize(stream: &Arc<CudaStream>, source: &Value, target: Binding) -> Result<Kernel> {
    ensure!(
        source.view.shape.iter().product::<usize>() == target.len,
        "mixed materialization extent mismatch"
    );
    let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
    let node = tiles.graph.add_node(TileIRBuilder::new().finish());
    tiles.set_materialized_view_dtype(node, &source.view, &source.source_shape, target.dtype)?;
    tiles.optimize_indices()?;
    let graph = PtxGraph::from(tiles).with_target(PtxTarget::from_context(stream.context())?);
    let name = graph
        .kernel_name(node)
        .context("mixed cast kernel absent")?;
    let module = stream
        .context()
        .load_module(Ptx::from_src(graph.to_ptx()))?;
    Ok(Kernel {
        owner: None,
        label: format!(
            "materialize {:?} to {:?} shape={:?}",
            source.binding.dtype, target.dtype, source.view.shape
        ),
        function: module.load_function(name)?,
        config: crate::ptx::target::pointwise_launch_config(u32::try_from(target.len)?),
        pointers: vec![source.binding.pointer, target.pointer],
        zero_before: None,
    })
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod column_reduction_tests;

#[cfg(test)]
mod large_tile_tests;

#[cfg(test)]
mod batched_attention_tile_tests;

#[cfg(test)]
mod split_k_square_tests;

#[cfg(test)]
mod ordinary_four_warp_tests;

#[cfg(test)]
mod indexed_loss_ordered_tests;

#[cfg(test)]
mod indexed_loss_forward_tests;
