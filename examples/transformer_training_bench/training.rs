use super::{memory::Memory, model::Model};
use anyhow::{Context, Result, ensure};
use clap::Parser;
use serde::Serialize;
use serde_json::json;
use std::{
    collections::HashMap,
    fs::OpenOptions,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};
use tnsr::{
    graph::{NodeIndex, TensorGraph, WithGrad},
    ptx::{
        mixed::{MixedExecutor, MixedInfo, MixedTransfers},
        resident::{AdamGroup, GradientNormPartialsStatus, ResidentAdam},
    },
    tile::{
        BatchedAttentionBf16TilePolicy, ColumnReductionPolicy, LargeBf16TilePolicy,
        MatMulPrecision, MediumBf16TilePolicy, SplitKBf16TilePolicy,
    },
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum, Serialize)]
#[serde(rename_all = "snake_case")]
enum MediumTilePolicy {
    #[default]
    Off,
    SquareOnly,
    SquareThenTall,
}
impl MediumTilePolicy {
    fn policy(self) -> MediumBf16TilePolicy {
        match self {
            Self::Off => MediumBf16TilePolicy::Off,
            Self::SquareOnly => MediumBf16TilePolicy::SquareOnly,
            Self::SquareThenTall => MediumBf16TilePolicy::SquareThenTall,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum, Serialize)]
#[serde(rename_all = "snake_case")]
enum ColumnPolicy {
    #[default]
    Off,
    Four,
    Eight,
}
impl ColumnPolicy {
    fn policy(self) -> ColumnReductionPolicy {
        match self {
            Self::Off => ColumnReductionPolicy::Off,
            Self::Four => ColumnReductionPolicy::Four,
            Self::Eight => ColumnReductionPolicy::Eight,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum, Serialize)]
#[serde(rename_all = "snake_case")]
enum LargeTilePolicy {
    #[default]
    Off,
    Tall,
    Wide,
    TallThenWide,
    #[value(name = "square128-streamed")]
    Square128Streamed,
}
impl LargeTilePolicy {
    fn policy(self) -> LargeBf16TilePolicy {
        match self {
            Self::Off => LargeBf16TilePolicy::Off,
            Self::Tall => LargeBf16TilePolicy::Tall,
            Self::Wide => LargeBf16TilePolicy::Wide,
            Self::TallThenWide => LargeBf16TilePolicy::TallThenWide,
            Self::Square128Streamed => LargeBf16TilePolicy::Square128Streamed,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum, Serialize)]
#[serde(rename_all = "snake_case")]
enum AttentionTilePolicy {
    #[default]
    Off,
    TallScores,
    WideScores,
    TallScoresByGrid,
}
impl AttentionTilePolicy {
    fn policy(self) -> BatchedAttentionBf16TilePolicy {
        match self {
            Self::Off => BatchedAttentionBf16TilePolicy::Off,
            Self::TallScores => BatchedAttentionBf16TilePolicy::TallScores,
            Self::WideScores => BatchedAttentionBf16TilePolicy::WideScores,
            Self::TallScoresByGrid => BatchedAttentionBf16TilePolicy::TallScoresByGrid,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum, Serialize)]
#[serde(rename_all = "snake_case")]
enum SplitTilePolicy {
    #[default]
    Off,
    Square64,
    #[value(name = "rectangle128x64")]
    Rectangle128x64,
}
impl SplitTilePolicy {
    fn policy(self) -> SplitKBf16TilePolicy {
        match self {
            Self::Off => SplitKBf16TilePolicy::Off,
            Self::Square64 => SplitKBf16TilePolicy::Square64,
            Self::Rectangle128x64 => SplitKBf16TilePolicy::Rectangle128x64,
        }
    }
}

#[derive(Parser)]
struct Args {
    #[arg(long)]
    archive: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 3)]
    warmups: usize,
    #[arg(long, default_value_t = 20)]
    repeats: usize,
    /// Capture only measured updates with an attached CUDA profiler.
    #[arg(long)]
    profile: bool,
    /// Use the opt-in compact derivative on eligible f32 softmax regions.
    #[arg(long)]
    compact_softmax: bool,
    /// Cast bijective weight views contiguously before applying their metadata.
    #[arg(long)]
    physical_order_casts: bool,
    /// Avoid clearing/reading gradient zeros before the first accumulation.
    #[arg(long)]
    first_accumulation_overwrite: bool,
    /// Fuse exact global norm partials into the first accumulation.
    #[arg(long, requires = "first_accumulation_overwrite")]
    gradient_norm_partials: bool,
    /// Borrow eligible BF16 gradient inputs during a single-microbatch Adam step.
    #[arg(long, requires_all = ["first_accumulation_overwrite", "gradient_norm_partials"])]
    borrowed_bf16_adam_source: bool,
    /// Vectorize eligible borrowed BF16-first Adam updates with scalar fallback.
    #[arg(long, requires = "borrowed_bf16_adam_source")]
    vector_bf16_first_adam: bool,
    /// Fuse declared whole-matrix BF16 weight conversions into staged Adam outputs.
    #[arg(long, requires_all = ["physical_order_casts", "direct_master_bindings"])]
    bf16_master_shadows: bool,
    /// Batch eligible small first-accumulation/norm kernels without changing their trees.
    #[arg(long, requires_all = ["first_accumulation_overwrite", "gradient_norm_partials"])]
    small_accumulation_norm_batches: bool,
    /// Launch standalone Adam kernels with 256 threads per block.
    #[arg(long)]
    adam_256_threads: bool,
    /// Reuse shared scratch for eligible ordinary BF16 GEMMs.
    #[arg(long)]
    shared_memory_reuse: bool,
    /// Reuse resident allocations after completed compilation prefixes.
    #[arg(long)]
    incremental_allocation_reuse: bool,
    /// Read resident masters directly with leased, address-keyed replay bindings.
    #[arg(long)]
    direct_master_bindings: bool,
    /// Fuse closed f32 attention normalization and its BF16 probability cast.
    #[arg(long, requires = "compact_softmax")]
    attention_softmax_fusion: bool,
    /// Fuse proven S512 training attention while retaining all four checkpoints.
    #[arg(long)]
    attention_training_forward_fusion: bool,
    /// Screen the numerical online attention pair while retaining old allocations.
    #[arg(long, conflicts_with = "attention_training_forward_fusion")]
    online_attention_pair_numerical_screen: bool,
    /// Write eligible online attention gradients directly into the closed QKV pack.
    #[arg(long, requires_all = ["online_attention_pair_numerical_screen", "qkv_grad_pack_fusion"])]
    online_attention_packed_backward: bool,
    #[arg(long, requires = "online_attention_pair_numerical_screen")]
    online_attention_register_statistics: bool,
    /// Fuse the closed compact attention derivative and its BF16 score cast.
    #[arg(long, requires = "compact_softmax")]
    attention_softmax_backward_fusion: bool,
    /// Simplify stores for proven full-tile plain BF16 GEMM outputs.
    #[arg(long)]
    plain_bf16_output: bool,
    /// Use the opt-in NT128 fused linear kernel after layout/resource proofs.
    #[arg(long)]
    fused_nt128_bias: bool,
    /// Use qualified collective operand loads in the two eligible NT128 shapes.
    #[arg(long, requires = "fused_nt128_bias")]
    fused_nt128_bias_ldmatrix: bool,
    /// Use qualified rectangular collective-load kernels for eligible FFN input gradients.
    #[arg(long, requires = "plain_bf16_output")]
    nn128x64_ldmatrix: bool,
    /// Experimental alias for proved contiguous BF16 bias-reduction reshapes.
    #[arg(long)]
    contiguous_bias_reshape_aliases: bool,
    /// Use the bounded larger tile for eligible N64 BF16 GEMMs.
    #[arg(long)]
    narrow_bf16_tile: bool,
    /// Use all eight warps in eligible 64x64 BF16 GEMM compute tiles.
    #[arg(long)]
    bf16_eight_compute_warps: bool,
    /// Use four block warps for eligible ordinary 64x64 BF16 GEMMs.
    #[arg(long)]
    ordinary_bf16_four_warp_block: bool,
    /// Select larger rank-two tiles using conservative grid coverage.
    #[arg(long, value_enum, default_value_t = MediumTilePolicy::Off)]
    medium_bf16_tile_policy: MediumTilePolicy,
    /// Compute eligible weight-gradient products in their final physical order.
    #[arg(long)]
    transposed_bf16_outputs: bool,
    /// Pack closed QKV gradient windows without materialized padding or joins.
    #[arg(long)]
    qkv_grad_pack_fusion: bool,
    /// Fuse closed mixed SwiGLU forward/backward pairs with primal recomputation.
    #[arg(long)]
    swiglu_pair_fusion: bool,
    /// Reduce adjacent columns together while preserving each reduction tree.
    #[arg(long, value_enum, default_value_t = ColumnPolicy::Off)]
    column_reduction_policy: ColumnPolicy,
    /// Select bounded larger ordinary BF16 tiles with proven shared reuse.
    #[arg(long, value_enum, default_value_t = LargeTilePolicy::Off)]
    large_bf16_tile_policy: LargeTilePolicy,
    /// Select exact BF16 rectangle geometry for eligible batched attention.
    #[arg(long, value_enum, default_value_t = AttentionTilePolicy::Off)]
    batched_attention_bf16_tile_policy: AttentionTilePolicy,
    /// Fuse eligible LayerNorm row pairs with original plan boundaries.
    #[arg(long)]
    layer_norm_pair_fusion: bool,
    /// Select exact square tiles for qualified long BF16 partition kernels.
    #[arg(long, value_enum, default_value_t = SplitTilePolicy::Off)]
    split_k_bf16_tile_policy: SplitTilePolicy,
    /// Split eligible small-grid BF16 matmuls into four F32 partial sums.
    #[arg(long)]
    small_grid_split_k_bf16: bool,
    /// Cooperatively prepare CE backward statistics with the original ordered sum.
    #[arg(long)]
    indexed_loss_cooperative_ordered_backward: bool,
    /// Cooperatively prepare CE forward statistics with the original ordered sum.
    #[arg(long)]
    indexed_loss_cooperative_ordered_forward: bool,
}
#[derive(Clone, Debug, Serialize)]
struct Sample {
    phase: String,
    index: usize,
    elapsed_ms: f64,
    loss: f32,
    gradient_norm_before_clip: f64,
    gradient_scale: f32,
    optimizer_step: usize,
    kernel_launches: usize,
    replay_active: bool,
    gradient_norm_partials: GradientNormPartialsStatus,
    borrowed_bf16_adam_source: serde_json::Value,
    vector_bf16_first_adam: serde_json::Value,
    bf16_master_shadows: serde_json::Value,
    small_accumulation_norm_batches: serde_json::Value,
    info: MixedInfo,
    transfers: MixedTransfers,
}
fn step(
    model: &Model,
    owner: &mut ResidentAdam,
    executor: &mut MixedExecutor,
    graph: &TensorGraph<f32, WithGrad>,
    output: NodeIndex,
    borrowed_bf16_adam_source: bool,
    (phase, index): (&str, usize),
) -> Result<Sample> {
    // Synchronization belongs to the complete step; previous steps have already
    // synchronized too. Exact-index feed construction/upload is intentionally timed.
    let started = Instant::now();
    owner.begin_step()?;
    let indices = model.config.feeds()?;
    owner.accumulate_mixed(executor, graph, HashMap::new(), indices)?;
    let loss = executor.read(output)?;
    ensure!(
        loss.len() == 1 && loss[0].is_finite(),
        "nonfinite or nonscalar language-model loss"
    );
    let status = if borrowed_bf16_adam_source {
        owner.step_mixed(executor, model.config.max_grad_norm)?
    } else {
        owner.step(model.config.max_grad_norm)?
    };
    executor.synchronize()?;
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    ensure!(
        status.gradient_norm.is_finite() && status.gradient_scale.is_finite(),
        "nonfinite update status"
    );
    Ok(Sample {
        phase: phase.into(),
        index,
        elapsed_ms,
        loss: loss[0],
        gradient_norm_before_clip: status.gradient_norm,
        gradient_scale: status.gradient_scale,
        optimizer_step: status.timestep,
        kernel_launches: status.kernel_launches,
        replay_active: executor.info().replay,
        gradient_norm_partials: owner.gradient_norm_partials_status(),
        borrowed_bf16_adam_source: serde_json::to_value(owner.borrowed_bf16_adam_status())?,
        vector_bf16_first_adam: serde_json::to_value(owner.vector_bf16_first_adam_status())?,
        bf16_master_shadows: serde_json::to_value(owner.bf16_master_shadow_status())?,
        small_accumulation_norm_batches: serde_json::to_value(
            owner.small_accumulation_norm_batches_status(),
        )?,
        info: executor.info().clone(),
        transfers: executor.transfers().clone(),
    })
}
fn writer(path: &Path) -> Result<BufWriter<std::fs::File>> {
    Ok(BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .with_context(|| format!("create fresh benchmark artifact {}", path.display()))?,
    ))
}
fn write_floats(writer: &mut impl Write, values: &[f32]) -> Result<()> {
    ensure!(
        values.iter().all(|v| v.is_finite()),
        "nonfinite qualification snapshot"
    );
    for chunk in values.chunks(4096) {
        let mut bytes = [0u8; 4096 * 4];
        for (value, target) in chunk.iter().zip(bytes.chunks_exact_mut(4)) {
            target.copy_from_slice(&value.to_le_bytes());
        }
        writer.write_all(&bytes[..chunk.len() * 4])?;
    }
    Ok(())
}
fn qualification(
    model: &Model,
    owner: &ResidentAdam,
    sample: &Sample,
    directory: &Path,
) -> Result<serde_json::Value> {
    let started = Instant::now();
    {
        // Existing diagnostic API returns all gradients. Consume their buffers
        // tensor-by-tensor and drop them before requesting any post-update weights.
        let mut gradients = owner.gradients()?;
        let mut output = writer(&directory.join("qual_gradients.f32"))?;
        for record in &model.records {
            let parameter = model.parameters.get(&record.name, &record.shape)?;
            let values = gradients
                .remove(&parameter.id())
                .with_context(|| format!("missing gradient {}", record.name))?;
            ensure!(
                values.len() == record.count,
                "gradient extent mismatch for {}",
                record.name
            );
            write_floats(&mut output, &values)?;
        }
        ensure!(gradients.is_empty(), "unexpected qualification gradients");
        output.flush()?;
    }
    {
        let mut output = writer(&directory.join("qual_weights.f32"))?;
        for record in &model.records {
            let values = model
                .parameters
                .get(&record.name, &record.shape)?
                .values()?;
            ensure!(
                values.len() == record.count,
                "weight extent mismatch for {}",
                record.name
            );
            write_floats(&mut output, &values)?;
        }
        output.flush()?;
    }
    let result = json!({"config":model.config,"loss":sample.loss,"gradient_norm_before_clip":sample.gradient_norm_before_clip,"gradient_scale":sample.gradient_scale,"optimizer_step":sample.optimizer_step,"gradients_file":"qual_gradients.f32","weights_file":"qual_weights.f32","dtype":"f32-le","offset_units":"f32 elements","tensors":model.records,"snapshot_seconds":started.elapsed().as_secs_f64(),"scope":"untimed first complete update; gradients are clipped; weights are post-Adam; snapshot tensors follow original archive order"});
    let mut output = writer(&directory.join("qualification.json"))?;
    serde_json::to_writer_pretty(&mut output, &result)?;
    output.flush()?;
    Ok(result)
}
pub fn run() -> Result<()> {
    let args = Args::parse();
    ensure!(args.repeats > 0, "repeats must be positive");
    std::fs::create_dir_all(&args.output).context("create benchmark output directory")?;
    ensure!(
        !args.output.join("results.json").exists()
            && !args.output.join("qualification.json").exists(),
        "output already contains benchmark results"
    );
    let setup_started = Instant::now();
    let memory = Memory::new()?;
    let initial_memory = memory.snapshot()?;
    let load_started = Instant::now();
    let model = Model::load(&args.archive)?;
    let load_seconds = load_started.elapsed().as_secs_f64();
    let graph_started = Instant::now();
    let (graph, output) = if args.compact_softmax {
        model.training_graph_with_options(tnsr::graph::GradientOptions {
            compact_softmax: true,
        })?
    } else {
        model.training_graph()?
    };
    let graph_seconds = graph_started.elapsed().as_secs_f64();
    let owner_started = Instant::now();
    let mut owner = ResidentAdam::new_with_graph_replay(
        model
            .parameters
            .iter()
            .map(|(_, entry)| (entry.parameter.clone(), 0))
            .collect(),
        vec![AdamGroup::new(model.config.learning_rate)],
    )?;
    owner.set_update_batch_size(8)?;
    owner.set_experimental_first_accumulation_overwrite(args.first_accumulation_overwrite)?;
    owner.set_experimental_gradient_norm_partials(args.gradient_norm_partials)?;
    owner.set_experimental_small_accumulation_norm_batches(args.small_accumulation_norm_batches)?;
    owner.set_experimental_adam_256_threads(args.adam_256_threads)?;
    owner.set_experimental_borrowed_bf16_adam_source(args.borrowed_bf16_adam_source)?;
    owner.set_experimental_vector_bf16_first_adam(args.vector_bf16_first_adam)?;
    let mixed_options = tnsr::ptx::mixed::MixedOptions {
        physical_order_casts: args.physical_order_casts,
        shared_memory_reuse: args.shared_memory_reuse,
        incremental_allocation_reuse: args.incremental_allocation_reuse,
        direct_master_bindings: args.direct_master_bindings,
        attention_softmax_fusion: args.attention_softmax_fusion,
        attention_training_forward_fusion: args.attention_training_forward_fusion,
        online_attention_pair_numerical_screen: args.online_attention_pair_numerical_screen,
        online_attention_packed_backward: args.online_attention_packed_backward,
        online_attention_register_statistics: args.online_attention_register_statistics,
        attention_softmax_backward_fusion: args.attention_softmax_backward_fusion,
        plain_bf16_output: args.plain_bf16_output,
        fused_nt128_bias: args.fused_nt128_bias,
        fused_nt128_bias_ldmatrix: args.fused_nt128_bias_ldmatrix,
        nn128x64_ldmatrix: args.nn128x64_ldmatrix,
        contiguous_bias_reshape_aliases: args.contiguous_bias_reshape_aliases,
        narrow_bf16_tile: args.narrow_bf16_tile,
        bf16_eight_compute_warps: args.bf16_eight_compute_warps,
        ordinary_bf16_four_warp_block: args.ordinary_bf16_four_warp_block,
        medium_bf16_tile_policy: args.medium_bf16_tile_policy.policy(),
        transposed_bf16_outputs: args.transposed_bf16_outputs,
        qkv_grad_pack_fusion: args.qkv_grad_pack_fusion,
        swiglu_pair_fusion: args.swiglu_pair_fusion,
        column_reduction_policy: args.column_reduction_policy.policy(),
        large_bf16_tile_policy: args.large_bf16_tile_policy.policy(),
        batched_attention_bf16_tile_policy: args.batched_attention_bf16_tile_policy.policy(),
        layer_norm_pair_fusion: args.layer_norm_pair_fusion,
        split_k_bf16_tile_policy: args.split_k_bf16_tile_policy.policy(),
        small_grid_split_k_bf16: args.small_grid_split_k_bf16,
        indexed_loss_cooperative_ordered_backward: args.indexed_loss_cooperative_ordered_backward,
        indexed_loss_cooperative_ordered_forward: args.indexed_loss_cooperative_ordered_forward,
    };
    if args.bf16_master_shadows {
        owner.declare_experimental_bf16_master_shadows(&graph, mixed_options)?;
    }
    owner.begin_step()?;
    let owner_seconds = owner_started.elapsed().as_secs_f64();
    let compile_started = Instant::now();
    let mut executor =
        owner.mixed_executor_with_options(&graph, MatMulPrecision::Tf32x3, mixed_options)?;
    executor.synchronize()?;
    let compile_seconds = compile_started.elapsed().as_secs_f64();
    let setup_seconds = setup_started.elapsed().as_secs_f64();
    let setup_memory = memory.snapshot()?;
    let first = step(
        &model,
        &mut owner,
        &mut executor,
        &graph,
        output,
        args.borrowed_bf16_adam_source,
        ("qualification", 0),
    )?;
    ensure!(
        first.optimizer_step == 1,
        "qualification must be the first optimizer update"
    );
    let qualification = qualification(&model, &owner, &first, &args.output)?;
    executor.synchronize()?;
    let qualification_memory = memory.snapshot()?;
    let mut warmups = Vec::new();
    for index in 0..args.warmups {
        warmups.push(step(
            &model,
            &mut owner,
            &mut executor,
            &graph,
            output,
            args.borrowed_bf16_adam_source,
            ("warmup", index),
        )?);
    }
    executor.synchronize()?;
    let lifetime_peak_before_reset = memory.snapshot()?;
    memory.reset_highs()?;
    let steady_initial_memory = memory.snapshot()?;
    if args.profile {
        cudarc::driver::profiler_start().context("start measured CUDA profile")?;
    }
    let mut samples = Vec::new();
    for index in 0..args.repeats {
        samples.push(step(
            &model,
            &mut owner,
            &mut executor,
            &graph,
            output,
            args.borrowed_bf16_adam_source,
            ("measured", index),
        )?);
    }
    executor.synchronize()?;
    if args.profile {
        cudarc::driver::profiler_stop().context("stop measured CUDA profile")?;
    }
    let steady_memory = memory.snapshot()?;
    let total_ms = samples.iter().map(|s| s.elapsed_ms).sum::<f64>();
    let tokens = model.config.rows()?;
    let mut sorted = samples.iter().map(|s| s.elapsed_ms).collect::<Vec<_>>();
    sorted.sort_by(f64::total_cmp);
    let median_ms = if sorted.len().is_multiple_of(2) {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    };
    let replay_steps = samples.iter().filter(|s| s.replay_active).count();
    let report = json!({
        "backend":"tnsr mixed BF16 resident Adam replay","kernel_labels":if args.profile {Some(executor.kernel_labels())} else {None},"profiled":args.profile,"transposed_bf16_outputs":args.transposed_bf16_outputs,"qkv_grad_pack_fusion":args.qkv_grad_pack_fusion,"swiglu_pair_fusion":args.swiglu_pair_fusion,"column_reduction_policy":args.column_reduction_policy,"large_bf16_tile_policy":args.large_bf16_tile_policy,"batched_attention_bf16_tile_policy":args.batched_attention_bf16_tile_policy,"layer_norm_pair_fusion":args.layer_norm_pair_fusion,"split_k_bf16_tile_policy":args.split_k_bf16_tile_policy,"small_grid_split_k_bf16":args.small_grid_split_k_bf16,"indexed_loss_cooperative_ordered_backward":args.indexed_loss_cooperative_ordered_backward,"indexed_loss_cooperative_ordered_forward":args.indexed_loss_cooperative_ordered_forward,"attention_softmax_fusion":args.attention_softmax_fusion,"attention_training_forward_fusion":args.attention_training_forward_fusion,"online_attention_pair_numerical_screen":args.online_attention_pair_numerical_screen,"online_attention_packed_backward":args.online_attention_packed_backward,"online_attention_register_statistics":args.online_attention_register_statistics,"attention_softmax_backward_fusion":args.attention_softmax_backward_fusion,"plain_bf16_output":args.plain_bf16_output,"fused_nt128_bias":args.fused_nt128_bias,"fused_nt128_bias_ldmatrix":args.fused_nt128_bias_ldmatrix,"nn128x64_ldmatrix":args.nn128x64_ldmatrix,"contiguous_bias_reshape_aliases":args.contiguous_bias_reshape_aliases,"narrow_bf16_tile":args.narrow_bf16_tile,"bf16_eight_compute_warps":args.bf16_eight_compute_warps,"ordinary_bf16_four_warp_block":args.ordinary_bf16_four_warp_block,"medium_bf16_tile_policy":args.medium_bf16_tile_policy,"incremental_allocation_reuse":args.incremental_allocation_reuse,"direct_master_bindings":args.direct_master_bindings,"shared_memory_reuse":args.shared_memory_reuse,"first_accumulation_overwrite":args.first_accumulation_overwrite,"gradient_norm_partials":args.gradient_norm_partials,"adam_256_threads":args.adam_256_threads,"borrowed_bf16_adam_source":args.borrowed_bf16_adam_source,"vector_bf16_first_adam":args.vector_bf16_first_adam,"bf16_master_shadows":args.bf16_master_shadows,"small_accumulation_norm_batches":args.small_accumulation_norm_batches,"physical_order_casts":args.physical_order_casts,"config":model.config,"archive":args.archive,"parameter_tensors":model.records.len(),"parameter_elements":model.parameter_count(),"master_parameter_bytes":model.parameter_count()*4,"graph_nodes_with_gradients":graph.graph.node_count(),
        "setup":{"total_seconds":setup_seconds,"archive_load_seconds":load_seconds,"graph_build_seconds":graph_seconds,"resident_owner_seconds":owner_seconds,"mixed_compile_seconds":compile_seconds},
        "qualification":qualification,"qualification_step":first,"warmups":warmups,"samples":samples,
        "timing":{"scope":"synchronized complete step: begin/zero, exact-index feed creation+upload, forward+loss, backward, gradient accumulation, norm+clip, Adam+health, device publications, scalar loss readback; excludes archive/setup/qualification snapshots; fixed batch", "mean_ms":total_ms/args.repeats as f64,"median_ms":median_ms,"tokens_per_second_from_total_time":(tokens as f64*args.repeats as f64)/(total_ms/1000.0),"measured_updates":args.repeats,"warmup_updates":args.warmups,"qualification_updates":1,"replay_active_steps":replay_steps,"replay_fallback_steps":args.repeats-replay_steps},
        "memory":{"scope":"CUDA current/default process memory-pool accounting only; USED_HIGH/RESERVED_HIGH are exact for each queried pool, not whole-process VRAM; excludes modules/context and other pools. Root separately samples whole-process NVML. High-water marks reset only after warmups; no allocations released by reset.","initial":initial_memory,"after_setup":setup_memory,"after_qualification":qualification_memory,"lifetime_peak_before_reset":lifetime_peak_before_reset,"steady_initial_after_reset":steady_initial_memory,"steady_after_timing":steady_memory,"tensor_payload_info":executor.info()},
        "numerics":{"storage":if executor.info().online_attention_pair_numerical_screen_pairs > 0 {"BF16 linears, attention outputs and FFN multiply; selected online attention checkpoints described separately; f32 residuals/LN/masters/Adam"} else {"explicit BF16 linears/attention/FFN multiply; f32 residuals/LN/softmax/masters/Adam"},"packed_backward_checkpoints":if executor.info().online_attention_packed_backward_regions > 0 {Some("Raw BF16 gradient rounds and original padded Join zero-add rounds preserved in packed epilogues; old raw gradient allocations retained")} else {None},"attention_checkpoints":if executor.info().online_attention_pair_numerical_screen_pairs > 0 {"selected online pairs: recomputed F32 scores, online unnormalized BF16 probabilities and saved m/l/delta; fallback pairs retain explicit BF16 checkpoints; old quadratic allocations retained"} else {"explicit BF16 QK/P/PV and score-gradient boundaries"},"matmul_precision":"Tf32x3","autograd":if args.compact_softmax {"compact eligible f32 softmax derivative"} else {"primitive default softmax derivative"},"compact_softmax":args.compact_softmax,"adam":{"learning_rate":model.config.learning_rate,"beta1":0.9,"beta2":0.999,"epsilon":1e-8,"weight_decay":0.0,"batch_size":8},"clip_epsilon":1e-6,"experimental_kernel_flags":args.online_attention_register_statistics || args.bf16_master_shadows || args.vector_bf16_first_adam || args.nn128x64_ldmatrix || args.online_attention_packed_backward || args.small_accumulation_norm_batches || args.fused_nt128_bias_ldmatrix || args.borrowed_bf16_adam_source || args.contiguous_bias_reshape_aliases || args.fused_nt128_bias || args.online_attention_pair_numerical_screen || args.plain_bf16_output || args.small_grid_split_k_bf16 || args.indexed_loss_cooperative_ordered_backward || args.indexed_loss_cooperative_ordered_forward || args.ordinary_bf16_four_warp_block || args.split_k_bf16_tile_policy != SplitTilePolicy::Off || args.layer_norm_pair_fusion || args.batched_attention_bf16_tile_policy != AttentionTilePolicy::Off || args.large_bf16_tile_policy != LargeTilePolicy::Off || args.column_reduction_policy != ColumnPolicy::Off || args.swiglu_pair_fusion || args.qkv_grad_pack_fusion || args.medium_bf16_tile_policy != MediumTilePolicy::Off || args.gradient_norm_partials || args.shared_memory_reuse || args.attention_training_forward_fusion || args.attention_softmax_fusion || args.attention_softmax_backward_fusion || args.narrow_bf16_tile || args.bf16_eight_compute_warps}
    });
    let mut output_file = writer(&args.output.join("results.json"))?;
    serde_json::to_writer_pretty(&mut output_file, &report)?;
    output_file.flush()?;
    println!("{}", serde_json::to_string(&report)?);
    Ok(())
}
