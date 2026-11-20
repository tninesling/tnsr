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
        resident::{AdamGroup, ResidentAdam},
    },
    tile::MatMulPrecision,
};

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
    info: MixedInfo,
    transfers: MixedTransfers,
}
fn step(
    model: &Model,
    owner: &mut ResidentAdam,
    executor: &mut MixedExecutor,
    graph: &TensorGraph<f32, WithGrad>,
    output: NodeIndex,
    phase: &str,
    index: usize,
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
    let status = owner.step(model.config.max_grad_norm)?;
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
    let (graph, output) = model.training_graph()?;
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
    owner.begin_step()?;
    let owner_seconds = owner_started.elapsed().as_secs_f64();
    let compile_started = Instant::now();
    let mut executor = owner.mixed_executor(&graph, MatMulPrecision::Tf32x3)?;
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
        "qualification",
        0,
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
            "warmup",
            index,
        )?);
    }
    executor.synchronize()?;
    let lifetime_peak_before_reset = memory.snapshot()?;
    memory.reset_highs()?;
    let steady_initial_memory = memory.snapshot()?;
    let mut samples = Vec::new();
    for index in 0..args.repeats {
        samples.push(step(
            &model,
            &mut owner,
            &mut executor,
            &graph,
            output,
            "measured",
            index,
        )?);
    }
    executor.synchronize()?;
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
        "backend":"tnsr mixed BF16 resident Adam replay","config":model.config,"archive":args.archive,"parameter_tensors":model.records.len(),"parameter_elements":model.parameter_count(),"master_parameter_bytes":model.parameter_count()*4,"graph_nodes_with_gradients":graph.graph.node_count(),
        "setup":{"total_seconds":setup_seconds,"archive_load_seconds":load_seconds,"graph_build_seconds":graph_seconds,"resident_owner_seconds":owner_seconds,"mixed_compile_seconds":compile_seconds},
        "qualification":qualification,"qualification_step":first,"warmups":warmups,"samples":samples,
        "timing":{"scope":"synchronized complete step: begin/zero, exact-index feed creation+upload, forward+loss, backward, gradient accumulation, norm+clip, Adam+health, device publications, scalar loss readback; excludes archive/setup/qualification snapshots; fixed batch", "mean_ms":total_ms/args.repeats as f64,"median_ms":median_ms,"tokens_per_second_from_total_time":(tokens as f64*args.repeats as f64)/(total_ms/1000.0),"measured_updates":args.repeats,"warmup_updates":args.warmups,"qualification_updates":1,"replay_active_steps":replay_steps,"replay_fallback_steps":args.repeats-replay_steps},
        "memory":{"scope":"CUDA current/default process memory-pool accounting only; USED_HIGH/RESERVED_HIGH are exact for each queried pool, not whole-process VRAM; excludes modules/context and other pools. Root separately samples whole-process NVML. High-water marks reset only after warmups; no allocations released by reset.","initial":initial_memory,"after_setup":setup_memory,"after_qualification":qualification_memory,"lifetime_peak_before_reset":lifetime_peak_before_reset,"steady_initial_after_reset":steady_initial_memory,"steady_after_timing":steady_memory,"tensor_payload_info":executor.info()},
        "numerics":{"storage":"explicit BF16 linears/attention/FFN multiply; f32 residuals/LN/softmax/masters/Adam","matmul_precision":"Tf32x3","autograd":"primitive default; no compact-softmax/split-K rewrite","adam":{"learning_rate":model.config.learning_rate,"beta1":0.9,"beta2":0.999,"epsilon":1e-8,"weight_decay":0.0,"batch_size":8},"clip_epsilon":1e-6,"experimental_kernel_flags":false}
    });
    let mut output_file = writer(&args.output.join("results.json"))?;
    serde_json::to_writer_pretty(&mut output_file, &report)?;
    output_file.flush()?;
    println!("{}", serde_json::to_string(&report)?);
    Ok(())
}
