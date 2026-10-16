//! Precision headroom experiment, not a mixed-precision training implementation.
//! Usage: bf16_gemm_probe SPEC.json OUTPUT_DIRECTORY.
#[path = "support/resident_gemm.rs"]
mod resident_gemm;
use anyhow::{Context, Result, ensure};
use cudarc::{
    driver::{CudaContext, CudaGraph, CudaStream, DevicePtr, LaunchConfig, sys},
    nvrtc::Ptx,
};
use resident_gemm::{Launch, ResidentGemm, Shape, submit};
use serde::Deserialize;
use std::{path::Path, sync::Arc};
use tnsr::{
    graph::TensorGraph,
    ptx::{
        PtxGraph,
        types::{BF16, F32},
    },
    tensor::TensorExpr,
    tile::{DType, Dim, Expr, MatMulPrecision, TileGraph, TileIRBuilder, ir::MatrixLayout},
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Spec {
    shape: Shape,
    samples: usize,
    graph_repeats: usize,
    replicas: usize,
}
struct Captured {
    graph: Option<CudaGraph>,
    stream: Arc<CudaStream>,
}
impl Drop for Captured {
    fn drop(&mut self) {
        if let Some(graph) = self.graph.take() {
            if self.stream.context().bind_to_thread().is_ok() {
                drop(graph);
            } else {
                std::mem::forget(graph);
            }
        }
    }
}
impl Captured {
    fn new(stream: &Arc<CudaStream>, launches: &[Launch], repeats: usize) -> Result<Self> {
        stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
        let recorded = (|| {
            for _ in 0..repeats {
                submit(stream, launches)?;
            }
            Ok::<_, anyhow::Error>(())
        })();
        // Always terminate capture, including on launch failure.
        let deferred = stream.context().check_err().map_err(anyhow::Error::from);
        let ended = stream.end_capture(
            sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        );
        recorded.and(deferred)?;
        Ok(Self {
            graph: Some(ended?.context("empty probe capture")?),
            stream: stream.clone(),
        })
    }
    fn launch(&self) -> Result<()> {
        self.graph
            .as_ref()
            .context("missing probe graph")?
            .launch()?;
        Ok(())
    }
}
fn conversion(
    device: &Arc<CudaContext>,
    extent: usize,
    input: DType,
    output: DType,
    src: u64,
    dst: u64,
) -> Result<Launch> {
    let mut builder = TileIRBuilder::new();
    builder.start_kernel("precision_cast");
    builder.add_param("input", input, true);
    builder.add_param("output", output, false);
    builder.bounds_check(extent);
    let value = builder.alloc_register(DType::F32, 1, 1);
    let index = Expr::Mul(
        Box::new(Expr::BlockIdx(Dim::X)),
        Box::new(Expr::BlockDim(Dim::X)),
    ) + Expr::ThreadIdx(Dim::X);
    let layout = MatrixLayout {
        rows: 1,
        cols: extent,
        row_stride: extent,
    };
    builder.load_global_predicated(
        value,
        "input",
        index.clone(),
        Expr::Const(0),
        index.clone(),
        layout,
    );
    builder.store_global_predicated("output", value, Expr::Const(0), index, layout);
    let dummy: TensorGraph<f32> = TensorExpr::input("placeholder", vec![extent]).into();
    let mut graph = TileGraph::from_graph(&dummy);
    let node = graph
        .graph
        .node_indices()
        .next()
        .context("missing cast node")?;
    graph.graph[node] = builder.finish();
    graph.optimize_indices()?;
    let source = PtxGraph::from(graph).to_ptx();
    let module = device.load_module(Ptx::from_src(source))?;
    Ok(Launch {
        function: module.load_function(&format!("precision_cast_{}", node.index()))?,
        config: LaunchConfig::for_num_elems(u32::try_from(extent)?),
        pointers: vec![src, dst],
    })
}
fn write_values(path: &Path, values: &[f32]) -> Result<()> {
    let bytes: Vec<_> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    std::fs::write(path, bytes)?;
    Ok(())
}
fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[(sorted.len() - 1) / 2]
}
fn main() -> Result<()> {
    let spec_path = std::env::args().nth(1).context("expected SPEC.json")?;
    let output = std::path::PathBuf::from(
        std::env::args()
            .nth(2)
            .context("expected output directory")?,
    );
    let spec: Spec = serde_json::from_slice(&std::fs::read(&spec_path)?)?;
    spec.shape.validate()?;
    ensure!(
        spec.samples >= 20 && spec.graph_repeats > 0,
        "need >=20 samples and positive graph repeats"
    );
    std::fs::create_dir(&output).context("probe output must be fresh")?;
    let device = CudaContext::new(0)?;
    let stream = device.new_stream()?;
    let f32 = ResidentGemm::<F32>::compile(&device, &stream, &spec.shape, MatMulPrecision::Tf32x3)?;
    let bf16 =
        ResidentGemm::<BF16>::compile(&device, &stream, &spec.shape, MatMulPrecision::AllowTf32)?;
    ensure!(
        bf16.schedules
            .iter()
            .any(|schedule| schedule.contains("TensorCoreBF16")),
        "probe did not select BF16 tensor cores"
    );
    ensure!(
        spec.replicas > 0 && spec.graph_repeats.is_multiple_of(spec.replicas),
        "graph repeats must be a positive multiple of replicas"
    );
    let mut floats = vec![f32];
    let mut halves = vec![bf16];
    for _ in 1..spec.replicas {
        floats.push(floats[0].replicate(&stream)?);
        halves.push(halves[0].replicate(&stream)?);
    }
    let mut promoted = Vec::new();
    let mut modes: Vec<Vec<Launch>> = vec![Vec::new(); 4];
    for (f32, bf16) in floats.iter().zip(&halves) {
        let mut casts = Vec::new();
        for (&(source, extent), &(destination, target_extent)) in f32.ports.iter().zip(&bf16.ports)
        {
            ensure!(extent == target_extent, "precision input extent mismatch");
            casts.push(conversion(
                &device,
                extent,
                DType::F32,
                DType::BF16,
                source,
                destination,
            )?);
        }
        let output = stream.alloc_zeros::<f32>(spec.shape.batch * spec.shape.m * spec.shape.n)?;
        let egress = conversion(
            &device,
            output.len(),
            DType::BF16,
            DType::F32,
            bf16.output_pointer,
            output.device_ptr(&stream).0,
        )?;
        modes[0].extend(f32.launches.clone());
        modes[1].extend(bf16.launches.clone());
        modes[2].extend(casts.clone());
        modes[2].extend(bf16.launches.clone());
        modes[3].extend(casts);
        modes[3].extend(bf16.launches.clone());
        modes[3].push(egress);
        promoted.push(output);
    }
    let labels = [
        "tf32x3_f32_storage",
        "bf16_resident",
        "bf16_input_casts",
        "bf16_roundtrip_f32",
    ];
    // Complete uploads/JIT/allocation and warm every kernel before timing or capture.
    stream.synchronize()?;
    for mode in &modes {
        for _ in 0..8 {
            submit(&stream, mode)?;
        }
    }
    stream.synchronize()?;
    let graphs = modes
        .iter()
        .map(|mode| Captured::new(&stream, mode, spec.graph_repeats / spec.replicas))
        .collect::<Result<Vec<_>>>()?;
    for graph in &graphs {
        for _ in 0..8 {
            graph.launch()?;
        }
    }
    stream.synchronize()?;
    let mut samples = vec![Vec::new(); 4];
    for iteration in 0..spec.samples {
        let mut order = [0, 1, 2, 3];
        order.rotate_left((iteration / 2) % 4);
        if iteration % 2 == 1 {
            order.reverse();
        }
        for index in order {
            let started = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
            graphs[index].launch()?;
            let ended = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
            samples[index]
                .push(f64::from(started.elapsed_ms(&ended)?) * 1000.0 / spec.graph_repeats as f64);
        }
    }
    stream.synchronize()?;
    let f32 = &floats[0];
    let bf16 = &halves[0];
    let a = f32.output(&stream)?;
    let b = bf16.output(&stream)?;
    ensure!(
        a.iter().chain(&b).all(|value| value.is_finite()),
        "nonfinite probe output"
    );
    for ((float, half), promotion) in floats.iter().zip(&halves).zip(&promoted) {
        ensure!(
            float.output(&stream)? == a && half.output(&stream)? == b,
            "replica outputs differ"
        );
        ensure!(
            stream.memcpy_dtov(promotion)? == b,
            "BF16 promotion changed a finite value"
        );
        half.validate_inputs(&stream, &spec.shape)?;
    }
    write_values(&output.join("tf32x3.f32"), &a)?;
    write_values(&output.join("bf16.f32"), &b)?;
    std::fs::write(output.join("tf32x3.plan"), &f32.plan)?;
    std::fs::write(output.join("bf16.plan"), &bf16.plan)?;
    std::fs::write(output.join("tf32x3.ptx"), &f32.ptx)?;
    std::fs::write(output.join("bf16.ptx"), &bf16.ptx)?;
    let physical_bytes = [
        floats
            .iter()
            .map(ResidentGemm::allocated_bytes)
            .sum::<usize>(),
        halves
            .iter()
            .map(ResidentGemm::allocated_bytes)
            .sum::<usize>(),
    ];
    let l2_bytes = device.attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)?;
    let result = serde_json::json!({"shape":spec.shape,"samples":spec.samples,"graph_repeats":spec.graph_repeats,
        "protocol":"one process; preallocated storage; warmed CUDA Graph of repeated work; CUDA-event times; rotated/reversed mode order",
        "labels":labels,"samples_us":samples,"median_us":samples.iter().map(|values|median(values)).collect::<Vec<_>>(),
        "replicas":spec.replicas,"physical_plan_bytes":physical_bytes,"device_l2_bytes":l2_bytes,"kernels_per_operation":modes.iter().map(|mode|mode.len()/spec.replicas).collect::<Vec<_>>(),
        "tf32x3_schedules":f32.schedules,"bf16_schedules":bf16.schedules,
        "input_elements":bf16.ports.iter().map(|(_,extent)|extent).collect::<Vec<_>>(),"output_elements":a.len(),
        "limits":["Synthetic deterministic operands with actual workload dimensions/layouts; not training or learning-quality parity.","Resident mode excludes cast cost; roundtrip includes fresh casts of both operands/bias and output promotion; reuse across operations is not modeled.","Accuracy against original PyTorch must be evaluated outside timed intervals."]});
    std::fs::write(
        output.join("report.json"),
        serde_json::to_vec_pretty(&result)?,
    )?;
    println!("{result}");
    Ok(())
}
