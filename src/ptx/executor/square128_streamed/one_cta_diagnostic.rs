//! Isolated resident-kernel diagnostic. It never changes production admission.
mod four_warp_square;
mod k64_diagnostic;
mod native_component;
mod plain_output_diagnostic;
use super::*;
use crate::{
    ptx::BF16,
    tensor::TensorExpr,
    tile::{MatMulPrecision, MatMulSchedule, MediumBf16TilePolicy},
};
use anyhow::ensure;
use cudarc::driver::{CudaGraph, DevicePtr, sys};
use half::bf16;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy)]
struct Case {
    name: &'static str,
    m: usize,
    n: usize,
    k: usize,
}
const CASES: [Case; 4] = [
    Case {
        name: "b8_gate_up_dx",
        m: 4096,
        n: 512,
        k: 1408,
    },
    Case {
        name: "b16_qkv_dx",
        m: 8192,
        n: 512,
        k: 1536,
    },
    Case {
        name: "b32_gate_up_dx",
        m: 16384,
        n: 512,
        k: 1408,
    },
    Case {
        name: "b8_down_dx",
        m: 4096,
        n: 1408,
        k: 512,
    },
];

struct Mode {
    // A graph is destroyed while its function/module/storage are still alive.
    capture: Option<CudaGraph>,
    function: CudaFunction,
    _module: Arc<CudaModule>,
    storage: Vec<CudaSlice<bf16>>,
    pointers: Vec<u64>,
    config: LaunchConfig,
    schedule: MatMulSchedule,
    name: &'static str,
    resources: Value,
    samples_us: Vec<f64>,
}
impl Mode {
    fn submit(&self, stream: &Arc<CudaStream>) -> Result<()> {
        let mut launch = stream.launch_builder(&self.function);
        for pointer in &self.pointers {
            launch.arg(pointer);
        }
        // SAFETY: this test owns three BF16 allocations with exactly the
        // compiled input/output extents. Geometry is this module's own plan.
        // Modules and all allocations outlive every launch/capture/readback.
        unsafe { launch.launch(self.config) }?;
        Ok(())
    }
    fn read(&self, stream: &Arc<CudaStream>) -> Result<Vec<bf16>> {
        Ok(stream.memcpy_dtov(&self.storage[2])?)
    }
    fn upload(&mut self, stream: &Arc<CudaStream>, a: &[bf16], b: &[bf16]) -> Result<()> {
        stream.memcpy_htod(a, &mut self.storage[0])?;
        stream.memcpy_htod(b, &mut self.storage[1])?;
        Ok(())
    }
    fn capture(&mut self, stream: &Arc<CudaStream>, repeats: usize) -> Result<()> {
        stream.synchronize()?;
        stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
        let recorded = (|| -> Result<()> {
            for _ in 0..repeats {
                self.submit(stream)?;
            }
            Ok(())
        })();
        let ended = stream.end_capture(
            sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        );
        recorded?;
        self.capture = Some(ended?.context("empty diagnostic capture")?);
        Ok(())
    }
}

fn schedule(
    plan: &PtxExecutionPlan,
) -> Result<(MatMulSchedule, &super::super::super::plan::PtxPlanStep)> {
    let step = plan
        .steps()
        .iter()
        .find(|s| {
            matches!(
                s.action,
                PtxPlanAction::MatMulRegion(_) | PtxPlanAction::Kernel
            )
        })
        .context("diagnostic GEMM absent")?;
    ensure!(
        step.inputs.len() == 2 && step.outputs.len() == 1,
        "plain GEMM ABI required"
    );
    let s = match step.action {
        PtxPlanAction::MatMulRegion(id) => plan.matmul_regions()[id].schedule,
        _ => *plan
            .matmul_schedules()
            .get(&step.node)
            .context("GEMM schedule absent")?,
    };
    Ok((s, step))
}
fn build(
    case: Case,
    candidate: bool,
    stream: &Arc<CudaStream>,
    a: &[bf16],
    b: &[bf16],
    dir: &Path,
) -> Result<Mode> {
    let graph: TensorGraph<bf16> = TensorExpr::input("a", vec![case.m, case.k])
        .matmul(TensorExpr::input("b", vec![case.k, case.n]))
        .into();
    let mut e = PtxExecutor::<BF16>::with_stream(
        stream.clone(),
        PtxReductionMode::DeterministicTree,
        Arc::new(Mutex::new(())),
    )?;
    e.set_matmul_precision(MatMulPrecision::NativeHalf);
    e.set_matmul_shared_layout(MatMulSharedLayout::Padded);
    e.set_matmul_pipeline(MatMulPipeline::Auto);
    e.set_experimental_shared_memory_reuse(true);
    e.set_experimental_medium_bf16_tile_policy(MediumBf16TilePolicy::SquareThenTall);
    e.compile(&graph)?;
    let mut plan = e.execution_plan().context("baseline plan absent")?.clone();
    let (baseline, base_step) = schedule(&plan)?;
    ensure!(
        baseline.block_tile
            == (crate::tile::MatMulTile {
                m: 64,
                n: 64,
                k: 32
            })
            && baseline.block_threads == (32, 8, 1)
            && baseline.warp_topology == (2, 2)
            && baseline.warp_fragments == (2, 2),
        "accepted baseline64² ownership not retained: {baseline:?}"
    );
    ensure!(
        baseline.pipeline_stages == 2
            && baseline.copy_bytes == 16
            && baseline.async_operands == (true, true)
            && baseline.reuse_shared_memory
            && !baseline.operand_layouts.0.column_major
            && !baseline.operand_layouts.1.column_major,
        "baseline must be NN padded async2/copy16/reuse"
    );
    ensure!(
        matches!(
            &graph[base_step.inputs[0]],
            TensorGraphNode::Input { name: "a", .. }
        ) && matches!(
            &graph[base_step.inputs[1]],
            TensorGraphNode::Input { name: "b", .. }
        ),
        "physical input order changed"
    );
    let (ptx, module) = if candidate {
        plan.configure_square128_streamed(
            MatMulSharedLayout::Padded,
            MatMulPipeline::Auto,
            e.target.supports_async_copy(),
            e.target.matmul_resources,
        );
        ensure!(
            plan.large_square128_streamed_count() == 1,
            "static Square candidate proof failed"
        );
        let (tiles, _) = e.lower_planned_tiles(&graph, &HashMap::new(), &mut plan)?;
        // This is the only bypass: compile exactly the candidate TileGraph,
        // without calling the production runtime >=2-block resource admission.
        e.compile_tile_variant(tiles)?
    } else {
        (
            PtxGraph::from(e.lower_planned_tiles(&graph, &HashMap::new(), &mut plan)?.0)
                .with_target(e.target),
            e.module.as_ref().context("baseline module absent")?.clone(),
        )
    };
    if !candidate {
        ensure!(
            ptx.module_source() == e.module_source().context("baseline source absent")?,
            "reconstructed baseline differs from loaded accepted module"
        );
    }
    let (selected, step) = schedule(&plan)?;
    let name = match step.action {
        PtxPlanAction::MatMulRegion(id) => ptx.matmul_region_kernel_name(id),
        _ => ptx.kernel_name(step.node),
    }
    .context("diagnostic function absent")?;
    let function = module.load_function(name)?;
    let (regs, local, shared, max_threads, capacity) = (
        function.num_regs()?,
        function.local_size_bytes()?,
        function.shared_size_bytes()?,
        function.max_threads_per_block()?,
        function.occupancy_max_active_blocks_per_multiprocessor(256, 0, None)?,
    );
    let mode = if candidate {
        "test_only_one_cta_square"
    } else {
        "accepted_baseline"
    };
    let resources = json!({"mode":mode,"function":name,"registers_per_thread":regs,"local_bytes":local,"shared_bytes":shared,"max_threads_per_block":max_threads,"static_active_block_capacity":capacity,"capacity_is_not_measured_occupancy":true,"schedule":format!("{selected:?}"),"production_runtime_gate_would_accept":resources_fit(regs,local,shared,max_threads,capacity,e.target.matmul_resources.shared_bytes_per_block),"production_runtime_gate_bypassed_in_test_only":candidate,"production_activation_claim":false});
    std::fs::write(dir.join(format!("{mode}.ptx")), ptx.module_source())?;
    write_json(&dir.join(format!("{mode}-resources.json")), &resources)?;
    ensure!(
        local == 0 && max_threads >= 256 && capacity >= 1,
        "diagnostic stopped: spill/thread/capacity rejection: {resources}"
    );
    ensure!(
        !candidate || regs <= 192,
        "diagnostic stopped: candidate exceeds192 registers: {resources}"
    );
    ensure!(
        usize::try_from(shared)? == selected.shared_mem_bytes(),
        "shared accounting mismatch"
    );
    let storage = vec![
        stream.memcpy_stod(a)?,
        stream.memcpy_stod(b)?,
        stream.alloc_zeros::<bf16>(case.m * case.n)?,
    ];
    let pointers = storage.iter().map(|s| s.device_ptr(stream).0).collect();
    Ok(Mode {
        capture: None,
        function,
        _module: module,
        storage,
        pointers,
        config: LaunchConfig {
            grid_dim: (
                u32::try_from(case.n / selected.block_tile.n)?,
                u32::try_from(case.m / selected.block_tile.m)?,
                1,
            ),
            block_dim: selected.block_threads,
            shared_mem_bytes: 0,
        },
        schedule: selected,
        name: mode,
        resources,
        samples_us: Vec::new(),
    })
}
fn write_json(path: &Path, value: &Value) -> Result<()> {
    std::fs::write(path, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}
fn compare(a: &[bf16], b: &[bf16], phase: &str) -> Result<()> {
    ensure!(a.len() == b.len(), "output extent mismatch");
    for (index, (&x, &y)) in a.iter().zip(b).enumerate() {
        ensure!(
            if x.is_nan() {
                y.is_nan()
            } else {
                x.to_bits() == y.to_bits()
            },
            "{phase}: output mismatch at{index}: {:04x}/{:04x}",
            x.to_bits(),
            y.to_bits()
        );
    }
    Ok(())
}
fn qualify(
    case: Case,
    modes: &mut [Mode],
    stream: &Arc<CudaStream>,
    a: &[bf16],
    b: &[bf16],
) -> Result<(Vec<bf16>, Value)> {
    let mut outputs = Vec::new();
    for m in modes.iter() {
        m.submit(stream)?;
        outputs.push(m.read(stream)?);
    }
    ensure!(
        outputs.iter().flatten().all(|v| v.is_finite()),
        "finite input produced nonfinite output"
    );
    compare(&outputs[0], &outputs[1], "finite")?;
    let expected = outputs.remove(0);
    let mut sanity = Vec::new();
    for sample in 0..32 {
        let (row, col) = (sample * 7919 % case.m, sample * 104729 % case.n);
        let reference = (0..case.k)
            .map(|j| {
                f64::from(a[row * case.k + j].to_f32()) * f64::from(b[j * case.n + col].to_f32())
            })
            .sum::<f64>();
        let actual = f64::from(expected[row * case.n + col].to_f32());
        let tolerance = 0.01 * (1. + reference.abs());
        ensure!(
            (reference - actual).abs() <= tolerance,
            "independent F64 sanity failed"
        );
        sanity.push(json!({"row":row,"column":col,"reference_f64":reference,"actual":actual,"absolute_error":(reference-actual).abs(),"tolerance":tolerance}));
    }
    let mut phases = Vec::new();
    for (name, left, right) in [
        ("positive_zero", 0., 1.),
        ("negative_zero", -0., 1.),
        ("positive_infinity", 1., f32::INFINITY),
        ("zero_times_infinity", 0., f32::INFINITY),
        ("nan", f32::NAN, 1.),
    ] {
        let aa = vec![bf16::from_f32(left); a.len()];
        let bb = vec![bf16::from_f32(right); b.len()];
        let mut values = Vec::new();
        for mode in modes.iter_mut() {
            mode.upload(stream, &aa, &bb)?;
            mode.submit(stream)?;
            values.push(mode.read(stream)?);
        }
        compare(&values[0], &values[1], name)?;
        phases.push(json!({"phase":name,"all_output_bits_equal_except_nan_payload":true,"elements":expected.len(),"baseline_nan_elements":values[0].iter().filter(|v|v.is_nan()).count()}));
    }
    for mode in modes.iter_mut() {
        mode.upload(stream, a, b)?;
        mode.submit(stream)?;
        compare(&expected, &mode.read(stream)?, "restored finite")?;
    }
    Ok((
        expected,
        json!({"finite_all_output_bits_equal":true,"finite_elements":case.m*case.n,"sampled_f64_sanity":sanity,"special_values":phases,"nan_contract":"categories only; payload not promised"}),
    ))
}
fn median(values: &[f64]) -> f64 {
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    (v[(v.len() - 1) / 2] + v[v.len() / 2]) / 2.
}

#[test]
#[ignore = "manual test-only one-CTA microbenchmark; production admission unchanged"]
fn square128_test_only_one_cta_diagnostic() -> Result<()> {
    let directory = PathBuf::from(
        std::env::var_os("TNSR_SQUARE_ONE_CTA_DIR")
            .context("set an explicit artifact directory")?,
    );
    std::fs::create_dir_all(&directory)?;
    let reverse = std::env::var("TNSR_SQUARE_ONE_CTA_ORDER").as_deref() == Ok("candidate,baseline");
    let timing = std::env::var("TNSR_SQUARE_ONE_CTA_TIMING").as_deref() == Ok("1");
    let repeats = 32;
    let samples = 32;
    let mut results = Vec::new();
    let mut cases = CASES.to_vec();
    if reverse {
        cases.reverse();
    }
    let context = CudaContext::new(0)?;
    let stream = context.new_stream()?;
    for case in cases {
        let dir = directory.join(case.name);
        std::fs::create_dir_all(&dir)?;
        let a = (0..case.m * case.k)
            .map(|i| bf16::from_f32(((i % 97 * 13 % 97) as f32 - 48.) / 97.))
            .collect::<Vec<_>>();
        let b = (0..case.k * case.n)
            .map(|i| bf16::from_f32(((i % 89 * 17 % 89) as f32 - 44.) / 89.))
            .collect::<Vec<_>>();
        let order = if reverse {
            [true, false]
        } else {
            [false, true]
        };
        let mut modes = Vec::new();
        for candidate in order {
            match build(case, candidate, &stream, &a, &b, &dir) {
                Ok(m) => modes.push(m),
                Err(error) => {
                    write_json(
                        &dir.join("result.json"),
                        &json!({"status":"stopped_resource_or_compile","error":format!("{error:#}"),"timing_performed":false}),
                    )?;
                    return Err(error);
                }
            }
        }
        let (expected, checks) = qualify(case, &mut modes, &stream, &a, &b)?;
        let mut pairs = Vec::new();
        let mut warmup_pairs = 0;
        let mut warmup_ms = 0.;
        if timing {
            for m in &mut modes {
                m.capture(&stream, repeats)?;
            }
            let began = Instant::now();
            while began.elapsed() < Duration::from_millis(500) {
                for index in if warmup_pairs % 2 == 0 {
                    [0, 1]
                } else {
                    [1, 0]
                } {
                    modes[index].capture.as_mut().unwrap().launch()?;
                }
                stream.synchronize()?;
                warmup_pairs += 1;
            }
            warmup_ms = began.elapsed().as_secs_f64() * 1000.;
            for sample in 0..samples {
                let mut execution_order = Vec::new();
                let mut baseline = 0.;
                let mut candidate = 0.;
                for index in if sample % 2 == 0 { [0, 1] } else { [1, 0] } {
                    let m = &mut modes[index];
                    let start = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
                    m.capture.as_mut().unwrap().launch()?;
                    let end = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
                    let us = f64::from(start.elapsed_ms(&end)?) * 1000. / repeats as f64;
                    ensure!(us > 0. && us.is_finite(), "invalid CUDA event interval");
                    m.samples_us.push(us);
                    execution_order.push(m.name);
                    if m.schedule.is_square128_streamed() {
                        candidate = us;
                    } else {
                        baseline = us;
                    }
                }
                pairs.push(json!({"sample":sample,"execution_order":execution_order,"baseline_us":baseline,"test_only_candidate_us":candidate,"baseline_over_candidate":baseline/candidate}));
            }
            stream.synchronize()?;
            for m in &modes {
                compare(&expected, &m.read(&stream)?, "post-capture finite")?;
            }
        }
        let summaries=modes.iter().map(|m|json!({"mode":m.name,"resources":m.resources,"samples_us":m.samples_us,"median_us":if timing{Some(median(&m.samples_us))}else{None},"mean_us":if timing{Some(m.samples_us.iter().sum::<f64>()/samples as f64)}else{None}})).collect::<Vec<_>>();
        let result = json!({"status":"qualified_test_fixture_only","case":case.name,"m":case.m,"n":case.n,"k":case.k,"orientation":"NN","dtype":"BF16","construction_order":if reverse{"candidate,baseline"}else{"baseline,candidate"},"qualification":checks,"timing_performed":timing,"samples":if timing{samples}else{0},"graph_kernel_repeats":repeats,"warmup_pairs":warmup_pairs,"warmup_elapsed_ms":warmup_ms,"pairs":pairs,"summaries":summaries,"production_gate_unchanged":true,"production_activation_claim":false,"scope":"Resident reused-input single-GEMM CUDA-event/graph microbenchmark. Compilation, allocation, upload, validation and readback excluded. Not complete updates or end-to-end parity; no TT timing claim."});
        write_json(&dir.join("result.json"), &result)?;
        results.push(result);
        write_json(
            &directory.join("report.json"),
            &json!({"cases":results,"scope":"test-only one-CTA diagnostic; strict production gate untouched"}),
        )?;
    }
    Ok(())
}
