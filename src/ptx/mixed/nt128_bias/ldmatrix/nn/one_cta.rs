//! Distinct one-CTA diagnostic on the exact rejected NN PTX.
//! Original strict test and rejection remain immutable; never production admission.
use super::*;
use cudarc::driver::{
    CudaFunction, CudaGraph, CudaModule, CudaStream, DevicePtr, LaunchConfig, sys,
};
use cudarc::nvrtc::Ptx;
use half::bf16;
use serde_json::{Value as Json, json};
use std::{path::Path, sync::Arc};
struct Built {
    function: CudaFunction,
    _module: Arc<CudaModule>,
    ptx: String,
    name: String,
}
fn build(stream: &Arc<CudaStream>, r: &MatMulRegion, target: PtxTarget) -> Result<Built> {
    let mut t = TileGraph::from_graph(&TensorGraph::<f32>::new());
    t.add_matmul_regions(std::slice::from_ref(r))?;
    t.matmul_region_kernels[0].ir = candidate(r, target)?;
    t.optimize_indices()?;
    let graph = PtxGraph::from(t).with_target(target);
    let ptx = graph.module_source();
    let name = graph
        .matmul_region_kernel_name(0)
        .context("NN entry absent")?
        .to_owned();
    let module = stream.context().load_module(Ptx::from_src(&ptx))?;
    let function = module.load_function(&name)?;
    Ok(Built {
        function,
        _module: module,
        ptx,
        name,
    })
}
fn resources(f: &CudaFunction) -> Result<(Json, bool)> {
    let registers = f.num_regs()?;
    let shared = f.shared_size_bytes()?;
    let local = f.local_size_bytes()?;
    let maximum = f.max_threads_per_block()?;
    let capacity = f.occupancy_max_active_blocks_per_multiprocessor(256, 0, None)?;
    let admitted =
        registers > 0 && shared <= 40960 && local == 0 && maximum >= 256 && capacity >= 1;
    Ok((
        json!({"registers":registers,"shared_bytes":shared,"local_bytes":local,"maximum_threads":maximum,"APIcapacity256":capacity,"capacity_is_not_measured_occupancy":true,"one_CTA_diagnostic_admitted":admitted}),
        admitted,
    ))
}
fn select(old: &Kernel, new: &Kernel, admitted: bool) -> Result<(Kernel, bool)> {
    ensure!(
        old.pointers.len() == 3
            && new.pointers.len() == 3
            && old.pointers[..2] == new.pointers[..2]
            && old.pointers[2] == new.pointers[2],
        "whole same-input/output ABI required"
    );
    let output = old.pointers[2];
    ensure!(
        output != 0 && output.is_multiple_of(16),
        "whole output BF16 range/alignment proof"
    );
    ensure!(
        output.checked_add(u64::try_from(M * N * 2)?).is_some(),
        "output pointer range overflow"
    );
    for (&p, len) in old.pointers[..2].iter().zip([M * K, K * N]) {
        ensure!(p != 0 && p.is_multiple_of(16), "input whole BF16 alignment");
        let end = p
            .checked_add(u64::try_from(len * 2)?)
            .context("input range overflow")?;
        ensure!(
            output + u64::try_from(M * N * 2)? <= p || end <= output,
            "output aliases input"
        );
    }
    if !admitted || !output.is_multiple_of(128) {
        return Ok((old.clone(), false));
    }
    Ok((new.clone(), true))
}
fn capture(stream: &Arc<CudaStream>, k: &Kernel) -> Result<CudaGraph> {
    stream.synchronize()?;
    stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
    let launched = k.run(stream);
    let ended = stream.end_capture(
        sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
    );
    launched?;
    ended?.context("empty graph")
}
fn bits(a: &[bf16], b: &[bf16]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
}
fn compare(a: &[bf16], b: &[bf16]) -> Json {
    assert_eq!(a.len(), b.len());
    let mut different = 0;
    let mut examples = Vec::new();
    for (i, (a, b)) in a.iter().zip(b).enumerate() {
        if a.to_bits() != b.to_bits() && !(a.is_nan() && b.is_nan()) {
            different += 1;
            if examples.len() < 16 {
                examples
                    .push(json!({"index":i,"wmma_bits":a.to_bits(),"ldmatrix_bits":b.to_bits()}));
            }
        }
    }
    json!({"elements":a.len(),"different_nonNaN_bits_or_NaN_classes":different,"examples":examples,"NaN_payload_identity_promised":false})
}
fn feed(phase: &str) -> (Vec<bf16>, Vec<bf16>) {
    let mut a = vec![bf16::ZERO; M * K];
    let mut b = vec![bf16::ZERO; K * N];
    let values = |n, seed| {
        (0..n)
            .map(|i| bf16::from_f32(((i % 97 * seed % 97) as f32 - 48.) / 97.))
            .collect::<Vec<_>>()
    };
    match phase {
        "finite" => return (values(M * K, 13usize), values(K * N, 17usize)),
        "changed" => return (values(M * K, 23usize), values(K * N, 31usize)),
        "zero" => b.fill(bf16::ONE),
        "rounding_boundary" => {
            for row in a.chunks_exact_mut(K) {
                row[0] = bf16::ONE;
                row[1] = bf16::from_f32(1. / 256.);
                row[2] = bf16::from_f32(1. / 65536.);
            }
            b[..2 * N].fill(bf16::ONE);
            for (col, v) in b[2 * N..3 * N].iter_mut().enumerate() {
                if col % 2 == 1 {
                    *v = bf16::ONE;
                }
            }
        }
        "cancellation" => {
            for row in a.chunks_exact_mut(K) {
                row[0] = bf16::from_f32(1e20);
                row[1] = bf16::from_f32(-1e20);
                row[16] = bf16::ONE;
                row[32] = bf16::from_f32(1. / 256.);
            }
            b.fill(bf16::ONE);
        }
        "subnormal" => {
            a.fill(bf16::from_bits(1));
            b.fill(bf16::ONE);
        }
        "signed_zero" => {
            for (row, v) in a.chunks_exact_mut(K).enumerate() {
                v.fill(bf16::from_bits(if row % 2 == 0 { 0x8000 } else { 0 }));
            }
            for (col, v) in b.iter_mut().enumerate() {
                *v = bf16::from_f32(if col % 2 == 0 { 1. } else { -1. });
            }
        }
        "nonfinite" => {
            a.fill(bf16::ONE);
            for (i, v) in b.iter_mut().enumerate() {
                *v = bf16::from_bits([0x7f80, 0xff80, 0x7fc1, 0x7f81][i % N % 4]);
            }
        }
        _ => unreachable!("fixed phases"),
    }
    (a, b)
}
fn f64_sanity(a: &[bf16], b: &[bf16], got: &[bf16]) -> Json {
    let mut samples = Vec::new();
    for i in 0..32 {
        let row = (i * 541) % M;
        let col = (i * 37) % N;
        let exact = (0..K)
            .map(|k| f64::from(a[row * K + k].to_f32()) * f64::from(b[k * N + col].to_f32()))
            .sum::<f64>();
        let value = f64::from(got[row * N + col].to_f32());
        samples.push(json!({"row":row,"column":col,"f64_dot":exact,"bf16_result":value,"absolute_error":(value-exact).abs()}));
    }
    json!({"scope":"Descriptive32 F64 dot samples, separate from mandatory full baseline bit equality; no fitted tolerance gate.","samples":samples})
}
fn write(p: &Path, value: &Json) -> Result<()> {
    std::fs::write(p, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}
#[test]
#[ignore = "manual distinct one-CTA NN diagnostic; original strict resource policy unchanged"]
fn nn_one_cta_same_rejected_ptx_fullbits_replay_readonly_whole_wmma_fallback() -> Result<()> {
    if !crate::ptx::mixed::tests::cuda_available() {
        return Ok(());
    }
    let directory = std::path::PathBuf::from(
        std::env::var_os("TNSR_NN_ONE_CTA_AUDIT").context("set fresh artifact directory")?,
    );
    std::fs::create_dir_all(&directory)?;
    ensure!(
        !directory.join("report.json").exists(),
        "preserve existing NN artifacts"
    );
    let context = CudaContext::new(0)?;
    let stream = context.new_stream()?;
    let target = PtxTarget::from_context(&context)?;
    let mut a = stream.alloc_zeros::<bf16>(M * K)?;
    let mut b = stream.alloc_zeros::<bf16>(K * N)?;
    let old_output = stream.alloc_zeros::<bf16>(M * N)?;
    let new_output = stream.alloc_zeros::<bf16>(M * N)?;
    let graph = graph();
    let inputs = graph
        .graph
        .node_indices()
        .filter_map(|node| {
            if let TensorGraphNode::Input { name, .. } = &graph[node] {
                Some((
                    node,
                    Binding {
                        pointer: if *name == "dy" {
                            a.device_ptr(&stream).0
                        } else {
                            b.device_ptr(&stream).0
                        },
                        len: if *name == "dy" { M * K } else { K * N },
                        dtype: DType::BF16,
                    },
                ))
            } else {
                None
            }
        })
        .collect();
    let baseline = Program::<BF16>::new_bound_with_plan_metadata(
        &stream,
        graph,
        inputs,
        Some(Binding {
            pointer: old_output.device_ptr(&stream).0,
            len: M * N,
            dtype: DType::BF16,
        }),
        crate::tile::MatMulPrecision::Tf32x3,
        ProgramKernelOptions {
            shared_memory_reuse: true,
            plain_bf16_output: true,
            medium_bf16_tile_policy: crate::tile::MediumBf16TilePolicy::SquareThenTall,
            ..ProgramKernelOptions::default()
        },
    )?;
    ensure!(
        baseline.kernels.len() == 1 && baseline.plain_bf16_output_counts == (1, 0),
        "strong accepted64 plain-output baseline required"
    );
    let metadata = baseline
        .metadata
        .as_ref()
        .context("baseline typed plan absent")?;
    let [r] = metadata.plan.matmul_regions() else {
        anyhow::bail!("one baseline typed region");
    };
    guard(r)?;
    let old = &baseline.kernels[0];
    ensure!(
        old.config.grid_dim == (8, 256, 1)
            && old.config.block_dim == (32, 8, 1)
            && old.config.shared_mem_bytes == 0
            && old.zero_before.is_none(),
        "complete accepted64 NN launch"
    );
    let inputs = [a.device_ptr(&stream).0, b.device_ptr(&stream).0];
    ensure!(
        old.pointers == [inputs[0], inputs[1], old_output.device_ptr(&stream).0],
        "baseline physical3-pointer ABI changed"
    );
    let new = build(&stream, r, target)?;
    let rejected_dir = std::env::var_os("TNSR_NN_STRICT_REJECTED_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::PathBuf::from(
                "target/paper-kernels/nn-bf16-ldmatrix-qualification/device-v1",
            )
        });
    ensure!(
        new.ptx == std::fs::read_to_string(rejected_dir.join("candidate.ptx"))?,
        "one-CTA diagnostic must use the complete original rejected candidate PTX"
    );
    ensure!(
        baseline.ptx == std::fs::read_to_string(rejected_dir.join("baseline.ptx"))?,
        "accepted64 baseline module changed"
    );
    let original_rejection: Json =
        serde_json::from_slice(&std::fs::read(rejected_dir.join("report.json"))?)?;
    ensure!(
        original_rejection["status"] == "RESOURCE_REJECTED_NO_CANDIDATE_EXECUTION",
        "preserve the strict rejection status"
    );
    write(
        &directory.join("original-module-identities.json"),
        &json!({"candidate_complete_PTX_byte_identical":true,"baseline_complete_PTX_byte_identical":true,"original_strict_rejection":original_rejection,"scope":"Separate hardware-valid one-CTA diagnostic only; original production selector and strict policy are unchanged."}),
    )?;

    let (old_resources, old_ok) = resources(&old.function)?;
    let (new_resources, admitted) = resources(&new.function)?;
    std::fs::write(directory.join("baseline.ptx"), &baseline.ptx)?;
    std::fs::write(directory.join("candidate.ptx"), &new.ptx)?;
    write(
        &directory.join("resources.json"),
        &json!({"baseline":old_resources,"candidate":new_resources,"actual_rank3_A_source":r.inputs[0].source_shape,"baseline_plan":baseline.plan}),
    )?;
    let new_kernel = Kernel {
        owner: None,
        label: new.name,
        function: new.function.clone(),
        config: LaunchConfig {
            grid_dim: (4, 128, 1),
            block_dim: (32, 8, 1),
            shared_mem_bytes: 0,
        },
        pointers: vec![inputs[0], inputs[1], new_output.device_ptr(&stream).0],
        zero_before: None,
    };
    // Selection must retain a COMPLETE function/config/ABI tuple on rejection.
    let old_same_output = Kernel {
        pointers: vec![inputs[0], inputs[1], new_output.device_ptr(&stream).0],
        ..old.clone()
    };
    let (rejected, selected) = select(&old_same_output, &new_kernel, false)?;
    assert!(!selected);
    assert_eq!(
        super::super::function_identity(&rejected.function)?,
        super::super::function_identity(&old.function)?
    );
    assert_eq!(rejected.config.grid_dim, (8, 256, 1));
    if !admitted || !old_ok {
        write(
            &directory.join("report.json"),
            &json!({"status":"ONE_CTA_DIAGNOSTIC_HARDWARE_REJECTED_NO_CANDIDATE_EXECUTION","baseline":old_resources,"candidate":new_resources,"complete_old_wmma_fallback":true,"timing_performed":false,"production_changed":false}),
        )?;
        anyhow::bail!("one-CTA diagnostic hardware admission failed; no candidate execution");
    }
    let (chosen, selected) = select(&old_same_output, &new_kernel, admitted)?;
    ensure!(selected, "positive output128B alignment required");
    let fallback_output = stream.alloc_zeros::<bf16>(M * N + 8)?;
    let suffix = fallback_output
        .device_ptr(&stream)
        .0
        .checked_add(16)
        .context("suffix overflow")?;
    ensure!(
        suffix.is_multiple_of(16) && !suffix.is_multiple_of(128),
        "misalignment fixture"
    );
    let fallback_old = Kernel {
        pointers: vec![inputs[0], inputs[1], suffix],
        ..old.clone()
    };
    let fallback_new = Kernel {
        pointers: fallback_old.pointers.clone(),
        ..new_kernel.clone()
    };
    let (fallback, selected) = select(&fallback_old, &fallback_new, admitted)?;
    ensure!(!selected, "unaligned output must use whole old WMMA");
    assert_eq!(
        super::super::function_identity(&fallback.function)?,
        super::super::function_identity(&old.function)?
    );
    assert_eq!(fallback.config.grid_dim, old.config.grid_dim);
    assert_eq!(fallback.config.block_dim, old.config.block_dim);
    assert_eq!(fallback.pointers, fallback_old.pointers);
    let (aa, bb) = feed("finite");
    stream.memcpy_htod(&aa, &mut a)?;
    stream.memcpy_htod(&bb, &mut b)?;
    old.run(&stream)?;
    fallback.run(&stream)?;
    stream.synchronize()?;
    let expected = stream.memcpy_dtov(&old_output)?;
    let got = stream.memcpy_dtov(&fallback_output)?;
    ensure!(
        got[..8].iter().all(|v| v.to_bits() == 0),
        "fallback prefix corrupted"
    );
    let fallback_bits = compare(&expected, &got[8..]);
    ensure!(
        fallback_bits["different_nonNaN_bits_or_NaN_classes"] == 0,
        "whole WMMA suffix differs"
    );
    write(
        &directory.join("whole-wmma-fallback.json"),
        &json!({"resource_rejection_whole_function_and_launch_retained":true,"output_mod128":suffix%128,"valid16B_suffix":true,"candidate_executed_on_unaligned_output":false,"prefix_guard_intact":true,"bits":fallback_bits}),
    )?;
    let old_graph = capture(&stream, old)?;
    let new_graph = capture(&stream, &chosen)?;
    let mut phases = Vec::new();
    let mut failed = false;
    for phase in [
        "finite",
        "changed",
        "zero",
        "rounding_boundary",
        "cancellation",
        "subnormal",
        "signed_zero",
        "nonfinite",
    ] {
        let (aa, bb) = feed(phase);
        stream.memcpy_htod(&aa, &mut a)?;
        stream.memcpy_htod(&bb, &mut b)?;
        let mut replays = Vec::new();
        let mut f64 = Json::Null;
        for replay in 0..2 {
            old_graph.launch()?;
            new_graph.launch()?;
            stream.synchronize()?;
            let expected = stream.memcpy_dtov(&old_output)?;
            let actual = stream.memcpy_dtov(&new_output)?;
            let comparison = compare(&expected, &actual);
            if comparison["different_nonNaN_bits_or_NaN_classes"] != 0 {
                failed = true;
            }
            if phase == "finite" && replay == 0 {
                f64 = f64_sanity(&aa, &bb, &actual);
            }
            if phase == "rounding_boundary" {
                ensure!(
                    actual.iter().enumerate().all(|(i, v)| v.to_bits()
                        == bf16::from_f32(if (i % N).is_multiple_of(2) {
                            1.
                        } else {
                            1. + 1. / 128.
                        })
                        .to_bits()),
                    "plain BF16 single-RN boundary oracle failed"
                );
            }
            if phase == "zero" {
                ensure!(actual.iter().all(|v| v.to_bits() == 0), "plain zero oracle");
            }
            replays.push(comparison);
        }
        let readonly = bits(&aa, &stream.memcpy_dtov(&a)?) && bits(&bb, &stream.memcpy_dtov(&b)?);
        ensure!(readonly, "NN source arrays modified");
        phases.push(json!({"phase":phase,"replays":replays,"inputs_bits_unchanged":readonly,"F64_descriptive_sanity":f64}));
        write(
            &directory.join("phases.json"),
            &json!({"phases":phases,"any_nonNaN_bit_or_class_difference":failed}),
        )?;
    }
    stream.synchronize()?;
    write(
        &directory.join("report.json"),
        &json!({"status":if failed{"ONE_CTA_DIAGNOSTIC_NUMERICAL_MISMATCH"}else{"ONE_CTA_DIAGNOSTIC_HARDWARE_ADMITTED_AND_EXECUTED_EXACT"},"shape":[M,N,K],"baseline":old_resources,"candidate":new_resources,"phases":phases,"whole_wmma_fallback_proved":true,"timing_performed":false,"production_changed":false,"synthetic_zero_bias_inserted":false,"production_strict_policy_changed":false,"diagnostic_policy":"hardware-valid256,positive_register_count,zero_local,shared<=40960,APIcapacity>=1; no register forcing or geometry change"}),
    )?;
    ensure!(
        !failed,
        "NN full-bit/nonfinite comparison differs; preserved raw outputs, no tolerance relaxation"
    );
    Ok(())
}

#[test]
#[ignore = "manual CPU full-module identity proof requires immutable strict-rejection PTX; no GPU"]
fn nn_one_cta_complete_candidate_ptx_identical_to_strict_rejection() -> Result<()> {
    let mut target = PtxTarget::sm80();
    target.compute_capability = (8, 9);
    target.matmul_resources = crate::tile::MatMulResources {
        shared_bytes_per_sm: 100 * 1024,
        threads_per_sm: 1536,
        ..crate::tile::MatMulResources::default()
    };
    let r = fixture(target)?;
    let ptx = source(&r, target, candidate(&r, target)?)?;
    let old = std::path::Path::new(
        "target/paper-kernels/nn-bf16-ldmatrix-qualification/device-v1/candidate.ptx",
    );
    assert_eq!(
        ptx,
        std::fs::read_to_string(old)?,
        "no schedule/geometry/math/load/store/directive changes for one-CTA diagnostic"
    );
    Ok(())
}
