//! Untimed new normalized backend; both scalar/coalesced epilogues plus whole-WMMA alignment fallback.
use super::*;
use cudarc::{
    driver::{CudaFunction, CudaGraph, CudaModule, CudaStream, DevicePtr, LaunchConfig, sys},
    nvrtc::Ptx,
};
use half::bf16;
use serde_json::{Value, json};
use std::{path::Path, sync::Arc};

struct Built {
    function: CudaFunction,
    _module: Arc<CudaModule>,
    source: String,
    name: String,
}
fn build(
    stream: &Arc<CudaStream>,
    r: &MatMulRegion,
    body: TileIR,
    target: PtxTarget,
) -> Result<Built> {
    let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
    tiles.add_matmul_regions(std::slice::from_ref(r))?;
    tiles.matmul_region_kernels[0].ir = body;
    tiles.optimize_indices()?;
    let graph = PtxGraph::from(tiles).with_target(target);
    let source = graph.module_source();
    let name = graph
        .matmul_region_kernel_name(0)
        .context("missing entry")?
        .to_owned();
    let module = stream.context().load_module(Ptx::from_src(&source))?;
    let function = module.load_function(&name)?;
    Ok(Built {
        function,
        _module: module,
        source,
        name,
    })
}
fn resources(b: &Built) -> Result<(Value, bool)> {
    let registers = b.function.num_regs()?;
    let shared = b.function.shared_size_bytes()?;
    let local = b.function.local_size_bytes()?;
    let max_threads = b.function.max_threads_per_block()?;
    let capacity = b
        .function
        .occupancy_max_active_blocks_per_multiprocessor(256, 0, None)?;
    let admitted = (1..=128).contains(&registers)
        && shared <= 40960
        && local == 0
        && max_threads >= 256
        && capacity >= 2;
    Ok((
        json!({"entry":b.name,"registers":registers,"shared_bytes":shared,"local_bytes":local,"maximum_threads":max_threads,"block_threads":256,"api_active_block_capacity":capacity,"capacity_is_not_measured_occupancy":true,"strict_admitted":admitted}),
        admitted,
    ))
}
fn capture(stream: &Arc<CudaStream>, kernel: &Kernel) -> Result<CudaGraph> {
    stream.synchronize()?;
    stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
    let recording = kernel.run(stream);
    let ended = stream.end_capture(
        sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
    );
    recording?;
    ended?.context("empty graph")
}
fn same_input(a: &[bf16], b: &[bf16]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}
fn values(len: usize, seed: usize) -> Vec<bf16> {
    (0..len)
        .map(|i| bf16::from_f32(((i % 97 * seed % 97) as f32 - 48.) / 97.))
        .collect()
}
fn feed(phase: &str, m: usize, n: usize) -> (Vec<bf16>, Vec<bf16>, Vec<bf16>) {
    let mut a = vec![bf16::ZERO; m * 512];
    let mut b = vec![bf16::ZERO; n * 512];
    let mut bias = vec![bf16::ZERO; n];
    match phase {
        "finite" => return (values(a.len(), 13), values(b.len(), 17), values(n, 19)),
        "changed" => return (values(a.len(), 23), values(b.len(), 31), values(n, 37)),
        "bias_once" => {
            b.fill(bf16::ONE);
            bias.fill(bf16::from_f32(1.5));
        }
        "rounding_boundary" => {
            for row in a.chunks_exact_mut(512) {
                row[0] = bf16::ONE;
                row[1] = bf16::from_f32(1. / 256.);
            }
            for row in b.chunks_exact_mut(512) {
                row[0] = bf16::ONE;
                row[1] = bf16::ONE;
            }
            bias.fill(bf16::from_f32(1. / 256.));
        }
        "cancellation" => {
            for row in a.chunks_exact_mut(512) {
                row[0] = bf16::from_f32(1e20);
                row[1] = bf16::from_f32(-1e20);
                row[16] = bf16::ONE;
                row[32] = bf16::from_f32(1. / 256.);
            }
            b.fill(bf16::ONE);
            bias.fill(bf16::from_f32(1. / 256.));
        }
        "subnormal" => {
            a.fill(bf16::from_bits(1));
            b.fill(bf16::ONE);
            bias.fill(bf16::from_bits(1));
        }
        "signed_zero" => {
            for (i, row) in a.chunks_exact_mut(512).enumerate() {
                row.fill(bf16::from_bits(if i % 2 == 0 { 0x8000 } else { 0 }));
            }
            b.fill(bf16::ONE);
            for (i, v) in bias.iter_mut().enumerate() {
                *v = bf16::from_bits(if i % 2 == 0 { 0x8000 } else { 0 });
            }
        }
        "nonfinite" => {
            a.fill(bf16::ONE);
            for (i, row) in b.chunks_exact_mut(512).enumerate() {
                row.fill(bf16::from_bits([0x7f80, 0xff80, 0x7fc1, 0x7f81][i % 4]));
            }
        }
        _ => unreachable!("fixed qualification phases"),
    }
    (a, b, bias)
}
fn comparison(expected: &[bf16], actual: &[bf16]) -> Value {
    assert_eq!(expected.len(), actual.len());
    let mut different = 0usize;
    let mut examples = Vec::new();
    for (i, (a, b)) in expected.iter().zip(actual).enumerate() {
        if a.to_bits() != b.to_bits() && !(a.is_nan() && b.is_nan()) {
            different += 1;
            if examples.len() < 16 {
                examples.push(json!({"index":i,"wmma_bits":a.to_bits(),"mma_bits":b.to_bits(),"wmma":format!("{a:?}"),"mma":format!("{b:?}")}));
            }
        }
    }
    json!({"elements":expected.len(),"different_nonNaN_bits_or_NaN_classes":different,"examples":examples,"NaN_payload_identity_promised":false})
}
fn write(path: &Path, value: &Value) -> Result<()> {
    std::fs::write(path, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}
#[test]
#[ignore = "manual untimed normalized-backend BF16 MMA resource/bit/fallback proof; no production activation"]
fn normalized_backend_two_nt_shapes_both_epilogues_resources_bits_replay_readonly_and_fallback()
-> Result<()> {
    let directory = std::path::PathBuf::from(
        std::env::var_os("TNSR_NORMALIZED_BF16_MMA_DIR").context("set fresh artifact directory")?,
    );
    std::fs::create_dir_all(&directory)?;
    ensure!(
        !directory.join("report.json").exists(),
        "preserve existing report"
    );
    let context = CudaContext::new(0)?;
    let stream = context.new_stream()?;
    let target = PtxTarget::from_context(&context)?;
    let mut reports = Vec::new();
    let mut failed = false;
    for (name, n, variant) in [
        ("ffn_scalar", 1408usize, false),
        ("qkv_scalar", 1536, false),
        ("ffn_coalesced", 1408, true),
        ("qkv_coalesced", 1536, true),
    ] {
        let m = 16384;
        let d = directory.join(name);
        std::fs::create_dir_all(&d)?;
        let r = fixture_for(m, n, target)?;
        let (old_ir, _, _) = ir::candidate(&r, target)?;
        let old = build(&stream, &r, old_ir, target)?;
        let new = build(&stream, &r, candidate(&r, target, variant, true)?, target)?;
        std::fs::write(d.join("baseline.ptx"), &old.source)?;
        std::fs::write(d.join("candidate.ptx"), &new.source)?;
        let (old_resource, old_admitted) = resources(&old)?;
        let (new_resource, admitted) = resources(&new)?;
        write(
            &d.join("resources.json"),
            &json!({"wmma":old_resource,"explicit_mma":new_resource}),
        )?;
        if !admitted || !old_admitted {
            failed = true;
            reports.push(json!({"case":name,"coalesced_epilogue":variant,"m":m,"n":n,"k":512,"status":"RESOURCE_REJECTED_NO_EXECUTION","wmma":old_resource,"explicit_mma":new_resource}));
            write(
                &directory.join("report.json"),
                &json!({"cases":reports,"production_changed":false,"timing_performed":false}),
            )?;
            continue;
        }
        let mut a = stream.alloc_zeros::<bf16>(m * 512)?;
        let mut b = stream.alloc_zeros::<bf16>(n * 512)?;
        let mut bias = stream.alloc_zeros::<bf16>(n)?;
        let old_output = stream.alloc_zeros::<bf16>(m * n)?;
        let new_output = stream.alloc_zeros::<bf16>(m * n)?;
        let config = LaunchConfig {
            grid_dim: (u32::try_from(n / 128)?, u32::try_from(m / 128)?, 1),
            block_dim: (32, 8, 1),
            shared_mem_bytes: 0,
        };
        let kernel = |built: &Built, output: u64| Kernel {
            owner: None,
            label: built.name.clone(),
            function: built.function.clone(),
            config,
            pointers: vec![
                a.device_ptr(&stream).0,
                b.device_ptr(&stream).0,
                bias.device_ptr(&stream).0,
                output,
            ],
            zero_before: None,
        };
        let old_kernel = kernel(&old, old_output.device_ptr(&stream).0);
        let new_kernel = kernel(&new, new_output.device_ptr(&stream).0);
        // Output128B alignment is a runtime preflight, not implied by a layout.
        let (guarded_body, selected) =
            prepare(&r, target, new_output.device_ptr(&stream).0, m * n, variant)?;
        ensure!(
            selected,
            "normalized positive output requires128B alignment"
        );
        ensure!(
            format!("{guarded_body:?}") == format!("{:?}", candidate(&r, target, variant, true)?),
            "preflight body differs from compiled candidate"
        );
        // A valid16B-aligned suffix of a larger allocation is deliberately not
        //128B-aligned. No explicit-MMA function is launched on this pointer.
        let fallback_output = stream.alloc_zeros::<bf16>(m * n + 8)?;
        let fallback_base = fallback_output.device_ptr(&stream).0;
        ensure!(
            fallback_base.is_multiple_of(128),
            "fallback test requires an aligned allocation base"
        );
        let fallback_pointer = fallback_base.checked_add(16).context("pointer overflow")?;
        ensure!(
            fallback_pointer.is_multiple_of(16) && !fallback_pointer.is_multiple_of(128),
            "misalignment fixture failed"
        );
        let (fallback_body, fallback_selected) =
            prepare(&r, target, fallback_pointer, m * n, variant)?;
        ensure!(
            !fallback_selected,
            "unaligned output selected normalized MMA"
        );
        let fallback = build(&stream, &r, fallback_body, target)?;
        ensure!(
            fallback.source == old.source,
            "whole WMMA module fallback changed"
        );
        let (fallback_resources, fallback_admitted) = resources(&fallback)?;
        ensure!(
            fallback_admitted && fallback_resources == old_resource,
            "whole WMMA resources differ"
        );
        let (aa, bb, bi) = feed("finite", m, n);
        stream.memcpy_htod(&aa, &mut a)?;
        stream.memcpy_htod(&bb, &mut b)?;
        stream.memcpy_htod(&bi, &mut bias)?;
        let fallback_kernel = Kernel {
            owner: None,
            label: "whole WMMA fallback for16B-not128B output".into(),
            function: fallback.function.clone(),
            config,
            pointers: vec![
                a.device_ptr(&stream).0,
                b.device_ptr(&stream).0,
                bias.device_ptr(&stream).0,
                fallback_pointer,
            ],
            zero_before: None,
        };
        old_kernel.run(&stream)?;
        fallback_kernel.run(&stream)?;
        stream.synchronize()?;
        let expected = stream.memcpy_dtov(&old_output)?;
        let got = stream.memcpy_dtov(&fallback_output)?;
        ensure!(
            got[..8].iter().all(|v| v.to_bits() == 0),
            "fallback prefix overwritten"
        );
        let fallback_bits = comparison(&expected, &got[8..]);
        ensure!(
            fallback_bits["different_nonNaN_bits_or_NaN_classes"].as_u64() == Some(0),
            "unaligned WMMA fallback output differs"
        );
        let fallback_readonly = same_input(&aa, &stream.memcpy_dtov(&a)?)
            && same_input(&bb, &stream.memcpy_dtov(&b)?)
            && same_input(&bi, &stream.memcpy_dtov(&bias)?);
        ensure!(fallback_readonly, "fallback modified inputs");
        let fallback_proof = json!({"pointer_mod128":fallback_pointer%128,"pointer_mod16":fallback_pointer%16,"normalized_selected":false,"whole_wmma_ptx_and_resources_identical":true,"normalized_function_launched_on_misaligned_output":false,"prefix_guard_intact":true,"output":fallback_bits,"inputs_readonly":fallback_readonly});
        write(&d.join("unaligned-output-fallback.json"), &fallback_proof)?;
        let old_graph = capture(&stream, &old_kernel)?;
        let new_graph = capture(&stream, &new_kernel)?;
        let mut phases = Vec::new();
        for phase in [
            "finite",
            "changed",
            "bias_once",
            "rounding_boundary",
            "cancellation",
            "subnormal",
            "signed_zero",
            "nonfinite",
        ] {
            let (aa, bb, bi) = feed(phase, m, n);
            stream.memcpy_htod(&aa, &mut a)?;
            stream.memcpy_htod(&bb, &mut b)?;
            stream.memcpy_htod(&bi, &mut bias)?;
            let mut replays = Vec::new();
            for _ in 0..2 {
                old_graph.launch()?;
                new_graph.launch()?;
                stream.synchronize()?;
                let expected = stream.memcpy_dtov(&old_output)?;
                let actual = stream.memcpy_dtov(&new_output)?;
                let result = comparison(&expected, &actual);
                if result["different_nonNaN_bits_or_NaN_classes"].as_u64() != Some(0) {
                    failed = true;
                }
                if phase == "bias_once" {
                    ensure!(
                        actual
                            .iter()
                            .all(|v| v.to_bits() == bf16::from_f32(1.5).to_bits()),
                        "bias once oracle failed"
                    );
                }
                if phase == "rounding_boundary" {
                    ensure!(
                        actual
                            .iter()
                            .all(|v| v.to_bits() == bf16::from_f32(1. + 1. / 128.).to_bits()),
                        "BF16 output rounding oracle failed"
                    );
                }
                replays.push(result);
            }
            let readonly = same_input(&aa, &stream.memcpy_dtov(&a)?)
                && same_input(&bb, &stream.memcpy_dtov(&b)?)
                && same_input(&bi, &stream.memcpy_dtov(&bias)?);
            ensure!(readonly, "input storage changed");
            phases.push(
                json!({"phase":phase,"replays":replays,"all_inputs_bits_unchanged":readonly}),
            );
            write(&d.join("phases.json"), &json!({"phases":phases}))?;
        }
        stream.synchronize()?;
        reports.push(json!({"case":name,"coalesced_epilogue":variant,"m":m,"n":n,"k":512,"status":"RESOURCE_ADMITTED_NUMERICS_RECORDED","wmma":old_resource,"explicit_mma":new_resource,"phases":phases,"unaligned_output_fallback":fallback_proof}));
        write(
            &directory.join("report.json"),
            &json!({"cases":reports,"production_changed":false,"timing_performed":false,"any_resource_or_numeric_failure":failed}),
        )?;
    }
    ensure!(
        !failed,
        "normalized-backend MMA resource gate or exact output comparison failed; see preserved report; no tolerance relaxation"
    );
    Ok(())
}
