//! One test-only four-warp wide-exchange fused NT128 geometry. No production selector/IR/API changes.
use super::*;
use crate::{
    tensor::BinaryOp,
    tile::{
        Block, MatMulAccumulatorExchange, MatMulFragmentLoad, MatMulPipeline, MatMulRegion,
        MatMulResources, MatMulSchedule, MatMulSharedLayout, MatMulTile, RegionOpKind,
        ScalarBinaryOp, ScalarExpr, ScalarMemory, ScalarPredicate, Stmt, TileIR, TileVar,
        VirtualTensor,
    },
};
mod ir;
mod tests;

fn region(c: Case, target: PtxTarget) -> Result<MatMulRegion> {
    let a = TensorExpr::<bf16>::input("a", vec![c.m, c.k]);
    let w = TensorExpr::<bf16>::input("b", vec![c.n, c.k]).transpose();
    let bias = TensorExpr::<bf16>::input("bias", vec![c.n])
        .reshape(vec![1, c.n])
        .broadcast(vec![c.m, c.n]);
    let graph: TensorGraph<bf16> = (a.matmul(w) + bias).into();
    // Use the actual planner's view rebasing (Transpose and bias broadcast),
    // rather than raw MatMulRegion::from_graph inputs, which are still virtual.
    let plan = crate::ptx::PtxExecutionPlan::build_with_reduction_mode(
        &graph,
        crate::ptx::PtxReductionMode::DeterministicTree,
    )?;
    ensure!(
        plan.matmul_regions().len() == 1,
        "one fused linear region required"
    );
    let mut r = plan.matmul_regions()[0].clone();
    r.schedule = MatMulSchedule::select_with_resources(
        c.m,
        c.n,
        c.k,
        DType::BF16,
        target.matmul_capabilities(),
        MatMulPrecision::Tf32x3,
        target.matmul_resources,
    )
    .with_shared_memory_reuse(true);
    r.configure_pipeline(
        MatMulSharedLayout::Padded,
        MatMulPipeline::Auto,
        target.supports_async_copy(),
        target.matmul_resources,
    )?;
    Ok(r)
}
fn equal(a: &[bf16], b: &[bf16]) -> Result<()> {
    ensure!(a.len() == b.len(), "output extents differ");
    for (i, (a, b)) in a.iter().zip(b).enumerate() {
        ensure!(
            a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()),
            "old/new output differs index{i}: {a:?}/{b:?}"
        );
    }
    Ok(())
}
fn capture_group(e: &MixedExecutor, kernels: &[Kernel]) -> Result<cudarc::driver::CudaGraph> {
    e.stream.synchronize()?;
    e.stream
        .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
    let recording = (|| -> Result<()> {
        for _ in 0..2 {
            submit(e, kernels)?;
        }
        Ok(())
    })();
    let ended = e.stream.end_capture(
        sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
    );
    recording?;
    ended?.context("empty probe capture")
}
#[test]
#[ignore = "manual strictly admitted ONEgeometry NT128 bias probe; no production selection/performance"]
fn fused_nt128_four_wide_actual_two_shapes_resource_bits_bias_and_replay() -> Result<()> {
    let input = PathBuf::from(
        std::env::var_os("TNSR_NON_NN_INPUTS").context("set qualified BF16 archive")?,
    );
    let directory = PathBuf::from(
        std::env::var_os("TNSR_FUSED_NT128_FOUR_WIDE_AUDIT")
            .context("set fresh NT128 audit directory")?,
    );
    std::fs::create_dir_all(&directory)?;
    ensure!(
        !directory.join("report.json").exists(),
        "preserve existing probe report"
    );
    let mut reports = Vec::new();
    let mut rejected = Vec::new();
    for c in CASES.into_iter().filter(|c| !c.split) {
        let d = directory.join(c.name);
        std::fs::create_dir_all(&d)?;
        let a = archive::read16(&input.join(c.name).join("a.bf16"), c.m * c.k)?;
        let b = archive::read16(&input.join(c.name).join("b.bf16"), c.k * c.n)?;
        let bias = archive::read16(&input.join(c.name).join("bias.bf16"), c.n)?;
        let (mut e, old, prefix) = construct(c, &a, &b, &bias)?;
        ensure!(
            old.len() == 1 && old[0].pointers.len() == 4,
            "actual strong-control ABI changed"
        );
        let target = PtxTarget::from_context(e.stream.context())?;
        let original = region(c, target)?;
        let (ir, schedule, _) = ir::candidate(&original, target)?;
        let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
        tiles.add_matmul_regions(&[original])?;
        tiles.matmul_region_kernels[0].ir = ir;
        tiles.optimize_indices()?;
        let ptx = PtxGraph::from(tiles).with_target(target);
        let source = ptx.module_source();
        std::fs::write(d.join("candidate.ptx"), &source)?;
        let module = e.stream.context().load_module(Ptx::from_src(source))?;
        let name = ptx
            .matmul_region_kernel_name(0)
            .context("candidate entry absent")?;
        let function = module.load_function(name)?;
        let registers = function.num_regs()?;
        let shared = function.shared_size_bytes()?;
        let local = function.local_size_bytes()?;
        let maximum = function.max_threads_per_block()?;
        let capacity = function.occupancy_max_active_blocks_per_multiprocessor(128, 0, None)?;
        let admitted = (1..=255).contains(&registers)
            && local == 0
            && shared == 40960
            && maximum >= 128
            && capacity >= 2;
        let resource = serde_json::json!({"entry":name,"registers":registers,"shared":shared,"local":local,"max_threads":maximum,"api_capacity":capacity,"api_capacity_is_not_measured_occupancy":true,"strict_admitted":admitted,"schedule":format!("{schedule:?}"),"ABI":["BF16Arow","BF16Wrow_with_logical_transpose","BF16bias_col","BF16output"],"production_selector_changed":false,"diagnostic_only_gate":{"max_registers":255,"min_api_capacity":2,"block_threads":128},"exchange_bytes":32768,"exchange_phases":2,"exchange_barriers":3});
        archive::json(&d.join("resources.json"), &resource)?;
        if !admitted {
            rejected.push(c.name);
            reports.push(serde_json::json!({"case":c.name,"status":"resource_rejected_no_candidate_execution","resources":resource}));
            continue;
        }
        let output = e.stream.alloc_zeros::<bf16>(c.m * c.n)?;
        let new = Kernel {
            owner: None,
            label: "test-only four-warp NT128 wide exchange single bias F32 before BF16".into(),
            function,
            config: LaunchConfig {
                grid_dim: (u32::try_from(c.n / 128)?, u32::try_from(c.m / 128)?, 1),
                block_dim: (32, 4, 1),
                shared_mem_bytes: 0,
            },
            pointers: vec![
                old[0].pointers[0],
                old[0].pointers[1],
                old[0].pointers[2],
                output.device_ptr(&e.stream).0,
            ],
            zero_before: None,
        };
        let old_graph = capture_group(&e, &old)?;
        let new_graph = capture_group(&e, std::slice::from_ref(&new))?;
        let mut phases = Vec::new();
        for phase in [
            "finite",
            "changed",
            "bias_once",
            "rounding_boundary",
            "positive_zero",
            "negative_zero",
            "infinity",
            "nan",
        ] {
            let (aa, bb, bi) = match phase {
                "finite" => (a.clone(), b.clone(), bias.clone()),
                "changed" => (
                    a.iter().map(|v| bf16::from_f32(-(*v).to_f32())).collect(),
                    b.clone(),
                    bias.iter()
                        .map(|v| bf16::from_f32((*v).to_f32() * 2.))
                        .collect(),
                ),
                "bias_once" => (
                    vec![bf16::ZERO; a.len()],
                    vec![bf16::ONE; b.len()],
                    vec![bf16::from_f32(1.5); c.n],
                ),
                "rounding_boundary" => {
                    let mut aa = vec![bf16::ZERO; a.len()];
                    let mut bb = vec![bf16::ZERO; b.len()];
                    for row in 0..c.m {
                        aa[row * c.k] = bf16::ONE;
                        aa[row * c.k + 1] = bf16::from_f32(1. / 256.);
                    }
                    for row in 0..c.n {
                        bb[row * c.k] = bf16::ONE;
                        bb[row * c.k + 1] = bf16::ONE;
                    }
                    (aa, bb, vec![bf16::from_f32(1. / 256.); c.n])
                }
                "positive_zero" => (
                    vec![bf16::ZERO; a.len()],
                    vec![bf16::ONE; b.len()],
                    vec![bf16::ZERO; c.n],
                ),
                "negative_zero" => (
                    vec![bf16::from_f32(-0.); a.len()],
                    vec![bf16::ONE; b.len()],
                    vec![bf16::from_f32(-0.); c.n],
                ),
                "infinity" => (
                    vec![bf16::ONE; a.len()],
                    vec![bf16::from_f32(f32::INFINITY); b.len()],
                    vec![bf16::ZERO; c.n],
                ),
                _ => (
                    vec![bf16::from_f32(f32::NAN); a.len()],
                    b.clone(),
                    bias.clone(),
                ),
            };
            e.upload(HashMap::from([
                ("a".into(), aa.iter().map(|v| (*v).to_f32()).collect()),
                ("b".into(), bb.iter().map(|v| (*v).to_f32()).collect()),
                ("bias".into(), bi.iter().map(|v| (*v).to_f32()).collect()),
            ]))?;
            for kernel in &e.kernels[..prefix] {
                kernel.run(&e.stream)?;
            }
            for _ in 0..2 {
                old_graph.launch()?;
                new_graph.launch()?;
                let baseline = read16(&e, old[0].pointers[3], c.m * c.n)?;
                let actual = e.stream.memcpy_dtov(&output)?;
                equal(&baseline, &actual)?;
                if phase == "bias_once" {
                    ensure!(
                        actual
                            .iter()
                            .all(|v| v.to_bits() == bf16::from_f32(1.5).to_bits()),
                        "bias added more than once"
                    );
                }
                if phase == "rounding_boundary" {
                    ensure!(
                        actual
                            .iter()
                            .all(|v| v.to_bits() == bf16::from_f32(1. + 1. / 128.).to_bits()),
                        "intermediate BF16 C rounding detected"
                    );
                }
            }
            phases.push(serde_json::json!({"phase":phase,"all_nonNaN_bits_and_NaNclasses_equal":true,"replays":2,"output_elements":c.m*c.n}));
        }
        reports.push(serde_json::json!({"case":c.name,"status":"strict_resource_and_full_output_pass","resources":resource,"phases":phases,"timing_performed":false}));
        archive::json(
            &directory.join("report.json"),
            &serde_json::json!({"cases":reports,"resource_rejected":rejected,"production_selector_changed":false}),
        )?;
        e.stream.synchronize()?;
        drop(new_graph);
        drop(old_graph);
        drop(output);
        drop(module);
    }
    archive::json(
        &directory.join("report.json"),
        &serde_json::json!({"cases":reports,"resource_rejected":rejected,"production_selector_changed":false}),
    )?;
    ensure!(
        rejected.is_empty(),
        "strict resource gate rejected ONEgeometry for {rejected:?}; no relaxed gate or geometry sweep"
    );
    Ok(())
}
