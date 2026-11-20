//! Direct old/new partial-kernel tests, including every F32 partial.
use super::*;
use crate::{
    ptx::{PtxGraph, PtxTarget},
    tile::{TileGraph, TileIR},
};
use cudarc::{
    driver::{CudaContext, CudaFunction, CudaModule, CudaStream, LaunchConfig, PushKernelArg},
    nvrtc::Ptx,
};
use half::bf16;
use std::sync::Arc;

struct Built {
    function: CudaFunction,
    _module: Arc<CudaModule>,
    ptx: String,
}
fn compile(stream: &Arc<CudaStream>, ir: TileIR) -> anyhow::Result<Built> {
    let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
    tiles.region_kernels = vec![crate::tile::graph::TileRegionKernel { region_id: 0, ir }];
    tiles.optimize_indices()?;
    let graph = PtxGraph::from(tiles).with_target(PtxTarget::from_context(stream.context())?);
    let name = graph.region_kernel_name(0).unwrap();
    let ptx = graph.module_source();
    let module = stream.context().load_module(Ptx::from_src(&ptx))?;
    let function = module.load_function(name)?;
    Ok(Built {
        function,
        _module: module,
        ptx,
    })
}
fn baseline_ir(region: &MatMulRegion) -> anyhow::Result<TileIR> {
    let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
    tiles.add_matmul_regions(std::slice::from_ref(region))?;
    let mut ir = tiles.matmul_region_kernels.remove(0).ir;
    for p in &mut ir.params {
        if !p.is_input {
            p.dtype = DType::F32;
        }
    }
    Ok(ir)
}
fn same(a: f32, b: f32) {
    assert!(
        (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits(),
        "{a:?}vs{b:?}"
    );
}
fn left_four(v: &[f32], index: usize, plane: usize) -> bf16 {
    bf16::from_f32(((v[index] + v[plane + index]) + v[2 * plane + index]) + v[3 * plane + index])
}

#[test]
fn split_square64_compares_every_f32_partial_before_final_bf16_and_changed_feeds()
-> anyhow::Result<()> {
    let available = CudaContext::new(0).is_ok();
    assert!(
        available || std::env::var_os("TNSR_REQUIRE_CUDA").is_none(),
        "CUDA required"
    );
    if !available {
        return Ok(());
    }
    let device = CudaContext::new(0)?;
    let stream = device.new_stream()?;
    let target = PtxTarget::from_context(&device)?;
    for (m, n, part_k) in [
        (256, 512, 512),
        (512, 1408, 4096),
        (1408, 512, 4096),
        (512, 1536, 4096),
        (512, 4096, 4096),
    ] {
        let r = fixture(m, n, part_k);
        let c = SplitKBf16TilePolicy::Square64
            .candidate(&r, 4, 4 * part_k, target.matmul_resources, true)
            .unwrap();
        let (schedule, ir) = c.partial_ir(&r, target.matmul_resources, true).unwrap();
        let old = compile(&stream, baseline_ir(&r)?)?;
        let new = compile(&stream, ir)?;
        assert_eq!(new.function.shared_size_bytes()?, 18432);
        assert_eq!(new.function.local_size_bytes()?, 0);
        assert!(new.function.num_regs()? <= 128);
        assert!(new.function.max_threads_per_block()? >= 256);
        let old_config = LaunchConfig {
            grid_dim: (u32::try_from(n / 32)?, u32::try_from(m / 32)?, 4),
            block_dim: (32, 8, 1),
            shared_mem_bytes: 0,
        };
        let new_config = LaunchConfig {
            grid_dim: (u32::try_from(n / 64)?, u32::try_from(m / 64)?, 4),
            block_dim: schedule.block_threads,
            shared_mem_bytes: 0,
        };
        let total_k = 4 * part_k;
        let plane = m * n;
        let mut left = stream.alloc_zeros::<bf16>(total_k * m)?;
        let mut right = stream.alloc_zeros::<bf16>(total_k * n)?;
        let mut old_partial = stream.alloc_zeros::<f32>(4 * plane)?;
        let mut new_partial = stream.alloc_zeros::<f32>(4 * plane)?;
        if let Some(directory) = std::env::var_os("TNSR_SPLIT_K_SQUARE_PTX_DIR") {
            std::fs::create_dir_all(&directory)?;
            let directory = std::path::PathBuf::from(directory);
            for (name, kernel) in [("old", &old), ("square64", &new)] {
                std::fs::write(
                    directory.join(format!("{m}-{n}-{part_k}-{name}.ptx")),
                    &kernel.ptx,
                )?;
                std::fs::write(
                    directory.join(format!("{m}-{n}-{part_k}-{name}-resources.json")),
                    serde_json::to_vec_pretty(
                        &serde_json::json!({"registers":kernel.function.num_regs()?,"shared":kernel.function.shared_size_bytes()?,"local":kernel.function.local_size_bytes()?,"static_cuda_block_capacity":kernel.function.occupancy_max_active_blocks_per_multiprocessor(256,0,None)?}),
                    )?,
                )?;
            }
        }
        for phase in 0..3 {
            let mut a: Vec<_> = (0..total_k * m)
                .map(|i| bf16::from_f32(((i + phase * 3) % 23) as f32 / 64. - 0.125))
                .collect();
            let mut b: Vec<_> = (0..total_k * n)
                .map(|i| bf16::from_f32(((i + phase) % 19) as f32 / 32. - 0.25))
                .collect();
            if phase == 1 {
                a.fill(bf16::from_f32(-0.));
                b.fill(bf16::from_f32(1.));
            }
            if phase == 2 {
                a[0] = bf16::INFINITY;
                b[0] = bf16::ZERO;
                a[m] = bf16::NAN;
                b[n] = bf16::NEG_INFINITY;
            }
            stream.memcpy_htod(&a, &mut left)?;
            stream.memcpy_htod(&b, &mut right)?;
            for (kernel, config, output) in [
                (&old, old_config, &mut old_partial),
                (&new, new_config, &mut new_partial),
            ] {
                let mut launch = stream.launch_builder(&kernel.function);
                launch.arg(&left).arg(&right).arg(output);
                // SAFETY: test owns two exact physical BF16 inputs and distinct
                // [4,M,N]F32 output, retained through synchronized readback.
                unsafe { launch.launch(config) }?;
            }
            stream.synchronize()?;
            let a = stream.memcpy_dtov(&old_partial)?;
            let b = stream.memcpy_dtov(&new_partial)?;
            // Critical gate: do NOT compare only narrowed final weights. A BF16
            // round could hide an incorrect F32 partial or partition boundary.
            assert_eq!(a.len(), 4 * plane);
            assert_eq!(b.len(), 4 * plane);
            for (&a, &b) in a.iter().zip(&b) {
                same(a, b);
            }
            for index in 0..plane {
                let expected = left_four(&a, index, plane);
                let actual = left_four(&b, index, plane);
                assert!(
                    (expected.is_nan() && actual.is_nan())
                        || expected.to_bits() == actual.to_bits()
                );
            }
        }
    }
    Ok(())
}

// The final BF16 comparison above is a CPU contract oracle over identical full
// GPU F32 partials; it is NOT a substitute for the unchanged production reducer.
// Required future integration tests use the actual Mixed path and its existing
// sum kernel, plus replay/resident invalid-update recovery/full fixed model gates.

#[path = "rectangle128_gpu.rs"]
mod rectangle128_gpu;
