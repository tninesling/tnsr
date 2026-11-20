//! Direct-GPU qualification under the shared serialized CUDA slot. No timing.
use super::*;
use cudarc::{
    driver::{CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg},
    nvrtc::Ptx,
};
use half::bf16;
use std::sync::Arc;

struct Built {
    function: CudaFunction,
    _module: Arc<CudaModule>,
    ptx: String,
}
fn compile(stream: &Arc<CudaStream>, ir: TileIR) -> Result<Built> {
    let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
    tiles.region_kernels = vec![crate::tile::graph::TileRegionKernel { region_id: 0, ir }];
    tiles.optimize_indices()?;
    let graph = PtxGraph::from(tiles).with_target(PtxTarget::from_context(stream.context())?);
    let name = graph
        .region_kernel_name(0)
        .context("warp qualification kernel missing")?;
    let ptx = graph.module_source();
    let module = stream.context().load_module(Ptx::from_src(&ptx))?;
    let function = module.load_function(name)?;
    Ok(Built {
        function,
        _module: module,
        ptx,
    })
}
fn forward(
    stream: &Arc<CudaStream>,
    kernel: &Built,
    config: LaunchConfig,
    qk: &CudaSlice<bf16>,
    mask: &CudaSlice<f32>,
    p: &mut CudaSlice<f32>,
    p16: &mut CudaSlice<bf16>,
) -> Result<()> {
    let mut launch = stream.launch_builder(&kernel.function);
    launch.arg(qk).arg(mask).arg(p).arg(p16);
    // SAFETY: the test owns disjoint exact-extent typed buffers through stream
    // synchronization; ABI order is qk16,mask32,P32,P16 in both kernels.
    unsafe { launch.launch(config) }?;
    Ok(())
}
fn backward(
    stream: &Arc<CudaStream>,
    kernel: &Built,
    config: LaunchConfig,
    p: &CudaSlice<f32>,
    g: &CudaSlice<bf16>,
    score: &mut CudaSlice<bf16>,
) -> Result<()> {
    let mut launch = stream.launch_builder(&kernel.function);
    launch.arg(p).arg(g).arg(score);
    // SAFETY: same ownership/extent contract; ABI order P32,dP16,dScore16.
    unsafe { launch.launch(config) }?;
    Ok(())
}
fn compare_f32(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (i, (&a, &b)) in a.iter().zip(b).enumerate() {
        assert!(
            (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits(),
            "f32index{i}:{a:?}vs{b:?}"
        );
    }
}
fn compare_bf16(a: &[bf16], b: &[bf16]) {
    assert_eq!(a.len(), b.len());
    for (i, (&a, &b)) in a.iter().zip(b).enumerate() {
        assert!(
            (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits(),
            "bf16index{i}:{a:?}vs{b:?}"
        );
    }
}

#[test]
fn warp_attention_direct_forward_backward_exact_bits_and_nonfinite_classes() -> Result<()> {
    if !super::super::super::tests::cuda_available() {
        return Ok(());
    }
    let device = CudaContext::new(0)?;
    let stream = device.new_stream()?;
    let shape = [8, 8, 512, 512];
    let launch = geometry(&shape, 64).expect("initial eligibility");
    let rows = launch.rows;
    let count = rows * 512;
    let old_forward = compile(
        &stream,
        super::super::super::attention_softmax::row_ir(0.125)?,
    )?;
    let new_forward = compile(&stream, forward_ir(0.125)?)?;
    let old_backward = compile(
        &stream,
        super::super::super::attention_softmax_backward::row_ir()?,
    )?;
    let new_backward = compile(&stream, backward_ir()?)?;
    let old_config = LaunchConfig {
        grid_dim: (u32::try_from(rows)?, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let new_config = LaunchConfig {
        grid_dim: launch.grid,
        block_dim: launch.block,
        shared_mem_bytes: 0,
    };
    for kernel in [&new_forward, &new_backward] {
        assert_eq!(kernel.function.shared_size_bytes()?, 0);
        assert!(kernel.function.max_threads_per_block()? >= 256);
        assert!(
            kernel
                .function
                .occupancy_max_active_blocks_per_multiprocessor(256, 0, None)?
                > 0
        );
        assert!(!kernel.ptx.contains("bar.sync") && !kernel.ptx.contains("shfl.sync.bfly"));
    }
    if let Some(directory) = std::env::var_os("TNSR_WARP_ATTENTION_PTX_DIR") {
        std::fs::create_dir_all(&directory)?;
        let directory = std::path::PathBuf::from(directory);
        let mut resources = Vec::new();
        for (name, kernel, threads) in [
            ("forward-old", &old_forward, 128),
            ("forward-warp", &new_forward, 256),
            ("backward-old", &old_backward, 128),
            ("backward-warp", &new_backward, 256),
        ] {
            std::fs::write(directory.join(format!("{name}.ptx")), &kernel.ptx)?;
            resources.push(serde_json::json!({"name":name,"registers":kernel.function.num_regs()?,"shared":kernel.function.shared_size_bytes()?,"local":kernel.function.local_size_bytes()?,"max_threads":kernel.function.max_threads_per_block()?,"cuda_static_block_capacity":kernel.function.occupancy_max_active_blocks_per_multiprocessor(threads,0,None)?}));
        }
        std::fs::write(
            directory.join("resources.json"),
            serde_json::to_vec_pretty(&resources)?,
        )?;
    }
    let mask: Vec<_> = (0..512 * 512)
        .map(|i| {
            if i % 512 > i / 512 {
                f32::NEG_INFINITY
            } else {
                0.
            }
        })
        .collect();
    let mask = stream.memcpy_stod(&mask)?;
    let mut qk = stream.alloc_zeros::<bf16>(count)?;
    let mut incoming = stream.alloc_zeros::<bf16>(count)?;
    let mut old_p = stream.alloc_zeros::<f32>(count)?;
    let mut new_p = stream.alloc_zeros::<f32>(count)?;
    let mut old_p16 = stream.alloc_zeros::<bf16>(count)?;
    let mut new_p16 = stream.alloc_zeros::<bf16>(count)?;
    let mut old_score = stream.alloc_zeros::<bf16>(count)?;
    let mut new_score = stream.alloc_zeros::<bf16>(count)?;
    for phase in 0..3 {
        let mut scores: Vec<_> = (0..count)
            .map(|i| {
                bf16::from_f32(if phase == 1 {
                    -0.
                } else {
                    ((i + phase * 7) % 37) as f32 / 16. - 1.
                })
            })
            .collect();
        let mut gradients: Vec<_> = (0..count)
            .map(|i| bf16::from_f32(((i + phase) % 29) as f32 / 32. - 0.5))
            .collect();
        if phase == 2 {
            for row in 0..rows {
                match row % 512 {
                    0 => scores[row * 512] = bf16::INFINITY,
                    1 => scores[row * 512] = bf16::NAN,
                    2 => scores[row * 512..(row + 1) * 512].fill(bf16::NEG_INFINITY),
                    3 => scores[row * 512 + 511] = bf16::INFINITY,
                    4 => gradients[row * 512 + 511] = bf16::INFINITY, // masked P0*Inf MUST still poison dot
                    5 => gradients[row * 512..(row + 1) * 512].fill(bf16::from_f32(-0.)),
                    6 => {
                        gradients[row * 512] = bf16::INFINITY;
                        gradients[row * 512 + 1] = bf16::NEG_INFINITY;
                    }
                    _ => {}
                }
            }
        }
        stream.memcpy_htod(&scores, &mut qk)?;
        stream.memcpy_htod(&gradients, &mut incoming)?;
        forward(
            &stream,
            &old_forward,
            old_config,
            &qk,
            &mask,
            &mut old_p,
            &mut old_p16,
        )?;
        forward(
            &stream,
            &new_forward,
            new_config,
            &qk,
            &mask,
            &mut new_p,
            &mut new_p16,
        )?;
        stream.synchronize()?;
        compare_f32(&stream.memcpy_dtov(&old_p)?, &stream.memcpy_dtov(&new_p)?);
        compare_bf16(
            &stream.memcpy_dtov(&old_p16)?,
            &stream.memcpy_dtov(&new_p16)?,
        );
        // Both VJPs consume the identical saved P32 buffer; forward agreement
        // above is a separate gate and cannot hide a backward-dot mismatch.
        backward(
            &stream,
            &old_backward,
            old_config,
            &old_p,
            &incoming,
            &mut old_score,
        )?;
        backward(
            &stream,
            &new_backward,
            new_config,
            &old_p,
            &incoming,
            &mut new_score,
        )?;
        stream.synchronize()?;
        compare_bf16(
            &stream.memcpy_dtov(&old_score)?,
            &stream.memcpy_dtov(&new_score)?,
        );
    }
    Ok(())
}

// REQUIRED AFTER actual Mixed integration (not fabricated by this standalone
// source preparation): Off/Fwd/Bwd/Both capture-key invalidation; changed-token
// replay and resident incremental allocation reuse; P32/P16/fullallgrad bits;
// recovery after invalid updates; mask/source-view/fanout/smallBH fallback;
// actualS512B8/B32 recognizer counters; full fixed loss/gradient/weight/update
// gates. Retain actualtarget PTX and offline-labeled SASS, checking new16 ordered
// FFMA steps in four independent4-term chains per physical lane (old4 steps
// per128-lane contributor; unchanged per-row count), canonical down16..1 and lane0IDX, zero
// shared/barriers and reported JIT registers/local memory. Do not time until
// all numerical/structural gates pass, and never infer speedup from barriers.

#[test]
fn warp_leader_primitive_preserves_raw_zero_and_nan_payload_bits() -> Result<()> {
    if !super::super::super::tests::cuda_available() {
        return Ok(());
    }
    let device = CudaContext::new(0)?;
    let stream = device.new_stream()?;
    let mut b = ScalarBuilder::default();
    let src = b.load(
        ScalarMemory::Global("input".into()),
        Expr::ThreadIdx(Dim::X),
    );
    let result = b.warp_broadcast_leader(src);
    b.store(
        ScalarMemory::Global("output".into()),
        Expr::ThreadIdx(Dim::X),
        result,
    );
    let ir = TileIR {
        kernel_name: "warp_leader_bits".into(),
        params: vec![
            KernelParam {
                name: "input".into(),
                dtype: DType::F32,
                is_input: true,
            },
            KernelParam {
                name: "output".into(),
                dtype: DType::F32,
                is_input: false,
            },
        ],
        body: Block { stmts: b.stmts },
        shared_mem_bytes: 0,
    };
    let kernel = compile(&stream, ir)?;
    assert!(kernel.ptx.contains("shfl.sync.idx.b32"));
    assert!(
        !kernel.ptx.contains("add.f32")
            && !kernel.ptx.contains("mul.f32")
            && !kernel.ptx.contains("shfl.sync.bfly")
    );
    let mut output = stream.alloc_zeros::<f32>(32)?;
    for bits in [
        0u32,
        0x8000_0000,
        0x3f80_0000,
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_1234,
        0x7f80_0001,
    ] {
        let mut values = [1.0f32; 32];
        values[0] = f32::from_bits(bits);
        let input = stream.memcpy_stod(&values)?;
        let mut launch = stream.launch_builder(&kernel.function);
        launch.arg(&input).arg(&mut output);
        // SAFETY: two exact disjoint32-element f32 buffers, one full warp,
        // retained until the synchronized readback; ABI is input32/output32.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            })
        }?;
        stream.synchronize()?;
        assert!(
            stream
                .memcpy_dtov(&output)?
                .iter()
                .all(|x| x.to_bits() == bits)
        );
    }
    Ok(())
}
