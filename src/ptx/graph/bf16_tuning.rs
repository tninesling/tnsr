//! Opt-in GPU schedule experiment. Never runs as part of the ordinary test suite.
//! TNSR_BF16_TUNE_OUTPUT must name a fresh directory; all samples and generated
//! kernels are retained. This measures GEMMs, not complete training throughput.
use std::path::PathBuf;

use anyhow::{Context, Result, ensure};
use cudarc::{
    driver::{CudaContext, CudaFunction, CudaSlice, DevicePtr, LaunchConfig, PushKernelArg, sys},
    nvrtc::Ptx,
};
use half::bf16;

use super::{PtxGraph, PtxTarget};
use crate::{
    graph::{TensorGraph, TensorGraphNode},
    tile::{
        DType, MatMulPipeline, MatMulPrecision, MatMulRegion, MatMulSchedule, MatMulSharedLayout,
        MatMulTile, TileGraph, VirtualTensor,
    },
};

struct Kernel {
    name: String,
    function: CudaFunction,
    output: CudaSlice<f32>,
    pointers: [u64; 3],
    schedule: MatMulSchedule,
    samples: Vec<f64>,
    normalized_only: bool,
}

#[test]
#[ignore = "opt-in CUDA performance experiment; requires a fresh output directory"]
fn bf16_schedule_sweep() -> Result<()> {
    let directory = PathBuf::from(std::env::var("TNSR_BF16_TUNE_OUTPUT")?);
    std::fs::create_dir(&directory).context("tuning output must be fresh")?;
    let device = CudaContext::new(0)?;
    let stream = device.new_stream()?;
    let target = PtxTarget::from_context(&device)?;
    let mut rows = Vec::new();
    // Full physical operands, including the interleaved split-K transpose used
    // by the real weight-gradient path, rather than independent packed batches.
    for (label, m, k, n, split, transpose_lhs, transpose_rhs) in [
        ("qkv-wgrad", 256, 4096, 768, 4, true, false),
        ("down-wgrad", 683, 4096, 256, 4, true, false),
        ("gate-wgrad", 256, 4096, 683, 4, true, false),
        ("qkv-forward", 4096, 256, 768, 1, false, true),
        ("down-forward", 4096, 683, 256, 1, false, true),
        ("qkv-input-grad", 4096, 768, 256, 1, false, false),
        ("gate-input-grad", 4096, 683, 256, 1, false, false),
        ("gate-forward", 4096, 256, 683, 1, false, true),
    ] {
        if std::env::var("TNSR_BF16_TUNE_CASE").is_ok_and(|wanted| wanted != label) {
            continue;
        }
        let kp = k / split;
        let mut graph = TensorGraph::<bf16>::new();
        let lhs = graph.graph.add_node(TensorGraphNode::Input {
            name: "a",
            shape: vec![split, m, kp],
        });
        let rhs = graph.graph.add_node(TensorGraphNode::Input {
            name: "b",
            shape: vec![split, kp, n],
        });
        let root = graph.graph.add_node(TensorGraphNode::MatMul {
            shape: vec![split, m, n],
        });
        graph.graph.add_edge(lhs, root, 0);
        graph.graph.add_edge(rhs, root, 1);
        let mut region = MatMulRegion::from_graph(&graph, root, vec![], vec![root])?;
        let lhs_shape = if transpose_lhs {
            vec![k, m]
        } else {
            vec![m, k]
        };
        let rhs_shape = if transpose_rhs {
            vec![n, k]
        } else {
            vec![k, n]
        };
        let a = VirtualTensor::identity(lhs, lhs_shape.clone(), DType::BF16);
        let b = VirtualTensor::identity(rhs, rhs_shape.clone(), DType::BF16);
        region.inputs[0].tensor = (if transpose_lhs {
            a.permute(&[1, 0])?
        } else {
            a
        })
        .reshape(vec![m, split, kp])?
        .permute(&[1, 0, 2])?;
        region.inputs[0].source_shape = lhs_shape;
        region.inputs[1].tensor = (if transpose_rhs {
            b.permute(&[1, 0])?
        } else {
            b
        })
        .reshape(vec![split, kp, n])?;
        region.inputs[1].source_shape = rhs_shape;
        let values = |extent: usize, seed: u32| -> Vec<bf16> {
            (0..extent)
                .map(|i| {
                    let bits = (i as u32).wrapping_mul(747796405).wrapping_add(seed);
                    bf16::from_f32(((bits ^ (bits >> 16)) % 1024) as f32 / 1024.0 - 0.5)
                })
                .collect()
        };
        let a = values(m * k, 123);
        let b = values(k * n, 987);
        let a_device = stream.memcpy_stod(&a)?;
        let b_device = stream.memcpy_stod(&b)?;
        let mut kernels = Vec::new();
        for (tm, tn, tk, wm, wn) in [
            (32, 32, 32, 2, 2),
            (32, 32, 64, 2, 2),
            (32, 64, 64, 2, 4),
            (64, 32, 64, 4, 2),
            (64, 64, 32, 2, 2),
            (64, 64, 64, 2, 2),
            (64, 128, 32, 2, 4),
            (128, 64, 32, 4, 2),
        ] {
            for normalized_only in [false, true] {
                for shared_layout in [MatMulSharedLayout::Contiguous, MatMulSharedLayout::Padded] {
                    for pipeline in [MatMulPipeline::Synchronous, MatMulPipeline::DoubleBuffered] {
                        let mut schedule = MatMulSchedule::select_with_resources(
                            m,
                            n,
                            kp,
                            DType::BF16,
                            target.matmul_capabilities(),
                            MatMulPrecision::AllowTf32,
                            target.matmul_resources,
                        );
                        schedule.block_tile = MatMulTile {
                            m: tm,
                            n: tn,
                            k: tk,
                        };
                        schedule.warp_topology = (wm, wn);
                        schedule.warp_fragments = (
                            tm / (16 * schedule.warp_topology.0),
                            tn / (16 * schedule.warp_topology.1),
                        );
                        schedule = schedule
                            .with_shared_layout(shared_layout, target.matmul_resources)?
                            .with_pipeline(
                                pipeline,
                                target.supports_async_copy(),
                                !transpose_lhs && !transpose_rhs,
                                target.matmul_resources,
                            );
                        region.schedule = schedule;
                        if pipeline == MatMulPipeline::DoubleBuffered {
                            region.configure_pipeline(
                                shared_layout,
                                pipeline,
                                target.supports_async_copy(),
                                target.matmul_resources,
                            )?;
                            schedule = region.schedule;
                        }
                        let name = format!(
                            "{label}-{tm}x{tn}x{tk}-w{wm}x{wn}-{pipeline:?}-{shared_layout:?}-normalize{normalized_only}"
                        );
                        let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
                        tiles.add_matmul_regions(&[region.clone()])?;
                        for parameter in &mut tiles.matmul_region_kernels[0].ir.params {
                            if !parameter.is_input {
                                parameter.dtype = DType::F32;
                            }
                        }
                        if normalized_only {
                            for kernel in &mut tiles.matmul_region_kernels {
                                kernel.ir.normalize_indices()?;
                            }
                        } else {
                            for kernel in &mut tiles.matmul_region_kernels {
                                kernel.ir.validate_layouts()?;
                                kernel.ir.optimize_indices_with_sharing(true)?;
                            }
                        }
                        let ptx = PtxGraph::from(tiles).with_target(target);
                        let source = ptx.to_ptx();
                        std::fs::write(directory.join(format!("{name}.ptx")), &source)?;
                        let module = device.load_module(Ptx::from_src(source))?;
                        let output = stream.alloc_zeros::<f32>(split * m * n)?;
                        let pointers = [
                            a_device.device_ptr(&stream).0,
                            b_device.device_ptr(&stream).0,
                            output.device_ptr(&stream).0,
                        ];
                        kernels.push(Kernel {
                            name,
                            function: module.load_function(
                                ptx.matmul_region_kernel_name(0).context("missing GEMM")?,
                            )?,
                            output,
                            pointers,
                            schedule,
                            samples: Vec::new(),
                            normalized_only,
                        });
                    }
                }
            }
        }
        let launch = |kernel: &Kernel| -> Result<()> {
            let mut launch = stream.launch_builder(&kernel.function);
            for pointer in &kernel.pointers {
                launch.arg(pointer);
            }
            // SAFETY: these retained allocations match the generated two-BF16,
            // one-f32 ABI. All accesses/captures are serialized on this stream.
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (
                        u32::try_from(n.div_ceil(kernel.schedule.block_tile.n))?,
                        u32::try_from(m.div_ceil(kernel.schedule.block_tile.m))?,
                        u32::try_from(split)?,
                    ),
                    block_dim: kernel.schedule.block_threads,
                    shared_mem_bytes: 0,
                })
            }?;
            Ok(())
        };
        stream.synchronize()?;
        for kernel in &kernels {
            for _ in 0..8 {
                launch(kernel)?;
            }
        }
        stream.synchronize()?;
        let reference = stream.memcpy_dtov(&kernels[0].output)?;
        // Independent f64 dot products sample boundaries and split partitions.
        for index in (0..reference.len()).step_by((reference.len() / 97).max(1)) {
            let (s, row, col) = (index / (m * n), (index / n) % m, index % n);
            let expected: f64 = (s * kp..(s + 1) * kp)
                .map(|inner| {
                    let ai = if transpose_lhs {
                        inner * m + row
                    } else {
                        row * k + inner
                    };
                    let bi = if transpose_rhs {
                        col * k + inner
                    } else {
                        inner * n + col
                    };
                    f64::from(a[ai].to_f32()) * f64::from(b[bi].to_f32())
                })
                .sum();
            ensure!(
                (f64::from(reference[index]) - expected).abs() < 2e-4,
                "reference dot mismatch {label} at {index}"
            );
        }
        for kernel in &kernels {
            let actual = stream.memcpy_dtov(&kernel.output)?;
            ensure!(
                actual == reference,
                "schedule changed f32 accumulations: {}",
                kernel.name
            );
        }
        let mut captures = Vec::new();
        for kernel in &kernels {
            stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
            let recorded = (|| {
                for _ in 0..8 {
                    launch(kernel)?;
                }
                Ok::<_, anyhow::Error>(())
            })();
            let ended = stream.end_capture(
                sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            );
            recorded?;
            captures.push(ended?.context("empty tuning graph")?);
        }
        for capture in &captures {
            for _ in 0..4 {
                capture.launch()?;
            }
        }
        stream.synchronize()?;
        for iteration in 0..48 {
            let mut order: Vec<usize> = (0..kernels.len()).collect();
            let count = order.len();
            order.rotate_left((iteration / 2) % count);
            if iteration % 2 == 1 {
                order.reverse();
            }
            for index in order {
                let start = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
                captures[index].launch()?;
                let end = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
                kernels[index]
                    .samples
                    .push(f64::from(start.elapsed_ms(&end)?) * 1000.0 / 8.0);
            }
        }
        stream.synchronize()?;
        let baseline = kernels[0].samples.clone();
        for kernel in &kernels {
            let median = |mut values: Vec<f64>| {
                values.sort_by(f64::total_cmp);
                (values[23] + values[24]) / 2.0
            };
            let ratio = median(
                baseline
                    .iter()
                    .zip(&kernel.samples)
                    .map(|(a, b)| a / b)
                    .collect(),
            );
            println!(
                "{}: {:.3} us; paired speedup {:.4}",
                kernel.name,
                median(kernel.samples.clone()),
                ratio
            );
            rows.push(serde_json::json!({"name":kernel.name,"shape":[m,k,n],"split":split,
                "transpose_lhs":transpose_lhs,"transpose_rhs":transpose_rhs,"schedule":format!("{:?}",kernel.schedule),
                "normalized_only":kernel.normalized_only,"samples_us":kernel.samples,"median_us":median(kernel.samples.clone()),"paired_speedup":ratio,
                "f32_output_exact_to_baseline":true,"reference_sample_dots":reference.len().div_ceil((reference.len() / 97).max(1))}));
        }
        // Destroy captures while their referenced allocations are still alive.
        drop(captures);
        std::fs::write(
            directory.join("report.json"),
            serde_json::to_vec_pretty(&rows)?,
        )?;
    }
    ensure!(!rows.is_empty(), "no tuning case selected");
    Ok(())
}
