//! Test-only actual mixed NT linear / TN four-F32-partial calibration.
//! No alternative native kernel or production selector is introduced.
use super::*;
use crate::{
    ptx::{BF16, linear::program::Kernel},
    tensor::{TensorExpr, ValueDType},
};
use half::bf16;
use std::{
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
mod archive;
mod fused_nt128;
mod fused_nt128_four_wide;
use archive::{CASES, Case};

fn owned_range(e: &MixedExecutor, pointer: u64, len: usize, dtype: DType) -> Result<()> {
    let end = pointer
        .checked_add(u64::try_from(
            len.checked_mul(dtype.size_bytes())
                .context("read extent overflow")?,
        )?)
        .context("read address overflow")?;
    let covered = e.buffers.iter().any(|b| {
        let (ptr, bytes, actual) = match b {
            Buffer::F32(b) => (b.device_ptr(&e.stream).0, b.len() * 4, DType::F32),
            Buffer::BF16(b) => (b.device_ptr(&e.stream).0, b.len() * 2, DType::BF16),
        };
        actual == dtype
            && pointer >= ptr
            && ptr
                .checked_add(bytes as u64)
                .is_some_and(|limit| end <= limit)
    });
    ensure!(
        covered,
        "readback is outside a live matching-dtype owned allocation"
    );
    Ok(())
}
fn read32(e: &MixedExecutor, p: u64, len: usize) -> Result<Vec<f32>> {
    owned_range(e, p, len, DType::F32)?;
    e.stream.context().bind_to_thread()?;
    let mut out = vec![0f32; len];
    // SAFETY: owned_range proves this entire source interval belongs to a live
    // F32 allocation; executor retains owners; host destination has exactlen.
    unsafe {
        cudarc::driver::result::memcpy_dtoh_async(&mut out, p, e.stream.cu_stream())?;
    }
    e.stream.synchronize()?;
    Ok(out)
}
fn read16(e: &MixedExecutor, p: u64, len: usize) -> Result<Vec<bf16>> {
    owned_range(e, p, len, DType::BF16)?;
    e.stream.context().bind_to_thread()?;
    let mut out = vec![bf16::ZERO; len];
    // SAFETY: same full-range/lifetime proof as read32, with BF16 storage.
    unsafe {
        cudarc::driver::result::memcpy_dtoh_async(&mut out, p, e.stream.cu_stream())?;
    }
    e.stream.synchronize()?;
    Ok(out)
}
fn expression(c: Case) -> Result<TensorExpr<f32>> {
    let aa = TensorExpr::input(
        "a",
        if c.split {
            vec![c.k, c.m]
        } else {
            vec![c.m, c.k]
        },
    );
    let bb = TensorExpr::input(
        "b",
        if c.split {
            vec![c.k, c.n]
        } else {
            vec![c.n, c.k]
        },
    );
    let expression = if c.split {
        aa.transpose()
            .cast(ValueDType::BF16)
            .matmul(bb.cast(ValueDType::BF16))
            .cast(ValueDType::F32)
    } else {
        aa.linear_bf16(bb.transpose(), TensorExpr::input("bias", vec![c.n]))?
            .cast(ValueDType::F32)
    };
    Ok(expression)
}
fn construct(
    c: Case,
    a: &[bf16],
    b: &[bf16],
    bias: &[bf16],
) -> Result<(MixedExecutor, Vec<Kernel>, usize)> {
    let expression = expression(c)?;
    let graph: TensorGraph<f32> = expression.into();
    let mut e = MixedExecutor::new_with_options(
        &graph,
        MatMulPrecision::Tf32x3,
        false,
        MixedOptions {
            physical_order_casts: true,
            shared_memory_reuse: true,
            narrow_bf16_tile: true,
            medium_bf16_tile_policy: crate::tile::MediumBf16TilePolicy::SquareThenTall,
            split_k_bf16_tile_policy: crate::tile::SplitKBf16TilePolicy::Square64,
            plain_bf16_output: true,
            ..MixedOptions::default()
        },
    )?;
    // The nonresident mini graph retains Program-owned storage instead of
    // running resident pooling. Transfer those SAME owners into the executor
    // buffer list so the isolated raw-copy range proof covers every allocation.
    // This performs no allocation, remapping or change to any compiled kernel.
    for program in &mut e.programs {
        if let Some(program) = program.downcast_mut::<Program<F32>>() {
            e.buffers.extend(program.drain_storage());
        }
        if let Some(program) = program.downcast_mut::<Program<BF16>>() {
            e.buffers.extend(program.drain_storage());
        }
    }
    let mut inputs = HashMap::from([
        ("a".into(), a.iter().map(|x| (*x).to_f32()).collect()),
        ("b".into(), b.iter().map(|x| (*x).to_f32()).collect()),
    ]);
    if !c.split {
        inputs.insert("bias".into(), bias.iter().map(|x| (*x).to_f32()).collect());
    }
    e.upload(inputs)?;
    let selected = e
        .kernels
        .iter()
        .enumerate()
        .filter(|(_, k)| {
            if c.split {
                k.label.contains("split-K partial") || k.label.contains("split-K sum")
            } else {
                k.label.contains("Linear BF16 MatMulRegion")
            }
        })
        .map(|(i, _)| i)
        .collect::<Vec<_>>();
    ensure!(
        selected.len() == if c.split { 2 } else { 1 },
        "actual operation group contract changed: {:?}",
        e.kernel_labels()
    );
    if c.split {
        ensure!(
            e.info().split_k_square64_programs == 1,
            "actual TN Square64partial not selected"
        );
    }
    let first = selected[0];
    // Uploads and casts occur once; only the actual operation group is captured.
    // Do not execute F32 promotion after the operation: it may reuse the dead
    // partial buffer. Full F32partial snapshots precede any such publication.
    for k in &e.kernels[..first] {
        k.run(&e.stream)?;
    }
    e.stream.synchronize()?;
    let kernels = selected
        .iter()
        .map(|&i| e.kernels[i].clone())
        .collect::<Vec<_>>();
    Ok((e, kernels, first))
}
fn resources(e: &MixedExecutor, kernels: &[Kernel]) -> Result<serde_json::Value> {
    let rows=kernels.iter().map(|k|Ok(serde_json::json!({"label":k.label,"owner":k.owner.map(|n|n.index()),"pointers":k.pointers,"grid":k.config.grid_dim,"block":k.config.block_dim,"registers":k.function.num_regs()?,"shared":k.function.shared_size_bytes()?,"local":k.function.local_size_bytes()?,"max_threads":k.function.max_threads_per_block()?,"api_capacity":k.function.occupancy_max_active_blocks_per_multiprocessor(k.config.block_dim.0*k.config.block_dim.1*k.config.block_dim.2,0,None)?}))).collect::<Result<Vec<_>>>()?;
    Ok(
        serde_json::json!({"info":e.info(),"kernels":rows,"scope":"actual executed groups and driverAPI resources; capacity is not achieved occupancy"}),
    )
}
fn submit(e: &MixedExecutor, k: &[Kernel]) -> Result<()> {
    for k in k {
        k.run(&e.stream)?;
    }
    Ok(())
}
fn bits16(a: &[bf16], b: &[bf16]) -> Result<()> {
    ensure!(a.len() == b.len(), "outputlen mismatch");
    ensure!(
        a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()),
        "output replay bits changed"
    );
    Ok(())
}
#[test]
#[ignore = "manual actual NT fused-linear and TN F32parts native calibration; no alternative kernel"]
fn non_nn_calibration_native_worker() -> Result<()> {
    let inputs =
        PathBuf::from(std::env::var_os("TNSR_NON_NN_INPUTS").context("set input archive")?);
    let partial_export = PathBuf::from(
        std::env::var_os("TNSR_SPLIT_K_SQUARE_PTX_DIR")
            .context("set actual bound SplitK PTX/identity export directory")?,
    );
    let out =
        PathBuf::from(std::env::var_os("TNSR_NON_NN_OUTPUT").context("set output directory")?);
    let timing = std::env::var("TNSR_NON_NN_TIMING").as_deref() == Ok("1");
    let reverse = std::env::var("TNSR_NON_NN_REVERSE").as_deref() == Ok("1");
    std::fs::create_dir_all(&out)?;
    ensure!(
        !out.join("report.json").exists(),
        "preserve existing worker report"
    );
    let mut cases = CASES.to_vec();
    if reverse {
        cases.reverse();
    }
    let mut reports = Vec::new();
    for c in cases {
        let d = out.join(c.name);
        std::fs::create_dir_all(&d)?;
        let a = archive::read16(&inputs.join(c.name).join("a.bf16"), c.m * c.k)?;
        let b = archive::read16(&inputs.join(c.name).join("b.bf16"), c.k * c.n)?;
        let bias = archive::read16(
            &inputs.join(c.name).join("bias.bf16"),
            if c.split { 0 } else { c.n },
        )?;
        let (e, kernels, prefix) = construct(c, &a, &b, &bias)?;
        let metadata = resources(&e, &kernels)?;
        archive::json(&d.join("actual-resources-and-abi.json"), &metadata)?;
        for (i, p) in e
            .programs
            .iter()
            .filter_map(|p| p.downcast_ref::<Program<BF16>>())
            .enumerate()
        {
            std::fs::write(d.join(format!("program-{i}.ptx")), &p.ptx)?;
            std::fs::write(d.join(format!("program-{i}-plan.txt")), &p.plan)?;
            let views=p.views.iter().map(|(node,v)|serde_json::json!({"node":node.index(),"view":format!("{v:?}"),"normalized_access":format!("{:?}",v.access.clone().normalize_in_domain(&v.shape))})).collect::<Vec<_>>();
            archive::json(
                &d.join(format!("program-{i}-bound-views.json")),
                &serde_json::json!({"views":views,"pointer_scope":"executed remapped pointers are in actual-resources-and-abi; semantic views do not own pointers"}),
            )?;
        }
        submit(&e, &kernels)?;
        let output_pointer = *kernels
            .last()
            .context("no output kernel")?
            .pointers
            .last()
            .context("no output ABI")?;
        let expected = read16(&e, output_pointer, c.m * c.n)?;
        ensure!(
            expected.iter().all(|x| x.is_finite()),
            "finite archive yielded nonfinite output"
        );
        archive::write16(&d.join("output.bf16"), &expected)?;
        let partial_pointer = if c.split {
            Some(kernels[0].pointers[2])
        } else {
            None
        };
        let expected_partials = if let Some(p) = partial_pointer {
            let data = read32(&e, p, 4 * c.m * c.n)?;
            ensure!(data.iter().all(|x| x.is_finite()), "nonfinite F32partial");
            archive::write32(&d.join("partials.f32"), &data)?;
            Some(data)
        } else {
            None
        };
        let repeats = 32;
        let samples = 32;
        e.stream.synchronize()?;
        e.stream
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
        let recording = (|| -> Result<()> {
            for _ in 0..repeats {
                submit(&e, &kernels)?;
            }
            Ok(())
        })();
        let ended = e.stream.end_capture(
            sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        );
        recording?;
        let captured = ended?.context("empty nativecomponentcapture")?;
        captured.launch()?;
        bits16(&expected, &read16(&e, output_pointer, c.m * c.n)?)?;
        let mut times = Vec::new();
        let mut warmups = 0;
        let mut warmup_ms = 0.;
        if timing {
            let start = Instant::now();
            while start.elapsed() < Duration::from_millis(500) {
                captured.launch()?;
                e.stream.synchronize()?;
                warmups += 1;
            }
            warmup_ms = start.elapsed().as_secs_f64() * 1000.;
            for _ in 0..samples {
                let start = e
                    .stream
                    .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
                captured.launch()?;
                let end = e
                    .stream
                    .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
                let us = f64::from(start.elapsed_ms(&end)?) * 1000. / repeats as f64;
                ensure!(us.is_finite() && us > 0., "invalid timing");
                times.push(us);
            }
        }
        bits16(&expected, &read16(&e, output_pointer, c.m * c.n)?)?;
        if let Some(p) = partial_pointer {
            let actual = read32(&e, p, 4 * c.m * c.n)?;
            ensure!(
                actual
                    .iter()
                    .zip(expected_partials.as_ref().context("partials absent")?)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "F32partial replay bits changed"
            );
            archive::write32(&d.join("partials-post.f32"), &actual)?;
        }
        reports.push(serde_json::json!({"case":c.name,"kind":if c.split {"tn_split4"}else{"nt_fused_linear"},"metadata":metadata,"excluded_input_setup_kernels":prefix,"component_kernel_count":kernels.len(),"typed_abi":if c.split {"BF16 A(T),BF16 B(N)→4F32partials;left3F32adds→BF16 output"}else{"BF16 A(N),BF16 W(T),BF16 bias→F32MMA+bias→singleBF16output"},"captured_repeats":repeats,"warmups":warmups,"warmup_ms":warmup_ms,"samples_us":times,"timing_performed":timing,"pre_post_output_bits_equal":true,"pre_post_partial_bits_equal":c.split.then_some(true),"scope":"isolated actual mixed operation group; upload/casts/promotion outside capture; no new kernel, fullmodel or convergence claim"}));
        archive::json(
            &out.join("report.json"),
            &serde_json::json!({"cases":reports,"reverse":reverse,"actual_partial_ptx_and_region_identity_directory":partial_export}),
        )?;
    }
    Ok(())
}

#[test]
fn non_nn_calibration_graph_contracts_and_physical_archive_shapes() -> Result<()> {
    for c in CASES {
        let graph: TensorGraph<f32> = expression(c)?.into();
        let linear = graph
            .graph
            .node_indices()
            .filter(|&n| matches!(graph[n], TensorGraphNode::Linear { .. }))
            .count();
        let matmul = graph
            .graph
            .node_indices()
            .filter(|&n| matches!(graph[n], TensorGraphNode::MatMul { .. }))
            .count();
        assert_eq!(linear, usize::from(!c.split));
        assert_eq!(matmul, usize::from(c.split));
        let dims = crate::graph::mixed::value_dtypes(&graph);
        let operation = graph
            .graph
            .node_indices()
            .find(|&n| {
                matches!(
                    graph[n],
                    TensorGraphNode::Linear { .. } | TensorGraphNode::MatMul { .. }
                )
            })
            .context("operation absent")?;
        assert_eq!(dims.get(&operation), Some(&ValueDType::BF16));
        assert_eq!(graph[operation].shape(), &vec![c.m, c.n]);
    }
    Ok(())
}
