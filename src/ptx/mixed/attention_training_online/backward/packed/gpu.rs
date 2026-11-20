//! Ignored, untimed complete-pair epilogue proof. No model integration.
use super::super::super::{
    forward,
    tests::{compile, data, finite_bits, kernel, physical, resources},
};
use super::*;
use crate::ptx::linear::{Captured, program::Kernel};
use cudarc::driver::{CudaContext, CudaStream, DevicePtr, LaunchConfig, sys};
use half::bf16;
use std::sync::Arc;

fn capture(stream: &Arc<CudaStream>, kernels: &[Kernel]) -> Result<Captured> {
    stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
    let result = kernels.iter().try_for_each(|k| k.run(stream));
    let ended = stream.end_capture(
        sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
    );
    result?;
    Ok(Captured {
        graph: Some(ended?.context("packed pair capture empty")?),
        stream: stream.clone(),
    })
}
fn disjoint(ranges: &[(u64, usize)]) -> Result<()> {
    for (i, &(pointer, bytes)) in ranges.iter().enumerate() {
        ensure!(pointer % 128 == 0, "fresh allocation is not 128B aligned");
        let end = pointer
            .checked_add(u64::try_from(bytes)?)
            .context("rangeoverflow")?;
        for &(other, count) in &ranges[..i] {
            let other_end = other
                .checked_add(u64::try_from(count)?)
                .context("rangeoverflow")?;
            ensure!(
                end <= other || other_end <= pointer,
                "leaf allocation ranges overlap"
            );
        }
    }
    Ok(())
}
#[test]
#[ignore = "manual GPU resource and packed output proof; root reservation required"]
fn online_packed_backward_epilogues_complete_pair_exact_bits_replay_resources() -> Result<()> {
    complete_pair(8)
}

#[test]
#[ignore = "manual BH128 GPU packed proof; separate fresh audit root required"]
fn online_packed_backward_bh128_complete_pair_exact_bits_replay_resources() -> Result<()> {
    complete_pair(16)
}

#[test]
#[ignore = "manual BH256 GPU packed proof; separate fresh audit root required"]
fn online_packed_backward_bh256_complete_pair_exact_bits_replay_resources() -> Result<()> {
    complete_pair(32)
}

fn complete_pair(batch: usize) -> Result<()> {
    if !crate::ptx::mixed::tests::cuda_available() {
        return Ok(());
    }
    ensure!(
        [8, 16, 32].contains(&batch),
        "bounded explicit leaf batch only"
    );
    if let Some(path) = std::env::var_os("TNSR_ONLINE_ATTENTION_AUDIT") {
        ensure!(
            !std::path::PathBuf::from(path)
                .join("packed-qualification.json")
                .exists(),
            "use a fresh per-case audit directory; never overwrite a qualified BH64/BH128/BH256 proof"
        );
    }
    let shape = Shape::new(batch, 8)?;
    let n = shape.elements();
    let layout = if std::env::var("TNSR_ONLINE_LAYOUT").as_deref() == Ok("head") {
        ReadLayout::HeadMajor
    } else {
        ReadLayout::SequenceMajor
    };
    let context = CudaContext::new(0)?;
    // One private ordered stream and all strong buffer/function owners outlive
    // both captures and their final synchronization. No raw pointer is retained
    // beyond this scope, and no external allocator or mutable alias participates.
    unsafe {
        context.disable_event_tracking();
    }
    let stream = context.new_stream()?;
    let mut inputs = [
        stream.alloc_zeros::<bf16>(n)?,
        stream.alloc_zeros(n)?,
        stream.alloc_zeros(n)?,
        stream.alloc_zeros(n)?,
    ];
    let mut output = stream.alloc_zeros::<bf16>(n)?;
    let raw = [
        stream.alloc_zeros::<bf16>(n)?,
        stream.alloc_zeros(n)?,
        stream.alloc_zeros(n)?,
    ];
    let mut packed = stream.alloc_zeros::<bf16>(3 * n)?;
    let mut statistics = stream.alloc_zeros::<f32>(shape.bh() * SEQUENCE * 3)?;
    let [q, k, v, incoming] = inputs.each_ref().map(|x| x.device_ptr(&stream).0);
    let [dq, dk, dv] = raw.each_ref().map(|x| x.device_ptr(&stream).0);
    let o = output.device_ptr(&stream).0;
    let target = packed.device_ptr(&stream).0;
    let stats = statistics.device_ptr(&stream).0;
    // Target is ONE owned allocation. CPU inverse mixed-radix proof partitions
    // it into disjoint Q/K/V windows. This additionally proves whole target vs
    // every read-only input, old raw checkpoint, O, and statistics byte range.
    let mut ranges = vec![
        (q, n * 2),
        (k, n * 2),
        (v, n * 2),
        (incoming, n * 2),
        (o, n * 2),
        (dq, n * 2),
        (dk, n * 2),
        (dv, n * 2),
        (target, 3 * n * 2),
        (stats, shape.bh() * SEQUENCE * 3 * 4),
    ];
    disjoint(&ranges)?;
    ranges.clear();
    let (region, values) = fixture(shape)?;
    let irs = [
        forward::ir_with_transport(shape, layout, Transport::Async16)?,
        super::super::d_query_ir_with_transport(shape, layout, layout, Transport::Async16)?,
        super::super::d_key_value_ir_with_transport(shape, layout, layout, Transport::Async16)?,
        dquery_ir(shape, layout, &region)?,
        dkeyvalue_ir(shape, layout, &region)?,
    ];
    let names = [
        "forward",
        "old-dquery",
        "old-dkeyvalue",
        "packed-dquery",
        "packed-dkeyvalue",
    ];
    let mut functions = Vec::new();
    let mut facts = Vec::new();
    for (ir, name) in irs.into_iter().zip(names) {
        let function = compile(&stream, ir, &format!("packed-{name}"))?;
        facts.push(resources(&function, name)?);
        functions.push(function);
    }
    let pack = compile(
        &stream,
        qkv_grad_pack::diagnostic_ir(&region, &values)?,
        "packed-existing-qkv-pack",
    )?;
    let pack_facts = serde_json::json!({"registers":pack.num_regs()?,"shared":pack.shared_size_bytes()?,"local":pack.local_size_bytes()?,"capacity":pack.occupancy_max_active_blocks_per_multiprocessor(256,0,None)?});
    let directory = std::env::var_os("TNSR_ONLINE_ATTENTION_AUDIT").map(std::path::PathBuf::from);
    if let Some(path) = &directory {
        std::fs::create_dir_all(path)?;
        std::fs::write(
            path.join("packed-resources.json"),
            serde_json::to_vec_pretty(
                &serde_json::json!({"functions":facts,"old_pack":pack_facts,"candidate_launched":false}),
            )?,
        )?;
    }
    // Nothing, including forward, launches until every complete baseline and
    // candidate function passes. A spill/resource failure is a rejected leaf.
    ensure!(
        facts
            .iter()
            .all(|x| x["strict_production_resource_admitted"] == true),
        "strict resource proof failed before any pair launch"
    );
    ensure!(
        pack.local_size_bytes()? == 0 && pack.max_threads_per_block()? >= 256,
        "existing pack resources invalid"
    );
    let fwd = kernel(
        functions[0].clone(),
        shape,
        vec![q, k, v, o, stats],
        "packed unchanged forward",
    )?;
    let mut pack_pointers = region
        .inputs
        .iter()
        .map(|input| [dq, dk, dv][input.window])
        .collect::<Vec<_>>();
    pack_pointers.push(target);
    let old_pack = Kernel {
        owner: None,
        label: "actual existing QKV gradient pack".into(),
        function: pack,
        config: LaunchConfig {
            grid_dim: (u32::try_from((3 * n).div_ceil(256))?, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        },
        pointers: pack_pointers,
        zero_before: None,
    };
    let old = vec![
        fwd.clone(),
        kernel(
            functions[1].clone(),
            shape,
            vec![q, k, v, o, incoming, stats, dq],
            "old dQ",
        )?,
        kernel(
            functions[2].clone(),
            shape,
            vec![q, k, v, incoming, stats, dk, dv],
            "old dKT+dV",
        )?,
        old_pack,
    ];
    let new = vec![
        fwd,
        kernel(
            functions[3].clone(),
            shape,
            vec![q, k, v, o, incoming, stats, target],
            "packed dQ",
        )?,
        kernel(
            functions[4].clone(),
            shape,
            vec![q, k, v, incoming, stats, target],
            "packed dKT+dV",
        )?,
    ];
    let captures = [capture(&stream, &old)?, capture(&stream, &new)?];
    let mut records = Vec::new();
    for revision in 0..8 {
        let mut logical = [
            data(shape, revision, 1),
            data(shape, revision, 2),
            data(shape, revision, 3),
            data(shape, revision, 4),
        ];
        if revision == 5 {
            logical[0][0] = bf16::from_bits(0x7f81);
        }
        if revision == 6 {
            logical[0][0] = bf16::INFINITY;
            logical[1][1] = bf16::NEG_INFINITY;
        }
        if revision == 7 {
            for (slot, values) in logical.iter_mut().enumerate() {
                for (index, value) in values.iter_mut().enumerate() {
                    *value = bf16::from_bits(
                        [0x0001, 0x8001, 0x007f, 0x807f, 0x0080, 0x8080][(index + slot) % 6],
                    );
                }
            }
        }
        let physical_inputs = logical.each_ref().map(|x| physical(x, shape, layout));
        for (target, source) in inputs.iter_mut().zip(&physical_inputs) {
            stream.memcpy_htod(source, target)?;
        }
        let mut reference = None;
        for (mode, captured) in captures.iter().enumerate() {
            stream.memcpy_htod(&vec![bf16::NAN; n], &mut output)?;
            stream.memcpy_htod(&vec![bf16::NAN; 3 * n], &mut packed)?;
            stream.memcpy_htod(&vec![f32::NAN; shape.bh() * SEQUENCE * 3], &mut statistics)?;
            captured.graph.as_ref().unwrap().launch()?;
            let first = (
                stream.memcpy_dtov(&output)?,
                stream.memcpy_dtov(&packed)?,
                stream.memcpy_dtov(&statistics)?,
            );
            captured.graph.as_ref().unwrap().launch()?;
            let second = (
                stream.memcpy_dtov(&output)?,
                stream.memcpy_dtov(&packed)?,
                stream.memcpy_dtov(&statistics)?,
            );
            finite_bits(&first.0, &second.0, "O self replay");
            finite_bits(&first.1, &second.1, "packed self replay");
            for (a, b) in first.2.iter().zip(&second.2) {
                ensure!(
                    (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits(),
                    "stats replay mismatch"
                );
            }
            if revision <= 2 || revision == 7 {
                ensure!(
                    first.0.iter().chain(&first.1).all(|x| x.is_finite())
                        && first.2.iter().all(|x| x.is_finite()),
                    "finite feed did not fully overwrite outputs"
                );
            }
            if mode == 0 {
                reference = Some(first);
            } else {
                let expected = reference.as_ref().unwrap();
                finite_bits(&expected.0, &first.0, "O old vs packed");
                finite_bits(&expected.1, &first.1, "all packed QKV bits / NaN classes");
                for (a, b) in expected.2.iter().zip(&first.2) {
                    ensure!(
                        (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits(),
                        "old/new statistic mismatch"
                    );
                }
            }
        }
        for (source, expected) in inputs.iter().zip(&physical_inputs) {
            let actual = stream.memcpy_dtov(source)?;
            ensure!(
                actual
                    .iter()
                    .zip(expected)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "read-only input changed, including NaN payload"
            );
        }
        records.push(serde_json::json!({"revision":revision,"full_packed_qkv_and_O_exact_non_nan_bits":true,"stats_exact_non_nan_bits":true,"nan_classes":true,"replay":2,"read_only_inputs_all_bits":true}));
    }
    stream.synchronize()?;
    if let Some(path) = directory {
        std::fs::write(
            path.join("packed-qualification.json"),
            serde_json::to_vec_pretty(
                &serde_json::json!({"batch":batch,"heads":shape.heads,"layout":format!("{layout:?}"),"resources":facts,"old_pack_resources":pack_facts,"revisions":records,"baseline_kernels":4,"candidate_kernels":3,"scope":"leaf epilogue-only, unchanged Async16 pair, actual existing QKV pack baseline","model_qualification":false,"performance_claim":false}),
            )?,
        )?;
    }
    Ok(())
}
