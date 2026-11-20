//! Whole-pair transport equivalence only; no timing or model runtime hook.
use super::super::tests::*;
use super::*;
use crate::ptx::linear::{Captured, program::Kernel};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, sys};
use half::bf16;
use std::sync::Arc;

fn capture(stream: &Arc<CudaStream>, kernels: &[Kernel]) -> Result<Captured> {
    stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
    let result = kernels.iter().try_for_each(|kernel| kernel.run(stream));
    let ended = stream.end_capture(
        sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
    );
    result?;
    Ok(Captured {
        graph: Some(ended?.ok_or_else(|| anyhow::anyhow!("empty complete transport capture"))?),
        stream: stream.clone(),
    })
}
fn read(stream: &Arc<CudaStream>, outputs: &[CudaSlice<bf16>; 4]) -> Result<[Vec<bf16>; 4]> {
    Ok([
        stream.memcpy_dtov(&outputs[0])?,
        stream.memcpy_dtov(&outputs[1])?,
        stream.memcpy_dtov(&outputs[2])?,
        stream.memcpy_dtov(&outputs[3])?,
    ])
}

#[test]
fn online_async16_complete_pair_finite_bits_nonfinite_classes_replay_and_resources() -> Result<()> {
    if !crate::ptx::mixed::tests::cuda_available() {
        return Ok(());
    }
    let shape = Shape::new(8, 8)?;
    let layout = if std::env::var("TNSR_ONLINE_LAYOUT").as_deref() == Ok("head") {
        ReadLayout::HeadMajor
    } else {
        ReadLayout::SequenceMajor
    };
    let golden = std::path::PathBuf::from(std::env::var("TNSR_ONLINE_SCALAR_GOLDENS")?);
    let context = CudaContext::new(0)?;
    // Both modes share one private ordered stream. Every allocation/function
    // outlives both complete captures; event tracking cannot inject dependencies.
    unsafe {
        context.disable_event_tracking();
    }
    let stream = context.new_stream()?;
    let n = shape.elements();
    let mut inputs = [
        stream.alloc_zeros::<bf16>(n)?,
        stream.alloc_zeros(n)?,
        stream.alloc_zeros(n)?,
        stream.alloc_zeros(n)?,
    ];
    let mut outputs = [
        stream.alloc_zeros::<bf16>(n)?,
        stream.alloc_zeros(n)?,
        stream.alloc_zeros(n)?,
        stream.alloc_zeros(n)?,
    ];
    let mut statistics = stream.alloc_zeros::<f32>(shape.bh() * 512 * 3)?;
    let descriptions = std::array::from_fn(|index| {
        let source_shape = if layout == ReadLayout::SequenceMajor {
            vec![shape.batch, 512, shape.heads * 64]
        } else {
            vec![shape.batch, shape.heads, 512, 64]
        };
        let root =
            VirtualTensor::identity(NodeIndex::new(index), source_shape.clone(), DType::BF16);
        let mut view = if layout == ReadLayout::SequenceMajor {
            root.reshape(vec![shape.batch, 512, shape.heads, 64])
                .unwrap()
                .permute(&[0, 2, 1, 3])
                .unwrap()
        } else {
            root
        };
        if index == 1 {
            view = view.permute(&[0, 1, 3, 2]).unwrap();
        }
        Descriptor {
            pointer: inputs[index].device_ptr(&stream).0,
            storage_len: n,
            storage_dtype: DType::BF16,
            source_shape,
            view,
        }
    });
    let plan = UncompiledPair::with_transport(shape, descriptions, Transport::Async16)?;
    ensure!(
        plan.transport == Transport::Async16,
        "aligned fresh allocations must admit ONE complete Async16 pair"
    );
    let [q, k, v, incoming] = inputs.each_ref().map(|x| x.device_ptr(&stream).0);
    let [o, dq, dk, dv] = outputs.each_ref().map(|x| x.device_ptr(&stream).0);
    let stats = statistics.device_ptr(&stream).0;
    let mut modes = Vec::new();
    let mut resource = Vec::new();
    for transport in [Transport::Scalar, plan.transport] {
        let irs = [
            forward::ir_with_transport(shape, layout, transport)?,
            backward::d_query_ir_with_transport(shape, layout, layout, transport)?,
            backward::d_key_value_ir_with_transport(shape, layout, layout, transport)?,
        ];
        let mut functions = Vec::new();
        for (slot, (ir, name)) in irs
            .into_iter()
            .zip(["forward", "dquery", "dkeyvalue"])
            .enumerate()
        {
            ir.validate_layouts()?;
            if transport == Transport::Scalar && layout == ReadLayout::SequenceMajor {
                let filename = [
                    "online-forward.ptx",
                    "online-dquery.ptx",
                    "online-dkeyvalue.ptx",
                ][slot];
                assert_eq!(
                    ptx(ir.clone())?,
                    std::fs::read_to_string(golden.join(filename))?,
                    "old scalar golden changed before resource admission"
                );
            }
            let function = compile(&stream, ir, &format!("transport-{transport:?}-{name}"))?;
            resource.push(serde_json::json!({"transport":format!("{transport:?}"),"function":resources(&function,name)?}));
            functions.push(function);
        }
        let mut functions = functions.into_iter();
        let kernels = [
            kernel(
                functions.next().unwrap(),
                shape,
                vec![q, k, v, o, stats],
                "transport forward",
            )?,
            kernel(
                functions.next().unwrap(),
                shape,
                vec![q, k, v, o, incoming, stats, dq],
                "transport dQ+delta",
            )?,
            kernel(
                functions.next().unwrap(),
                shape,
                vec![q, k, v, incoming, stats, dk, dv],
                "transport dKT+dV",
            )?,
        ];
        modes.push((transport, kernels));
    }
    let directory = std::env::var_os("TNSR_ONLINE_ATTENTION_AUDIT").map(std::path::PathBuf::from);
    if let Some(path) = &directory {
        std::fs::create_dir_all(path)?;
        std::fs::write(
            path.join("async16-resources.json"),
            serde_json::to_vec_pretty(
                &serde_json::json!({"functions":resource,"leaf_kernels_launched":false}),
            )?,
        )?;
    }
    ensure!(
        resource
            .iter()
            .all(|r| r["function"]["strict_production_resource_admitted"] == true),
        "ALL Scalar/Async16 functions must pass strict resources before ANY leaf launch"
    );
    let captures = modes
        .iter()
        .map(|(_, kernels)| capture(&stream, kernels))
        .collect::<Result<Vec<_>>>()?;
    let mut revisions = Vec::new();
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
            // BF16 subnormals and minimum normals exercise the old conversion
            // roundtrip against raw transport without changing arithmetic.
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
        let mut expected: Option<[Vec<bf16>; 4]> = None;
        let mut expected_stats: Option<Vec<f32>> = None;
        let mut mode_records = Vec::new();
        for (index, captured) in captures.iter().enumerate() {
            let sentinel = vec![bf16::NAN; n];
            for target in &mut outputs {
                stream.memcpy_htod(&sentinel, target)?;
            }
            stream.memcpy_htod(&vec![f32::NAN; shape.bh() * 512 * 3], &mut statistics)?;
            captured.graph.as_ref().unwrap().launch()?;
            let first = read(&stream, &outputs)?;
            let first_stats = stream.memcpy_dtov(&statistics)?;
            if revision <= 2 || revision == 7 {
                ensure!(
                    first.iter().flatten().all(|x| x.is_finite())
                        && first_stats.iter().all(|x| x.is_finite()),
                    "finite transport fixture must overwrite all outputs/statistics"
                );
            }
            captured.graph.as_ref().unwrap().launch()?;
            let second = read(&stream, &outputs)?;
            let second_stats = stream.memcpy_dtov(&statistics)?;
            for (a, b) in first.iter().zip(&second) {
                finite_bits(a, b, "complete transport self replay");
            }
            for (&a, &b) in first_stats.iter().zip(&second_stats) {
                ensure!(
                    (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits(),
                    "statistics self replay mismatch"
                );
            }
            if let Some(reference) = &expected {
                for (a, b) in first.iter().zip(reference) {
                    finite_bits(a, b, "complete Async16 vs Scalar finite bits/NaN classes");
                }
                for (&a, &b) in first_stats.iter().zip(expected_stats.as_ref().unwrap()) {
                    ensure!(
                        (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits(),
                        "Async16 statistic differs from Scalar"
                    );
                }
            }
            let classes:Vec<_>=first.iter().map(|x| {
                let values:Vec<_>=x.iter().map(|x|f64::from(x.to_f32())).collect();let c=super::super::oracle::classify(&values);
                serde_json::json!({"finite":c.finite,"nan":c.nan,"positive_infinity":c.positive_infinity,"negative_infinity":c.negative_infinity})
            }).collect();
            mode_records.push(serde_json::json!({"transport":format!("{:?}",modes[index].0),"replay_exact_finite_bits_nan_classes":true,"classes":classes}));
            if index == 0 {
                expected = Some(first);
                expected_stats = Some(first_stats);
            }
        }
        for (source, reference) in inputs.iter().zip(&physical_inputs) {
            finite_bits(
                reference,
                &stream.memcpy_dtov(source)?,
                "transport inputs remain read-only",
            );
        }
        revisions.push(serde_json::json!({"revision":revision,"modes":mode_records,"all_four_outputs_and_three_stats_fields_match":true}));
        if let Some(path) = &directory {
            std::fs::write(
                path.join("async16-qualification.json"),
                serde_json::to_vec_pretty(&serde_json::json!({
                    "layout":format!("{layout:?}"),"transport_selection":format!("{:?}",plan.transport),"resources":resource,"revisions":revisions,
                    "scope":"disconnected whole-pair raw BF16 transport; finite bits and NaN classes, not NaN payload equality",
                    "model_qualification":false,"performance_claim":false
                }))?,
            )?;
        }
    }
    stream.synchronize()?;
    Ok(())
}
