//! Untimed complete-pair qualification. Timing is a separately reserved stage.
use super::super::{backward, forward, leaf::ReadLayout, oracle, tests::*};
use super::{prepare::prepare, *};
use crate::{
    ptx::linear::{Binding, Captured, program::Kernel},
    tile::DType,
};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, sys};
use half::bf16;
use std::{path::Path, sync::Arc};

pub(super) fn capture(stream: &Arc<CudaStream>, kernels: &[Kernel]) -> Result<Captured> {
    stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
    let result = (|| -> Result<()> {
        for kernel in kernels {
            kernel.run(stream)?;
        }
        Ok(())
    })();
    let ended = stream.end_capture(
        sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
    );
    result?;
    Ok(Captured {
        graph: Some(ended?.context("empty attention component capture")?),
        stream: stream.clone(),
    })
}
pub(super) fn read(
    stream: &Arc<CudaStream>,
    outputs: &[CudaSlice<bf16>; 4],
) -> Result<[Vec<bf16>; 4]> {
    Ok([
        stream.memcpy_dtov(&outputs[0])?,
        stream.memcpy_dtov(&outputs[1])?,
        stream.memcpy_dtov(&outputs[2])?,
        stream.memcpy_dtov(&outputs[3])?,
    ])
}
pub(super) fn logical(values: &[bf16], shape: Shape, output: usize) -> Vec<f64> {
    (0..shape.elements())
        .map(|i| {
            let physical = if output == 2 {
                ((i / (512 * 64)) * 64 + i % 64) * 512 + (i / 64) % 512
            } else {
                i
            };
            f64::from(values[physical].to_f32())
        })
        .collect()
}
pub(super) fn raw(path: &Path) -> Result<Vec<bf16>> {
    let bytes = std::fs::read(path)?;
    ensure!(
        bytes.len().is_multiple_of(2),
        "invalid BF16 raw payload length"
    );
    Ok(bytes
        .chunks_exact(2)
        .map(|x| bf16::from_bits(u16::from_le_bytes([x[0], x[1]])))
        .collect())
}

#[test]
fn old_attention_pair_component_external_seed_full_dot_and_online_qualification() -> Result<()> {
    if !crate::ptx::mixed::tests::cuda_available() {
        return Ok(());
    }
    let batch: usize = std::env::var("TNSR_ONLINE_COMPONENT_BATCH")
        .ok()
        .map(|x| x.parse())
        .transpose()?
        .unwrap_or(8);
    ensure!(
        [8, 16, 32].contains(&batch),
        "bounded component B8/B16/B32 only"
    );
    let shape = Shape::new(batch, 8)?;
    let context = CudaContext::new(0)?;
    // Single private ordered stream per capture; retained owners/buffers outlive
    // every graph. No event-tracked cross-stream dependency enters capture.
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
    let statistics = stream.alloc_zeros::<f32>(shape.bh() * 512 * 3)?;
    let binding = |slice: &CudaSlice<bf16>| Binding {
        pointer: slice.device_ptr(&stream).0,
        len: n,
        dtype: DType::BF16,
    };
    let input_bindings = inputs.each_ref().map(binding);
    let output_bindings = outputs.each_ref().map(binding);
    let fwd = compile(
        &stream,
        forward::ir(shape, ReadLayout::SequenceMajor)?,
        "component-online-forward",
    )?;
    let dq = compile(
        &stream,
        backward::d_query_ir(shape, ReadLayout::SequenceMajor, ReadLayout::SequenceMajor)?,
        "component-online-dq",
    )?;
    let dkv = compile(
        &stream,
        backward::d_key_value_ir(shape, ReadLayout::SequenceMajor, ReadLayout::SequenceMajor)?,
        "component-online-dkv",
    )?;
    let resource = [
        resources(&fwd, "forward")?,
        resources(&dq, "dQ+delta")?,
        resources(&dkv, "dKT+dV")?,
    ];
    ensure!(
        resource.iter().all(|r| r["diagnostic_admitted"] == true),
        "all three online kernels must admit before any component launch"
    );
    // Mixed constructors perform setup warmups, so construct even the old
    // baselines only after complete new-pair resource admission.
    let ordinary = prepare(shape, graph(shape)?, false, input_bindings, output_bindings)?;
    let mut reference = prepare(
        shape,
        validation_graph(shape)?,
        false,
        input_bindings,
        output_bindings,
    )?;
    let checkpoint = if shape.bh() >= 128 {
        Some(prepare(
            shape,
            graph(shape)?,
            true,
            input_bindings,
            output_bindings,
        )?)
    } else {
        None
    };
    let q = input_bindings[0].pointer;
    let k = input_bindings[1].pointer;
    let v = input_bindings[2].pointer;
    let incoming = input_bindings[3].pointer;
    let o = output_bindings[0].pointer;
    let dquery = output_bindings[1].pointer;
    let dkey = output_bindings[2].pointer;
    let dvalue = output_bindings[3].pointer;
    let stats = statistics.device_ptr(&stream).0;
    let online = [
        kernel(
            fwd,
            shape,
            vec![q, k, v, o, stats],
            "online complete forward",
        )?,
        kernel(
            dq,
            shape,
            vec![q, k, v, o, incoming, stats, dquery],
            "online complete dQ+delta",
        )?,
        kernel(
            dkv,
            shape,
            vec![q, k, v, incoming, stats, dkey, dvalue],
            "online complete dKT+dV",
        )?,
    ];
    let reference_capture = capture(&stream, &reference.kernels)?;
    let mut captures = vec![
        ("old-ordinary", capture(&stream, &ordinary.kernels)?),
        ("online", capture(&stream, &online)?),
    ];
    if let Some(checkpoint) = &checkpoint {
        captures.push((
            "old-checkpoint-fused",
            capture(&stream, &checkpoint.kernels)?,
        ));
    }
    let directory = std::env::var_os("TNSR_ONLINE_ATTENTION_AUDIT").map(std::path::PathBuf::from);
    let mut revisions = Vec::new();
    for revision in 0..5 {
        let logical_inputs = [
            data(shape, revision, 1),
            data(shape, revision, 2),
            data(shape, revision, 3),
            data(shape, revision, 4),
        ];
        for (target, source) in inputs.iter_mut().zip(&logical_inputs) {
            stream.memcpy_htod(&physical(source, shape, ReadLayout::SequenceMajor), target)?;
        }
        // Independent full-dot reference retains and EXECUTES its original
        // F32->BF16 narrowing kernels. Exact BF16 values widened for host upload
        // exercise the removed identity casts as well as external seed pruning.
        let feeds = ["q", "k", "v", "do"]
            .into_iter()
            .zip(&logical_inputs)
            .map(|(name, values)| {
                (
                    name.into(),
                    physical(values, shape, ReadLayout::SequenceMajor)
                        .into_iter()
                        .map(bf16::to_f32)
                        .collect(),
                )
            })
            .collect();
        reference.owner.upload(feeds)?;
        reference.owner.execution_stream().synchronize()?;
        let reset = |targets: &mut [CudaSlice<bf16>; 4]| -> Result<()> {
            let sentinel = vec![bf16::NAN; n];
            for target in targets {
                stream.memcpy_htod(&sentinel, target)?;
            }
            Ok(())
        };
        reset(&mut outputs)?;
        reference_capture.graph.as_ref().unwrap().launch()?;
        let expected = read(&stream, &outputs)?;
        let mut modes = Vec::new();
        for (name, captured) in &captures {
            reset(&mut outputs)?;
            captured.graph.as_ref().unwrap().launch()?;
            let first = read(&stream, &outputs)?;
            captured.graph.as_ref().unwrap().launch()?;
            let second = read(&stream, &outputs)?;
            for (slot, (a, b)) in first.iter().zip(&second).enumerate() {
                finite_bits(a, b, &format!("{name} replay output{slot}"));
            }
            if *name != "online" {
                for (slot, (a, b)) in expected.iter().zip(&first).enumerate() {
                    finite_bits(
                        a,
                        b,
                        &format!("{name} fullDot VJP output{slot} revision{revision}"),
                    );
                }
            }
            let mut metrics = Vec::new();
            for slot in 0..4 {
                metrics.push(metric_json(
                    &logical(&expected[slot], shape, slot),
                    &logical(&first[slot], shape, slot),
                )?);
            }
            let mut f64 = Vec::new();
            if revision <= 2 {
                let oracle_inputs: Vec<Vec<f64>> = logical_inputs
                    .iter()
                    .map(|x| {
                        x[..512 * 64]
                            .iter()
                            .map(|x| f64::from(x.to_f32()))
                            .collect()
                    })
                    .collect();
                let exact = oracle::evaluate(
                    oracle::Shape::new(1, 512, 64)?,
                    oracle::Inputs {
                        query: &oracle_inputs[0],
                        key: &oracle_inputs[1],
                        value: &oracle_inputs[2],
                        incoming: &oracle_inputs[3],
                    },
                )?;
                for (slot, reference) in
                    [&exact.output, &exact.d_query, &exact.d_key, &exact.d_value]
                        .into_iter()
                        .enumerate()
                {
                    f64.push(metric_json(
                        reference,
                        &logical(&first[slot], shape, slot)[..512 * 64],
                    )?);
                }
            }
            let mut torch_metrics = Vec::new();
            if let (Ok(exports), Ok(torch)) = (
                std::env::var("TNSR_ONLINE_COMPONENT_REFERENCE_EXPORTS"),
                std::env::var("TNSR_ONLINE_COMPONENT_REFERENCE_TORCH"),
            ) {
                let base = Path::new(&exports).join(format!("revision-{revision}"));
                let torch = Path::new(&torch).join(format!("revision-{revision}"));
                let count = 64 * 512 * 64;
                for (slot, name) in ["query", "key", "value", "incoming"].iter().enumerate() {
                    let gold = raw(&base.join(format!("{name}.bf16")))?;
                    ensure!(
                        gold.len() == count,
                        "qualified reference prefix must be exactly BH64"
                    );
                    finite_bits(
                        &gold,
                        &logical_inputs[slot][..count],
                        "reference canonical input prefix",
                    );
                }
                for (slot, output) in ["output", "d_query", "d_key", "d_value"].iter().enumerate() {
                    let actual = logical(&first[slot], shape, slot);
                    if *name == "online" {
                        let native = raw(&base.join(format!("{output}.bf16")))?;
                        let prefix: Vec<_> = actual[..count]
                            .iter()
                            .map(|x| bf16::from_f32(*x as f32))
                            .collect();
                        finite_bits(
                            &native,
                            &prefix,
                            "current online matches qualified reference prefix",
                        );
                    }
                    let reference = raw(&torch.join(format!("{output}.bf16")))?;
                    let reference: Vec<_> =
                        reference.iter().map(|x| f64::from(x.to_f32())).collect();
                    torch_metrics.push(metric_json(&reference, &actual[..count])?);
                }
            }
            modes.push(serde_json::json!({"mode":name,"replay_exact_bits_or_nan_classes":true,
                "old_full_dot_exact_bits_or_nan_classes":*name!="online","vs_full_dot":metrics,
                "f64_head0":f64,"torch_qualified_BH64_prefix":torch_metrics,"threshold_applied":false}));
        }
        revisions.push(serde_json::json!({"revision":revision,"modes":modes}));
        if let Some(path) = &directory {
            std::fs::create_dir_all(path)?;
            std::fs::write(
                path.join(format!("complete-pair-component-b{batch}.json")),
                serde_json::to_vec_pretty(&serde_json::json!({
                    "batch":batch,"heads":8,"ordinary":ordinary.metadata,"full_dot_untimed_reference":reference.metadata,
                    "checkpoint_fused":checkpoint.as_ref().map(|p|&p.metadata),"online_resources":resource,
                    "online_kernel_count":3,"revisions":revisions,"performance_claim":false,"full_model_qualification":false,
                    "reference_scope":"full-dot actual AD graph includes dummy forward loss work only for untimed equivalence; no dummy work in ordinary/checkpoint timing baselines",
                    "torch_scope":"optional exact BH64 input/output prefix linkage to previously profiled FLASH forward/backward leaf; not a new Torch backend run"
                }))?,
            )?;
        }
    }
    // Observe owners here explicitly; capture objects drop first, then their
    // modules/allocations remain owned until the end of this private scope.
    ensure!(
        ordinary.owner.info().kernels > 0 && reference.owner.info().kernels > 0,
        "baseline owners missing"
    );
    stream.synchronize()?;
    Ok(())
}
