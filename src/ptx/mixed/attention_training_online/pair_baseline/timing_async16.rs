//! Reserved three-mode transport comparison; every old operation stays included.
use super::super::{
    backward, forward,
    leaf::{Descriptor, ReadLayout, Transport, UncompiledPair},
    tests::*,
};
use super::{
    gpu::{capture, logical, raw, read},
    prepare::prepare,
    timing::{median, repeated},
    *,
};
use crate::{
    ptx::linear::{Binding, program::Kernel},
    tile::{DType, VirtualTensor},
};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, sys};
use half::bf16;
use std::{
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};

const REPEATS: usize = 8;
const SAMPLES: usize = 32;
const WARMUP_MS: u64 = 500;
const LABELS: [&str; 3] = ["old", "scalar", "async16"];

fn order(sample: usize, reverse: bool) -> [usize; 3] {
    if sample.is_multiple_of(2) != reverse {
        [0, 1, 2]
    } else {
        [2, 1, 0]
    }
}
fn stats_exact(expected: &[f32], actual: &[f32]) -> Result<()> {
    ensure!(expected.len() == actual.len(), "statistics extent mismatch");
    ensure!(
        expected
            .iter()
            .zip(actual)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "finite Scalar/Async16 m/l/delta bits differ"
    );
    Ok(())
}
fn online(
    stream: &Arc<CudaStream>,
    shape: Shape,
    transport: Transport,
    inputs: [Binding; 4],
    outputs: [Binding; 4],
    stats: u64,
) -> Result<(Vec<Kernel>, serde_json::Value)> {
    let irs = [
        forward::ir_with_transport(shape, ReadLayout::SequenceMajor, transport)?,
        backward::d_query_ir_with_transport(
            shape,
            ReadLayout::SequenceMajor,
            ReadLayout::SequenceMajor,
            transport,
        )?,
        backward::d_key_value_ir_with_transport(
            shape,
            ReadLayout::SequenceMajor,
            ReadLayout::SequenceMajor,
            transport,
        )?,
    ];
    let mut functions = Vec::new();
    let mut resource = Vec::new();
    for (ir, name) in irs.into_iter().zip(["forward", "dquery", "dkeyvalue"]) {
        ir.validate_layouts()?;
        let function = compile(stream, ir, &format!("three-mode-{transport:?}-{name}"))?;
        resource.push(resources(&function, name)?);
        functions.push(function);
    }
    let [q, k, v, incoming] = inputs.map(|binding| binding.pointer);
    let [o, dq, dk, dv] = outputs.map(|binding| binding.pointer);
    let mut functions = functions.into_iter();
    Ok((
        vec![
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
        ],
        serde_json::json!({"transport":format!("{transport:?}"),"functions":resource}),
    ))
}

#[test]
fn async16_component_timing_order_is_abc_cba_and_reversed_construction() {
    for reverse in [false, true] {
        assert_eq!(
            (0..32)
                .filter(|&sample| order(sample, reverse) == [0, 1, 2])
                .count(),
            16
        );
        assert_eq!(
            (0..32)
                .filter(|&sample| order(sample, reverse) == [2, 1, 0])
                .count(),
            16
        );
        assert_ne!(order(0, reverse), order(1, reverse));
    }
    assert_eq!(order(0, true), [2, 1, 0]);
}

#[test]
#[ignore = "exclusive root-clean three-mode timing; two fresh reversed processes required"]
fn online_async16_complete_pair_three_mode_timing() -> Result<()> {
    ensure!(
        crate::ptx::mixed::tests::cuda_available(),
        "real CUDA device required"
    );
    let output = PathBuf::from(std::env::var("TNSR_ONLINE_TRIPLE_OUTPUT")?);
    ensure!(
        !output.exists(),
        "use a new immutable three-mode run directory"
    );
    std::fs::create_dir_all(&output)?;
    let provenance: serde_json::Value = serde_json::from_slice(&std::fs::read(std::env::var(
        "TNSR_ONLINE_TRIPLE_PROVENANCE",
    )?)?)?;
    let exports = PathBuf::from(std::env::var("TNSR_ONLINE_PAIR_REFERENCE_EXPORTS")?);
    let reverse = match std::env::var("TNSR_ONLINE_TRIPLE_ORDER").as_deref() {
        Ok("forward") => false,
        Ok("reverse") => true,
        _ => {
            anyhow::bail!("explicit forward/reverse construction, cases and ABC/CBA order required")
        }
    };
    let case_order = if reverse { [32, 16, 8] } else { [8, 16, 32] };
    let construction = order(0, reverse);
    let transport_order = if reverse {
        [Transport::Async16, Transport::Scalar]
    } else {
        [Transport::Scalar, Transport::Async16]
    };
    let mut results = Vec::new();
    for batch in case_order {
        let shape = Shape::new(batch, 8)?;
        let n = shape.elements();
        let exact_checkpoint = shape.bh() >= 128;
        let context = CudaContext::new(0)?;
        // One private ordered stream, raw kernel pointers, and every graph's
        // allocation/function owners retained until after synchronized checks.
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
        let mut outputs = [
            stream.alloc_zeros::<bf16>(n)?,
            stream.alloc_zeros(n)?,
            stream.alloc_zeros(n)?,
            stream.alloc_zeros(n)?,
        ];
        let mut statistics = stream.alloc_zeros::<f32>(shape.bh() * 512 * 3)?;
        let binding = |slice: &CudaSlice<bf16>| Binding {
            pointer: slice.device_ptr(&stream).0,
            len: n,
            dtype: DType::BF16,
        };
        let input_bindings = inputs.each_ref().map(binding);
        let output_bindings = outputs.each_ref().map(binding);
        let descriptors = std::array::from_fn(|index| {
            let source_shape = if index == 3 {
                vec![batch * 512, shape.heads * 64]
            } else {
                vec![batch, 512, shape.heads * 64]
            };
            let root =
                VirtualTensor::identity(NodeIndex::new(index), source_shape.clone(), DType::BF16);
            let root = if index == 3 {
                root.reshape(vec![batch, 512, shape.heads * 64]).unwrap()
            } else {
                root
            };
            let mut view = root
                .reshape(vec![batch, 512, shape.heads, 64])
                .unwrap()
                .permute(&[0, 2, 1, 3])
                .unwrap();
            if index == 1 {
                view = view.permute(&[0, 1, 3, 2]).unwrap();
            }
            Descriptor {
                pointer: input_bindings[index].pointer,
                storage_len: n,
                storage_dtype: DType::BF16,
                source_shape,
                view,
            }
        });
        let plan = UncompiledPair::with_transport(shape, descriptors, Transport::Async16)?;
        ensure!(
            plan.transport == Transport::Async16
                && plan.read_layouts == [ReadLayout::SequenceMajor; 4],
            "actual Async16 whole-pair alignment/canonical binding admission required, no relabeled fallback"
        );
        let mut built: [Option<Vec<Kernel>>; 2] = [None, None];
        let mut resources = Vec::new();
        for transport in transport_order {
            let (kernels, resource) = online(
                &stream,
                shape,
                transport,
                input_bindings,
                output_bindings,
                statistics.device_ptr(&stream).0,
            )?;
            built[usize::from(transport == Transport::Async16)] = Some(kernels);
            resources.push(resource);
        }
        ensure!(
            resources.iter().all(|r| r["functions"]
                .as_array()
                .unwrap()
                .iter()
                .all(|f| f["strict_production_resource_admitted"] == true)),
            "all SIX functions must admit before any component setup launch"
        );
        let (baseline, mut reference) = if reverse {
            let reference = prepare(
                shape,
                validation_graph(shape)?,
                false,
                input_bindings,
                output_bindings,
            )?;
            let baseline = prepare(
                shape,
                graph(shape)?,
                exact_checkpoint,
                input_bindings,
                output_bindings,
            )?;
            (baseline, reference)
        } else {
            let baseline = prepare(
                shape,
                graph(shape)?,
                exact_checkpoint,
                input_bindings,
                output_bindings,
            )?;
            let reference = prepare(
                shape,
                validation_graph(shape)?,
                false,
                input_bindings,
                output_bindings,
            )?;
            (baseline, reference)
        };
        let scalar = built[0].take().unwrap();
        let asynchronous = built[1].take().unwrap();
        let kernel_sets = [
            baseline.kernels.as_slice(),
            scalar.as_slice(),
            asynchronous.as_slice(),
        ];
        let mut captures = [None, None, None];
        for mode in construction {
            captures[mode] = Some(repeated(&stream, kernel_sets[mode])?);
        }
        let logical_inputs = [
            data(shape, 0, 1),
            data(shape, 0, 2),
            data(shape, 0, 3),
            data(shape, 0, 4),
        ];
        let physical_inputs = logical_inputs
            .each_ref()
            .map(|values| physical(values, shape, ReadLayout::SequenceMajor));
        for (target, values) in inputs.iter_mut().zip(&physical_inputs) {
            stream.memcpy_htod(values, target)?;
        }
        reference.owner.upload(
            ["q", "k", "v", "do"]
                .into_iter()
                .zip(&physical_inputs)
                .map(|(name, values)| (name.into(), values.iter().map(|x| x.to_f32()).collect()))
                .collect(),
        )?;
        reference.owner.execution_stream().synchronize()?;
        let full_dot = capture(&stream, &reference.kernels)?;
        full_dot.graph.as_ref().unwrap().launch()?;
        let full_expected = read(&stream, &outputs)?;
        ensure!(
            full_expected.iter().flatten().all(|x| x.is_finite()),
            "finite old full-dot reference required"
        );
        let mut expected: [Option<[Vec<bf16>; 4]>; 3] = [None, None, None];
        let mut expected_stats: [Option<Vec<f32>>; 3] = [None, None, None];
        for mode in construction {
            let sentinel = vec![bf16::NAN; n];
            for target in &mut outputs {
                stream.memcpy_htod(&sentinel, target)?;
            }
            if mode > 0 {
                stream.memcpy_htod(&vec![f32::NAN; shape.bh() * 512 * 3], &mut statistics)?;
            }
            captures[mode]
                .as_ref()
                .unwrap()
                .graph
                .as_ref()
                .unwrap()
                .launch()?;
            let actual = read(&stream, &outputs)?;
            ensure!(
                actual.iter().flatten().all(|x| x.is_finite()),
                "all four complete pre-timing outputs must be finite"
            );
            if mode == 0 {
                for (a, b) in full_expected.iter().zip(&actual) {
                    finite_bits(a, b, "strong old component versus full-dot precheck");
                }
            } else {
                let stats = stream.memcpy_dtov(&statistics)?;
                ensure!(
                    stats.iter().all(|x| x.is_finite()),
                    "all m/l/delta fields overwritten/finite at every shape"
                );
                expected_stats[mode] = Some(stats);
            }
            expected[mode] = Some(actual);
        }
        for (a, b) in expected[1]
            .as_ref()
            .unwrap()
            .iter()
            .zip(expected[2].as_ref().unwrap())
        {
            finite_bits(
                a,
                b,
                "full Scalar/Async16 pre-timing outputs at all three sizes",
            );
        }
        stats_exact(
            expected_stats[1].as_ref().unwrap(),
            expected_stats[2].as_ref().unwrap(),
        )?;
        for (slot, name) in ["output", "d_query", "d_key", "d_value"].iter().enumerate() {
            let gold = raw(&exports.join("revision-0").join(format!("{name}.bf16")))?;
            ensure!(
                gold.len() == 64 * 512 * 64,
                "native prefix must be exactly64heads"
            );
            let values = logical(&expected[1].as_ref().unwrap()[slot], shape, slot);
            let prefix: Vec<_> = values[..gold.len()]
                .iter()
                .map(|x| bf16::from_f32(*x as f32))
                .collect();
            finite_bits(&gold, &prefix, "qualified Scalar/native prefix");
        }
        for (slot, name) in ["query", "key", "value", "incoming"].iter().enumerate() {
            let gold = raw(&exports.join("revision-0").join(format!("{name}.bf16")))?;
            finite_bits(
                &gold,
                &logical_inputs[slot][..gold.len()],
                "qualified fixed input prefix",
            );
        }
        let started = Instant::now();
        let mut warm_samples = 0;
        while started.elapsed() < Duration::from_millis(WARMUP_MS) {
            for mode in order(warm_samples, reverse) {
                captures[mode]
                    .as_ref()
                    .unwrap()
                    .graph
                    .as_ref()
                    .unwrap()
                    .launch()?;
            }
            stream.synchronize()?;
            warm_samples += 1;
        }
        let warm_ms = started.elapsed().as_secs_f64() * 1000.;
        let mut samples = Vec::new();
        let mut old_us = Vec::new();
        let mut scalar_us = Vec::new();
        let mut async_us = Vec::new();
        for sample in 0..SAMPLES {
            let sequence = order(sample, reverse);
            let mut times = [0.; 3];
            for mode in sequence {
                let start = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
                captures[mode]
                    .as_ref()
                    .unwrap()
                    .graph
                    .as_ref()
                    .unwrap()
                    .launch()?;
                let end = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
                times[mode] = f64::from(start.elapsed_ms(&end)?) * 1000. / REPEATS as f64;
                ensure!(
                    times[mode].is_finite() && times[mode] > 0.,
                    "invalid complete three-mode timing"
                );
            }
            old_us.push(times[0]);
            scalar_us.push(times[1]);
            async_us.push(times[2]);
            samples.push(serde_json::json!({"sample":sample,"order":sequence.map(|index|LABELS[index]),"old_us":times[0],"scalar_us":times[1],"async16_us":times[2]}));
        }
        stream.synchronize()?;
        for mode in construction {
            captures[mode]
                .as_ref()
                .unwrap()
                .graph
                .as_ref()
                .unwrap()
                .launch()?;
            let actual = read(&stream, &outputs)?;
            ensure!(
                actual.iter().flatten().all(|x| x.is_finite()),
                "all post-timing output elements must be finite"
            );
            for (a, b) in expected[mode].as_ref().unwrap().iter().zip(&actual) {
                finite_bits(a, b, "full pre/post three-mode output proof");
            }
            if mode > 0 {
                let stats = stream.memcpy_dtov(&statistics)?;
                ensure!(
                    stats.iter().all(|x| x.is_finite()),
                    "all post m/l/delta elements finite"
                );
                stats_exact(expected_stats[1].as_ref().unwrap(), &stats)?;
            }
        }
        for (input, values) in inputs.iter().zip(&physical_inputs) {
            finite_bits(
                values,
                &stream.memcpy_dtov(input)?,
                "all three modes preserve resident input bits",
            );
        }
        results.push(serde_json::json!({"batch":batch,"heads":8,"old_baseline":if exact_checkpoint{"old-checkpoint-fused"}else{"old-ordinary"},
            "old_metadata":baseline.metadata,"untimed_reference_metadata":reference.metadata,"online_resources":resources,
            "actual_transport":"Async16","capture_construction_order":construction.map(|index|LABELS[index]),"transport_compile_order":transport_order.map(|t|format!("{t:?}")),
            "old_reference_construction_order":if reverse{["full-dot","baseline"]}else{["baseline","full-dot"]},
            "samples":samples,"old_median_us":median(&old_us),"scalar_median_us":median(&scalar_us),"async16_median_us":median(&async_us),
            "old_over_scalar":median(&old_us)/median(&scalar_us),"old_over_async16":median(&old_us)/median(&async_us),"scalar_over_async16":median(&scalar_us)/median(&async_us),
            "repeats":REPEATS,"explicit_captured_kernel_launches":{"old":baseline.kernels.len()*REPEATS,"scalar":24,"async16":24},
            "actual_warm_ms":warm_ms,"warm_samples":warm_samples,"full_four_outputs_pre_post_exact":true,"full_scalar_async_stats_pre_post_exact":true,"all_inputs_readonly":true,
            "memory_scope":"old+full-dot owners includingunused source/output buffers retained; common fully overwritten outputs, online stats; no memory-saving measurement"}));
        std::fs::write(
            output.join("progress.json"),
            serde_json::to_vec_pretty(&serde_json::json!({"terminal":false,"cases":results}))?,
        )?;
        stream.synchronize()?;
    }
    std::fs::write(
        output.join("results.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"terminal":true,"cases":results,"case_count":3,"samples_per_case":SAMPLES,"repeats":REPEATS,"minimum_warmup_ms":WARMUP_MS,"reverse":reverse,"compiled_source_provenance":provenance,
        "protocol":"three complete modes in real ABC/CBA order; two fresh processes reverse cases/compile/construct/capture order; setup/full-dot/uploads/JIT/readback excluded",
        "scope":"resident warm component timing only; no native-model/performance-parity or memory-saving claim"}),
        )?,
    )?;
    Ok(())
}
