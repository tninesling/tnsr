//! Explicitly reserved, ignored complete-pair timing; no model-performance claim.
use super::super::{backward, forward, leaf::ReadLayout, tests::*};
use super::{
    gpu::{capture, logical, raw, read},
    prepare::prepare,
    *,
};
use crate::{
    ptx::linear::{Binding, Captured, program::Kernel},
    tile::DType,
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

fn order(sample: usize, reverse_start: bool) -> [usize; 2] {
    if sample.is_multiple_of(2) != reverse_start {
        [0, 1]
    } else {
        [1, 0]
    }
}
fn cases(reverse: bool) -> Vec<(usize, bool)> {
    let mut result = vec![(8, false), (16, false), (16, true), (32, false), (32, true)];
    if reverse {
        result.reverse();
    }
    result
}
pub(super) fn median(values: &[f64]) -> f64 {
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    (values[values.len() / 2 - 1] + values[values.len() / 2]) / 2.
}
pub(super) fn repeated(stream: &Arc<CudaStream>, kernels: &[Kernel]) -> Result<Captured> {
    ensure!(
        kernels.iter().all(|k| k.zero_before.is_none()),
        "whole-kernel launch count requires no hidden zero/memset work"
    );
    let repeated: Vec<_> = (0..REPEATS).flat_map(|_| kernels.iter().cloned()).collect();
    capture(stream, &repeated)
}

#[test]
fn complete_pair_component_timing_order_is_genuine_ab_ba_and_five_comparisons() {
    assert_eq!(cases(false).len(), 5);
    assert_eq!(
        cases(true),
        cases(false).into_iter().rev().collect::<Vec<_>>()
    );
    for start in [false, true] {
        assert_ne!(order(0, start), order(1, start));
        assert_eq!(
            (0..SAMPLES).filter(|&i| order(i, start) == [0, 1]).count(),
            16
        );
        assert_eq!(
            (0..SAMPLES).filter(|&i| order(i, start) == [1, 0]).count(),
            16
        );
    }
    assert_ne!(order(0, false), order(0, true));
}

#[test]
#[ignore = "requires explicit root clean timing reservation; two fresh reversed processes"]
fn online_training_complete_pair_component_paired_timing() -> Result<()> {
    ensure!(
        crate::ptx::mixed::tests::cuda_available(),
        "real CUDA required"
    );
    let output = PathBuf::from(std::env::var("TNSR_ONLINE_PAIR_TIMING_OUTPUT")?);
    ensure!(
        !output.exists(),
        "timing output must be a new immutable run directory"
    );
    std::fs::create_dir_all(&output)?;
    let provenance = PathBuf::from(std::env::var("TNSR_ONLINE_PAIR_SOURCE_PROVENANCE")?);
    let source: serde_json::Value = serde_json::from_slice(&std::fs::read(&provenance)?)?;
    let exports = PathBuf::from(std::env::var("TNSR_ONLINE_PAIR_REFERENCE_EXPORTS")?);
    let reverse_start = match std::env::var("TNSR_ONLINE_PAIR_START").as_deref() {
        Ok("baseline,online") => false,
        Ok("online,baseline") => true,
        _ => anyhow::bail!("explicit baseline,online or online,baseline start order required"),
    };
    let reverse_cases = match std::env::var("TNSR_ONLINE_PAIR_CASE_ORDER").as_deref() {
        Ok("forward") => false,
        Ok("reverse") => true,
        _ => anyhow::bail!("explicit forward/reverse case order required"),
    };
    let mut results = Vec::new();
    for (batch, exact_checkpoint_forward) in cases(reverse_cases) {
        let shape = Shape::new(batch, 8)?;
        let context = CudaContext::new(0)?;
        // Every capture is single-stream and owns all allocations/functions;
        // no event-tracked allocation dependency enters the captured graph.
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
            "timing-online-forward",
        )?;
        let dq = compile(
            &stream,
            backward::d_query_ir(shape, ReadLayout::SequenceMajor, ReadLayout::SequenceMajor)?,
            "timing-online-dq",
        )?;
        let dkv = compile(
            &stream,
            backward::d_key_value_ir(shape, ReadLayout::SequenceMajor, ReadLayout::SequenceMajor)?,
            "timing-online-dkv",
        )?;
        let resource = [
            resources(&fwd, "forward")?,
            resources(&dq, "dQ+delta")?,
            resources(&dkv, "dKT+dV")?,
        ];
        ensure!(
            resource
                .iter()
                .all(|r| r["strict_production_resource_admitted"] == true),
            "resource change requires review before any timing"
        );
        let baseline = prepare(
            shape,
            graph(shape)?,
            exact_checkpoint_forward,
            input_bindings,
            output_bindings,
        )?;
        let mut reference = prepare(
            shape,
            validation_graph(shape)?,
            false,
            input_bindings,
            output_bindings,
        )?;
        let [q, k, v, incoming] = input_bindings.map(|b| b.pointer);
        let [o, dquery, dkey, dvalue] = output_bindings.map(|b| b.pointer);
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
        let logical_inputs = [
            data(shape, 0, 1),
            data(shape, 0, 2),
            data(shape, 0, 3),
            data(shape, 0, 4),
        ];
        let physical_inputs = logical_inputs
            .each_ref()
            .map(|x| physical(x, shape, ReadLayout::SequenceMajor));
        for (target, values) in inputs.iter_mut().zip(&physical_inputs) {
            stream.memcpy_htod(values, target)?;
        }
        reference.owner.upload(
            ["q", "k", "v", "do"]
                .into_iter()
                .zip(&physical_inputs)
                .map(|(name, x)| (name.into(), x.iter().map(|x| x.to_f32()).collect()))
                .collect(),
        )?;
        reference.owner.execution_stream().synchronize()?;
        let full_dot = capture(&stream, &reference.kernels)?;
        full_dot.graph.as_ref().unwrap().launch()?;
        let full_expected = read(&stream, &outputs)?;
        ensure!(
            full_expected.iter().flatten().all(|x| x.is_finite()),
            "finite full-dot reference required"
        );
        let captures = [
            repeated(&stream, &baseline.kernels)?,
            repeated(&stream, &online)?,
        ];
        let mut expected = Vec::new();
        for (mode, captured) in captures.iter().enumerate() {
            let sentinel = vec![bf16::NAN; n];
            for target in &mut outputs {
                stream.memcpy_htod(&sentinel, target)?;
            }
            captured.graph.as_ref().unwrap().launch()?;
            let values = read(&stream, &outputs)?;
            ensure!(
                values.iter().flatten().all(|x| x.is_finite()),
                "finite pre-timing outputs required for both complete modes"
            );
            if mode == 0 {
                for (full, actual) in full_expected.iter().zip(&values) {
                    finite_bits(full, actual, "timing old baseline full-dot precheck");
                }
            } else {
                for (slot, name) in ["output", "d_query", "d_key", "d_value"].iter().enumerate() {
                    let gold = raw(&exports.join("revision-0").join(format!("{name}.bf16")))?;
                    ensure!(
                        gold.len() == 64 * 512 * 64,
                        "qualified native prefix must be exactly BH64"
                    );
                    let actual = logical(&values[slot], shape, slot);
                    let prefix: Vec<_> = actual[..gold.len()]
                        .iter()
                        .map(|x| bf16::from_f32(*x as f32))
                        .collect();
                    finite_bits(&gold, &prefix, "timing online qualified prefix precheck");
                }
            }
            expected.push(values);
        }
        for (slot, name) in ["query", "key", "value", "incoming"].iter().enumerate() {
            let gold = raw(&exports.join("revision-0").join(format!("{name}.bf16")))?;
            finite_bits(
                &gold,
                &logical_inputs[slot][..gold.len()],
                "timing identical qualified input prefix",
            );
        }
        // ALL setup/JIT/uploads/captures/full-output/full-dot reference work above is
        // excluded. Warm both finished graphs with true alternating order.
        let warm = Instant::now();
        let mut warm_pairs = 0;
        while warm.elapsed() < Duration::from_millis(WARMUP_MS) {
            for mode in order(warm_pairs, reverse_start) {
                captures[mode].graph.as_ref().unwrap().launch()?;
            }
            stream.synchronize()?;
            warm_pairs += 1;
        }
        let warm_ms = warm.elapsed().as_secs_f64() * 1000.;
        let mut samples = Vec::new();
        let mut baseline_us = Vec::new();
        let mut online_us = Vec::new();
        for sample in 0..SAMPLES {
            let sequence = order(sample, reverse_start);
            let mut times = [0.; 2];
            for mode in sequence {
                let start = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
                captures[mode].graph.as_ref().unwrap().launch()?;
                let end = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
                times[mode] = f64::from(start.elapsed_ms(&end)?) * 1000. / REPEATS as f64;
                ensure!(
                    times[mode].is_finite() && times[mode] > 0.,
                    "invalid complete-component event measurement"
                );
            }
            baseline_us.push(times[0]);
            online_us.push(times[1]);
            samples.push(serde_json::json!({"pair":sample,"order":sequence.map(|i|if i==0{"baseline"}else{"online"}),
                "baseline_us":times[0],"online_us":times[1],"throughput_ratio":times[0]/times[1]}));
        }
        stream.synchronize()?;
        for (mode, captured) in captures.iter().enumerate() {
            captured.graph.as_ref().unwrap().launch()?;
            let values = read(&stream, &outputs)?;
            ensure!(
                values.iter().flatten().all(|x| x.is_finite()),
                "finite post-timing outputs required"
            );
            for (before, after) in expected[mode].iter().zip(&values) {
                finite_bits(before, after, "timing full post/pre self-output check");
            }
        }
        for (slice, expected) in inputs.iter().zip(&physical_inputs) {
            finite_bits(
                expected,
                &stream.memcpy_dtov(slice)?,
                "timing resident inputs remain read-only",
            );
        }
        results.push(serde_json::json!({"batch":batch,"heads":8,"sequence":512,"width":64,
            "baseline":if exact_checkpoint_forward{"old-checkpoint-fused"}else{"old-ordinary"},
            "baseline_metadata":baseline.metadata,"untimed_full_dot_metadata":reference.metadata,
            "online_resources":resource,"samples":samples,"repeats":REPEATS,
            "explicit_captured_kernel_launches":{"baseline":baseline.kernels.len()*REPEATS,"online":3*REPEATS},
            "graph_node_count_scope":"explicit Kernel.run count; no graph-node API query or hidden zero_before work",
            "warm_pairs":warm_pairs,"actual_warm_ms":warm_ms,"baseline_median_us":median(&baseline_us),
            "online_median_us":median(&online_us),"median_throughput_ratio":median(&baseline_us)/median(&online_us),
            "pre_post_all_four_outputs_exact":true,"all_inputs_unchanged":true,
            "memory_scope":"ALL old baseline AND full-dot owner allocations retained, including bypassed F32 inputs/BF16 outputs; shared external BF16 inputs/outputs and online stats additional; no memory-saving claim",
            "external_common_bf16_bytes":8*n*2,"online_stats_bytes":shape.bh()*512*3*4}));
        // Intermediate files are explicitly nonterminal if a later comparison fails.
        std::fs::write(
            output.join("progress.json"),
            serde_json::to_vec_pretty(
                &serde_json::json!({"terminal":false,"comparisons":results}),
            )?,
        )?;
        stream.synchronize()?;
    }
    std::fs::write(
        output.join("results.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "terminal":true,"comparisons":results,"comparison_count":5,"pairs_per_comparison":SAMPLES,
            "repeats":REPEATS,"minimum_warmup_ms":WARMUP_MS,"reverse_start":reverse_start,"reverse_cases":reverse_cases,
            "compiled_source_provenance":source,"protocol":"same resident read-only BF16 feeds, common fully overwritten outputs, CUDA-event repeated WHOLE component captures; real AB/BA only; no uploads/JIT/reference/readback/loss/input-cast work timed",
            "scope":"component timing only; no model integration, full-update qualification, memory saving or parity claim"
        }))?,
    )?;
    Ok(())
}
