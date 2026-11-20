//! Complete-pair resource audit and numerical diagnostics. No model hook.
use super::{
    backward, forward,
    leaf::{Descriptor, PairedFunctions, ReadLayout, Shape, UncompiledPair},
    oracle,
};
use crate::{
    graph::{NodeIndex, TensorGraph},
    ptx::{
        PtxGraph, PtxTarget,
        linear::{Captured, program::Kernel},
    },
    tile::{DType, TileGraph, TileIR, VirtualTensor},
};
use anyhow::{Context, Result, ensure};
use cudarc::{
    driver::{CudaContext, CudaFunction, CudaStream, DevicePtr, LaunchConfig, sys},
    nvrtc::Ptx,
};
use half::bf16;
use std::sync::Arc;

fn directory() -> Option<std::path::PathBuf> {
    std::env::var_os("TNSR_ONLINE_ATTENTION_AUDIT").map(Into::into)
}
fn save(name: &str, value: &impl serde::Serialize) -> Result<()> {
    if let Some(path) = directory() {
        std::fs::create_dir_all(&path)?;
        std::fs::write(path.join(name), serde_json::to_vec_pretty(value)?)?;
    }
    Ok(())
}
pub(super) fn compile(stream: &Arc<CudaStream>, ir: TileIR, tag: &str) -> Result<CudaFunction> {
    let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
    tiles
        .region_kernels
        .push(crate::tile::graph::TileRegionKernel { region_id: 0, ir });
    tiles.optimize_indices()?;
    let ptx = PtxGraph::from(tiles).with_target(PtxTarget::from_context(stream.context())?);
    let source = ptx.to_ptx();
    if let Some(path) = directory() {
        std::fs::create_dir_all(&path)?;
        std::fs::write(path.join(format!("{tag}.ptx")), &source)?;
    }
    let module = stream.context().load_module(Ptx::from_src(source))?;
    Ok(module.load_function(
        ptx.region_kernel_name(0)
            .context("online region name missing")?,
    )?)
}
pub(super) fn resources(function: &CudaFunction, name: &str) -> Result<serde_json::Value> {
    let registers = function.num_regs()?;
    let local = function.local_size_bytes()?;
    let shared = function.shared_size_bytes()?;
    let capacity = function.occupancy_max_active_blocks_per_multiprocessor(256, 0, None)?;
    let base = local == 0 && shared == 49152 && function.max_threads_per_block()? >= 256;
    Ok(
        serde_json::json!({"kernel":name,"registers":registers,"local":local,"shared":shared,"block":[32,8,1],"api_capacity":capacity,"diagnostic_admitted":base && registers<=192 && capacity>=1,"strict_production_resource_admitted":base && registers<=128 && capacity>=2,"scope":"actual driver resources; API capacity is not achieved occupancy"}),
    )
}
pub(super) fn kernel(
    function: CudaFunction,
    shape: Shape,
    pointers: Vec<u64>,
    label: &str,
) -> Result<Kernel> {
    Ok(Kernel {
        owner: None,
        label: label.into(),
        function,
        config: LaunchConfig {
            grid_dim: (8, u32::try_from(shape.bh())?, 1),
            block_dim: (32, 8, 1),
            shared_mem_bytes: 0,
        },
        pointers,
        zero_before: None,
    })
}
pub(super) fn physical(logical: &[bf16], shape: Shape, layout: ReadLayout) -> Vec<bf16> {
    if layout == ReadLayout::HeadMajor {
        return logical.to_vec();
    }
    let mut values = vec![bf16::ZERO; logical.len()];
    for head in 0..shape.bh() {
        for row in 0..512 {
            for feature in 0..64 {
                let source = (head * 512 + row) * 64 + feature;
                let dest = ((head / shape.heads * 512 + row) * shape.heads + head % shape.heads)
                    * 64
                    + feature;
                values[dest] = logical[source];
            }
        }
    }
    values
}
pub(super) fn finite_bits(expected: &[bf16], actual: &[bf16], name: &str) {
    assert_eq!(expected.len(), actual.len());
    if let Some(i) = expected
        .iter()
        .zip(actual)
        .position(|(a, b)| !(a.is_nan() && b.is_nan()) && a.to_bits() != b.to_bits())
    {
        panic!(
            "{name} replay mismatch at{i}: {:?}/{:?}",
            expected[i], actual[i]
        );
    }
}
pub(super) fn metric_json(expected: &[f64], actual: &[f64]) -> Result<serde_json::Value> {
    let m = oracle::error_metrics(expected, actual)?;
    Ok(
        serde_json::json!({"finite":m.finite,"relative_l2":m.relative_l2,"max_absolute":m.max_absolute,"reference_l2":m.reference_l2,"threshold_applied":false}),
    )
}
pub(super) fn data(shape: Shape, revision: usize, salt: usize) -> Vec<bf16> {
    (0..shape.elements())
        .map(|i| {
            let row = (i / 64) % 512;
            let x = match revision {
                1 if salt == 1 => 0.25,
                1 if salt == 2 => row as f32 / 128. - 1.,
                2 => {
                    if i.is_multiple_of(2) {
                        0.
                    } else {
                        -0.
                    }
                }
                3 if salt == 1 && i == 0 => f32::NAN,
                4 if salt <= 2 => 0.,
                4 if salt == 3 => bf16::MAX.to_f32(),
                4 => 1e-30,
                _ => {
                    (((i * 13 + salt * 17) % 97) as f32 / 127. - 0.31)
                        * if salt == 4 { 0.02 } else { 1. }
                }
            };
            bf16::from_f32(x)
        })
        .collect()
}

#[test]
fn online_training_complete_pair_resource_audit_and_leaf_diagnostics() -> Result<()> {
    if !crate::ptx::mixed::tests::cuda_available() {
        return Ok(());
    }
    let shape = Shape::new(8, 8)?;
    let layout = if std::env::var("TNSR_ONLINE_LAYOUT").ok().as_deref() == Some("head") {
        ReadLayout::HeadMajor
    } else {
        ReadLayout::SequenceMajor
    };
    let context = CudaContext::new(0)?;
    // Every allocation/copy/kernel uses this private ordered stream. The whole
    // three-kernel graph retains its buffers/functions throughout every replay.
    unsafe {
        context.disable_event_tracking();
    }
    let stream = context.new_stream()?;
    let fwd = compile(&stream, forward::ir(shape, layout)?, "online-forward")?;
    let dq = compile(
        &stream,
        backward::d_query_ir(shape, layout, layout)?,
        "online-dquery",
    )?;
    let dkv = compile(
        &stream,
        backward::d_key_value_ir(shape, layout, layout)?,
        "online-dkeyvalue",
    )?;
    let resource = vec![
        resources(&fwd, "forward")?,
        resources(&dq, "dQ+delta")?,
        resources(&dkv, "dKT+dV")?,
    ];
    let admitted = resource.iter().all(|r| r["diagnostic_admitted"] == true);
    let production = resource
        .iter()
        .all(|r| r["strict_production_resource_admitted"] == true);
    save(
        "complete-pair-resources.json",
        &serde_json::json!({"functions":resource,"all_three_diagnostic_admitted":admitted,"all_three_strict_resource_admitted":production,"model_or_leaf_kernels_executed":false}),
    )?;
    ensure!(
        admitted,
        "complete online pair rejected by prototype resources; no leaf kernel executed"
    );
    if std::env::var_os("TNSR_ONLINE_RESOURCE_ONLY").is_some() {
        return Ok(());
    }
    // No forward or backward launch is possible before all three functions have
    // compiled and independently passed the same declared diagnostic resource gate.
    let functions = PairedFunctions::from_all(Some(fwd), Some(dq), Some(dkv))?;
    let n = shape.elements();
    let mut query = stream.alloc_zeros::<bf16>(n)?;
    let mut key = stream.alloc_zeros::<bf16>(n)?;
    let mut value = stream.alloc_zeros::<bf16>(n)?;
    let mut incoming = stream.alloc_zeros::<bf16>(n)?;
    let mut output = stream.alloc_zeros::<bf16>(n)?;
    let mut d_query = stream.alloc_zeros::<bf16>(n)?;
    let mut d_key = stream.alloc_zeros::<bf16>(n)?;
    let mut d_value = stream.alloc_zeros::<bf16>(n)?;
    let mut statistics = stream.alloc_zeros::<f32>(shape.bh() * 512 * 3)?;
    let describe =
        |pointer: u64, source: usize, is_key: bool, is_incoming: bool| -> Result<Descriptor> {
            let source_shape = match layout {
                ReadLayout::HeadMajor => vec![shape.bh(), 512, 64],
                ReadLayout::SequenceMajor if is_incoming => {
                    vec![shape.batch * 512, shape.heads * 64]
                }
                ReadLayout::SequenceMajor => vec![shape.batch, 512, shape.heads * 64],
            };
            let mut view =
                VirtualTensor::identity(NodeIndex::new(source), source_shape.clone(), DType::BF16);
            view = match layout {
                ReadLayout::HeadMajor => view.reshape(vec![shape.batch, shape.heads, 512, 64])?,
                ReadLayout::SequenceMajor => view
                    .reshape(vec![shape.batch, 512, shape.heads, 64])?
                    .permute(&[0, 2, 1, 3])?,
            };
            if is_key {
                view = view.permute(&[0, 1, 3, 2])?;
            }
            Ok(Descriptor {
                pointer,
                storage_len: n,
                storage_dtype: DType::BF16,
                source_shape,
                view,
            })
        };
    let abi = UncompiledPair::new(
        shape,
        [
            describe(query.device_ptr(&stream).0, 0, false, false)?,
            describe(key.device_ptr(&stream).0, 1, true, false)?,
            describe(value.device_ptr(&stream).0, 2, false, false)?,
            describe(incoming.device_ptr(&stream).0, 3, false, true)?,
        ],
    )?;
    ensure!(
        abi.read_layouts == [layout; 4],
        "prepared leaf binding layouts differ from compiled pair"
    );
    let fwd = kernel(
        functions.forward,
        shape,
        vec![
            query.device_ptr(&stream).0,
            key.device_ptr(&stream).0,
            value.device_ptr(&stream).0,
            output.device_ptr(&stream).0,
            statistics.device_ptr(&stream).0,
        ],
        "paired-online forward",
    )?;
    let dq = kernel(
        functions.d_query_and_delta,
        shape,
        vec![
            query.device_ptr(&stream).0,
            key.device_ptr(&stream).0,
            value.device_ptr(&stream).0,
            output.device_ptr(&stream).0,
            incoming.device_ptr(&stream).0,
            statistics.device_ptr(&stream).0,
            d_query.device_ptr(&stream).0,
        ],
        "paired-online dQ+delta",
    )?;
    let dkv = kernel(
        functions.d_key_value,
        shape,
        vec![
            query.device_ptr(&stream).0,
            key.device_ptr(&stream).0,
            value.device_ptr(&stream).0,
            incoming.device_ptr(&stream).0,
            statistics.device_ptr(&stream).0,
            d_key.device_ptr(&stream).0,
            d_value.device_ptr(&stream).0,
        ],
        "paired-online dKT+dV",
    )?;
    let kernels = [fwd, dq, dkv];
    stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
    let recorded = (|| -> Result<()> {
        for k in &kernels {
            k.run(&stream)?;
        }
        Ok(())
    })();
    let ended = stream.end_capture(
        sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
    );
    recorded?;
    let captured = Captured {
        graph: Some(ended?.context("empty complete-pair capture")?),
        stream: stream.clone(),
    };
    let mut revisions = Vec::new();
    for revision in 0..5 {
        let logical = [
            data(shape, revision, 1),
            data(shape, revision, 2),
            data(shape, revision, 3),
            data(shape, revision, 4),
        ];
        stream.memcpy_htod(&physical(&logical[0], shape, layout), &mut query)?;
        stream.memcpy_htod(&physical(&logical[1], shape, layout), &mut key)?;
        stream.memcpy_htod(&physical(&logical[2], shape, layout), &mut value)?;
        stream.memcpy_htod(&physical(&logical[3], shape, layout), &mut incoming)?;
        let sentinel = vec![bf16::NAN; n];
        for target in [&mut output, &mut d_query, &mut d_key, &mut d_value] {
            stream.memcpy_htod(&sentinel, target)?;
        }
        stream.memcpy_htod(&vec![f32::NAN; shape.bh() * 512 * 3], &mut statistics)?;
        captured.graph.as_ref().unwrap().launch()?;
        let first = [
            stream.memcpy_dtov(&output)?,
            stream.memcpy_dtov(&d_query)?,
            stream.memcpy_dtov(&d_key)?,
            stream.memcpy_dtov(&d_value)?,
        ];
        let first_stats = stream.memcpy_dtov(&statistics)?;
        if revision <= 2 {
            ensure!(
                first.iter().flatten().all(|x| x.is_finite())
                    && first_stats.iter().all(|x| x.is_finite()),
                "finite fixture produced incomplete/nonfinite output"
            );
        }
        if revision == 2 {
            ensure!(
                first.iter().flatten().all(|x| x.to_f32() == 0.),
                "zero fixture must have zero outputs/gradients"
            );
        }
        captured.graph.as_ref().unwrap().launch()?;
        let second = [
            stream.memcpy_dtov(&output)?,
            stream.memcpy_dtov(&d_query)?,
            stream.memcpy_dtov(&d_key)?,
            stream.memcpy_dtov(&d_value)?,
        ];
        let second_stats = stream.memcpy_dtov(&statistics)?;
        for (i, (a, b)) in first.iter().zip(&second).enumerate() {
            finite_bits(a, b, &format!("output{i}"));
        }
        for (i, (&a, &b)) in first_stats.iter().zip(&second_stats).enumerate() {
            ensure!(
                (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits(),
                "statistics replay mismatch at{i}"
            );
        }
        super::export::revision(shape, revision, &logical, &first, &first_stats)?;
        let mut sampled = Vec::new();
        if revision != 3 {
            for head in [0, shape.bh() - 1] {
                let range = head * 512 * 64..(head + 1) * 512 * 64;
                let inputs: Vec<Vec<f64>> = logical
                    .iter()
                    .map(|x| {
                        x[range.clone()]
                            .iter()
                            .map(|x| f64::from(x.to_f32()))
                            .collect()
                    })
                    .collect();
                let expected = oracle::evaluate(
                    oracle::Shape::new(1, 512, 64)?,
                    oracle::Inputs {
                        query: &inputs[0],
                        key: &inputs[1],
                        value: &inputs[2],
                        incoming: &inputs[3],
                    },
                )?;
                let actual: Vec<Vec<f64>> = first
                    .iter()
                    .enumerate()
                    .map(|(i, x)| {
                        if i == 2 {
                            (0..512)
                                .flat_map(|row| {
                                    (0..64).map(move |feature| {
                                        f64::from(x[(head * 64 + feature) * 512 + row].to_f32())
                                    })
                                })
                                .collect()
                        } else {
                            x[range.clone()]
                                .iter()
                                .map(|x| f64::from(x.to_f32()))
                                .collect()
                        }
                    })
                    .collect();
                let stats: Vec<Vec<f64>> = (0..3)
                    .map(|field| {
                        (0..512)
                            .map(|row| f64::from(first_stats[(head * 512 + row) * 3 + field]))
                            .collect()
                    })
                    .collect();
                let rounded_delta: Vec<f64> = (0..512)
                    .map(|row| {
                        (0..64)
                            .map(|feature| {
                                actual[0][row * 64 + feature] * inputs[3][row * 64 + feature]
                            })
                            .sum()
                    })
                    .collect();
                sampled.push(serde_json::json!({"head":head,"output":metric_json(&expected.output,&actual[0])?,"d_query":metric_json(&expected.d_query,&actual[1])?,"d_key":metric_json(&expected.d_key,&actual[2])?,"d_value":metric_json(&expected.d_value,&actual[3])?,"maximum":metric_json(&expected.maximum,&stats[0])?,"denominator":metric_json(&expected.denominator,&stats[1])?,"delta_ideal_O":metric_json(&expected.delta,&stats[2])?,"delta_rounded_O_f64_sum":metric_json(&rounded_delta,&stats[2])?,"oracle_quadratic_workspace_elements":expected.quadratic_workspace_elements}));
            }
        }
        let classes:Vec<_>=first.iter().map(|x| {
            let values:Vec<_>=x.iter().map(|x|f64::from(x.to_f32())).collect();let c=oracle::classify(&values);
            serde_json::json!({"finite":c.finite,"nan":c.nan,"positive_infinity":c.positive_infinity,"negative_infinity":c.negative_infinity})
        }).collect();
        revisions.push(serde_json::json!({"revision":revision,"f64_sampled_heads":sampled,"all_heads_replay_bits_or_nan_classes_match":true,"all_heads_output_classes":classes,"scope":"F64 metrics descriptive; no data-fitted thresholds or full-update acceptance"}));
        save(
            "complete-pair-leaf-diagnostics.json",
            &serde_json::json!({"shape":{"batch":8,"heads":8,"sequence":512,"width":64},"layout":format!("{layout:?}"),"resources":resource,"division_policy":"existing div.approx.f32 in forward/backward normalization; add/sub/mul explicit RN; no new IR", "strict_resource_verdict":production,"revisions":revisions,"oracle_scope":"heads0,last; all-head output/replay checks; ideal F64 VJP not exact differentiation through BF16 rounding","model_hook":false,"torch_sdpa_leaf_comparison":"pending separate independent diagnostic","full_model_gates":"unchanged and not executed","performance_claim":false}),
        )?;
    }
    stream.synchronize()?;
    Ok(())
}
