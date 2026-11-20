use super::*;
use crate::{
    tensor::{Parameter, TensorExpr},
    tile::{MatMulAccumulatorExchange, MatMulFragmentLoad, MatMulTile},
};
use half::bf16;

fn same_bits(actual: &[f32], expected: &[f32], context: &str) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits(),
            "{context} index{i}: {a:?}/{:08x} != {b:?}/{:08x}",
            a.to_bits(),
            b.to_bits()
        );
    }
}
fn host_physical(canonical: &[f32], rows: usize, cols: usize, column: bool) -> Vec<f32> {
    if !column {
        return canonical.to_vec();
    }
    (0..rows * cols)
        .map(|i| canonical[(i % rows) * cols + i / rows])
        .collect()
}
fn graph(m: usize, n: usize, k: usize, orientation: (bool, bool)) -> (TensorGraph<f32>, NodeIndex) {
    let a = TensorExpr::input(
        "a",
        if orientation.0 {
            vec![k, m]
        } else {
            vec![m, k]
        },
    )
    .cast(ValueDType::BF16);
    let b = TensorExpr::input(
        "b",
        if orientation.1 {
            vec![n, k]
        } else {
            vec![k, n]
        },
    )
    .cast(ValueDType::BF16);
    let result = (if orientation.0 { a.transpose() } else { a })
        .matmul(if orientation.1 { b.transpose() } else { b })
        .cast(ValueDType::F32);
    let g: TensorGraph<f32> = result.into();
    let out = *g.toposort().last().unwrap();
    (g, out)
}

/// Independent oracle: pack each contiguous original K slice on the host,
/// launch FOUR ordinary rank-two BF16-input/F32-output GEMMs, then perform the
/// explicit binary32 left-four sum on the CPU. No batched partition addresses,
/// BF16 partials or old unsplit output are used as an oracle.
fn oracle(
    stream: &Arc<CudaStream>,
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
    orientation: (bool, bool),
) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
    let target = PtxTarget::from_context(stream.context())?;
    let part_k = k / 4;
    let mut partials = Vec::new();
    for part in 0..4 {
        let aa = (0..m * part_k)
            .map(|i| a[(i / part_k) * k + part * part_k + i % part_k])
            .collect::<Vec<_>>();
        let bb = (0..part_k * n)
            .map(|i| b[(part * part_k + i / n) * n + i % n])
            .collect::<Vec<_>>();
        let aa = host_physical(&aa, m, part_k, orientation.0)
            .into_iter()
            .map(bf16::from_f32)
            .collect::<Vec<_>>();
        let bb = host_physical(&bb, part_k, n, orientation.1)
            .into_iter()
            .map(bf16::from_f32)
            .collect::<Vec<_>>();
        let a_gpu = stream.memcpy_stod(&aa)?;
        let b_gpu = stream.memcpy_stod(&bb)?;
        let output = stream.alloc_zeros::<f32>(m * n)?;
        let mut g = TensorGraph::<bf16>::new();
        let x = g.graph.add_node(TensorGraphNode::Input {
            name: "a",
            shape: vec![m, part_k],
        });
        let y = g.graph.add_node(TensorGraphNode::Input {
            name: "b",
            shape: vec![part_k, n],
        });
        let z = g
            .graph
            .add_node(TensorGraphNode::MatMul { shape: vec![m, n] });
        g.graph.add_edge(x, z, 0);
        g.graph.add_edge(y, z, 1);
        let mut region = MatMulRegion::from_graph(&g, z, vec![], vec![z])?;
        region.schedule = MatMulSchedule::select_with_resources(
            m,
            n,
            part_k,
            DType::BF16,
            target.matmul_capabilities(),
            MatMulPrecision::NativeHalf,
            target.matmul_resources,
        );
        region.schedule.block_tile = MatMulTile {
            m: 64,
            n: 64,
            k: 32,
        };
        region.schedule.block_threads = (32, 8, 1);
        region.schedule.warp_topology = (2, 2);
        region.schedule.warp_fragments = (2, 2);
        region.schedule.reuse_shared_memory = true;
        region.schedule.accumulator_exchange = MatMulAccumulatorExchange::FullTile;
        region.schedule.fragment_load = MatMulFragmentLoad::PreloadAll;
        for (i, column) in [orientation.0, orientation.1].into_iter().enumerate() {
            let shape = match (i, column) {
                (0, false) => vec![m, part_k],
                (0, true) => vec![part_k, m],
                (1, false) => vec![part_k, n],
                (1, true) => vec![n, part_k],
                _ => unreachable!(),
            };
            let input = &mut region.inputs[i];
            input.source_shape = shape.clone();
            input.tensor = VirtualTensor::identity(input.node, shape, DType::BF16);
            if column {
                input.tensor = input.tensor.permute(&[1, 0])?;
            }
        }
        region.configure_pipeline(
            crate::tile::MatMulSharedLayout::Padded,
            crate::tile::MatMulPipeline::Auto,
            true,
            target.matmul_resources,
        )?;
        let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
        tiles.add_matmul_regions(&[region])?;
        let mut ir = tiles.matmul_region_kernels.remove(0).ir;
        ir.kernel_name = format!("small_split_independent_part{part}");
        for p in &mut ir.params {
            if !p.is_input {
                p.dtype = DType::F32;
            }
        }
        let function = compile_ir(
            stream,
            target,
            ir,
            "independent ordinary BF16/F32 contiguous-K slice oracle",
            &format!(
                "oracle-{m}-{n}-{k}-{}-{}-{part}",
                orientation.0, orientation.1
            ),
        )?;
        Kernel {
            owner: None,
            label: "independent F32 partial oracle".into(),
            function,
            config: LaunchConfig {
                grid_dim: (u32::try_from(n / 64)?, u32::try_from(m / 64)?, 1),
                block_dim: (32, 8, 1),
                shared_mem_bytes: 0,
            },
            pointers: vec![
                a_gpu.device_ptr(stream).0,
                b_gpu.device_ptr(stream).0,
                output.device_ptr(stream).0,
            ],
            zero_before: None,
        }
        .run(stream)?;
        partials.push(stream.memcpy_dtov(&output)?);
    }
    let output = (0..m * n)
        .map(|i| {
            bf16::from_f32(((partials[0][i] + partials[1][i]) + partials[2][i]) + partials[3][i])
                .to_f32()
        })
        .collect();
    Ok((output, partials))
}

#[test]
fn small_grid_split_matches_independent_four_f32_part_oracle_all_maps_and_replay() -> Result<()> {
    if !super::super::tests::cuda_available() {
        return Ok(());
    }
    let (m, n, k) = (128, 128, 512);
    let mut differences = Vec::new();
    for orientation in [(false, false), (false, true), (true, false), (true, true)] {
        let (graph, out) = graph(m, n, k, orientation);
        let mut e = MixedExecutor::new_with_options(
            &graph,
            MatMulPrecision::NativeHalf,
            true,
            MixedOptions {
                small_grid_split_k_bf16: true,
                shared_memory_reuse: true,
                ..Default::default()
            },
        )?;
        assert_eq!(e.info().small_grid_split_k_programs, 1);
        assert_eq!(e.info().small_grid_split_k_fallback_programs, 0);
        assert_eq!(e.info().split_k_square64_programs, 0);
        assert!(e.info().replay);
        let mut old = MixedExecutor::new_with_options(
            &graph,
            MatMulPrecision::NativeHalf,
            true,
            MixedOptions {
                shared_memory_reuse: true,
                ..Default::default()
            },
        )?;
        for revision in 0..4 {
            let values = |len: usize, salt: usize| {
                (0..len)
                    .map(|i| {
                        if revision == 2 && i % 4093 == 0 {
                            return [
                                0.,
                                -0.,
                                f32::INFINITY,
                                f32::NEG_INFINITY,
                                f32::NAN,
                                f32::MAX,
                                f32::MIN_POSITIVE,
                            ][(i / 4093 + salt) % 7];
                        }
                        if revision == 0 {
                            ((i * 7 + salt) % 17) as f32 / 32. - 0.25
                        } else {
                            ((i * 131 + salt) % 257) as f32 / 97. - 1.23
                        }
                    })
                    .collect::<Vec<_>>()
            };
            let (a, b) = if revision == 3 {
                // Deliberately cancellation-sensitive checkpoint: observe the
                // numerical experiment, never assert equality to the old sum.
                let a = (0..m * k)
                    .map(|i| match i % k {
                        0 => -16777216.0,
                        128 => 16777216.0,
                        129..=255 => 1.0,
                        _ => 0.0,
                    })
                    .collect();
                (a, vec![1.0; k * n])
            } else {
                (values(m * k, 3), values(k * n, 7))
            };
            let feeds = HashMap::from([
                ("a".into(), host_physical(&a, m, k, orientation.0)),
                ("b".into(), host_physical(&b, k, n, orientation.1)),
            ]);
            e.upload(feeds.clone())?;
            old.upload(feeds)?;
            old.run()?;
            let (expected, expected_partials) = oracle(&e.stream, &a, &b, m, n, k, orientation)?;
            for replay in 0..2 {
                e.run()?;
                let partial_pointer = e
                    .kernels
                    .iter()
                    .find(|kernel| kernel.label.contains("F32 partials4"))
                    .expect("selected small split partial kernel")
                    .pointers[2];
                let partial = e
                    .buffers
                    .iter()
                    .find_map(|buffer| match buffer {
                        Buffer::F32(values)
                            if values.device_ptr(&e.stream).0 == partial_pointer =>
                        {
                            Some(values)
                        }
                        _ => None,
                    })
                    .expect("retained typed partial workspace");
                let actual_partials = e.stream.memcpy_dtov(partial)?;
                for part in 0..4 {
                    same_bits(
                        &actual_partials[part * m * n..(part + 1) * m * n],
                        &expected_partials[part],
                        &format!("{orientation:?}/{revision}/{replay} F32part{part}"),
                    );
                }
                same_bits(
                    &e.read(out)?,
                    &expected,
                    &format!("{orientation:?}/{revision}/{replay}"),
                );
            }
            let (new, previous) = (e.read(out)?, old.read(out)?);
            let finite_pairs = new
                .iter()
                .zip(&previous)
                .filter(|(a, b)| a.is_finite() && b.is_finite())
                .collect::<Vec<_>>();
            let error2 = finite_pairs
                .iter()
                .map(|(a, b)| (f64::from(**a) - f64::from(**b)).powi(2))
                .sum::<f64>();
            let norm2 = finite_pairs
                .iter()
                .map(|(_, b)| f64::from(**b).powi(2))
                .sum::<f64>();
            differences.push(serde_json::json!({"orientation":orientation,"revision":revision,
                "finite_pairs":finite_pairs.len(),"elements":new.len(),
                "different_non_nan_bits":new.iter().zip(&previous).filter(|(a,b)|!(a.is_nan()&&b.is_nan())&&a.to_bits()!=b.to_bits()).count(),
                "max_absolute_error_finite_pairs":(!finite_pairs.is_empty()).then(||finite_pairs.iter().map(|(a,b)|(f64::from(**a)-f64::from(**b)).abs()).fold(0.0,f64::max)),
                "finite_pair_relative_l2":(!finite_pairs.is_empty()).then(||error2.sqrt()/norm2.sqrt().max(1e-12)),
                "scope":"Observation vs old unsplit BF16 output, not a bit-equivalence assertion or full-model numerical gate"}));
        }
    }
    if let Some(path) = std::env::var_os("TNSR_SMALL_GRID_SPLIT_K_AUDIT") {
        std::fs::create_dir_all(&path)?;
        std::fs::write(
            std::path::Path::new(&path).join("observed-unsplit-differences.json"),
            serde_json::to_vec_pretty(&differences)?,
        )?;
    }
    Ok(())
}

#[test]
fn small_grid_split_reducer_keeps_no_initial_zero_order_and_one_bf16_round() -> Result<()> {
    if !super::super::tests::cuda_available() {
        return Ok(());
    }
    let context = CudaContext::new(0)?;
    let stream = context.new_stream()?;
    let patterns = [
        [-0., -0., -0., -0.],
        [1., 1. / 256., 0., 0.],
        [1.0078125, 1. / 256., 0., 0.],
        [1e20, -1e20, 1., 0.0078125],
        [f32::MAX, f32::MAX, f32::NEG_INFINITY, 0.],
        [f32::INFINITY, 1., 0., 0.],
        [f32::NAN, 0., 0., 0.],
        [f32::from_bits(1), 0., 0., 0.],
    ];
    let len = 64 * 64;
    let host = (0..4 * len)
        .map(|i| patterns[i % len % patterns.len()][i / len])
        .collect::<Vec<_>>();
    let partial = stream.memcpy_stod(&host)?;
    let output = stream.alloc_zeros::<bf16>(len)?;
    let target = PtxTarget::from_context(&context)?;
    let function = compile_ir(
        &stream,
        target,
        reducer_ir(64, 64)?,
        "ordered left-four oracle test",
        "reducer-patterns",
    )?;
    Kernel {
        owner: None,
        label: "reducer primitive checkpoint".into(),
        function,
        config: LaunchConfig::for_num_elems(u32::try_from(len)?),
        pointers: vec![partial.device_ptr(&stream).0, output.device_ptr(&stream).0],
        zero_before: None,
    }
    .run(&stream)?;
    let actual = stream
        .memcpy_dtov(&output)?
        .into_iter()
        .map(bf16::to_f32)
        .collect::<Vec<_>>();
    let expected = (0..len)
        .map(|i| {
            let p = patterns[i % patterns.len()];
            bf16::from_f32(((p[0] + p[1]) + p[2]) + p[3]).to_f32()
        })
        .collect::<Vec<_>>();
    same_bits(&actual, &expected, "left4 primitive");
    assert_eq!(actual[0].to_bits(), (-0.0_f32).to_bits());
    Ok(())
}

#[test]
fn small_grid_split_declined_views_linear_and_batch_keep_complete_baseline() -> Result<()> {
    if !super::super::tests::cuda_available() {
        return Ok(());
    }
    let a = TensorExpr::input("a", vec![1, 512])
        .cast(ValueDType::BF16)
        .broadcast(vec![128, 512]);
    let b = TensorExpr::input("b", vec![512, 128]).cast(ValueDType::BF16);
    let g: TensorGraph<f32> = a.matmul(b).cast(ValueDType::F32).into();
    let out = *g.toposort().last().unwrap();
    let mut outputs = Vec::new();
    let mut sources = Vec::new();
    for enabled in [false, true] {
        let mut e = MixedExecutor::new_with_options(
            &g,
            MatMulPrecision::NativeHalf,
            true,
            MixedOptions {
                small_grid_split_k_bf16: enabled,
                shared_memory_reuse: true,
                ..Default::default()
            },
        )?;
        assert_eq!(e.info().small_grid_split_k_programs, 0);
        assert_eq!(
            e.info().small_grid_split_k_fallback_programs,
            usize::from(enabled)
        );
        e.upload(HashMap::from([
            ("a".into(), vec![0.125; 512]),
            ("b".into(), vec![-0.25; 512 * 128]),
        ]))?;
        e.run()?;
        outputs.push(e.read(out)?);
        sources.push(
            e.programs
                .iter()
                .filter_map(|p| p.downcast_ref::<Program<BF16>>().map(|p| p.ptx.clone()))
                .collect::<Vec<_>>(),
        );
    }
    same_bits(&outputs[0], &outputs[1], "whole fallback");
    assert_eq!(sources[0], sources[1]);
    for linear in [true, false] {
        let expr = if linear {
            TensorExpr::input("a", vec![128, 512]).linear_bf16(
                TensorExpr::input("b", vec![512, 128]),
                TensorExpr::input("bias", vec![128]),
            )?
        } else {
            TensorExpr::input("a", vec![2, 128, 512])
                .cast(ValueDType::BF16)
                .matmul(TensorExpr::input("b", vec![2, 512, 128]).cast(ValueDType::BF16))
        };
        let g: TensorGraph<f32> = expr.cast(ValueDType::F32).into();
        let e = MixedExecutor::new_with_options(
            &g,
            MatMulPrecision::NativeHalf,
            true,
            MixedOptions {
                small_grid_split_k_bf16: true,
                shared_memory_reuse: true,
                ..Default::default()
            },
        )?;
        assert_eq!(e.info().small_grid_split_k_programs, 0);
        assert_eq!(e.info().small_grid_split_k_fallback_programs, 0);
    }
    Ok(())
}

#[test]
fn small_grid_split_actual_decoder_structural_scope_is_explicit() -> Result<()> {
    use super::super::test_decoder::{Config, synthetic};
    let mut records = Vec::new();
    for (width, sequence, batch, heads, hidden) in [
        (2048, 128, 1, 32, 5504),
        (1024, 512, 1, 16, 2752),
        (512, 512, 8, 8, 1408),
        (512, 512, 16, 8, 1408),
        (512, 512, 32, 8, 1408),
        (512, 128, 2, 8, 1408),
        (512, 512, 2, 8, 1408),
        (1024, 128, 2, 16, 2752),
    ] {
        let model = synthetic(Config {
            width,
            sequence,
            batch,
            heads,
            hidden,
            layers: 4,
            vocab: 4096,
            epsilon: 1e-5,
            learning_rate: 1e-4,
            max_grad_norm: 1.,
        });
        let (graph, _) = model.training_graph_with_options(crate::graph::GradientOptions {
            compact_softmax: true,
        })?;
        let types = value_dtypes(&graph);
        let mut shapes = Vec::new();
        for node in graph.toposort() {
            let TensorGraphNode::MatMul { shape } = &graph[node] else {
                continue;
            };
            if shape.len() != 2 || types.get(&node) != Some(&ValueDType::BF16) {
                continue;
            }
            let inputs = graph.inputs(node);
            let lhs = graph[inputs[0]].shape();
            let rhs = graph[inputs[1]].shape();
            if lhs.len() == 2
                && rhs.len() == 2
                && SmallGridCandidate::eligible_shape(shape[0], shape[1], lhs[1])
            {
                shapes.push((node.index(), shape[0], shape[1], lhs[1]));
            }
        }
        records.push(serde_json::json!({"width":width,"sequence":sequence,"batch":batch,
            "config":{"width":width,"sequence":sequence,"batch":batch,"heads":heads,"hidden":hidden,
                "layers":4,"vocab":4096,"epsilon":1e-5,"learning_rate":1e-4,"max_grad_norm":1.0},
            "structural_nodes":shapes,"scope":"Established run.py configuration; shape/type eligibility only; actual map/alignment/JIT activation must be measured separately"}));
    }
    assert!(
        !records[0]["structural_nodes"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    if let Some(path) = std::env::var_os("TNSR_SMALL_GRID_SPLIT_K_AUDIT") {
        std::fs::create_dir_all(&path)?;
        std::fs::write(
            std::path::Path::new(&path).join("actual-decoder-structural-scope.json"),
            serde_json::to_vec_pretty(&records)?,
        )?;
    }
    Ok(())
}

#[test]
fn small_grid_split_resident_incremental_replay_microbatches_and_recovery() -> Result<()> {
    use crate::ptx::resident::{AdamGroup, ResidentAdam};
    if !super::super::tests::cuda_available() {
        return Ok(());
    }
    let (m, n, k) = (128, 128, 512);
    let mut parameters = Vec::new();
    let mut graphs = Vec::new();
    for _ in 0..2 {
        let x = Parameter::new(
            (0..m * k).map(|i| (i % 17) as f32 / 64. - 0.125).collect(),
            vec![m, k],
        );
        let w = Parameter::new(
            (0..n * k).map(|i| (i % 13) as f32 / 32. - 0.0625).collect(),
            vec![n, k],
        );
        let y = (TensorExpr::from(x.clone()) + TensorExpr::input("delta", vec![m, k]))
            .cast(ValueDType::BF16)
            .matmul(
                TensorExpr::from(w.clone())
                    .cast(ValueDType::BF16)
                    .transpose(),
            )
            .cast(ValueDType::F32);
        let mut graph = TensorGraph::new();
        let loss = graph.add_expr(
            &(y * TensorExpr::input("seed", vec![m, n])).mean_all(),
            false,
        );
        graphs.push((graph.with_gradients(loss), loss));
        parameters.push([x, w]);
    }
    let mut owners = parameters
        .iter()
        .map(|p| {
            ResidentAdam::new_with_graph_replay(
                p.iter().cloned().map(|p| (p, 0)).collect(),
                vec![AdamGroup::new(1e-4)],
            )
        })
        .collect::<Result<Vec<_>>>()?;
    for owner in &mut owners {
        owner.set_experimental_first_accumulation_overwrite(true)?;
        owner.set_experimental_gradient_norm_partials(true)?;
        owner.begin_step()?;
    }
    let mut executors = owners
        .iter()
        .enumerate()
        .map(|(i, owner)| {
            owner.mixed_executor_with_options(
                &graphs[i].0,
                MatMulPrecision::Tf32x3,
                MixedOptions {
                    small_grid_split_k_bf16: true,
                    shared_memory_reuse: true,
                    incremental_allocation_reuse: i == 1,
                    direct_master_bindings: true,
                    ..Default::default()
                },
            )
        })
        .collect::<Result<Vec<_>>>()?;
    for e in &executors {
        assert_eq!(e.info().small_grid_split_k_programs, 1);
        assert_eq!(e.info().small_grid_split_k_fallback_programs, 0);
        assert!(e.info().replay);
    }
    for (step, microbatches) in [1, 2, 1].into_iter().enumerate() {
        let mut statuses = Vec::new();
        for i in 0..2 {
            if step > 0 {
                owners[i].begin_step()?;
            }
            for micro in 0..microbatches {
                owners[i].accumulate_mixed(
                    &mut executors[i],
                    &graphs[i].0,
                    HashMap::from([
                        (
                            "delta".into(),
                            (0..m * k)
                                .map(|j| ((j + step * 7 + micro * 13) % 11) as f32 / 128.)
                                .collect(),
                        ),
                        (
                            "seed".into(),
                            (0..m * n)
                                .map(|j| ((j + step * 3 + micro) % 7) as f32 / 32. - 0.0625)
                                .collect(),
                        ),
                    ]),
                    HashMap::new(),
                )?;
            }
            statuses.push(owners[i].step(1.)?);
            assert_eq!(
                owners[i].gradient_norm_partials_status().last_step_used,
                microbatches == 1
            );
            assert_eq!(executors[i].transfers().parameter_copies, 0);
        }
        assert_eq!(
            statuses[0].gradient_norm.to_bits(),
            statuses[1].gradient_norm.to_bits()
        );
        assert_eq!(
            statuses[0].gradient_scale.to_bits(),
            statuses[1].gradient_scale.to_bits()
        );
        same_bits(
            &executors[0].read(graphs[0].1)?,
            &executors[1].read(graphs[1].1)?,
            "candidate lifecycle loss",
        );
        let gradients = [owners[0].gradients()?, owners[1].gradients()?];
        let moments = [owners[0].moments()?, owners[1].moments()?];
        for (a, b) in parameters[0].iter().zip(&parameters[1]) {
            same_bits(
                &gradients[0][&a.id()],
                &gradients[1][&b.id()],
                "candidate gradient",
            );
            same_bits(&a.values()?, &b.values()?, "candidate master");
            same_bits(
                &moments[0][&a.id()].first,
                &moments[1][&b.id()].first,
                "candidate firstmoment",
            );
            same_bits(
                &moments[0][&a.id()].second,
                &moments[1][&b.id()].second,
                "candidate secondmoment",
            );
        }
    }
    for e in &executors {
        assert_eq!(e.info().direct_capture_variants, 2);
        assert_eq!(e.info().direct_capture_misses, 2);
        assert_eq!(e.info().direct_capture_hits, 1);
    }
    for i in 0..2 {
        let old = parameters[i]
            .iter()
            .map(Parameter::values)
            .collect::<Result<Vec<_>>>()?;
        owners[i].begin_step()?;
        owners[i].accumulate_mixed(
            &mut executors[i],
            &graphs[i].0,
            HashMap::from([
                ("delta".into(), vec![f32::NAN; m * k]),
                ("seed".into(), vec![1.; m * n]),
            ]),
            HashMap::new(),
        )?;
        assert!(owners[i].step(1.).is_err(), "nonfinite interval rejected");
        for (p, before) in parameters[i].iter().zip(&old) {
            same_bits(&p.values()?, before, "failed interval retains master");
        }
        owners[i].begin_step()?;
        owners[i].accumulate_mixed(
            &mut executors[i],
            &graphs[i].0,
            HashMap::from([
                ("delta".into(), vec![0.; m * k]),
                ("seed".into(), vec![1.; m * n]),
            ]),
            HashMap::new(),
        )?;
        assert!(owners[i].step(1.)?.gradient_norm.is_finite());
    }
    for (a, b) in parameters[0].iter().zip(&parameters[1]) {
        same_bits(&a.values()?, &b.values()?, "recovered candidate state");
    }
    Ok(())
}

#[test]
fn small_grid_split_permission_and_long_priority_keep_original_kernels() -> Result<()> {
    if !super::super::tests::cuda_available() {
        return Ok(());
    }
    for (k, reuse, expected_fallback) in [(512, false, 1), (2048, true, 0)] {
        let (g, out) = graph(128, 128, k, (false, false));
        let mut outputs = Vec::new();
        let mut labels = Vec::new();
        let mut sources = Vec::new();
        for enabled in [false, true] {
            let mut e = MixedExecutor::new_with_options(
                &g,
                MatMulPrecision::NativeHalf,
                true,
                MixedOptions {
                    small_grid_split_k_bf16: enabled,
                    shared_memory_reuse: reuse,
                    ..Default::default()
                },
            )?;
            assert_eq!(e.info().small_grid_split_k_programs, 0);
            assert_eq!(
                e.info().small_grid_split_k_fallback_programs,
                if enabled { expected_fallback } else { 0 }
            );
            labels.push(
                e.kernels
                    .iter()
                    .map(|kernel| kernel.label.clone())
                    .collect::<Vec<_>>(),
            );
            sources.push(
                e.programs
                    .iter()
                    .filter_map(|p| p.downcast_ref::<Program<BF16>>().map(|p| p.ptx.clone()))
                    .collect::<Vec<_>>(),
            );
            e.upload(HashMap::from([
                (
                    "a".into(),
                    (0..128 * k)
                        .map(|i| (i % 17) as f32 / 32. - 0.125)
                        .collect(),
                ),
                (
                    "b".into(),
                    (0..k * 128)
                        .map(|i| (i % 13) as f32 / 64. - 0.0625)
                        .collect(),
                ),
            ]))?;
            e.run()?;
            outputs.push(e.read(out)?);
        }
        assert_eq!(labels[0], labels[1]);
        assert_eq!(sources[0], sources[1]);
        same_bits(
            &outputs[0],
            &outputs[1],
            "declined or unchanged long-path output",
        );
    }
    Ok(())
}
