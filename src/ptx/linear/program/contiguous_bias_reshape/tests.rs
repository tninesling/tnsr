use super::*;
use crate::{ptx::PtxExecutionPlan, tensor::TensorExpr};
use half::bf16;

fn fixture() -> Result<(TensorGraph<bf16>, PtxExecutionPlan)> {
    let expression = TensorExpr::<bf16>::input("x", vec![32, 512, 512])
        .reshape(vec![16384, 512])
        .reduce_sum(0);
    let graph: TensorGraph<bf16> = expression.into();
    let plan =
        PtxExecutionPlan::build_with_reduction_mode(&graph, PtxReductionMode::DeterministicTree)?;
    Ok((graph, plan))
}
#[test]
fn contiguous_bias_alias_selected_plan_census_and_closed_typed_proof() -> Result<()> {
    let (graph, plan) = fixture()?;
    let count = requests(&graph, &plan);
    eprintln!("standalone BF16 reshape physical-copy census={count}");
    // This is a planner census, not permission to broaden admission. A planner
    // that already virtualizes the reshape has no redundant copy to remove.
    if count == 0 {
        assert!(prove(&graph, &plan, DType::BF16, false).is_err());
        return Ok(());
    }
    assert_eq!(count, 1, "{:#?}", plan.steps());
    let c = prove(&graph, &plan, DType::BF16, false)?;
    assert_eq!(c.len, 8388608);
    assert_ne!(c.source, c.output);
    assert!(matches!(graph[c.output], TensorGraphNode::Reshape { .. }));
    assert!(prove(&graph, &plan, DType::F32, false).is_err());
    assert!(prove(&graph, &plan, DType::BF16, true).is_err());
    Ok(())
}
#[test]
fn contiguous_bias_alias_rejects_mean_escaped_reshape_and_invalid_bindings() -> Result<()> {
    let (mut graph, plan) = fixture()?;
    if let Some(root) = plan.graph_output()
        && let TensorGraphNode::ReduceAxis { op, .. } = &mut graph.graph[root]
    {
        *op = ReduceOp::Mean;
    }
    assert!(prove(&graph, &plan, DType::BF16, false).is_err());
    let c = Candidate {
        step: 0,
        source: NodeIndex::new(0),
        output: NodeIndex::new(1),
        len: 16,
    };
    let good = Binding {
        pointer: 256,
        len: 16,
        dtype: DType::BF16,
    };
    let mut values = HashMap::from([(c.source, good)]);
    assert_eq!(c.binding(&values)?.pointer, 256);
    for bad in [
        Binding {
            pointer: 257,
            ..good
        },
        Binding { pointer: 0, ..good },
        Binding { len: 15, ..good },
        Binding {
            dtype: DType::F32,
            ..good
        },
        Binding {
            pointer: u64::MAX - 15,
            ..good
        },
    ] {
        values.insert(c.source, bad);
        assert!(c.binding(&values).is_err());
    }
    let (mut graph, plan) = fixture()?;
    if let Some(copy) = plan.steps().iter().find(|s| {
        s.action == PtxPlanAction::Kernel
            && matches!(graph[s.node], TensorGraphNode::Reshape { .. })
    }) {
        let escape = graph.graph.add_node(TensorGraphNode::Reshape {
            shape: copy.shape.clone(),
        });
        graph.graph.add_edge(copy.node, escape, 0);
        assert!(prove(&graph, &plan, DType::BF16, false).is_err());
    }
    Ok(())
}

#[test]
#[ignore = "manual CUDA all65536-BF16-pattern alias/reducer/pointer/replay proof before model qualification"]
fn contiguous_bias_alias_all_bf16_patterns_pointer_metadata_and_replay() -> Result<()> {
    use crate::ptx::BF16;
    use cudarc::driver::{CudaContext, sys};
    let context = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            assert!(
                std::env::var_os("TNSR_REQUIRE_CUDA").is_none(),
                "CUDA required: {e}"
            );
            return Ok(());
        }
    };
    let stream = context.new_stream()?;
    const ROWS: usize = 256;
    const COLS: usize = 65536;
    let graph: TensorGraph<bf16> = TensorExpr::<bf16>::input("x", vec![2, 128, COLS])
        .reshape(vec![ROWS, COLS])
        .reduce_sum(0)
        .into();
    let input_node = graph
        .graph
        .node_indices()
        .find(|&n| matches!(graph[n], TensorGraphNode::Input { .. }))
        .context("input missing")?;
    let mut source = stream.alloc_zeros::<bf16>(ROWS * COLS)?;
    let source_binding = Binding {
        pointer: source.device_ptr(&stream).0,
        len: ROWS * COLS,
        dtype: DType::BF16,
    };
    let build = |enabled| {
        Program::<BF16>::new_bound_with_plan_metadata(
            &stream,
            graph.clone(),
            HashMap::from([(input_node, source_binding)]),
            None,
            crate::tile::MatMulPrecision::NativeHalf,
            ProgramKernelOptions {
                contiguous_bias_reshape_aliases: enabled,
                column_reduction_policy: crate::tile::ColumnReductionPolicy::Four,
                ..ProgramKernelOptions::default()
            },
        )
    };
    let baseline = build(false)?;
    let selected = build(true)?;
    let explicit_off = build(false)?;
    assert_eq!(baseline.kernels.len(), 2);
    assert_eq!(selected.kernels.len(), 1);
    assert_eq!(explicit_off.kernels.len(), 2);
    assert_eq!(selected.contiguous_bias_reshape_alias_steps, 1);
    assert_eq!(selected.contiguous_bias_reshape_fallback_steps, 0);
    assert_eq!(
        selected.contiguous_bias_reshape_avoided_bytes,
        ROWS * COLS * 2
    );
    assert_eq!(baseline.contiguous_bias_reshape_alias_steps, 0);
    assert_eq!(baseline.ptx, selected.ptx, "reducer module changed");
    assert_eq!(baseline.ptx, explicit_off.ptx, "default-off module changed");
    assert_eq!(baseline.bytes() - selected.bytes(), ROWS * COLS * 2);
    let metadata = selected
        .metadata
        .as_ref()
        .context("selected metadata absent")?;
    let c = prove(&graph, &metadata.plan, DType::BF16, false)?;
    assert_eq!(baseline.metadata.as_ref().unwrap().kernel_steps, vec![1, 2]);
    assert_eq!(metadata.kernel_steps, vec![2]);
    assert_eq!(
        explicit_off.metadata.as_ref().unwrap().kernel_steps,
        vec![1, 2]
    );
    assert_eq!(selected.values[&c.output].pointer, source_binding.pointer);
    assert_ne!(baseline.values[&c.output].pointer, source_binding.pointer);
    assert_ne!(selected.output.pointer, source_binding.pointer);
    assert_eq!(
        selected.kernels[0].pointers,
        vec![source_binding.pointer, selected.output.pointer]
    );
    assert_eq!(
        baseline.kernels[1].config.grid_dim,
        selected.kernels[0].config.grid_dim
    );
    assert_eq!(
        baseline.kernels[1].config.block_dim,
        selected.kernels[0].config.block_dim
    );
    assert_eq!(
        baseline.kernels[1].config.shared_mem_bytes,
        selected.kernels[0].config.shared_mem_bytes
    );
    assert_eq!(baseline.kernels[0].label, explicit_off.kernels[0].label);
    assert_eq!(baseline.kernels[1].label, explicit_off.kernels[1].label);
    let make_capture = |p: &Program<BF16>| -> Result<_> {
        stream.synchronize()?;
        stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
        let run = p.kernels.iter().try_for_each(|k| k.run(&stream));
        let captured = stream.end_capture(
            sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        );
        run?;
        captured?.context("empty reduction capture")
    };
    let graphs = [
        make_capture(&baseline)?,
        make_capture(&selected)?,
        make_capture(&explicit_off)?,
    ];
    let read = |p: &Program<BF16>| -> Result<Vec<bf16>> {
        let owned = p
            .storage
            .iter()
            .find(|b| b.device_ptr(&stream).0 == p.output.pointer)
            .context("owned reduction output missing")?;
        let mut host = vec![bf16::ZERO; COLS];
        stream.memcpy_dtoh(owned, &mut host)?;
        stream.synchronize()?;
        Ok(host)
    };
    let mut phases = Vec::new();
    for phase in 0..3 {
        let host = (0..ROWS * COLS)
            .map(|i| {
                let column = i % COLS;
                let row = i / COLS;
                let bits = match phase {
                    0 => column as u16,
                    1 => column as u16 ^ if row.is_multiple_of(2) { 0 } else { 0x8000 },
                    _ => (column as u16).wrapping_add((row % 7) as u16),
                };
                bf16::from_bits(bits)
            })
            .collect::<Vec<_>>();
        stream.memcpy_htod(&host, &mut source)?;
        for _ in 0..2 {
            for g in &graphs {
                g.launch()?;
            }
            stream.synchronize()?;
            let expected = read(&baseline)?;
            for p in [&selected, &explicit_off] {
                let actual = read(p)?;
                for (i, (a, b)) in expected.iter().zip(&actual).enumerate() {
                    assert!(
                        a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()),
                        "BF16 alias phase{phase} output{i}: {a:?}/{b:?}"
                    );
                }
            }
        }
        let mut readonly = vec![bf16::ZERO; host.len()];
        stream.memcpy_dtoh(&source, &mut readonly)?;
        stream.synchronize()?;
        assert!(
            host.iter()
                .zip(readonly)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "source was written"
        );
        phases.push(serde_json::json!({"phase":phase,"all65536patterns":true,"replays":2,"finite_and_nonNaN_bits_exact":true,"NaNclasses_equal":true,"source_readonly_bits_exact":true}));
    }
    // An externally supplied root may overlap the original input. The old
    // copy makes that safe; the new alias must reject BEFORE allocation and
    // retain the complete two-kernel path, including its publication order.
    let fallback = Program::<BF16>::new_bound_with_plan_metadata(
        &stream,
        graph.clone(),
        HashMap::from([(input_node, source_binding)]),
        Some(Binding {
            pointer: source_binding.pointer,
            len: COLS,
            dtype: DType::BF16,
        }),
        crate::tile::MatMulPrecision::NativeHalf,
        ProgramKernelOptions {
            contiguous_bias_reshape_aliases: true,
            column_reduction_policy: crate::tile::ColumnReductionPolicy::Four,
            ..ProgramKernelOptions::default()
        },
    )?;
    assert_eq!(fallback.contiguous_bias_reshape_alias_steps, 0);
    assert_eq!(fallback.contiguous_bias_reshape_fallback_steps, 1);
    assert_eq!(fallback.kernels.len(), 2);
    assert_eq!(fallback.ptx, baseline.ptx);
    assert_eq!(fallback.metadata.as_ref().unwrap().kernel_steps, vec![1, 2]);
    for (a, b) in fallback.kernels.iter().zip(&baseline.kernels) {
        assert_eq!(a.label, b.label);
        assert_eq!(a.config.grid_dim, b.config.grid_dim);
        assert_eq!(a.config.block_dim, b.config.block_dim);
        assert_eq!(a.config.shared_mem_bytes, b.config.shared_mem_bytes);
    }
    graphs[0].launch()?;
    stream.synchronize()?;
    let expected = read(&baseline)?;
    let fallback_graph = make_capture(&fallback)?;
    fallback_graph.launch()?;
    stream.synchronize()?;
    let mut external = vec![bf16::ZERO; COLS];
    stream.memcpy_dtoh(&source.slice(0..COLS), &mut external)?;
    stream.synchronize()?;
    assert!(
        expected
            .iter()
            .zip(external)
            .all(|(a, b)| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())),
        "overlapping external root complete fallback differs"
    );
    drop(fallback_graph);
    if let Some(dir) = std::env::var_os("TNSR_CONTIGUOUS_BIAS_RESHAPE_AUDIT") {
        let dir = std::path::PathBuf::from(dir);
        std::fs::create_dir_all(&dir)?;
        std::fs::write(dir.join("leaf-reducer.ptx"), &selected.ptx)?;
        std::fs::write(
            dir.join("leaf-proof.json"),
            serde_json::to_vec_pretty(
                &serde_json::json!({"rows":ROWS,"cols":COLS,"selected_aliases":1,"baseline_kernels":2,"selected_kernels":1,"kernel_steps_before":[1,2],"kernel_steps_after":[2],"reducer_and_defaultoff_PTX_identical":true,"source_output_disjoint":true,"external_overlapping_root_complete_fallback":true,"nominal_avoided_bytes":ROWS*COLS*2,"phases":phases,"scope":"Owned leaf allocations; not pooled model-memory savings. Full model lifetimes/state pending."}),
            )?,
        )?;
    }
    stream.synchronize()?;
    drop(graphs);
    Ok(())
}
