//! Device-free relocation tests. Real-device lease/replay tests are added when
//! the direct mode is integrated after the preceding binary/source freeze.
use super::*;
fn binding(pointer: u64, len: usize) -> Binding {
    Binding {
        pointer,
        len,
        dtype: DType::F32,
    }
}
#[test]
fn direct_relocations_preserve_parameter_aliases_and_leave_workspace_untouched() {
    let a = NodeIndex::new(1);
    let duplicate = NodeIndex::new(2);
    let b = NodeIndex::new(3);
    let view = NodeIndex::new(4);
    let work = NodeIndex::new(5);
    let sources = HashMap::from([
        (a, binding(100, 16)),
        (duplicate, binding(100, 16)),
        (b, binding(200, 32)),
    ]);
    let values = HashMap::from([
        (a, binding(100, 16)),
        (duplicate, binding(100, 16)),
        (b, binding(200, 32)),
        (view, binding(100, 16)),
        (work, binding(3000, 16)),
    ]);
    let layout = DirectLayout::new(
        &sources,
        &[vec![100, 3000, 200], vec![3000, 100]],
        &[None, Some(3000)],
        &values,
    )
    .unwrap();
    let next = HashMap::from([
        (a, binding(400, 16)),
        (duplicate, binding(400, 16)),
        (b, binding(500, 32)),
    ]);
    let plan = layout.rebind(&next).unwrap();
    assert_eq!(plan.key, vec![400, 400, 500]);
    assert_eq!(plan.arguments, vec![(0, 0, 400), (0, 2, 500), (1, 1, 400)]);
    let aliases: HashMap<_, _> = plan.aliases.into_iter().collect();
    assert_eq!(aliases[&view], 400);
    assert!(!aliases.contains_key(&work));
    // Reusing a physical address is valid after a new version owns it; identity
    // of the Arc is not part of this purely physical replay key.
    assert_eq!(layout.rebind(&sources).unwrap().key, vec![100, 100, 200]);
}
#[test]
fn direct_relocation_rejects_partial_sets_extent_changes_and_alias_changes() {
    let a = NodeIndex::new(1);
    let alias = NodeIndex::new(2);
    let b = NodeIndex::new(3);
    let initial = HashMap::from([
        (a, binding(100, 16)),
        (alias, binding(100, 16)),
        (b, binding(200, 16)),
    ]);
    let layout = DirectLayout::new(&initial, &[vec![100, 200]], &[None], &initial).unwrap();
    let mut next = initial.clone();
    next.remove(&b);
    assert!(layout.rebind(&next).is_err());
    next = initial.clone();
    next.insert(b, binding(200, 17));
    assert!(layout.rebind(&next).is_err());
    next = initial.clone();
    next.insert(alias, binding(300, 16));
    assert!(layout.rebind(&next).is_err());
    next = initial.clone();
    next.insert(b, binding(100, 16));
    assert!(layout.rebind(&next).is_err());
    assert!(DirectLayout::new(&initial, &[vec![100]], &[Some(100)], &initial).is_err());
    assert!(DirectLayout::new(&initial, &[vec![100]], &[], &initial).is_err());
    assert_eq!(layout.rebind(&initial).unwrap().key, vec![100, 100, 200]);
}

#[test]
fn direct_typed_shadow_layout_relocates_both_types_and_rejects_overlap() -> Result<()> {
    let master = NodeIndex::new(1);
    let cast = NodeIndex::new(3);
    let alias = NodeIndex::new(4);
    let f32 = binding(4096, 8192);
    let bf16 = Binding {
        pointer: 65536,
        len: 8192,
        dtype: DType::BF16,
    };
    let initial = HashMap::from([(master, f32), (cast, bf16)]);
    let values = HashMap::from([(master, f32), (cast, bf16), (alias, bf16)]);
    let layout = DirectLayout::new(&initial, &[vec![4096, 65536, 131072]], &[None], &values)?;
    let next = HashMap::from([
        (
            master,
            Binding {
                pointer: 262144,
                ..f32
            },
        ),
        (
            cast,
            Binding {
                pointer: 327680,
                ..bf16
            },
        ),
    ]);
    let plan = layout.rebind(&next)?;
    assert_eq!(plan.key, vec![262144, 327680]);
    assert_eq!(plan.arguments, vec![(0, 0, 262144), (0, 1, 327680)]);
    assert!(plan.aliases.contains(&(alias, 327680)));
    // A new revision may return to the same physical address pair. Revisions
    // and Arc generations do not participate in this compatibility key.
    assert_eq!(layout.rebind(&initial)?.key, vec![4096, 65536]);
    let mut wrong = next.clone();
    wrong.get_mut(&cast).unwrap().dtype = DType::F32;
    assert!(layout.rebind(&wrong).is_err());
    wrong = next.clone();
    wrong.get_mut(&cast).unwrap().pointer = 262148;
    assert!(layout.rebind(&wrong).is_err());
    wrong = next.clone();
    wrong.get_mut(&cast).unwrap().len -= 1;
    assert!(layout.rebind(&wrong).is_err());
    assert!(DirectLayout::new(&initial, &[vec![65536]], &[Some(65536)], &values).is_err());
    assert!(DirectLayout::new(&initial, &[vec![65536]], &[Some(65538)], &values).is_err());
    assert!(DirectLayout::new(&initial, &[vec![65538]], &[None], &values).is_err());
    let mut interior = values.clone();
    interior.get_mut(&alias).unwrap().pointer += 2;
    assert!(DirectLayout::new(&initial, &[vec![65536]], &[None], &interior).is_err());
    let fixed = Binding {
        pointer: 131072,
        len: 16384,
        dtype: DType::F32,
    };
    let guarded = DirectLayout::new_with_writes(
        &initial,
        &[vec![4096, 65536, fixed.pointer]],
        &[Some(fixed.pointer)],
        &values,
        &[fixed],
    )?;
    let mut onto_write = initial.clone();
    onto_write.get_mut(&master).unwrap().pointer = fixed.pointer + 4;
    assert!(guarded.rebind(&onto_write).is_err());
    onto_write = initial.clone();
    onto_write.get_mut(&cast).unwrap().pointer = fixed.pointer - 2;
    assert!(guarded.rebind(&onto_write).is_err()); // spans overlap despite distinct starts
    let indices = Binding {
        pointer: 524288,
        len: 4096,
        dtype: DType::U32,
    };
    let guarded_indices = DirectLayout::new_with_writes(
        &initial,
        &[vec![4096, 65536, indices.pointer]],
        &[None],
        &values,
        &[indices],
    )?;
    onto_write = initial.clone();
    onto_write.get_mut(&master).unwrap().pointer = indices.pointer + 4;
    assert!(guarded_indices.rebind(&onto_write).is_err());
    onto_write = initial.clone();
    onto_write.get_mut(&cast).unwrap().pointer = indices.pointer - 2;
    assert!(guarded_indices.rebind(&onto_write).is_err());
    let zero_only = DirectLayout::new(
        &initial,
        &[vec![4096, fixed.pointer]],
        &[Some(fixed.pointer)],
        &values,
    )?;
    onto_write = initial.clone();
    onto_write.get_mut(&master).unwrap().pointer = fixed.pointer;
    assert!(zero_only.rebind(&onto_write).is_err());
    let mut bad_alias = values;
    bad_alias.get_mut(&alias).unwrap().dtype = DType::F32;
    assert!(DirectLayout::new(&initial, &[vec![65536]], &[None], &bad_alias).is_err());
    Ok(())
}

#[test]
fn direct_failed_capture_keeps_old_sources_graph_and_readback_then_recovers() {
    use crate::tensor::{Parameter, TensorExpr};
    let available = CudaContext::new(0).is_ok();
    assert!(
        available || std::env::var_os("TNSR_REQUIRE_CUDA").is_none(),
        "CUDA required"
    );
    if !available {
        return;
    }
    let p = Parameter::new(vec![0.25_f32, 0.5, 0.75], vec![3]);
    let mut graph = TensorGraph::<f32>::new();
    let source = graph.add_expr(&TensorExpr::from(p), false);
    let output = graph.add_expr(
        &(TensorExpr::node_ref(source, vec![3]) * TensorExpr::input("x", vec![3])),
        false,
    );
    let stream = CudaContext::new(0).unwrap().new_stream().unwrap();
    let original = Arc::new(stream.memcpy_stod(&[0.25_f32, 0.5, 0.75]).unwrap());
    let parameters = HashMap::from([(source, original)]);
    let mut executor = MixedExecutor::for_resident_with_options(
        &graph,
        MatMulPrecision::Tf32x3,
        true,
        ResidentBindings {
            stream: stream.clone(),
            gate: Arc::new(Mutex::new(())),
            parameters,
        },
        MixedOptions {
            direct_master_bindings: true,
            ..MixedOptions::default()
        },
    )
    .unwrap();
    executor
        .upload(HashMap::from([("x".into(), vec![1.0; 3])]))
        .unwrap();
    executor.run().unwrap();
    let before = executor.read(source).unwrap();
    let loss_buffer = executor.read(output).unwrap();
    let output_pointer = executor.readbacks.borrow()[&output]
        .buffer
        .device_ptr(&stream)
        .0;
    let replacement = HashMap::from([(
        source,
        Arc::new(stream.memcpy_stod(&[1.0_f32, 2.0, 3.0]).unwrap()),
    )]);
    let key = executor.direct.as_ref().unwrap().key.clone();
    let mut direct = executor.direct.take().unwrap();
    let error = direct
        .refresh_using(
            &stream,
            &replacement,
            DirectFrame {
                kernels: &mut executor.kernels,
                values: &mut executor.values,
                captured: &mut executor.captured,
                readbacks: executor.readbacks.get_mut(),
            },
            |_, _| Err(anyhow::anyhow!("injected graph instantiation failure")),
        )
        .unwrap_err();
    assert!(error.to_string().contains("injected"));
    assert_eq!(direct.key, key);
    assert_eq!(direct.capture_misses, 1);
    executor.direct = Some(direct);
    assert_eq!(executor.read(source).unwrap(), before);
    executor.run().unwrap();
    assert_eq!(executor.read(output).unwrap(), loss_buffer);
    // Full validation must reject partial sets without changing any source.
    assert!(executor.refresh_parameters(&HashMap::new()).is_err());
    assert_eq!(executor.read(source).unwrap(), before);
    let retired = executor.refresh_parameters(&replacement).unwrap();
    drop(retired);
    assert_eq!(executor.read(source).unwrap(), vec![1.0, 2.0, 3.0]);
    assert_eq!(
        executor.readbacks.borrow()[&output]
            .buffer
            .device_ptr(&stream)
            .0,
        output_pointer
    );
    executor.run().unwrap();
    assert_eq!(executor.read(output).unwrap(), vec![1.0, 2.0, 3.0]);
    assert_eq!(executor.transfers().parameter_copies, 0);
    assert_eq!(executor.transfers().parameter_bytes, 0);
    assert_eq!(executor.info().direct_capture_variants, 2);
}

#[test]
#[ignore = "requires reserved CUDA device; paired shadow refresh and replay proof"]
fn direct_shadow_refresh_rejects_stale_pairs_and_capture_failure_then_reuses_two_variants()
-> Result<()> {
    use crate::tensor::ParameterStorage;
    let stream = CudaContext::new(0)?.new_stream()?;
    let storage = Arc::new(ParameterStorage::new(vec![0.; 8192]));
    let mut graph = TensorGraph::<f32>::new();
    let parameter = graph.graph.add_node(TensorGraphNode::Parameter {
        id: 901,
        data: storage.clone(),
        shape: vec![64, 128],
    });
    let transpose = graph.graph.add_node(TensorGraphNode::Transpose {
        shape: vec![128, 64],
    });
    graph.graph.add_edge(parameter, transpose, 0);
    let cast = graph.graph.add_node(TensorGraphNode::Cast {
        dtype: crate::tensor::ValueDType::BF16,
        shape: vec![128, 64],
    });
    graph.graph.add_edge(transpose, cast, 0);
    let promoted = graph.graph.add_node(TensorGraphNode::Cast {
        dtype: crate::tensor::ValueDType::F32,
        shape: vec![128, 64],
    });
    graph.graph.add_edge(cast, promoted, 0);
    let output = graph.graph.add_node(TensorGraphNode::Unary {
        op: crate::tensor::UnaryOp::Neg,
        shape: vec![128, 64],
    });
    graph.graph.add_edge(promoted, output, 0);
    let f32_user = graph.graph.add_node(TensorGraphNode::Unary {
        op: crate::tensor::UnaryOp::Neg,
        shape: vec![64, 128],
    });
    graph.graph.add_edge(parameter, f32_user, 0);
    let first: Vec<f32> = (0..8192).map(|i| (i % 97) as f32 * 0.03125).collect();
    let second: Vec<f32> = first.iter().map(|&x| -x - 0.125).collect();
    let expected = |source: &[f32]| -> Vec<f32> {
        (0..128)
            .flat_map(|column| {
                (0..64).map(move |row| -half::bf16::from_f32(source[row * 128 + column]).to_f32())
            })
            .collect()
    };
    let a = Arc::new(stream.memcpy_stod(&first)?);
    let b = Arc::new(stream.memcpy_stod(&second)?);
    let sa = Arc::new(
        stream.memcpy_stod(
            &first
                .iter()
                .copied()
                .map(half::bf16::from_f32)
                .collect::<Vec<_>>(),
        )?,
    );
    let sb = Arc::new(
        stream.memcpy_stod(
            &second
                .iter()
                .copied()
                .map(half::bf16::from_f32)
                .collect::<Vec<_>>(),
        )?,
    );
    let shadows = |master: &Arc<CudaSlice<f32>>, shadow: &Arc<CudaSlice<half::bf16>>, revision| {
        HashMap::from([(
            cast,
            ShadowBinding {
                parameter_node: parameter,
                parameter_id: 901,
                storage: storage.clone(),
                revision,
                master: Arc::downgrade(master),
                shadow: shadow.clone(),
            },
        )])
    };
    let ma = HashMap::from([(parameter, a.clone())]);
    let mb = HashMap::from([(parameter, b.clone())]);
    let mut executor = MixedExecutor::for_resident_with_shadows(
        &graph,
        MatMulPrecision::Tf32x3,
        true,
        ResidentBindings {
            stream: stream.clone(),
            gate: Arc::new(Mutex::new(())),
            parameters: ma.clone(),
        },
        MixedOptions {
            direct_master_bindings: true,
            physical_order_casts: true,
            ..MixedOptions::default()
        },
        shadows(&a, &sa, 0),
    )?;
    assert_eq!(executor.info().bf16_master_shadow_cast_aliases, 1);
    assert_eq!(executor.info().physical_order_cast_materializations, 0);
    assert_eq!(executor.info().bf16_bytes, 0);
    executor.upload(HashMap::new())?;
    executor.run()?;
    assert_eq!(executor.read(output)?, expected(&first));
    assert_eq!(
        executor.read(f32_user)?,
        first.iter().map(|&x| -x).collect::<Vec<_>>()
    );
    assert!(executor.refresh_parameters(&ma).is_err());
    assert!(
        executor
            .refresh_parameters_with_shadows(&ma, &HashMap::new())
            .is_err()
    );
    let revision = storage.publish_device(0, Arc::new(|| Ok(vec![0.; 8192])))?;
    assert!(
        executor
            .refresh_parameters_with_shadows(&mb, &shadows(&b, &sb, 0))
            .is_err()
    );
    assert!(
        executor
            .refresh_parameters_with_shadows(&mb, &shadows(&a, &sb, revision))
            .is_err()
    );
    let next = shadows(&b, &sb, revision);
    let key = executor
        .direct
        .as_ref()
        .context("direct bindings absent")?
        .key
        .clone();
    let mut direct = executor.direct.take().context("direct bindings absent")?;
    let error = direct
        .refresh_using_with_shadows(
            &stream,
            &mb,
            DirectFrame {
                kernels: &mut executor.kernels,
                values: &mut executor.values,
                captured: &mut executor.captured,
                readbacks: executor.readbacks.get_mut(),
            },
            &next,
            |_, _| Err(anyhow::anyhow!("injected shadow capture failure")),
        )
        .err()
        .context("injected failure accepted")?;
    assert!(error.to_string().contains("injected"));
    assert_eq!(direct.key, key);
    executor.direct = Some(direct);
    assert_eq!(executor.read(output)?, expected(&first));
    drop(executor.refresh_parameters_with_shadows(&mb, &next)?);
    drop(next);
    executor.run()?;
    assert_eq!(executor.read(output)?, expected(&second));
    assert_eq!(
        executor.read(f32_user)?,
        second.iter().map(|&x| -x).collect::<Vec<_>>()
    );
    // Inactive captured graphs must not pin either old allocation.
    assert_eq!(Arc::strong_count(&sa), 1);
    assert_eq!(
        stream.memcpy_dtov(sa.as_ref())?,
        first
            .iter()
            .copied()
            .map(half::bf16::from_f32)
            .collect::<Vec<_>>()
    );
    let revision = storage.publish_device(revision, Arc::new(|| Ok(vec![0.; 8192])))?;
    drop(executor.refresh_parameters_with_shadows(&ma, &shadows(&a, &sa, revision))?);
    executor.run()?;
    assert_eq!(executor.read(output)?, expected(&first));
    assert_eq!(Arc::strong_count(&sb), 1);
    assert_eq!(executor.info().direct_capture_variants, 2);
    assert_eq!(executor.info().direct_capture_misses, 2);
    assert!(executor.info().direct_capture_hits >= 1);
    assert_eq!(executor.transfers().parameter_copies, 0);
    Ok(())
}
