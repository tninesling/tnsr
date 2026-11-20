//! Test-only observation of existing casts. No shadow allocation, stores or selection.
use super::*;
use crate::tensor::ParameterStorage;
use petgraph::Direction;
use serde_json::{Value as Json, json};

pub(crate) struct RegisteredMaster {
    pub parameter_id: usize,
    pub storage: Arc<ParameterStorage<f32>>,
    pub revision: u64,
    pub allocation: Arc<CudaSlice<f32>>,
}
/// Constructor metadata only. Physical-order proof limits the copied chain to
/// Reshape/Flatten/Permute/Transpose nodes, so it contains no storage/device Arc.
/// Raw construction pointers are evidence, not launch bindings or live leases.
pub(super) struct Observation {
    cast: NodeIndex,
    input: NodeIndex,
    origin: std::result::Result<NodeIndex, String>,
    parameter_id: Option<usize>,
    storage_identity: Option<usize>,
    source: Value,
    target: Value,
    kernel_index: usize,
    kernel_label: String,
}
pub(super) fn observe<G>(
    graph: &TensorGraph<f32, G>,
    cast: NodeIndex,
    source: &Value,
    target: &Value,
    kernel_index: usize,
    kernel_label: &str,
) -> Result<Observation> {
    ensure!(
        physical_order_cast_source(source, DType::BF16).is_some(),
        "constructor observation lacks actual physical-order proof"
    );
    let inputs = graph.inputs(cast);
    ensure!(inputs.len() == 1, "observed cast does not have one input");
    let input = inputs[0];
    let origin = parameter_origin(graph, input).map_err(|e| format!("{e:#}"));
    let (parameter_id, storage_identity) = origin
        .as_ref()
        .ok()
        .and_then(|&n| {
            if let TensorGraphNode::Parameter { id, data, .. } = &graph[n] {
                Some((Some(*id), Some(Arc::as_ptr(data) as usize)))
            } else {
                None
            }
        })
        .unwrap_or((None, None));
    Ok(Observation {
        cast,
        input,
        origin,
        parameter_id,
        storage_identity,
        source: source.clone(),
        target: target.clone(),
        kernel_index,
        kernel_label: kernel_label.into(),
    })
}
fn parameter_origin<G>(graph: &TensorGraph<f32, G>, mut node: NodeIndex) -> Result<NodeIndex> {
    loop {
        match &graph[node] {
            TensorGraphNode::Parameter { .. } => return Ok(node),
            TensorGraphNode::Reshape { .. }
            | TensorGraphNode::Flatten { .. }
            | TensorGraphNode::Permute { .. }
            | TensorGraphNode::Transpose { .. } => {
                let inputs = graph.inputs(node);
                ensure!(inputs.len() == 1, "view does not have one source");
                node = inputs[0];
            }
            other => anyhow::bail!("non-parameter/view origin: {}", other.name()),
        }
    }
}
fn same_view_chain(a: &[TensorGraphNode<f32>], b: &[TensorGraphNode<f32>]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| match (a, b) {
        (TensorGraphNode::Reshape { shape: a }, TensorGraphNode::Reshape { shape: b })
        | (TensorGraphNode::Flatten { shape: a }, TensorGraphNode::Flatten { shape: b })
        | (TensorGraphNode::Transpose { shape: a }, TensorGraphNode::Transpose { shape: b }) => {
            a == b
        }
        (
            TensorGraphNode::Permute { shape: a, axes: aa },
            TensorGraphNode::Permute { shape: b, axes: ba },
        ) => a == b && aa == ba,
        _ => false,
    })
}
fn owner_metadata_matches(source: &Value, owner: Binding, expected: u64, current: u64) -> bool {
    expected == current
        && source.binding.dtype == DType::F32
        && source.view.dtype == DType::F32
        && owner.dtype == DType::F32
        && source.binding.pointer == owner.pointer
        && source.binding.len == owner.len
        && source
            .source_shape
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            == Some(owner.len)
}
/// `registered` is built from the actual owner's matched graph-node/storage map.
/// All owners are kept alive until the complete census has finished.
pub(crate) fn census<G>(
    executor: &MixedExecutor,
    graph: &TensorGraph<f32, G>,
    registered: &HashMap<NodeIndex, RegisteredMaster>,
) -> Result<Json> {
    ensure!(
        executor.has_run,
        "cast census requires a completed real run"
    );
    let mut rows = Vec::new();
    let mut nodes = graph.graph.node_indices().collect::<Vec<_>>();
    nodes.sort_by_key(|n| n.index());
    for node in nodes {
        if !matches!(
            graph[node],
            TensorGraphNode::Cast {
                dtype: ValueDType::BF16,
                ..
            }
        ) {
            continue;
        }
        let inputs = graph.inputs(node);
        let mut row = json!({"cast_node":node.index(),"cast_shape":graph[node].shape(),
            "admitted_candidate":false,"candidate_not_production_admission":true,
            "consumers":graph.graph.neighbors_directed(node,Direction::Outgoing).map(|n|json!({"node":n.index(),"kind":graph[n].name(),"shape":graph[n].shape()})).collect::<Vec<_>>()});
        let checked = (|| -> Result<()> {
            ensure!(inputs.len() == 1, "cast does not have one source");
            let input = inputs[0];
            row["input_node"] = input.index().into();
            let origin = parameter_origin(graph, input)?;
            row["parameter_node"] = origin.index().into();
            let TensorGraphNode::Parameter { id, data, shape } = &graph[origin] else {
                anyhow::bail!("origin is not a parameter");
            };
            row["parameter_id"] = (*id).into();
            row["parameter_storage_identity"] = (Arc::as_ptr(data) as usize).into();
            row["parameter_shape"] = json!(shape);
            ensure!(shape.len() == 2, "parameter is not a matrix");
            let master = registered
                .get(&origin)
                .context("parameter is not registered with owner")?;
            ensure!(
                master.parameter_id == *id && Arc::ptr_eq(&master.storage, data),
                "registered parameter storage mismatch"
            );
            ensure!(
                Arc::ptr_eq(master.allocation.context(), executor.stream.context()),
                "registered master context mismatch"
            );
            let revision = data.revision()?;
            row["expected_revision"] = master.revision.into();
            row["current_revision"] = revision.into();
            row["master_allocation_identity"] = (Arc::as_ptr(&master.allocation) as usize).into();
            let owner = Binding {
                pointer: master.allocation.device_ptr(&executor.stream).0,
                len: master.allocation.len(),
                dtype: DType::F32,
            };
            let observations = executor
                .shadow_cast_observations
                .iter()
                .filter(|o| o.cast == node)
                .collect::<Vec<_>>();
            ensure!(
                observations.len() == 1,
                "cast lacks one actual constructor physical-order observation"
            );
            let observed = observations[0];
            ensure!(
                observed.input == input
                    && observed.origin.as_ref().ok() == Some(&origin)
                    && observed.parameter_id == Some(*id)
                    && observed.storage_identity == Some(Arc::as_ptr(data) as usize),
                "constructor cast origin/storage identity changed"
            );
            row["constructor_source_pointer"] = observed.source.binding.pointer.into();
            row["constructor_target_pointer"] = observed.target.binding.pointer.into();
            row["constructor_kernel_index"] = observed.kernel_index.into();
            row["constructor_kernel_label"] = observed.kernel_label.clone().into();
            row["source_retained_after_pruning"] = executor.values.contains_key(&input).into();
            row["target_retained_after_pruning"] = executor.values.contains_key(&node).into();
            // The exact registered current F32 input and recorded materializer
            // kind/label must identify one real post-rebind kernel. Only then
            // may constructor metadata be rebased to its two live pointers.
            let live = executor
                .kernels
                .iter()
                .enumerate()
                .filter(|(_, k)| {
                    k.label == observed.kernel_label
                        && k.label.starts_with("materialize F32 to BF16")
                        && k.label.contains("physical-order")
                        && k.pointers.len() == 2
                        && k.pointers[0] == owner.pointer
                })
                .collect::<Vec<_>>();
            ensure!(
                live.len() == 1,
                "current registered master does not identify one actual recorded materializer"
            );
            let (live_index, live_kernel) = live[0];
            let mut source = observed.source.clone();
            let mut target = observed.target.clone();
            source.binding.pointer = live_kernel.pointers[0];
            target.binding.pointer = live_kernel.pointers[1];
            let source = &source;
            let target = &target;
            row["live_kernel_index"] = live_index.into();
            row["source_dtype"] = format!("{:?}", source.binding.dtype).into();
            row["source_view_dtype"] = format!("{:?}", source.view.dtype).into();
            row["source_pointer"] = source.binding.pointer.into();
            row["source_elements"] = source.binding.len.into();
            row["source_shape"] = json!(source.source_shape);
            row["logical_shape"] = json!(source.view.shape);
            row["source_chain"] = json!(
                source
                    .chain
                    .iter()
                    .map(|n| json!({"kind":n.name(),"shape":n.shape()}))
                    .collect::<Vec<_>>()
            );
            row["source_normalized_map"] = format!(
                "{:?}",
                source
                    .view
                    .access
                    .clone()
                    .normalize_in_domain(&source.view.shape)?
            )
            .into();
            ensure!(
                owner_metadata_matches(source, owner, master.revision, revision),
                "stale/partial/non-F32/current-owner mismatch"
            );
            let physical = physical_order_cast_source(source, DType::BF16)
                .context("whole bijective physical-order proof rejected")?;
            ensure!(
                physical.binding.pointer == owner.pointer && physical.binding.len == owner.len,
                "physical source differs from current master"
            );
            row["physical_normalized_map"] = format!(
                "{:?}",
                physical
                    .view
                    .access
                    .clone()
                    .normalize_in_domain(&physical.view.shape)?
            )
            .into();
            ensure!(
                target.binding.dtype == DType::BF16
                    && target.view.dtype == DType::BF16
                    && target.binding.len == owner.len
                    && target.source_shape == source.source_shape,
                "BF16 cast target dtype/extent/physical shape mismatch"
            );
            if let Some(retained) = executor.values.get(&node) {
                ensure!(
                    retained.binding.pointer == target.binding.pointer
                        && retained.binding.len == target.binding.len
                        && retained.binding.dtype == target.binding.dtype
                        && retained.source_shape == target.source_shape
                        && same_view_chain(&retained.chain, &target.chain),
                    "retained target disagrees with recorded metadata/current materializer"
                );
            }
            let normalized_source = source
                .view
                .access
                .clone()
                .normalize_in_domain(&source.view.shape)?;
            let normalized_target = target
                .view
                .access
                .clone()
                .normalize_in_domain(&target.view.shape)?;
            ensure!(
                target.view.shape == source.view.shape
                    && normalized_target == normalized_source
                    && same_view_chain(&target.chain, &source.chain),
                "cast result did not preserve logical view/chain"
            );
            let target_bytes = u64::try_from(target.binding.len)?
                .checked_mul(2)
                .context("cast target byte extent overflow")?;
            let target_end = target
                .binding
                .pointer
                .checked_add(target_bytes)
                .context("cast target pointer overflow")?;
            let owners = executor.buffers.iter().enumerate().filter_map(|(index,buffer)| {
                let Buffer::BF16(owner) = buffer else {return None;};
                let pointer = owner.device_ptr(&executor.stream).0;
                let bytes = u64::try_from(owner.len()).ok()?.checked_mul(2)?;
                let end = pointer.checked_add(bytes)?;
                (Arc::ptr_eq(owner.context(),executor.stream.context())
                    && pointer <= target.binding.pointer && target_end <= end).then(||json!({
                        "buffer_index":index,"dtype":"BF16","base_pointer":pointer,"elements":owner.len(),
                        "offset_bytes":target.binding.pointer-pointer,"same_context":true,
                    }))
            }).collect::<Vec<_>>();
            ensure!(
                owners.len() == 1,
                "BF16 target does not have one current typed containing allocation"
            );
            row["target_typed_owners"] = json!(owners);
            row["target_alias_value_count"] = executor
                .values
                .values()
                .filter(|v| v.binding.pointer == target.binding.pointer)
                .count()
                .into();
            row["target_pointer_kernel_uses"] = json!(
                executor
                    .kernels
                    .iter()
                    .enumerate()
                    .filter(|(_, k)| k.pointers.contains(&target.binding.pointer))
                    .map(|(index, k)| json!({"kernel_index":index,"label":k.label}))
                    .collect::<Vec<_>>()
            );
            row["target_pointer"] = target.binding.pointer.into();
            row["target_elements"] = target.binding.len.into();
            row["target_dtype"] = format!("{:?}", target.binding.dtype).into();
            row["target_source_shape"] = json!(target.source_shape);
            row["target_chain"] = json!(
                target
                    .chain
                    .iter()
                    .map(|n| json!({"kind":n.name(),"shape":n.shape()}))
                    .collect::<Vec<_>>()
            );
            row["target_normalized_map"] = format!("{normalized_target:?}").into();
            let materializers = executor.kernels.iter().enumerate().filter(|(_,k)| {
                k.label.starts_with("materialize F32 to BF16") && k.label.contains("physical-order")
                    && k.pointers == [source.binding.pointer,target.binding.pointer]
            }).map(|(index,k)|json!({"kernel_index":index,"label":k.label,"pointers":k.pointers})).collect::<Vec<_>>();
            ensure!(
                materializers.len() == 1,
                "expected exactly one current physical-order materializer"
            );
            row["materializers"] = json!(materializers);
            let parameter_aliases = graph
                .graph
                .node_indices()
                .filter(|&n| {
                    matches!(&graph[n],TensorGraphNode::Parameter{id:other,data:other_data,..}
                    if other == id && Arc::ptr_eq(other_data,data))
                })
                .collect::<Vec<_>>();
            row["parameter_graph_aliases"] = json!(
                parameter_aliases
                    .iter()
                    .map(|n| n.index())
                    .collect::<Vec<_>>()
            );
            row["parameter_f32_consumers"] = json!(parameter_aliases.iter().flat_map(|&p| {
                graph.graph.neighbors_directed(p,Direction::Outgoing)
                    .map(move |n|json!({"parameter_node":p.index(),"node":n.index(),"kind":graph[n].name(),"shape":graph[n].shape()}))
            }).collect::<Vec<_>>());
            row["f32_master_binding_preserved"] = parameter_aliases
                .iter()
                .all(|p| {
                    executor.values.get(p).is_some_and(|v| {
                        v.binding.dtype == DType::F32
                            && v.binding.pointer == owner.pointer
                            && v.binding.len == owner.len
                    })
                })
                .into();
            row["admitted_candidate"] = true.into();
            Ok(())
        })();
        row["fallback_reason"] = checked
            .err()
            .map(|e| format!("{e:#}"))
            .unwrap_or_default()
            .into();
        rows.push(row);
    }
    Ok(
        json!({"scope":"Actual completed executor cast/registered-owner census only. No shadow buffers/stores or production shadow selector exist.",
        "physical_order_cast_materializations":executor.info.physical_order_cast_materializations,
        "direct_master_parameters":executor.info.direct_master_parameters,"direct_master_bytes":executor.info.direct_master_bytes,
        "constructor_physical_cast_observations":executor.shadow_cast_observations.len(),
        "observation_version":"constructor_metadata_live_rebind_v2",
        "rows":rows}),
    )
}

#[test]
fn shadow_cast_owner_metadata_rejects_stale_partial_and_wrong_storage() {
    let value = Value {
        binding: Binding {
            pointer: 4096,
            len: 1024,
            dtype: DType::F32,
        },
        view: VirtualTensor::identity(NodeIndex::new(0), vec![32, 32], DType::F32),
        source_shape: vec![32, 32],
        chain: Vec::new(),
    };
    let owner = value.binding;
    assert!(owner_metadata_matches(&value, owner, 7, 7));
    assert!(!owner_metadata_matches(&value, owner, 7, 8));
    for bad in [
        Binding {
            pointer: 4100,
            ..owner
        },
        Binding { len: 1023, ..owner },
        Binding {
            dtype: DType::BF16,
            ..owner
        },
    ] {
        assert!(!owner_metadata_matches(&value, bad, 7, 7));
    }
}

#[test]
fn shadow_cast_parameter_trace_rejects_arithmetic_partial_and_broadcast() -> Result<()> {
    let mut graph = TensorGraph::<f32>::new();
    let root = graph.graph.add_node(TensorGraphNode::Parameter {
        id: 17,
        data: Arc::new(ParameterStorage::new(vec![0.; 1024])),
        shape: vec![32, 32],
    });
    let transpose = graph.graph.add_node(TensorGraphNode::Transpose {
        shape: vec![32, 32],
    });
    graph.graph.add_edge(root, transpose, 0);
    assert_eq!(parameter_origin(&graph, transpose)?, root);
    for rejected in [
        TensorGraphNode::Narrow {
            axis: 0,
            start: 1,
            shape: vec![31, 32],
        },
        TensorGraphNode::BroadcastAxis {
            axis: 0,
            shape: vec![64, 32],
        },
        TensorGraphNode::Linear {
            shape: vec![32, 32],
        },
    ] {
        let node = graph.graph.add_node(rejected);
        graph.graph.add_edge(root, node, 0);
        assert!(parameter_origin(&graph, node).is_err());
    }
    Ok(())
}
