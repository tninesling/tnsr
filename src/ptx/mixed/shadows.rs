//! Sparse owner-provided BF16 versions. Never part of the F32 upload registry.
use super::*;
use crate::tensor::ParameterStorage;

pub(crate) struct ShadowDeclaration {
    pub cast_node: NodeIndex,
    pub parameter_node: NodeIndex,
    pub parameter_id: usize,
    pub storage: Arc<ParameterStorage<f32>>,
    pub elements: usize,
    pub source_shape: Vec<usize>,
}

#[derive(Clone)]
pub(crate) struct ShadowBinding {
    pub parameter_node: NodeIndex,
    pub parameter_id: usize,
    pub storage: Arc<ParameterStorage<f32>>,
    pub revision: u64,
    /// Identity proof only; the separate master map owns the F32 allocation.
    pub master: Weak<CudaSlice<f32>>,
    pub shadow: Arc<CudaSlice<half::bf16>>,
}

pub(crate) struct RetiredShadowBindings {
    pub masters: HashMap<NodeIndex, Arc<CudaSlice<f32>>>,
    pub shadows: HashMap<usize, Arc<CudaSlice<half::bf16>>>,
}

/// Device-free declaration uses exactly the constructor's full-storage proof.
/// Unsupported casts retain their ordinary materialization.
pub(crate) fn shadow_declarations<G>(
    graph: &TensorGraph<f32, G>,
) -> Result<Vec<ShadowDeclaration>> {
    let mut declarations = Vec::new();
    for cast in graph.graph.node_indices() {
        if !matches!(
            graph[cast],
            TensorGraphNode::Cast {
                dtype: ValueDType::BF16,
                ..
            }
        ) {
            continue;
        }
        let Some((parameter, source)) = declaration_source(graph, cast) else {
            continue;
        };
        let TensorGraphNode::Parameter { id, data, shape } = &graph[parameter] else {
            continue;
        };
        if shape.len() != 2
            || source.binding.len <= 4096
            || physical_order_cast_source(&source, DType::BF16).is_none()
        {
            continue;
        }
        declarations.push(ShadowDeclaration {
            cast_node: cast,
            parameter_node: parameter,
            parameter_id: *id,
            storage: data.clone(),
            elements: source.binding.len,
            source_shape: shape.clone(),
        });
    }
    Ok(declarations)
}

fn declaration_source<G>(
    graph: &TensorGraph<f32, G>,
    cast: NodeIndex,
) -> Option<(NodeIndex, Value)> {
    let inputs = graph.inputs(cast);
    if inputs.len() != 1 {
        return None;
    }
    let mut node = inputs[0];
    let mut chain = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let parameter = loop {
        if !seen.insert(node) {
            return None;
        }
        match &graph[node] {
            TensorGraphNode::Parameter { .. } => break node,
            op @ (TensorGraphNode::Reshape { .. }
            | TensorGraphNode::Flatten { .. }
            | TensorGraphNode::Permute { .. }
            | TensorGraphNode::Transpose { .. }) => {
                chain.push(op.clone());
                let inputs = graph.inputs(node);
                let [input] = inputs.as_slice() else {
                    return None;
                };
                node = *input;
            }
            _ => return None,
        }
    };
    let shape = graph[parameter].shape().clone();
    let len = shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d))?;
    let mut view = VirtualTensor::identity(parameter, shape.clone(), DType::F32);
    chain.reverse();
    for op in &chain {
        view = match op {
            TensorGraphNode::Reshape { shape } | TensorGraphNode::Flatten { shape } => {
                view.reshape(shape.clone()).ok()?
            }
            TensorGraphNode::Permute { axes, .. } => view.permute(axes).ok()?,
            TensorGraphNode::Transpose { .. } => {
                let mut axes: Vec<_> = (0..view.shape.len()).collect();
                if axes.len() < 2 {
                    return None;
                }
                let n = axes.len();
                axes.swap(n - 2, n - 1);
                view.permute(&axes).ok()?
            }
            _ => return None,
        };
        if view.shape != *op.shape() {
            return None;
        }
    }
    if view.shape != *graph[cast].shape() {
        return None;
    }
    Some((
        parameter,
        Value {
            binding: Binding {
                pointer: 0,
                len,
                dtype: DType::F32,
            },
            view,
            source_shape: shape,
            chain,
        },
    ))
}

pub(super) fn validate_binding(
    stream: &Arc<CudaStream>,
    masters: &HashMap<NodeIndex, Arc<CudaSlice<f32>>>,
    shadow: &ShadowBinding,
) -> Result<()> {
    let master = masters
        .get(&shadow.parameter_node)
        .context("shadow master missing")?;
    let identity = shadow
        .master
        .upgrade()
        .context("shadow master identity expired")?;
    ensure!(
        Arc::ptr_eq(master, &identity),
        "shadow/master version identity mismatch"
    );
    ensure!(
        shadow.storage.revision()? == shadow.revision,
        "shadow revision is stale"
    );
    ensure!(
        master.len() > 4096
            && master.len() == shadow.shadow.len()
            && Arc::ptr_eq(master.context(), stream.context())
            && Arc::ptr_eq(shadow.shadow.context(), stream.context()),
        "shadow context/extent mismatch"
    );
    Ok(())
}

pub(super) fn validate_declarations<G>(
    graph: &TensorGraph<f32, G>,
    stream: &Arc<CudaStream>,
    masters: &HashMap<NodeIndex, Arc<CudaSlice<f32>>>,
    shadows: &HashMap<NodeIndex, ShadowBinding>,
) -> Result<()> {
    if shadows.is_empty() {
        return Ok(());
    }
    let declarations = shadow_declarations(graph)?;
    for (&cast, binding) in shadows {
        let declaration = declarations
            .iter()
            .find(|d| d.cast_node == cast)
            .context("shadow cast lacks full-storage declaration")?;
        ensure!(
            binding.parameter_node == declaration.parameter_node
                && binding.parameter_id == declaration.parameter_id
                && Arc::ptr_eq(&binding.storage, &declaration.storage)
                && binding.shadow.len() == declaration.elements
                && declaration.source_shape == *graph[declaration.parameter_node].shape(),
            "shadow declaration identity mismatch"
        );
        validate_binding(stream, masters, binding)?;
    }
    Ok(())
}

fn same_declared_view(expected: &Value, actual: &Value) -> Result<bool> {
    let same_chain = expected.chain.len() == actual.chain.len()
        && expected
            .chain
            .iter()
            .zip(&actual.chain)
            .all(|(a, b)| match (a, b) {
                (TensorGraphNode::Reshape { shape: a }, TensorGraphNode::Reshape { shape: b })
                | (TensorGraphNode::Flatten { shape: a }, TensorGraphNode::Flatten { shape: b })
                | (
                    TensorGraphNode::Transpose { shape: a },
                    TensorGraphNode::Transpose { shape: b },
                ) => a == b,
                (
                    TensorGraphNode::Permute { shape: a, axes: aa },
                    TensorGraphNode::Permute { shape: b, axes: ba },
                ) => a == b && aa == ba,
                _ => false,
            });
    if expected.binding.dtype != actual.binding.dtype
        || expected.binding.len != actual.binding.len
        || expected.source_shape != actual.source_shape
        || expected.view.dtype != actual.view.dtype
        || expected.view.shape != actual.view.shape
        || actual.view.predicate.is_some()
        || actual.view.out_of_bounds.is_some()
        || !same_chain
    {
        return Ok(false);
    }
    // f32_region's Values carry its local graph source labels. Only that
    // namespace label is ignored: the declared chain and complete map agree.
    Ok(expected
        .view
        .access
        .clone()
        .normalize_in_domain(&expected.view.shape)?
        == actual
            .view
            .access
            .clone()
            .normalize_in_domain(&actual.view.shape)?)
}

pub(super) fn cast_alias<G>(
    graph: &TensorGraph<f32, G>,
    node: NodeIndex,
    source: &Value,
    shadow: &ShadowBinding,
    stream: &Arc<CudaStream>,
    masters: &HashMap<NodeIndex, Arc<CudaSlice<f32>>>,
) -> Result<Value> {
    validate_binding(stream, masters, shadow)?;
    let physical = physical_order_cast_source(source, DType::BF16)
        .context("shadow cast lost full-storage physical-order proof")?;
    let (declared_parameter, declared_source) =
        declaration_source(graph, node).context("shadow cast graph declaration disappeared")?;
    ensure!(
        declared_parameter == shadow.parameter_node
            && same_declared_view(&declared_source, source)?,
        "shadow runtime view differs from declared parameter view: cast={} parameter={} physical_source={}",
        node.index(),
        shadow.parameter_node.index(),
        physical.view.source.index()
    );
    let master = &masters[&shadow.parameter_node];
    ensure!(
        physical.binding.pointer == master.device_ptr(stream).0
            && physical.binding.len == master.len(),
        "shadow cast master source changed: cast={} parameter={} physical_source={} source_pointer={} source_len={} master_pointer={} master_len={} source_shape={:?} logical_shape={:?} chain={:?}",
        node.index(),
        shadow.parameter_node.index(),
        physical.view.source.index(),
        physical.binding.pointer,
        physical.binding.len,
        master.device_ptr(stream).0,
        master.len(),
        source.source_shape,
        source.view.shape,
        source
            .chain
            .iter()
            .map(|op| (op.name(), op.shape()))
            .collect::<Vec<_>>()
    );
    Ok(physical_order_cast_result(
        source,
        node,
        Binding {
            pointer: shadow.shadow.device_ptr(stream).0,
            len: shadow.shadow.len(),
            dtype: DType::BF16,
        },
    ))
}

/// External shadows never become private output/gradient storage or zero targets.
pub(super) fn private_storage(executor: &MixedExecutor) -> Vec<Binding> {
    let mut private: Vec<_> = executor
        .buffers
        .iter()
        .map(|buffer| binding(buffer, &executor.stream))
        .collect();
    for program in &executor.programs {
        if let Some(program) = program.downcast_ref::<Program<F32>>() {
            private.extend(program.storage_bindings(&executor.stream));
        }
        if let Some(program) = program.downcast_ref::<Program<BF16>>() {
            private.extend(program.storage_bindings(&executor.stream));
        }
    }
    private.extend(executor.index_buffers.values().map(|buffer| Binding {
        pointer: buffer.device_ptr(&executor.stream).0,
        len: buffer.len(),
        dtype: DType::U32,
    }));
    private.extend(
        executor
            .kernels
            .iter()
            .filter_map(|kernel| kernel.zero_before),
    );
    private
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shadow_declared_view_ignores_only_source_namespace_not_square_bijection() -> Result<()> {
        let mut graph = TensorGraph::<f32>::new();
        let parameter = graph.graph.add_node(TensorGraphNode::Parameter {
            id: 19,
            data: Arc::new(ParameterStorage::new(vec![0.; 16384])),
            shape: vec![128, 128],
        });
        let transpose = graph.graph.add_node(TensorGraphNode::Transpose {
            shape: vec![128, 128],
        });
        graph.graph.add_edge(parameter, transpose, 0);
        let cast = graph.graph.add_node(TensorGraphNode::Cast {
            dtype: ValueDType::BF16,
            shape: vec![128, 128],
        });
        graph.graph.add_edge(transpose, cast, 0);
        let (_, declared) = declaration_source(&graph, cast).context("declared source missing")?;
        let mut local = declared.clone();
        local.view.source = NodeIndex::new(999);
        assert!(physical_order_cast_source(&local, DType::BF16).is_some());
        assert!(same_declared_view(&declared, &local)?);
        // Two transposes are a whole-storage bijection with the same square
        // shape. They still disagree with this graph's declared single transpose.
        let mut wrong = local.clone();
        wrong.view = wrong.view.permute(&[1, 0])?;
        wrong.chain.push(TensorGraphNode::Transpose {
            shape: vec![128, 128],
        });
        assert!(physical_order_cast_source(&wrong, DType::BF16).is_some());
        assert!(!same_declared_view(&declared, &wrong)?);
        let mut wrong_shape = local.clone();
        wrong_shape.source_shape = vec![64, 256];
        assert!(!same_declared_view(&declared, &wrong_shape)?);
        let mut wrong_dtype = local;
        wrong_dtype.binding.dtype = DType::BF16;
        assert!(!same_declared_view(&declared, &wrong_dtype)?);
        Ok(())
    }

    #[test]
    fn shadow_declaration_admits_only_complete_reordered_large_matrices() -> Result<()> {
        let mut graph = TensorGraph::<f32>::new();
        let storage = Arc::new(ParameterStorage::new(vec![0.; 8192]));
        let parameter = graph.graph.add_node(TensorGraphNode::Parameter {
            id: 71,
            data: storage.clone(),
            shape: vec![64, 128],
        });
        let transpose = graph.graph.add_node(TensorGraphNode::Transpose {
            shape: vec![128, 64],
        });
        graph.graph.add_edge(parameter, transpose, 0);
        let cast = graph.graph.add_node(TensorGraphNode::Cast {
            dtype: ValueDType::BF16,
            shape: vec![128, 64],
        });
        graph.graph.add_edge(transpose, cast, 0);
        // Identity casts, partial windows and arithmetic do not qualify.
        let plain = graph.graph.add_node(TensorGraphNode::Cast {
            dtype: ValueDType::BF16,
            shape: vec![64, 128],
        });
        graph.graph.add_edge(parameter, plain, 0);
        for rejected in [
            TensorGraphNode::Narrow {
                axis: 0,
                start: 1,
                shape: vec![63, 128],
            },
            TensorGraphNode::BroadcastAxis {
                axis: 0,
                shape: vec![128, 128],
            },
            TensorGraphNode::Linear {
                shape: vec![64, 128],
            },
        ] {
            let shape = rejected.shape().clone();
            let view = graph.graph.add_node(rejected);
            graph.graph.add_edge(parameter, view, 0);
            let rejected_cast = graph.graph.add_node(TensorGraphNode::Cast {
                dtype: ValueDType::BF16,
                shape,
            });
            graph.graph.add_edge(view, rejected_cast, 0);
        }
        let declared = shadow_declarations(&graph)?;
        assert_eq!(declared.len(), 1);
        assert_eq!(declared[0].cast_node, cast);
        assert_eq!(declared[0].parameter_node, parameter);
        assert_eq!(declared[0].parameter_id, 71);
        assert_eq!(declared[0].elements, 8192);
        assert!(Arc::ptr_eq(&declared[0].storage, &storage));
        Ok(())
    }

    #[test]
    fn shadow_declaration_keeps_small_batch_members_and_bad_shapes_ordinary() -> Result<()> {
        let mut graph = TensorGraph::<f32>::new();
        for (id, shape, transposed) in [
            (1, vec![64, 64], vec![64, 64]),
            (2, vec![8192], vec![8192]),
            (3, vec![64, 128], vec![64, 128]), // wrong transpose metadata
        ] {
            let len = shape.iter().product();
            let parameter = graph.graph.add_node(TensorGraphNode::Parameter {
                id,
                data: Arc::new(ParameterStorage::new(vec![0.; len])),
                shape,
            });
            let view = graph.graph.add_node(TensorGraphNode::Transpose {
                shape: transposed.clone(),
            });
            graph.graph.add_edge(parameter, view, 0);
            let cast = graph.graph.add_node(TensorGraphNode::Cast {
                dtype: ValueDType::BF16,
                shape: transposed,
            });
            graph.graph.add_edge(view, cast, 0);
        }
        assert!(shadow_declarations(&graph)?.is_empty());
        Ok(())
    }
}
