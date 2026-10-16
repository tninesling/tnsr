use super::*;

impl MixedExecutor {
    pub(super) fn f32_region<G>(
        &mut self,
        graph: &TensorGraph<f32, G>,
        nodes: &[NodeIndex],
        precision: MatMulPrecision,
    ) -> Result<()> {
        use petgraph::Direction;
        use std::collections::HashSet;
        if nodes.is_empty() {
            return Ok(());
        }
        let members: HashSet<_> = nodes.iter().copied().collect();
        let mut plain = TensorGraph::<f32>::new();
        let mut mapped = HashMap::new();
        let mut bindings = HashMap::new();
        for &node in nodes {
            let mut operands = Vec::new();
            for input in graph.inputs(node) {
                let end = if let Some(&end) = mapped.get(&input) {
                    end
                } else {
                    let source = self
                        .values
                        .get(&input)
                        .context("f32 region boundary missing")?
                        .clone();
                    let n = if source.binding.dtype == DType::U32 {
                        let TensorGraphNode::IndexSource { source: index } =
                            &self.index_graph[source.view.source]
                        else {
                            anyhow::bail!("region integer source missing")
                        };
                        TensorGraphNode::IndexSource {
                            source: index.clone(),
                        }
                    } else {
                        TensorGraphNode::Input {
                            name: "boundary",
                            shape: source.source_shape.clone(),
                        }
                    };
                    let mut end = plain.graph.add_node(n);
                    bindings.insert(end, source.binding);
                    for view in &source.chain {
                        let next = plain.graph.add_node(typed_node(view)?);
                        plain.graph.add_edge(end, next, 0);
                        end = next;
                    }
                    mapped.insert(input, end);
                    end
                };
                operands.push(end);
            }
            let end = if matches!(
                graph[node],
                TensorGraphNode::Cast {
                    dtype: ValueDType::F32,
                    ..
                }
            ) {
                *operands.first().context("identity cast has no input")?
            } else {
                let end = plain.graph.add_node(typed_node(&graph[node])?);
                for (slot, input) in operands.into_iter().enumerate() {
                    plain.graph.add_edge(input, end, slot);
                }
                end
            };
            mapped.insert(node, end);
        }
        // Make every value needed by a later region an explicit sink. The
        // ordinary f32 planner can then fuse without discarding saved primals.
        for &node in nodes {
            let consumers: Vec<_> = graph
                .graph
                .neighbors_directed(node, Direction::Outgoing)
                .collect();
            if consumers.is_empty() || consumers.iter().any(|n| !members.contains(n)) {
                let sink = plain.graph.add_node(TensorGraphNode::Reshape {
                    shape: graph[node].shape().clone(),
                });
                plain.graph.add_edge(mapped[&node], sink, 0);
            }
        }
        let program =
            Program::<F32>::new_bound(&self.stream, plain.clone(), bindings, None, precision)?;
        self.info.f32_bytes += program.bytes();
        self.kernels.extend(program.kernels.clone());
        for &original in nodes {
            let node = mapped[&original];
            let Some(&physical) = program.values.get(&node) else {
                continue;
            };
            let value = if let Some(view) = program.views.get(&node) {
                let mut chain = Vec::new();
                let mut cursor = node;
                while cursor != view.source {
                    ensure!(
                        is_view(&plain[cursor]),
                        "f32 region virtual path crosses arithmetic"
                    );
                    chain.push(plain[cursor].clone());
                    cursor = *plain
                        .inputs(cursor)
                        .first()
                        .context("f32 virtual path has no input")?;
                }
                chain.reverse();
                Value {
                    binding: *program
                        .values
                        .get(&view.source)
                        .context("f32 region virtual source missing")?,
                    view: view.clone(),
                    source_shape: plain[view.source].shape().clone(),
                    chain,
                }
            } else {
                Value {
                    binding: physical,
                    view: VirtualTensor::identity(
                        original,
                        graph[original].shape().clone(),
                        physical.dtype,
                    ),
                    source_shape: graph[original].shape().clone(),
                    chain: Vec::new(),
                }
            };
            self.values.insert(original, value);
        }
        self.programs.push(Box::new(program));
        Ok(())
    }
}
