//! Late, atomic replacement of original compiled programs before first run or
//! capture. One executor owns one set of real allocations. Metadata owns no CUDA
//! slices/master leases, and prospective reads enter every prefix reuse pass.
use super::*;
use crate::ptx::{PtxExecutionPlan, PtxPlanAction};
use std::collections::{BTreeMap, HashSet};
struct Record {
    plan: PtxExecutionPlan,
    mapped: HashMap<NodeIndex, NodeIndex>,
    kernel_for_step: HashMap<usize, usize>,
}
pub(in crate::ptx::mixed) struct Preparation {
    pairs: Vec<LayerNormPair>,
    records: Vec<Record>,
    values: HashMap<NodeIndex, Value>,
    arithmetic: HashSet<NodeIndex>,
    inputs: HashMap<NodeIndex, Vec<NodeIndex>>,
    add_nodes: HashSet<NodeIndex>,
    aliases: HashMap<NodeIndex, NodeIndex>,
}
impl Record {
    fn owner(&self, n: NodeIndex) -> Option<usize> {
        let plain = self.mapped.get(&n)?;
        self.plan.steps().iter().enumerate().find_map(|(i, s)| {
            (s.members.contains(plain) && self.kernel_for_step.contains_key(&i)).then_some(i)
        })
    }
    fn original_members(&self, step: usize) -> HashSet<NodeIndex> {
        let members = &self.plan.steps()[step].members;
        self.mapped
            .iter()
            .filter_map(|(&old, new)| members.contains(new).then_some(old))
            .collect()
    }
}
impl Preparation {
    pub(in crate::ptx::mixed) fn new<G>(g: &TensorGraph<f32, G>) -> Result<Self> {
        let pairs = recognize(g)?
            .into_iter()
            .filter(|p| planned_contiguous(p, g))
            .collect();
        let dtypes = value_dtypes(g);
        let mut aliases = HashMap::new();
        for n in g.graph.node_indices() {
            if matches!(
                g[n],
                TensorGraphNode::Cast {
                    dtype: ValueDType::F32,
                    ..
                }
            ) && let Some(a) = one(g, n)
                && dtypes.get(&a) == Some(&ValueDType::F32)
                && g[a].shape() == g[n].shape()
            {
                aliases.insert(n, a);
            }
        }
        Ok(Self {
            pairs,
            records: Vec::new(),
            values: HashMap::new(),
            arithmetic: g
                .graph
                .node_indices()
                .filter(|n| {
                    !is_view(&g[*n])
                        && !matches!(
                            g[*n],
                            TensorGraphNode::Input { .. }
                                | TensorGraphNode::Parameter { .. }
                                | TensorGraphNode::Constant { .. }
                                | TensorGraphNode::IndexSource { .. }
                                | TensorGraphNode::Cast {
                                    dtype: ValueDType::F32,
                                    ..
                                }
                        )
                })
                .collect(),
            inputs: g.graph.node_indices().map(|n| (n, g.inputs(n))).collect(),
            add_nodes: g
                .graph
                .node_indices()
                .filter(|n| {
                    matches!(
                        g[*n],
                        TensorGraphNode::Binary {
                            op: BinaryOp::Add,
                            ..
                        }
                    )
                })
                .collect(),
            aliases,
        })
    }
    pub(in crate::ptx::mixed) fn extend_last_consumers(
        &self,
        positions: &HashMap<NodeIndex, usize>,
        last: &mut HashMap<NodeIndex, usize>,
    ) -> Result<()> {
        for pair in &self.pairs {
            lifetime::extend_last_consumers(pair, positions, last)?;
        }
        Ok(())
    }
    pub(in crate::ptx::mixed) fn record(
        &mut self,
        program: &mut Program<F32>,
        mapped: &HashMap<NodeIndex, NodeIndex>,
        start: usize,
        values: &HashMap<NodeIndex, Value>,
    ) -> Result<()> {
        let metadata = program
            .metadata
            .take()
            .context("LN original program metadata missing")?;
        ensure!(
            metadata.kernel_steps.len() == program.kernels.len(),
            "LN original kernel/step mapping mismatch"
        );
        #[cfg(test)]
        if let Some(directory) = std::env::var_os("TNSR_LN_PTX_DIR") {
            let directory = std::path::PathBuf::from(directory);
            std::fs::create_dir_all(&directory)?;
            let prefix = directory.join(format!("original-program-{}", self.records.len()));
            std::fs::write(prefix.with_extension("ptx"), &program.ptx)?;
            std::fs::write(
                prefix.with_extension("txt"),
                format!("plan={:#?}\nmapping={mapped:#?}\n", metadata.plan),
            )?;
        }
        self.records.push(Record {
            plan: metadata.plan,
            mapped: mapped.clone(),
            kernel_for_step: metadata
                .kernel_steps
                .into_iter()
                .enumerate()
                .map(|(k, s)| (s, start + k))
                .collect(),
        });
        self.values
            .extend(values.iter().map(|(&n, v)| (n, v.clone())));
        Ok(())
    }
    pub(in crate::ptx::mixed) fn prospective_uses(&self) -> Vec<(usize, u64)> {
        let mut uses = Vec::new();
        for pair in &self.pairs {
            let target = self.records.iter().find_map(|r| {
                r.owner(pair.backward)
                    .and_then(|s| r.kernel_for_step.get(&s).copied())
            });
            let Some(target) = target else {
                continue;
            };
            for n in [
                pair.inputs[0],
                pair.inputs[1],
                pair.inputs[3],
                pair.storage_roots[0],
                pair.storage_roots[1],
                pair.storage_roots[3],
                pair.mean,
                pair.variance_plus_epsilon,
                pair.inverse_std,
            ] {
                if let Some(v) = self.values.get(&n) {
                    uses.push((target, v.binding.pointer));
                }
            }
        }
        uses
    }
    pub(in crate::ptx::mixed) fn remap(&mut self, remap: &HashMap<u64, u64>) {
        for v in self.values.values_mut() {
            if let Some(p) = remap.get(&v.binding.pointer) {
                v.binding.pointer = *p;
            }
        }
    }
    fn canonical(&self, mut n: NodeIndex) -> Option<NodeIndex> {
        for _ in 0..=self.aliases.len() {
            match self.aliases.get(&n) {
                Some(&a) => n = a,
                None => return Some(n),
            }
        }
        None
    }
    fn evidence(&self, pair: &LayerNormPair) -> Result<Option<kernel::TailBoundary>> {
        use boundaries::Boundary;
        let mut merged: BTreeMap<&str, Boundary> = BTreeMap::new();
        let mut row_trees = HashSet::new();
        for record in &self.records {
            let Some(proved) =
                boundaries::row_trees_from_original_plan(&record.plan, &record.mapped, pair)?
            else {
                return Ok(None);
            };
            row_trees.extend(proved);
            for edge in boundaries::from_original_plan(&record.plan, &record.mapped, pair)? {
                ensure!(
                    edge.producer != edge.consumer,
                    "LN original edge collapsed in symbolic template"
                );
                if edge.boundary == Boundary::OutsideProgram {
                    continue;
                }
                // Repeated boundary aliases must agree on category, not a local
                // step number from another original program.
                if let Some(old) = merged.get(edge.label) {
                    if std::mem::discriminant(old) != std::mem::discriminant(&edge.boundary) {
                        #[cfg(test)]
                        eprintln!(
                            "LN contradictory edge {}: {old:?} vs {:?}, pair {:?}",
                            edge.label, edge.boundary, pair.forward
                        );
                        return Ok(None);
                    }
                } else {
                    merged.insert(edge.label, edge.boundary);
                }
            }
        }
        if merged.len() != 18 || row_trees.len() != 4 {
            #[cfg(test)]
            eprintln!(
                "LN incomplete edge coverage: {:?}, pair {:?}",
                merged, pair.forward
            );
            return Ok(None);
        }
        for label in [
            "mean -> center",
            "center -> square",
            "square -> variance Mean",
            "variance Mean -> epsilon Add",
            "quotient -> variance-gradient Sum",
            "variance-gradient Sum -> dS scale",
            "dS scale -> first square VJP",
            "dS scale -> second square VJP",
            "direct center VJP -> first Add",
            "first square VJP -> first Add",
            "second square VJP -> second Add",
            "negative center VJP -> mean-gradient Sum",
            "mean-gradient Sum -> mean scale",
        ] {
            if !matches!(merged.get(label), Some(Boundary::Stored { .. })) {
                #[cfg(test)]
                eprintln!(
                    "LN unsupported stored edge {label}: {:?}, pair {:?}",
                    merged.get(label),
                    pair.forward
                );
                return Ok(None);
            }
        }
        for label in [
            "epsilon Add -> Log",
            "inverse std -> normalized",
            "normalized -> affine Mul",
            "affine Mul -> bias Add",
        ] {
            if !matches!(merged.get(label), Some(Boundary::SameKernel { .. })) {
                #[cfg(test)]
                eprintln!(
                    "LN unsupported same-kernel edge {label}: {:?}, pair {:?}",
                    merged.get(label),
                    pair.forward
                );
                return Ok(None);
            }
        }
        Ok(match merged.get("mean scale -> inner dZ Add") {
            Some(Boundary::Stored { .. }) => Some(kernel::TailBoundary::SeparateMeanTerm),
            Some(Boundary::SameKernel {
                action: PtxPlanAction::PointwiseRegion(_),
                ..
            }) => Some(kernel::TailBoundary::OriginalPointwiseOpportunity),
            _ => None,
        })
    }
    pub(in crate::ptx::mixed) fn apply(self, executor: &mut MixedExecutor) -> Result<()> {
        let mut removals = HashSet::new();
        let mut inserts: BTreeMap<usize, Vec<Kernel>> = BTreeMap::new();
        let mut updates = HashMap::new();
        let mut dead = HashSet::new();
        let mut accepted = 0;
        for pair in &self.pairs {
            let Some(patch) = self.patch(pair, executor)? else {
                continue;
            };
            if patch.remove.iter().any(|i| removals.contains(i)) {
                continue;
            }
            removals.extend(patch.remove);
            for (i, k) in patch.insert {
                inserts.entry(i).or_default().extend(k);
            }
            updates.extend(patch.values);
            dead.extend(pair.members.iter().copied());
            accepted += 1;
        }
        // Commit only after all source/layout/18-edge/resource checks and
        // candidate compilation succeed. No old kernel has run or captured.
        let old = std::mem::take(&mut executor.kernels);
        for (i, k) in old.into_iter().enumerate() {
            if let Some(new) = inserts.remove(&i) {
                executor.kernels.extend(new);
            }
            if !removals.contains(&i) {
                executor.kernels.push(k);
            }
        }
        ensure!(
            inserts.is_empty(),
            "LN insertion target beyond original program"
        );
        executor.values.retain(|n, _| !dead.contains(n));
        executor.values.extend(updates);
        executor.info.layer_norm_pair_forward_regions = accepted;
        executor.info.layer_norm_pair_backward_regions = accepted;
        executor.info.layer_norm_pair_fallback_regions = self.pairs.len() - accepted;
        Ok(())
    }
    fn patch(&self, pair: &LayerNormPair, executor: &MixedExecutor) -> Result<Option<Patch>> {
        let Some(tail) = self.evidence(pair)? else {
            #[cfg(test)]
            eprintln!(
                "LN fallback at rewrite source line 289, pair {:?}",
                pair.forward
            );
            return Ok(None);
        };
        let Some(forward_inputs) = contiguous_bindings(pair, &self.values, false) else {
            #[cfg(test)]
            eprintln!(
                "LN fallback at rewrite source line 292, pair {:?}",
                pair.forward
            );
            return Ok(None);
        };
        let Some(back_inputs) = contiguous_bindings(pair, &self.values, true) else {
            #[cfg(test)]
            eprintln!(
                "LN fallback at rewrite source line 295, pair {:?}",
                pair.forward
            );
            return Ok(None);
        };
        let total = pair.shape.iter().product::<usize>();
        let rows = total / pair.shape[2];
        let output = |n, len| {
            self.values
                .get(&n)
                .filter(|v| {
                    v.binding.dtype == DType::F32
                        && v.binding.len == len
                        && v.view.predicate.is_none()
                        && v.view.out_of_bounds.is_none()
                        && v.view.access.is_identity()
                })
                .cloned()
        };
        let Some(y) = output(pair.forward, total) else {
            #[cfg(test)]
            eprintln!(
                "LN fallback at rewrite source line 312, pair {:?}",
                pair.forward
            );
            return Ok(None);
        };
        let Some(n) = output(pair.normalized, total) else {
            #[cfg(test)]
            eprintln!(
                "LN fallback at rewrite source line 315, pair {:?}",
                pair.forward
            );
            return Ok(None);
        };
        let Some(mean) = output(pair.mean, rows) else {
            #[cfg(test)]
            eprintln!(
                "LN fallback at rewrite source line 318, pair {:?}",
                pair.forward
            );
            return Ok(None);
        };
        let Some(a) = output(pair.variance_plus_epsilon, total) else {
            #[cfg(test)]
            eprintln!(
                "LN fallback at rewrite source line 321, pair {:?}",
                pair.forward
            );
            return Ok(None);
        };
        let Some(r) = output(pair.inverse_std, total) else {
            #[cfg(test)]
            eprintln!(
                "LN fallback at rewrite source line 324, pair {:?}",
                pair.forward
            );
            return Ok(None);
        };
        let forward_members = pair.forward_members.clone();
        let mut fw = HashSet::new();
        let mut bw = HashSet::new();
        let mut outer: Option<(NodeIndex, NodeIndex, bool)> = None;
        for record in &self.records {
            for (&step, &index) in &record.kernel_for_step {
                let members = record.original_members(step);
                if !members.iter().any(|n| pair.members.contains(n)) {
                    continue;
                }
                let is_forward = members.iter().any(|n| forward_members.contains(n));
                let extra: Vec<_> = members
                    .iter()
                    .copied()
                    .filter(|n| self.arithmetic.contains(n) && !pair.members.contains(n))
                    .collect();
                if !extra.is_empty() {
                    if is_forward
                        || extra.len() != 1
                        || record.owner(pair.backward) != Some(step)
                        || outer.is_some()
                    {
                        #[cfg(test)]
                        eprintln!(
                            "LN fallback at rewrite source line 348, pair {:?}",
                            pair.forward
                        );
                        return Ok(None);
                    }
                    let out = extra[0];
                    if !self.add_nodes.contains(&out) {
                        #[cfg(test)]
                        eprintln!(
                            "LN fallback at rewrite source line 352, pair {:?}",
                            pair.forward
                        );
                        return Ok(None);
                    }
                    let Some(args) = self.inputs.get(&out) else {
                        #[cfg(test)]
                        eprintln!(
                            "LN fallback at rewrite source line 355, pair {:?}",
                            pair.forward
                        );
                        return Ok(None);
                    };
                    if args.len() != 2 {
                        #[cfg(test)]
                        eprintln!(
                            "LN fallback at rewrite source line 358, pair {:?}",
                            pair.forward
                        );
                        return Ok(None);
                    }
                    let x = self.canonical(args[0]);
                    let z = self.canonical(args[1]);
                    outer = if x == Some(pair.backward) {
                        Some((out, args[1], true))
                    } else if z == Some(pair.backward) {
                        Some((out, args[0], false))
                    } else {
                        #[cfg(test)]
                        eprintln!(
                            "LN fallback at rewrite source line 367, pair {:?}",
                            pair.forward
                        );
                        return Ok(None);
                    };
                }
                if is_forward {
                    fw.insert(index);
                } else {
                    bw.insert(index);
                }
            }
        }
        let (Some(&fwd_at), Some(&back_at)) = (fw.iter().max(), bw.iter().max()) else {
            #[cfg(test)]
            eprintln!(
                "LN fallback at rewrite source line 378, pair {:?}",
                pair.forward
            );
            return Ok(None);
        };
        if fwd_at >= *bw.iter().min().unwrap_or(&0) || fw.iter().any(|i| bw.contains(i)) {
            #[cfg(test)]
            eprintln!(
                "LN fallback at rewrite source line 381, pair {:?}",
                pair.forward
            );
            return Ok(None);
        }
        let cp =
            ssa::checkpoint_candidates(pair).context("LN ordered checkpoint anchors missing")?;
        ensure!(
            cp.requires_baseline_contraction_audit,
            "LN baseline proof contract changed"
        );
        let cc2 = pair
            .program
            .iter()
            .find(|op| op.node == pair.backward)
            .and_then(|op| op.inputs.first())
            .and_then(|i| pair.program.get(*i))
            .map(|op| op.node)
            .context("LN center-gradient anchor missing")?;
        let dz = match output(pair.backward, total) {
            Some(v) => v,
            None => match output(cc2, total) {
                Some(mut v) => {
                    v.view = VirtualTensor::identity(pair.backward, pair.shape.clone(), DType::F32);
                    v.source_shape = pair.shape.clone();
                    v.chain.clear();
                    v
                }
                None => return Ok(None),
            },
        };
        let mut stats = Vec::new();
        for layout in lifetime::stat_layouts(pair)? {
            let old = if layout.node == pair.mean {
                &mean
            } else if layout.node == pair.variance_plus_epsilon {
                &a
            } else {
                &r
            };
            let mut binding = old.binding;
            binding.len = rows;
            stats.push((
                layout.node,
                Value {
                    binding,
                    view: layout.view,
                    source_shape: layout.source_shape,
                    chain: layout.chain,
                },
            ));
        }
        let mut fp: Vec<_> = forward_inputs.iter().map(|v| v.binding.pointer).collect();
        fp.extend([
            y.binding.pointer,
            n.binding.pointer,
            mean.binding.pointer,
            a.binding.pointer,
            r.binding.pointer,
        ]);
        if fp.iter().collect::<HashSet<_>>().len() != fp.len() {
            #[cfg(test)]
            eprintln!(
                "LN fallback at rewrite source line 439, pair {:?}",
                pair.forward
            );
            return Ok(None);
        }
        let mut bp: Vec<_> = back_inputs.iter().map(|v| v.binding.pointer).collect();
        bp.extend([mean.binding.pointer, a.binding.pointer, r.binding.pointer]);
        if bp.contains(&dz.binding.pointer) {
            #[cfg(test)]
            eprintln!(
                "LN fallback at rewrite source line 444, pair {:?}",
                pair.forward
            );
            return Ok(None);
        }
        bp.push(dz.binding.pointer);
        let Some(fwd) = compile(
            executor,
            kernel::forward(pair)?,
            fp,
            rows,
            128,
            "LayerNorm pair forward F32",
        )?
        else {
            #[cfg(test)]
            eprintln!(
                "LN fallback at rewrite source line 456, pair {:?}",
                pair.forward
            );
            return Ok(None);
        };
        let Some(back) = compile(
            executor,
            kernel::backward(pair, tail, back_inputs[2].binding.dtype)?,
            bp,
            rows,
            128,
            "LayerNorm pair backward F32",
        )?
        else {
            #[cfg(test)]
            eprintln!(
                "LN fallback at rewrite source line 467, pair {:?}",
                pair.forward
            );
            return Ok(None);
        };
        let mut back_kernels = vec![back];
        let mut values = HashMap::from([
            (pair.forward, y),
            (pair.normalized, n),
            (pair.backward, dz.clone()),
        ]);
        values.extend(stats);
        if let Some((out, other, dz_first)) = outer {
            let Some(other) = self.values.get(&other).and_then(|v| flat(v, total)) else {
                #[cfg(test)]
                eprintln!(
                    "LN fallback at rewrite source line 478, pair {:?}",
                    pair.forward
                );
                return Ok(None);
            };
            let Some(target) = output(out, total) else {
                #[cfg(test)]
                eprintln!(
                    "LN fallback at rewrite source line 481, pair {:?}",
                    pair.forward
                );
                return Ok(None);
            };
            if target.binding.pointer == dz.binding.pointer
                || target.binding.pointer == other.pointer
            {
                #[cfg(test)]
                eprintln!(
                    "LN fallback at rewrite source line 486, pair {:?}",
                    pair.forward
                );
                return Ok(None);
            }
            let ins = if dz_first {
                [dz.binding, other]
            } else {
                [other, dz.binding]
            };
            let ir = outer_add_ir(total, ins[0].dtype, ins[1].dtype);
            let Some(add) = compile(
                executor,
                ir,
                vec![ins[0].pointer, ins[1].pointer, target.binding.pointer],
                total.div_ceil(256),
                256,
                "LayerNorm preserved outer Add F32",
            )?
            else {
                #[cfg(test)]
                eprintln!(
                    "LN fallback at rewrite source line 503, pair {:?}",
                    pair.forward
                );
                return Ok(None);
            };
            back_kernels.push(add);
            values.insert(out, target);
        }
        let mut remove = fw;
        remove.extend(bw);
        #[cfg(test)]
        if let Some(directory) = std::env::var_os("TNSR_LN_PTX_DIR") {
            let records = self.records.iter().enumerate().map(|(program,record)| {
                Ok(serde_json::json!({"program":program,"edges": boundaries::from_original_plan(&record.plan,&record.mapped,pair)?.iter().map(|edge|serde_json::json!({"label":edge.label,"producer":edge.producer.index(),"consumer":edge.consumer.index(),"boundary":format!("{:?}",edge.boundary)})).collect::<Vec<_>>()}))
            }).collect::<Result<Vec<_>>>()?;
            let proof = serde_json::json!({"forward":pair.forward.index(),"backward":pair.backward.index(),"shape":&pair.shape,"incoming_storage":format!("{:?}",back_inputs[2].binding.dtype),"tail":format!("{tail:?}"),"original_program_edges":records});
            let path = std::path::PathBuf::from(directory)
                .join(format!("pair-{}-evidence.json", pair.forward.index()));
            std::fs::write(path, serde_json::to_vec_pretty(&proof)?)?;
        }
        Ok(Some(Patch {
            remove,
            insert: BTreeMap::from([(fwd_at, vec![fwd]), (back_at, back_kernels)]),
            values,
        }))
    }
}
struct Patch {
    remove: HashSet<usize>,
    insert: BTreeMap<usize, Vec<Kernel>>,
    values: HashMap<NodeIndex, Value>,
}
fn flat(v: &Value, count: usize) -> Option<Binding> {
    if !matches!(v.binding.dtype, DType::F32 | DType::BF16)
        || v.view.dtype != v.binding.dtype
        || v.binding.len != count
        || v.view.predicate.is_some()
        || v.view.out_of_bounds.is_some()
    {
        return None;
    }
    let mut expected =
        VirtualTensor::identity(v.view.source, v.source_shape.clone(), v.binding.dtype);
    for alias in &v.chain {
        match alias {
            TensorGraphNode::Reshape { shape } | TensorGraphNode::Flatten { shape }
                if shape.iter().product::<usize>() == count =>
            {
                expected = expected.reshape(shape.clone()).ok()?
            }
            _ => return None,
        }
    }
    (expected == v.view).then_some(v.binding)
}
fn compile(
    executor: &MixedExecutor,
    ir: crate::tile::TileIR,
    pointers: Vec<u64>,
    blocks: usize,
    threads: u32,
    label: &str,
) -> Result<Option<Kernel>> {
    let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
    tiles.region_kernels = vec![crate::tile::graph::TileRegionKernel { region_id: 0, ir }];
    tiles.optimize_indices()?;
    let graph =
        PtxGraph::from(tiles).with_target(PtxTarget::from_context(executor.stream.context())?);
    let name = graph.region_kernel_name(0).context("LN kernel missing")?;
    let ptx = graph.to_ptx();
    #[cfg(test)]
    let artifact_prefix = if let Some(directory) = std::env::var_os("TNSR_LN_PTX_DIR") {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let directory = std::path::PathBuf::from(directory);
        std::fs::create_dir_all(&directory)?;
        let sequence = NEXT.fetch_add(1, Ordering::Relaxed);
        let prefix = directory.join(format!("candidate-{sequence}"));
        std::fs::write(prefix.with_extension("ptx"), &ptx)?;
        std::fs::write(prefix.with_extension("txt"), label)?;
        Some(prefix)
    } else {
        None
    };
    let module = executor.stream.context().load_module(Ptx::from_src(ptx))?;
    let function = module.load_function(name)?;
    #[cfg(test)]
    if let Some(prefix) = artifact_prefix {
        let resources = serde_json::json!({
            "label":label,"function":name,"registers_per_thread":function.num_regs()?,
            "static_shared_bytes":function.shared_size_bytes()?,"local_bytes_per_thread":function.local_size_bytes()?,
            "block_threads":threads,"grid_blocks":blocks,"max_threads_per_block":function.max_threads_per_block()?,
            "api_max_active_blocks_per_sm":function.occupancy_max_active_blocks_per_multiprocessor(threads,0,None)?,
            "scope":"actual driver JIT resources; capacity is not achieved occupancy"});
        std::fs::write(
            prefix.with_extension("resources.json"),
            serde_json::to_vec_pretty(&resources)?,
        )?;
    }
    if threads
        > u32::try_from(function.max_threads_per_block()?).context("LN negative thread limit")?
        || function.occupancy_max_active_blocks_per_multiprocessor(threads, 0, None)? == 0
    {
        #[cfg(test)]
        eprintln!("LN resource fallback for {label}");
        return Ok(None);
    }
    Ok(Some(Kernel {
        owner: None,
        label: label.into(),
        function,
        config: LaunchConfig {
            grid_dim: (u32::try_from(blocks)?, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        },
        pointers,
        zero_before: None,
    }))
}
fn outer_add_ir(total: usize, a: DType, c: DType) -> crate::tile::TileIR {
    use crate::tile::{
        Block, Dim, Expr, KernelParam, ScalarBinaryOp, ScalarBuilder, ScalarExpr, ScalarMemory,
        Stmt, TileIR,
    };
    let mut b = ScalarBuilder::default();
    b.stmts.push(Stmt::BoundsCheck { extent: total });
    let i = Expr::BlockIdx(Dim::X) * 256usize + Expr::ThreadIdx(Dim::X);
    let x = b.load(ScalarMemory::Global("left".into()), i.clone());
    let y = b.load(ScalarMemory::Global("right".into()), i.clone());
    let z = b.value(ScalarExpr::binary(
        ScalarBinaryOp::Arithmetic(BinaryOp::Add),
        x,
        y,
    ));
    b.store(ScalarMemory::Global("output".into()), i, z);
    TileIR {
        kernel_name: "layer_norm_outer_add".into(),
        params: vec![
            KernelParam {
                name: "left".into(),
                dtype: a,
                is_input: true,
            },
            KernelParam {
                name: "right".into(),
                dtype: c,
                is_input: true,
            },
            KernelParam {
                name: "output".into(),
                dtype: DType::F32,
                is_input: false,
            },
        ],
        body: Block { stmts: b.stmts },
        shared_mem_bytes: 0,
    }
}
