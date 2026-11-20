//! Late checkpoint-retaining replacement. Preparation owns only metadata/function
//! clones, never a second allocation arena. Original kernels are installed first.
use super::*;
use recognize::Region;
use std::collections::{BTreeMap, HashMap, HashSet};

#[derive(Clone)]
struct Counts {
    tensor: usize,
    eight: usize,
    four: usize,
    medium: (usize, usize),
    large: (usize, usize),
    square: (usize, usize),
    batched: crate::tile::BatchedAttentionTileCounts,
    column: (usize, usize),
}
impl Counts {
    fn program<D: crate::ptx::CudaDType>(p: &Program<D>) -> Self {
        Self {
            tensor: usize::from(p.ptx.contains("f32.bf16.bf16.f32")),
            eight: usize::from(p.bf16_eight_compute_warps),
            four: p.ordinary_bf16_four_warp_count,
            medium: (
                usize::from(p.medium_bf16_tile_counts.0 > 0),
                usize::from(p.medium_bf16_tile_counts.1 > 0),
            ),
            large: (
                usize::from(p.large_bf16_tile_counts.0 > 0),
                usize::from(p.large_bf16_tile_counts.1 > 0),
            ),
            square: p.large_square128_streamed_counts,
            batched: p.batched_attention_bf16_tile_counts,
            column: p.column_reduction_counts,
        }
    }
    fn debit(&self, i: &mut MixedInfo) -> Option<()> {
        macro_rules! dec {
            ($field:ident,$n:expr) => {
                i.$field = i.$field.checked_sub($n)?;
            };
        }
        dec!(bf16_tensor_core_programs, self.tensor);
        dec!(bf16_eight_compute_warp_programs, self.eight);
        dec!(ordinary_bf16_four_warp_programs, self.four);
        dec!(medium_bf16_square_programs, self.medium.0);
        dec!(medium_bf16_tall_programs, self.medium.1);
        dec!(large_bf16_tall_programs, self.large.0);
        dec!(large_bf16_wide_programs, self.large.1);
        dec!(large_bf16_square128_streamed_programs, self.square.0);
        dec!(
            large_bf16_square128_streamed_fallback_programs,
            self.square.1
        );
        dec!(batched_attention_score_programs, self.batched.scores);
        dec!(
            batched_attention_query_width_programs,
            self.batched.query_width
        );
        dec!(
            batched_attention_width_query_programs,
            self.batched.width_query
        );
        dec!(batched_attention_tall_programs, self.batched.tall);
        dec!(batched_attention_wide_programs, self.batched.wide);
        dec!(column_reduction_four_regions, self.column.0);
        dec!(column_reduction_eight_regions, self.column.1);
        Some(())
    }
}
struct Group {
    kernels: Vec<Kernel>,
    counts: Option<Counts>,
}
pub(in crate::ptx::mixed) struct Preparation {
    enabled: bool,
    regions: Vec<Region>,
    values: HashMap<NodeIndex, Value>,
    groups: HashMap<NodeIndex, Group>,
    needed: HashSet<NodeIndex>,
}
fn policy_eligible(r: &Region) -> bool {
    r.shape[0]
        .checked_mul(r.shape[1])
        .is_some_and(|bh| bh >= 128)
}
fn nodes(r: &Region) -> [NodeIndex; 8] {
    [
        r.query,
        r.key,
        r.value,
        r.mask,
        r.qk,
        r.probability,
        r.narrowed,
        r.output,
    ]
}
impl Preparation {
    pub(in crate::ptx::mixed) fn new<G>(g: &TensorGraph<f32, G>, enabled: bool) -> Self {
        let regions = recognize::recognize(g);
        let needed = regions
            .iter()
            .filter(|r| policy_eligible(r))
            .flat_map(nodes)
            .collect();
        Self {
            enabled,
            regions,
            values: HashMap::new(),
            groups: HashMap::new(),
            needed,
        }
    }
    pub(in crate::ptx::mixed) fn extend_last_consumers(
        &self,
        positions: &HashMap<NodeIndex, usize>,
        last: &mut HashMap<NodeIndex, usize>,
    ) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }
        for r in self.regions.iter().filter(|r| policy_eligible(r)) {
            let end = *positions
                .get(&r.output)
                .context("attention PV absent from topology")?;
            for n in nodes(r)
                .into_iter()
                .chain(r.early_checkpoint_aliases.iter().copied())
            {
                last.entry(n)
                    .and_modify(|v| *v = (*v).max(end))
                    .or_insert(end);
            }
        }
        Ok(())
    }
    pub(in crate::ptx::mixed) fn record_operation<D: crate::ptx::CudaDType>(
        &mut self,
        node: NodeIndex,
        program: &Program<D>,
        output: &Value,
        values: &HashMap<NodeIndex, Value>,
    ) {
        if !self.enabled
            || !self
                .regions
                .iter()
                .any(|r| policy_eligible(r) && (node == r.qk || node == r.output))
        {
            return;
        }
        self.groups.insert(
            node,
            Group {
                kernels: program.kernels.clone(),
                counts: Some(Counts::program(program)),
            },
        );
        self.values.insert(node, output.clone());
        self.capture_values(values);
    }
    fn capture_values(&mut self, values: &HashMap<NodeIndex, Value>) {
        for n in &self.needed {
            if let Some(v) = values.get(n) {
                self.values.insert(*n, v.clone());
            }
        }
    }
    /// Must precede EVERY prefix/final compaction, before values are pruned.
    pub(in crate::ptx::mixed) fn capture(
        &mut self,
        values: &HashMap<NodeIndex, Value>,
        kernels: &[Kernel],
    ) {
        if !self.enabled {
            return;
        }
        self.capture_values(values);
        for r in self.regions.iter().filter(|r| policy_eligible(r)) {
            if let std::collections::hash_map::Entry::Vacant(entry) =
                self.groups.entry(r.probability)
            {
                let group: Vec<_> = kernels
                    .iter()
                    .filter(|k| k.owner == Some(r.probability))
                    .cloned()
                    .collect();
                if !group.is_empty() {
                    entry.insert(Group {
                        kernels: group,
                        counts: None,
                    });
                }
            }
        }
    }
    /// Semantic owners are resolved anew: LN's late rewrite changes positions.
    pub(in crate::ptx::mixed) fn prospective_uses(&self, kernels: &[Kernel]) -> Vec<(usize, u64)> {
        self.prospective_from_owners(kernels.iter().map(|k| k.owner))
    }
    fn prospective_from_owners(
        &self,
        owners: impl Iterator<Item = Option<NodeIndex>> + Clone,
    ) -> Vec<(usize, u64)> {
        let mut uses = Vec::new();
        if !self.enabled {
            return uses;
        }
        for r in self.regions.iter().filter(|r| policy_eligible(r)) {
            if let Some(target) = owner_end(owners.clone(), r.output) {
                for n in nodes(r) {
                    if let Some(v) = self.values.get(&n) {
                        uses.push((target, v.binding.pointer));
                    }
                }
            }
        }
        uses
    }
    pub(in crate::ptx::mixed) fn remap(&mut self, map: &HashMap<u64, u64>) {
        for v in self.values.values_mut() {
            v.binding.pointer = *map.get(&v.binding.pointer).unwrap_or(&v.binding.pointer);
        }
        for g in self.groups.values_mut() {
            for k in &mut g.kernels {
                for p in &mut k.pointers {
                    *p = *map.get(p).unwrap_or(p);
                }
                if let Some(b) = &mut k.zero_before {
                    b.pointer = *map.get(&b.pointer).unwrap_or(&b.pointer);
                }
            }
        }
    }
    fn group_indices(&self, node: NodeIndex, kernels: &[Kernel]) -> Option<Vec<usize>> {
        let original = self.groups.get(&node)?;
        let indices: Vec<_> = kernels
            .iter()
            .enumerate()
            .filter_map(|(i, k)| (k.owner == Some(node)).then_some(i))
            .collect();
        if indices.is_empty()
            || indices.len() != original.kernels.len()
            || !indices.windows(2).all(|a| a[0] + 1 == a[1])
        {
            return None;
        }
        for (&index, old) in indices.iter().zip(&original.kernels) {
            let k = &kernels[index];
            if k.owner != old.owner
                || k.label != old.label
                || k.pointers != old.pointers
                || k.zero_before.is_some()
                || old.zero_before.is_some()
                || k.config.grid_dim != old.config.grid_dim
                || k.config.block_dim != old.config.block_dim
                || k.config.shared_mem_bytes != old.config.shared_mem_bytes
            {
                return None;
            }
        }
        Some(indices)
    }
    fn patch(&self, r: &Region, e: &MixedExecutor) -> Result<Option<Patch>> {
        macro_rules! fallback {
            () => {{
                #[cfg(test)]
                eprintln!(
                    "attention training fallback line {} PV {:?}",
                    line!(),
                    r.output
                );
                return Ok(None);
            }};
        }
        if !policy_eligible(r) {
            fallback!();
        }
        if r.scale.to_bits() != 0.125f32.to_bits()
            || ![r.qk, r.probability, r.output]
                .iter()
                .all(|n| r.members.contains(n))
        {
            fallback!();
        }
        let Some(bindings) = nodes(r)
            .iter()
            .map(|n| self.values.get(n).cloned())
            .collect::<Option<Vec<_>>>()
        else {
            fallback!();
        };
        let [q, k, v, mask, qk, p, p16, out]: [Value; 8] = match bindings.try_into() {
            Ok(v) => v,
            Err(_) => return Ok(None),
        };
        let Some(layout) = input_layout::prove_triplet(&q, &k, &v, r.shape[0], r.shape[1]) else {
            fallback!();
        };
        let mask_shape = vec![1, 1, SEQUENCE, SEQUENCE];
        if !identity(&mask, &mask_shape, DType::F32)
            || !identity(&qk, &r.shape, DType::BF16)
            || !identity(&p, &r.shape, DType::F32)
            || !identity(&p16, &r.shape, DType::BF16)
            || !identity(
                &out,
                &[r.shape[0], r.shape[1], SEQUENCE, WIDTH],
                DType::BF16,
            )
        {
            fallback!();
        }
        if [
            (r.qk, &qk),
            (r.probability, &p),
            (r.narrowed, &p16),
            (r.output, &out),
        ]
        .iter()
        .any(|(node, value)| value.view.source != *node)
        {
            fallback!();
        }
        let outputs = [qk.binding, p.binding, p16.binding, out.binding];
        let inputs = [q.binding, k.binding, v.binding, mask.binding];
        if !disjoint(&outputs, &inputs) {
            fallback!();
        }
        let Some(q_indices) = self.group_indices(r.qk, &e.kernels) else {
            fallback!();
        };
        let Some(n_indices) = self.group_indices(r.probability, &e.kernels) else {
            fallback!();
        };
        let Some(o_indices) = self.group_indices(r.output, &e.kernels) else {
            fallback!();
        };
        if n_indices.len() != 1
            || e.kernels[n_indices[0]].pointers
                != vec![
                    qk.binding.pointer,
                    mask.binding.pointer,
                    p.binding.pointer,
                    p16.binding.pointer,
                ]
        {
            fallback!();
        }
        let Some(q_counts) = self.groups.get(&r.qk).and_then(|g| g.counts.clone()) else {
            fallback!();
        };
        let Some(o_counts) = self.groups.get(&r.output).and_then(|g| g.counts.clone()) else {
            fallback!();
        };
        // Actual original TensorCoreBF16 programs prove the required K16 MMA family.
        if q_counts.tensor != 1
            || o_counts.tensor != 1
            || q_indices.last() >= n_indices.first()
            || n_indices.last() >= o_indices.first()
        {
            fallback!();
        }
        for (indices, value) in [(&q_indices, &qk), (&o_indices, &out)] {
            if !indices
                .iter()
                .any(|&i| e.kernels[i].pointers.contains(&value.binding.pointer))
            {
                fallback!();
            }
        }
        let geometry = Geometry::new(
            r.shape[0]
                .checked_mul(r.shape[1])
                .context("attention BH overflow")?,
            32,
        )?;
        let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
        tiles.region_kernels = vec![crate::tile::graph::TileRegionKernel {
            region_id: 0,
            ir: ir_with_input_layout(geometry, true, layout)?,
        }];
        tiles.optimize_indices()?;
        let graph = PtxGraph::from(tiles).with_target(PtxTarget::from_context(e.stream.context())?);
        let Some(name) = graph.region_kernel_name(0) else {
            fallback!();
        };
        let module = match e
            .stream
            .context()
            .load_module(Ptx::from_src(graph.to_ptx()))
        {
            Ok(m) => m,
            Err(_) => return Ok(None),
        };
        let function = match module.load_function(name) {
            Ok(f) => f,
            Err(_) => return Ok(None),
        };
        if !runtime_admitted(
            &function,
            geometry,
            PtxTarget::from_context(e.stream.context())?
                .matmul_resources
                .shared_bytes_per_block,
        )
        .unwrap_or(false)
        {
            fallback!();
        }
        #[cfg(test)]
        if FORCE_FALLBACK.with(|f| f.get()) {
            fallback!();
        }
        let kernel = Kernel {
            owner: Some(r.output),
            label: format!(
                "attention training forward four checkpoints q32 shape={:?}",
                r.shape
            ),
            function,
            config: geometry.launch()?,
            pointers: vec![
                q.binding.pointer,
                k.binding.pointer,
                v.binding.pointer,
                mask.binding.pointer,
                qk.binding.pointer,
                p.binding.pointer,
                p16.binding.pointer,
                out.binding.pointer,
            ],
            zero_before: None,
        };
        Ok(Some(Patch {
            remove: q_indices
                .into_iter()
                .chain(n_indices)
                .chain(o_indices.iter().copied())
                .collect(),
            insert: *o_indices.last().context("attention PV group empty")?,
            kernel,
            counts: [q_counts, o_counts],
            values: [
                (r.qk, qk),
                (r.probability, p),
                (r.narrowed, p16),
                (r.output, out),
            ],
        }))
    }
    pub(in crate::ptx::mixed) fn apply(mut self, e: &mut MixedExecutor) -> Result<usize> {
        self.capture(&e.values, &e.kernels);
        let mut remove = HashSet::new();
        let mut inserts = BTreeMap::new();
        let mut updates = HashMap::new();
        let mut info = e.info.clone();
        let mut accepted = 0;
        for r in self.regions.iter().filter(|r| policy_eligible(r)) {
            if !self.enabled {
                continue;
            }
            let Ok(Some(patch)) = self.patch(r, e) else {
                continue;
            };
            if patch.remove.iter().any(|i| remove.contains(i)) {
                continue;
            }
            let mut next = info.clone();
            if patch.counts.iter().any(|c| c.debit(&mut next).is_none()) {
                continue;
            }
            let Some(norms) = next.attention_softmax_regions.checked_sub(1) else {
                continue;
            };
            next.attention_softmax_regions = norms;
            info = next;
            remove.extend(patch.remove);
            inserts.insert(patch.insert, patch.kernel);
            updates.extend(patch.values);
            accepted += 1;
        }
        // No executor state is changed until every accepted complete patch and
        // counter debit has been validated; every fallback keeps all old groups.
        if accepted > 0 {
            let old = std::mem::take(&mut e.kernels);
            for (i, k) in old.into_iter().enumerate() {
                if let Some(new) = inserts.remove(&i) {
                    e.kernels.push(new);
                }
                if !remove.contains(&i) {
                    e.kernels.push(k);
                }
            }
            e.values.extend(updates);
        }
        info.attention_training_forward_regions = accepted;
        info.attention_training_forward_fallback_regions = self.regions.len() - accepted;
        info.attention_training_forward_materializations = 0;
        e.info = info;
        Ok(accepted)
    }
}
struct Patch {
    remove: Vec<usize>,
    insert: usize,
    kernel: Kernel,
    counts: [Counts; 2],
    values: [(NodeIndex, Value); 4],
}
fn identity(v: &Value, shape: &[usize], dtype: DType) -> bool {
    shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .is_some_and(|n| v.binding.len == n)
        && v.binding.dtype == dtype
        && v.view.dtype == dtype
        && v.source_shape == shape
        && v.view.shape == shape
        && v.view.access == crate::tile::IndexMap::identity(shape.len())
        && v.view.predicate.is_none()
        && v.view.out_of_bounds.is_none()
        && v.chain.is_empty()
}
fn range(b: Binding) -> Option<(u64, u64)> {
    if b.pointer == 0 || b.len == 0 || !b.pointer.is_multiple_of(b.dtype.size_bytes() as u64) {
        return None;
    }
    Some((
        b.pointer,
        b.pointer
            .checked_add(u64::try_from(b.len.checked_mul(b.dtype.size_bytes())?).ok()?)?,
    ))
}
fn disjoint(outputs: &[Binding], inputs: &[Binding]) -> bool {
    let Some(out) = outputs
        .iter()
        .copied()
        .map(range)
        .collect::<Option<Vec<_>>>()
    else {
        return false;
    };
    let Some(input) = inputs
        .iter()
        .copied()
        .map(range)
        .collect::<Option<Vec<_>>>()
    else {
        return false;
    };
    let overlaps = |(a, b): (u64, u64), (c, d): (u64, u64)| a < d && c < b;
    out.iter().enumerate().all(|(i, &a)| {
        out.iter().skip(i + 1).all(|&b| !overlaps(a, b)) && input.iter().all(|&b| !overlaps(a, b))
    })
}
#[cfg(test)]
thread_local! {static FORCE_FALLBACK:std::cell::Cell<bool>=const {std::cell::Cell::new(false)};}
#[cfg(test)]
mod tests;

// Production policy is stricter than the exploratory leaf's diagnostic gate.
fn resources_fit(
    registers: i32,
    local: i32,
    shared: i32,
    max_threads: i32,
    capacity: u32,
    block_limit: usize,
) -> bool {
    (1..=128).contains(&registers)
        && local == 0
        && shared == 44 * 1024
        && usize::try_from(shared).is_ok_and(|s| s <= block_limit)
        && max_threads >= 256
        && capacity >= 2
}
fn runtime_admitted(
    function: &cudarc::driver::CudaFunction,
    geometry: Geometry,
    block_limit: usize,
) -> Result<bool> {
    Ok(geometry.query_tile == 32
        && geometry.shared_bytes() == 44 * 1024
        && resources_fit(
            function.num_regs()?,
            function.local_size_bytes()?,
            function.shared_size_bytes()?,
            function.max_threads_per_block()?,
            function.occupancy_max_active_blocks_per_multiprocessor(256, 0, None)?,
            block_limit,
        ))
}

fn owner_end(owners: impl Iterator<Item = Option<NodeIndex>>, target: NodeIndex) -> Option<usize> {
    let mut matched = None;
    let mut last = None;
    for (i, owner) in owners.enumerate() {
        last = Some(i);
        if owner == Some(target) {
            matched = Some(i);
        }
    }
    matched.or(last)
}

#[cfg(test)]
mod gpu_tests;
