//! Default-off numerical-screen preparation and atomic complete-pair rewrite.
//!
//! Retain the entire original allocation pool. All old computation is installed
//! first; only complete, preflighted three-kernel pairs may replace it before
//! constructor warmup/capture. This is not an allocation/performance milestone.
use super::{
    backward, forward,
    leaf::{self, Descriptor, PairedFunctions, Shape},
    pair_recognize::{self, CompletePair},
};
use crate::{
    graph::{NodeIndex, TensorGraph, TensorGraphNode},
    ptx::{
        CudaDType, PtxGraph, PtxTarget,
        linear::{
            Binding, Buffer,
            program::{Kernel, Program},
        },
        mixed::{MixedExecutor, MixedInfo, Value},
    },
    tile::{DType, TileGraph, TileIR, VirtualTensor},
};
use anyhow::{Context, Result, ensure};
use cudarc::{
    driver::{CudaFunction, CudaSlice, DevicePtr, LaunchConfig},
    nvrtc::Ptx,
};
use std::collections::{BTreeMap, HashMap, HashSet};
mod packed;

#[derive(Clone, Debug)]
pub(in crate::ptx::mixed) enum Rejection {
    MissingValues,
    MissingSemanticOwner(NodeIndex, &'static str),
    ChangedOriginalGroup,
    UnsupportedOriginalGroups,
    UnverifiedPack,
    UnsupportedLayout,
    AliasedBindings,
    EarlySurvivingReader,
    Compilation,
    Resources,
    CounterInvariant,
    OverlappingPair,
}
impl std::fmt::Display for Rejection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingSemanticOwner(node, role) => {
                write!(f, "missing {role} owner {}", node.index())
            }
            other => write!(f, "{other:?}"),
        }
    }
}
#[derive(Debug)]
pub(in crate::ptx::mixed) struct PairFallback {
    pub output: NodeIndex,
    pub reason: Rejection,
}
#[derive(Debug, Default)]
pub(in crate::ptx::mixed) struct CommittedScreen {
    pub selected_pairs: usize,
    pub register_statistics_pairs: usize,
    pub global_statistics_pairs: usize,
    pub register_statistics_fallbacks: Vec<PairFallback>,
    pub packed_outputs: Vec<NodeIndex>,
    pub packed_fallbacks: Vec<PairFallback>,
    pub packed_pack_kernels_removed: usize,
    pub async16_pairs: usize,
    pub scalar_pairs: usize,
    pub fallbacks: Vec<PairFallback>,
    pub workspace_bytes: usize,
    pub retained_outputs: Vec<[NodeIndex; 4]>,
    /// Raw pair owner is dKT; packed pair owners are the single pack output.
    /// This explicit metadata describes both logical writes.
    pub joint_logical_outputs: Vec<[NodeIndex; 2]>,
    /// Caller must not run any later rewrite or allocation reuse after commit.
    pub terminal_late_pass: bool,
}
#[derive(Clone)]
struct Counts {
    tensor: usize,
    eight: usize,
    four: usize,
    plain: (usize, usize),
    medium: (usize, usize),
    large: (usize, usize),
    square: (usize, usize),
    batched: crate::tile::BatchedAttentionTileCounts,
    column: (usize, usize),
}
impl Counts {
    fn program<D: CudaDType>(p: &Program<D>) -> Self {
        Self {
            tensor: usize::from(p.ptx.contains("f32.bf16.bf16.f32")),
            eight: usize::from(p.bf16_eight_compute_warps),
            four: p.ordinary_bf16_four_warp_count,
            plain: p.plain_bf16_output_counts,
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
        macro_rules! sub {
            ($field:ident,$n:expr) => {
                i.$field = i.$field.checked_sub($n)?;
            };
        }
        sub!(bf16_tensor_core_programs, self.tensor);
        sub!(bf16_eight_compute_warp_programs, self.eight);
        sub!(ordinary_bf16_four_warp_programs, self.four);
        sub!(plain_bf16_output_programs, self.plain.0);
        sub!(plain_bf16_output_fallback_programs, self.plain.1);
        sub!(medium_bf16_square_programs, self.medium.0);
        sub!(medium_bf16_tall_programs, self.medium.1);
        sub!(large_bf16_tall_programs, self.large.0);
        sub!(large_bf16_wide_programs, self.large.1);
        sub!(large_bf16_square128_streamed_programs, self.square.0);
        sub!(
            large_bf16_square128_streamed_fallback_programs,
            self.square.1
        );
        sub!(batched_attention_score_programs, self.batched.scores);
        sub!(
            batched_attention_query_width_programs,
            self.batched.query_width
        );
        sub!(
            batched_attention_width_query_programs,
            self.batched.width_query
        );
        sub!(batched_attention_tall_programs, self.batched.tall);
        sub!(batched_attention_wide_programs, self.batched.wide);
        sub!(column_reduction_four_regions, self.column.0);
        sub!(column_reduction_eight_regions, self.column.1);
        Some(())
    }
}
#[derive(Clone)]
struct Group {
    kernels: Vec<Kernel>,
    counts: Option<Counts>,
}
#[derive(Clone)]
struct PackInput {
    source: NodeIndex,
    views: Vec<TensorGraphNode<f32>>,
    window: usize,
}
struct Pack {
    output: NodeIndex,
    shape: Vec<usize>,
    inputs: Vec<PackInput>,
    tree: crate::ptx::mixed::qkv_grad_pack::Join,
    members: HashSet<NodeIndex>,
    readers: Option<packed::Readers>,
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum StorageKey {
    Global(NodeIndex),
    F32 { program: usize, physical: NodeIndex },
}
#[derive(Clone)]
struct Origin {
    key: StorageKey,
    binding: Binding,
    producer: Vec<Kernel>,
}
struct Region {
    proof: CompletePair,
    mask: NodeIndex,
    /// Logical aliases and their physical-source roots must all remain visible
    /// to prefix future-live accounting. Captured Values alone do not pin them.
    reservations: HashSet<NodeIndex>,
}

pub(in crate::ptx::mixed) struct Preparation {
    packed_requested: bool,
    register_statistics_requested: bool,
    #[cfg(test)]
    force_register_statistics_rejection: bool,
    regions: Vec<Region>,
    needed: HashSet<NodeIndex>,
    never_prune: HashSet<NodeIndex>,
    values: HashMap<NodeIndex, Value>,
    groups: HashMap<NodeIndex, Group>,
    packs: HashMap<NodeIndex, Pack>,
    aliases: HashMap<NodeIndex, NodeIndex>,
    boundaries: HashMap<NodeIndex, Vec<Kernel>>,
    origins: HashMap<NodeIndex, Origin>,
    unknown_origins: HashSet<NodeIndex>,
    f32_programs: usize,
}
fn target_from_positions(prefix_len: usize, ends: &[Option<usize>]) -> Option<usize> {
    let completed = ends.iter().copied().collect::<Option<Vec<_>>>();
    match completed {
        Some(ends) if !ends.is_empty() => ends.into_iter().max(),
        _ => prefix_len.checked_sub(1),
    }
}
#[cfg(test)]
fn target_from_owners(prefix: &[Option<NodeIndex>], owners: &[NodeIndex]) -> Option<usize> {
    target_from_positions(
        prefix.len(),
        &owners
            .iter()
            .map(|owner| prefix.iter().rposition(|n| *n == Some(*owner)))
            .collect::<Vec<_>>(),
    )
}
fn physical_member_kernels(
    node: NodeIndex,
    members: &[Vec<NodeIndex>],
    kernel_steps: &[usize],
) -> Option<Vec<usize>> {
    let steps = members
        .iter()
        .enumerate()
        .filter_map(|(step, nodes)| {
            (nodes.contains(&node) && kernel_steps.contains(&step)).then_some(step)
        })
        .collect::<Vec<_>>();
    let [step] = steps.as_slice() else {
        return None;
    };
    Some(
        kernel_steps
            .iter()
            .enumerate()
            .filter_map(|(index, &owner)| (owner == *step).then_some(index))
            .collect(),
    )
}
fn unique_stamp_indices(old: &[Stamp], current: &[Stamp]) -> Option<Vec<usize>> {
    if old.is_empty() || old.iter().chain(current).any(|s| s.function.is_none()) {
        return None;
    }
    let mut result = Vec::new();
    for expected in old {
        let matches = current
            .iter()
            .enumerate()
            .filter_map(|(i, actual)| (actual == expected).then_some(i))
            .collect::<Vec<_>>();
        let [index] = matches.as_slice() else {
            return None;
        };
        result.push(*index);
    }
    result
        .windows(2)
        .all(|w| w[0] + 1 == w[1])
        .then_some(result)
}
fn group_indices(old: &[Kernel], current: &[Kernel]) -> Option<Vec<usize>> {
    unique_stamp_indices(
        &old.iter().map(stamp).collect::<Vec<_>>(),
        &current.iter().map(stamp).collect::<Vec<_>>(),
    )
}
fn remap_pointer_uses(pointers: &mut [u64], zero: &mut Option<Binding>, map: &HashMap<u64, u64>) {
    for p in pointers {
        *p = *map.get(p).unwrap_or(p);
    }
    if let Some(b) = zero {
        b.pointer = *map.get(&b.pointer).unwrap_or(&b.pointer);
    }
}
fn remap_kernel(k: &mut Kernel, map: &HashMap<u64, u64>) {
    remap_pointer_uses(&mut k.pointers, &mut k.zero_before, map);
}
fn storage_aliases<G>(g: &TensorGraph<f32, G>) -> HashMap<NodeIndex, NodeIndex> {
    let dtypes = crate::graph::mixed::value_dtypes(g);
    let mut result = HashMap::new();
    for n in g.graph.node_indices() {
        let inputs = g.inputs(n);
        let [parent] = inputs.as_slice() else {
            continue;
        };
        let alias = match &g[n] {
            TensorGraphNode::Reshape { .. }
            | TensorGraphNode::Flatten { .. }
            | TensorGraphNode::Transpose { .. }
            | TensorGraphNode::Permute { .. } => true,
            TensorGraphNode::Cast { dtype, .. } => {
                *dtype == crate::tensor::ValueDType::F32 || dtypes.get(parent) == Some(dtype)
            }
            _ => false,
        };
        if alias {
            result.insert(n, *parent);
        }
    }
    result
}
fn reservation_ancestors<G>(
    g: &TensorGraph<f32, G>,
    nodes: impl IntoIterator<Item = NodeIndex>,
) -> HashSet<NodeIndex> {
    let mut result = HashSet::new();
    for mut n in nodes {
        loop {
            if !result.insert(n) {
                break;
            }
            if !matches!(
                g[n],
                TensorGraphNode::Reshape { .. }
                    | TensorGraphNode::Flatten { .. }
                    | TensorGraphNode::Transpose { .. }
                    | TensorGraphNode::Permute { .. }
                    | TensorGraphNode::Cast { .. }
            ) {
                break;
            }
            let inputs = g.inputs(n);
            let [parent] = inputs.as_slice() else {
                break;
            };
            n = *parent;
        }
    }
    result
}
fn discarded_global_alias_keys(
    seeds: &HashSet<NodeIndex>,
    aliases: &HashMap<NodeIndex, NodeIndex>,
    protected: &HashSet<NodeIndex>,
) -> HashSet<NodeIndex> {
    let mut discarded = seeds.difference(protected).copied().collect::<HashSet<_>>();
    loop {
        let before = discarded.len();
        for (&child, parent) in aliases {
            if !protected.contains(&child) && discarded.contains(parent) {
                discarded.insert(child);
            }
        }
        if before == discarded.len() {
            return discarded;
        }
    }
}
fn requested_transport(bindings: &[Binding]) -> leaf::Transport {
    if bindings.len() == 8
        && bindings
            .iter()
            .all(|b| b.dtype == DType::BF16 && b.pointer != 0 && b.pointer.is_multiple_of(16))
    {
        leaf::Transport::Async16
    } else {
        leaf::Transport::Scalar
    }
}
fn pair_ir(
    shape: Shape,
    layouts: [leaf::ReadLayout; 4],
    transport: leaf::Transport,
    statistics: forward::StatisticsStorage,
) -> Result<[TileIR; 3]> {
    match transport {
        leaf::Transport::Scalar => Ok([
            forward::ir_with_statistics(shape, layouts[0], transport, statistics)?,
            backward::d_query_ir(shape, layouts[0], layouts[3])?,
            backward::d_key_value_ir(shape, layouts[0], layouts[3])?,
        ]),
        leaf::Transport::Async16 => Ok([
            forward::ir_with_statistics(shape, layouts[0], transport, statistics)?,
            backward::d_query_ir_with_transport(shape, layouts[0], layouts[3], transport)?,
            backward::d_key_value_ir_with_transport(shape, layouts[0], layouts[3], transport)?,
        ]),
    }
}
fn register_statistics_scope(shape: Shape, layouts: [leaf::ReadLayout; 4]) -> bool {
    shape.heads == 8
        && [64, 128, 256].contains(&shape.bh())
        && layouts
            .iter()
            .all(|layout| *layout == leaf::ReadLayout::SequenceMajor)
}
// A failed new mode never commits a partial pair: retry the complete old pair.
fn select_statistics_pair<T>(
    requested: bool,
    eligible: bool,
    mut compile: impl FnMut(forward::StatisticsStorage) -> std::result::Result<T, Rejection>,
) -> std::result::Result<(T, forward::StatisticsStorage, Option<Rejection>), Rejection> {
    use forward::StatisticsStorage::{Global, Registers};
    let rejected = if requested {
        if eligible {
            match compile(Registers) {
                Ok(pair) => return Ok((pair, Registers, None)),
                Err(reason) => Some(reason),
            }
        } else {
            Some(Rejection::UnsupportedLayout)
        }
    } else {
        None
    };
    Ok((compile(Global)?, Global, rejected))
}
fn compile_pair(
    e: &MixedExecutor,
    irs: [TileIR; 3],
) -> std::result::Result<PairedFunctions<CudaFunction>, Rejection> {
    let [forward, dq, dkv] = irs;
    let functions = PairedFunctions::from_all(
        Some(compile(e, forward).map_err(|_| Rejection::Compilation)?),
        Some(compile(e, dq).map_err(|_| Rejection::Compilation)?),
        Some(compile(e, dkv).map_err(|_| Rejection::Compilation)?),
    )
    .map_err(|_| Rejection::Compilation)?;
    for f in [
        &functions.forward,
        &functions.d_query_and_delta,
        &functions.d_key_value,
    ] {
        if !resources(e, f).unwrap_or(false) {
            return Err(Rejection::Resources);
        }
    }
    Ok(functions)
}
impl Preparation {
    /// The caller must reject simultaneous checkpoint-forward fusion. Candidate
    /// packs are conditional here; stage verifies their actual owners and ABI.
    #[cfg(test)]
    pub(in crate::ptx::mixed) fn new<G>(g: &TensorGraph<f32, G>) -> Self {
        Self::with_packed_backward(g, false)
    }
    #[cfg(test)]
    pub(in crate::ptx::mixed) fn with_packed_backward<G>(
        g: &TensorGraph<f32, G>,
        packed_requested: bool,
    ) -> Self {
        Self::with_statistics(g, packed_requested, false, false)
    }
    pub(in crate::ptx::mixed) fn with_statistics<G>(
        g: &TensorGraph<f32, G>,
        packed_requested: bool,
        register_statistics_requested: bool,
        #[cfg(test)] force_register_statistics_rejection: bool,
    ) -> Self {
        let packs = crate::ptx::mixed::qkv_grad_pack::recognize(g)
            .into_iter()
            .map(|p| {
                (
                    p.output,
                    Pack {
                        output: p.output,
                        shape: p.shape,
                        inputs: p
                            .inputs
                            .into_iter()
                            .map(|i| PackInput {
                                source: i.source,
                                views: i.views,
                                window: i.window,
                            })
                            .collect(),
                        tree: p.tree,
                        members: p.members,
                        readers: if packed_requested {
                            packed::readers(g, p.output)
                        } else {
                            None
                        },
                    },
                )
            })
            .collect::<HashMap<_, _>>();
        let selected = packs.keys().copied().collect();
        let masks = crate::ptx::mixed::attention_training_forward::online_pair_forward_seeds(g)
            .into_iter()
            .map(|r| (r.output, r.mask))
            .collect::<HashMap<_, _>>();
        let regions = pair_recognize::recognize(g, &selected)
            .into_iter()
            .filter_map(|proof| {
                let mut reservations =
                    reservation_ancestors(g, proof.inputs.into_iter().chain(proof.outputs));
                if packed_requested {
                    for output in &proof.required_qkv_packs {
                        reservations.extend(reservation_ancestors(g, [*output]));
                    }
                }
                Some(Region {
                    mask: *masks.get(&proof.outputs[0])?,
                    reservations,
                    proof,
                })
            })
            .collect::<Vec<_>>();
        let mut needed = HashSet::new();
        for r in &regions {
            needed.extend(r.reservations.iter().copied());
            needed.extend(r.proof.quadratic);
            needed.insert(r.mask);
            needed.extend(r.proof.surviving_consumers.iter().map(|(_, n, _)| *n));
            for p in &r.proof.required_qkv_packs {
                if let Some(p) = packs.get(p) {
                    needed.insert(p.output);
                    needed.extend(p.inputs.iter().map(|i| i.source));
                    if packed_requested && let Some(readers) = &p.readers {
                        needed.extend(readers.numeric.iter().copied());
                        needed.extend(
                            readers
                                .identity_casts
                                .iter()
                                .flat_map(|p| [p.node, p.parent]),
                        );
                    }
                }
            }
        }
        // Constants, including the immutable mask shared across layers, remain
        // sources even though new kernels generate the proved causal mask.
        let never_prune = g
            .graph
            .node_indices()
            .filter(|&n| {
                matches!(
                    g[n],
                    TensorGraphNode::Constant { .. }
                        | TensorGraphNode::Parameter { .. }
                        | TensorGraphNode::Input { .. }
                        | TensorGraphNode::IndexSource { .. }
                )
            })
            .chain(regions.iter().map(|r| r.mask))
            .collect();
        Self {
            packed_requested,
            register_statistics_requested,
            #[cfg(test)]
            force_register_statistics_rejection,
            regions,
            needed,
            never_prune,
            values: HashMap::new(),
            groups: HashMap::new(),
            packs,
            aliases: storage_aliases(g),
            boundaries: HashMap::new(),
            origins: HashMap::new(),
            unknown_origins: HashSet::new(),
            f32_programs: 0,
        }
    }
    pub(in crate::ptx::mixed) fn extend_last_consumers(
        &self,
        last: &mut HashMap<NodeIndex, usize>,
    ) {
        for r in &self.regions {
            for &n in &r.reservations {
                last.entry(n)
                    .and_modify(|p| *p = (*p).max(r.proof.backward_position))
                    .or_insert(r.proof.backward_position);
            }
        }
    }
    /// A known constructor decoration must be published to the same retained
    /// instance. Verify the before group and exact suffix before changing any
    /// registry: never weaken matching or rediscover a producer by labels.
    pub(in crate::ptx::mixed) fn refresh_decorated_operation(
        &mut self,
        node: NodeIndex,
        groups: (&[Kernel], &[Kernel]),
        suffix: &str,
        output: &Value,
    ) -> Result<()> {
        if !self.needed.contains(&node) {
            return Ok(());
        }
        let before = groups.0.iter().map(stamp).collect::<Vec<_>>();
        let after = groups.1.iter().map(stamp).collect::<Vec<_>>();
        ensure!(
            exact_decoration(node, &before, &after, suffix),
            "decorated operation changed non-label identity for node {}",
            node.index()
        );
        let full = Some((0..groups.0.len()).collect::<Vec<_>>());
        let boundary = self
            .boundaries
            .get(&node)
            .context("decorated operation boundary was not registered")?;
        ensure!(
            group_indices(boundary, groups.0) == full,
            "decorated boundary does not match exact original group"
        );
        let origin = self
            .origins
            .get(&node)
            .context("decorated operation origin was not registered")?;
        ensure!(
            origin.key == StorageKey::Global(node)
                && range(origin.binding) == range(output.binding)
                && origin.binding.dtype == output.binding.dtype
                && group_indices(&origin.producer, groups.0) == full,
            "decorated operation origin changed"
        );
        if let Some(group) = self.groups.get(&node) {
            ensure!(
                group_indices(&group.kernels, groups.0) == full,
                "decorated contraction group changed"
            );
        }
        // The same modules/owners remain held by groups and the executor. No
        // fallible operation follows the first registry publication.
        if let Some(group) = self.groups.get_mut(&node) {
            group.kernels = groups.1.to_vec();
        }
        self.boundaries.insert(node, groups.1.to_vec());
        if let Some(origin) = self.origins.get_mut(&node) {
            origin.producer = groups.1.to_vec();
        }
        self.values.insert(node, output.clone());
        Ok(())
    }
    /// Existing selected split paths have no single retained Program. Capture
    /// their exact installed group for surviving boundaries/source provenance;
    /// absence of removable Program counters still rejects contractions.
    pub(in crate::ptx::mixed) fn record_emitted(
        &mut self,
        node: NodeIndex,
        kernels: &[Kernel],
        output: &Value,
        values: &HashMap<NodeIndex, Value>,
    ) {
        if self.needed.contains(&node) {
            self.boundaries.insert(node, kernels.to_vec());
            self.origins.insert(
                node,
                Origin {
                    key: StorageKey::Global(node),
                    binding: output.binding,
                    producer: kernels.to_vec(),
                },
            );
            self.values.insert(node, output.clone());
        }
        self.capture_values(values);
    }
    pub(in crate::ptx::mixed) fn record_operation<D: CudaDType>(
        &mut self,
        node: NodeIndex,
        p: &Program<D>,
        output: &Value,
        values: &HashMap<NodeIndex, Value>,
    ) {
        if self
            .regions
            .iter()
            .any(|r| r.proof.contractions.contains(&node))
        {
            self.groups.insert(
                node,
                Group {
                    kernels: p.kernels.clone(),
                    counts: Some(Counts::program(p)),
                },
            );
            self.values.insert(node, output.clone());
        }
        self.record_emitted(node, &p.kernels, output, values);
    }
    /// Borrow immutable typed metadata BEFORE LN consumes it. Numeric logical
    /// members may share one physical step; local IDs never become global IDs.
    pub(in crate::ptx::mixed) fn record_f32(
        &mut self,
        p: &Program<crate::ptx::F32>,
        mapped: &HashMap<NodeIndex, NodeIndex>,
        values: &HashMap<NodeIndex, Value>,
    ) -> Result<()> {
        let metadata = p
            .metadata
            .as_ref()
            .context("screen original F32 metadata missing")?;
        ensure!(
            metadata.kernel_steps.len() == p.kernels.len(),
            "screen F32 step/kernel mapping mismatch"
        );
        let members = metadata
            .plan
            .steps()
            .iter()
            .map(|step| step.members.clone())
            .collect::<Vec<_>>();
        let outputs = metadata
            .plan
            .steps()
            .iter()
            .map(|step| step.outputs.clone())
            .collect::<Vec<_>>();
        let program = self.f32_programs;
        self.f32_programs = self
            .f32_programs
            .checked_add(1)
            .context("screen F32 program count overflow")?;
        for (&original, &local) in mapped {
            if !self.needed.contains(&original) {
                continue;
            }
            if let Some(indices) = physical_member_kernels(local, &members, &metadata.kernel_steps)
            {
                self.boundaries.insert(
                    original,
                    indices.iter().map(|&i| p.kernels[i].clone()).collect(),
                );
            }
            let physical = p.views.get(&local).map_or(local, |v| v.source);
            if let (Some(indices), Some(&binding)) = (
                physical_member_kernels(physical, &outputs, &metadata.kernel_steps),
                p.values.get(&physical),
            ) {
                self.origins.insert(
                    original,
                    Origin {
                        key: StorageKey::F32 { program, physical },
                        binding,
                        producer: indices.iter().map(|&i| p.kernels[i].clone()).collect(),
                    },
                );
            } else if outputs.iter().enumerate().any(|(step, nodes)| {
                nodes.contains(&physical) && metadata.kernel_steps.contains(&step)
            }) {
                // A real owned producer exists but its mapping is ambiguous or
                // incomplete. Do not reinterpret this materialized view as an
                // ancestor alias merely because graph-level view syntax fits.
                self.origins.remove(&original);
                self.unknown_origins.insert(original);
            }
        }
        self.capture_values(values);
        Ok(())
    }
    fn origin(&self, mut node: NodeIndex) -> std::result::Result<&Origin, Rejection> {
        for _ in 0..=self.aliases.len() {
            if self.unknown_origins.contains(&node) {
                break;
            }
            if let Some(origin) = self.origins.get(&node) {
                return Ok(origin);
            }
            match self.aliases.get(&node) {
                Some(&parent) => node = parent,
                None => break,
            }
        }
        Err(Rejection::MissingSemanticOwner(
            node,
            "typed storage provenance",
        ))
    }
    fn boundary(
        &self,
        node: NodeIndex,
        kernels: &[Kernel],
    ) -> std::result::Result<Vec<usize>, Rejection> {
        let old = self
            .boundaries
            .get(&node)
            .or_else(|| self.groups.get(&node).map(|g| &g.kernels))
            .ok_or(Rejection::MissingSemanticOwner(
                node,
                "typed semantic boundary",
            ))?;
        group_indices(old, kernels).ok_or(Rejection::ChangedOriginalGroup)
    }
    fn capture_values(&mut self, values: &HashMap<NodeIndex, Value>) {
        for n in &self.needed {
            if let Some(v) = values.get(n) {
                self.values.insert(*n, v.clone());
            }
        }
    }
    /// Must run before every prefix/final reuse, including post-LN reuse.
    pub(in crate::ptx::mixed) fn capture_before_reuse(
        &mut self,
        values: &HashMap<NodeIndex, Value>,
        kernels: &[Kernel],
    ) {
        self.capture_values(values);
        let owners = self
            .regions
            .iter()
            .flat_map(|r| {
                [r.proof.quadratic[1], r.proof.quadratic[4]]
                    .into_iter()
                    .chain(r.proof.required_qkv_packs.iter().copied())
            })
            .collect::<HashSet<_>>();
        for n in owners {
            self.groups.entry(n).or_insert_with(|| Group {
                kernels: kernels
                    .iter()
                    .filter(|k| k.owner == Some(n))
                    .cloned()
                    .collect(),
                counts: None,
            });
            if self.groups[&n].kernels.is_empty() {
                self.groups.remove(&n);
            }
        }
    }
    fn prefix_target(&self, kernels: &[Kernel], owners: &[NodeIndex]) -> Option<usize> {
        let ends = owners
            .iter()
            .map(|owner| {
                self.groups
                    .get(owner)
                    .and_then(|g| group_indices(&g.kernels, kernels))
                    .and_then(|indices| indices.last().copied())
            })
            .collect::<Vec<_>>();
        target_from_positions(kernels.len(), &ends)
    }
    /// Read and WRITE reservations are explicit. Gradient writes are no earlier
    /// than their old births: both backward kernels are inserted at the latest
    /// old gradient boundary. Graph future-live ends cover not-yet-seen owners.
    pub(in crate::ptx::mixed) fn prospective_uses(&self, kernels: &[Kernel]) -> Vec<(usize, u64)> {
        let mut uses = Vec::new();
        for r in &self.regions {
            if let Some(back) = self.prefix_target(kernels, &r.proof.outputs[1..]) {
                for &n in &r.reservations {
                    if let Some(v) = self.values.get(&n) {
                        uses.push((back, v.binding.pointer));
                    }
                }
            }
            if let Some(fwd) = self.prefix_target(kernels, &[r.proof.outputs[0]])
                && let Some(v) = self.values.get(&r.proof.outputs[0])
            {
                uses.push((fwd, v.binding.pointer));
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
                remap_kernel(k, map);
            }
        }
        for group in self.boundaries.values_mut() {
            for k in group {
                remap_kernel(k, map);
            }
        }
        for origin in self.origins.values_mut() {
            origin.binding.pointer = *map
                .get(&origin.binding.pointer)
                .unwrap_or(&origin.binding.pointer);
            for k in &mut origin.producer {
                remap_kernel(k, map);
            }
        }
    }
    fn indices(
        &self,
        owner: NodeIndex,
        kernels: &[Kernel],
    ) -> std::result::Result<Vec<usize>, Rejection> {
        let old = self
            .groups
            .get(&owner)
            .ok_or(Rejection::MissingSemanticOwner(owner, "captured group"))?;
        let indices =
            group_indices(&old.kernels, kernels).ok_or(Rejection::ChangedOriginalGroup)?;
        if old.kernels.iter().any(|k| k.zero_before.is_some()) {
            return Err(Rejection::ChangedOriginalGroup);
        }
        Ok(indices)
    }
    fn value(&self, n: NodeIndex) -> std::result::Result<&Value, Rejection> {
        self.values.get(&n).ok_or(Rejection::MissingValues)
    }
    fn verify_pack(
        &self,
        n: NodeIndex,
        e: &MixedExecutor,
    ) -> std::result::Result<Vec<usize>, Rejection> {
        let p = self.packs.get(&n).ok_or(Rejection::UnverifiedPack)?;
        let indices = self.indices(n, &e.kernels)?;
        if indices.len() != 1 || p.inputs.len() != 3 {
            return Err(Rejection::UnverifiedPack);
        }
        let out = self.value(p.output)?;
        if !identity(out, &p.shape, DType::BF16) {
            return Err(Rejection::UnverifiedPack);
        }
        let mut pointers = Vec::new();
        for input in &p.inputs {
            let v = self.value(input.source)?;
            if !identity(v, &v.source_shape, DType::BF16) {
                return Err(Rejection::UnverifiedPack);
            }
            let mut view = v.view.clone();
            for op in &input.views {
                view = apply_view(view, op).ok_or(Rejection::UnverifiedPack)?;
            }
            if view.shape != [p.shape[0], p.shape[1], p.shape[2] / 3] {
                return Err(Rejection::UnverifiedPack);
            }
            pointers.push(v.binding.pointer);
        }
        pointers.push(out.binding.pointer);
        let k = &e.kernels[indices[0]];
        if k.pointers != pointers
            || k.config.block_dim != (256, 1, 1)
            || k.config.grid_dim
                != (
                    u32::try_from(out.binding.len)
                        .map_err(|_| Rejection::UnverifiedPack)?
                        .div_ceil(256),
                    1,
                    1,
                )
            || k.config.shared_mem_bytes != 0
        {
            return Err(Rejection::UnverifiedPack);
        }
        Ok(indices)
    }
    fn draft(&self, r: &Region, e: &MixedExecutor) -> std::result::Result<Draft, Rejection> {
        let p = &r.proof;
        let shape = Shape::new(p.shape[0], p.shape[1]).map_err(|_| Rejection::UnsupportedLayout)?;
        let input = p
            .inputs
            .map(|n| self.value(n).cloned())
            .into_iter()
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let outputs = p
            .outputs
            .map(|n| self.value(n).cloned())
            .into_iter()
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let requested = requested_transport(
            &input
                .iter()
                .chain(&outputs)
                .map(|v| v.binding)
                .collect::<Vec<_>>(),
        );
        let descriptors = input
            .iter()
            .map(descriptor)
            .collect::<Vec<_>>()
            .try_into()
            .map_err(|_| Rejection::UnsupportedLayout)?;
        let abi = match requested {
            leaf::Transport::Scalar => leaf::UncompiledPair::new(shape, descriptors),
            leaf::Transport::Async16 => {
                leaf::UncompiledPair::with_transport(shape, descriptors, requested)
            }
        }
        .map_err(|_| Rejection::UnsupportedLayout)?;
        let expected_payload = shape
            .elements()
            .checked_mul(8)
            .and_then(|n| shape.bh().checked_mul(512)?.checked_mul(12)?.checked_add(n))
            .ok_or(Rejection::UnsupportedLayout)?;
        if abi.shape != shape
            || abi.statistics_shape != [shape.bh(), 512, 3]
            || abi.global_payload_bytes != expected_payload
            || abi.transport != requested
            || p.forward_position >= p.backward_position
            || !p.outputs[1..].contains(&p.backward_boundary)
        {
            return Err(Rejection::UnsupportedLayout);
        }
        for (((v, expected), &n), proved_view) in outputs
            .iter()
            .zip(&abi.output_shapes)
            .zip(&p.outputs)
            .zip(&p.output_views)
        {
            if !identity(v, expected, DType::BF16) || v.view.source != n || v.view != *proved_view {
                return Err(Rejection::UnsupportedLayout);
            }
        }
        let input_bindings = input.iter().map(|v| v.binding).collect::<Vec<_>>();
        let output_bindings = outputs.iter().map(|v| v.binding).collect::<Vec<_>>();
        if !disjoint(&output_bindings, &input_bindings) {
            return Err(Rejection::AliasedBindings);
        }
        let qk = self.value(p.quadratic[0])?;
        let probability = self.value(p.quadratic[1])?;
        let p16 = self.value(p.quadratic[2])?;
        let dp = self.value(p.quadratic[3])?;
        let ds = self.value(p.quadratic[4])?;
        let mask = self.value(r.mask)?;
        for (v, dtype) in [
            (qk, DType::BF16),
            (probability, DType::F32),
            (p16, DType::BF16),
            (dp, DType::BF16),
            (ds, DType::BF16),
        ] {
            if !identity(v, &p.shape, dtype) {
                return Err(Rejection::UnsupportedLayout);
            }
        }
        if !identity(mask, &[1, 1, 512, 512], DType::F32) {
            return Err(Rejection::UnsupportedLayout);
        }
        let mut remove = HashSet::new();
        let mut counts = Vec::new();
        let mut contraction_indices = HashMap::new();
        for owner in p.contractions {
            let indices = self.indices(owner, &e.kernels)?;
            let count = self
                .groups
                .get(&owner)
                .and_then(|g| g.counts.clone())
                .ok_or(Rejection::UnsupportedOriginalGroups)?;
            if count.tensor != 1
                || e.kernels[*indices.last().ok_or(Rejection::ChangedOriginalGroup)?]
                    .pointers
                    .last()
                    != Some(&self.value(owner)?.binding.pointer)
            {
                return Err(Rejection::UnsupportedOriginalGroups);
            }
            remove.extend(indices.iter().copied());
            contraction_indices.insert(owner, indices);
            counts.push(count);
        }
        let norm = self.indices(p.quadratic[1], &e.kernels)?;
        let back = self.indices(p.quadratic[4], &e.kernels)?;
        if norm.len() != 1
            || back.len() != 1
            || e.kernels[norm[0]].pointers
                != [
                    qk.binding.pointer,
                    mask.binding.pointer,
                    probability.binding.pointer,
                    p16.binding.pointer,
                ]
            || e.kernels[back[0]].pointers
                != [
                    probability.binding.pointer,
                    dp.binding.pointer,
                    ds.binding.pointer,
                ]
        {
            return Err(Rejection::UnsupportedOriginalGroups);
        }
        remove.extend(norm);
        remove.extend(back);
        if e.kernels
            .iter()
            .enumerate()
            .any(|(i, k)| k.owner.is_some_and(|n| p.members.contains(&n)) && !remove.contains(&i))
        {
            return Err(Rejection::UnsupportedOriginalGroups);
        }
        let fwd = *contraction_indices[&p.outputs[0]]
            .last()
            .ok_or(Rejection::ChangedOriginalGroup)?;
        let bwd = p.outputs[1..]
            .iter()
            .filter_map(|n| {
                contraction_indices
                    .get(n)
                    .and_then(|indices| indices.last())
                    .copied()
            })
            .max()
            .ok_or(Rejection::MissingSemanticOwner(
                p.backward_boundary,
                "gradient boundary",
            ))?;
        if !remove.contains(&fwd) || !remove.contains(&bwd) || fwd >= bwd {
            return Err(Rejection::ChangedOriginalGroup);
        }
        for n in &p.required_qkv_packs {
            let indices = self.verify_pack(*n, e)?;
            if indices[0] <= bwd {
                return Err(Rejection::EarlySurvivingReader);
            }
        }
        // Every surviving numeric member resolves through its typed original
        // physical group, including fused multi-output F32 steps. Unknown or
        // removed/ambiguous groups cause whole-pair fallback.
        for &(output, consumer, _) in &p.surviving_consumers {
            let point = if output == p.outputs[0] { fwd } else { bwd };
            let indices = self.boundary(consumer, &e.kernels)?;
            if indices[0] <= point {
                return Err(Rejection::EarlySurvivingReader);
            }
        }
        for ((&node, value), point) in p.inputs.iter().zip(&input).zip([fwd, fwd, fwd, bwd]) {
            let origin = self.origin(node)?;
            if range(origin.binding) != range(value.binding)
                || origin.binding.dtype != value.binding.dtype
            {
                return Err(Rejection::UnsupportedLayout);
            }
            let indices = group_indices(&origin.producer, &e.kernels)
                .ok_or(Rejection::ChangedOriginalGroup)?;
            if *indices.last().ok_or(Rejection::ChangedOriginalGroup)? >= point {
                return Err(Rejection::EarlySurvivingReader);
            }
        }
        // Only these four reads and four writes change physical lifetime. All
        // graph/view/cast ancestors remain reserved in the global slot planner,
        // but a Value.view.source from a local F32 graph is never an owner ID.
        let mut reserved = Vec::new();
        for (&node, value) in p
            .inputs
            .iter()
            .zip(&input)
            .chain(p.outputs.iter().zip(&outputs))
        {
            let origin = self.origin(node)?;
            if range(origin.binding) != range(value.binding)
                || origin.binding.dtype != value.binding.dtype
            {
                return Err(Rejection::UnsupportedLayout);
            }
            let producer = group_indices(&origin.producer, &e.kernels)
                .ok_or(Rejection::ChangedOriginalGroup)?;
            let first = producer[0];
            let mut last = bwd;
            for &(output, consumer, _) in &p.surviving_consumers {
                if self.origin(output)?.key == origin.key {
                    let indices = self.boundary(consumer, &e.kernels)?;
                    last = last.max(*indices.last().ok_or(Rejection::ChangedOriginalGroup)?);
                }
            }
            if first > last {
                return Err(Rejection::ChangedOriginalGroup);
            }
            reserved.push(ReservedSpan {
                source: origin.key,
                binding: value.binding,
                first,
                last,
            });
        }
        if reserved.iter().enumerate().any(|(i, &a)| {
            reserved
                .iter()
                .skip(i + 1)
                .any(|&b| !compatible_spans(a, b))
        }) {
            return Err(Rejection::AliasedBindings);
        }
        let (functions, statistics_mode, statistics_rejection) = select_statistics_pair(
            self.register_statistics_requested,
            register_statistics_scope(shape, abi.read_layouts),
            |mode| {
                #[cfg(test)]
                if mode == forward::StatisticsStorage::Registers
                    && self.force_register_statistics_rejection
                {
                    return Err(Rejection::Resources);
                }
                let irs = pair_ir(shape, abi.read_layouts, abi.transport, mode)
                    .map_err(|_| Rejection::Compilation)?;
                compile_pair(e, irs)
            },
        )?;
        Ok(Draft {
            outputs: p.outputs,
            input: input_bindings
                .try_into()
                .map_err(|_| Rejection::MissingValues)?,
            output: output_bindings
                .try_into()
                .map_err(|_| Rejection::MissingValues)?,
            values: outputs,
            shape,
            transport: abi.transport,
            statistics_mode,
            statistics_rejection,
            functions,
            remove,
            fwd,
            bwd,
            counts,
            reserved,
            packed: None,
            prune: p
                .members
                .difference(&p.outputs.into_iter().collect())
                .copied()
                .filter(|n| !self.never_prune.contains(n))
                .collect(),
        })
    }
    /// The exclusive borrow spans all validation, compilation, allocation and
    /// commit. No caller may mutate bindings/info between those phases.
    pub(in crate::ptx::mixed) fn apply(self, e: &mut MixedExecutor) -> Result<CommittedScreen> {
        self.stage(e)?.commit(e)
    }
    fn stage(mut self, e: &MixedExecutor) -> Result<StagedScreen> {
        ensure!(
            !e.has_run && e.captured.is_none() && e.direct.is_none(),
            "numerical screen must precede all warmup/capture/direct initialization"
        );
        self.capture_before_reuse(&e.values, &e.kernels);
        let mut report = CommittedScreen {
            terminal_late_pass: true,
            ..Default::default()
        };
        let mut drafts = Vec::new();
        let mut remove = HashSet::new();
        let mut info = e.info.clone();
        for r in &self.regions {
            let mut next = info.clone();
            match self.draft(r, e) {
                Ok(mut draft) => {
                    if self.packed_requested {
                        match packed::prepare(&self, r, e, &draft) {
                            Ok(candidate) if next.qkv_grad_pack_regions > 0 => {
                                next.qkv_grad_pack_regions -= 1;
                                draft.functions.d_query_and_delta = candidate.d_query.clone();
                                draft.functions.d_key_value = candidate.d_key_value.clone();
                                draft.remove.extend(candidate.remove.iter().copied());
                                draft.reserved.push(candidate.reserved);
                                draft.prune.extend(
                                    candidate
                                        .members
                                        .iter()
                                        .copied()
                                        .filter(|n| *n != candidate.output),
                                );
                                draft.prune.extend(draft.outputs[1..].iter().copied());
                                draft.packed = Some(candidate);
                            }
                            Ok(_) => report.packed_fallbacks.push(PairFallback {
                                output: r.proof.outputs[0],
                                reason: Rejection::CounterInvariant,
                            }),
                            Err(reason) => report.packed_fallbacks.push(PairFallback {
                                output: r.proof.outputs[0],
                                reason,
                            }),
                        }
                    }
                    if drafts.iter().any(|old: &Draft| {
                        old.reserved
                            .iter()
                            .any(|&a| draft.reserved.iter().any(|&b| !compatible_spans(a, b)))
                    }) {
                        report.fallbacks.push(PairFallback {
                            output: r.proof.outputs[0],
                            reason: Rejection::AliasedBindings,
                        });
                        continue;
                    }
                    if draft.remove.iter().any(|n| remove.contains(n)) {
                        report.fallbacks.push(PairFallback {
                            output: r.proof.outputs[0],
                            reason: Rejection::OverlappingPair,
                        });
                        continue;
                    }
                    if draft.counts.iter().any(|c| c.debit(&mut next).is_none())
                        || next.attention_softmax_regions.checked_sub(1).is_none()
                        || next
                            .attention_softmax_backward_regions
                            .checked_sub(1)
                            .is_none()
                    {
                        report.fallbacks.push(PairFallback {
                            output: r.proof.outputs[0],
                            reason: Rejection::CounterInvariant,
                        });
                        continue;
                    }
                    next.attention_softmax_regions -= 1;
                    next.attention_softmax_backward_regions -= 1;
                    let bytes = draft
                        .shape
                        .bh()
                        .checked_mul(512)
                        .and_then(|n| n.checked_mul(12))
                        .context("online statistics bytes overflow")?;
                    next.f32_bytes = next
                        .f32_bytes
                        .checked_add(bytes)
                        .context("online owned F32 bytes overflow")?;
                    next.unpooled_bytes = next
                        .unpooled_bytes
                        .checked_add(bytes)
                        .context("online unpooled bytes overflow")?;
                    report.workspace_bytes = report
                        .workspace_bytes
                        .checked_add(bytes)
                        .context("online total workspace bytes overflow")?;
                    info = next;
                    remove.extend(draft.remove.iter().copied());
                    drafts.push(draft);
                }
                Err(reason) => {
                    if self.packed_requested {
                        report.packed_fallbacks.push(PairFallback {
                            output: r.proof.outputs[0],
                            reason: reason.clone(),
                        });
                    }
                    report.fallbacks.push(PairFallback {
                        output: r.proof.outputs[0],
                        reason,
                    });
                }
            }
        }
        if self.packed_requested {
            for fallback in &report.fallbacks {
                if !report
                    .packed_fallbacks
                    .iter()
                    .any(|old| old.output == fallback.output)
                {
                    report.packed_fallbacks.push(PairFallback {
                        output: fallback.output,
                        reason: fallback.reason.clone(),
                    });
                }
            }
        }
        // All counter arithmetic precedes allocation and all executor mutation.
        let added = drafts
            .len()
            .checked_mul(3)
            .context("online kernel count overflow")?;
        let final_len = e
            .kernels
            .len()
            .checked_sub(remove.len())
            .and_then(|n| n.checked_add(added))
            .context("online replacement length overflow")?;
        let mut statistics = Vec::new();
        for d in &drafts {
            let len = d
                .shape
                .bh()
                .checked_mul(512)
                .and_then(|n| n.checked_mul(3))
                .context("online workspace extent overflow")?;
            match e.stream.alloc_zeros::<f32>(len) {
                Ok(v) => statistics.push(v),
                Err(err) => {
                    e.stream.synchronize()?;
                    return Err(err.into());
                }
            }
        }
        // Private zero-initialization has completed before any staged owner can
        // be dropped by a subsequent error. No model function has executed.
        e.stream.synchronize()?;
        let mut inserts = BTreeMap::<usize, Vec<Kernel>>::new();
        let mut values = e.values.clone();
        let mut prune = HashSet::new();
        let mut retired_raw = HashSet::new();
        for (d, s) in drafts.into_iter().zip(&statistics) {
            let stats = Binding {
                pointer: s.device_ptr(&e.stream).0,
                len: s.len(),
                dtype: DType::F32,
            };
            ensure!(
                disjoint(
                    &[stats],
                    &d.input.iter().chain(&d.output).copied().collect::<Vec<_>>()
                ),
                "new statistics overlaps existing binding"
            );
            let [q, k, v, incoming] = d.input.map(|b| b.pointer);
            let [o, dq, dkt, dv] = d.output.map(|b| b.pointer);
            let make =
                |function: CudaFunction, owner: NodeIndex, pointers: Vec<u64>, role: &str| Kernel {
                    owner: Some(owner),
                    label: format!(
                        "online attention NUMERICAL SCREEN {role} outputs={:?} transport={:?}",
                        d.outputs, d.transport
                    ) + if d.statistics_mode == forward::StatisticsStorage::Registers {
                        " statistics=Registers"
                    } else {
                        ""
                    },
                    function,
                    config: LaunchConfig {
                        grid_dim: (8, d.shape.bh() as u32, 1),
                        block_dim: (32, 8, 1),
                        shared_mem_bytes: 0,
                    },
                    pointers,
                    zero_before: None,
                };
            inserts.entry(d.fwd).or_default().push(make(
                d.functions.forward,
                d.outputs[0],
                vec![q, k, v, o, stats.pointer],
                "forward",
            ));
            let (dq_owner, dkv_owner, dq_target, dkv_targets, roles) =
                if let Some(packed) = &d.packed {
                    ensure!(
                        disjoint(&[stats], &[packed.value.binding]),
                        "new statistics overlaps packed target"
                    );
                    report.packed_outputs.push(packed.output);
                    report.packed_pack_kernels_removed = report
                        .packed_pack_kernels_removed
                        .checked_add(packed.remove.len())
                        .context("packed removed kernel count overflow")?;
                    retired_raw.extend(d.outputs[1..].iter().copied());
                    values.insert(packed.output, packed.value.clone());
                    report.retained_outputs.push([
                        d.outputs[0],
                        packed.output,
                        packed.output,
                        packed.output,
                    ]);
                    report
                        .joint_logical_outputs
                        .push([packed.output, packed.output]);
                    (
                        packed.output,
                        packed.output,
                        packed.value.binding.pointer,
                        vec![packed.value.binding.pointer],
                        ["packed Q+delta", "packed K+V"],
                    )
                } else {
                    report.retained_outputs.push(d.outputs);
                    report
                        .joint_logical_outputs
                        .push([d.outputs[2], d.outputs[3]]);
                    (
                        d.outputs[1],
                        d.outputs[2],
                        dq,
                        vec![dkt, dv],
                        ["dQ+delta", "dKT+dV"],
                    )
                };
            inserts.entry(d.bwd).or_default().extend([
                make(
                    d.functions.d_query_and_delta,
                    dq_owner,
                    vec![q, k, v, o, incoming, stats.pointer, dq_target],
                    roles[0],
                ),
                make(
                    d.functions.d_key_value,
                    dkv_owner,
                    [vec![q, k, v, incoming, stats.pointer], dkv_targets].concat(),
                    roles[1],
                ),
            ]);
            values.extend(d.outputs.into_iter().zip(d.values));
            prune.extend(d.prune);
            report.selected_pairs += 1;
            match d.statistics_mode {
                forward::StatisticsStorage::Registers => report.register_statistics_pairs += 1,
                forward::StatisticsStorage::Global => report.global_statistics_pairs += 1,
            }
            if let Some(reason) = d.statistics_rejection {
                report.register_statistics_fallbacks.push(PairFallback {
                    output: d.outputs[0],
                    reason,
                });
            }

            match d.transport {
                leaf::Transport::Scalar => report.scalar_pairs += 1,
                leaf::Transport::Async16 => report.async16_pairs += 1,
            }
        }
        let mut protected = self.never_prune.clone();
        for region in &self.regions {
            protected.extend(
                region
                    .proof
                    .outputs
                    .into_iter()
                    .filter(|n| !retired_raw.contains(n)),
            );
            protected.insert(region.mask);
        }
        let discarded = discarded_global_alias_keys(&prune, &self.aliases, &protected);
        // F32 Value.view.source may be a LOCAL original-program node ID. Only
        // original GLOBAL graph alias edges may decide which map keys die.
        #[cfg(test)]
        let legacy_metadata = values
            .iter()
            .filter(|(n, v)| !prune.contains(n) && !prune.contains(&v.view.source))
            .map(|(&n, v)| {
                (
                    n,
                    (
                        v.binding.pointer,
                        v.binding.len,
                        v.binding.dtype,
                        v.view.clone(),
                        v.source_shape.clone(),
                    ),
                )
            })
            .collect::<HashMap<_, _>>();
        values.retain(|n, _| !discarded.contains(n));
        #[cfg(test)]
        if let Some(directory) = std::env::var_os("TNSR_ONLINE_SCREEN_AUDIT") {
            let current_metadata = values
                .iter()
                .map(|(&n, v)| {
                    (
                        n,
                        (
                            v.binding.pointer,
                            v.binding.len,
                            v.binding.dtype,
                            v.view.clone(),
                            v.source_shape.clone(),
                        ),
                    )
                })
                .collect::<HashMap<_, _>>();
            let mut legacy_keys = legacy_metadata
                .keys()
                .map(|n| n.index())
                .collect::<Vec<_>>();
            legacy_keys.sort_unstable();
            let mut current_keys = current_metadata
                .keys()
                .map(|n| n.index())
                .collect::<Vec<_>>();
            current_keys.sort_unstable();
            let directory = std::path::PathBuf::from(directory);
            std::fs::create_dir_all(&directory)?;
            std::fs::write(
                directory.join(format!(
                    "global-prune-{}-pairs-{}-values.json",
                    report.selected_pairs,
                    values.len()
                )),
                serde_json::to_vec_pretty(&serde_json::json!({
                    "scope":"Test-only comparison; invalid local-ID interpretation is NOT used to drive execution. Kernels were constructed before pruning and no buffers are freed.",
                    "kept_binding_metadata_identical":legacy_metadata == current_metadata,
                    "legacy_kept_global_keys":legacy_keys,"corrected_kept_global_keys":current_keys,
                    "inserted_kernel_count":added,"retained_pair_outputs":report.retained_outputs.iter().map(|ns|ns.map(|n|n.index())).collect::<Vec<_>>()
                }))?,
            )?;
        }
        let mut replacement = Vec::with_capacity(final_len);
        for (i, k) in e.kernels.iter().enumerate() {
            if let Some(new) = inserts.remove(&i) {
                replacement.extend(new);
            }
            if !remove.contains(&i) {
                replacement.push(k.clone());
            }
        }
        ensure!(
            replacement.len() == final_len && inserts.is_empty(),
            "online insertion transaction mismatch"
        );
        info.kernels = final_len;
        ensure!(
            report.terminal_late_pass
                && report.retained_outputs.len() == report.selected_pairs
                && report.joint_logical_outputs.len() == report.selected_pairs
                && report.async16_pairs.checked_add(report.scalar_pairs)
                    == Some(report.selected_pairs),
            "online numerical screen report mismatch"
        );
        Ok(StagedScreen {
            old: e.kernels.iter().map(stamp).collect(),
            replacement,
            values,
            info,
            statistics,
            report,
        })
    }
}
#[derive(Clone, Copy)]
struct ReservedSpan {
    source: StorageKey,
    binding: Binding,
    first: usize,
    last: usize,
}
fn compatible_spans(a: ReservedSpan, b: ReservedSpan) -> bool {
    let (Some((a0, a1)), Some((b0, b1))) = (range(a.binding), range(b.binding)) else {
        return false;
    };
    if a.source == b.source {
        return a0 == b0 && a1 == b1 && a.binding.dtype == b.binding.dtype;
    }
    a.last < b.first || b.last < a.first || a1 <= b0 || b1 <= a0
}
struct Draft {
    outputs: [NodeIndex; 4],
    input: [Binding; 4],
    output: [Binding; 4],
    values: Vec<Value>,
    shape: Shape,
    transport: leaf::Transport,
    statistics_mode: forward::StatisticsStorage,
    statistics_rejection: Option<Rejection>,
    functions: PairedFunctions<CudaFunction>,
    remove: HashSet<usize>,
    fwd: usize,
    bwd: usize,
    counts: Vec<Counts>,
    prune: HashSet<NodeIndex>,
    reserved: Vec<ReservedSpan>,
    packed: Option<packed::Packed>,
}
#[derive(Clone, PartialEq, Eq)]
struct Stamp {
    function: Option<String>,
    owner: Option<NodeIndex>,
    label: String,
    pointers: Vec<u64>,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    shared: u32,
    zero: Option<(u64, usize, DType)>,
}
fn exact_decoration(node: NodeIndex, before: &[Stamp], after: &[Stamp], suffix: &str) -> bool {
    !before.is_empty()
        && before.len() == after.len()
        && !suffix.is_empty()
        && before.iter().zip(after).all(|(old, new)| {
            if old.function.is_none() || old.owner != Some(node) {
                return false;
            }
            let mut expected = old.clone();
            expected.label.push_str(suffix);
            expected == *new
        })
}
fn stamp(k: &Kernel) -> Stamp {
    Stamp {
        function: crate::ptx::mixed::nt128_bias::ldmatrix::function_identity(&k.function).ok(),
        owner: k.owner,
        label: k.label.clone(),
        pointers: k.pointers.clone(),
        grid: k.config.grid_dim,
        block: k.config.block_dim,
        shared: k.config.shared_mem_bytes,
        zero: k.zero_before.map(|b| (b.pointer, b.len, b.dtype)),
    }
}
struct StagedScreen {
    old: Vec<Stamp>,
    replacement: Vec<Kernel>,
    values: HashMap<NodeIndex, Value>,
    info: MixedInfo,
    statistics: Vec<CudaSlice<f32>>,
    report: CommittedScreen,
}
impl StagedScreen {
    fn commit(self, e: &mut MixedExecutor) -> Result<CommittedScreen> {
        ensure!(
            e.kernels.iter().map(stamp).eq(self.old.iter().cloned()),
            "executor changed after numerical-screen preflight"
        );
        ensure!(
            !e.has_run && e.captured.is_none() && e.direct.is_none(),
            "late numerical-screen commit"
        );
        e.buffers
            .try_reserve(self.statistics.len())
            .context("reserve online owner slots")?;
        // No fallible operations after the first publication. Original programs
        // and every old quadratic allocation remain owned; caller skips reuse.
        e.kernels = self.replacement;
        e.values = self.values;
        e.info = self.info;
        e.buffers
            .extend(self.statistics.into_iter().map(Buffer::F32));
        Ok(self.report)
    }
}
fn descriptor(v: &Value) -> Descriptor {
    Descriptor {
        pointer: v.binding.pointer,
        storage_len: v.binding.len,
        storage_dtype: v.binding.dtype,
        source_shape: v.source_shape.clone(),
        view: v.view.clone(),
    }
}
fn identity(v: &Value, shape: &[usize], dtype: DType) -> bool {
    shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .is_some_and(|n| n == v.binding.len)
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
    let (Some(o), Some(i)) = (
        outputs
            .iter()
            .copied()
            .map(range)
            .collect::<Option<Vec<_>>>(),
        inputs
            .iter()
            .copied()
            .map(range)
            .collect::<Option<Vec<_>>>(),
    ) else {
        return false;
    };
    o.iter().enumerate().all(|(n, &(a, b))| {
        o.iter()
            .skip(n + 1)
            .chain(i.iter())
            .all(|&(c, d)| b <= c || d <= a)
    })
}
fn apply_view(v: VirtualTensor, n: &TensorGraphNode<f32>) -> Option<VirtualTensor> {
    match n {
        TensorGraphNode::Reshape { shape } | TensorGraphNode::Flatten { shape } => {
            v.reshape(shape.clone()).ok()
        }
        TensorGraphNode::Permute { axes, .. } => v.permute(axes).ok(),
        TensorGraphNode::Transpose { .. } => {
            let mut a = (0..v.shape.len()).collect::<Vec<_>>();
            if a.len() < 2 {
                return None;
            }
            let n = a.len();
            a.swap(n - 2, n - 1);
            v.permute(&a).ok()
        }
        _ => None,
    }
}
fn compile(e: &MixedExecutor, ir: TileIR) -> Result<CudaFunction> {
    let mut t = TileGraph::from_graph(&TensorGraph::<f32>::new());
    t.region_kernels
        .push(crate::tile::graph::TileRegionKernel { region_id: 0, ir });
    t.optimize_indices()?;
    let g = PtxGraph::from(t).with_target(PtxTarget::from_context(e.stream.context())?);
    let m = e.stream.context().load_module(Ptx::from_src(g.to_ptx()))?;
    Ok(m.load_function(g.region_kernel_name(0).context("online function missing")?)?)
}
fn resources(e: &MixedExecutor, f: &CudaFunction) -> Result<bool> {
    Ok((1..=128).contains(&f.num_regs()?)
        && f.local_size_bytes()? == 0
        && f.shared_size_bytes()? == 49152
        && PtxTarget::from_context(e.stream.context())?
            .matmul_resources
            .shared_bytes_per_block
            >= 49152
        && f.max_threads_per_block()? >= 256
        && f.occupancy_max_active_blocks_per_multiprocessor(256, 0, None)? >= 2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_backward_owners_reserve_through_every_prefix_end() {
        let dq = NodeIndex::new(1);
        let dkt = NodeIndex::new(2);
        let dv = NodeIndex::new(3);
        let unrelated = NodeIndex::new(4);
        let required = [dq, dkt, dv];
        let emitted = [
            Some(dq),
            Some(unrelated),
            None,
            Some(dkt),
            Some(unrelated),
            Some(dv),
            None,
        ];
        assert_eq!(target_from_owners(&[], &required), None);
        for len in 1..6 {
            assert_eq!(
                target_from_owners(&emitted[..len], &required),
                Some(len - 1)
            );
        }
        assert_eq!(target_from_owners(&emitted[..6], &required), Some(5));
        assert_eq!(target_from_owners(&emitted[..7], &required), Some(5));
        // Owner duplication is a multi-kernel original Program, not completion
        // at its first kernel.
        let complete = [Some(dq), Some(dkt), Some(dv), Some(dkt), None];
        assert_eq!(target_from_owners(&complete, &required), Some(3));
    }
    #[test]
    fn cross_pair_ranges_allow_only_disjoint_lifetimes_or_true_aliases() {
        let span = |source, pointer, len, first, last| ReservedSpan {
            source: StorageKey::Global(NodeIndex::new(source)),
            binding: Binding {
                pointer,
                len,
                dtype: DType::BF16,
            },
            first,
            last,
        };
        let first = span(1, 128, 32, 1, 8);
        assert!(compatible_spans(first, span(1, 128, 32, 4, 12)));
        assert!(!compatible_spans(first, span(1, 128, 16, 4, 12)));
        assert!(!compatible_spans(first, span(2, 128, 32, 4, 12)));
        assert!(!compatible_spans(first, span(2, 160, 32, 8, 12)));
        assert!(compatible_spans(first, span(2, 160, 32, 9, 12)));
        assert!(compatible_spans(first, span(2, 192, 32, 4, 12)));
        // Delaying the first pair's new read to 10 invalidates the formerly
        // legal second pair's birth at 9; original-birth-only proof is unsound.
        assert!(!compatible_spans(
            span(1, 128, 32, 1, 10),
            span(2, 128, 32, 9, 12)
        ));
    }
    #[test]
    fn typed_numeric_members_share_one_step_and_reject_ambiguous_membership() {
        let a = NodeIndex::new(7);
        let b = NodeIndex::new(8);
        let missing = NodeIndex::new(9);
        let steps = [vec![], vec![a, b], vec![]];
        assert_eq!(
            physical_member_kernels(a, &steps, &[1, 1, 2]),
            Some(vec![0, 1])
        );
        assert_eq!(
            physical_member_kernels(b, &steps, &[1, 1, 2]),
            Some(vec![0, 1])
        );
        assert_eq!(physical_member_kernels(missing, &steps, &[1, 1, 2]), None);
        assert_eq!(
            physical_member_kernels(a, &[vec![a], vec![a]], &[0, 1]),
            None
        );
        // Upload-only membership does not invent a runnable producer.
        assert_eq!(physical_member_kernels(a, &[vec![a], vec![b]], &[1]), None);
    }
    #[test]
    fn post_transpose_decoration_refresh_requires_same_complete_retained_instance() {
        let node = NodeIndex::new(7);
        let old = Stamp {
            function: Some("strongly retained function handle".into()),
            owner: Some(node),
            label: "node=7 MatMul original".into(),
            pointers: vec![128, 256, 384],
            grid: (8, 24, 1),
            block: (32, 8, 1),
            shared: 0,
            zero: None,
        };
        let suffix = " transposed-physical-output logical=[512, 1536]";
        let mut decorated = old.clone();
        decorated.label.push_str(suffix);
        assert!(
            unique_stamp_indices(std::slice::from_ref(&old), std::slice::from_ref(&decorated))
                .is_none()
        );
        assert!(exact_decoration(
            node,
            std::slice::from_ref(&old),
            std::slice::from_ref(&decorated),
            suffix
        ));
        let mut negatives = Vec::new();
        let mut k = decorated.clone();
        k.function = Some("different function".into());
        negatives.push(k);
        let mut k = decorated.clone();
        k.owner = Some(NodeIndex::new(8));
        negatives.push(k);
        let mut k = decorated.clone();
        k.pointers[0] += 128;
        negatives.push(k);
        let mut k = decorated.clone();
        k.grid.0 += 1;
        negatives.push(k);
        let mut k = decorated.clone();
        k.block.1 = 4;
        negatives.push(k);
        let mut k = decorated.clone();
        k.shared = 128;
        negatives.push(k);
        let mut k = decorated.clone();
        k.zero = Some((384, 32, DType::BF16));
        negatives.push(k);
        let mut k = decorated.clone();
        k.label.push_str(" unexpected mutation");
        negatives.push(k);
        let mut k = decorated.clone();
        k.function = None;
        negatives.push(k);
        for new in negatives {
            assert!(!exact_decoration(
                node,
                std::slice::from_ref(&old),
                &[new],
                suffix
            ));
        }
        assert!(!exact_decoration(node, &[], &[], suffix));
        assert!(!exact_decoration(
            node,
            std::slice::from_ref(&old),
            std::slice::from_ref(&decorated),
            ""
        ));
        // Once the known exact decoration has been registered, full normal
        // equality is required again; the suffix is never ignored by matching.
        assert_eq!(
            unique_stamp_indices(
                std::slice::from_ref(&decorated),
                std::slice::from_ref(&decorated)
            ),
            Some(vec![0])
        );
    }
    #[test]
    fn typed_boundary_stamp_matches_are_unique_and_pointer_remaps_include_zeroing() {
        let stamp = |pointer| Stamp {
            function: Some("retained-function".into()),
            owner: Some(NodeIndex::new(1)),
            label: "local physical step".into(),
            pointers: vec![pointer],
            grid: (1, 1, 1),
            block: (256, 1, 1),
            shared: 0,
            zero: None,
        };
        let a = stamp(128);
        let b = stamp(256);
        let mut different_function = a.clone();
        different_function.function = Some("different retained handle".into());
        assert!(unique_stamp_indices(std::slice::from_ref(&a), &[different_function]).is_none());
        let mut unknown_function = a.clone();
        unknown_function.function = None;
        assert!(
            unique_stamp_indices(
                std::slice::from_ref(&unknown_function),
                std::slice::from_ref(&unknown_function)
            )
            .is_none()
        );
        assert_eq!(
            unique_stamp_indices(std::slice::from_ref(&a), &[b.clone(), a.clone()]),
            Some(vec![1])
        );
        assert_eq!(
            unique_stamp_indices(std::slice::from_ref(&a), std::slice::from_ref(&b)),
            None
        );
        assert_eq!(
            unique_stamp_indices(std::slice::from_ref(&a), &[a.clone(), a.clone()]),
            None
        );
        let mut pointers = vec![128, 256];
        let mut zero = Some(Binding {
            pointer: 256,
            len: 8,
            dtype: DType::F32,
        });
        remap_pointer_uses(
            &mut pointers,
            &mut zero,
            &HashMap::from([(128, 1024), (256, 2048)]),
        );
        assert_eq!(pointers, [1024, 2048]);
        let zero = zero.unwrap();
        assert_eq!((zero.pointer, zero.len, zero.dtype), (2048, 8, DType::F32));
        assert_eq!(unique_stamp_indices(&[stamp(1024)], &[stamp(128)]), None);
        assert_eq!(
            unique_stamp_indices(&[stamp(1024)], &[stamp(1024)]),
            Some(vec![0])
        );
        // Equal LOCAL node IDs do not identify an allocation across programs.
        assert!(
            StorageKey::F32 {
                program: 1,
                physical: NodeIndex::new(3)
            } != StorageKey::F32 {
                program: 2,
                physical: NodeIndex::new(3)
            }
        );
    }
    #[test]
    fn async_pair_request_requires_all_eight_aligned_bf16_bases() {
        let bases = (0..8)
            .map(|i| Binding {
                pointer: 128 + i * 128,
                len: 64,
                dtype: DType::BF16,
            })
            .collect::<Vec<_>>();
        assert_eq!(requested_transport(&bases), leaf::Transport::Async16);
        for i in 0..8 {
            let mut changed = bases.clone();
            changed[i].pointer += 2;
            assert_eq!(
                requested_transport(&changed),
                leaf::Transport::Scalar,
                "unaligned operand/output {i}"
            );
        }
        assert_eq!(requested_transport(&bases[..7]), leaf::Transport::Scalar);
        let mut wrong = bases;
        wrong[7].dtype = DType::F32;
        assert_eq!(requested_transport(&wrong), leaf::Transport::Scalar);
    }
    #[test]
    fn global_pruning_preserves_local_id_collisions_and_protected_sources() {
        let n = NodeIndex::new;
        let aliases = HashMap::from([
            (n(2), n(1)),
            (n(3), n(2)),
            (n(4), n(3)),
            (n(5), n(4)),
            (n(9), n(8)),
            (n(10), n(9)),
        ]);
        let discarded = discarded_global_alias_keys(
            &HashSet::from([n(1), n(8)]),
            &aliases,
            &HashSet::from([n(4), n(8)]),
        );
        assert_eq!(discarded, HashSet::from([n(1), n(2), n(3)]));
        // Global100 has an unrelated local view.source1. That numerical
        // coincidence must never remove global100 from executor.values.
        let values = [
            (n(1), n(0)),
            (n(2), n(1)),
            (n(100), n(1)),
            (n(4), n(3)),
            (n(8), n(1)),
            (n(9), n(8)),
            (n(10), n(9)),
        ];
        let kept = values
            .into_iter()
            .filter_map(|(key, _local_source)| (!discarded.contains(&key)).then_some(key))
            .collect::<HashSet<_>>();
        assert_eq!(kept, HashSet::from([n(100), n(4), n(8), n(9), n(10)]));
    }
    #[test]
    fn empty_preparation_preserves_every_source_and_reports_no_activation() {
        let graph = TensorGraph::<f32>::new();
        let mut preparation = Preparation::new(&graph);
        let mut last = HashMap::new();
        preparation.extend_last_consumers(&mut last);
        preparation.capture_before_reuse(&HashMap::new(), &[]);
        preparation.remap(&HashMap::new());
        assert!(preparation.prospective_uses(&[]).is_empty());
        assert!(last.is_empty());
        assert!(preparation.regions.is_empty());
        // Compile the constructor-only API without creating a CUDA context or
        // executing a model. Runtime activation is a separate qualification.
        let _apply: fn(Preparation, &mut MixedExecutor) -> Result<CommittedScreen> =
            Preparation::apply;
        let _record = Preparation::record_operation::<crate::ptx::F32>;
        let report = CommittedScreen::default();
        assert_eq!(report.selected_pairs, 0);
        assert_eq!(report.async16_pairs + report.scalar_pairs, 0);
        assert!(report.fallbacks.is_empty());
        assert_eq!(report.workspace_bytes, 0);
        assert!(report.retained_outputs.is_empty());
        assert!(report.joint_logical_outputs.is_empty());
        assert!(!report.terminal_late_pass);
        let rejection = PairFallback {
            output: NodeIndex::new(0),
            reason: Rejection::MissingValues,
        };
        assert_eq!(rejection.output, NodeIndex::new(0));
        assert!(matches!(rejection.reason, Rejection::MissingValues));
    }
}

#[cfg(test)]
mod gpu_tests;

#[cfg(test)]
mod register_statistics_tests;
