//! Exact late contraction of ONE closed QKV pack into its complete online pair.
//! Prospective output writes are reserved before any pool reuse. Owners remain
//! in the original allocation pool; this pass never reclaims raw checkpoints.
use super::*;
use crate::ptx::{
    linear::program::binding,
    mixed::qkv_grad_pack::{Input, QkvGradPack},
};
use petgraph::Direction;

pub(super) struct Packed {
    pub output: NodeIndex,
    pub value: Value,
    pub remove: Vec<usize>,
    pub d_query: CudaFunction,
    pub d_key_value: CudaFunction,
    pub reserved: ReservedSpan,
    pub members: HashSet<NodeIndex>,
}

#[derive(Clone)]
pub(super) struct IdentityCast {
    pub node: NodeIndex,
    pub parent: NodeIndex,
    pub expected: VirtualTensor,
    pub chain: Vec<TensorGraphNode<f32>>,
}
#[derive(Clone)]
pub(super) struct Readers {
    pub numeric: Vec<NodeIndex>,
    pub identity_casts: Vec<IdentityCast>,
}

/// Only full-volume BF16 view operations and SAME-BF16 identity casts are
/// metadata. A cast records the exact root-derived view/chain for a subsequent
/// live Value proof. F32 promotions, views, and operations remain boundaries.
pub(super) fn readers<G>(g: &TensorGraph<f32, G>, output: NodeIndex) -> Option<Readers> {
    let dtypes = crate::graph::mixed::value_dtypes(g);
    let volume = |shape: &[usize]| shape.iter().try_fold(1_usize, |n, &d| n.checked_mul(d));
    let root = VirtualTensor::identity(output, g[output].shape().clone(), DType::BF16);
    let mut pending = vec![(output, root, Vec::<TensorGraphNode<f32>>::new())];
    let mut visited = HashSet::new();
    let mut result = HashSet::new();
    let mut identity_casts = Vec::new();
    while let Some((node, expected, chain)) = pending.pop() {
        if !visited.insert(node) {
            continue;
        }
        let next = g
            .graph
            .neighbors_directed(node, Direction::Outgoing)
            .collect::<Vec<_>>();
        if next.is_empty() {
            return None;
        }
        for child in next {
            let whole = volume(g[node].shape()).is_some()
                && volume(g[node].shape()) == volume(g[child].shape());
            let one_input = g.inputs(child).as_slice() == [node];
            let both_bf16 = dtypes.get(&node) == Some(&crate::tensor::ValueDType::BF16)
                && dtypes.get(&child) == Some(&crate::tensor::ValueDType::BF16);
            let identity_cast = one_input
                && whole
                && both_bf16
                && g[node].shape() == g[child].shape()
                && matches!(
                    g[child],
                    TensorGraphNode::Cast {
                        dtype: crate::tensor::ValueDType::BF16,
                        ..
                    }
                );
            let metadata = one_input
                && whole
                && both_bf16
                && matches!(
                    g[child],
                    TensorGraphNode::Reshape { .. }
                        | TensorGraphNode::Flatten { .. }
                        | TensorGraphNode::Transpose { .. }
                        | TensorGraphNode::Permute { .. }
                );
            if identity_cast {
                identity_casts.push(IdentityCast {
                    node: child,
                    parent: node,
                    expected: expected.clone(),
                    chain: chain.clone(),
                });
                pending.push((child, expected.clone(), chain.clone()));
            } else if metadata {
                let view = apply_view(expected.clone(), &g[child])?;
                if view.shape != *g[child].shape()
                    || view.dtype != DType::BF16
                    || view.predicate.is_some()
                    || view.out_of_bounds.is_some()
                {
                    return None;
                }
                let mut next_chain = chain.clone();
                next_chain.push(g[child].clone());
                pending.push((child, view, next_chain));
            } else {
                result.insert(child);
            }
        }
    }
    let mut numeric = result.into_iter().collect::<Vec<_>>();
    numeric.sort_unstable();
    identity_casts.sort_by_key(|p| p.node);
    Some(Readers {
        numeric,
        identity_casts,
    })
}

fn extent(b: Binding) -> std::result::Result<(u64, u64), Rejection> {
    range(b).ok_or(Rejection::AliasedBindings)
}
fn owner(e: &MixedExecutor, required: Binding) -> std::result::Result<Binding, Rejection> {
    owner_from_bindings(
        required,
        &e.buffers
            .iter()
            .map(|buffer| binding(buffer, &e.stream))
            .collect::<Vec<_>>(),
    )
}
fn owner_from_bindings(
    required: Binding,
    owners: &[Binding],
) -> std::result::Result<Binding, Rejection> {
    let (lo, hi) = extent(required)?;
    let matches = owners
        .iter()
        .filter_map(|&candidate| {
            let (a, z) = range(candidate)?;
            (candidate.dtype == required.dtype && a == lo && hi <= z).then_some(candidate)
        })
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [one] => Ok(*one),
        _ => Err(Rejection::AliasedBindings),
    }
}
fn same_logical_view(a: &VirtualTensor, b: &VirtualTensor) -> bool {
    a.source == b.source
        && a.dtype == b.dtype
        && a.shape == b.shape
        && a.predicate.is_none()
        && a.out_of_bounds.is_none()
        && a.access.normalize_in_domain(&a.shape).ok()
            == b.access.normalize_in_domain(&b.shape).ok()
        && a.access.normalize_in_domain(&a.shape).is_ok()
}
fn expected_view(
    node: NodeIndex,
    shape: Shape,
    window: usize,
) -> std::result::Result<VirtualTensor, Rejection> {
    let raw_shape = if window == 1 {
        vec![shape.batch, 8, 64, 512]
    } else {
        vec![shape.batch, 8, 512, 64]
    };
    let mut v = VirtualTensor::identity(node, raw_shape, DType::BF16);
    if window == 1 {
        v = v
            .permute(&[0, 1, 3, 2])
            .map_err(|_| Rejection::UnsupportedLayout)?;
    }
    v.permute(&[0, 2, 1, 3])
        .and_then(|v| v.reshape(vec![shape.batch, 512, 512]))
        .map_err(|_| Rejection::UnsupportedLayout)
}
fn same_chain(a: &[TensorGraphNode<f32>], b: &[TensorGraphNode<f32>]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| match (a, b) {
        (TensorGraphNode::Reshape { shape: a }, TensorGraphNode::Reshape { shape: b })
        | (TensorGraphNode::Flatten { shape: a }, TensorGraphNode::Flatten { shape: b })
        | (TensorGraphNode::Transpose { shape: a }, TensorGraphNode::Transpose { shape: b }) => {
            a == b
        }
        (
            TensorGraphNode::Permute { shape: a, axes: aa },
            TensorGraphNode::Permute { shape: b, axes: bb },
        ) => a == b && aa == bb,
        _ => false,
    })
}
fn physical_identity_cast(
    output: NodeIndex,
    shape: &[usize],
    root: &Value,
    proof: &IdentityCast,
    pair: (&Value, &Value),
) -> bool {
    identity(root, shape, DType::BF16)
        && root.view.source == output
        && proof.expected.source == output
        && proof.expected.dtype == DType::BF16
        && shape.iter().try_fold(1_usize, |n, &d| n.checked_mul(d))
            == proof
                .expected
                .shape
                .iter()
                .try_fold(1_usize, |n, &d| n.checked_mul(d))
        && [pair.0, pair.1].iter().all(|value| {
            value.binding.dtype == DType::BF16
                && value.source_shape == shape
                && range(value.binding) == range(root.binding)
                && same_logical_view(&value.view, &proof.expected)
                && same_chain(&value.chain, &proof.chain)
        })
}
fn exact_resources(e: &MixedExecutor, function: &CudaFunction) -> bool {
    resources(e, function).unwrap_or(false)
        && function.shared_size_bytes().is_ok_and(|n| n == 49152)
}
/// The selected IR has one global output allocation and proved mixed-radix
/// windows. It cannot receive independent, potentially overlapping Q/K/V args.
pub(super) fn prepare(
    preparation: &Preparation,
    r: &Region,
    e: &MixedExecutor,
    d: &Draft,
) -> std::result::Result<Packed, Rejection> {
    #[cfg(test)]
    if FORCE_REJECT.with(|flag| flag.get()) {
        return Err(Rejection::Resources);
    }
    if d.shape.heads != 8
        || ![64, 128, 256].contains(&d.shape.bh())
        || d.transport != leaf::Transport::Async16
        || d.shape.batch.checked_mul(8) != Some(d.shape.bh())
        || r.proof.required_qkv_packs.len() != 1
    {
        return Err(Rejection::UnsupportedLayout);
    }
    // Runtime scope is the measured canonical sequence-major Q/K/V/dO layout;
    // structural support for another leaf layout is not a device qualification.
    let input = d
        .input
        .iter()
        .zip(&r.proof.inputs)
        .map(|(_, node)| preparation.value(*node).map(descriptor))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let abi = leaf::UncompiledPair::with_transport(
        d.shape,
        input.try_into().map_err(|_| Rejection::UnsupportedLayout)?,
        leaf::Transport::Async16,
    )
    .map_err(|_| Rejection::UnsupportedLayout)?;
    if abi
        .read_layouts
        .iter()
        .any(|layout| *layout != leaf::ReadLayout::SequenceMajor)
    {
        return Err(Rejection::UnsupportedLayout);
    }
    let output = *r
        .proof
        .required_qkv_packs
        .iter()
        .next()
        .ok_or(Rejection::UnverifiedPack)?;
    let p = preparation
        .packs
        .get(&output)
        .ok_or(Rejection::UnverifiedPack)?;
    if p.shape != [d.shape.batch, 512, 1536] || p.inputs.len() != 3 {
        return Err(Rejection::UnverifiedPack);
    }
    if r.proof
        .surviving_consumers
        .iter()
        .any(|(node, consumer, _)| d.outputs[1..].contains(node) && *consumer != output)
    {
        return Err(Rejection::EarlySurvivingReader);
    }
    let remove = match preparation.verify_pack(output, e) {
        Ok(indices) => indices,
        Err(reason) => {
            #[cfg(test)]
            group_diagnostic(preparation, e, output, output, "verify-pack", &reason);
            return Err(reason);
        }
    };
    let [pack_position] = remove.as_slice() else {
        return Err(Rejection::UnverifiedPack);
    };
    if *pack_position <= d.bwd {
        return Err(Rejection::EarlySurvivingReader);
    }
    let value = preparation.value(output)?.clone();
    if !identity(&value, &p.shape, DType::BF16)
        || value.view.source != output
        || !value.binding.pointer.is_multiple_of(16)
    {
        return Err(Rejection::UnsupportedLayout);
    }
    let storage = owner(e, value.binding)?;
    if !disjoint(
        &[value.binding],
        &d.input.iter().chain(&d.output).copied().collect::<Vec<_>>(),
    ) {
        return Err(Rejection::AliasedBindings);
    }
    let mut windows = HashSet::new();
    for i in &p.inputs {
        if i.window >= 3 || !windows.insert(i.window) || i.source != d.outputs[i.window + 1] {
            return Err(Rejection::UnverifiedPack);
        }
        let mut view = preparation.value(i.source)?.view.clone();
        for op in &i.views {
            view = apply_view(view, op).ok_or(Rejection::UnsupportedLayout)?;
        }
        if !same_logical_view(&view, &expected_view(i.source, d.shape, i.window)?) {
            return Err(Rejection::UnsupportedLayout);
        }
    }
    let readers = p
        .readers
        .as_ref()
        .filter(|v| !v.numeric.is_empty())
        .ok_or(Rejection::EarlySurvivingReader)?;
    for proof in &readers.identity_casts {
        let parent = preparation.value(proof.parent)?;
        let cast = preparation.value(proof.node)?;
        // Mixed's same-binding-dtype branch clones the source Value. Prove the
        // exact permitted BF16 view chain, not a logical Cast label alone.
        if !physical_identity_cast(output, &p.shape, &value, proof, (parent, cast)) {
            return Err(Rejection::UnsupportedLayout);
        }
    }
    let consumers = &readers.numeric;
    let mut allowed = HashSet::new();
    let mut last = *pack_position;
    for &reader in consumers {
        let group = match preparation.boundary(reader, &e.kernels) {
            Ok(indices) => indices,
            Err(reason) => {
                #[cfg(test)]
                group_diagnostic(preparation, e, output, reader, "numeric-reader", &reason);
                return Err(reason);
            }
        };
        if group.first().is_none_or(|i| *i <= *pack_position) {
            return Err(Rejection::EarlySurvivingReader);
        }
        last = last.max(*group.last().ok_or(Rejection::ChangedOriginalGroup)?);
        allowed.extend(group);
    }
    // Full owning-range census covers EVERY surviving physical alias, not just
    // graph Value keys retained by the pool planner. Numeric reader groups are
    // resolved through exact function+owner stamps; labels alone never admit.
    let (owner_lo, owner_hi) = extent(storage)?;
    for (position, kernel) in e.kernels.iter().enumerate().take(last + 1).skip(d.bwd) {
        if d.remove.contains(&position) || remove.contains(&position) {
            continue;
        }
        let touches = kernel
            .pointers
            .iter()
            .any(|pointer| owner_lo <= *pointer && *pointer < owner_hi);
        let zero = kernel
            .zero_before
            .and_then(range)
            .is_some_and(|(a, z)| a < owner_hi && owner_lo < z);
        if zero || (touches && !allowed.contains(&position)) {
            return Err(Rejection::AliasedBindings);
        }
    }
    let origin = match preparation.origin(output) {
        Ok(origin) => origin,
        Err(reason) => {
            #[cfg(test)]
            group_diagnostic(
                preparation,
                e,
                output,
                output,
                "pack-origin-lookup",
                &reason,
            );
            return Err(reason);
        }
    };
    if origin.key != StorageKey::Global(output)
        || range(origin.binding) != range(value.binding)
        || origin.binding.dtype != DType::BF16
        || group_indices(&origin.producer, &e.kernels) != Some(remove.clone())
    {
        #[cfg(test)]
        group_diagnostic(
            preparation,
            e,
            output,
            output,
            "pack-origin-proof",
            &Rejection::UnverifiedPack,
        );
        return Err(Rejection::UnverifiedPack);
    }
    let region = QkvGradPack {
        output,
        shape: p.shape.clone(),
        inputs: p
            .inputs
            .iter()
            .map(|i| Input {
                source: i.source,
                window: i.window,
                views: i.views.clone(),
            })
            .collect(),
        tree: p.tree.clone(),
        members: p.members.clone(),
    };
    let dq = backward::packed::dquery_ir(d.shape, abi.read_layouts[0], &region)
        .map_err(|_| Rejection::Compilation)?;
    let dkv = backward::packed::dkeyvalue_ir(d.shape, abi.read_layouts[0], &region)
        .map_err(|_| Rejection::Compilation)?;
    let d_query = compile(e, dq).map_err(|_| Rejection::Compilation)?;
    let d_key_value = compile(e, dkv).map_err(|_| Rejection::Compilation)?;
    if !exact_resources(e, &d_query) || !exact_resources(e, &d_key_value) {
        return Err(Rejection::Resources);
    }
    Ok(Packed {
        output,
        value,
        remove,
        d_query,
        d_key_value,
        reserved: ReservedSpan {
            source: origin.key,
            binding: origin.binding,
            first: d.bwd,
            last,
        },
        members: p.members.clone(),
    })
}

#[cfg(test)]
mod tests;

#[cfg(test)]
thread_local! { static FORCE_REJECT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }
#[cfg(test)]
struct ForcedReject(bool);
#[cfg(test)]
impl ForcedReject {
    fn install() -> Self {
        Self(FORCE_REJECT.with(|flag| flag.replace(true)))
    }
}
#[cfg(test)]
impl Drop for ForcedReject {
    fn drop(&mut self) {
        FORCE_REJECT.with(|flag| flag.set(self.0));
    }
}
#[cfg(test)]
mod gpu_tests;

/// Diagnostic matching is explanatory ONLY. Admission continues to use the
/// unchanged exact function+owner+pointers+config stamp equality above.
#[cfg(test)]
fn group_diagnostic(
    preparation: &Preparation,
    e: &MixedExecutor,
    packed: NodeIndex,
    node: NodeIndex,
    phase: &str,
    reason: &Rejection,
) {
    let Some(path) = std::env::var_os("TNSR_ONLINE_PACKED_RUNTIME_AUDIT") else {
        return;
    };
    let detail = |k: &Kernel| serde_json::json!({"function":stamp(k).function,"owner":k.owner.map(|n|n.index()),"label":k.label,"pointers":k.pointers,"grid":k.config.grid_dim,"block":k.config.block_dim,"shared":k.config.shared_mem_bytes,"zero_before":k.zero_before.map(|b|serde_json::json!({"pointer":b.pointer,"len":b.len,"dtype":format!("{:?}",b.dtype)}))});
    let old = preparation
        .boundaries
        .get(&node)
        .or_else(|| preparation.groups.get(&node).map(|g| &g.kernels));
    let expected=old.into_iter().flatten().map(|k|{
        let expected=stamp(k);
        let candidates=e.kernels.iter().enumerate().filter(|(_,current)|{
            let actual=stamp(current);
            current.owner==Some(node) || (expected.function.is_some()&&actual.function==expected.function)
                || actual.label==expected.label || actual.pointers==expected.pointers
        }).take(24).map(|(position,k)|{
            let actual=stamp(k);let mut changed=Vec::new();
            if expected.function!=actual.function{changed.push("function");}
            if expected.owner!=actual.owner{changed.push("owner");}
            if expected.label!=actual.label{changed.push("label");}
            if expected.pointers!=actual.pointers{changed.push("pointers");}
            if expected.grid!=actual.grid{changed.push("grid");}
            if expected.block!=actual.block{changed.push("block");}
            if expected.shared!=actual.shared{changed.push("shared");}
            if expected.zero!=actual.zero{changed.push("zero_before");}
            serde_json::json!({"position":position,"stamp":detail(k),"changed_fields":changed})
        }).collect::<Vec<_>>();
        serde_json::json!({"expected":detail(k),"candidate_count_limit":24,"candidates":candidates})
    }).collect::<Vec<_>>();
    let origin=preparation.origin(node).ok().map(|o|{
        let key=match o.key{StorageKey::Global(n)=>serde_json::json!({"kind":"global","node":n.index()}),StorageKey::F32{program,physical}=>serde_json::json!({"kind":"f32","program":program,"physical":physical.index()})};
        serde_json::json!({"key":key,"binding":{"pointer":o.binding.pointer,"len":o.binding.len,"dtype":format!("{:?}",o.binding.dtype)},"producer":o.producer.iter().map(&detail).collect::<Vec<_>>(),"exact_current_group":group_indices(&o.producer,&e.kernels)})
    });
    let report = serde_json::json!({"packed":packed.index(),"node":node.index(),"phase":phase,"reason":format!("{reason}"),"expected_groups":expected,"origin":origin,"graph_alias_parent":preparation.aliases.get(&node).map(|n|n.index()),"scope":"test-only mismatch diagnostics, never used to admit by labels or approximate identity"});
    let path = std::path::PathBuf::from(path);
    if let Err(error) = std::fs::create_dir_all(&path).and_then(|_| {
        std::fs::write(
            path.join(format!(
                "group-mismatch-pack{}-{phase}-node{}.json",
                packed.index(),
                node.index()
            )),
            serde_json::to_vec_pretty(&report).expect("finite diagnostic JSON"),
        )
    }) {
        eprintln!("packed mismatch diagnostic write failed: {error}");
    }
}
