use super::*;
use petgraph::Direction;
fn one<G>(g: &TensorGraph<f32, G>, n: NodeIndex) -> Option<NodeIndex> {
    let [v] = g.inputs(n).try_into().ok()?;
    Some(v)
}
fn count(shape: &[usize]) -> Option<usize> {
    shape.iter().try_fold(
        1_usize,
        |a, n| if *n > 0 { a.checked_mul(*n) } else { None },
    )
}
fn zero<G>(
    g: &TensorGraph<f32, G>,
    n: NodeIndex,
    dtypes: &HashMap<NodeIndex, ValueDType>,
    members: &mut HashSet<NodeIndex>,
) -> Option<()> {
    if dtypes.get(&n) != Some(&ValueDType::F32) {
        return None;
    }
    match &g[n] {
        TensorGraphNode::Constant { data, shape }
            if data.len() == 1 && data[0].to_bits() == 0 && count(shape) == Some(1) =>
        {
            Some(())
        }
        TensorGraphNode::Reshape { shape } | TensorGraphNode::Flatten { shape }
            if count(shape) == Some(1) =>
        {
            members.insert(n);
            zero(g, one(g, n)?, dtypes, members)
        }
        TensorGraphNode::BroadcastAxis { axis, shape } => {
            let input = one(g, n)?;
            let source = g[input].shape();
            if source.len() != shape.len()
                || *axis >= shape.len()
                || source[*axis] != 1
                || source
                    .iter()
                    .enumerate()
                    .any(|(i, v)| i != *axis && *v != shape[i])
                || count(shape).is_none()
            {
                return None;
            }
            members.insert(n);
            zero(g, input, dtypes, members)
        }
        _ => None,
    }
}
fn trace<G>(
    g: &TensorGraph<f32, G>,
    n: NodeIndex,
    dtypes: &HashMap<NodeIndex, ValueDType>,
    members: &mut HashSet<NodeIndex>,
    views: &mut Vec<TensorGraphNode<f32>>,
) -> Option<NodeIndex> {
    let input = one(g, n);
    match &g[n] {
        TensorGraphNode::Reshape { shape }
        | TensorGraphNode::Flatten { shape }
        | TensorGraphNode::Permute { shape, .. }
        | TensorGraphNode::Transpose { shape } => {
            let input = input?;
            let source = g[input].shape();
            if count(source) != count(shape) || count(shape).is_none() {
                return None;
            }
            let identity = VirtualTensor::identity(input, source.clone(), DType::BF16);
            let mapped = view(&identity, &g[n]).ok()?;
            if mapped.shape != *shape {
                return None;
            }
            let leaf = trace(g, input, dtypes, members, views)?;
            members.insert(n);
            views.push(g[n].clone());
            Some(leaf)
        }
        TensorGraphNode::Cast { dtype, .. } => {
            let input = input?;
            if g[n].shape() != g[input].shape() {
                return None;
            }
            if *dtype == ValueDType::BF16
                && dtypes.get(&input) == Some(&ValueDType::F32)
                && matches!(
                    g[input],
                    TensorGraphNode::Input { .. }
                        | TensorGraphNode::Parameter { .. }
                        | TensorGraphNode::Constant { .. }
                )
            {
                return Some(n);
            }
            // Only exact BF16 promotion (or an identity cast) is transparent.
            // A second narrowing conversion is not a transparent view.
            let source_dtype = dtypes.get(&input)?;
            if !(*dtype == *source_dtype
                || (*dtype == ValueDType::F32 && *source_dtype == ValueDType::BF16))
            {
                return None;
            }
            let leaf = trace(g, input, dtypes, members, views)?;
            members.insert(n);
            Some(leaf)
        }
        TensorGraphNode::MatMul { .. } if dtypes.get(&n) == Some(&ValueDType::BF16) => Some(n),
        _ => None,
    }
}
fn pad<G>(
    g: &TensorGraph<f32, G>,
    n: NodeIndex,
    shape: &[usize],
    dtypes: &HashMap<NodeIndex, ValueDType>,
    members: &mut HashSet<NodeIndex>,
) -> Option<Input> {
    if !matches!(
        g[n],
        TensorGraphNode::Cast {
            dtype: ValueDType::BF16,
            ..
        }
    ) || g[n].shape() != shape
    {
        return None;
    }
    let concat = one(g, n)?;
    if !matches!(g[concat], TensorGraphNode::Concat { axis: 2, .. })
        || g[concat].shape() != shape
        || dtypes.get(&concat) != Some(&ValueDType::F32)
    {
        return None;
    }
    let pieces = g.inputs(concat);
    if !(2..=3).contains(&pieces.len()) {
        return None;
    }
    let width = shape[2] / 3;
    let mut offset: usize = 0;
    let mut selected = None;
    let mut all = HashSet::new();
    for piece in pieces.iter().copied() {
        let s = g[piece].shape();
        if s.len() != 3 || s[..2] != shape[..2] || count(s).is_none() {
            return None;
        }
        let mut zeros = HashSet::new();
        if zero(g, piece, dtypes, &mut zeros).is_some() {
            all.extend(zeros);
        } else {
            if selected.is_some() || s[2] != width || !offset.is_multiple_of(width) {
                return None;
            }
            let mut branch = HashSet::new();
            let mut views = Vec::new();
            let source = trace(g, piece, dtypes, &mut branch, &mut views)?;
            if dtypes.get(&source) != Some(&ValueDType::BF16) {
                return None;
            }
            let raw = g[source].shape();
            if raw.len() != 4 || raw[0] != shape[0] || raw[1] < 2 || !width.is_multiple_of(raw[1]) {
                return None;
            }
            let depth = width / raw[1];
            if raw != &vec![shape[0], raw[1], shape[1], depth]
                && raw != &vec![shape[0], raw[1], depth, shape[1]]
            {
                return None;
            }
            let mut virtual_value = VirtualTensor::identity(source, raw.clone(), DType::BF16);
            for v in &views {
                virtual_value = view(&virtual_value, v).ok()?;
            }
            if virtual_value.shape != *s {
                return None;
            }
            all.extend(branch);
            selected = Some(Input {
                source,
                window: offset / width,
                views,
            });
        }
        offset = offset.checked_add(s[2])?;
    }
    if offset != shape[2] {
        return None;
    }
    let selected = selected?;
    if selected.window > 2 || pieces.len() != if selected.window == 1 { 3 } else { 2 } {
        return None;
    }
    members.extend(all);
    members.extend([n, concat]);
    Some(selected)
}
fn tree<G>(
    g: &TensorGraph<f32, G>,
    n: NodeIndex,
    shape: &[usize],
    dtypes: &HashMap<NodeIndex, ValueDType>,
    members: &mut HashSet<NodeIndex>,
    inputs: &mut Vec<Input>,
    adds: &mut usize,
) -> Option<Join> {
    if g[n].shape() != shape || dtypes.get(&n) != Some(&ValueDType::BF16) {
        return None;
    }
    if matches!(
        g[n],
        TensorGraphNode::Binary {
            op: BinaryOp::Add,
            ..
        }
    ) {
        *adds += 1;
        if *adds > 2 {
            return None;
        }
        let [a, b] = g.inputs(n).try_into().ok()?;
        members.insert(n);
        Some(Join::Add(
            Box::new(tree(g, a, shape, dtypes, members, inputs, adds)?),
            Box::new(tree(g, b, shape, dtypes, members, inputs, adds)?),
        ))
    } else {
        let input = pad(g, n, shape, dtypes, members)?;
        let index = inputs.len();
        inputs.push(input);
        Some(Join::Input(index))
    }
}
pub(in crate::ptx::mixed) fn recognize<G>(g: &TensorGraph<f32, G>) -> Vec<QkvGradPack> {
    let dtypes = value_dtypes(g);
    let mut regions = Vec::new();
    let mut claimed = HashSet::new();
    for output in g.toposort() {
        if !matches!(
            g[output],
            TensorGraphNode::Binary {
                op: BinaryOp::Add,
                ..
            }
        ) {
            continue;
        }
        let shape = g[output].shape();
        if shape.len() != 3
            || shape[2] == 0
            || !shape[2].is_multiple_of(3)
            || count(shape).is_none_or(|n| u32::try_from(n).is_err())
        {
            continue;
        }
        let mut members = HashSet::new();
        let mut inputs = Vec::new();
        let mut adds = 0;
        let Some(tree) = tree(
            g,
            output,
            shape,
            &dtypes,
            &mut members,
            &mut inputs,
            &mut adds,
        ) else {
            continue;
        };
        if adds != 2 || inputs.len() != 3 {
            continue;
        }
        let mut windows: Vec<_> = inputs.iter().map(|i| i.window).collect();
        windows.sort_unstable();
        if windows != [0, 1, 2] {
            continue;
        }
        if members.iter().any(|node| {
            claimed.contains(node)
                || (*node != output
                    && (g
                        .graph
                        .neighbors_directed(*node, Direction::Outgoing)
                        .next()
                        .is_none()
                        || g.graph
                            .neighbors_directed(*node, Direction::Outgoing)
                            .any(|n| !members.contains(&n))))
        }) {
            continue;
        }
        claimed.extend(members.iter().copied());
        regions.push(QkvGradPack {
            output,
            shape: shape.clone(),
            inputs,
            tree,
            members,
        });
    }
    regions
}
