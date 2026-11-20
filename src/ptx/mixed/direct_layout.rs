//! Pointer relocation metadata for borrowed resident masters. No CUDA calls.
use super::*;

pub(super) struct DirectLayout {
    sources: Vec<(NodeIndex, usize, DType)>,
    alias_groups: Vec<Vec<NodeIndex>>,
    arguments: Vec<(usize, usize, NodeIndex)>,
    aliases: Vec<(NodeIndex, NodeIndex)>,
    writes: Vec<(u64, u64)>,
    zero_targets: Vec<u64>,
}
pub(super) struct Rebinding {
    pub key: Vec<u64>,
    pub arguments: Vec<(usize, usize, u64)>,
    pub aliases: Vec<(NodeIndex, u64)>,
}
impl DirectLayout {
    #[cfg(test)]
    pub fn new(
        sources: &HashMap<NodeIndex, Binding>,
        arguments: &[Vec<u64>],
        zero_targets: &[Option<u64>],
        values: &HashMap<NodeIndex, Binding>,
    ) -> Result<Self> {
        Self::new_with_writes(sources, arguments, zero_targets, values, &[])
    }
    pub fn new_with_writes(
        sources: &HashMap<NodeIndex, Binding>,
        arguments: &[Vec<u64>],
        zero_targets: &[Option<u64>],
        values: &HashMap<NodeIndex, Binding>,
        writes: &[Binding],
    ) -> Result<Self> {
        ensure!(!sources.is_empty(), "direct resident sources are empty");
        let mut ordered: Vec<_> = sources
            .iter()
            .map(|(&node, &binding)| (node, binding))
            .collect();
        ordered.sort_by_key(|(node, _)| node.index());
        let mut physical: HashMap<u64, (NodeIndex, usize, DType)> = HashMap::new();
        for &(node, binding) in &ordered {
            ensure!(
                matches!(binding.dtype, DType::F32 | DType::BF16) && binding.len > 0,
                "direct source must be nonempty f32 or BF16"
            );
            if let Some(&(previous, len, dtype)) = physical.get(&binding.pointer) {
                ensure!(
                    len == binding.len && dtype == binding.dtype,
                    "direct source aliases disagree on extent/dtype"
                );
                // Multiple graph parameter nodes may name one registered master.
                ensure!(
                    previous.index() < node.index(),
                    "direct source ordering is inconsistent"
                );
            } else {
                physical.insert(binding.pointer, (node, binding.len, binding.dtype));
            }
        }
        let ranges = source_ranges(sources)?;
        let writes = writes
            .iter()
            .copied()
            .map(binding_span)
            .collect::<Result<Vec<_>>>()?;
        validate_writes(&ranges, &writes)?;
        ensure!(
            zero_targets.len() == arguments.len(),
            "direct kernel metadata count mismatch"
        );
        ensure!(
            !zero_targets.iter().flatten().any(|pointer| ranges
                .iter()
                .any(|range| *pointer >= range.0 && *pointer < range.1)),
            "direct master is a mutable zero target"
        );
        let mut references = Vec::new();
        for (kernel, arguments) in arguments.iter().enumerate() {
            for (argument, pointer) in arguments.iter().enumerate() {
                if let Some(&(node, _, _)) = physical.get(pointer) {
                    references.push((kernel, argument, node));
                } else {
                    ensure!(
                        !ranges
                            .iter()
                            .any(|range| *pointer > range.0 && *pointer < range.1),
                        "direct kernel has an unrelocatable interior source pointer"
                    );
                }
            }
        }
        let mut aliases = Vec::new();
        for (&node, value) in values {
            if let Some(&(source, len, dtype)) = physical.get(&value.pointer) {
                ensure!(
                    value.dtype == dtype && value.len == len,
                    "direct value alias changes its physical extent/dtype"
                );
                aliases.push((node, source));
            } else {
                ensure!(
                    !ranges
                        .iter()
                        .any(|range| value.pointer > range.0 && value.pointer < range.1),
                    "direct value has an unrelocatable interior source pointer"
                );
            }
        }
        let mut grouped: HashMap<u64, Vec<NodeIndex>> = HashMap::new();
        for &(node, binding) in &ordered {
            grouped.entry(binding.pointer).or_default().push(node);
        }
        Ok(Self {
            alias_groups: grouped.into_values().collect(),
            sources: ordered
                .into_iter()
                .map(|(node, binding)| (node, binding.len, binding.dtype))
                .collect(),
            arguments: references,
            aliases,
            writes,
            zero_targets: zero_targets.iter().flatten().copied().collect(),
        })
    }
    pub fn rebind(&self, sources: &HashMap<NodeIndex, Binding>) -> Result<Rebinding> {
        ensure!(
            sources.len() == self.sources.len(),
            "direct master set changed"
        );
        let mut key = Vec::with_capacity(self.sources.len());
        for &(node, len, dtype) in &self.sources {
            let binding = sources.get(&node).context("direct master missing")?;
            ensure!(
                binding.len == len && binding.dtype == dtype,
                "direct master extent/dtype changed"
            );
            key.push(binding.pointer);
        }
        let ranges = source_ranges(sources)?;
        validate_writes(&ranges, &self.writes)?;
        ensure!(
            !self.zero_targets.iter().any(|pointer| ranges
                .iter()
                .any(|range| *pointer >= range.0 && *pointer < range.1)),
            "direct source relocated onto a mutable zero target"
        );
        let mut addresses = std::collections::HashSet::new();
        for group in &self.alias_groups {
            let pointer = sources[&group[0]].pointer;
            ensure!(
                addresses.insert(pointer),
                "distinct direct masters became aliases"
            );
            ensure!(
                group.iter().all(|node| sources[node].pointer == pointer),
                "direct master alias group split"
            );
        }
        Ok(Rebinding {
            key,
            arguments: self
                .arguments
                .iter()
                .map(|&(kernel, argument, node)| (kernel, argument, sources[&node].pointer))
                .collect(),
            aliases: self
                .aliases
                .iter()
                .map(|&(node, source)| (node, sources[&source].pointer))
                .collect(),
        })
    }
}

fn source_ranges(sources: &HashMap<NodeIndex, Binding>) -> Result<Vec<(u64, u64, DType)>> {
    let mut spans = Vec::new();
    for binding in sources.values() {
        let width = match binding.dtype {
            DType::F32 => 4u64,
            DType::BF16 => 2,
            _ => anyhow::bail!("unsupported direct source dtype"),
        };
        let (start, end) = binding_span(*binding)?;
        ensure!(
            binding.pointer % width == 0,
            "direct typed source is misaligned"
        );
        spans.push((start, end, binding.dtype));
    }
    for (i, a) in spans.iter().enumerate() {
        for b in &spans[i + 1..] {
            ensure!(
                a == b || a.0 >= b.1 || b.0 >= a.1,
                "distinct direct source ranges overlap"
            );
        }
    }
    Ok(spans)
}

fn binding_span(binding: Binding) -> Result<(u64, u64)> {
    let width = match binding.dtype {
        DType::BF16 => 2,
        DType::F32 | DType::U32 => 4,
        _ => anyhow::bail!("unsupported direct storage dtype"),
    };
    let bytes = u64::try_from(binding.len)?
        .checked_mul(width)
        .context("direct storage size overflow")?;
    Ok((
        binding.pointer,
        binding
            .pointer
            .checked_add(bytes)
            .context("direct storage address overflow")?,
    ))
}

fn validate_writes(sources: &[(u64, u64, DType)], writes: &[(u64, u64)]) -> Result<()> {
    for source in sources {
        for write in writes {
            ensure!(
                source.0 >= write.1 || write.0 >= source.1,
                "direct source aliases private/writable storage"
            );
        }
    }
    Ok(())
}
