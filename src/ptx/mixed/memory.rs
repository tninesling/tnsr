//! Rebind private intermediates by their complete captured-kernel lifetimes.
use super::*;
use std::collections::HashSet;

struct Allocation {
    binding: Binding,
    first: usize,
    last: usize,
    pinned: bool,
}
struct Slot {
    original: usize,
    binding: Binding,
    last: usize,
    pinned: bool,
}
impl MixedExecutor {
    pub(super) fn reuse_intermediates<G>(&mut self, graph: &TensorGraph<f32, G>) -> Result<()> {
        // Resident consumers read graph sinks after execution. Sources retain
        // stable addresses for feed uploads and immutable constant caching.
        let retained: HashSet<_> = graph
            .graph
            .node_indices()
            .filter(|n| {
                self.sources.contains_key(n)
                    || graph
                        .graph
                        .neighbors_directed(*n, petgraph::Direction::Outgoing)
                        .next()
                        .is_none()
            })
            .collect();
        let pinned: HashSet<_> = retained
            .iter()
            .filter_map(|n| self.values.get(n).map(|v| v.binding.pointer))
            .collect();
        for program in &mut self.programs {
            if let Some(p) = program.downcast_mut::<Program<F32>>() {
                self.buffers.extend(p.drain_storage());
            }
            if let Some(p) = program.downcast_mut::<Program<BF16>>() {
                self.buffers.extend(p.drain_storage());
            }
        }
        let end = self.kernels.len();
        let mut allocations: Vec<_> = self
            .buffers
            .iter()
            .map(|buffer| {
                let binding = binding(buffer, &self.stream);
                Allocation {
                    binding,
                    first: end,
                    last: 0,
                    pinned: pinned.contains(&binding.pointer),
                }
            })
            .collect();
        let indices: HashMap<_, _> = allocations
            .iter()
            .enumerate()
            .map(|(i, a)| (a.binding.pointer, i))
            .collect();
        for (step, kernel) in self.kernels.iter().enumerate() {
            for pointer in kernel
                .pointers
                .iter()
                .copied()
                .chain(kernel.zero_before.map(|b| b.pointer))
            {
                if let Some(&index) = indices.get(&pointer) {
                    let a = &mut allocations[index];
                    a.first = a.first.min(step);
                    a.last = step;
                }
            }
        }
        for a in &mut allocations {
            if a.pinned {
                a.first = 0;
                a.last = end;
            }
        }
        let mut order: Vec<_> = (0..allocations.len()).collect();
        order.sort_by_key(|&i| allocations[i].first);
        let mut slots: Vec<Slot> = Vec::new();
        let mut remap = HashMap::new();
        for index in order {
            let a = &allocations[index];
            // Match the smallest available allocation. A kernel's inputs and
            // outputs overlap at that step, so only strictly earlier deaths
            // can be reused. Dtypes never share a slot.
            let available = (!a.pinned)
                .then(|| {
                    slots
                        .iter()
                        .enumerate()
                        .filter(|(_, s)| {
                            !s.pinned
                                && s.last < a.first
                                && s.binding.dtype == a.binding.dtype
                                && s.binding.len >= a.binding.len
                        })
                        .min_by_key(|(_, s)| s.binding.len)
                        .map(|(i, _)| i)
                })
                .flatten();
            let slot = if let Some(i) = available {
                self.info.reused_allocations += 1;
                slots[i].last = a.last;
                i
            } else {
                slots.push(Slot {
                    original: index,
                    binding: a.binding,
                    last: a.last,
                    pinned: a.pinned,
                });
                slots.len() - 1
            };
            remap.insert(a.binding.pointer, slots[slot].binding.pointer);
        }
        let old_sources: Vec<_> = self
            .sources
            .iter()
            .map(|(&node, &index)| (node, allocations[index].binding.pointer))
            .collect();
        let mut buffers: Vec<_> = std::mem::take(&mut self.buffers)
            .into_iter()
            .map(Some)
            .collect();
        for slot in slots {
            self.buffers.push(
                buffers[slot.original]
                    .take()
                    .context("memory slot owner reused")?,
            );
        }
        let source_slots: HashMap<_, _> = self
            .buffers
            .iter()
            .enumerate()
            .map(|(i, b)| (binding(b, &self.stream).pointer, i))
            .collect();
        for (node, pointer) in old_sources {
            self.sources.insert(
                node,
                *source_slots
                    .get(&remap[&pointer])
                    .context("source slot missing")?,
            );
        }
        for kernel in &mut self.kernels {
            for pointer in &mut kernel.pointers {
                if let Some(&replacement) = remap.get(pointer) {
                    *pointer = replacement;
                }
            }
            if let Some(b) = &mut kernel.zero_before {
                b.pointer = *remap.get(&b.pointer).unwrap_or(&b.pointer);
            }
        }
        self.values.retain(|node, _| retained.contains(node));
        for value in self.values.values_mut() {
            value.binding.pointer = *remap
                .get(&value.binding.pointer)
                .unwrap_or(&value.binding.pointer);
        }
        self.info.unpooled_bytes = self.info.f32_bytes + self.info.bf16_bytes;
        self.info.f32_bytes = 0;
        self.info.bf16_bytes = 0;
        for buffer in &self.buffers {
            let b = binding(buffer, &self.stream);
            match b.dtype {
                DType::F32 => self.info.f32_bytes += b.len * 4,
                DType::BF16 => self.info.bf16_bytes += b.len * 2,
                _ => unreachable!(),
            }
        }
        // Original alloc_zeros may still be queued. Fence before dropping any
        // discarded owner; the remapped program has not launched or captured.
        self.stream.synchronize()?;
        Ok(())
    }
}
