//! Rebind private intermediates by their complete captured-kernel lifetimes.
use super::*;
use std::collections::HashSet;

mod plan;
use plan::{Allocation, plan_slots};

impl MixedExecutor {
    pub(super) fn reuse_intermediates<G>(
        &mut self,
        graph: &TensorGraph<f32, G>,
        future: &HashSet<NodeIndex>,
        previous_retained_bytes: usize,
    ) -> Result<usize> {
        self.freshness.dirty();
        self.borrowed_prefixes.clear();
        if let Some(preparation) = &mut self.attention_training_preparation {
            preparation.capture(&self.values, &self.kernels);
        }
        if let Some(preparation) = &mut self.online_attention_pair_preparation {
            preparation.capture_before_reuse(&self.values, &self.kernels);
        }
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
        let live: HashSet<_> = future
            .iter()
            .filter_map(|node| self.values.get(node).map(|v| v.binding.pointer))
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
        let allocations: Vec<_> = self
            .buffers
            .iter()
            .map(|buffer| {
                let binding = binding(buffer, &self.stream);
                Allocation {
                    binding,
                    pinned: pinned.contains(&binding.pointer),
                    future_live: live.contains(&binding.pointer),
                }
            })
            .collect();
        let uses = self.kernels.iter().enumerate().flat_map(|(step, kernel)| {
            kernel
                .pointers
                .iter()
                .copied()
                .chain(kernel.zero_before.map(|b| b.pointer))
                .map(move |pointer| (step, pointer))
        });
        // Prospective LN reads are reserved at their ORIGINAL target kernel
        // positions before any prefix can recycle an early-dead saved buffer.
        let prospective = self
            .layer_norm_preparation
            .as_ref()
            .map(|p| p.prospective_uses())
            .unwrap_or_default();
        let attention_uses = self
            .attention_training_preparation
            .as_ref()
            .map(|p| p.prospective_uses(&self.kernels))
            .unwrap_or_default();
        let online_uses = self
            .online_attention_pair_preparation
            .as_ref()
            .map(|p| p.prospective_uses(&self.kernels))
            .unwrap_or_default();
        let (slots, remap) = plan_slots(
            &allocations,
            uses.chain(prospective)
                .chain(attention_uses)
                .chain(online_uses),
            end,
        )?;
        if let Some(preparation) = &mut self.attention_training_preparation {
            preparation.remap(&remap);
        }

        if let Some(preparation) = &mut self.layer_norm_preparation {
            preparation.remap(&remap);
        }
        if let Some(preparation) = &mut self.online_attention_pair_preparation {
            preparation.remap(&remap);
        }
        self.info.reused_allocations += allocations.len() - slots.len();
        let old_sources: Vec<_> = self
            .sources
            .iter()
            .filter_map(|(&node, &storage)| match storage {
                SourceStorage::Owned(index) => Some((node, allocations[index].binding.pointer)),
                SourceStorage::External => None,
            })
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
                SourceStorage::Owned(
                    *source_slots
                        .get(&remap[&pointer])
                        .context("source slot missing")?,
                ),
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
        self.values
            .retain(|node, _| retained.contains(node) || future.contains(node));
        for value in self.values.values_mut() {
            value.binding.pointer = *remap
                .get(&value.binding.pointer)
                .unwrap_or(&value.binding.pointer);
        }
        let allocated_bytes = self
            .info
            .f32_bytes
            .checked_add(self.info.bf16_bytes)
            .context("mixed allocation byte count overflow")?;
        let newly_allocated = allocated_bytes
            .checked_sub(previous_retained_bytes)
            .context("mixed retained allocation accounting decreased")?;
        self.info.unpooled_bytes = self
            .info
            .unpooled_bytes
            .checked_add(newly_allocated)
            .context("mixed unpooled byte count overflow")?;
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
        Ok(self.info.f32_bytes + self.info.bf16_bytes)
    }
}
