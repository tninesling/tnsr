//! Private, stable integer source buffers. No allocation enters a capture.
use super::*;
use crate::graph::NodeIndex;
use std::sync::Weak;

pub(super) struct DeviceIndex {
    pub values: CudaSlice<u32>,
    pub maximum: Option<u32>,
    source: Weak<Vec<u32>>,
}

#[derive(Default)]
pub(super) struct IndexBuffers {
    pub buffers: HashMap<NodeIndex, DeviceIndex>,
    pub allocated_bytes: usize,
    pub allocations: usize,
}

impl IndexBuffers {
    /// Called under the executor gate, after compilation has invalidated any
    /// graph whose structure could change these allocations. Same-shape updates
    /// overwrite the existing allocation before replay on the same stream.
    pub fn stage(
        &mut self,
        stream: &Arc<CudaStream>,
        sources: HashMap<NodeIndex, Arc<Vec<u32>>>,
    ) -> Result<usize> {
        self.buffers.retain(|node, _| sources.contains_key(node));
        let mut copies = 0;
        for (node, source) in sources {
            let needs_allocation = self
                .buffers
                .get(&node)
                .is_none_or(|buffer| buffer.values.len() != source.len());
            if needs_allocation {
                self.buffers.insert(
                    node,
                    DeviceIndex {
                        values: if source.is_empty() {
                            stream.null()?
                        } else {
                            stream.alloc_zeros(source.len())?
                        },
                        maximum: None,
                        source: Weak::new(),
                    },
                );
                if !source.is_empty() {
                    self.allocated_bytes += source.len() * std::mem::size_of::<u32>();
                    self.allocations += 1;
                }
            }
            let buffer = self
                .buffers
                .get_mut(&node)
                .context("integer source allocation missing")?;
            if !buffer.source.ptr_eq(&Arc::downgrade(&source)) {
                if !source.is_empty() {
                    stream.memcpy_htod(source.as_slice(), &mut buffer.values)?;
                    copies += 1;
                }
                buffer.maximum = source.iter().copied().max();
                buffer.source = Arc::downgrade(&source);
            }
        }
        Ok(copies)
    }

    pub fn bytes(&self) -> usize {
        self.buffers
            .values()
            .map(|buffer| buffer.values.len() * std::mem::size_of::<u32>())
            .sum()
    }
}

pub(super) enum IndexArgument<'a, D: CudaDType> {
    Exact {
        values: &'a CudaSlice<u32>,
        maximum: Option<u32>,
    },
    Floating(&'a Arc<CudaSlice<D::HostType>>),
}

impl<D: CudaDType> IndexArgument<'_, D> {
    pub fn len(&self) -> usize {
        match self {
            Self::Exact { values, .. } => values.len(),
            Self::Floating(values) => values.len(),
        }
    }
}

impl<D: CudaDType> PtxExecutor<D> {
    pub(super) fn index_argument(&self, node: NodeIndex) -> Result<IndexArgument<'_, D>> {
        if let Some(buffer) = self.exact_indices.buffers.get(&node) {
            Ok(IndexArgument::Exact {
                values: &buffer.values,
                maximum: buffer.maximum,
            })
        } else {
            Ok(IndexArgument::Floating(
                self.values
                    .get(&node)
                    .context("indexed argument not computed")?,
            ))
        }
    }

    pub(super) fn validate_index_argument(
        &self,
        argument: &IndexArgument<'_, D>,
        upper: usize,
        operation: &str,
    ) -> Result<()> {
        match argument {
            IndexArgument::Exact { maximum, .. } => {
                if let Some(value) = maximum {
                    crate::tensor::indices::checked_index(*value, upper, operation)?;
                }
                Ok(())
            }
            IndexArgument::Floating(values) => self.validate_indices(values, upper, operation),
        }
    }
}

#[cfg(test)]
mod tests;
