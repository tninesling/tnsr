//! Immutable constants and revision-checked parameter uploads.
use super::types::CudaDType;
use crate::graph::TensorGraphNode;
use crate::tensor::ParameterStorage;
use anyhow::{Context, Result};
use cudarc::driver::{CudaContext, CudaSlice};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, Weak};

pub(super) struct SourceBinding<D: CudaDType> {
    pub value: Arc<CudaSlice<D::HostType>>,
    pub uploaded: bool,
}

struct ParameterEntry<D: CudaDType> {
    source: Weak<ParameterStorage<D::HostType>>,
    revision: u64,
    value: Arc<CudaSlice<D::HostType>>,
}
struct ConstantEntry<D: CudaDType> {
    source: Weak<Vec<D::HostType>>,
    value: Arc<CudaSlice<D::HostType>>,
}
pub(super) struct SourceCache<D: CudaDType> {
    parameters: RefCell<HashMap<usize, ParameterEntry<D>>>,
    constants: RefCell<HashMap<usize, ConstantEntry<D>>>,
    index_facts: RefCell<HashMap<usize, IndexEntry<D>>>,
}

struct IndexEntry<D: CudaDType> {
    allocation: Weak<CudaSlice<D::HostType>>,
    facts: IndexFacts,
}

struct IndexFacts {
    maximum: Option<usize>,
    error: Option<String>,
}
impl<D: CudaDType> Default for SourceCache<D> {
    fn default() -> Self {
        Self {
            parameters: RefCell::new(HashMap::new()),
            constants: RefCell::new(HashMap::new()),
            index_facts: RefCell::new(HashMap::new()),
        }
    }
}
impl<D: CudaDType> SourceCache<D> {
    pub fn prune(&self) {
        self.index_facts.borrow_mut().clear();
        self.parameters
            .borrow_mut()
            .retain(|_, entry| entry.source.strong_count() > 0);
        self.constants
            .borrow_mut()
            .retain(|_, entry| entry.source.strong_count() > 0);
    }

    /// Host-provided inputs can be validated without reading them back. Weak
    /// allocation identity prevents stale facts when a pooled pointer is reused.
    pub fn record_input(&self, device: &Arc<CudaSlice<D::HostType>>, values: &[D::HostType]) {
        use num_traits::ToPrimitive;
        let mut facts = IndexFacts {
            maximum: None,
            error: None,
        };
        for (position, value) in values.iter().enumerate() {
            let Some(value) = value.to_f32() else {
                facts.error = Some("index cannot be represented as f32".into());
                break;
            };
            let index = value as usize;
            let reason = if !value.is_finite() {
                Some("must be finite")
            } else if value < 0.0 {
                Some("must be non-negative")
            } else if value.fract() != 0.0 {
                Some("must be an integer")
            } else if index as f32 != value {
                Some("cannot be represented as usize")
            } else {
                None
            };
            if let Some(reason) = reason {
                facts.error = Some(format!(
                    "index at position {position} {reason}, got {value}"
                ));
                break;
            }
            facts.maximum = Some(facts.maximum.map_or(index, |old| old.max(index)));
        }
        self.index_facts.borrow_mut().insert(
            Arc::as_ptr(device) as usize,
            IndexEntry {
                allocation: Arc::downgrade(device),
                facts,
            },
        );
    }

    pub fn validate_input(
        &self,
        device: &Arc<CudaSlice<D::HostType>>,
        upper: usize,
        operation: &str,
    ) -> Option<Result<()>> {
        let key = Arc::as_ptr(device) as usize;
        if !self.index_facts.borrow().contains_key(&key) {
            // Constant indices are also known on the host. Derive facts lazily
            // so large non-index constants are never scanned for this purpose.
            let host = self
                .constants
                .borrow()
                .values()
                .find(|entry| Arc::ptr_eq(&entry.value, device))
                .and_then(|entry| entry.source.upgrade());
            if let Some(host) = host {
                self.record_input(device, host.as_slice());
            }
        }
        let facts = self.index_facts.borrow();
        let IndexEntry { allocation, facts } = facts.get(&(Arc::as_ptr(device) as usize))?;
        if !allocation.ptr_eq(&Arc::downgrade(device)) || allocation.strong_count() == 0 {
            return None;
        }
        Some((|| {
            if let Some(error) = &facts.error {
                anyhow::bail!("{operation} {error}");
            }
            anyhow::ensure!(
                facts.maximum.is_none_or(|index| index < upper),
                "{operation} index is out of range: {:?} >= {upper}",
                facts.maximum
            );
            Ok(())
        })())
    }

    /// Return the source buffer and whether this call uploaded host data.
    pub fn get(
        &self,
        node: &TensorGraphNode<D::HostType>,
        device: &Arc<CudaContext>,
        allocate: impl FnOnce(usize) -> Result<CudaSlice<D::HostType>>,
    ) -> Result<Option<SourceBinding<D>>> {
        match node {
            TensorGraphNode::Parameter { id, data, .. } => {
                let revision = data.revision()?;
                if let Some(entry) = self.parameters.borrow().get(id)
                    && entry.revision == revision
                    && entry.source.ptr_eq(&Arc::downgrade(data))
                {
                    return Ok(Some(SourceBinding {
                        value: entry.value.clone(),
                        uploaded: false,
                    }));
                }
                let host = data.lock()?;
                let revision = host.revision();
                let mut value = allocate(host.len())?;
                if !host.is_empty() {
                    device
                        .default_stream()
                        .memcpy_htod(host.as_slice(), &mut value)
                        .context("upload parameter")?;
                }
                let value = Arc::new(value);
                self.parameters.borrow_mut().insert(
                    *id,
                    ParameterEntry {
                        source: Arc::downgrade(data),
                        revision,
                        value: value.clone(),
                    },
                );
                Ok(Some(SourceBinding {
                    value,
                    uploaded: !host.is_empty(),
                }))
            }
            TensorGraphNode::Constant { data, .. } => {
                let key = Arc::as_ptr(data) as usize;
                if let Some(entry) = self.constants.borrow().get(&key)
                    && entry.source.ptr_eq(&Arc::downgrade(data))
                {
                    return Ok(Some(SourceBinding {
                        value: entry.value.clone(),
                        uploaded: false,
                    }));
                }
                let mut value = allocate(data.len())?;
                if !data.is_empty() {
                    device
                        .default_stream()
                        .memcpy_htod(data.as_slice(), &mut value)
                        .context("upload constant")?;
                }
                let value = Arc::new(value);
                self.constants.borrow_mut().insert(
                    key,
                    ConstantEntry {
                        source: Arc::downgrade(data),
                        value: value.clone(),
                    },
                );
                Ok(Some(SourceBinding {
                    value,
                    uploaded: !data.is_empty(),
                }))
            }
            _ => Ok(None),
        }
    }
}
