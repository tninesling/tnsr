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
}
impl<D: CudaDType> Default for SourceCache<D> {
    fn default() -> Self {
        Self {
            parameters: RefCell::new(HashMap::new()),
            constants: RefCell::new(HashMap::new()),
        }
    }
}
impl<D: CudaDType> SourceCache<D> {
    pub fn prune(&self) {
        self.parameters
            .borrow_mut()
            .retain(|_, entry| entry.source.strong_count() > 0);
        self.constants
            .borrow_mut()
            .retain(|_, entry| entry.source.strong_count() > 0);
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
