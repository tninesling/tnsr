//! Private successful-submission freshness and lifetime-bound gradient borrows.
use super::*;
use cudarc::driver::CudaView;
use half::bf16;

#[derive(Clone)]
pub(crate) struct CompletedRunStamp {
    identity: Weak<()>,
    epoch: u64,
    upload: u64,
}
pub(super) struct Freshness {
    identity: Arc<()>,
    epoch: Option<u64>,
    upload: Option<u64>,
    completed: Option<(u64, u64)>,
}
impl Default for Freshness {
    fn default() -> Self {
        Self {
            identity: Arc::new(()),
            epoch: Some(0),
            upload: None,
            completed: None,
        }
    }
}
impl Freshness {
    /// Invalidate before validation, locks or any possibly partial device work.
    /// Overflow disables borrowing permanently; it never wraps into an old stamp.
    pub fn dirty(&mut self) {
        self.completed = None;
        self.epoch = self.epoch.and_then(|e| e.checked_add(1));
    }
    pub fn uploaded(&mut self) {
        self.upload = self.epoch;
    }
    pub fn ran(&mut self) {
        self.completed = self.epoch.zip(self.upload);
    }
    pub fn stamp(&self) -> Option<CompletedRunStamp> {
        let (epoch, upload) = self.completed?;
        Some(CompletedRunStamp {
            identity: Arc::downgrade(&self.identity),
            epoch,
            upload,
        })
    }
    pub fn matches(&self, stamp: &CompletedRunStamp) -> bool {
        Weak::ptr_eq(&stamp.identity, &Arc::downgrade(&self.identity))
            && self.epoch == Some(stamp.epoch)
            && self.upload == Some(stamp.upload)
            && self.completed == Some((stamp.epoch, stamp.upload))
    }
}
#[derive(Default)]
pub(super) struct PrefixCache(
    RefCell<HashMap<(NodeIndex, usize), PrefixProof>>,
    std::cell::Cell<usize>,
    std::cell::Cell<usize>,
);
struct PrefixProof {
    owner: Option<usize>,
    binding: Binding,
    view: VirtualTensor,
    source_shape: Vec<usize>,
}
impl PrefixProof {
    fn matches(&self, value: &Value) -> bool {
        self.binding.pointer == value.binding.pointer
            && self.binding.len == value.binding.len
            && self.binding.dtype == value.binding.dtype
            && self.view == value.view
            && self.source_shape == value.source_shape
    }
}
impl PrefixCache {
    pub fn clear(&self) {
        self.0.borrow_mut().clear();
    }
}
fn logical_prefix(
    view: &VirtualTensor,
    source_shape: &[usize],
    binding_len: usize,
    len: usize,
) -> bool {
    if len == 0
        || binding_len != len
        || view.dtype != DType::BF16
        || view.predicate.is_some()
        || view.out_of_bounds.is_some()
    {
        return false;
    }
    let volume = |s: &[usize]| s.iter().try_fold(1usize, |n, &d| n.checked_mul(d));
    if volume(source_shape) != Some(len) || volume(&view.shape) != Some(len) {
        return false;
    }
    let Ok(expected) = VirtualTensor::identity(view.source, source_shape.to_vec(), DType::BF16)
        .reshape(view.shape.clone())
    else {
        return false;
    };
    match (
        expected.access.normalize_in_domain(&view.shape),
        view.access.clone().normalize_in_domain(&view.shape),
    ) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    }
}
fn owner_prefix(base: u64, owner_len: usize, pointer: u64, len: usize) -> bool {
    base != 0
        && base == pointer
        && base.is_multiple_of(16)
        && len > 0
        && owner_len >= len
        && len
            .checked_mul(2)
            .and_then(|n| u64::try_from(n).ok())
            .and_then(|n| base.checked_add(n))
            .is_some()
}
impl MixedExecutor {
    #[cfg(test)]
    pub(crate) fn borrowed_gradient_proof_counts(&self) -> (usize, usize) {
        (
            self.borrowed_prefixes.1.get(),
            self.borrowed_prefixes.2.get(),
        )
    }
    /// Metadata-only test census; no address or allocation lease escapes.
    #[cfg(test)]
    pub(crate) fn borrowed_gradient_cached_owner_extent(
        &self,
        node: NodeIndex,
        len: usize,
    ) -> Option<(usize, bool)> {
        let value = self.values.get(&node)?;
        let proofs = self.borrowed_prefixes.0.borrow();
        let proof = proofs.get(&(node, len))?;
        if !proof.matches(value) {
            return None;
        }
        let Buffer::BF16(owner) = self.buffers.get(proof.owner?)? else {
            return None;
        };
        Some((
            owner.len(),
            owner.device_ptr(&self.stream).0 == value.binding.pointer,
        ))
    }
    pub(crate) fn borrowed_gradient_stamp(&self) -> Option<CompletedRunStamp> {
        self.freshness.stamp()
    }
    pub(crate) fn matches_borrowed_gradient_stamp(&self, stamp: &CompletedRunStamp) -> bool {
        self.freshness.matches(stamp)
    }
    /// No address escapes: the typed zero-offset prefix keeps the real owner
    /// borrowed. A larger pooled allocation is allowed; logical volume is exact.
    pub(crate) fn borrowed_bf16_gradient(
        &self,
        node: NodeIndex,
        len: usize,
    ) -> Option<CudaView<'_, bf16>> {
        let value = self.values.get(&node)?;
        let origin = self.index_graph.graph.node_weight(value.view.source)?;
        if value.binding.dtype != DType::BF16
            || value.view.dtype != DType::BF16
            || value.binding.len != len
            || len == 0
            || value.binding.pointer == 0
            || !value.binding.pointer.is_multiple_of(16)
            || value.view.predicate.is_some()
            || value.view.out_of_bounds.is_some()
            || matches!(origin, TensorGraphNode::EmbeddingBackward { .. })
        {
            return None;
        }
        // Buffers/indices are immutable after construction. Reuse/remapping
        // explicitly clears this cache; direct refresh changes only F32 sources.
        // A hit compares current proof metadata, without normalizing maps or
        // asking CUDA for every allocation address on every update.
        if let Some(proof) = self.borrowed_prefixes.0.borrow().get(&(node, len))
            && proof.matches(value)
        {
            let index = proof.owner?; // Exact-metadata negative proof hit.
            if let Some(Buffer::BF16(owner)) = self.buffers.get(index)
                && owner.len() >= len
                && Arc::ptr_eq(owner.context(), self.stream.context())
            {
                return Some(owner.slice(0..len));
            }
        }
        let remember = |owner| {
            self.borrowed_prefixes.0.borrow_mut().insert(
                (node, len),
                PrefixProof {
                    owner,
                    binding: value.binding,
                    view: value.view.clone(),
                    source_shape: value.source_shape.clone(),
                },
            );
        };
        self.borrowed_prefixes
            .1
            .set(self.borrowed_prefixes.1.get().saturating_add(1));
        if !logical_prefix(&value.view, &value.source_shape, value.binding.len, len) {
            remember(None);
            return None;
        }
        self.borrowed_prefixes
            .2
            .set(self.borrowed_prefixes.2.get().saturating_add(1));
        let mut owners = self.buffers.iter().enumerate().filter_map(|(index, b)| {
            let Buffer::BF16(owner) = b else {
                return None;
            };
            (owner_prefix(
                owner.device_ptr(&self.stream).0,
                owner.len(),
                value.binding.pointer,
                len,
            ) && Arc::ptr_eq(owner.context(), self.stream.context()))
            .then_some((index, owner))
        });
        let Some((index, owner)) = owners.next() else {
            remember(None);
            return None;
        };
        if owners.next().is_some() {
            remember(None);
            return None;
        }
        remember(Some(index));
        Some(owner.slice(0..len))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn borrowed_gradient_cache_rejects_binding_or_map_changes() {
        let node = NodeIndex::new(0);
        let value = Value {
            binding: Binding {
                pointer: 256,
                len: 15,
                dtype: DType::BF16,
            },
            view: VirtualTensor::identity(node, vec![3, 5], DType::BF16),
            source_shape: vec![3, 5],
            chain: Vec::new(),
        };
        let proof = PrefixProof {
            owner: Some(4),
            binding: value.binding,
            view: value.view.clone(),
            source_shape: value.source_shape.clone(),
        };
        assert!(proof.matches(&value));
        let mut changed = value.clone();
        changed.binding.pointer = 512;
        assert!(!proof.matches(&changed));
        changed = value.clone();
        changed.view = changed.view.permute(&[1, 0]).unwrap();
        assert!(!proof.matches(&changed));
        changed = value.clone();
        changed.source_shape = vec![15];
        assert!(!proof.matches(&changed));
        changed = value.clone();
        changed.binding.len -= 1;
        assert!(!proof.matches(&changed));
        let cache = PrefixCache::default();
        cache.0.borrow_mut().insert((node, 15), proof);
        assert_eq!(cache.0.borrow().len(), 1);
        cache.clear();
        assert!(cache.0.borrow().is_empty());
    }
    #[test]
    fn borrowed_gradient_logical_maps_and_pooled_prefix_bounds() {
        let node = NodeIndex::new(0);
        let identity = VirtualTensor::identity(node, vec![3, 5], DType::BF16);
        assert!(logical_prefix(&identity, &[3, 5], 15, 15));
        assert!(logical_prefix(
            &identity.clone().reshape(vec![15]).unwrap(),
            &[3, 5],
            15,
            15
        ));
        assert!(!logical_prefix(
            &identity.clone().permute(&[1, 0]).unwrap(),
            &[3, 5],
            15,
            15
        ));
        assert!(!logical_prefix(&identity, &[3, 5], 16, 15));
        assert!(!logical_prefix(&identity, &[15], 15, 14));
        assert!(owner_prefix(256, 15, 256, 15));
        assert!(owner_prefix(256, 1024, 256, 15));
        for (base, capacity, ptr, len) in [
            (256, 15, 258, 14),
            (256, 14, 256, 15),
            (0, 15, 0, 15),
            (256, 15, 256, 0),
            (u64::MAX - 15, 16, u64::MAX - 15, 16),
        ] {
            assert!(!owner_prefix(base, capacity, ptr, len));
        }
    }
    #[test]
    fn borrowed_gradient_freshness_attempts_identity_and_overflow() {
        let mut a = Freshness::default();
        assert!(a.stamp().is_none());
        a.dirty();
        a.uploaded();
        a.dirty();
        a.ran();
        let first = a.stamp().unwrap();
        assert!(a.matches(&first));
        let mut b = Freshness::default();
        b.dirty();
        b.uploaded();
        b.dirty();
        b.ran();
        assert!(!b.matches(&first));
        // Even a failed mutation attempt leaves no usable completion stamp.
        a.dirty();
        assert!(a.stamp().is_none());
        assert!(!a.matches(&first));
        a.dirty();
        a.ran();
        assert!(!a.matches(&first));
        a.epoch = Some(u64::MAX);
        a.dirty();
        a.uploaded();
        a.ran();
        assert!(a.stamp().is_none());
        drop(a);
        assert!(first.identity.upgrade().is_none());
        assert!(!b.matches(&first));
    }
}

#[cfg(test)]
mod gpu_tests;
