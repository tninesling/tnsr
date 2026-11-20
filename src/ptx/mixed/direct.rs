//! Opt-in resident master leases and bounded address-keyed graph variants.
//! Active leases preserve source diagnostics until the next complete refresh.
use super::*;
use std::collections::VecDeque;
#[path = "direct_layout.rs"]
mod layout;
use layout::DirectLayout;

pub(super) type MasterLeases = HashMap<NodeIndex, Arc<CudaSlice<f32>>>;
const CAPTURE_VARIANTS: usize = 2;
struct Variant {
    key: Vec<u64>,
    captured: Captured,
}
pub(super) struct DirectBindings {
    // Inactive graphs own no masters. Destroy them before dropping active leases.
    inactive: VecDeque<Variant>,
    active: MasterLeases,
    active_shadows: HashMap<NodeIndex, ShadowBinding>,
    active_addresses: HashMap<NodeIndex, Binding>,
    key: Vec<u64>,
    layout: DirectLayout,
    replay: bool,
    pub capture_hits: usize,
    pub capture_misses: usize,
}
pub(super) struct DirectFrame<'a> {
    pub kernels: &'a mut Vec<Kernel>,
    pub values: &'a mut HashMap<NodeIndex, Value>,
    pub captured: &'a mut Option<Captured>,
    pub readbacks: &'a mut HashMap<NodeIndex, Readback>,
}
fn bindings(
    stream: &Arc<CudaStream>,
    parameters: &MasterLeases,
) -> Result<HashMap<NodeIndex, Binding>> {
    parameters
        .iter()
        .map(|(&node, source)| {
            ensure!(
                Arc::ptr_eq(source.context(), stream.context()),
                "direct master context mismatch"
            );
            Ok((
                node,
                Binding {
                    pointer: source.device_ptr(stream).0,
                    len: source.len(),
                    dtype: DType::F32,
                },
            ))
        })
        .collect()
}
impl DirectBindings {
    /// Call after final allocation remapping and before first warmup/capture.
    pub fn new_with_shadows(
        stream: &Arc<CudaStream>,
        parameters: &MasterLeases,
        kernels: &[Kernel],
        values: &HashMap<NodeIndex, Value>,
        replay: bool,
        shadows: &HashMap<NodeIndex, ShadowBinding>,
        private: &[Binding],
    ) -> Result<Self> {
        let sources = typed_bindings(stream, parameters, shadows)?;
        let layout = DirectLayout::new_with_writes(
            &sources,
            &kernels
                .iter()
                .map(|k| k.pointers.clone())
                .collect::<Vec<_>>(),
            &kernels
                .iter()
                .map(|k| k.zero_before.map(|b| b.pointer))
                .collect::<Vec<_>>(),
            &values.iter().map(|(&node, v)| (node, v.binding)).collect(),
            private,
        )?;
        let key = layout.rebind(&sources)?.key;
        Ok(Self {
            inactive: VecDeque::new(),
            active: parameters.clone(),
            active_shadows: shadows.clone(),
            active_addresses: sources,
            key,
            layout,
            replay: replay && !kernels.is_empty(),
            capture_hits: 0,
            capture_misses: usize::from(replay && !kernels.is_empty()),
        })
    }
    pub fn has_shadows(&self) -> bool {
        !self.active_shadows.is_empty()
    }
    pub fn variants(&self, active_capture: bool) -> usize {
        self.inactive.len() + usize::from(active_capture)
    }

    /// Caller holds the common stream gate. All fallible validation/capture work
    /// precedes the commit. Retired leases must be processed outside that gate.
    pub fn refresh_with_shadows(
        &mut self,
        stream: &Arc<CudaStream>,
        parameters: &MasterLeases,
        frame: DirectFrame<'_>,
        shadows: &HashMap<NodeIndex, ShadowBinding>,
    ) -> Result<RetiredShadowBindings> {
        self.refresh_using_with_shadows(stream, parameters, frame, shadows, capture)
    }

    #[cfg(test)]
    fn refresh_using(
        &mut self,
        stream: &Arc<CudaStream>,
        parameters: &MasterLeases,
        frame: DirectFrame<'_>,
        make_capture: impl FnOnce(&Arc<CudaStream>, &[Kernel]) -> Result<Captured>,
    ) -> Result<MasterLeases> {
        Ok(self
            .refresh_using_with_shadows(stream, parameters, frame, &HashMap::new(), make_capture)?
            .masters)
    }

    fn refresh_using_with_shadows(
        &mut self,
        stream: &Arc<CudaStream>,
        parameters: &MasterLeases,
        frame: DirectFrame<'_>,
        shadows: &HashMap<NodeIndex, ShadowBinding>,
        make_capture: impl FnOnce(&Arc<CudaStream>, &[Kernel]) -> Result<Captured>,
    ) -> Result<RetiredShadowBindings> {
        ensure!(
            shadows.len() == self.active_shadows.len(),
            "direct shadow set changed"
        );
        for (node, current) in &self.active_shadows {
            let next = shadows.get(node).context("direct shadow missing")?;
            ensure!(
                next.parameter_node == current.parameter_node
                    && next.parameter_id == current.parameter_id
                    && Arc::ptr_eq(&next.storage, &current.storage),
                "direct shadow ownership changed"
            );
        }
        let sources = typed_bindings(stream, parameters, shadows)?;
        let plan = self.layout.rebind(&sources)?;
        // A matching numeric address must still name the active typed Arc,
        // even if some other source changed and the complete graph key differs.
        for (node, source) in parameters {
            if sources[node].pointer == self.active_addresses[node].pointer {
                ensure!(
                    Arc::ptr_eq(source, &self.active[node]),
                    "direct master identity changed at an active allocation address"
                );
            }
        }
        for (node, source) in shadows {
            if sources[node].pointer == self.active_addresses[node].pointer {
                ensure!(
                    Arc::ptr_eq(&source.shadow, &self.active_shadows[node].shadow),
                    "direct shadow identity changed at an active allocation address"
                );
            }
        }
        if self.key == plan.key {
            ensure!(
                parameters
                    .iter()
                    .all(|(node, source)| Arc::ptr_eq(source, &self.active[node])),
                "direct master identity changed at an active allocation address"
            );
            ensure!(
                shadows.iter().all(|(node, source)| Arc::ptr_eq(
                    &source.shadow,
                    &self.active_shadows[node].shadow
                )),
                "direct shadow identity changed at an active allocation address"
            );
            // Revisions validate freshness but never change graph compatibility.
            self.active_shadows = shadows.clone();
            return Ok(RetiredShadowBindings {
                masters: HashMap::new(),
                shadows: HashMap::new(),
            });
        }
        let mut kernels = frame.kernels.clone();
        for (kernel, argument, pointer) in plan.arguments {
            *kernels
                .get_mut(kernel)
                .and_then(|k| k.pointers.get_mut(argument))
                .context("direct kernel argument relocation missing")? = pointer;
        }
        for (node, _) in &plan.aliases {
            ensure!(
                frame.values.contains_key(node),
                "direct value relocation missing"
            );
        }
        let cached = self
            .inactive
            .iter()
            .position(|variant| variant.key == plan.key);
        let fresh = if self.replay && cached.is_none() {
            Some(make_capture(stream, &kernels)?)
        } else {
            None
        };
        // No eviction/free while prior raw-pointer launches could be in flight.
        // Normal two-address ping-pong uses the cache without this fence.
        if self.replay && cached.is_none() && self.inactive.len() + 2 > CAPTURE_VARIANTS {
            stream.synchronize()?;
        }
        // Commit: all remaining operations are infallible ownership changes.
        let next = if let Some(index) = cached {
            let captured = self
                .inactive
                .remove(index)
                .context("validated direct cache entry disappeared")?
                .captured;
            self.capture_hits += 1;
            Some(captured)
        } else {
            if self.replay {
                self.capture_misses += 1;
            }
            fresh
        };
        if let Some(captured) = frame.captured.take() {
            self.inactive.push_back(Variant {
                key: self.key.clone(),
                captured,
            });
        }
        while self.inactive.len() + usize::from(next.is_some()) > CAPTURE_VARIANTS {
            self.inactive.pop_front();
        }
        *frame.captured = next;
        *frame.kernels = kernels;
        for (node, pointer) in plan.aliases {
            // Validated above. Only master-dependent diagnostics need rebuilding.
            if let Some(value) = frame.values.get_mut(&node) {
                value.binding.pointer = pointer;
            }
            frame.readbacks.remove(&node);
        }
        self.key = plan.key;
        self.active_addresses = sources;
        let previous = std::mem::replace(&mut self.active, parameters.clone());
        let previous_shadows = std::mem::replace(&mut self.active_shadows, shadows.clone());
        // Drop duplicate graph-node aliases before owner attempts Arc::try_unwrap.
        let mut retired = HashMap::new();
        let mut seen = std::collections::HashSet::new();
        for (node, source) in previous {
            if !Arc::ptr_eq(&source, &parameters[&node]) && seen.insert(Arc::as_ptr(&source)) {
                retired.insert(node, source);
            }
        }
        let mut retired_shadows = HashMap::new();
        for (node, source) in previous_shadows {
            if !Arc::ptr_eq(&source.shadow, &shadows[&node].shadow) {
                retired_shadows
                    .entry(source.parameter_id)
                    .or_insert(source.shadow);
            }
        }
        Ok(RetiredShadowBindings {
            masters: retired,
            shadows: retired_shadows,
        })
    }
}

fn typed_bindings(
    stream: &Arc<CudaStream>,
    parameters: &MasterLeases,
    shadows: &HashMap<NodeIndex, ShadowBinding>,
) -> Result<HashMap<NodeIndex, Binding>> {
    let mut sources = bindings(stream, parameters)?;
    let mut owners: HashMap<usize, &ShadowBinding> = HashMap::new();
    for (&node, shadow) in shadows {
        super::shadows::validate_binding(stream, parameters, shadow)?;
        if let Some(previous) = owners.insert(shadow.parameter_id, shadow) {
            ensure!(
                Arc::ptr_eq(&previous.storage, &shadow.storage)
                    && previous.revision == shadow.revision
                    && Weak::ptr_eq(&previous.master, &shadow.master)
                    && Arc::ptr_eq(&previous.shadow, &shadow.shadow),
                "duplicate parameter shadows disagree"
            );
        }
        ensure!(
            !sources.contains_key(&node),
            "shadow cast collides with F32 source"
        );
        sources.insert(
            node,
            Binding {
                pointer: shadow.shadow.device_ptr(stream).0,
                len: shadow.shadow.len(),
                dtype: DType::BF16,
            },
        );
    }
    Ok(sources)
}
fn capture(stream: &Arc<CudaStream>, kernels: &[Kernel]) -> Result<Captured> {
    stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
    let launched = kernels.iter().try_for_each(|kernel| kernel.run(stream));
    // Always end capture, including failed launches, so the stream can recover.
    let ended = stream.end_capture(
        sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
    );
    launched?;
    Ok(Captured {
        graph: Some(ended?.context("empty direct binding capture")?),
        stream: stream.clone(),
    })
}

#[cfg(test)]
#[path = "direct_tests.rs"]
mod tests;
