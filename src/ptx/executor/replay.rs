//! Private-stream model replay. Host staging/validation and optimizer publication
//! stay outside capture; a warmed executor owns every captured allocation/module.
use super::*;
use crate::graph::NodeIndex;
use cudarc::driver::{CudaGraph, sys};
use std::mem::ManuallyDrop;
use std::sync::Weak;

struct ReplayGraph {
    graph: ManuallyDrop<CudaGraph>,
    stream: Arc<CudaStream>,
}

impl ReplayGraph {
    fn new(graph: CudaGraph, stream: Arc<CudaStream>) -> Self {
        Self {
            graph: ManuallyDrop::new(graph),
            stream,
        }
    }
    fn launch(&mut self) -> Result<()> {
        self.graph.launch().map_err(Into::into)
    }
}

impl Drop for ReplayGraph {
    fn drop(&mut self) {
        if self.stream.context().bind_to_thread().is_ok() {
            // SAFETY: this is the sole owner, on the correct current context,
            // and this destructor runs before any graph-referenced allocation.
            unsafe { ManuallyDrop::drop(&mut self.graph) };
        }
        // A lost/unbindable context cannot safely destroy its graph; retain the
        // driver handle in that error case rather than destroying on another context.
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PtxReplayMetrics {
    pub active: bool,
    pub captures: usize,
    pub graph_launches: usize,
    pub warmup_executions: usize,
    pub fallbacks: usize,
    pub stable_binding_bytes: usize,
    pub last_fallback: Option<String>,
}

pub(super) struct IndexRequirement {
    pub key: usize,
    pub upper: usize,
    pub operation: String,
}

struct StableParameter<D: CudaDType> {
    target: Arc<CudaSlice<D::HostType>>,
    // Weak control-block identity prevents address reuse without retaining an
    // obsolete generation of GPU weights for every cached shape bucket.
    source: Weak<CudaSlice<D::HostType>>,
}

pub(super) struct ReplayState<D: CudaDType> {
    // Field order destroys graph references before releasing their buffers.
    graph: Option<ReplayGraph>,
    inputs: HashMap<NodeIndex, Arc<CudaSlice<D::HostType>>>,
    parameters: HashMap<NodeIndex, StableParameter<D>>,
    constants: Vec<(NodeIndex, Arc<Vec<D::HostType>>)>,
    requirements: Vec<IndexRequirement>,
    metrics: PtxExecutionMetrics,
}

impl<D: CudaDType> ReplayState<D> {
    fn matches_constants<G>(&self, graph: &TensorGraph<D::HostType, G>) -> bool {
        self.constants.iter().all(|(node, data)| {
            matches!(graph.graph.node_weight(*node), Some(TensorGraphNode::Constant { data: current, .. }) if Arc::ptr_eq(current, data))
        })
    }

    fn aliases_bound_source(&self) -> bool {
        self.inputs
            .values()
            .chain(self.parameters.values().map(|p| &p.target))
            .any(|value| Arc::strong_count(value) != 1)
    }
}

impl<D: CudaDType> PtxExecutor<D> {
    pub(crate) fn enable_graph_replay(&mut self, enabled: bool) {
        self.invalidate_replay();
        self.replay_enabled = enabled;
    }

    pub fn replay_metrics(&self) -> &PtxReplayMetrics {
        &self.replay_metrics
    }

    pub(super) fn invalidate_replay(&mut self) {
        self.replay = None;
        self.replay_blocked = false;
        self.replay_metrics.stable_binding_bytes = 0;
        self.replay_metrics.active = false;
    }

    fn block_replay(&mut self, reason: String) {
        self.replay_blocked = true;
        self.replay_metrics.fallbacks += 1;
        self.replay_metrics.last_fallback = Some(reason);
        self.replay_metrics.stable_binding_bytes = 0;
        self.replay_metrics.active = false;
    }

    fn new_replay_state<G>(
        &self,
        graph: &TensorGraph<D::HostType, G>,
        bindings: &HashMap<NodeIndex, Arc<CudaSlice<D::HostType>>>,
    ) -> Result<ReplayState<D>> {
        let mut state = ReplayState {
            graph: None,
            inputs: HashMap::new(),
            parameters: HashMap::new(),
            constants: Vec::new(),
            requirements: Vec::new(),
            metrics: PtxExecutionMetrics::default(),
        };
        let allocate = |len| -> Result<Arc<CudaSlice<D::HostType>>> {
            Ok(Arc::new(if len == 0 {
                self.stream.null()?
            } else {
                self.stream.alloc_zeros(len)?
            }))
        };
        for node in graph.graph.node_indices() {
            match &graph[node] {
                TensorGraphNode::Input { shape, .. } => {
                    anyhow::ensure!(
                        !bindings.contains_key(&node),
                        "graph replay requires host-fed inputs"
                    );
                    state.inputs.insert(
                        node,
                        allocate(checked_element_count(shape, "replay input")?)?,
                    );
                }
                TensorGraphNode::Parameter { .. } => {
                    let source = bindings
                        .get(&node)
                        .context("replay parameter binding missing")?;
                    state.parameters.insert(
                        node,
                        StableParameter {
                            target: allocate(source.len())?,
                            source: Arc::downgrade(source),
                        },
                    );
                }
                TensorGraphNode::Constant { data, .. } => {
                    state.constants.push((node, data.clone()))
                }
                _ => {}
            }
        }
        // The first copy is unconditional; targets initially contain zeros.
        for (node, parameter) in &mut state.parameters {
            self.stream.memcpy_dtod(
                bindings[node].as_ref(),
                Arc::get_mut(&mut parameter.target).context("new replay parameter aliased")?,
            )?;
        }
        Ok(state)
    }

    fn stage_replay<G>(
        &mut self,
        state: &mut ReplayState<D>,
        graph: &TensorGraph<D::HostType, G>,
        inputs: &HashMap<String, Vec<D::HostType>>,
        bindings: &HashMap<NodeIndex, Arc<CudaSlice<D::HostType>>>,
    ) -> Result<usize> {
        // Drop input-fact Weak references before Arc::get_mut. Facts for this
        // frame are rebuilt only after every stable input has been overwritten.
        self.sources.prune();
        for (&node, target) in &state.inputs {
            let TensorGraphNode::Input { name, .. } = &graph[node] else {
                anyhow::bail!("replay input structure changed");
            };
            let host = inputs
                .get::<str>(name)
                .context("replay host input missing")?;
            anyhow::ensure!(
                host.len() == target.len(),
                "replay input '{name}' length mismatch"
            );
        }
        let mut copies = 0;
        for (&node, parameter) in &mut state.parameters {
            let current = bindings.get(&node).context("replay parameter missing")?;
            if !parameter.source.ptr_eq(&Arc::downgrade(current)) {
                self.stream.memcpy_dtod(
                    current.as_ref(),
                    Arc::get_mut(&mut parameter.target)
                        .context("replay parameter escapes as an output")?,
                )?;
                parameter.source = Arc::downgrade(current);
                copies += 1;
            }
        }
        for (&node, target) in &mut state.inputs {
            let TensorGraphNode::Input { name, .. } = &graph[node] else {
                unreachable!()
            };
            let host = &inputs[&**name];
            if !host.is_empty() {
                self.stream.memcpy_htod(
                    host.as_slice(),
                    Arc::get_mut(target).context("replay input escapes as an output")?,
                )?;
                copies += 1;
            }
            self.sources.record_input(target, host);
        }
        Ok(copies)
    }

    fn run_replay_frame<G>(
        &mut self,
        state: &ReplayState<D>,
        graph: &TensorGraph<D::HostType, G>,
        inputs: &HashMap<String, Vec<D::HostType>>,
    ) -> Result<()> {
        self.bindings = state
            .inputs
            .iter()
            .map(|(&node, value)| (node, value.clone()))
            .chain(
                state
                    .parameters
                    .iter()
                    .map(|(&node, parameter)| (node, parameter.target.clone())),
            )
            .collect();
        self.replay_frame_active = true;
        let result = self.execute_compiled_impl(graph, inputs.clone(), false);
        self.replay_frame_active = false;
        self.bindings.clear();
        result.map(|_| ())
    }

    pub(super) fn run_replay<G>(
        &mut self,
        graph: &TensorGraph<D::HostType, G>,
        inputs: &HashMap<String, Vec<D::HostType>>,
        bindings: &HashMap<NodeIndex, Arc<CudaSlice<D::HostType>>>,
    ) -> Result<bool> {
        if self
            .replay
            .as_ref()
            .is_some_and(|state| !state.matches_constants(graph))
        {
            self.invalidate_replay();
        }
        let (mut state, initial_copies) = if let Some(state) = self.replay.take() {
            (state, 0)
        } else {
            match self.new_replay_state(graph, bindings) {
                Ok(state) => {
                    let copies = state.parameters.len();
                    (state, copies)
                }
                Err(error) => {
                    self.block_replay(error.to_string());
                    return Ok(false);
                }
            }
        };
        if state.graph.is_none() {
            // A previous invalid frame can retain source aliases until its
            // values are released. Do so before mutating stable input buffers.
            let old: Vec<_> = self.values.drain().map(|(_, value)| value).collect();
            for value in old {
                self.recycle_value(value);
            }
        }
        let staged = self.stage_replay(&mut state, graph, inputs, bindings);
        let copies = match staged {
            Ok(copies) => copies + initial_copies,
            Err(error) => {
                self.replay = Some(state);
                return Err(error);
            }
        };
        self.replay_metrics.stable_binding_bytes = state
            .inputs
            .values()
            .chain(state.parameters.values().map(|p| &p.target))
            .map(|value| value.len() * std::mem::size_of::<D::HostType>())
            .sum();

        if let Some(captured) = &mut state.graph {
            for requirement in &state.requirements {
                let result = self
                    .sources
                    .validate_key(requirement.key, requirement.upper, &requirement.operation)
                    .context("captured index source is no longer host-verifiable");
                if let Err(error) = result.and_then(|value| value) {
                    self.replay = Some(state);
                    return Err(error);
                }
            }
            captured.launch()?;
            self.replay_metrics.graph_launches += 1;
            self.execution_metrics = state.metrics.clone();
            self.execution_metrics.device_copies += copies;
            self.execution_metrics.graph_launches = 1;
            self.kernel_launches.set(state.metrics.kernel_launches);
            self.replay = Some(state);
            return Ok(true);
        }

        // First-use constant uploads can pin buffers which earlier operations
        // reused. A second uncaptured evaluation settles that workspace before
        // freezing allocation. These pure model evaluations never accumulate
        // owner gradients or update optimizer state.
        let mut warmup = PtxExecutionMetrics {
            device_copies: copies,
            ..Default::default()
        };
        for _ in 0..2 {
            if let Err(error) = self.run_replay_frame(&state, graph, inputs) {
                self.replay = Some(state);
                return Err(error);
            }
            self.replay_metrics.warmup_executions += 1;
            warmup.add_work(&self.execution_metrics);
        }
        if state.aliases_bound_source() {
            self.block_replay("a source binding escapes as a retained output".into());
            self.execution_metrics = warmup;
            self.kernel_launches
                .set(self.execution_metrics.kernel_launches);
            return Ok(true);
        }

        if let Err(error) = self
            .stream
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
        {
            self.block_replay(format!("begin CUDA capture: {error}"));
            self.pending_replay_work = warmup;
            return Ok(false);
        }
        self.capture_indices.borrow_mut().clear();
        self.capturing.set(true);
        self.allocation_frozen.set(true);
        let recorded = self.run_replay_frame(&state, graph, inputs);
        self.allocation_frozen.set(false);
        self.capturing.set(false);
        // Always end capture, even if recording returned an error. No CUDA work
        // from an unsuccessful capture has executed; the checked path runs once.
        // cudarc can defer errors from launch/event guards into the context.
        // Drain them before end_capture: otherwise its bind_to_thread check can
        // return early without actually ending an invalidated stream capture.
        let deferred = self.device.check_err().map_err(anyhow::Error::from);
        let recorded = recorded.and(deferred);
        let ended = self.stream.end_capture(
            sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        );
        match (recorded, ended) {
            (Ok(()), Ok(Some(captured))) => {
                state.metrics = self.execution_metrics.clone();
                state.requirements = self.capture_indices.take();
                let mut captured = ReplayGraph::new(captured, self.stream.clone());
                captured.launch()?;
                state.graph = Some(captured);
                self.replay_metrics.captures += 1;
                self.replay_metrics.active = true;
                self.replay_metrics.graph_launches += 1;
                self.execution_metrics.graph_launches = 1;
                self.execution_metrics.add_work(&warmup);
                self.kernel_launches
                    .set(self.execution_metrics.kernel_launches);
                self.replay = Some(state);
                Ok(true)
            }
            (recorded, ended) => {
                let reason = recorded
                    .err()
                    .map(|error| format!("{error:#}"))
                    .or_else(|| ended.err().map(|error| error.to_string()))
                    .unwrap_or_else(|| "empty CUDA capture".into());
                // `ended` (and any partial graph) has been destroyed before
                // this state/buffer set can be released or reused.
                self.block_replay(reason);
                self.pending_replay_work = warmup;
                Ok(false)
            }
        }
    }
}
