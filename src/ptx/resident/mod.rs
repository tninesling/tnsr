//! CUDA-resident f32 parameters, accumulated gradients and grouped Adam state.
mod batch;
mod small_accumulation_norm;
pub use small_accumulation_norm::SmallAccumulationNormBatchStatus;
mod borrowed_bf16;
pub use borrowed_bf16::BorrowedBf16AdamStatus;
mod bf16_shadow;
pub use bf16_shadow::Bf16MasterShadowStatus;
#[cfg(test)]
mod borrowed_bf16_golden;
mod config;
mod direct;
mod kernels;
mod launch;
mod mixed;
mod norm_partials;
mod reduction;
mod staging;
#[cfg(test)]
mod staging_tests;
mod vector_adam;
use super::types::{CudaDType, F32};
use super::{PtxExecutor, PtxReductionMode};
use crate::graph::{TensorGraph, TensorGraphNode, WithGrad};
use crate::tensor::{Parameter, ParameterStorage, Shape};
use crate::tile::VirtualTensor;
use anyhow::{Context, Result, ensure};
pub use config::{AdamGroup, AdamMoments, ResidentStep};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
pub use norm_partials::GradientNormPartialsStatus;
use reduction::Reduction;
use std::collections::{BTreeMap, HashMap};
use std::ops::Range;
use std::sync::{Arc, Mutex};
pub use vector_adam::{VectorBf16FirstAdamResources, VectorBf16FirstAdamStatus};

pub(crate) struct DeviceValue<D: CudaDType = F32> {
    pub storage: Arc<CudaSlice<D::HostType>>,
    pub view: VirtualTensor,
    pub source_shape: Shape,
}

#[cfg(test)]
#[derive(serde::Serialize)]
pub(crate) struct AdamDispatchCensus {
    pub parameter_id: usize,
    pub elements: usize,
    pub group: usize,
    pub batched: bool,
}

type AccumulationKey = (bool, bool, Vec<(VirtualTensor, Shape)>);

struct ResidentParameter {
    id: usize,
    storage: Arc<ParameterStorage<f32>>,
    revision: u64,
    weight: Arc<CudaSlice<f32>>,
    staged_weight: Option<CudaSlice<f32>>,
    range: Range<usize>,
    /// One health partial for each logical 256-element group, independent of
    /// standalone/batched launch geometry and parameter element alignment.
    health_range: Range<usize>,
    group: usize,
}

/// An explicit training owner. Parameters/moments and gradients stay on CUDA
/// between microbatches. Host reads synchronize through shared parameter storage.
/// Writers must be serialized; host changes during an update are rejected.
pub struct ResidentAdam {
    borrowed_bf16: borrowed_bf16::BorrowState,
    vector_bf16_first_adam: vector_adam::State,
    bf16_master_shadows: bf16_shadow::State,
    small_accumulation_norm: small_accumulation_norm::State,
    device: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    stream_gate: Arc<Mutex<()>>,
    readback_stream: Arc<CudaStream>,
    replay: bool,
    parameters: Vec<ResidentParameter>,
    groups: Vec<AdamGroup>,
    gradients: Arc<CudaSlice<f32>>,
    moment: CudaSlice<f32>,
    variance: CudaSlice<f32>,
    staging: staging::Staging,
    padded_len: usize,
    norm: Reduction,
    gradient_norm_partials: bool,
    norm_partial_workspace: Option<norm_partials::Workspace>,
    norm_partial_validity: norm_partials::Validity,
    norm_partial_attempt: bool,
    last_norm_partials_used: bool,
    health: Reduction,
    update_kernels: kernels::Kernels,
    update_ids: HashMap<usize, usize>,
    update_batch_size: usize,
    adam_256_threads: bool,
    batched_updates: Option<batch::Updates>,
    accumulate_kernels: HashMap<AccumulationKey, kernels::Kernels>,
    active: bool,
    first_accumulation_overwrite: bool,
    pending_zero: bool,
    accumulation_failed: bool,
    timestep: usize,
    last_scale: f32,
    kernel_launches: usize,
}

impl ResidentAdam {
    /// Inspect the plan actually used by a completed legacy update. This does
    /// not construct an alternative plan or expose device storage ownership.
    #[cfg(test)]
    pub(crate) fn actual_adam_dispatch_for_census(&self) -> Result<Vec<AdamDispatchCensus>> {
        ensure!(
            !self.active && self.timestep > 0,
            "dispatch census requires a completed resident step"
        );
        ensure!(
            self.update_batch_size <= 1 || self.batched_updates.is_some(),
            "completed step has no materialized batch dispatch plan"
        );
        Ok(self
            .parameters
            .iter()
            .enumerate()
            .map(|(index, parameter)| AdamDispatchCensus {
                parameter_id: parameter.id,
                elements: parameter.range.len(),
                group: parameter.group,
                batched: self
                    .batched_updates
                    .as_ref()
                    .is_some_and(|updates| updates.covers(index)),
            })
            .collect())
    }

    /// Register shared parameters once, with explicit optimizer group indices.
    /// Duplicate references to the same storage/group are deduplicated.
    pub fn new(parameters: Vec<(Parameter<f32>, usize)>, groups: Vec<AdamGroup>) -> Result<Self> {
        Self::initialize(parameters, groups, false)
    }

    /// Use a private stream and cache graph-safe repeated model executions.
    /// Unsupported graphs retain the checked uncaptured path.
    pub fn new_with_graph_replay(
        parameters: Vec<(Parameter<f32>, usize)>,
        groups: Vec<AdamGroup>,
    ) -> Result<Self> {
        Self::initialize(parameters, groups, true)
    }

    fn initialize(
        parameters: Vec<(Parameter<f32>, usize)>,
        groups: Vec<AdamGroup>,
        replay: bool,
    ) -> Result<Self> {
        ensure!(
            !parameters.is_empty() && !groups.is_empty(),
            "resident training requires parameters/groups"
        );
        for group in &groups {
            group.validate()?;
        }
        let device = CudaContext::new(0).context("initialize resident training CUDA device")?;
        let stream = if replay {
            device.new_stream()?
        } else {
            device.default_stream()
        };
        // Published parameter readers can run on another host thread while a
        // model capture is being recorded. Keep those copies off its stream.
        let readback_stream = if replay {
            device.new_stream()?
        } else {
            stream.clone()
        };
        let stream_gate = Arc::new(Mutex::new(()));
        if replay {
            // SAFETY: this context/stream cannot escape the resident API. Every
            // model/accumulation/optimizer operation uses the same ordered stream;
            // executor entry points and destruction share a gate with capture.
            // The only second-stream reads are immutable published weights: the
            // health readback completes their writes before publication, and the
            // reader retains their allocation through a blocking D2H copy.
            unsafe { device.disable_event_tracking() };
        }
        let mut unique = BTreeMap::new();
        for (parameter, group) in parameters {
            ensure!(
                group < groups.len(),
                "parameter optimizer group is out of bounds"
            );
            if let Some((old, old_group)) = unique.get(&parameter.id()) {
                let old: &Parameter<f32> = old;
                ensure!(
                    Arc::ptr_eq(&old.storage(), &parameter.storage()) && *old_group == group,
                    "conflicting shared parameter identity/group"
                );
            } else {
                unique.insert(parameter.id(), (parameter, group));
            }
        }
        let mut offset = 0usize;
        let mut health_offset = 0usize;
        let mut resident = Vec::new();
        let mut update_ids = HashMap::new();
        let mut ir = Vec::new();
        for (id, (parameter, group)) in unique {
            let storage = parameter.storage();
            let host = storage.lock()?;
            let len = parameter
                .shape()
                .iter()
                .try_fold(1usize, |n, &d| n.checked_mul(d))
                .context("parameter size overflow")?;
            ensure!(
                host.len() == len && len > 0,
                "resident parameter shape/length mismatch or empty parameter"
            );
            ensure!(
                host.iter().all(|x| x.is_finite()),
                "nonfinite initial parameter"
            );
            let weight = Arc::new(stream.memcpy_stod(host.as_slice())?);
            let end = offset
                .checked_add(len)
                .context("resident parameter total overflow")?;
            let health_end = health_offset
                .checked_add(len.div_ceil(kernels::HEALTH_GROUP_SIZE))
                .context("resident health partial count overflow")?;
            resident.push(ResidentParameter {
                id,
                storage: storage.clone(),
                revision: host.revision(),
                weight,
                staged_weight: None,
                range: offset..end,
                health_range: health_offset..health_end,
                group,
            });
            offset = end;
            health_offset = health_end;
            if let std::collections::hash_map::Entry::Vacant(entry) = update_ids.entry(len) {
                let index = ir.len();
                ir.push(kernels::adam(len)?);
                entry.insert(index);
            }
        }
        let padded_len = offset
            .checked_add(1023)
            .context("resident padding overflow")?
            / 1024
            * 1024;
        ensure!(
            padded_len <= u32::MAX as usize,
            "resident gradient storage exceeds launch range"
        );
        let gradients = Arc::new(stream.alloc_zeros(padded_len)?);
        let moment = stream.alloc_zeros(padded_len)?;
        let variance = stream.alloc_zeros(padded_len)?;
        let norm = Reduction::new(stream.clone(), stream_gate.clone(), padded_len, true)?;
        let health_len = health_offset
            .checked_add(1023)
            .context("resident health padding overflow")?
            / 1024
            * 1024;
        let health = Reduction::new(stream.clone(), stream_gate.clone(), health_len, false)?;
        let update_kernels = kernels::Kernels::compile(&device, ir)?;
        let staging = staging::Staging::new(&stream, padded_len, health_len, groups.len())?;
        Ok(Self {
            borrowed_bf16: borrowed_bf16::BorrowState::default(),
            vector_bf16_first_adam: vector_adam::State::default(),
            bf16_master_shadows: bf16_shadow::State::default(),
            small_accumulation_norm: small_accumulation_norm::State::default(),
            device,
            stream,
            stream_gate,
            readback_stream,
            replay,
            parameters: resident,
            groups,
            gradients,
            moment,
            variance,
            staging,
            padded_len,
            norm,
            gradient_norm_partials: false,
            norm_partial_workspace: None,
            norm_partial_validity: norm_partials::Validity::default(),
            norm_partial_attempt: false,
            last_norm_partials_used: false,
            health,
            update_kernels,
            update_ids,
            update_batch_size: 8,
            adam_256_threads: false,
            batched_updates: None,
            accumulate_kernels: HashMap::new(),
            active: false,
            first_accumulation_overwrite: false,
            pending_zero: false,
            accumulation_failed: false,
            timestep: 0,
            last_scale: 1.0,
            kernel_launches: 0,
        })
    }

    /// Batch up to this many small same-group parameter updates in one launch.
    /// The default is eight; one retains the per-parameter path. Legal sizes are 1 through 16.
    /// Batches include only tensors of at most 4096 elements and compile lazily.
    pub fn set_update_batch_size(&mut self, size: usize) -> Result<()> {
        self.borrowed_bf16.invalidate();
        ensure!(
            (1..=16).contains(&size),
            "resident Adam batch size must be 1 through 16"
        );
        if size != self.update_batch_size {
            self.update_batch_size = size;
            self.batched_updates = None;
        }
        Ok(())
    }

    /// Use 256 threads instead of the existing 1024 for standalone Adam
    /// launches. This changes only scheduling: generated arithmetic, staged
    /// state, 256-element health groups and batched launches are unchanged.
    /// Disabled by default; select before starting an accumulation interval.
    pub fn set_experimental_adam_256_threads(&mut self, enabled: bool) -> Result<()> {
        self.borrowed_bf16.invalidate();
        ensure!(
            !self.active,
            "cannot change Adam launch geometry during an active interval"
        );
        self.adam_256_threads = enabled;
        Ok(())
    }

    /// Defer gradient clearing until the first accumulation, which writes
    /// `+0 + incoming` without reading previous gradients. Disabled by default.
    /// An interval without microbatches still clears gradients before Adam.
    pub fn set_experimental_first_accumulation_overwrite(&mut self, enabled: bool) -> Result<()> {
        self.borrowed_bf16.invalidate();
        ensure!(
            !self.active,
            "cannot change gradient clearing during an active interval"
        );
        ensure!(
            enabled || !self.gradient_norm_partials,
            "disable gradient norm partials before first-overwrite"
        );
        self.first_accumulation_overwrite = enabled;
        Ok(())
    }

    /// Fuse exact global norm rows into a single first-overwrite accumulation.
    /// Later microbatches, empty intervals and failed intervals retain full norm.
    pub fn set_experimental_gradient_norm_partials(&mut self, enabled: bool) -> Result<()> {
        self.borrowed_bf16.invalidate();
        ensure!(
            !self.active,
            "cannot change norm partials during an active interval"
        );
        ensure!(
            !enabled || self.first_accumulation_overwrite,
            "norm partials require first-accumulation overwrite"
        );
        if enabled && self.norm_partial_workspace.is_none() {
            let layout = norm_partials::PartialLayout::new(
                self.parameters.iter().map(|p| p.range.clone()).collect(),
                self.padded_len,
            )?;
            self.norm_partial_workspace = Some(norm_partials::Workspace::new(
                &self.stream,
                self.stream_gate.clone(),
                layout,
            )?);
        }
        self.gradient_norm_partials = enabled;
        self.norm_partial_validity.begin_step();
        self.last_norm_partials_used = false;
        Ok(())
    }
    pub fn gradient_norm_partials_status(&self) -> GradientNormPartialsStatus {
        GradientNormPartialsStatus {
            enabled: self.gradient_norm_partials,
            last_step_used: self.last_norm_partials_used,
            rows: self
                .norm_partial_workspace
                .as_ref()
                .map_or(0, |w| w.layout.rows),
            repair_rows: self
                .norm_partial_workspace
                .as_ref()
                .map_or(0, |w| w.layout.repair_rows.len()),
            workspace_bytes: self
                .norm_partial_workspace
                .as_ref()
                .map_or(0, |w| w.layout.rows * 4 + w.layout.repair_rows.len() * 4),
        }
    }
    fn begin_norm_accumulation(&mut self) {
        self.borrowed_bf16.invalidate();
        self.small_accumulation_norm_reset();
        self.norm_partial_attempt = self.norm_partial_validity.begin_accumulation(
            self.gradient_norm_partials,
            self.first_accumulation_overwrite,
            self.pending_zero,
        );
    }

    fn finish_accumulation(&mut self, result: Result<()>) -> Result<()> {
        self.norm_partial_validity
            .finish_accumulation(self.norm_partial_attempt, result.is_ok());
        self.norm_partial_attempt = false;
        if result.is_ok() {
            self.pending_zero = false;
        } else {
            // Some disjoint parameter ranges may already have been enqueued.
            // Never retry or publish this interval; begin_step is the recovery boundary.
            self.accumulation_failed = true;
        }
        result
    }

    /// Create a model executor on the same CUDA context/stream as this owner.
    pub fn executor(&self) -> Result<PtxExecutor> {
        self.executor_with_reduction_mode(PtxReductionMode::Strict)
    }

    /// Select reduction ordering explicitly; parameters, gradients and Adam state
    /// remain f32. Tree ordering can differ from increasing-index accumulation.
    pub fn executor_with_reduction_mode(&self, mode: PtxReductionMode) -> Result<PtxExecutor> {
        PtxExecutor::with_stream(self.stream.clone(), mode, self.stream_gate.clone()).map(
            |mut executor| {
                executor.enable_graph_replay(self.replay);
                executor
            },
        )
    }

    /// Start a new accumulation interval, importing explicit host mutations once.
    /// Host changes retain Adam moments, matching an in-place host weight edit.
    pub fn begin_step(&mut self) -> Result<()> {
        self.borrowed_bf16.invalidate();
        self.small_accumulation_norm.begin_step();
        self.vector_bf16_first_adam.reset_status();
        self.bf16_master_shadows.reset_step_status();
        self.active = true;
        self.accumulation_failed = true;
        self.norm_partial_validity.begin_step();
        self.norm_partial_attempt = false;
        self.last_norm_partials_used = false;
        let stream = self.stream.clone();
        self.reclaim_shadow_pairs()?;
        for index in 0..self.parameters.len() {
            let imported = {
                let parameter = &self.parameters[index];
                if parameter.storage.revision()? != parameter.revision {
                    let host = parameter.storage.lock()?;
                    ensure!(
                        host.len() == parameter.range.len() && host.iter().all(|x| x.is_finite()),
                        "invalid host parameter mutation"
                    );
                    Some((
                        Arc::new(stream.memcpy_stod(host.as_slice())?),
                        host.revision(),
                    ))
                } else {
                    None
                }
            };
            if let Some((weight, revision)) = imported {
                if self
                    .bf16_master_shadows
                    .parameters
                    .contains_key(&self.parameters[index].id)
                {
                    self.import_host_shadow_pair(index, weight, revision)?;
                } else {
                    self.parameters[index].weight = weight;
                    self.parameters[index].revision = revision;
                }
            }
        }
        self.reclaim_shadow_pairs()?;
        if !self.first_accumulation_overwrite {
            stream.memset_zeros(
                Arc::get_mut(&mut self.gradients).context("resident gradients still borrowed")?,
            )?;
        }
        // Padding starts zero and is never written by accumulation kernels.
        self.pending_zero = self.first_accumulation_overwrite;
        self.accumulation_failed = false;
        self.kernel_launches = 0;
        Ok(())
    }

    /// Execute forward/backward with resident weights, then accumulate gradients
    /// through their logical index maps on device. The graph must already weight
    /// each microbatch's loss appropriately; accumulation adds its gradients once.
    pub fn accumulate(
        &mut self,
        executor: &mut PtxExecutor,
        graph: &TensorGraph<f32, WithGrad>,
        inputs: HashMap<String, Vec<f32>>,
    ) -> Result<()> {
        self.accumulate_with_indices(executor, graph, inputs, HashMap::new())
    }

    /// Accumulate with exact integer feeds for lookup/indexed-loss operations.
    pub fn accumulate_with_indices(
        &mut self,
        executor: &mut PtxExecutor,
        graph: &TensorGraph<f32, WithGrad>,
        inputs: HashMap<String, Vec<f32>>,
        indices: crate::tensor::IndexInputs,
    ) -> Result<()> {
        self.borrowed_bf16.invalidate();
        ensure!(self.active, "begin_step must precede accumulation");
        ensure!(
            !self.accumulation_failed,
            "failed accumulation interval; call begin_step to recover"
        );
        self.begin_norm_accumulation();
        let result = self.accumulate_with_indices_inner(executor, graph, inputs, indices);
        self.finish_accumulation(result)
    }

    fn accumulate_with_indices_inner(
        &mut self,
        executor: &mut PtxExecutor,
        graph: &TensorGraph<f32, WithGrad>,
        inputs: HashMap<String, Vec<f32>>,
        indices: crate::tensor::IndexInputs,
    ) -> Result<()> {
        ensure!(
            Arc::ptr_eq(&executor.context(), &self.device),
            "resident executor context mismatch"
        );
        ensure!(
            Arc::ptr_eq(&executor.execution_stream(), &self.stream),
            "resident executor stream mismatch"
        );
        let mut bindings = HashMap::new();
        for node in graph.graph.node_indices() {
            if let TensorGraphNode::Parameter { id, data, .. } = &graph[node] {
                let parameter = self
                    .parameters
                    .iter()
                    .find(|p| p.id == *id)
                    .context("unregistered graph parameter")?;
                ensure!(
                    Arc::ptr_eq(data, &parameter.storage) && data.revision()? == parameter.revision,
                    "parameter changed during resident accumulation"
                );
                bindings.insert(node, parameter.weight.clone());
            }
        }
        if indices.is_empty() {
            executor.run_device(graph, inputs, bindings)?;
        } else {
            executor.run_device_with_indices(graph, inputs, indices, bindings)?;
        }
        self.kernel_launches += executor.execution_metrics().kernel_launches;
        let values = self
            .parameters
            .iter()
            .map(|parameter| {
                let node = graph
                    .gradient_metadata()
                    .param_to_grad
                    .get(&parameter.id)
                    .context("missing resident parameter gradient")?;
                let value = executor.device_value(*node)?;
                ensure!(
                    value.view.shape.iter().product::<usize>() == parameter.range.len(),
                    "resident gradient shape mismatch"
                );
                Ok(value)
            })
            .collect::<Result<Vec<_>>>()?;
        let key = (
            self.pending_zero,
            self.norm_partial_attempt,
            values
                .iter()
                .map(|v| (v.view.clone(), v.source_shape.clone()))
                .collect::<Vec<_>>(),
        );
        if !self.accumulate_kernels.contains_key(&key) {
            let ir = values
                .iter()
                .zip(&self.parameters)
                .map(|(value, parameter)| {
                    if self.norm_partial_attempt {
                        norm_partials::accumulate_ir(
                            &value.view,
                            &value.source_shape,
                            parameter.range.clone(),
                        )
                    } else {
                        kernels::accumulate(value, parameter.range.len(), self.pending_zero)
                    }
                })
                .collect::<Result<_>>()?;
            if self.accumulate_kernels.len() >= 8 {
                self.accumulate_kernels.clear();
            }
            self.accumulate_kernels
                .insert(key.clone(), kernels::Kernels::compile(&self.device, ir)?);
        }
        self.kernel_launches += self.parameters.len();
        let kernels = &self.accumulate_kernels[&key];
        let stream = self.stream.clone();
        let gradients =
            Arc::get_mut(&mut self.gradients).context("resident gradient buffer still borrowed")?;
        for (index, (parameter, value)) in self.parameters.iter().zip(&values).enumerate() {
            let mut target = gradients.slice_mut(parameter.range.clone());
            let function = kernels.module.load_function(&kernels.names[index])?;
            let mut launch = stream.launch_builder(&function);
            launch.arg(value.storage.as_ref()).arg(&mut target);
            let config = if self.norm_partial_attempt {
                let workspace = self
                    .norm_partial_workspace
                    .as_mut()
                    .context("norm partial workspace absent")?;
                let config = workspace.layout.accumulation_config(index)?;
                launch.arg(workspace.partials_mut()?);
                config
            } else {
                LaunchConfig::for_num_elems(u32::try_from(parameter.range.len())?)
            };
            // SAFETY: generated index maps address the retained source allocation;
            // the output is this parameter's disjoint, checked flat gradient range.
            unsafe { launch.launch(config) }?;
            self.small_accumulation_norm.record_standalone();
        }
        Ok(())
    }

    /// Clip the global gradient norm and apply one grouped Adam update. A scalar
    /// finite check validates staged weights/moments before publishing any update.
    pub fn step(&mut self, max_grad_norm: f32) -> Result<ResidentStep> {
        self.borrowed_bf16.invalidate();
        self.borrowed_bf16.reset_status("ordinary_f32_step");
        self.step_impl(max_grad_norm, None)
    }

    fn step_impl(
        &mut self,
        max_grad_norm: f32,
        borrowed_sources: Option<&[Option<cudarc::driver::CudaView<'_, half::bf16>>]>,
    ) -> Result<ResidentStep> {
        self.vector_bf16_first_adam.reset_status();
        self.bf16_master_shadows.reset_step_status();
        ensure!(self.active, "no active resident accumulation interval");
        ensure!(
            !self.accumulation_failed,
            "failed accumulation interval; call begin_step to recover"
        );
        ensure!(
            max_grad_norm.is_finite() && max_grad_norm >= 0.0,
            "invalid gradient norm limit"
        );
        if self.pending_zero {
            self.stream.memset_zeros(
                Arc::get_mut(&mut self.gradients).context("resident gradients still borrowed")?,
            )?;
            self.pending_zero = false;
        }
        let use_partials = self.gradient_norm_partials && self.norm_partial_validity.valid();
        let squared = if use_partials {
            let workspace = self
                .norm_partial_workspace
                .as_mut()
                .context("norm partial workspace absent")?;
            let value = workspace.run(&self.stream, self.gradients.as_ref())?;
            self.kernel_launches += workspace.last_launches();
            value
        } else {
            let value = self.norm.run(self.gradients.clone())?;
            self.kernel_launches += self.norm.executor.execution_metrics().kernel_launches;
            value
        };
        self.last_norm_partials_used = use_partials;
        ensure!(
            squared.is_finite() && squared >= 0.0,
            "nonfinite resident gradients/norm"
        );
        let norm = f64::from(squared).sqrt();
        let scale = if max_grad_norm > 0.0 {
            (f64::from(max_grad_norm) / (norm + 1e-6)).min(1.0) as f32
        } else {
            1.0
        };
        let next_step = self
            .timestep
            .checked_add(1)
            .context("Adam timestep overflow")?;
        let exponent =
            i32::try_from(next_step).context("Adam bias-correction timestep exceeds i32")?;
        let stream = self.stream.clone();
        for (g, settings) in self.groups.iter().zip(&mut self.staging.settings) {
            stream.memcpy_htod(
                &[
                    g.learning_rate,
                    g.beta1,
                    g.beta2,
                    1.0 - g.beta1,
                    1.0 - g.beta2,
                    1.0 - g.beta1.powi(exponent),
                    1.0 - g.beta2.powi(exponent),
                    g.epsilon,
                    g.weight_decay,
                    scale,
                ],
                settings,
            )?;
        }
        if self.update_batch_size > 1 && self.batched_updates.is_none() {
            self.batched_updates = Some(batch::Updates::new(self)?);
        }
        let borrowed_lengths = borrowed_sources
            .map(|sources| {
                self.parameters
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| {
                        !self
                            .batched_updates
                            .as_ref()
                            .is_some_and(|u| u.covers(*index))
                            && sources.get(*index).is_some_and(Option::is_some)
                    })
                    .map(|(_, p)| p.range.len())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        if !borrowed_lengths.is_empty()
            && let Err(reason) = self
                .borrowed_bf16
                .compile_missing(&self.device, &borrowed_lengths)
        {
            self.borrowed_bf16.status.reason =
                format!("BF16 kernel unavailable; F32 fallback: {reason:#}");
        }
        self.vector_bf16_first_adam
            .prepare(&self.device, &borrowed_lengths);
        for parameter in &mut self.parameters {
            ensure!(
                parameter.storage.revision()? == parameter.revision,
                "parameter changed during resident update"
            );
            if parameter.staged_weight.is_none() {
                parameter.staged_weight = Some(stream.alloc_zeros::<f32>(parameter.range.len())?);
            }
        }
        self.bf16_master_shadows.prepare_staging(&stream)?;
        let next_m = &mut self.staging.moment;
        let next_v = &mut self.staging.variance;
        let health = Arc::get_mut(&mut self.staging.health)
            .context("resident health staging still borrowed")?;
        for (index, parameter) in self.parameters.iter_mut().enumerate() {
            if self
                .batched_updates
                .as_ref()
                .is_some_and(|updates| updates.covers(index))
            {
                if self.vector_bf16_first_adam.enabled {
                    self.vector_bf16_first_adam.status.batched_f32_parameters += 1;
                }
                if self.borrowed_bf16.enabled {
                    self.borrowed_bf16.status.batched_f32_parameters += 1;
                    self.borrowed_bf16.status.gradient_read_bytes += parameter.range.len() * 4;
                }
                continue;
            }
            self.kernel_launches += 1;
            let len = parameter.range.len();
            let g = self.gradients.slice(parameter.range.clone());
            let m = self.moment.slice(parameter.range.clone());
            let v = self.variance.slice(parameter.range.clone());
            let mut nm = next_m.slice_mut(parameter.range.clone());
            let mut nv = next_v.slice_mut(parameter.range.clone());
            let mut status = health.slice_mut(parameter.health_range.clone());
            let weight = parameter
                .staged_weight
                .as_mut()
                .context("resident weight staging absent")?;
            let config = launch::standalone_adam(len, self.adam_256_threads)?;
            let source = borrowed_sources
                .and_then(|v| v.get(index))
                .and_then(Option::as_ref);
            let bf16_function = source.and_then(|_| {
                let maximum = self.borrowed_bf16.limits.get(&len)?;
                if *maximum < config.block_dim.0 as usize {
                    return None;
                }
                let kernels = self.borrowed_bf16.kernels.as_ref()?;
                let id = self.borrowed_bf16.ids.get(&len)?;
                kernels.module.load_function(&kernels.names[*id]).ok()
            });
            let use_bf16 = bf16_function.is_some();
            let function = if let Some(function) = bf16_function {
                function
            } else {
                let id = self.update_ids[&len];
                self.update_kernels
                    .module
                    .load_function(&self.update_kernels.names[id])?
            };
            // The optional vector path may only replace an already-admitted old
            // BF16 function. Current typed bindings are checked on every call;
            // the cache stores no parameter/gradient addresses.
            let vector_function = if self.vector_bf16_first_adam.enabled {
                if let Some(source) = source.filter(|_| use_bf16) {
                    if vector_adam::bindings_admitted(
                        len,
                        &stream,
                        parameter.weight.as_ref(),
                        source,
                        &m,
                        &v,
                        &self.staging.settings[parameter.group],
                        weight,
                        &nm,
                        &nv,
                        &status,
                    ) {
                        self.vector_bf16_first_adam.function(len)
                    } else {
                        self.vector_bf16_first_adam
                            .fallback("binding_alignment_range_or_alias");
                        None
                    }
                } else {
                    self.vector_bf16_first_adam
                        .fallback("old_bf16_source_not_selected");
                    None
                }
            } else {
                None
            };
            let mut vector_selected = vector_function.is_some();
            let scalar_config = config;
            let (mut function, mut config) = if let Some(vector_function) = vector_function {
                (
                    vector_function,
                    LaunchConfig {
                        grid_dim: (u32::try_from(len / 1024)?, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                )
            } else {
                (function, config)
            };
            let paired_kind = if self
                .bf16_master_shadows
                .parameters
                .contains_key(&parameter.id)
            {
                let requested = if vector_selected {
                    bf16_shadow::AdamKind::Vector4
                } else if use_bf16 {
                    bf16_shadow::AdamKind::Bf16First
                } else {
                    bf16_shadow::AdamKind::F32
                };
                let (paired, kind) = self.bf16_master_shadows.paired_function(len, requested)?;
                if vector_selected && kind != bf16_shadow::AdamKind::Vector4 {
                    vector_selected = false;
                    config = scalar_config;
                    self.vector_bf16_first_adam
                        .fallback("paired_vector_unavailable");
                    self.bf16_master_shadows.record_vector_fallback(len);
                }
                let gradient = if use_bf16 {
                    bf16_shadow::operand_range(
                        source.context("paired BF16 gradient absent")?,
                        &stream,
                    )?
                } else {
                    bf16_shadow::operand_range(&g, &stream)?
                };
                let operands = [
                    bf16_shadow::operand_range(parameter.weight.as_ref(), &stream)?,
                    gradient,
                    bf16_shadow::operand_range(&m, &stream)?,
                    bf16_shadow::operand_range(&v, &stream)?,
                    bf16_shadow::operand_range(&self.staging.settings[parameter.group], &stream)?,
                    bf16_shadow::operand_range(weight, &stream)?,
                    bf16_shadow::operand_range(&nm, &stream)?,
                    bf16_shadow::operand_range(&nv, &stream)?,
                    bf16_shadow::operand_range(&status, &stream)?,
                ];
                self.bf16_master_shadows
                    .validate_staged(parameter.id, &stream, len, &operands)?;
                function = paired;
                Some(kind)
            } else {
                None
            };
            let mut launch = stream.launch_builder(&function);
            launch.arg(parameter.weight.as_ref());
            if use_bf16 {
                launch.arg(source.context("borrowed gradient source absent")?);
            } else {
                launch.arg(&g);
            }
            launch
                .arg(&m)
                .arg(&v)
                .arg(&self.staging.settings[parameter.group])
                .arg(weight)
                .arg(&mut nm)
                .arg(&mut nv)
                .arg(&mut status);
            if paired_kind.is_some() {
                let shadow = self
                    .bf16_master_shadows
                    .parameters
                    .get_mut(&parameter.id)
                    .and_then(|p| p.staged.as_mut())
                    .context("paired shadow staging disappeared")?;
                launch.arg(shadow);
            }
            // SAFETY: weight/state operands cover len f32 elements; status covers
            // ceil(len/256) partials. A 256/1024-thread launch emits one/four
            // bounded partials per CTA with identical logical element groups.
            // Outputs remain staged in distinct allocations/disjoint parameter
            // ranges until health validation.
            unsafe { launch.launch(config) }?;
            if let Some(kind) = paired_kind {
                self.bf16_master_shadows.record_submission(kind, len);
            }
            if vector_selected {
                self.vector_bf16_first_adam.status.submitted_parameters += 1;
                self.vector_bf16_first_adam.status.submitted_elements += len;
            }
            if self.borrowed_bf16.enabled {
                if use_bf16 {
                    self.borrowed_bf16.status.standalone_bf16_parameters += 1;
                    self.borrowed_bf16.status.bf16_gradient_elements += len;
                    self.borrowed_bf16.status.logical_gradient_read_bytes_saved += len * 2;
                    self.borrowed_bf16.status.gradient_read_bytes += len * 2;
                } else {
                    self.borrowed_bf16.status.standalone_f32_parameters += 1;
                    self.borrowed_bf16.status.gradient_read_bytes += len * 4;
                }
            }
        }
        if let Some(updates) = &self.batched_updates {
            updates.launch(
                &stream,
                &mut self.parameters,
                batch::State {
                    gradient: self.gradients.as_ref(),
                    moment: &self.moment,
                    variance: &self.variance,
                    next_moment: next_m,
                    next_variance: next_v,
                    health,
                    settings: &self.staging.settings,
                },
            )?;
            self.kernel_launches += updates.launches();
        }
        ensure!(
            self.health.run(self.staging.health.clone())? == 0.0,
            "Adam produced nonfinite parameters/moments"
        );
        self.kernel_launches += self.health.executor.execution_metrics().kernel_launches;
        self.bf16_master_shadows
            .validate_publication(&self.parameters)?;
        for parameter in &mut self.parameters {
            self.bf16_master_shadows
                .publish_parameter(parameter, &self.readback_stream)?;
        }
        std::mem::swap(&mut self.moment, &mut self.staging.moment);
        std::mem::swap(&mut self.variance, &mut self.staging.variance);
        self.timestep = next_step;
        self.last_scale = scale;
        self.vector_bf16_first_adam.status.committed_parameters =
            self.vector_bf16_first_adam.status.submitted_parameters;
        self.vector_bf16_first_adam.status.committed_elements =
            self.vector_bf16_first_adam.status.submitted_elements;
        self.bf16_master_shadows.status.committed_parameters =
            self.bf16_master_shadows.status.submitted_parameters;
        self.bf16_master_shadows.status.committed_elements =
            self.bf16_master_shadows.status.submitted_elements;
        self.bf16_master_shadows.refresh_accounting();
        self.active = false;
        Ok(ResidentStep {
            gradient_norm: norm,
            gradient_scale: scale,
            timestep: next_step,
            kernel_launches: self.kernel_launches,
        })
    }

    /// Change a group's rate without resetting its moments or timestep.
    pub fn set_learning_rate(&mut self, group: usize, rate: f32) -> Result<()> {
        self.borrowed_bf16.invalidate();
        ensure!(
            rate.is_finite() && rate >= 0.0,
            "invalid Adam learning rate"
        );
        self.groups
            .get_mut(group)
            .context("Adam group is out of bounds")?
            .learning_rate = rate;
        Ok(())
    }

    /// Explicit diagnostic readback of the last clipped gradients.
    pub fn gradients(&self) -> Result<HashMap<usize, Vec<f32>>> {
        ensure!(
            !self.active && self.timestep > 0,
            "gradient diagnostics require a successful step"
        );
        let mut host = vec![0.0; self.padded_len];
        self.stream
            .clone()
            .memcpy_dtoh(self.gradients.as_ref(), &mut host)?;
        Ok(self
            .parameters
            .iter()
            .map(|p| {
                (
                    p.id,
                    host[p.range.clone()]
                        .iter()
                        .map(|x| x * self.last_scale)
                        .collect(),
                )
            })
            .collect())
    }

    /// Explicit diagnostic moment readback, keyed by shared parameter identity.
    pub fn moments(&self) -> Result<HashMap<usize, AdamMoments>> {
        let mut m = vec![0.0; self.padded_len];
        let mut v = vec![0.0; self.padded_len];
        let stream = self.stream.clone();
        stream.memcpy_dtoh(&self.moment, &mut m)?;
        stream.memcpy_dtoh(&self.variance, &mut v)?;
        Ok(self
            .parameters
            .iter()
            .map(|p| {
                (
                    p.id,
                    AdamMoments {
                        first: m[p.range.clone()].to_vec(),
                        second: v[p.range.clone()].to_vec(),
                    },
                )
            })
            .collect())
    }
}
