//! CUDA-resident f32 parameters, accumulated gradients and grouped Adam state.
mod config;
mod kernels;
mod reduction;
use super::types::{CudaDType, F32};
use super::{PtxExecutor, PtxReductionMode};
use crate::graph::{TensorGraph, TensorGraphNode, WithGrad};
use crate::tensor::{Parameter, ParameterStorage, Shape};
use crate::tile::VirtualTensor;
use anyhow::{Context, Result, ensure};
pub use config::{AdamGroup, AdamMoments, ResidentStep};
use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
use reduction::Reduction;
use std::collections::{BTreeMap, HashMap};
use std::ops::Range;
use std::sync::Arc;

pub(crate) struct DeviceValue<D: CudaDType = F32> {
    pub storage: Arc<CudaSlice<D::HostType>>,
    pub view: VirtualTensor,
    pub source_shape: Shape,
}

struct ResidentParameter {
    id: usize,
    storage: Arc<ParameterStorage<f32>>,
    revision: u64,
    weight: Arc<CudaSlice<f32>>,
    range: Range<usize>,
    group: usize,
}

/// An explicit training owner. Parameters/moments and gradients stay on CUDA
/// between microbatches. Host reads synchronize through shared parameter storage.
/// Writers must be serialized; host changes during an update are rejected.
pub struct ResidentAdam {
    device: Arc<CudaContext>,
    parameters: Vec<ResidentParameter>,
    groups: Vec<AdamGroup>,
    gradients: Arc<CudaSlice<f32>>,
    moment: CudaSlice<f32>,
    variance: CudaSlice<f32>,
    padded_len: usize,
    norm: Reduction,
    health: Reduction,
    update_kernels: kernels::Kernels,
    update_ids: HashMap<usize, usize>,
    accumulate_kernels: HashMap<Vec<(VirtualTensor, Shape)>, kernels::Kernels>,
    active: bool,
    timestep: usize,
    last_scale: f32,
    kernel_launches: usize,
}

impl ResidentAdam {
    /// Register shared parameters once, with explicit optimizer group indices.
    /// Duplicate references to the same storage/group are deduplicated.
    pub fn new(parameters: Vec<(Parameter<f32>, usize)>, groups: Vec<AdamGroup>) -> Result<Self> {
        ensure!(
            !parameters.is_empty() && !groups.is_empty(),
            "resident training requires parameters/groups"
        );
        for group in &groups {
            group.validate()?;
        }
        let device = CudaContext::new(0).context("initialize resident training CUDA device")?;
        let stream = device.default_stream();
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
            resident.push(ResidentParameter {
                id,
                storage: storage.clone(),
                revision: host.revision(),
                weight,
                range: offset..end,
                group,
            });
            offset = end;
            update_ids.entry(len).or_insert_with(|| {
                let index = ir.len();
                ir.push(kernels::adam(len));
                index
            });
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
        let norm = Reduction::new(device.clone(), padded_len, true)?;
        let health = Reduction::new(device.clone(), padded_len, false)?;
        let update_kernels = kernels::Kernels::compile(&device, ir)?;
        Ok(Self {
            device,
            parameters: resident,
            groups,
            gradients,
            moment,
            variance,
            padded_len,
            norm,
            health,
            update_kernels,
            update_ids,
            accumulate_kernels: HashMap::new(),
            active: false,
            timestep: 0,
            last_scale: 1.0,
            kernel_launches: 0,
        })
    }

    /// Create a model executor on the same CUDA context/stream as this owner.
    pub fn executor(&self) -> Result<PtxExecutor> {
        self.executor_with_reduction_mode(PtxReductionMode::Strict)
    }

    /// Select reduction ordering explicitly; parameters, gradients and Adam state
    /// remain f32. Tree ordering can differ from increasing-index accumulation.
    pub fn executor_with_reduction_mode(&self, mode: PtxReductionMode) -> Result<PtxExecutor> {
        PtxExecutor::with_context(self.device.clone(), mode)
    }

    /// Start a new accumulation interval, importing explicit host mutations once.
    /// Host changes retain Adam moments, matching an in-place host weight edit.
    pub fn begin_step(&mut self) -> Result<()> {
        let stream = self.device.default_stream();
        for parameter in &mut self.parameters {
            if parameter.storage.revision()? != parameter.revision {
                let host = parameter.storage.lock()?;
                ensure!(
                    host.len() == parameter.range.len() && host.iter().all(|x| x.is_finite()),
                    "invalid host parameter mutation"
                );
                parameter.weight = Arc::new(stream.memcpy_stod(host.as_slice())?);
                parameter.revision = host.revision();
            }
        }
        stream.memset_zeros(
            Arc::get_mut(&mut self.gradients).context("resident gradients still borrowed")?,
        )?;
        self.kernel_launches = 0;
        self.active = true;
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
        ensure!(self.active, "begin_step must precede accumulation");
        ensure!(
            Arc::ptr_eq(&executor.context(), &self.device),
            "resident executor context mismatch"
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
        executor.run_device(graph, inputs, bindings)?;
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
        let key: Vec<_> = values
            .iter()
            .map(|v| (v.view.clone(), v.source_shape.clone()))
            .collect();
        if !self.accumulate_kernels.contains_key(&key) {
            let ir = values
                .iter()
                .zip(&self.parameters)
                .map(|(value, parameter)| kernels::accumulate(value, parameter.range.len()))
                .collect::<Result<_>>()?;
            if self.accumulate_kernels.len() >= 8 {
                self.accumulate_kernels.clear();
            }
            self.accumulate_kernels
                .insert(key.clone(), kernels::Kernels::compile(&self.device, ir)?);
        }
        self.kernel_launches += self.parameters.len();
        let kernels = &self.accumulate_kernels[&key];
        let stream = self.device.default_stream();
        let gradients =
            Arc::get_mut(&mut self.gradients).context("resident gradient buffer still borrowed")?;
        for (index, (parameter, value)) in self.parameters.iter().zip(&values).enumerate() {
            let mut target = gradients.slice_mut(parameter.range.clone());
            let function = kernels.module.load_function(&kernels.names[index])?;
            let mut launch = stream.launch_builder(&function);
            launch.arg(value.storage.as_ref()).arg(&mut target);
            // SAFETY: generated index maps address the retained source allocation;
            // the output is this parameter's disjoint, checked flat gradient range.
            unsafe {
                launch.launch(LaunchConfig::for_num_elems(u32::try_from(
                    parameter.range.len(),
                )?))
            }?;
        }
        Ok(())
    }

    /// Clip the global gradient norm and apply one grouped Adam update. A scalar
    /// finite check validates staged weights/moments before publishing any update.
    pub fn step(&mut self, max_grad_norm: f32) -> Result<ResidentStep> {
        ensure!(self.active, "no active resident accumulation interval");
        ensure!(
            max_grad_norm.is_finite() && max_grad_norm >= 0.0,
            "invalid gradient norm limit"
        );
        let squared = self.norm.run(self.gradients.clone())?;
        self.kernel_launches += self.norm.executor.execution_metrics().kernel_launches;
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
        let stream = self.device.default_stream();
        let settings = self
            .groups
            .iter()
            .map(|g| {
                stream
                    .memcpy_stod(&[
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
                    ])
                    .map_err(Into::into)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut next_m = stream.alloc_zeros::<f32>(self.padded_len)?;
        let mut next_v = stream.alloc_zeros::<f32>(self.padded_len)?;
        let mut health = stream.alloc_zeros::<f32>(self.padded_len)?;
        let mut weights = Vec::new();
        self.kernel_launches += self.parameters.len();
        for parameter in &self.parameters {
            ensure!(
                parameter.storage.revision()? == parameter.revision,
                "parameter changed during resident update"
            );
            let len = parameter.range.len();
            let g = self.gradients.slice(parameter.range.clone());
            let m = self.moment.slice(parameter.range.clone());
            let v = self.variance.slice(parameter.range.clone());
            let mut nm = next_m.slice_mut(parameter.range.clone());
            let mut nv = next_v.slice_mut(parameter.range.clone());
            let mut status = health.slice_mut(parameter.range.clone());
            let mut weight = stream.alloc_zeros::<f32>(len)?;
            let id = self.update_ids[&len];
            let function = self
                .update_kernels
                .module
                .load_function(&self.update_kernels.names[id])?;
            let mut launch = stream.launch_builder(&function);
            launch
                .arg(parameter.weight.as_ref())
                .arg(&g)
                .arg(&m)
                .arg(&v)
                .arg(&settings[parameter.group])
                .arg(&mut weight)
                .arg(&mut nm)
                .arg(&mut nv)
                .arg(&mut status);
            // SAFETY: all operands cover exactly len f32 elements. Output weights
            // and state are staged in distinct allocations/disjoint parameter ranges.
            unsafe { launch.launch(LaunchConfig::for_num_elems(u32::try_from(len)?)) }?;
            weights.push(Arc::new(weight));
        }
        ensure!(
            self.health.run(Arc::new(health))? == 0.0,
            "Adam produced nonfinite parameters/moments"
        );
        self.kernel_launches += self.health.executor.execution_metrics().kernel_launches;
        for (parameter, weight) in self.parameters.iter_mut().zip(weights) {
            let device = self.device.clone();
            let latest = weight.clone();
            parameter.revision = parameter.storage.publish_device(
                parameter.revision,
                Arc::new(move || {
                    let mut host = vec![0.0; latest.len()];
                    device
                        .default_stream()
                        .memcpy_dtoh(latest.as_ref(), &mut host)
                        .context("read resident parameter")?;
                    Ok(host)
                }),
            )?;
            parameter.weight = weight;
        }
        self.moment = next_m;
        self.variance = next_v;
        self.timestep = next_step;
        self.last_scale = scale;
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
        self.device
            .default_stream()
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
        let stream = self.device.default_stream();
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
