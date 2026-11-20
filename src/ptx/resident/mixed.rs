use super::*;
use crate::ptx::mixed::{MixedExecutor, MixedOptions, ResidentBindings};
use crate::tile::MatMulPrecision;

impl ResidentAdam {
    fn mixed_bindings(
        &self,
        graph: &TensorGraph<f32, WithGrad>,
    ) -> Result<HashMap<crate::graph::NodeIndex, Arc<CudaSlice<f32>>>> {
        let mut bindings = HashMap::new();
        for node in graph.graph.node_indices() {
            if let TensorGraphNode::Parameter { id, data, .. } = &graph[node] {
                let parameter = self
                    .parameters
                    .iter()
                    .find(|p| p.id == *id)
                    .context("unregistered mixed parameter")?;
                ensure!(
                    Arc::ptr_eq(data, &parameter.storage) && data.revision()? == parameter.revision,
                    "parameter changed during mixed resident accumulation"
                );
                bindings.insert(node, parameter.weight.clone());
            }
        }
        Ok(bindings)
    }

    /// Compile stable f32 master bindings; subsequent Adam publications refresh
    /// those addresses device-to-device without recapturing or downloading.
    pub fn mixed_executor(
        &self,
        graph: &TensorGraph<f32, WithGrad>,
        precision: MatMulPrecision,
    ) -> Result<MixedExecutor> {
        if self.bf16_master_shadows.enabled {
            return self.mixed_executor_with_options(
                graph,
                precision,
                self.bf16_master_shadows.options,
            );
        }
        ensure!(self.active, "begin_step must precede mixed compilation");
        MixedExecutor::for_resident(
            graph,
            precision,
            self.replay,
            ResidentBindings {
                stream: self.stream.clone(),
                gate: self.stream_gate.clone(),
                parameters: self.mixed_bindings(graph)?,
            },
        )
    }

    /// Compile explicitly selected mixed-layout experiments on the resident
    /// stream. Default options retain the established compilation policy.
    pub fn mixed_executor_with_options(
        &self,
        graph: &TensorGraph<f32, WithGrad>,
        precision: MatMulPrecision,
        options: MixedOptions,
    ) -> Result<MixedExecutor> {
        if self.bf16_master_shadows.enabled {
            ensure!(self.active, "begin_step must precede mixed compilation");
            ensure!(
                options == self.bf16_master_shadows.options,
                "mixed options differ from sparse shadow declaration"
            );
            return MixedExecutor::for_resident_with_shadows(
                graph,
                precision,
                self.replay,
                ResidentBindings {
                    stream: self.stream.clone(),
                    gate: self.stream_gate.clone(),
                    parameters: self.mixed_bindings(graph)?,
                },
                options,
                self.mixed_shadow_bindings(graph)?,
            );
        }
        if options == MixedOptions::default() {
            return self.mixed_executor(graph, precision);
        }
        ensure!(self.active, "begin_step must precede mixed compilation");
        MixedExecutor::for_resident_with_options(
            graph,
            precision,
            self.replay,
            ResidentBindings {
                stream: self.stream.clone(),
                gate: self.stream_gate.clone(),
                parameters: self.mixed_bindings(graph)?,
            },
            options,
        )
    }

    /// Execute mixed graph autograd and accumulate promoted f32 gradients into
    /// the same clipping, finite-check and grouped Adam path as homogeneous f32.
    pub fn accumulate_mixed(
        &mut self,
        executor: &mut MixedExecutor,
        graph: &TensorGraph<f32, WithGrad>,
        inputs: HashMap<String, Vec<f32>>,
        indices: crate::tensor::IndexInputs,
    ) -> Result<()> {
        self.borrowed_bf16.invalidate();
        ensure!(self.active, "begin_step must precede mixed accumulation");
        ensure!(
            !self.accumulation_failed,
            "failed accumulation interval; call begin_step to recover"
        );
        self.begin_norm_accumulation();
        let result = self.accumulate_mixed_inner(executor, graph, inputs, indices);
        let result = self.finish_accumulation(result);
        if result.is_ok()
            && self.borrowed_bf16.enabled
            && self.norm_partial_validity.single_attempt()
            && let Some(stamp) = executor.borrowed_gradient_stamp()
            && let Some(nodes) = self
                .parameters
                .iter()
                .map(|p| graph.gradient_metadata().param_to_grad.get(&p.id).copied())
                .collect::<Option<Vec<_>>>()
        {
            self.borrowed_bf16.candidate = Some(super::borrowed_bf16::Candidate { stamp, nodes });
        }
        result
    }

    fn accumulate_mixed_inner(
        &mut self,
        executor: &mut MixedExecutor,
        graph: &TensorGraph<f32, WithGrad>,
        inputs: HashMap<String, Vec<f32>>,
        indices: crate::tensor::IndexInputs,
    ) -> Result<()> {
        ensure!(
            Arc::ptr_eq(executor.execution_stream(), &self.stream),
            "mixed resident stream mismatch"
        );
        let bindings = self.mixed_bindings(graph)?;
        executor.upload_with_indices(inputs, indices)?;
        if self.bf16_master_shadows.enabled {
            let shadows = self.mixed_shadow_bindings(graph)?;
            let retired = executor.refresh_parameters_with_shadows(&bindings, &shadows)?;
            drop(shadows);
            self.reclaim_mixed_shadow_bindings(graph, retired)?;
        } else {
            let retired = executor.refresh_parameters(&bindings)?;
            self.reclaim_mixed_masters(graph, retired)?;
        }
        executor.run()?;
        if self.bf16_master_shadows.enabled {
            self.bf16_master_shadows.status.bound_cast_aliases =
                executor.info().bf16_master_shadow_cast_aliases;
            self.bf16_master_shadows.status.bound_cast_alias_bytes =
                executor.info().bf16_master_shadow_alias_bytes;
            self.bf16_master_shadows.status.removed_cast_kernels =
                executor.info().bf16_master_shadow_cast_aliases;
            self.bf16_master_shadows.status.last_parameter_copies =
                executor.transfers().parameter_copies;
        }
        self.kernel_launches += executor.info().kernels;
        let values = self
            .parameters
            .iter()
            .map(|p| {
                let node = *graph
                    .gradient_metadata()
                    .param_to_grad
                    .get(&p.id)
                    .context("missing mixed parameter gradient")?;
                let (binding, view, shape) = executor.device_binding(node)?;
                ensure!(
                    view.shape.iter().product::<usize>() == p.range.len(),
                    "mixed gradient shape mismatch"
                );
                Ok((binding, view, shape))
            })
            .collect::<Result<Vec<_>>>()?;
        if self.small_accumulation_norm.enabled
            && self.pending_zero
            && self.norm_partial_attempt
            && self.try_small_accumulation_norm(&values)?
        {
            return Ok(());
        }
        let key = (
            self.pending_zero,
            self.norm_partial_attempt,
            values
                .iter()
                .map(|(_, view, shape)| (view.clone(), shape.clone()))
                .collect::<Vec<_>>(),
        );
        if !self.accumulate_kernels.contains_key(&key) {
            let ir = values
                .iter()
                .zip(&self.parameters)
                .map(|((_, view, shape), p)| {
                    if self.norm_partial_attempt {
                        norm_partials::accumulate_ir(view, shape, p.range.clone())
                    } else {
                        kernels::accumulate_view(view, shape, p.range.len(), self.pending_zero)
                    }
                })
                .collect::<Result<_>>()?;
            if self.accumulate_kernels.len() >= 8 {
                self.accumulate_kernels.clear();
            }
            self.accumulate_kernels
                .insert(key.clone(), kernels::Kernels::compile(&self.device, ir)?);
        }
        let gate = self.stream_gate.clone();
        let _guard = gate
            .lock()
            .map_err(|_| anyhow::anyhow!("mixed accumulation stream lock poisoned"))?;
        let kernels = &self.accumulate_kernels[&key];
        let gradients = Arc::get_mut(&mut self.gradients).context("resident gradients borrowed")?;
        for (index, (parameter, (binding, _, _))) in self.parameters.iter().zip(&values).enumerate()
        {
            let mut target = gradients.slice_mut(parameter.range.clone());
            let function = kernels.module.load_function(&kernels.names[index])?;
            let mut launch = self.stream.launch_builder(&function);
            launch.arg(&binding.pointer).arg(&mut target);
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
            // SAFETY: the mixed owner retains the validated source allocation;
            // both owners use this ordered stream, and target ranges are disjoint.
            unsafe { launch.launch(config) }?;
            self.small_accumulation_norm.record_standalone();
        }
        self.kernel_launches += self.parameters.len();
        Ok(())
    }
}
