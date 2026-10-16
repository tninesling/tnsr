use super::*;
use crate::ptx::mixed::{MixedExecutor, ResidentBindings};
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

    /// Execute mixed graph autograd and accumulate promoted f32 gradients into
    /// the same clipping, finite-check and grouped Adam path as homogeneous f32.
    pub fn accumulate_mixed(
        &mut self,
        executor: &mut MixedExecutor,
        graph: &TensorGraph<f32, WithGrad>,
        inputs: HashMap<String, Vec<f32>>,
        indices: crate::tensor::IndexInputs,
    ) -> Result<()> {
        ensure!(self.active, "begin_step must precede mixed accumulation");
        ensure!(
            Arc::ptr_eq(executor.execution_stream(), &self.stream),
            "mixed resident stream mismatch"
        );
        let bindings = self.mixed_bindings(graph)?;
        executor.upload_with_indices(inputs, indices)?;
        executor.refresh_parameters(&bindings)?;
        executor.run()?;
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
        let key: Vec<_> = values
            .iter()
            .map(|(_, view, shape)| (view.clone(), shape.clone()))
            .collect();
        if !self.accumulate_kernels.contains_key(&key) {
            let ir = values
                .iter()
                .zip(&self.parameters)
                .map(|((_, view, shape), p)| kernels::accumulate_view(view, shape, p.range.len()))
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
            // SAFETY: the mixed owner retains the validated source allocation;
            // both owners use this ordered stream, and target ranges are disjoint.
            unsafe {
                launch.launch(LaunchConfig::for_num_elems(u32::try_from(
                    parameter.range.len(),
                )?))
            }?;
        }
        self.kernel_launches += self.parameters.len();
        Ok(())
    }
}
