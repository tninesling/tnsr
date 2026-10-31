//! Bounded horizontal Adam fusion using ordinary tracked pointer arguments.
use super::{ResidentParameter, kernels};
use anyhow::{Context, Result};
use cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use std::collections::HashSet;
use std::sync::Arc;

struct Batch {
    parameters: Vec<usize>,
    max_len: u32,
    group: usize,
}

pub(super) struct Updates {
    batches: Vec<Batch>,
    kernels: Option<kernels::Kernels>,
    covered: HashSet<usize>,
}

pub(super) struct State<'a> {
    pub gradient: &'a CudaSlice<f32>,
    pub moment: &'a CudaSlice<f32>,
    pub variance: &'a CudaSlice<f32>,
    pub next_moment: &'a mut CudaSlice<f32>,
    pub next_variance: &'a mut CudaSlice<f32>,
    pub health: &'a mut CudaSlice<f32>,
    pub settings: &'a [CudaSlice<f32>],
}

impl Updates {
    pub fn new(owner: &super::ResidentAdam) -> Result<Self> {
        let mut batches = Vec::new();
        let mut ir = Vec::new();
        let mut covered = HashSet::new();
        for group in 0..owner.groups.len() {
            let mut indices: Vec<_> = owner
                .parameters
                .iter()
                .enumerate()
                .filter(|(_, p)| p.group == group && p.range.len() <= 4096)
                .map(|(i, _)| i)
                .collect();
            indices.sort_by_key(|&i| owner.parameters[i].range.len());
            for chunk in indices.chunks(owner.update_batch_size) {
                if chunk.len() < 2 {
                    continue;
                }
                let descriptors: Vec<_> = chunk
                    .iter()
                    .map(|&i| {
                        let range = &owner.parameters[i].range;
                        (range.start, range.len())
                    })
                    .collect();
                let max_len = descriptors
                    .iter()
                    .map(|&(_, len)| len)
                    .max()
                    .context("empty resident update batch")?;
                ir.push(kernels::adam_batch(&descriptors, owner.padded_len)?);
                covered.extend(chunk.iter().copied());
                batches.push(Batch {
                    parameters: chunk.to_vec(),
                    max_len: u32::try_from(max_len)?,
                    group,
                });
            }
        }
        let kernels = if ir.is_empty() {
            None
        } else {
            Some(kernels::Kernels::compile(&owner.device, ir)?)
        };
        Ok(Self {
            batches,
            kernels,
            covered,
        })
    }

    pub fn covers(&self, index: usize) -> bool {
        self.covered.contains(&index)
    }
    pub fn launches(&self) -> usize {
        self.batches.len()
    }

    pub fn launch(
        &self,
        stream: &Arc<CudaStream>,
        parameters: &mut [ResidentParameter],
        state: State<'_>,
    ) -> Result<()> {
        let Some(kernels) = &self.kernels else {
            return Ok(());
        };
        for (index, batch) in self.batches.iter().enumerate() {
            let function = kernels.module.load_function(&kernels.names[index])?;
            let mut selected: Vec<_> = parameters
                .iter_mut()
                .enumerate()
                .filter_map(|(i, p)| {
                    batch
                        .parameters
                        .iter()
                        .position(|&member| member == i)
                        .map(|position| (position, p))
                })
                .collect();
            selected.sort_by_key(|(position, _)| *position);
            let mut launch = stream.launch_builder(&function);
            launch
                .arg(state.gradient)
                .arg(state.moment)
                .arg(state.variance)
                .arg(&state.settings[batch.group])
                .arg(&mut *state.next_moment)
                .arg(&mut *state.next_variance)
                .arg(&mut *state.health);
            for (_, parameter) in &mut selected {
                launch.arg(parameter.weight.as_ref()).arg(
                    parameter
                        .staged_weight
                        .as_mut()
                        .context("batched resident weight staging absent")?,
                );
            }
            // SAFETY: batch members are unique, nonempty, disjoint flat ranges.
            // Grid Y selects exactly one member and each generated body checks
            // its own length. All weight allocations and tracked state buffers
            // remain alive until the ordered stream has completed the update.
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (
                        batch.max_len.div_ceil(256),
                        u32::try_from(batch.parameters.len())?,
                        1,
                    ),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ptx::resident::{AdamGroup, ResidentAdam};
    use crate::tensor::Parameter;
    use std::time::Instant;

    fn owner(lengths: &[usize], batch: usize) -> Option<ResidentAdam> {
        let parameters = lengths
            .iter()
            .enumerate()
            .map(|(i, &len)| {
                let data = (0..len)
                    .map(|j| (j % 11) as f32 * 0.03125 - 0.125)
                    .collect();
                (Parameter::new(data, vec![len]), i % 2)
            })
            .collect();
        let mut a = AdamGroup::new(0.003);
        a.weight_decay = 0.01;
        let mut b = AdamGroup::new(0.007);
        b.beta1 = 0.8;
        let mut owner = match ResidentAdam::new(parameters, vec![a, b]) {
            Ok(owner) => owner,
            Err(error)
                if std::env::var("TNSR_REQUIRE_CUDA").as_deref() != Ok("1")
                    && error
                        .to_string()
                        .starts_with("initialize resident training CUDA device") =>
            {
                return None;
            }
            Err(error) => panic!("{error:#}"),
        };
        owner.set_update_batch_size(batch).unwrap();
        Some(owner)
    }

    fn gradients(owner: &mut ResidentAdam) {
        owner.begin_step().unwrap();
        let mut data = vec![0.0; owner.padded_len];
        for p in &owner.parameters {
            for (j, value) in data[p.range.clone()].iter_mut().enumerate() {
                *value = [0.25, -0.5, 0.0, 1.0][j % 4];
            }
        }
        owner
            .stream
            .memcpy_htod(&data, Arc::get_mut(&mut owner.gradients).unwrap())
            .unwrap();
        owner.stream.synchronize().unwrap();
    }

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|v| v.to_bits()).collect()
    }

    #[test]
    fn batched_adam_preserves_groups_tails_moments_and_retained_versions() {
        let lengths = [1, 3, 17, 33, 129, 255, 257, 1023, 1024, 4096, 4097, 8193];
        let Some(mut scalar) = owner(&lengths, 1) else {
            return;
        };
        let mut batch = owner(&lengths, 8).unwrap();
        assert!(batch.set_update_batch_size(0).is_err());
        assert!(batch.set_update_batch_size(17).is_err());
        let retained = batch.parameters[0].weight.clone();
        let old = batch.stream.memcpy_dtov(retained.as_ref()).unwrap();
        for step in 0..4 {
            if step == 2 {
                batch.set_update_batch_size(3).unwrap();
            }
            gradients(&mut scalar);
            gradients(&mut batch);
            let reference = scalar.step(0.25).unwrap();
            let actual = batch.step(0.25).unwrap();
            assert_eq!(
                reference.gradient_norm.to_bits(),
                actual.gradient_norm.to_bits()
            );
            assert_eq!(
                reference.gradient_scale.to_bits(),
                actual.gradient_scale.to_bits()
            );
            assert_eq!(reference.timestep, actual.timestep);
            assert!(actual.kernel_launches < reference.kernel_launches);
            for (a, b) in scalar.parameters.iter().zip(&batch.parameters) {
                assert_eq!(
                    bits(&a.storage.lock().unwrap()),
                    bits(&b.storage.lock().unwrap())
                );
            }
            assert_eq!(
                bits(&scalar.stream.memcpy_dtov(&scalar.moment).unwrap()),
                bits(&batch.stream.memcpy_dtov(&batch.moment).unwrap())
            );
            assert_eq!(
                bits(&scalar.stream.memcpy_dtov(&scalar.variance).unwrap()),
                bits(&batch.stream.memcpy_dtov(&batch.variance).unwrap())
            );
        }
        assert_eq!(
            bits(&old),
            bits(&batch.stream.memcpy_dtov(retained.as_ref()).unwrap())
        );
        batch.set_update_batch_size(1).unwrap();
        gradients(&mut batch);
        batch.step(0.25).unwrap();
    }

    #[test]
    fn batched_adam_rejects_nonfinite_staging_without_publication() {
        let Some(mut owner) = owner(&[17, 33, 65, 129], 8) else {
            return;
        };
        for p in &owner.parameters {
            let values = vec![2.0; p.range.len()];
            p.storage.lock().unwrap().copy_from_slice(&values);
        }
        gradients(&mut owner);
        for group in &mut owner.groups {
            group.weight_decay = f32::MAX;
        }
        assert!(owner.step(0.0).is_err());
        assert_eq!(owner.timestep, 0);
        assert!(
            owner
                .stream
                .memcpy_dtov(&owner.moment)
                .unwrap()
                .iter()
                .all(|v| *v == 0.0)
        );
        assert!(
            owner
                .parameters
                .iter()
                .all(|p| p.storage.lock().unwrap().iter().all(|v| *v == 2.0))
        );
        for group in &mut owner.groups {
            group.weight_decay = 0.0;
        }
        gradients(&mut owner);
        assert_eq!(owner.step(0.0).unwrap().timestep, 1);
    }

    #[test]
    #[ignore = "manual GPU optimizer-step timing"]
    fn resident_batched_update_timing() {
        for count in [16, 64, 128] {
            let lengths: &[usize] = if std::env::var("TNSR_ADAM_BENCH_LARGE").as_deref() == Ok("1")
            {
                &[4096, 8192]
            } else {
                &[16, 256, 1024]
            };
            for &len in lengths {
                let order = if std::env::var("TNSR_ADAM_BENCH_REVERSE").as_deref() == Ok("1") {
                    [8, 1]
                } else {
                    [1, 8]
                };
                for batch_size in order {
                    let mut owner = owner(&vec![len; count], batch_size).unwrap();
                    for _ in 0..3 {
                        gradients(&mut owner);
                        owner.step(1.0).unwrap();
                    }
                    let mut ns = Vec::new();
                    let mut launches = 0;
                    for _ in 0..24 {
                        gradients(&mut owner);
                        let started = Instant::now();
                        launches = owner.step(1.0).unwrap().kernel_launches;
                        ns.push(started.elapsed().as_nanos() as f64);
                    }
                    let mean = ns.iter().sum::<f64>() / ns.len() as f64;
                    let variance =
                        ns.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (ns.len() - 1) as f64;
                    println!(
                        "{}",
                        serde_json::json!({"count":count,"len":len,"batch":batch_size,
                        "mean_ns":mean,"ci95_ns":[mean-2.2*(variance/ns.len() as f64).sqrt(),mean+2.2*(variance/ns.len() as f64).sqrt()],"launches":launches})
                    );
                }
            }
        }
    }
}
