//! Opt-in horizontal composition of first-overwrite accumulation/norm rows.
use super::*;
use crate::ptx::linear::Binding;
use crate::tile::DType;
use cudarc::driver::{CudaFunction, DevicePtr};
use std::collections::HashSet;
mod ir;
const SMALL_ELEMENTS: usize = 4096;
const BATCH_MEMBERS: usize = 8;
type Key = Vec<(VirtualTensor, Shape)>;
type Source = (Binding, VirtualTensor, Shape);
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct SmallAccumulationNormBatchStatus {
    pub enabled: bool,
    pub last_accumulation_used: bool,
    pub selected_members: usize,
    pub batched_launches: usize,
    pub standalone_launches: usize,
    pub baseline_launches: usize,
    pub actual_launches: usize,
    pub fallback_reason: String,
}
#[derive(Default)]
pub(super) struct State {
    pub enabled: bool,
    status: SmallAccumulationNormBatchStatus,
    cache: HashMap<Key, std::result::Result<Prepared, String>>,
}
impl State {
    pub fn record_standalone(&mut self) {
        if self.enabled {
            self.status.standalone_launches += 1;
            self.status.actual_launches += 1;
        }
    }
    pub fn begin_step(&mut self) {
        if self.enabled {
            self.status = SmallAccumulationNormBatchStatus {
                enabled: true,
                fallback_reason: "no_accumulation".into(),
                ..Default::default()
            };
        }
    }
}
struct Step {
    function: CudaFunction,
    members: Vec<usize>,
    config: LaunchConfig,
}
struct Prepared {
    _kernels: kernels::Kernels,
    steps: Vec<Step>,
}
fn span(range: &Range<usize>) -> Result<usize> {
    ensure!(
        range.start < range.end,
        "nonempty ordered accumulation range required"
    );
    range
        .end
        .div_ceil(norm_partials::ROW_ELEMENTS)
        .checked_sub(range.start / norm_partials::ROW_ELEMENTS)
        .context("row span underflow")
}
fn volume(s: &[usize]) -> Option<usize> {
    s.iter().try_fold(1usize, |n, &d| n.checked_mul(d))
}
fn source_proof(view: &VirtualTensor, shape: &[usize], len: usize) -> Result<()> {
    ensure!(
        matches!(view.dtype, DType::F32 | DType::BF16)
            && view.predicate.is_none()
            && view.out_of_bounds.is_none(),
        "small source dtype/predicate unsupported"
    );
    ensure!(
        volume(shape) == Some(len) && volume(&view.shape) == Some(len),
        "small source must cover exact logical binding"
    );
    let identity = VirtualTensor::identity(view.source, shape.to_vec(), view.dtype);
    // A no-op reshape rewrites singleton coordinates to zero, while a direct
    // identity retains IterDim. Use the exact identity map for equal domains;
    // retain the original normalized reshape proof for all different shapes.
    let expected = if shape == view.shape.as_slice() {
        identity
    } else {
        identity.reshape(view.shape.clone())?
    };
    ensure!(
        expected.access.normalize_in_domain(&view.shape)?
            == view.access.clone().normalize_in_domain(&view.shape)?,
        "small source must have proven whole contiguous map"
    );
    Ok(())
}
fn groups(ranges: &[Range<usize>]) -> Vec<Vec<usize>> {
    let small = ranges
        .iter()
        .enumerate()
        .filter(|(_, r)| r.len() <= SMALL_ELEMENTS)
        .map(|(i, _)| i)
        .collect::<Vec<_>>();
    small
        .chunks(BATCH_MEMBERS)
        .filter(|c| c.len() > 1)
        .map(<[usize]>::to_vec)
        .collect()
}
fn bounds(pointer: u64, len: usize, dtype: DType) -> Result<(u64, u64)> {
    ensure!(
        pointer > 0 && pointer.is_multiple_of(u64::try_from(dtype.size_bytes())?),
        "source alignment/base invalid"
    );
    let bytes = len
        .checked_mul(dtype.size_bytes())
        .context("source span overflow")?;
    ensure!(bytes > 0, "source empty");
    Ok((
        pointer,
        pointer
            .checked_add(u64::try_from(bytes)?)
            .context("source address overflow")?,
    ))
}
fn overlap(a: (u64, u64), b: (u64, u64)) -> bool {
    a.0 < b.1 && b.0 < a.1
}
fn preflight(
    values: &[Source],
    ranges: &[Range<usize>],
    padded_len: usize,
    gradient: u64,
    partials: u64,
    rows: usize,
) -> Result<()> {
    ensure!(values.len() == ranges.len(), "source/member count changed");
    let g = bounds(gradient, padded_len, DType::F32)?;
    let p = bounds(partials, rows, DType::F32)?;
    ensure!(!overlap(g, p), "gradient/partial outputs alias");
    for ((binding, view, _), range) in values.iter().zip(ranges) {
        ensure!(
            binding.dtype == view.dtype && binding.len == range.len(),
            "source typed binding mismatch"
        );
        let s = bounds(binding.pointer, binding.len, binding.dtype)?;
        ensure!(
            !overlap(s, g) && !overlap(s, p),
            "source aliases mutable accumulation outputs"
        );
    }
    Ok(())
}
impl Prepared {
    fn new(
        device: &Arc<CudaContext>,
        values: &[Source],
        ranges: &[Range<usize>],
        padded: usize,
    ) -> Result<Self> {
        ensure!(values.len() == ranges.len(), "batch source count mismatch");
        let batches = groups(ranges);
        ensure!(
            !batches.is_empty(),
            "fewer than two supported small members"
        );
        // Complete old layout validation checks contiguous, unique flat ranges.
        let _ = norm_partials::PartialLayout::new(ranges.to_vec(), padded)?;
        let covered = batches.iter().flatten().copied().collect::<HashSet<_>>();
        for &i in &covered {
            source_proof(&values[i].1, &values[i].2, ranges[i].len())?;
        }
        let mut kernels_ir = Vec::new();
        let mut descriptors = Vec::new();
        for members in &batches {
            let inputs = members
                .iter()
                .map(|&i| (values[i].1.clone(), values[i].2.clone(), ranges[i].clone()))
                .collect::<Vec<_>>();
            kernels_ir.push(ir::batch_ir(&inputs, padded)?);
            let q = members
                .iter()
                .map(|&i| span(&ranges[i]))
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .max()
                .context("empty batch")?;
            descriptors.push((
                members.clone(),
                LaunchConfig {
                    grid_dim: (u32::try_from(q)?, u32::try_from(members.len())?, 1),
                    block_dim: (norm_partials::ROW_THREADS as u32, 1, 1),
                    shared_mem_bytes: 0,
                },
            ));
        }
        for (i, range) in ranges
            .iter()
            .enumerate()
            .filter(|(i, _)| !covered.contains(i))
        {
            kernels_ir.push(norm_partials::accumulate_ir(
                &values[i].1,
                &values[i].2,
                range.clone(),
            )?);
            descriptors.push((
                vec![i],
                LaunchConfig {
                    grid_dim: (u32::try_from(span(range)?)?, 1, 1),
                    block_dim: (norm_partials::ROW_THREADS as u32, 1, 1),
                    shared_mem_bytes: 0,
                },
            ));
        }
        let compiled = kernels::Kernels::compile(device, kernels_ir)?;
        let mut steps = Vec::new();
        for (i, (members, config)) in descriptors.into_iter().enumerate() {
            let f = compiled.module.load_function(&compiled.names[i])?;
            // Admission is complete BEFORE the first member writes anything.
            ensure!(
                f.max_threads_per_block()? >= config.block_dim.0 as i32
                    && f.local_size_bytes()? == 0
                    && f.num_regs()? <= 128
                    && f.shared_size_bytes()? <= i32::try_from(norm_partials::ROW_THREADS * 4)?,
                "small accumulation resource admission failed"
            );
            ensure!(
                f.occupancy_max_active_blocks_per_multiprocessor(config.block_dim.0, 0, None)? > 0,
                "small accumulation has no resident block capacity"
            );
            steps.push(Step {
                function: f,
                members,
                config,
            });
        }
        Ok(Self {
            _kernels: compiled,
            steps,
        })
    }
}
impl ResidentAdam {
    pub fn set_experimental_small_accumulation_norm_batches(
        &mut self,
        enabled: bool,
    ) -> Result<()> {
        self.borrowed_bf16.invalidate();
        ensure!(
            !self.active,
            "select small accumulation norm batching before begin_step"
        );
        ensure!(
            !enabled || (self.first_accumulation_overwrite && self.gradient_norm_partials),
            "small accumulation norm batches require first overwrite and norm partials"
        );
        self.small_accumulation_norm.enabled = enabled;
        self.small_accumulation_norm.status = SmallAccumulationNormBatchStatus {
            enabled,
            ..Default::default()
        };
        Ok(())
    }
    pub fn small_accumulation_norm_batches_status(&self) -> &SmallAccumulationNormBatchStatus {
        &self.small_accumulation_norm.status
    }
    pub(super) fn small_accumulation_norm_reset(&mut self) {
        if self.small_accumulation_norm.enabled {
            self.small_accumulation_norm.status = SmallAccumulationNormBatchStatus {
                enabled: true,
                baseline_launches: self.parameters.len(),
                actual_launches: 0,
                standalone_launches: 0,
                fallback_reason: "not_first_overwrite_norm_attempt".into(),
                ..Default::default()
            };
        }
    }
    /// False means no writes occurred; caller may use the exact old path. Once
    /// any launch is attempted, errors propagate into the invalid-attempt path.
    pub(super) fn try_small_accumulation_norm(&mut self, values: &[Source]) -> Result<bool> {
        let ranges = self
            .parameters
            .iter()
            .map(|p| p.range.clone())
            .collect::<Vec<_>>();
        let key = values
            .iter()
            .map(|(_, v, s)| (v.clone(), s.clone()))
            .collect::<Vec<_>>();
        if !self.small_accumulation_norm.cache.contains_key(&key) {
            let result = Prepared::new(&self.device, values, &ranges, self.padded_len)
                .map_err(|e| format!("{e:#}"));
            if self.small_accumulation_norm.cache.len() >= 8 {
                self.small_accumulation_norm.cache.clear();
            }
            self.small_accumulation_norm
                .cache
                .insert(key.clone(), result);
        }
        let prepared = match &self.small_accumulation_norm.cache[&key] {
            Ok(p) => p,
            Err(reason) => {
                self.small_accumulation_norm.status.fallback_reason = reason.clone();
                return Ok(false);
            }
        };
        let gate = self.stream_gate.clone();
        let _guard = gate
            .lock()
            .map_err(|_| anyhow::anyhow!("small accumulation stream lock poisoned"))?;
        let gradient = Arc::get_mut(&mut self.gradients).context("resident gradients borrowed")?;
        let workspace = self
            .norm_partial_workspace
            .as_mut()
            .context("norm partial workspace absent")?;
        let rows = workspace.layout.rows;
        let partials = workspace.partials_mut()?;
        if let Err(reason) = preflight(
            values,
            &ranges,
            self.padded_len,
            gradient.device_ptr(&self.stream).0,
            partials.device_ptr(&self.stream).0,
            rows,
        ) {
            self.small_accumulation_norm.status.fallback_reason = format!("{reason:#}");
            return Ok(false);
        }
        self.small_accumulation_norm.status = SmallAccumulationNormBatchStatus {
            enabled: true,
            last_accumulation_used: true,
            selected_members: 0,
            batched_launches: 0,
            standalone_launches: 0,
            baseline_launches: self.parameters.len(),
            actual_launches: 0,
            fallback_reason: String::new(),
        };
        for step in &prepared.steps {
            let mut launch = self.stream.launch_builder(&step.function);
            if step.members.len() > 1 {
                for &i in &step.members {
                    launch.arg(&values[i].0.pointer);
                }
                // Exactly ONE tracked mutable global gradient argument.
                launch.arg(&mut *gradient).arg(&mut *partials);
                // SAFETY: all sources are live in Mixed on this ordered stream;
                // full writable spans were checked disjoint before enqueue.
                unsafe { launch.launch(step.config) }?;
            } else {
                let i = step.members[0];
                let mut target = gradient.slice_mut(ranges[i].clone());
                launch
                    .arg(&values[i].0.pointer)
                    .arg(&mut target)
                    .arg(&mut *partials);
                // SAFETY: this is the unchanged standalone member ABI/config.
                unsafe { launch.launch(step.config) }?;
            }
            self.kernel_launches += 1;
            self.small_accumulation_norm.status.actual_launches += 1;
            if step.members.len() > 1 {
                self.small_accumulation_norm.status.batched_launches += 1;
                self.small_accumulation_norm.status.selected_members += step.members.len();
            } else {
                self.small_accumulation_norm.status.standalone_launches += 1;
            }
        }
        Ok(true)
    }
}
#[cfg(test)]
mod tests;
