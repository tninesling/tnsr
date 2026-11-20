//! Optional single-microbatch standalone Adam source. Never owns GPU gradients.
use super::*;
use crate::graph::NodeIndex;
use crate::ptx::mixed::{CompletedRunStamp, MixedExecutor};
use cudarc::driver::CudaView;
use half::bf16;

#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct BorrowedBf16AdamStatus {
    pub enabled: bool,
    pub standalone_bf16_parameters: usize,
    pub standalone_f32_parameters: usize,
    pub batched_f32_parameters: usize,
    pub bf16_gradient_elements: usize,
    pub gradient_read_bytes: usize,
    /// Logical source payload difference only, not measured DRAM or allocation.
    pub logical_gradient_read_bytes_saved: usize,
    pub reason: String,
}
pub(super) struct Candidate {
    pub stamp: CompletedRunStamp,
    pub nodes: Vec<NodeIndex>,
}
#[derive(Default)]
pub(super) struct BorrowState {
    pub enabled: bool,
    pub candidate: Option<Candidate>,
    pub kernels: Option<kernels::Kernels>,
    pub ids: HashMap<usize, usize>,
    pub status: BorrowedBf16AdamStatus,
    pub limits: HashMap<usize, usize>,
}
impl BorrowState {
    pub fn invalidate(&mut self) {
        self.candidate = None;
    }
    pub fn reset_status(&mut self, reason: &str) {
        self.status = BorrowedBf16AdamStatus {
            enabled: self.enabled,
            reason: reason.into(),
            ..BorrowedBf16AdamStatus::default()
        };
    }
    pub fn compile_missing(&mut self, device: &Arc<CudaContext>, lengths: &[usize]) -> Result<()> {
        let mut all = self.ids.keys().copied().collect::<Vec<_>>();
        all.extend(lengths);
        all.sort_unstable();
        all.dedup();
        if all.len() == self.ids.len() {
            return Ok(());
        }
        let ir = all
            .iter()
            .map(|&n| kernels::adam_bf16_first(n))
            .collect::<Result<Vec<_>>>()?;
        let compiled = kernels::Kernels::compile(device, ir)?;
        let mut limits = HashMap::new();
        for (index, &len) in all.iter().enumerate() {
            let function = compiled.module.load_function(&compiled.names[index])?;
            let maximum = if function.local_size_bytes()? == 0 {
                usize::try_from(function.max_threads_per_block()?)?
            } else {
                0
            };
            limits.insert(len, maximum);
        }
        let ids = all.into_iter().enumerate().map(|(i, n)| (n, i)).collect();
        self.kernels = Some(compiled);
        self.ids = ids;
        self.limits = limits;
        Ok(())
    }
}
impl ResidentAdam {
    pub fn set_experimental_borrowed_bf16_adam_source(&mut self, enabled: bool) -> Result<()> {
        self.borrowed_bf16.invalidate();
        ensure!(
            !self.active,
            "select borrowed BF16 Adam source before begin_step"
        );
        ensure!(
            enabled || !self.vector_bf16_first_adam.enabled,
            "disable vector BF16-first Adam before disabling borrowed source"
        );
        self.borrowed_bf16.enabled = enabled;
        self.borrowed_bf16
            .reset_status(if enabled { "not_used" } else { "disabled" });
        Ok(())
    }
    pub fn borrowed_bf16_adam_status(&self) -> &BorrowedBf16AdamStatus {
        &self.borrowed_bf16.status
    }
    /// Borrow only for this call. Wrong/stale/unsupported sources use the
    /// established accumulated F32 gradient path. No raw address is recorded.
    pub fn step_mixed(
        &mut self,
        executor: &MixedExecutor,
        max_grad_norm: f32,
    ) -> Result<ResidentStep> {
        let candidate = self.borrowed_bf16.candidate.take();
        self.borrowed_bf16.reset_status("stale_or_unsupported");
        let matching = self.borrowed_bf16.enabled
            && self.active
            && !self.accumulation_failed
            && self.norm_partial_validity.single_attempt()
            && Arc::ptr_eq(executor.execution_stream(), &self.stream)
            && Arc::ptr_eq(executor.execution_stream().context(), &self.device)
            && self
                .parameters
                .iter()
                .all(|p| p.storage.revision().ok() == Some(p.revision));
        let sources: Vec<Option<CudaView<'_, bf16>>> = if matching {
            candidate
                .filter(|c| {
                    c.nodes.len() == self.parameters.len()
                        && executor.matches_borrowed_gradient_stamp(&c.stamp)
                })
                .map(|c| {
                    c.nodes
                        .iter()
                        .zip(&self.parameters)
                        .enumerate()
                        .map(|(index, (&node, p))| {
                            if self
                                .batched_updates
                                .as_ref()
                                .is_some_and(|u| u.covers(index))
                            {
                                None
                            } else {
                                executor.borrowed_bf16_gradient(node, p.range.len())
                            }
                        })
                        .collect()
                })
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        if !sources.is_empty() {
            self.borrowed_bf16.status.reason = if sources.iter().any(Option::is_some) {
                "fresh_single_mixed_accumulation"
            } else {
                "fresh_no_supported_bf16_source"
            }
            .into();
        }
        // Existing norm/health helpers lock the same gate internally: do not
        // nest an outer gate. The owner's serialized-writer contract remains.
        let result = self.step_impl(max_grad_norm, Some(&sources));
        // Success already synchronized via the health scalar readback. Errors
        // may have queued only a prefix: finish that work before ending borrow.
        let finished = if result.is_err() {
            self.stream.synchronize()
        } else {
            Ok(())
        };
        // Successful health readback completes the reads. On CUDA errors the
        // cleanup synchronization is attempted and both error contexts retained.
        drop(sources);
        match (result, finished) {
            (Err(error), Err(cleanup)) => Err(error.context(format!(
                "borrowed-source cleanup synchronization also failed: {cleanup}"
            ))),
            (Err(error), Ok(())) => Err(error),
            (Ok(_), Err(error)) => Err(error.into()),
            (Ok(step), Ok(())) => Ok(step),
        }
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod gpu_tests;

#[cfg(test)]
mod model_tests;
