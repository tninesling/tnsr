//! Owner-only, optional vector dispatch. Cached objects contain code, never data pointers.
use super::*;
use cudarc::driver::{CudaFunction, CudaView, CudaViewMut, DevicePtr, DeviceRepr};
use half::bf16;

#[derive(Clone, Debug, serde::Serialize)]
pub struct VectorBf16FirstAdamResources {
    pub elements: usize,
    pub registers: i32,
    pub shared_bytes: i32,
    pub local_bytes: i32,
    pub max_threads: i32,
    /// Driver capacity bound, not measured occupancy.
    pub capacity_at_256: u32,
}
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct VectorBf16FirstAdamStatus {
    pub enabled: bool,
    pub submitted_parameters: usize,
    pub submitted_elements: usize,
    pub committed_parameters: usize,
    pub committed_elements: usize,
    pub fallback_parameters: usize,
    pub batched_f32_parameters: usize,
    pub fallback_reasons: BTreeMap<String, usize>,
    pub resources: Vec<VectorBf16FirstAdamResources>,
}
struct Compiled {
    // CudaFunction owns an Arc<CudaModule>; retaining this code handle keeps
    // the qualified module alive without caching any launch/data bindings.
    function: CudaFunction,
    resources: VectorBf16FirstAdamResources,
}
#[derive(Default)]
pub(in crate::ptx::resident) struct State {
    pub enabled: bool,
    pub status: VectorBf16FirstAdamStatus,
    cache: HashMap<usize, std::result::Result<Compiled, String>>,
}
impl State {
    pub fn reset_status(&mut self) {
        self.status = VectorBf16FirstAdamStatus {
            enabled: self.enabled,
            ..VectorBf16FirstAdamStatus::default()
        };
    }
    pub fn fallback(&mut self, reason: &str) {
        if self.enabled {
            self.status.fallback_parameters += 1;
            *self
                .status
                .fallback_reasons
                .entry(reason.into())
                .or_default() += 1;
        }
    }
    pub fn prepare(&mut self, device: &Arc<CudaContext>, lengths: &[usize]) {
        if !self.enabled {
            return;
        }
        for &len in lengths {
            self.cache.entry(len).or_insert_with(|| {
                (|| -> Result<Compiled> {
                    let kernels = kernels::Kernels::compile(device, vec![ir(len)?])?;
                    let name = kernels
                        .names
                        .first()
                        .context("vector Adam function absent")?;
                    let function = kernels.module.load_function(name)?;
                    let resources = VectorBf16FirstAdamResources {
                        elements: len,
                        registers: function.num_regs()?,
                        shared_bytes: function.shared_size_bytes()?,
                        local_bytes: function.local_size_bytes()?,
                        max_threads: function.max_threads_per_block()?,
                        capacity_at_256: function
                            .occupancy_max_active_blocks_per_multiprocessor(256, 0, None)?,
                    };
                    ensure!(
                        resources.registers > 0
                            && resources.registers <= 128
                            && resources.shared_bytes == 32
                            && resources.local_bytes == 0
                            && resources.max_threads >= 256
                            && resources.capacity_at_256 >= 2,
                        "vector Adam resources outside qualified bounds: {resources:?}"
                    );
                    Ok(Compiled {
                        function,
                        resources,
                    })
                })()
                .map_err(|error| format!("{error:#}"))
            });
        }
    }
    pub fn function(&mut self, len: usize) -> Option<CudaFunction> {
        if !self.enabled {
            return None;
        }
        let Some(Ok(compiled)) = self.cache.get(&len) else {
            self.fallback("kernel_or_resources_unavailable");
            return None;
        };
        if !self.status.resources.iter().any(|r| r.elements == len) {
            self.status.resources.push(compiled.resources.clone());
        }
        Some(compiled.function.clone())
    }
}
fn span<T: DeviceRepr, S: DevicePtr<T>>(value: &S, stream: &Arc<CudaStream>) -> Option<Span> {
    if !Arc::ptr_eq(value.stream().context(), stream.context()) {
        return None;
    }
    Some(Span {
        address: usize::try_from(value.device_ptr(stream).0).ok()?,
        bytes: value.len().checked_mul(std::mem::size_of::<T>())?,
    })
}
/// All borrowed references are live for this call and launch. No address escapes.
/// The BF16 view has already passed the existing completed-run/zero-offset owner
/// proof; this check adds vector alignment and current read/write disjointness.
#[allow(clippy::too_many_arguments)]
pub(in crate::ptx::resident) fn bindings_admitted(
    len: usize,
    stream: &Arc<CudaStream>,
    weight: &CudaSlice<f32>,
    gradient: &CudaView<'_, bf16>,
    moment: &CudaView<'_, f32>,
    variance: &CudaView<'_, f32>,
    settings: &CudaSlice<f32>,
    next_weight: &CudaSlice<f32>,
    next_moment: &CudaViewMut<'_, f32>,
    next_variance: &CudaViewMut<'_, f32>,
    health: &CudaViewMut<'_, f32>,
) -> bool {
    let Some(spans) = (|| -> Option<[Span; 8]> {
        Some([
            span(weight, stream)?,
            span(moment, stream)?,
            span(variance, stream)?,
            span(gradient, stream)?,
            span(next_weight, stream)?,
            span(next_moment, stream)?,
            span(next_variance, stream)?,
            span(health, stream)?,
        ])
    })() else {
        return false;
    };
    let Some(settings) = span(settings, stream) else {
        return false;
    };
    admitted_with_settings(len, &spans, settings, true)
}
impl ResidentAdam {
    pub fn set_experimental_vector_bf16_first_adam(&mut self, enabled: bool) -> Result<()> {
        ensure!(
            !self.active,
            "select vector BF16-first Adam before begin_step"
        );
        ensure!(
            !enabled || self.borrowed_bf16.enabled,
            "vector BF16-first Adam requires borrowed BF16 Adam source"
        );
        self.vector_bf16_first_adam.enabled = enabled;
        self.vector_bf16_first_adam.reset_status();
        Ok(())
    }
    pub fn vector_bf16_first_adam_status(&self) -> &VectorBf16FirstAdamStatus {
        &self.vector_bf16_first_adam.status
    }
}
