use crate::graph::{TensorGraph, WithGrad};
use crate::{Executor, SimpleExecutor};
use anyhow::Result;
use petgraph::graph::NodeIndex;
use std::collections::HashMap;

#[cfg(feature = "cuda")]
use crate::ptx::PtxExecutor;
#[cfg(feature = "cuda")]
use crate::ptx::types::{BF16, F16, F32};
#[cfg(feature = "cuda")]
use anyhow::Context;

/// Available execution backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// CPU-based execution using SimpleExecutor
    Cpu,
    /// Compatibility alias for tiled PTX execution (requires `cuda` feature)
    #[cfg(feature = "cuda")]
    Cuda,
    /// GPU-based execution using PTX JIT compilation (requires `cuda` feature)
    #[cfg(feature = "cuda")]
    Ptx,
}

/// Unified runtime that automatically selects the best available backend.
///
/// Wraps either a CPU or GPU executor and provides a single consistent interface.
/// Create with [`Runtime::new`] for automatic backend selection, or use
/// [`Runtime::with_backend`] to explicitly choose a backend.
///
pub enum Runtime<D = f32> {
    /// CPU-based executor
    Cpu(SimpleExecutor<D>),
    /// PTX JIT executor
    #[cfg(feature = "cuda")]
    Ptx(PtxExecutor),
    /// Half-precision PTX JIT executor with f32 computation.
    #[cfg(feature = "cuda")]
    PtxF16(PtxExecutor<F16>),
    /// Brain-float PTX JIT executor with f32 computation.
    #[cfg(feature = "cuda")]
    PtxBF16(PtxExecutor<BF16>),
}

impl Runtime<f32> {
    /// Create a runtime with automatic backend selection.
    ///
    /// Prefers tiled PTX execution when `cuda` is enabled, falling back to CPU
    /// if GPU initialization fails. The `ptx` feature is a compatibility alias.
    pub fn new() -> Self {
        #[cfg(feature = "cuda")]
        {
            match PtxExecutor::<F32>::try_new() {
                Ok(executor) => {
                    tracing::info!("Runtime initialized with tiled PTX backend");
                    Runtime::Ptx(executor)
                }
                Err(error) => {
                    tracing::warn!("PTX initialization failed: {}, falling back to CPU", error);
                    Runtime::Cpu(SimpleExecutor::new())
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            Runtime::Cpu(SimpleExecutor::new())
        }
    }

    /// Create a runtime with an explicitly specified backend.
    pub fn with_backend(backend: Backend) -> Result<Self> {
        match backend {
            Backend::Cpu => Ok(Runtime::Cpu(SimpleExecutor::new())),
            #[cfg(feature = "cuda")]
            Backend::Cuda | Backend::Ptx => {
                let executor =
                    PtxExecutor::<F32>::try_new().context("Failed to initialize PTX executor")?;
                Ok(Runtime::Ptx(executor))
            }
        }
    }
}

macro_rules! impl_half_runtime {
    ($dtype:ty, $marker:ty, $variant:ident, $name:literal) => {
        impl Runtime<$dtype> {
            /// Create a half-precision runtime with automatic backend selection.
            pub fn new() -> Self {
                #[cfg(feature = "cuda")]
                {
                    match PtxExecutor::<$marker>::try_new_for_dtype() {
                        Ok(executor) => Runtime::$variant(executor),
                        Err(error) => {
                            tracing::warn!(
                                concat!(
                                    "PTX initialization failed for ",
                                    $name,
                                    ": {}, falling back to CPU"
                                ),
                                error
                            );
                            Runtime::Cpu(SimpleExecutor::new())
                        }
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Runtime::Cpu(SimpleExecutor::new())
                }
            }

            /// Create a half-precision runtime with an explicitly selected backend.
            pub fn with_backend(backend: Backend) -> Result<Self> {
                match backend {
                    Backend::Cpu => Ok(Runtime::Cpu(SimpleExecutor::new())),
                    #[cfg(feature = "cuda")]
                    Backend::Cuda | Backend::Ptx => Ok(Runtime::$variant(
                        PtxExecutor::<$marker>::try_new_for_dtype()
                            .context("Failed to initialize PTX executor")?,
                    )),
                }
            }
        }

        impl Default for Runtime<$dtype> {
            fn default() -> Self {
                Self::new()
            }
        }
    };
}

impl_half_runtime!(half::f16, F16, PtxF16, "f16");
impl_half_runtime!(half::bf16, BF16, PtxBF16, "bf16");

impl<D> Runtime<D>
where
    D: num_traits::Float + Send + Sync + 'static,
{
    /// Returns the backend currently in use by this runtime.
    pub fn backend(&self) -> Backend {
        match self {
            Runtime::Cpu(_) => Backend::Cpu,
            #[cfg(feature = "cuda")]
            Runtime::Ptx(_) => Backend::Ptx,
            #[cfg(feature = "cuda")]
            Runtime::PtxF16(_) | Runtime::PtxBF16(_) => Backend::Ptx,
        }
    }

    /// Returns `true` if this runtime is using the CPU backend.
    pub fn is_cpu(&self) -> bool {
        matches!(self, Runtime::Cpu(_))
    }

    /// Returns `true` if this runtime is using the CUDA backend.
    pub fn is_cuda(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            matches!(
                self,
                Runtime::Ptx(_) | Runtime::PtxF16(_) | Runtime::PtxBF16(_)
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }

    /// Get the computed value for a specific node after execution.
    ///
    pub fn get_value(&self, node_idx: NodeIndex) -> Option<Vec<D>> {
        match self {
            Runtime::Cpu(executor) => executor.get_value(node_idx),
            #[cfg(feature = "cuda")]
            Runtime::Ptx(executor) => executor.get_value(node_idx).map(|values| {
                values
                    .into_iter()
                    .map(|value| D::from(value).unwrap())
                    .collect()
            }),
            #[cfg(feature = "cuda")]
            Runtime::PtxF16(executor) => executor.get_value(node_idx).map(|values| {
                values
                    .into_iter()
                    .map(|value| D::from(value.to_f32()).unwrap())
                    .collect()
            }),
            #[cfg(feature = "cuda")]
            Runtime::PtxBF16(executor) => executor.get_value(node_idx).map(|values| {
                values
                    .into_iter()
                    .map(|value| D::from(value.to_f32()).unwrap())
                    .collect()
            }),
        }
    }
}

impl Default for Runtime<f32> {
    fn default() -> Self {
        Self::new()
    }
}

impl<D> Executor<D> for Runtime<D>
where
    D: num_traits::Float + Send + Sync + 'static,
{
    fn execute<G>(
        &mut self,
        graph: &TensorGraph<D, G>,
        inputs: HashMap<String, Vec<D>>,
    ) -> Result<Vec<D>>
    where
        TensorGraph<D, G>: Clone,
        TensorGraph<f32, G>: Clone,
    {
        match self {
            Runtime::Cpu(executor) => executor.execute(graph, inputs),
            #[cfg(feature = "cuda")]
            Runtime::Ptx(executor) => execute_native_ptx(executor, graph, inputs, HashMap::new()),
            #[cfg(feature = "cuda")]
            Runtime::PtxF16(executor) => {
                execute_native_ptx(executor, graph, inputs, HashMap::new())
            }
            #[cfg(feature = "cuda")]
            Runtime::PtxBF16(executor) => {
                execute_native_ptx(executor, graph, inputs, HashMap::new())
            }
        }
    }

    fn execute_with_indices<G>(
        &mut self,
        graph: &TensorGraph<D, G>,
        inputs: HashMap<String, Vec<D>>,
        indices: crate::tensor::IndexInputs,
    ) -> Result<Vec<D>>
    where
        TensorGraph<D, G>: Clone,
        TensorGraph<f32, G>: Clone,
    {
        match self {
            Runtime::Cpu(executor) => executor.execute_with_indices(graph, inputs, indices),
            #[cfg(feature = "cuda")]
            Runtime::Ptx(executor) => execute_native_ptx(executor, graph, inputs, indices),
            #[cfg(feature = "cuda")]
            Runtime::PtxF16(executor) => execute_native_ptx(executor, graph, inputs, indices),
            #[cfg(feature = "cuda")]
            Runtime::PtxBF16(executor) => execute_native_ptx(executor, graph, inputs, indices),
        }
    }

    fn get_gradients(&self, graph: &TensorGraph<D, WithGrad>) -> HashMap<usize, Vec<D>> {
        match self {
            Runtime::Cpu(executor) => executor.get_gradients(graph),
            #[cfg(feature = "cuda")]
            Runtime::Ptx(executor) => graph
                .gradient_metadata()
                .param_to_grad
                .iter()
                .filter_map(|(param_id, node_idx)| {
                    executor.get_value(*node_idx).map(|values| {
                        (
                            *param_id,
                            values
                                .into_iter()
                                .map(|value| D::from(value).unwrap())
                                .collect(),
                        )
                    })
                })
                .collect(),
            #[cfg(feature = "cuda")]
            Runtime::PtxF16(executor) => graph
                .gradient_metadata()
                .param_to_grad
                .iter()
                .filter_map(|(param_id, node_idx)| {
                    executor.get_value(*node_idx).map(|values| {
                        (
                            *param_id,
                            values
                                .into_iter()
                                .map(|value| D::from(value.to_f32()).unwrap())
                                .collect(),
                        )
                    })
                })
                .collect(),
            #[cfg(feature = "cuda")]
            Runtime::PtxBF16(executor) => graph
                .gradient_metadata()
                .param_to_grad
                .iter()
                .filter_map(|(param_id, node_idx)| {
                    executor.get_value(*node_idx).map(|values| {
                        (
                            *param_id,
                            values
                                .into_iter()
                                .map(|value| D::from(value.to_f32()).unwrap())
                                .collect(),
                        )
                    })
                })
                .collect(),
        }
    }
}

/// Native PTX buffers already use the runtime scalar type. Preserve their
/// ownership and parameter sharing instead of converting the graph on each call.
#[cfg(feature = "cuda")]
fn execute_native_ptx<M: crate::ptx::types::CudaDType, D: 'static, G>(
    executor: &mut PtxExecutor<M>,
    graph: &TensorGraph<D, G>,
    inputs: HashMap<String, Vec<D>>,
    indices: crate::tensor::IndexInputs,
) -> Result<Vec<D>> {
    anyhow::ensure!(
        std::any::TypeId::of::<D>() == std::any::TypeId::of::<M::HostType>(),
        "PTX executor storage dtype does not match the runtime dtype"
    );
    // SAFETY: TypeId equality proves D and M::HostType are the same type.
    // G is unchanged; all graph and buffer layouts and ownership are identical.
    let graph =
        unsafe { &*(graph as *const TensorGraph<D, G> as *const TensorGraph<M::HostType, G>) };
    let inputs = unsafe {
        std::mem::transmute::<HashMap<String, Vec<D>>, HashMap<String, Vec<M::HostType>>>(inputs)
    };
    let output = executor.compile_and_execute_with_indices(graph, inputs, indices)?;
    // SAFETY: The checked scalar types are identical; transfer the allocation.
    Ok(unsafe { std::mem::transmute::<Vec<M::HostType>, Vec<D>>(output) })
}
