use crate::graph::{TensorGraph, WithGrad};
use crate::{Executor, SimpleExecutor};
use anyhow::Result;
use petgraph::graph::NodeIndex;
use std::collections::HashMap;

#[cfg(feature = "cuda")]
use crate::cuda::CudaExecutor;
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
    /// GPU-based execution using CUDA (requires `cuda` feature)
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
    /// GPU-based executor
    #[cfg(feature = "cuda")]
    Cuda(CudaExecutor),
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
    /// When the `ptx` feature is enabled, prefers the Tile IR/PTX executor.
    /// Otherwise, prefers static CUDA and falls back to CPU if CUDA is unavailable.
    pub fn new() -> Self {
        #[cfg(all(feature = "cuda", feature = "ptx"))]
        {
            // PTX feature enabled: try PTX first, then CUDA, then CPU
            match PtxExecutor::<F32>::try_new() {
                Ok(executor) => {
                    tracing::info!("Runtime initialized with PTX JIT backend");
                    Runtime::Ptx(executor)
                }
                Err(e) => {
                    tracing::warn!("PTX initialization failed: {}, trying CUDA", e);
                    match CudaExecutor::try_new() {
                        Ok(executor) => {
                            tracing::info!("Runtime initialized with CUDA backend");
                            Runtime::Cuda(executor)
                        }
                        Err(e) => {
                            tracing::warn!(
                                "CUDA initialization failed: {}, falling back to CPU",
                                e
                            );
                            Runtime::Cpu(SimpleExecutor::new())
                        }
                    }
                }
            }
        }
        #[cfg(all(feature = "cuda", not(feature = "ptx")))]
        {
            match CudaExecutor::try_new() {
                Ok(executor) => {
                    tracing::info!("Runtime initialized with CUDA backend");
                    Runtime::Cuda(executor)
                }
                Err(e) => {
                    tracing::warn!("CUDA initialization failed: {}, falling back to CPU", e);
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
            Backend::Cuda => {
                let executor =
                    CudaExecutor::try_new().context("Failed to initialize CUDA executor")?;
                Ok(Runtime::Cuda(executor))
            }
            #[cfg(feature = "cuda")]
            Backend::Ptx => {
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
                #[cfg(all(feature = "cuda", feature = "ptx"))]
                {
                    match PtxExecutor::<$marker>::try_new_for_dtype() {
                        Ok(executor) => Runtime::$variant(executor),
                        Err(error) => {
                            tracing::warn!(
                                concat!(
                                    "PTX initialization failed for ",
                                    $name,
                                    ": {}, trying CUDA"
                                ),
                                error
                            );
                            CudaExecutor::try_new()
                                .map_or_else(|_| Runtime::Cpu(SimpleExecutor::new()), Runtime::Cuda)
                        }
                    }
                }
                #[cfg(all(feature = "cuda", not(feature = "ptx")))]
                {
                    match CudaExecutor::try_new() {
                        Ok(executor) => {
                            tracing::info!(concat!(
                                "Runtime initialized with CUDA backend (",
                                $name,
                                ")"
                            ));
                            Runtime::Cuda(executor)
                        }
                        Err(error) => {
                            tracing::warn!(
                                concat!(
                                    "CUDA initialization failed for ",
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
                    Backend::Cuda => Ok(Runtime::Cuda(
                        CudaExecutor::try_new().context("Failed to initialize CUDA executor")?,
                    )),
                    #[cfg(feature = "cuda")]
                    Backend::Ptx => Ok(Runtime::$variant(
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
            Runtime::Cuda(_) => Backend::Cuda,
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
                Runtime::Cuda(_) | Runtime::Ptx(_) | Runtime::PtxF16(_) | Runtime::PtxBF16(_)
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
            Runtime::Cuda(executor) => {
                use std::any::TypeId;
                if TypeId::of::<D>() == TypeId::of::<f32>() {
                    executor
                        .get_value(node_idx)
                        // Safety: D is f32 at runtime, so the transmute is a no-op.
                        .map(|v| unsafe { std::mem::transmute::<Vec<f32>, Vec<D>>(v) })
                } else if TypeId::of::<D>() == TypeId::of::<half::f16>() {
                    executor.get_value(node_idx).map(|values| {
                        let values: Vec<half::f16> =
                            values.into_iter().map(half::f16::from_f32).collect();
                        // Safety: the TypeId check proves D is f16.
                        unsafe { std::mem::transmute::<Vec<half::f16>, Vec<D>>(values) }
                    })
                } else if TypeId::of::<D>() == TypeId::of::<half::bf16>() {
                    executor.get_value(node_idx).map(|values| {
                        let values: Vec<half::bf16> =
                            values.into_iter().map(half::bf16::from_f32).collect();
                        // Safety: the TypeId check proves D is bf16.
                        unsafe { std::mem::transmute::<Vec<half::bf16>, Vec<D>>(values) }
                    })
                } else {
                    None
                }
            }
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
            Runtime::Cuda(executor) => {
                use std::any::TypeId;
                if TypeId::of::<D>() != TypeId::of::<f32>()
                    && TypeId::of::<D>() != TypeId::of::<half::f16>()
                    && TypeId::of::<D>() != TypeId::of::<half::bf16>()
                {
                    anyhow::bail!("CUDA backend only supports f32, f16, and bf16")
                }

                if TypeId::of::<D>() == TypeId::of::<f32>() {
                    // Safety: the TypeId check proves D is f32, so these
                    // container transmutations preserve layout and ownership.
                    let graph = unsafe {
                        &*(graph as *const TensorGraph<D, G> as *const TensorGraph<f32, G>)
                    };
                    let inputs = unsafe {
                        std::mem::transmute::<HashMap<String, Vec<D>>, HashMap<String, Vec<f32>>>(
                            inputs,
                        )
                    };
                    let result = executor.execute(graph, inputs);
                    return unsafe {
                        std::mem::transmute::<Result<Vec<f32>>, Result<Vec<D>>>(result)
                    };
                }

                let graph_f32 = graph.clone().map_dtype(|value| value.to_f32().unwrap());
                let inputs_f32 = inputs
                    .into_iter()
                    .map(|(name, values)| {
                        (
                            name,
                            values
                                .into_iter()
                                .map(|value| value.to_f32().unwrap())
                                .collect(),
                        )
                    })
                    .collect();
                executor
                    .execute(&graph_f32, inputs_f32)?
                    .into_iter()
                    .map(|value| {
                        D::from(value)
                            .context("CUDA result cannot be represented by the requested dtype")
                    })
                    .collect()
            }
            #[cfg(feature = "cuda")]
            Runtime::Ptx(executor) => execute_native_ptx(executor, graph, inputs),
            #[cfg(feature = "cuda")]
            Runtime::PtxF16(executor) => execute_native_ptx(executor, graph, inputs),
            #[cfg(feature = "cuda")]
            Runtime::PtxBF16(executor) => execute_native_ptx(executor, graph, inputs),
        }
    }

    fn get_gradients(&self, graph: &TensorGraph<D, WithGrad>) -> HashMap<usize, Vec<D>> {
        match self {
            Runtime::Cpu(executor) => executor.get_gradients(graph),
            #[cfg(feature = "cuda")]
            Runtime::Cuda(executor) => {
                use std::any::TypeId;
                if TypeId::of::<D>() == TypeId::of::<f32>() {
                    // Safety: D is f32 at runtime, so transmute is a no-op.
                    let graph = unsafe {
                        &*(graph as *const TensorGraph<D, WithGrad>
                            as *const TensorGraph<f32, WithGrad>)
                    };
                    let result = executor.get_gradients(graph);
                    unsafe {
                        std::mem::transmute::<HashMap<usize, Vec<f32>>, HashMap<usize, Vec<D>>>(
                            result,
                        )
                    }
                } else if TypeId::of::<D>() == TypeId::of::<half::f16>() {
                    graph
                        .gradient_metadata()
                        .param_to_grad
                        .iter()
                        .filter_map(|(param_id, node_idx)| {
                            executor.get_value(*node_idx).map(|values| {
                                let values: Vec<half::f16> =
                                    values.into_iter().map(half::f16::from_f32).collect();
                                // Safety: the TypeId check proves D is f16.
                                let values = unsafe {
                                    std::mem::transmute::<Vec<half::f16>, Vec<D>>(values)
                                };
                                (*param_id, values)
                            })
                        })
                        .collect()
                } else if TypeId::of::<D>() == TypeId::of::<half::bf16>() {
                    graph
                        .gradient_metadata()
                        .param_to_grad
                        .iter()
                        .filter_map(|(param_id, node_idx)| {
                            executor.get_value(*node_idx).map(|values| {
                                let values: Vec<half::bf16> =
                                    values.into_iter().map(half::bf16::from_f32).collect();
                                // Safety: the TypeId check proves D is bf16.
                                let values = unsafe {
                                    std::mem::transmute::<Vec<half::bf16>, Vec<D>>(values)
                                };
                                (*param_id, values)
                            })
                        })
                        .collect()
                } else {
                    HashMap::new()
                }
            }
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
    let output = executor.compile_and_execute(graph, inputs)?;
    // SAFETY: The checked scalar types are identical; transfer the allocation.
    Ok(unsafe { std::mem::transmute::<Vec<M::HostType>, Vec<D>>(output) })
}
