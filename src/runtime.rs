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
/// Note: GPU backends (CUDA/PTX) currently only support f32.
pub enum Runtime<D = f32> {
    /// CPU-based executor
    Cpu(SimpleExecutor<D>),
    /// GPU-based executor (only available with `cuda` feature, f32 only)
    #[cfg(feature = "cuda")]
    Cuda(CudaExecutor),
    /// PTX JIT executor (only available with `cuda` feature, f32 only)
    #[cfg(feature = "cuda")]
    Ptx(PtxExecutor),
}

impl Runtime<f32> {
    /// Create a runtime with automatic backend selection.
    ///
    /// When the `ptx` feature is enabled, prefers PTX JIT executor.
    /// Otherwise, prefers static CUDA executor if available.
    /// Falls back to CPU if CUDA initialization fails.
    pub fn new() -> Self {
        #[cfg(all(feature = "cuda", feature = "ptx"))]
        {
            // PTX feature enabled: try PTX first, then CUDA, then CPU
            match PtxExecutor::try_new() {
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
            // CUDA feature enabled but not PTX: try CUDA, then CPU
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
                    PtxExecutor::try_new().context("Failed to initialize PTX executor")?;
                Ok(Runtime::Ptx(executor))
            }
        }
    }
}

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
            matches!(self, Runtime::Cuda(_) | Runtime::Ptx(_))
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }

    /// Get the computed value for a specific node after execution.
    ///
    /// Note: For GPU backends, this always returns f32 values.
    /// For CPU backend, returns values of type D.
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
                } else {
                    None
                }
            }
            #[cfg(feature = "cuda")]
            Runtime::Ptx(_executor) => {
                use std::any::TypeId;
                if TypeId::of::<D>() == TypeId::of::<f32>() {
                    unimplemented!("PTX get_value not yet implemented")
                } else {
                    None
                }
            }
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
                if TypeId::of::<D>() == TypeId::of::<f32>() {
                    // Safety: D is f32 at runtime, so transmute is a no-op.
                    let graph = unsafe {
                        &*(graph as *const TensorGraph<D, G> as *const TensorGraph<f32, G>)
                    };
                    let inputs = unsafe {
                        std::mem::transmute::<HashMap<String, Vec<D>>, HashMap<String, Vec<f32>>>(
                            inputs,
                        )
                    };
                    let result = executor.execute(graph, inputs);
                    unsafe { std::mem::transmute::<Result<Vec<f32>>, Result<Vec<D>>>(result) }
                } else {
                    anyhow::bail!("CUDA backend only supports f32")
                }
            }
            #[cfg(feature = "cuda")]
            Runtime::Ptx(_executor) => {
                anyhow::bail!("PTX backend not yet supported through Runtime")
            }
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
                } else {
                    HashMap::new()
                }
            }
            #[cfg(feature = "cuda")]
            Runtime::Ptx(_executor) => {
                unimplemented!("PTX gradients not yet implemented")
            }
        }
    }
}
