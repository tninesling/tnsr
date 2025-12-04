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
pub enum Runtime {
    /// CPU-based executor
    Cpu(SimpleExecutor),
    /// GPU-based executor (only available with `cuda` feature)
    #[cfg(feature = "cuda")]
    Cuda(CudaExecutor),
    /// PTX JIT executor (only available with `cuda` feature)
    #[cfg(feature = "cuda")]
    Ptx(PtxExecutor),
}

impl Runtime {
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
    pub fn get_value(&self, node_idx: NodeIndex) -> Option<Vec<f32>> {
        match self {
            Runtime::Cpu(executor) => executor.get_value(node_idx).cloned(),
            #[cfg(feature = "cuda")]
            Runtime::Cuda(executor) => executor.get_value(node_idx),
            #[cfg(feature = "cuda")]
            Runtime::Ptx(executor) => executor.get_value(node_idx),
        }
    }
}

impl Default for Runtime {
    fn default() -> Self {
        Self::new()
    }
}

impl Executor<f32> for Runtime {
    fn execute<G>(
        &mut self,
        graph: &TensorGraph<f32, G>,
        inputs: HashMap<String, Vec<f32>>,
    ) -> Result<Vec<f32>>
    where
        TensorGraph<f32, G>: Clone,
    {
        match self {
            Runtime::Cpu(executor) => executor.execute(graph, inputs),
            #[cfg(feature = "cuda")]
            Runtime::Cuda(executor) => executor.execute(graph, inputs),
            #[cfg(feature = "cuda")]
            Runtime::Ptx(executor) => executor.execute(graph, inputs),
        }
    }

    fn get_gradients(&self, graph: &TensorGraph<f32, WithGrad>) -> HashMap<usize, Vec<f32>> {
        match self {
            Runtime::Cpu(executor) => executor.get_gradients(graph),
            #[cfg(feature = "cuda")]
            Runtime::Cuda(executor) => executor.get_gradients(graph),
            #[cfg(feature = "cuda")]
            Runtime::Ptx(executor) => executor.get_gradients(graph),
        }
    }
}
