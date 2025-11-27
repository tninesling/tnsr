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
/// The [`Runtime`] enum wraps either a CPU or GPU executor and provides
/// a single consistent interface. Users can create a runtime with automatic
/// backend detection using [`Runtime::new`], or explicitly choose a backend
/// with [`Runtime::with_backend`].
///
/// # Examples
///
/// ```rust
/// use runtime::{Executor, Runtime};
/// use tensor::{TensorExpr, graph::TensorGraph};
/// use std::collections::HashMap;
///
/// // Automatic backend selection (prefers CUDA if available)
/// let mut runtime = Runtime::new();
///
/// let x = TensorExpr::<f32>::input("x", vec![2, 2]);
/// let graph: TensorGraph<f32> = x.into();
///
/// let mut inputs = HashMap::new();
/// inputs.insert("x".to_string(), vec![1.0, 2.0, 3.0, 4.0]);
///
/// let result = runtime.execute(&graph, inputs).unwrap();
/// assert_eq!(result.len(), 4);
/// ```
///
/// # Explicit Backend Selection
///
/// ```rust
/// use runtime::{Runtime, Backend};
///
/// // Force CPU execution
/// let mut cpu_runtime = Runtime::with_backend(Backend::Cpu).unwrap();
///
/// // Try CUDA, will error if not available
/// # #[cfg(feature = "cuda")]
/// let mut gpu_runtime = Runtime::with_backend(Backend::Cuda);
/// ```
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
    /// Prefers CUDA if the `cuda` feature is enabled and CUDA initialization succeeds,
    /// otherwise falls back to CPU execution.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use runtime::Runtime;
    ///
    /// let runtime = Runtime::new();
    /// println!("Using backend: {:?}", runtime.backend());
    /// ```
    pub fn new() -> Self {
        #[cfg(feature = "cuda")]
        {
            // Try CUDA first, fall back to CPU if initialization fails
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
    ///
    /// # Arguments
    ///
    /// * `backend` - The execution backend to use
    ///
    /// # Returns
    ///
    /// Returns `Ok(Runtime)` if the backend is available and initialization succeeds,
    /// or an error if the backend is unavailable or initialization fails.
    ///
    /// # Errors
    ///
    /// - Returns an error if the CUDA backend is requested but the `cuda` feature is not enabled
    /// - Returns an error if CUDA initialization fails (device unavailable, driver issues, etc.)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use runtime::{Runtime, Backend};
    ///
    /// // CPU is always available
    /// let cpu_runtime = Runtime::with_backend(Backend::Cpu).unwrap();
    ///
    /// // CUDA may not be available
    /// # #[cfg(feature = "cuda")]
    /// match Runtime::with_backend(Backend::Cuda) {
    ///     Ok(runtime) => println!("CUDA runtime created"),
    ///     Err(e) => println!("CUDA not available: {}", e),
    /// }
    /// ```
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
    ///
    /// # Examples
    ///
    /// ```rust
    /// use runtime::{Runtime, Backend};
    ///
    /// let runtime = Runtime::new();
    /// match runtime.backend() {
    ///     Backend::Cpu => println!("Using CPU"),
    ///     # #[cfg(feature = "cuda")]
    ///     Backend::Cuda => println!("Using CUDA"),
    ///     # #[cfg(feature = "cuda")]
    ///     Backend::Ptx => println!("Using PTX"),
    /// }
    /// ```
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

    /// Get the computed value for a specific node in the graph.
    ///
    /// This method allows retrieving intermediate values after execution.
    ///
    /// # Arguments
    ///
    /// * `node_idx` - The index of the node whose value to retrieve
    ///
    /// # Returns
    ///
    /// The computed tensor value as a flat vector, or `None` if the node hasn't been computed.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use runtime::{Executor, Runtime};
    /// use tensor::{TensorExpr, graph::TensorGraph};
    /// use std::collections::HashMap;
    ///
    /// let mut runtime = Runtime::new();
    /// let x = TensorExpr::<f32>::input("x", vec![2, 2]);
    /// let graph: TensorGraph<f32> = x.into();
    /// let node_idx = *graph.toposort().last().unwrap();
    ///
    /// let mut inputs = HashMap::new();
    /// inputs.insert("x".to_string(), vec![1.0, 2.0, 3.0, 4.0]);
    ///
    /// runtime.execute(&graph, inputs).unwrap();
    /// let value = runtime.get_value(node_idx).unwrap();
    /// assert_eq!(value.len(), 4);
    /// ```
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
