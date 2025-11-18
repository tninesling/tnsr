//! Runtime execution engine for tensor computation graphs.
//!
//! This crate provides executors that perform forward and backward passes on computation graphs
//! lowered from the `tensor` crate. Execution is where errors can occur—resource allocation,
//! shape validation, and arithmetic operations may fail.
//!
//! # Architecture
//!
//! - **Executor Trait**: Common interface for forward and backward passes, returning
//!   `Result<Vec<D>>` and `Result<BackwardResult<D>>` respectively for proper error handling.
//!
//! - **SimpleExecutor**: CPU-based executor with optional parallelism (via `parallel` feature).
//!   Uses naive algorithms suitable for testing and small models.
//!
//! - **CudaExecutor** (optional): GPU-accelerated executor using CUDA kernels (via `cuda` feature).
//!   Automatically manages device memory and kernel launches.
//!
//! # Error Handling
//!
//! Execution failures return `Result` types with `anyhow::Error` for:
//! - Missing inputs or forward values
//! - Shape mismatches
//! - Memory allocation failures
//! - Invalid operations (e.g., unsupported dimensions)
//! - CUDA-specific errors (kernel launch failures, device errors)
//!
//! Use `.unwrap()` in tests or the `?` operator in production code to handle errors.
//!
//! # Features
//!
//! - `cuda`: Enable GPU acceleration via CUDA (requires CUDA toolkit)
//! - `parallel`: Enable CPU parallelism via Rayon
//!
//! # Example
//!
//! ```rust
//! use runtime::{Executor, SimpleExecutor};
//! use tensor::{TensorExpr, graph::TensorGraph};
//! use std::collections::HashMap;
//!
//! let mut executor = SimpleExecutor::new();
//! let x = TensorExpr::<f32>::input("x", vec![2, 3]);
//! let graph: TensorGraph<f32> = x.into();
//!
//! let mut inputs = HashMap::new();
//! inputs.insert("x".to_string(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
//!
//! let result = executor.forward(&graph, inputs).unwrap();
//! assert_eq!(result.len(), 6);
//! ```

use std::collections::HashMap;

use anyhow::{Context, Result};

#[cfg(feature = "cuda")]
pub mod cuda;
pub mod optimizer;

#[cfg(feature = "parallel")]
use rayon::prelude::*;
use tensor::graph::TensorGraph;
use tensor::graph::TensorGraphNode;
use tracing::trace_span;
use tracing_chrome::ChromeLayerBuilder;
use tracing_subscriber::prelude::*;

/// Iterator type for slice traversal, conditionally parallel based on features.
///
/// With the `parallel` feature enabled, uses Rayon for parallel iteration.
/// Otherwise, uses standard sequential iteration.
#[cfg(feature = "parallel")]
pub type SliceIter<'a, T> = rayon::iter::MinLen<rayon::slice::Iter<'a, T>>;

/// Iterator type for slice traversal, conditionally parallel based on features.
///
/// With the `parallel` feature enabled, uses Rayon for parallel iteration.
/// Otherwise, uses standard sequential iteration.
#[cfg(not(feature = "parallel"))]
pub type SliceIter<'a, T> = std::slice::Iter<'a, T>;

#[cfg(feature = "parallel")]
mod parallel_config {
    /// Minimum number of elements for parallel iteration
    pub const ELEMENTWISE_THRESHOLD: usize = 8192;

    /// Minimum matrix dimension for parallel matmul
    pub const MATMUL_THRESHOLD: usize = 128;
}

/// RAII guard for Chrome tracing session.
///
/// Ensures trace data is properly flushed to disk when dropped.
/// Created by [`init_chrome_tracing`].
pub struct TracingGuard {
    _guard: tracing_chrome::FlushGuard,
}

/// Initialize Chrome tracing and write output to the specified file.
///
/// Returns a [`TracingGuard`] that must be kept alive for the duration of tracing.
/// When dropped, all trace data is flushed to the file.
///
/// # Example
///
/// ```no_run
/// use runtime::init_chrome_tracing;
///
/// let _guard = init_chrome_tracing("trace.json").unwrap();
/// // ... perform traced operations ...
/// // Trace is flushed when _guard is dropped
/// ```
pub fn init_chrome_tracing(file_path: &str) -> Result<TracingGuard, Box<dyn std::error::Error>> {
    let (chrome_layer, guard) = ChromeLayerBuilder::new().file(file_path).build();

    tracing_subscriber::registry().with(chrome_layer).init();

    Ok(TracingGuard { _guard: guard })
}

/// Result of a backward pass through a computation graph.
///
/// Contains gradients computed via reverse-mode automatic differentiation,
/// organized by both graph nodes and parameter IDs for efficient access.
pub struct BackwardResult<D> {
    /// Gradients indexed by graph node position.
    ///
    /// Useful for inspecting intermediate gradients during debugging.
    pub grads_by_node: HashMap<petgraph::graph::NodeIndex, Vec<D>>,

    /// Gradients indexed by parameter ID.
    ///
    /// Used by optimizers to update trainable parameters. Each parameter's
    /// gradient is accumulated across all uses in the graph.
    pub grads_by_param: HashMap<usize, Vec<D>>,

    /// The final scalar loss value computed during the forward pass.
    ///
    /// Typically a single element `Vec<D>` representing the loss to minimize.
    pub loss_value: Vec<D>,
}

/// Executes forward and backward passes on computation graphs.
///
/// Implementors provide different execution strategies (CPU, GPU, etc.)
/// while maintaining a common interface for graph evaluation.
pub trait Executor<D> {
    /// Execute a forward pass through the computation graph.
    ///
    /// # Arguments
    ///
    /// * `graph` - The computation graph to execute
    /// * `inputs` - Named input tensors as flat vectors
    ///
    /// # Returns
    ///
    /// The output tensor as a flat vector, or an error if execution fails.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Required inputs are missing
    /// - Shapes are incompatible
    /// - Memory allocation fails
    /// - Invalid operations are encountered
    fn forward(
        &mut self,
        graph: &TensorGraph<D>,
        inputs: HashMap<String, Vec<D>>,
    ) -> Result<Vec<D>>;

    /// Execute a backward pass to compute gradients via automatic differentiation.
    ///
    /// Performs reverse-mode autodiff starting from `loss_node` and propagating
    /// gradients back through the graph to all parameters.
    ///
    /// # Arguments
    ///
    /// * `graph` - The computation graph (must match the last forward pass)
    /// * `loss_node` - Graph node representing the scalar loss to differentiate
    /// * `seed_grad` - Optional initial gradient (defaults to `vec![1.0]`)
    ///
    /// # Returns
    ///
    /// A [`BackwardResult`] containing gradients for all nodes and parameters,
    /// or an error if backward pass fails.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Forward pass was not called first
    /// - Graph structure is invalid
    /// - Memory allocation fails
    fn backward(
        &mut self,
        graph: &TensorGraph<D>,
        loss_node: petgraph::graph::NodeIndex,
        seed_grad: Option<Vec<D>>,
    ) -> Result<BackwardResult<D>>;
}

/// CPU-based executor for tensor computation graphs.
///
/// Implements the [`Executor`] trait using CPU operations. When the `parallel`
/// feature is enabled, uses Rayon for parallelism on larger tensors.
///
/// Suitable for testing, small models, and platforms without GPU support.
///
/// # Example
///
/// ```rust
/// use runtime::{Executor, SimpleExecutor};
/// use tensor::{TensorExpr, graph::TensorGraph};
/// use std::collections::HashMap;
///
/// let mut executor = SimpleExecutor::new();
/// let x = TensorExpr::<f32>::input("x", vec![2, 2]);
/// let graph: TensorGraph<f32> = x.into();
///
/// let mut inputs = HashMap::new();
/// inputs.insert("x".to_string(), vec![1.0, 2.0, 3.0, 4.0]);
///
/// let result = executor.forward(&graph, inputs).unwrap();
/// assert_eq!(result.len(), 4);
/// ```
#[derive(Default)]
pub struct SimpleExecutor {
    values: HashMap<petgraph::graph::NodeIndex, Vec<f32>>,
    grads: HashMap<petgraph::graph::NodeIndex, Vec<f32>>,
}

impl SimpleExecutor {
    /// Create a new CPU executor.
    pub fn new() -> Self {
        Self::default()
    }

    /// Retrieve the computed value for a specific graph node.
    ///
    /// Only available after a forward pass has been executed.
    /// Useful for debugging intermediate values.
    pub fn get_value(&self, node_idx: petgraph::graph::NodeIndex) -> Option<&Vec<f32>> {
        self.values.get(&node_idx)
    }

    fn neg(&self, x: &[f32]) -> Vec<f32> {
        let _span = trace_span!("neg").entered();
        get_iter(x).map(|v| -v).collect()
    }

    fn exp(&self, x: &[f32]) -> Vec<f32> {
        let _span = trace_span!("exp").entered();

        get_iter(x).copied().map(f32::exp).collect()
    }

    fn log(&self, x: &[f32]) -> Vec<f32> {
        let _span = trace_span!("log").entered();

        get_iter(x).copied().map(f32::ln).collect()
    }

    fn relu(&self, x: &[f32]) -> Vec<f32> {
        let _span = trace_span!("relu").entered();

        get_iter(x).map(|v| v.max(0.0)).collect()
    }

    fn add(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        let _span = trace_span!("add").entered();

        get_iter(a).zip(get_iter(b)).map(|(x, y)| x + y).collect()
    }

    fn sub(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        let _span = trace_span!("sub").entered();

        get_iter(a).zip(get_iter(b)).map(|(x, y)| x - y).collect()
    }

    fn mul(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        let _span = trace_span!("mul").entered();
        get_iter(a).zip(get_iter(b)).map(|(x, y)| x * y).collect()
    }

    fn div(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        let _span = trace_span!("div").entered();
        get_iter(a).zip(get_iter(b)).map(|(x, y)| x / y).collect()
    }

    fn neg_grad(&self, dy: &[f32]) -> Vec<f32> {
        let _span = trace_span!("neg").entered();
        get_iter(dy).map(|g| -g).collect()
    }

    fn exp_grad(&self, dy: &[f32], y: &[f32]) -> Vec<f32> {
        let _span = trace_span!("exp").entered();
        get_iter(dy).zip(get_iter(y)).map(|(g, y)| g * y).collect()
    }

    fn log_grad(&self, dy: &[f32], x: &[f32]) -> Vec<f32> {
        let _span = trace_span!("log").entered();
        get_iter(dy).zip(get_iter(x)).map(|(g, x)| g / x).collect()
    }

    fn relu_grad(&self, dy: &[f32], x: &[f32]) -> Vec<f32> {
        let _span = trace_span!("relu").entered();
        get_iter(dy)
            .zip(get_iter(x))
            .map(|(g, x)| if *x > 0.0 { *g } else { 0.0 })
            .collect()
    }

    fn add_grad(&self, dy: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let _span = trace_span!("add").entered();
        (dy.to_vec(), dy.to_vec())
    }

    fn sub_grad(&self, dy: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let _span = trace_span!("sub").entered();
        let da = dy.to_vec();
        let db = dy.iter().map(|v| -v).collect();
        (da, db)
    }

    fn mul_grad(&self, dy: &[f32], a_val: &[f32], b_val: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let _span = trace_span!("mul").entered();
        let da: Vec<f32> = get_iter(dy)
            .zip(get_iter(b_val))
            .map(|(g, b)| g * b)
            .collect();
        let db: Vec<f32> = get_iter(dy)
            .zip(get_iter(a_val))
            .map(|(g, a)| g * a)
            .collect();
        (da, db)
    }

    fn div_grad(&self, dy: &[f32], a_val: &[f32], b_val: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let _span = trace_span!("div").entered();
        let da: Vec<f32> = get_iter(dy)
            .zip(get_iter(b_val))
            .map(|(g, b)| g / b)
            .collect();
        let tmp_b: Vec<f32> = get_iter(dy)
            .zip(get_iter(a_val))
            .map(|(g, a)| -g * a)
            .collect();
        let mut b_sq = b_val.to_vec();
        for v in b_sq.iter_mut() {
            *v = *v * *v;
        }
        let db: Vec<f32> = get_iter(&tmp_b)
            .zip(get_iter(&b_sq))
            .map(|(t, bsq)| t / bsq)
            .collect();
        (da, db)
    }
}

impl Executor<f32> for SimpleExecutor {
    fn forward(
        &mut self,
        graph: &TensorGraph<f32>,
        inputs: HashMap<String, Vec<f32>>,
    ) -> Result<Vec<f32>> {
        let order = graph.toposort();
        let _fwd_span = trace_span!("forward", nodes = order.len()).entered();

        for node_idx in order.iter() {
            let node = &graph[*node_idx];
            let result = match node {
                TensorGraphNode::Constant { data } => {
                    let _span = trace_span!("constant", node = node_idx.index()).entered();
                    data.as_ref().clone()
                }
                TensorGraphNode::Input { name } => {
                    let _span =
                        trace_span!("input", node = node_idx.index(), name = name).entered();
                    inputs
                        .get::<str>(name)
                        .with_context(|| format!("Input '{}' not found", name))?
                        .clone()
                }
                TensorGraphNode::Parameter { data, .. } => {
                    let _span = trace_span!("parameter", node = node_idx.index()).entered();
                    data.lock().unwrap().clone()
                }
                TensorGraphNode::Unary { op } => {
                    let inputs = graph.inputs(*node_idx);
                    let x = self.values.get(&inputs[0]).with_context(|| {
                        format!("Value for node {} not computed", inputs[0].index())
                    })?;
                    match op {
                        tensor::UnaryOp::Neg => self.neg(x),
                        tensor::UnaryOp::Exp => self.exp(x),
                        tensor::UnaryOp::Log => self.log(x),
                        tensor::UnaryOp::Relu => self.relu(x),
                    }
                }
                TensorGraphNode::Binary { op } => {
                    let ins = graph.inputs(*node_idx);
                    let a = self.values.get(&ins[0]).with_context(|| {
                        format!("Value for node {} not computed", ins[0].index())
                    })?;
                    let b = self.values.get(&ins[1]).with_context(|| {
                        format!("Value for node {} not computed", ins[1].index())
                    })?;
                    anyhow::ensure!(
                        a.len() == b.len(),
                        "Binary op shape mismatch: {} vs {}",
                        a.len(),
                        b.len()
                    );
                    match op {
                        tensor::BinaryOp::Add => self.add(a, b),
                        tensor::BinaryOp::Sub => self.sub(a, b),
                        tensor::BinaryOp::Mul => self.mul(a, b),
                        tensor::BinaryOp::Div => self.div(a, b),
                    }
                }
                TensorGraphNode::MatMul => {
                    let _span = trace_span!("matmul", node = node_idx.index()).entered();
                    matmul_forward(graph, &self.values, *node_idx)?
                }
                TensorGraphNode::Transpose => {
                    let _span = trace_span!("transpose", node = node_idx.index()).entered();
                    transpose_forward(graph, &self.values, *node_idx)?
                }
                TensorGraphNode::BroadcastAxis { axis } => {
                    let _span = trace_span!("broadcast_axis", node = node_idx.index()).entered();
                    broadcast_axis_forward(graph, &self.values, *node_idx, *axis)?
                }
                TensorGraphNode::ReduceAxis { op, axis } => {
                    let _span = trace_span!(
                        "reduce_axis",
                        op = node.name(),
                        axis = *axis,
                        node = node_idx.index()
                    )
                    .entered();
                    reduce_axis_forward(graph, &self.values, *node_idx, op, *axis)?
                }
            };
            self.values.insert(*node_idx, result);
        }
        let last_node = order
            .last()
            .context("Graph is empty, no nodes to execute")?;
        let out = self
            .values
            .get(last_node)
            .context("Output value not computed")?;
        Ok(out.clone())
    }

    fn backward(
        &mut self,
        graph: &TensorGraph<f32>,
        loss_node: petgraph::graph::NodeIndex,
        seed_grad: Option<Vec<f32>>,
    ) -> Result<BackwardResult<f32>> {
        let order = graph.toposort();
        let _bwd_span = trace_span!("backward", nodes = order.len()).entered();

        let mut param_grads: HashMap<usize, Vec<f32>> = HashMap::new();

        // Initialize gradient for loss node
        let loss_shape = graph
            .shapes
            .get(&loss_node)
            .context("Loss node shape missing")?;

        let seed = seed_grad.unwrap_or_else(|| vec![1.0f32; get_iter(loss_shape).product()]);

        self.grads.insert(loss_node, seed);

        // Backward pass in reverse topological order
        for &node_idx in order.iter().rev() {
            if let Some(dy) = self.grads.get(&node_idx).cloned() {
                let node = &graph[node_idx];
                match node {
                    TensorGraphNode::Constant { .. } => {}
                    TensorGraphNode::Input { .. } => {}
                    TensorGraphNode::Parameter { id, .. } => {
                        param_grads
                            .entry(*id)
                            .and_modify(|g| add_inplace(g, &dy))
                            .or_insert(dy);
                    }
                    TensorGraphNode::Unary { op } => {
                        let x_idx = graph.inputs(node_idx)[0];
                        let dx = match op {
                            tensor::UnaryOp::Neg => self.neg_grad(&dy),
                            tensor::UnaryOp::Exp => {
                                let y = self.values.get(&node_idx).with_context(|| {
                                    format!("Forward value for node {} not found", node_idx.index())
                                })?;
                                self.exp_grad(&dy, y)
                            }
                            tensor::UnaryOp::Log => {
                                let x = self.values.get(&x_idx).with_context(|| {
                                    format!("Forward value for node {} not found", x_idx.index())
                                })?;
                                self.log_grad(&dy, x)
                            }
                            tensor::UnaryOp::Relu => {
                                let x = self.values.get(&x_idx).with_context(|| {
                                    format!("Forward value for node {} not found", x_idx.index())
                                })?;
                                self.relu_grad(&dy, x)
                            }
                        };
                        accumulate_grad(&mut self.grads, x_idx, dx);
                    }
                    TensorGraphNode::Binary { op } => {
                        let ins = graph.inputs(node_idx);
                        let a_idx = ins[0];
                        let b_idx = ins[1];
                        let a_val = self.values.get(&a_idx).with_context(|| {
                            format!("Forward value for node {} not found", a_idx.index())
                        })?;
                        let b_val = self.values.get(&b_idx).with_context(|| {
                            format!("Forward value for node {} not found", b_idx.index())
                        })?;

                        let (da, db) = match op {
                            tensor::BinaryOp::Add => self.add_grad(&dy),
                            tensor::BinaryOp::Sub => self.sub_grad(&dy),
                            tensor::BinaryOp::Mul => self.mul_grad(&dy, a_val, b_val),
                            tensor::BinaryOp::Div => self.div_grad(&dy, a_val, b_val),
                        };
                        accumulate_grad(&mut self.grads, a_idx, da);
                        accumulate_grad(&mut self.grads, b_idx, db);
                    }
                    TensorGraphNode::MatMul => {
                        let _span = trace_span!("matmul", node = node_idx.index()).entered();
                        let ins = graph.inputs(node_idx);
                        let a_idx = ins[0];
                        let b_idx = ins[1];
                        let a_val = self.values.get(&a_idx).with_context(|| {
                            format!("Forward value for node {} not found", a_idx.index())
                        })?;
                        let b_val = self.values.get(&b_idx).with_context(|| {
                            format!("Forward value for node {} not found", b_idx.index())
                        })?;
                        let a_shape = graph
                            .shapes
                            .get(&a_idx)
                            .context("Shape missing for matmul left operand")?;
                        let b_shape = graph
                            .shapes
                            .get(&b_idx)
                            .context("Shape missing for matmul right operand")?;
                        let dy_shape = graph
                            .shapes
                            .get(&node_idx)
                            .context("Shape missing for matmul output")?;

                        // dA = dY * B^T
                        let da = matmul_grad_left(&dy, dy_shape, b_val, b_shape);
                        // dB = A^T * dY
                        let db = matmul_grad_right(a_val, a_shape, &dy, dy_shape);
                        accumulate_grad(&mut self.grads, a_idx, da);
                        accumulate_grad(&mut self.grads, b_idx, db);
                    }
                    TensorGraphNode::Transpose => {
                        let _span = trace_span!("transpose", node = node_idx.index()).entered();
                        let x_idx = graph.inputs(node_idx)[0];
                        let dy_shape = graph
                            .shapes
                            .get(&node_idx)
                            .context("Shape missing for transpose output")?;
                        
                        // Gradient of transpose is transpose of gradient
                        // If Y = X^T, then dX = (dY)^T
                        let dx = transpose_2d(&dy, dy_shape);
                        accumulate_grad(&mut self.grads, x_idx, dx);
                    }
                    TensorGraphNode::BroadcastAxis { axis } => {
                        let _span =
                            trace_span!("broadcast_axis", node = node_idx.index()).entered();
                        let x_idx = graph.inputs(node_idx)[0];
                        let y_shape = graph
                            .shapes
                            .get(&node_idx)
                            .context("Shape missing for broadcast_axis output")?;
                        let dx = broadcast_axis_backward(&dy, y_shape, *axis);
                        accumulate_grad(&mut self.grads, x_idx, dx);
                    }
                    TensorGraphNode::ReduceAxis { op, axis } => {
                        let _span = trace_span!(
                            "reduce_axis",
                            op = node.name(),
                            axis = *axis,
                            node = node_idx.index()
                        )
                        .entered();
                        let x_idx = graph.inputs(node_idx)[0];
                        let x_shape = graph
                            .shapes
                            .get(&x_idx)
                            .context("Shape missing for reduce_axis input")?;
                        let y_shape = graph
                            .shapes
                            .get(&node_idx)
                            .context("Shape missing for reduce_axis output")?;
                        match op {
                            tensor::ReduceOp::Sum => {
                                let dx = reduce_axis_backward(&dy, y_shape, *axis, x_shape[*axis]);
                                accumulate_grad(&mut self.grads, x_idx, dx);
                            }
                            tensor::ReduceOp::Mean => {
                                let mut dx =
                                    reduce_axis_backward(&dy, y_shape, *axis, x_shape[*axis]);
                                let axis_size = x_shape[*axis];
                                for v in dx.iter_mut() {
                                    *v /= axis_size as f32;
                                }
                                accumulate_grad(&mut self.grads, x_idx, dx);
                            }
                            tensor::ReduceOp::Max => {
                                accumulate_grad(
                                    &mut self.grads,
                                    x_idx,
                                    vec![0.0f32; get_iter(x_shape).product()],
                                );
                            }
                        }
                    }
                }
            }
        }

        let loss_value = self
            .values
            .get(&loss_node)
            .context("Loss value not computed")?
            .clone();

        Ok(BackwardResult {
            grads_by_node: self.grads.drain().collect(),
            grads_by_param: param_grads,
            loss_value,
        })
    }
}

fn rowmajor_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }
    let mut s = vec![0usize; shape.len()];
    s[shape.len() - 1] = 1;
    for i in (0..shape.len() - 1).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

fn matmul_forward(
    graph: &TensorGraph<f32>,
    values: &HashMap<petgraph::graph::NodeIndex, Vec<f32>>,
    node_idx: petgraph::graph::NodeIndex,
) -> Result<Vec<f32>> {
    let inputs_idx = graph.inputs(node_idx);
    let a_idx = inputs_idx[0];
    let b_idx = inputs_idx[1];
    let a = values
        .get(&a_idx)
        .context("Left operand value not computed for matmul")?;
    let b = values
        .get(&b_idx)
        .context("Right operand value not computed for matmul")?;
    let a_shape = graph
        .shapes
        .get(&a_idx)
        .context("Shape missing for matmul left operand")?;
    let b_shape = graph
        .shapes
        .get(&b_idx)
        .context("Shape missing for matmul right operand")?;
    anyhow::ensure!(
        a_shape.len() == 2,
        "Matmul left operand must be 2D, got {}D",
        a_shape.len()
    );
    anyhow::ensure!(
        b_shape.len() == 2,
        "Matmul right operand must be 2D, got {}D",
        b_shape.len()
    );
    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[1];

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;

        if m >= parallel_config::MATMUL_THRESHOLD {
            let mut out = vec![0.0f32; m * n];
            out.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
                for j in 0..n {
                    let mut sum = 0.0f32;
                    for p in 0..k {
                        sum += a[i * k + p] * b[p * n + j];
                    }
                    row[j] = sum;
                }
            });
            return Ok(out);
        }
    }

    // Sequential fallback
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            out[i * n + j] = sum;
        }
    }
    Ok(out)
}

fn transpose_forward(
    graph: &TensorGraph<f32>,
    values: &HashMap<petgraph::graph::NodeIndex, Vec<f32>>,
    node_idx: petgraph::graph::NodeIndex,
) -> Result<Vec<f32>> {
    let inputs_idx = graph.inputs(node_idx);
    let a_idx = inputs_idx[0];
    let a = values
        .get(&a_idx)
        .context("Input value not computed for transpose")?;
    let a_shape = graph
        .shapes
        .get(&a_idx)
        .context("Shape missing for transpose input")?;
    
    anyhow::ensure!(
        a_shape.len() == 2,
        "Transpose currently only supports 2D matrices, got {}D",
        a_shape.len()
    );
    
    let (m, n) = (a_shape[0], a_shape[1]);
    let mut out = vec![0.0f32; m * n];
    
    for i in 0..m {
        for j in 0..n {
            out[j * m + i] = a[i * n + j];
        }
    }
    
    Ok(out)
}

fn add_inplace(acc: &mut [f32], src: &[f32]) {
    for (a, s) in acc.iter_mut().zip(src.iter()) {
        *a += *s;
    }
}

fn transpose_2d(a: &[f32], a_shape: &[usize]) -> Vec<f32> {
    assert_eq!(a_shape.len(), 2, "transpose_2d requires 2D shape");
    let (m, n) = (a_shape[0], a_shape[1]);
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            out[j * m + i] = a[i * n + j];
        }
    }
    out
}

fn accumulate_grad(
    map: &mut HashMap<petgraph::graph::NodeIndex, Vec<f32>>,
    idx: petgraph::graph::NodeIndex,
    add: Vec<f32>,
) {
    map.entry(idx)
        .and_modify(|g| add_inplace(g, &add))
        .or_insert(add);
}

fn matmul_grad_left(dy: &[f32], dy_shape: &[usize], b: &[f32], b_shape: &[usize]) -> Vec<f32> {
    let (m, n) = (dy_shape[0], dy_shape[1]);
    let (kb, nb) = (b_shape[0], b_shape[1]);
    assert_eq!(n, nb);
    let k = kb;

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;

        if m >= parallel_config::MATMUL_THRESHOLD {
            let mut da = vec![0.0f32; m * k];
            da.par_chunks_mut(k).enumerate().for_each(|(i, row)| {
                for p in 0..k {
                    let mut sum = 0.0f32;
                    for j in 0..n {
                        sum += dy[i * n + j] * b[p * n + j];
                    }
                    row[p] = sum;
                }
            });
            return da;
        }
    }

    // Sequential fallback
    let mut da = vec![0.0f32; m * k];
    for i in 0..m {
        for p in 0..k {
            let mut sum = 0.0f32;
            for j in 0..n {
                sum += dy[i * n + j] * b[p * n + j];
            }
            da[i * k + p] = sum;
        }
    }
    da
}

fn matmul_grad_right(a: &[f32], a_shape: &[usize], dy: &[f32], dy_shape: &[usize]) -> Vec<f32> {
    let (m, k) = (a_shape[0], a_shape[1]);
    let (mdy, n) = (dy_shape[0], dy_shape[1]);
    assert_eq!(m, mdy);

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;

        if k >= parallel_config::MATMUL_THRESHOLD {
            let mut db = vec![0.0f32; k * n];
            db.par_chunks_mut(n).enumerate().for_each(|(p, row)| {
                for j in 0..n {
                    let mut sum = 0.0f32;
                    for i in 0..m {
                        sum += a[i * k + p] * dy[i * n + j];
                    }
                    row[j] = sum;
                }
            });
            return db;
        }
    }

    // Sequential fallback
    let mut db = vec![0.0f32; k * n];
    for p in 0..k {
        for j in 0..n {
            let mut sum = 0.0f32;
            for i in 0..m {
                sum += a[i * k + p] * dy[i * n + j];
            }
            db[p * n + j] = sum;
        }
    }
    db
}

fn broadcast_axis_forward(
    graph: &TensorGraph<f32>,
    values: &HashMap<petgraph::graph::NodeIndex, Vec<f32>>,
    node_idx: petgraph::graph::NodeIndex,
    axis: usize,
) -> Result<Vec<f32>> {
    let in_idx = graph.inputs(node_idx)[0];
    let in_val = values
        .get(&in_idx)
        .context("Input value not computed for broadcast_axis")?;
    let in_shape = graph
        .shapes
        .get(&in_idx)
        .context("Input shape missing for broadcast_axis")?;
    let out_shape = graph
        .shapes
        .get(&node_idx)
        .context("Output shape missing for broadcast_axis")?;
    let out_size: usize = out_shape.iter().product();
    let in_strides = rowmajor_strides(in_shape);
    let out_strides = rowmajor_strides(out_shape);

    let mut out = vec![0.0f32; out_size];
    for (out_idx, out_elem) in out.iter_mut().enumerate() {
        let mut in_linear = 0usize;
        let mut rem = out_idx;
        for (dim, &stride) in out_strides.iter().enumerate() {
            let coord = rem / stride;
            rem %= stride;
            let in_coord = if dim == axis { 0 } else { coord };
            in_linear += in_coord * in_strides[dim];
        }
        *out_elem = in_val[in_linear];
    }
    Ok(out)
}

fn reduce_axis_forward(
    graph: &TensorGraph<f32>,
    values: &HashMap<petgraph::graph::NodeIndex, Vec<f32>>,
    node_idx: petgraph::graph::NodeIndex,
    op: &tensor::ReduceOp,
    axis: usize,
) -> Result<Vec<f32>> {
    let in_idx = graph.inputs(node_idx)[0];
    let x = values
        .get(&in_idx)
        .context("Input value not computed for reduce_axis")?;
    let in_shape = graph
        .shapes
        .get(&in_idx)
        .context("Input shape missing for reduce_axis")?;
    let out_shape = graph
        .shapes
        .get(&node_idx)
        .context("Output shape missing for reduce_axis")?;
    let out_size: usize = out_shape.iter().product();
    let axis_size = in_shape[axis];
    let in_strides = rowmajor_strides(in_shape);
    let out_strides = rowmajor_strides(out_shape);

    let mut out = match op {
        tensor::ReduceOp::Max => vec![f32::NEG_INFINITY; out_size],
        _ => vec![0.0f32; out_size],
    };

    let total: usize = in_shape.iter().product();
    for (idx, &val) in x.iter().enumerate().take(total) {
        let mut rem = idx;
        let mut out_linear = 0usize;
        for (dim, &stride) in in_strides.iter().enumerate() {
            let coord = rem / stride;
            rem %= stride;
            let out_coord = if dim == axis { 0 } else { coord };
            out_linear += out_coord * out_strides[dim];
        }
        match op {
            tensor::ReduceOp::Sum => {
                out[out_linear] += val;
            }
            tensor::ReduceOp::Mean => {
                out[out_linear] += val / axis_size as f32;
            }
            tensor::ReduceOp::Max => {
                out[out_linear] = out[out_linear].max(val);
            }
        }
    }
    Ok(out)
}

fn broadcast_axis_backward(dy: &[f32], dy_shape: &[usize], axis: usize) -> Vec<f32> {
    let mut dx_shape = dy_shape.to_vec();
    dx_shape[axis] = 1;
    let dx_size: usize = dx_shape.iter().product();
    let mut dx = vec![0.0f32; dx_size];

    let dy_strides = rowmajor_strides(dy_shape);
    let dx_strides = rowmajor_strides(&dx_shape);

    for (dy_idx, &grad_val) in dy.iter().enumerate() {
        let mut rem = dy_idx;
        let mut dx_linear = 0usize;
        for (dim, &stride) in dy_strides.iter().enumerate() {
            let coord = rem / stride;
            rem %= stride;
            let dx_coord = if dim == axis { 0 } else { coord };
            dx_linear += dx_coord * dx_strides[dim];
        }
        dx[dx_linear] += grad_val;
    }
    dx
}

fn reduce_axis_backward(
    dy: &[f32],
    dy_shape: &[usize],
    axis: usize,
    target_size: usize,
) -> Vec<f32> {
    let mut dx_shape = dy_shape.to_vec();
    dx_shape[axis] = target_size;
    let dx_size: usize = dx_shape.iter().product();
    let mut dx = vec![0.0f32; dx_size];

    let dx_strides = rowmajor_strides(&dx_shape);
    let dy_strides = rowmajor_strides(dy_shape);

    for (dx_idx, dx_elem) in dx.iter_mut().enumerate() {
        let mut rem = dx_idx;
        let mut dy_linear = 0usize;
        for (dim, &stride) in dx_strides.iter().enumerate() {
            let coord = rem / stride;
            rem %= stride;
            let dy_coord = if dim == axis { 0 } else { coord };
            dy_linear += dy_coord * dy_strides[dim];
        }
        *dx_elem = dy[dy_linear];
    }
    dx
}

#[cfg(feature = "cuda")]
pub(crate) fn expand_to(x: &[f32], x_shape: &[usize], target_shape: &[usize]) -> Vec<f32> {
    if x_shape == target_shape {
        return x.to_vec();
    }

    let in_shape = x_shape;
    let out_shape = target_shape;
    let out_size: usize = out_shape.iter().product();
    let mut out = vec![0.0f32; out_size];
    let in_rank = in_shape.len();
    let out_rank = out_shape.len();
    let out_strides = rowmajor_strides(out_shape);
    let in_strides = rowmajor_strides(in_shape);

    for (out_idx, out_elem) in out.iter_mut().enumerate().take(out_size) {
        let mut rem = out_idx;
        let mut in_linear = 0usize;
        for (dim, stride) in out_strides.iter().enumerate().take(out_rank) {
            let coord = if out_rank == 0 { 0 } else { rem / *stride };
            if out_rank > 0 {
                rem %= *stride;
            }
            let in_dim_opt = if dim + in_rank >= out_rank {
                Some(dim + in_rank - out_rank)
            } else {
                None
            };
            if let Some(in_dim) = in_dim_opt {
                let in_dim_size = in_shape[in_dim];
                let idx_in_dim = if in_dim_size == 1 { 0 } else { coord };
                let stride = if in_strides.is_empty() {
                    0
                } else {
                    in_strides[in_dim]
                };
                in_linear += idx_in_dim * stride;
            }
        }
        *out_elem = x[in_linear];
    }
    out
}

/// Get an iterator over a slice, parallel if the `parallel` feature is enabled.
///
/// Automatically uses parallel iteration for large slices when compiled with
/// the `parallel` feature, falling back to sequential iteration otherwise.
#[cfg(feature = "parallel")]
pub fn get_iter<T: Sync>(slice: &[T]) -> SliceIter<'_, T> {
    slice
        .par_iter()
        .with_min_len(parallel_config::ELEMENTWISE_THRESHOLD)
}

/// Get an iterator over a slice, parallel if the `parallel` feature is enabled.
///
/// Automatically uses parallel iteration for large slices when compiled with
/// the `parallel` feature, falling back to sequential iteration otherwise.
#[cfg(not(feature = "parallel"))]
pub fn get_iter<T>(slice: &[T]) -> SliceIter<'_, T> {
    slice.iter()
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-5;

    macro_rules! assert_approx_eq {
        ($a:expr, $b:expr) => {
            assert_approx_eq!($a, $b, EPSILON)
        };
        ($a:expr, $b:expr, $eps:expr) => {
            if $a
                .iter()
                .zip($b.iter())
                .any(|(a, b)| (*a - *b).abs() > $eps)
            {
                let diffs: Vec<_> = $a
                    .iter()
                    .zip($b.iter())
                    .enumerate()
                    .filter(|(_, (a, b))| (*a - *b).abs() > $eps)
                    .collect();
                panic!(
                    "assertion failed: `(left ~= right)` with epsilon {}\nDifferences at {} positions (showing first 5): {:?}\nleft: `{:?}`\nright: `{:?}`",
                    $eps,
                    diffs.len(),
                    &diffs[..diffs.len().min(5)],
                    $a,
                    $b
                );
            }
        };
    }

    fn test_simple_backward_pass_impl<E: Executor<f32>>(mut executor: E) {
        // Create a simple computation graph: x^2 where x is a parameter
        let x = tensor::Parameter::new(vec![2.0f32], vec![1]);
        let x_id = x.id();
        let x_sq = x.clone() * x;
        let graph: TensorGraph<f32> = x_sq.into();

        // Test backward pass - gradient of x^2 at x=2 should be 2*x = 4
        executor.forward(&graph, Default::default()).unwrap();
        let param_grads = executor
            .backward(&graph, 1.into(), Some(vec![1.0f32]))
            .unwrap();

        assert_eq!(param_grads.grads_by_param.len(), 1);
        let grad = param_grads
            .grads_by_param
            .get(&x_id)
            .expect("Parameter gradient missing");
        assert_eq!(grad.len(), 1);
        assert!(
            (grad[0] - 4.0).abs() < 1e-5,
            "Expected gradient 4.0, got {}",
            grad[0]
        );
    }

    #[test]
    fn test_simple_backward_pass_cpu() {
        test_simple_backward_pass_impl(SimpleExecutor::new());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_simple_backward_pass_cuda() {
        test_simple_backward_pass_impl(crate::cuda::CudaExecutor::new());
    }

    fn test_unary_backward_pass_impl<E: Executor<f32>>(mut executor: E) {
        // Test backward pass for exp(x) where x = [1.0]
        let x = tensor::Parameter::new(vec![1.0f32], vec![1]);
        let x_id = x.id();
        let exp = x.exp();
        let graph: TensorGraph<f32> = exp.into();

        // Gradient of exp(x) at x=1 should be exp(1) ≈ 2.718
        executor.forward(&graph, Default::default()).unwrap();
        let param_grads = executor
            .backward(&graph, 1.into(), Some(vec![1.0f32]))
            .unwrap();

        assert_eq!(param_grads.grads_by_param.len(), 1);
        let grad = param_grads
            .grads_by_param
            .get(&x_id)
            .expect("Parameter gradient missing");
        assert_eq!(grad.len(), 1);
        let expected = 1.0f32.exp();
        assert!(
            (grad[0] - expected).abs() < 1e-5,
            "Expected gradient {}, got {}",
            expected,
            grad[0]
        );
    }

    #[test]
    fn test_unary_backward_pass_cpu() {
        test_unary_backward_pass_impl(SimpleExecutor::new());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_unary_backward_pass_cuda() {
        test_unary_backward_pass_impl(crate::cuda::CudaExecutor::new());
    }

    #[test]
    fn test_matmul_grad_left_simple() {
        // Test matmul_grad_left: dA = dC @ B^T
        // Forward: A[2,3] @ B[3,2] = C[2,2]
        // Given dC[2,2] and B[3,2], compute dA[2,3]

        let dy = vec![1.0f32, 2.0, 3.0, 4.0]; // dC: [2,2]
        let dy_shape = vec![2, 2];
        let b = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // B: [3,2]
        let b_shape = vec![3, 2];

        let da = matmul_grad_left(&dy, &dy_shape, &b, &b_shape);

        // Expected: dC @ B^T = [[1,2],[3,4]] @ [[1,3,5],[2,4,6]]
        // = [[1*1+2*2, 1*3+2*4, 1*5+2*6], [3*1+4*2, 3*3+4*4, 3*5+4*6]]
        // = [[5, 11, 17], [11, 25, 39]]
        let expected = vec![5.0f32, 11.0, 17.0, 11.0, 25.0, 39.0];

        assert_eq!(da.len(), 6);
        assert_approx_eq!(da, expected);
    }

    #[test]
    fn test_matmul_grad_right_simple() {
        // Test matmul_grad_right: dB = A^T @ dC
        // Forward: A[2,3] @ B[3,2] = C[2,2]
        // Given A[2,3] and dC[2,2], compute dB[3,2]

        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // A: [2,3]
        let a_shape = vec![2, 3];
        let dy = vec![1.0f32, 2.0, 3.0, 4.0]; // dC: [2,2]
        let dy_shape = vec![2, 2];

        let db = matmul_grad_right(&a, &a_shape, &dy, &dy_shape);

        // Expected: A^T @ dC = [[1,4],[2,5],[3,6]] @ [[1,2],[3,4]]
        // = [[1*1+4*3, 1*2+4*4], [2*1+5*3, 2*2+5*4], [3*1+6*3, 3*2+6*4]]
        // = [[13, 18], [17, 24], [21, 30]]
        let expected = vec![13.0f32, 18.0, 17.0, 24.0, 21.0, 30.0];

        assert_eq!(db.len(), 6);
        assert_approx_eq!(db, expected);
    }

    fn test_matmul_grad_identity_impl<E: Executor<f32>>(mut executor: E) {
        // Test matmul gradient with identity matrix
        // Forward: A[3,3] @ I[3,3] = A[3,3]
        // dA should equal dC when B is identity

        let a_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let identity = vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

        let a = tensor::Parameter::new(a_data.clone(), vec![3, 3]);
        let a_id = a.id();
        let b = tensor::Parameter::new(identity, vec![3, 3]);
        let node = tensor::TensorExpr::from(a).matmul(b);

        let mut graph = tensor::graph::TensorGraph::new();
        let loss_node = node.lower_to_graph(&mut graph);

        let inputs = std::collections::HashMap::new();
        let seed_grad = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        executor.forward(&graph, inputs).unwrap();
        let result = executor
            .backward(&graph, loss_node, Some(seed_grad.clone()))
            .unwrap();

        let grad_a = result.grads_by_param.get(&a_id).unwrap();

        // dA = dC @ I^T = dC @ I = dC
        assert_eq!(grad_a.len(), 9);
        assert_approx_eq!(grad_a, &seed_grad);
    }

    #[test]
    fn test_matmul_grad_identity_cpu() {
        test_matmul_grad_identity_impl(SimpleExecutor::new());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_matmul_grad_identity_cuda() {
        test_matmul_grad_identity_impl(crate::cuda::CudaExecutor::new());
    }

    fn test_matmul_grad_asymmetric_impl<E: Executor<f32>>(mut executor: E) {
        // Test with very asymmetric matrices to catch dimension errors
        // Forward: A[5,100] @ B[100,3] = C[5,3]

        let m = 5;
        let k = 100;
        let n = 3;

        let a_data = vec![0.1f32; m * k];
        let b_data = vec![0.2f32; k * n];

        let a = tensor::Parameter::new(a_data, vec![m, k]);
        let a_id = a.id();
        let b = tensor::Parameter::new(b_data, vec![k, n]);
        let b_id = b.id();
        let node = tensor::TensorExpr::from(a).matmul(b);

        let mut graph = tensor::graph::TensorGraph::new();
        let loss_node = node.lower_to_graph(&mut graph);

        let inputs = std::collections::HashMap::new();
        let seed_grad = vec![1.0f32; m * n];

        executor.forward(&graph, inputs).unwrap();
        let result = executor
            .backward(&graph, loss_node, Some(seed_grad))
            .unwrap();

        let grad_a = result.grads_by_param.get(&a_id).unwrap();
        let grad_b = result.grads_by_param.get(&b_id).unwrap();

        assert_eq!(grad_a.len(), m * k, "dA should have shape [5, 100]");
        assert_eq!(grad_b.len(), k * n, "dB should have shape [100, 3]");

        // Verify gradients are non-zero and finite
        for &val in grad_a.iter() {
            assert!(val.is_finite(), "dA contains non-finite value");
        }
        for &val in grad_b.iter() {
            assert!(val.is_finite(), "dB contains non-finite value");
        }
    }

    #[test]
    fn test_matmul_grad_asymmetric_cpu() {
        test_matmul_grad_asymmetric_impl(SimpleExecutor::new());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_matmul_grad_asymmetric_cuda() {
        test_matmul_grad_asymmetric_impl(crate::cuda::CudaExecutor::new());
    }

    fn test_matmul_backward_in_graph_impl<E: Executor<f32>>(mut executor: E) {
        // Test matmul backward pass integrated in graph
        // Forward: A[2,3] @ B[3,2] = C[2,2]
        // Backward: dA = dC @ B^T, dB = A^T @ dC

        let a_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2, 3]
        let b_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // [3, 2]

        let a = tensor::Parameter::new(a_data.clone(), vec![2, 3]);
        let a_id = a.id();
        let b = tensor::Parameter::new(b_data.clone(), vec![3, 2]);
        let b_id = b.id();
        let node = tensor::TensorExpr::from(a).matmul(b);

        let mut graph = tensor::graph::TensorGraph::new();
        let loss_node = node.lower_to_graph(&mut graph);

        let inputs = std::collections::HashMap::new();
        let seed_grad = vec![1.0f32, 1.0, 1.0, 1.0]; // [2, 2]

        executor.forward(&graph, inputs).unwrap();
        let result = executor
            .backward(&graph, loss_node, Some(seed_grad.clone()))
            .unwrap();

        // Verify we have gradients for both parameters
        assert_eq!(result.grads_by_param.len(), 2);

        let grad_a = result.grads_by_param.get(&a_id).unwrap();
        let grad_b = result.grads_by_param.get(&b_id).unwrap();

        assert_eq!(grad_a.len(), 6, "dA should have 6 elements [2,3]");
        assert_eq!(grad_b.len(), 6, "dB should have 6 elements [3,2]");

        // Manually compute expected gradients
        let expected_da = matmul_grad_left(&seed_grad, &[2, 2], &b_data, &[3, 2]);
        let expected_db = matmul_grad_right(&a_data, &[2, 3], &seed_grad, &[2, 2]);

        assert_approx_eq!(grad_a, &expected_da);
        assert_approx_eq!(grad_b, &expected_db);
    }

    #[test]
    fn test_matmul_backward_in_graph_cpu() {
        test_matmul_backward_in_graph_impl(SimpleExecutor::new());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_matmul_backward_in_graph_cuda() {
        test_matmul_backward_in_graph_impl(crate::cuda::CudaExecutor::new());
    }

    fn test_matmul_grad_chain_rule_impl<E: Executor<f32>>(mut executor: E) {
        // Test that matmul gradients compose correctly with chain rule
        // z = (A @ B) @ C, verify dA and dB are correct

        let a = tensor::Parameter::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
        let a_id = a.id();
        let b = tensor::Parameter::new(vec![1.0f32, 0.0, 0.0, 1.0], vec![2, 2]); // Identity
        let c = tensor::Parameter::new(vec![2.0f32, 0.0, 0.0, 2.0], vec![2, 2]); // 2*Identity

        let ab = tensor::TensorExpr::from(a).matmul(b);
        let z = ab.matmul(c);

        let mut graph = tensor::graph::TensorGraph::new();
        let loss_node = z.lower_to_graph(&mut graph);

        let inputs = std::collections::HashMap::new();
        let seed_grad = vec![1.0f32; 4]; // [2, 2]

        executor.forward(&graph, inputs).unwrap();
        let result = executor
            .backward(&graph, loss_node, Some(seed_grad))
            .unwrap();

        // With B=I and C=2I, z = 2A, so dA should be 2*seed_grad
        let grad_a = result.grads_by_param.get(&a_id).unwrap();

        assert_eq!(grad_a.len(), 4);
        for &val in grad_a.iter() {
            assert!(
                (val - 2.0).abs() < 1e-4,
                "Expected gradient 2.0, got {}",
                val
            );
        }
    }

    #[test]
    fn test_matmul_grad_chain_rule_cpu() {
        test_matmul_grad_chain_rule_impl(SimpleExecutor::new());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_matmul_grad_chain_rule_cuda() {
        test_matmul_grad_chain_rule_impl(crate::cuda::CudaExecutor::new());
    }

    fn test_matmul_grad_rectangular_case<E: Executor<f32>>(
        mut executor: E,
        m: usize,
        k: usize,
        n: usize,
    ) {
        let a_data = vec![0.5f32; m * k];
        let b_data = vec![0.3f32; k * n];

        let a = tensor::Parameter::new(a_data, vec![m, k]);
        let b = tensor::Parameter::new(b_data, vec![k, n]);
        let node = tensor::TensorExpr::from(a).matmul(b);

        let mut graph = tensor::graph::TensorGraph::new();
        let loss_node = node.lower_to_graph(&mut graph);

        let inputs = std::collections::HashMap::new();
        let seed_grad = vec![1.0f32; m * n];

        executor.forward(&graph, inputs).unwrap();
        let result = executor
            .backward(&graph, loss_node, Some(seed_grad))
            .unwrap();

        // Get all available parameter IDs (sorted)
        let mut param_ids: Vec<_> = result.grads_by_param.keys().copied().collect();
        param_ids.sort();

        assert_eq!(
            param_ids.len(),
            2,
            "Expected 2 parameters, found {} for test case ({}, {}) @ ({}, {})",
            param_ids.len(),
            m,
            k,
            k,
            n
        );

        let grad_a = result.grads_by_param.get(&param_ids[0]).unwrap();
        let grad_b = result.grads_by_param.get(&param_ids[1]).unwrap();

        assert_eq!(
            grad_a.len(),
            m * k,
            "dA should have shape [{}, {}] for test case ({}, {}) @ ({}, {})",
            m,
            k,
            m,
            k,
            k,
            n
        );
        assert_eq!(
            grad_b.len(),
            k * n,
            "dB should have shape [{}, {}] for test case ({}, {}) @ ({}, {})",
            k,
            n,
            m,
            k,
            k,
            n
        );

        // Verify all gradients are finite
        for &val in grad_a.iter().chain(grad_b.iter()) {
            assert!(val.is_finite(), "Gradient contains non-finite value");
        }
    }

    #[test]
    fn test_matmul_grad_rectangular_cpu() {
        // Test matmul gradient with rectangular matrices of various sizes
        test_matmul_grad_rectangular_case(SimpleExecutor::new(), 2, 3, 4);
        test_matmul_grad_rectangular_case(SimpleExecutor::new(), 1, 5, 1);
        test_matmul_grad_rectangular_case(SimpleExecutor::new(), 10, 20, 15);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_matmul_grad_rectangular_cuda() {
        // Test matmul gradient with rectangular matrices of various sizes
        test_matmul_grad_rectangular_case(crate::cuda::CudaExecutor::new(), 2, 3, 4);
        test_matmul_grad_rectangular_case(crate::cuda::CudaExecutor::new(), 1, 5, 1);
        test_matmul_grad_rectangular_case(crate::cuda::CudaExecutor::new(), 10, 20, 15);
    }

    fn test_matmul_grad_numerical_impl<E: Executor<f32>>(mut executor: E) {
        // Numerical gradient check using finite differences
        let a_data = vec![1.0f32, 2.0, 3.0, 4.0]; // [2, 2]
        let b_data = vec![0.5f32, 1.0, 1.5, 2.0]; // [2, 2]

        let epsilon = 1e-4;

        // Create graph for forward pass
        let a = tensor::Parameter::new(a_data.clone(), vec![2, 2]);
        let a_id = a.id();
        let b = tensor::Parameter::new(b_data.clone(), vec![2, 2]);
        let node = tensor::TensorExpr::from(a).matmul(b);

        let mut graph = tensor::graph::TensorGraph::new();
        let loss_node = node.lower_to_graph(&mut graph);

        // Forward and backward
        executor.forward(&graph, Default::default()).unwrap();
        let seed_grad = vec![1.0f32; 4];
        let result = executor
            .backward(&graph, loss_node, Some(seed_grad.clone()))
            .unwrap();

        let grad_a = result.grads_by_param.get(&a_id).unwrap();

        // Numerical gradient for each element of A
        for idx in 0..4 {
            let mut a_plus = a_data.clone();
            a_plus[idx] += epsilon;
            let a_param_plus = tensor::Parameter::new(a_plus, vec![2, 2]);
            let b_param = tensor::Parameter::new(b_data.clone(), vec![2, 2]);
            let node_plus = tensor::TensorExpr::from(a_param_plus).matmul(b_param);
            let mut graph_plus = tensor::graph::TensorGraph::new();
            let _loss_plus_node = node_plus.lower_to_graph(&mut graph_plus);
            let out_plus = executor.forward(&graph_plus, Default::default()).unwrap();

            let mut a_minus = a_data.clone();
            a_minus[idx] -= epsilon;
            let a_param_minus = tensor::Parameter::new(a_minus, vec![2, 2]);
            let b_param = tensor::Parameter::new(b_data.clone(), vec![2, 2]);
            let node_minus = tensor::TensorExpr::from(a_param_minus).matmul(b_param);
            let mut graph_minus = tensor::graph::TensorGraph::new();
            let _loss_minus_node = node_minus.lower_to_graph(&mut graph_minus);
            let out_minus = executor.forward(&graph_minus, Default::default()).unwrap();

            // Compute numerical gradient: (f(x+h) - f(x-h)) / 2h
            let mut numerical_grad = 0.0f32;
            for i in 0..4 {
                numerical_grad += (out_plus[i] - out_minus[i]) / (2.0 * epsilon) * seed_grad[i];
            }

            let analytical_grad = grad_a[idx];
            let rel_error = ((analytical_grad - numerical_grad).abs()
                / (analytical_grad.abs() + numerical_grad.abs() + 1e-8))
                .abs();

            assert!(
                rel_error < 1e-3,
                "Numerical gradient check failed for A[{}]: analytical={}, numerical={}, rel_error={}",
                idx,
                analytical_grad,
                numerical_grad,
                rel_error
            );
        }
    }

    #[test]
    fn test_matmul_grad_numerical_cpu() {
        test_matmul_grad_numerical_impl(SimpleExecutor::new());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_matmul_grad_numerical_cuda() {
        test_matmul_grad_numerical_impl(crate::cuda::CudaExecutor::new());
    }
}
