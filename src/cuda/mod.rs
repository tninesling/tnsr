use crate::Executor;
use crate::alloc::{AllocStats, CudaBufferPool};
use crate::graph::{TensorGraph, TensorGraphNode, WithGrad, liveness};
use crate::tensor;
use anyhow::{Context as _, Result, anyhow};
use cudarc::driver::{CudaContext, CudaModule, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::trace_span;

static PTX: &str = include_str!("kernels.ptx");

fn checked_product(shape: &[usize], description: &str) -> Result<usize> {
    shape.iter().try_fold(1usize, |size, &dimension| {
        size.checked_mul(dimension)
            .with_context(|| format!("{description} size overflows usize for shape {shape:?}"))
    })
}

fn rowmajor_strides_u64(shape: &[usize], description: &str) -> Result<Vec<u64>> {
    let mut strides = vec![0u64; shape.len()];
    let mut stride = 1u64;
    for axis in (0..shape.len()).rev() {
        strides[axis] = stride;
        let dimension = u64::try_from(shape[axis]).with_context(|| {
            format!(
                "{description} dimension {} does not fit in u64",
                shape[axis]
            )
        })?;
        stride = stride
            .checked_mul(dimension)
            .with_context(|| format!("{description} strides overflow u64 for shape {shape:?}"))?;
    }
    Ok(strides)
}

fn broadcast_batch_index(
    output_index: usize,
    output_shape: &[usize],
    output_strides: &[usize],
    input_shape: &[usize],
    input_strides: &[usize],
) -> usize {
    let rank_difference = output_shape.len() - input_shape.len();
    let mut input_index = 0usize;
    let mut remainder = output_index;
    for (axis, &stride) in output_strides.iter().enumerate() {
        let coordinate = remainder / stride;
        remainder %= stride;
        if axis >= rank_difference {
            let input_axis = axis - rank_difference;
            if input_shape[input_axis] != 1 {
                input_index += coordinate * input_strides[input_axis];
            }
        }
    }
    input_index
}

enum ConvKind {
    Forward,
    Transpose,
    BackwardWeight,
}

enum PoolKind {
    Forward,
    Backward,
}

pub struct CudaExecutor {
    device: Arc<CudaContext>,
    module: Arc<CudaModule>,
    values: HashMap<petgraph::graph::NodeIndex, CudaSlice<f32>>,
    pool: RefCell<CudaBufferPool>,
    stats: AllocStats,
}

impl Default for CudaExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl CudaExecutor {
    /// Create a new CUDA executor, panicking if CUDA initialization fails.
    ///
    /// For fallible initialization, use [`CudaExecutor::try_new`].
    pub fn new() -> Self {
        Self::try_new().expect("Failed to initialize CUDA executor")
    }

    /// Try to create a new CUDA executor, returning an error if initialization fails.
    ///
    /// This is useful for runtime backend detection where you want to fall back
    /// to CPU if CUDA is not available.
    pub fn try_new() -> Result<Self> {
        // cudarc's dynamic loader panics when libcuda is absent; keep this API fallible.
        let device = match std::panic::catch_unwind(|| CudaContext::new(0)) {
            Ok(device) => device.context("Failed to initialize CUDA device 0")?,
            Err(_) => anyhow::bail!(
                "Failed to initialize CUDA device 0: CUDA driver library is unavailable"
            ),
        };
        let module = device
            .load_module(Ptx::from_src(PTX))
            .context("Failed to load PTX module with cudarc")?;
        Ok(CudaExecutor {
            device,
            module,
            values: HashMap::new(),
            pool: RefCell::new(CudaBufferPool::default()),
            stats: AllocStats::default(),
        })
    }

    fn take_buffer(&self, len: usize) -> CudaSlice<f32> {
        self.pool
            .borrow_mut()
            .take(&self.device.default_stream(), len)
            .expect("Failed to allocate CUDA output buffer")
    }

    #[cfg(feature = "fusion")]
    fn give_buffer(&self, buf: CudaSlice<f32>) {
        self.pool.borrow_mut().give(buf);
    }

    fn neg(&self, input: &CudaSlice<f32>) -> CudaSlice<f32> {
        let _span = trace_span!("neg").entered();
        let len = input.len();
        let len_u64 = len as u64;
        let stream = self.device.default_stream();
        let mut out = self.take_buffer(len);
        let f = self.module.load_function("neg").unwrap();
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let mut launcher = stream.launch_builder(&f);
        launcher.arg(input);
        launcher.arg(&len_u64);
        launcher.arg(&mut out);
        launcher.arg(&len_u64);
        unsafe { launcher.launch(cfg) }.expect("CUDA neg failed");
        out
    }

    fn exp(&self, input: &CudaSlice<f32>) -> CudaSlice<f32> {
        let _span = trace_span!("exp").entered();
        let len = input.len();
        let len_u64 = len as u64;
        let stream = self.device.default_stream();
        let mut out = self.take_buffer(len);
        let f = self.module.load_function("exp").unwrap();
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let mut launcher = stream.launch_builder(&f);
        launcher.arg(input);
        launcher.arg(&len_u64);
        launcher.arg(&mut out);
        launcher.arg(&len_u64);
        unsafe { launcher.launch(cfg) }.expect("CUDA exp failed");
        out
    }

    fn log(&self, input: &CudaSlice<f32>) -> CudaSlice<f32> {
        let _span = trace_span!("log").entered();
        let len = input.len();
        let len_u64 = len as u64;
        let stream = self.device.default_stream();
        let mut out = self.take_buffer(len);
        let f = self.module.load_function("log").unwrap();
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let mut launcher = stream.launch_builder(&f);
        launcher.arg(input);
        launcher.arg(&len_u64);
        launcher.arg(&mut out);
        launcher.arg(&len_u64);
        unsafe { launcher.launch(cfg) }.expect("CUDA log failed");
        out
    }

    fn relu(&self, input: &CudaSlice<f32>) -> CudaSlice<f32> {
        let _span = trace_span!("relu").entered();
        let len = input.len();
        let len_u64 = len as u64;
        let stream = self.device.default_stream();
        let mut out = self.take_buffer(len);
        let f = self.module.load_function("relu").unwrap();
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let mut launcher = stream.launch_builder(&f);
        launcher.arg(input);
        launcher.arg(&len_u64);
        launcher.arg(&mut out);
        launcher.arg(&len_u64);
        unsafe { launcher.launch(cfg) }.expect("CUDA relu failed");
        out
    }

    fn add(&self, lhs: &CudaSlice<f32>, rhs: &CudaSlice<f32>) -> CudaSlice<f32> {
        let _span = trace_span!("add").entered();
        assert_eq!(lhs.len(), rhs.len(), "binary op input length mismatch");
        let len = lhs.len();
        let len_u64 = len as u64;
        let stream = self.device.default_stream();
        let mut out = self.take_buffer(len);
        let f = self.module.load_function("add").unwrap();
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let mut launcher = stream.launch_builder(&f);
        launcher.arg(lhs);
        launcher.arg(&len_u64);
        launcher.arg(rhs);
        launcher.arg(&len_u64);
        launcher.arg(&mut out);
        launcher.arg(&len_u64);
        unsafe { launcher.launch(cfg) }.expect("CUDA add failed");
        out
    }

    fn sub(&self, lhs: &CudaSlice<f32>, rhs: &CudaSlice<f32>) -> CudaSlice<f32> {
        let _span = trace_span!("sub").entered();
        assert_eq!(lhs.len(), rhs.len(), "binary op input length mismatch");
        let len = lhs.len();
        let len_u64 = len as u64;
        let stream = self.device.default_stream();
        let mut out = self.take_buffer(len);
        let f = self.module.load_function("sub").unwrap();
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let mut launcher = stream.launch_builder(&f);
        launcher.arg(lhs);
        launcher.arg(&len_u64);
        launcher.arg(rhs);
        launcher.arg(&len_u64);
        launcher.arg(&mut out);
        launcher.arg(&len_u64);
        unsafe { launcher.launch(cfg) }.expect("CUDA sub failed");
        out
    }

    fn mul(&self, lhs: &CudaSlice<f32>, rhs: &CudaSlice<f32>) -> CudaSlice<f32> {
        let _span = trace_span!("mul").entered();
        assert_eq!(lhs.len(), rhs.len(), "binary op input length mismatch");
        let len = lhs.len();
        let len_u64 = len as u64;
        let stream = self.device.default_stream();
        let mut out = self.take_buffer(len);
        let f = self.module.load_function("mul").unwrap();
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let mut launcher = stream.launch_builder(&f);
        launcher.arg(lhs);
        launcher.arg(&len_u64);
        launcher.arg(rhs);
        launcher.arg(&len_u64);
        launcher.arg(&mut out);
        launcher.arg(&len_u64);
        unsafe { launcher.launch(cfg) }.expect("CUDA mul failed");
        out
    }

    fn div(&self, lhs: &CudaSlice<f32>, rhs: &CudaSlice<f32>) -> CudaSlice<f32> {
        let _span = trace_span!("div").entered();
        assert_eq!(lhs.len(), rhs.len(), "binary op input length mismatch");
        let len = lhs.len();
        let len_u64 = len as u64;
        let stream = self.device.default_stream();
        let mut out = self.take_buffer(len);
        let f = self.module.load_function("div").unwrap();
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let mut launcher = stream.launch_builder(&f);
        launcher.arg(lhs);
        launcher.arg(&len_u64);
        launcher.arg(rhs);
        launcher.arg(&len_u64);
        launcher.arg(&mut out);
        launcher.arg(&len_u64);
        unsafe { launcher.launch(cfg) }.expect("CUDA div failed");
        out
    }

    fn gt(&self, lhs: &CudaSlice<f32>, rhs: &CudaSlice<f32>) -> CudaSlice<f32> {
        let _span = trace_span!("gt").entered();
        assert_eq!(lhs.len(), rhs.len(), "binary op input length mismatch");
        let len = lhs.len();
        let len_u64 = len as u64;
        let stream = self.device.default_stream();
        let mut out = self.take_buffer(len);
        let f = self.module.load_function("gt").unwrap();
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let mut launcher = stream.launch_builder(&f);
        launcher.arg(lhs);
        launcher.arg(&len_u64);
        launcher.arg(rhs);
        launcher.arg(&len_u64);
        launcher.arg(&mut out);
        launcher.arg(&len_u64);
        unsafe { launcher.launch(cfg) }.expect("CUDA gt failed");
        out
    }

    fn mask(&self, values: &CudaSlice<f32>, condition: &CudaSlice<f32>) -> CudaSlice<f32> {
        let _span = trace_span!("mask").entered();
        assert_eq!(
            values.len(),
            condition.len(),
            "binary op input length mismatch"
        );
        let len = values.len();
        let len_u64 = len as u64;
        let stream = self.device.default_stream();
        let mut out = self.take_buffer(len);
        let f = self.module.load_function("mask").unwrap();
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let mut launcher = stream.launch_builder(&f);
        launcher.arg(values);
        launcher.arg(&len_u64);
        launcher.arg(condition);
        launcher.arg(&len_u64);
        launcher.arg(&mut out);
        launcher.arg(&len_u64);
        unsafe { launcher.launch(cfg) }.expect("CUDA mask failed");
        out
    }

    fn matmul(
        &self,
        lhs: &CudaSlice<f32>,
        rhs: &CudaSlice<f32>,
        lhs_shape: &[usize],
        rhs_shape: &[usize],
        output_shape: &[usize],
    ) -> Result<CudaSlice<f32>> {
        let _span = trace_span!("matmul").entered();
        anyhow::ensure!(
            lhs_shape.len() >= 2 && rhs_shape.len() >= 2,
            "MatMul operands must have rank >= 2, got {}D and {}D",
            lhs_shape.len(),
            rhs_shape.len()
        );
        anyhow::ensure!(
            output_shape.len() >= 2,
            "MatMul output must have rank >= 2, got {}D",
            output_shape.len()
        );

        let m = lhs_shape[lhs_shape.len() - 2];
        let k = lhs_shape[lhs_shape.len() - 1];
        let rhs_k = rhs_shape[rhs_shape.len() - 2];
        let n = rhs_shape[rhs_shape.len() - 1];
        anyhow::ensure!(
            k == rhs_k,
            "MatMul inner dimension mismatch: lhs has k={k}, rhs has {rhs_k}"
        );
        anyhow::ensure!(
            output_shape[output_shape.len() - 2..] == [m, n],
            "MatMul output shape {output_shape:?} does not end in [{m}, {n}]"
        );

        let lhs_batch_shape = &lhs_shape[..lhs_shape.len() - 2];
        let rhs_batch_shape = &rhs_shape[..rhs_shape.len() - 2];
        let output_batch_shape = &output_shape[..output_shape.len() - 2];
        for (name, input_batch_shape) in [("left", lhs_batch_shape), ("right", rhs_batch_shape)] {
            anyhow::ensure!(
                input_batch_shape.len() <= output_batch_shape.len(),
                "MatMul {name} batch rank {} exceeds output batch rank {}",
                input_batch_shape.len(),
                output_batch_shape.len()
            );
            let rank_difference = output_batch_shape.len() - input_batch_shape.len();
            for (axis, &dimension) in input_batch_shape.iter().enumerate() {
                let output_dimension = output_batch_shape[rank_difference + axis];
                anyhow::ensure!(
                    dimension == 1 || dimension == output_dimension,
                    "MatMul {name} batch dimension {dimension} cannot broadcast to {output_dimension}"
                );
            }
        }
        for output_axis in 0..output_batch_shape.len() {
            let lhs_axis = output_axis
                .checked_sub(output_batch_shape.len() - lhs_batch_shape.len())
                .map(|axis| lhs_batch_shape[axis])
                .unwrap_or(1);
            let rhs_axis = output_axis
                .checked_sub(output_batch_shape.len() - rhs_batch_shape.len())
                .map(|axis| rhs_batch_shape[axis])
                .unwrap_or(1);
            let expected = if lhs_axis == 1 { rhs_axis } else { lhs_axis };
            anyhow::ensure!(
                output_batch_shape[output_axis] == expected,
                "MatMul output batch shape {output_batch_shape:?} does not match broadcast of {lhs_batch_shape:?} and {rhs_batch_shape:?}"
            );
        }

        let lhs_len = checked_product(lhs_shape, "MatMul left operand")?;
        let rhs_len = checked_product(rhs_shape, "MatMul right operand")?;
        let output_len = checked_product(output_shape, "MatMul output")?;
        anyhow::ensure!(
            lhs.len() == lhs_len,
            "MatMul left buffer length {} does not match shape {lhs_shape:?} ({lhs_len} elements)",
            lhs.len()
        );
        anyhow::ensure!(
            rhs.len() == rhs_len,
            "MatMul right buffer length {} does not match shape {rhs_shape:?} ({rhs_len} elements)",
            rhs.len()
        );

        let batch_count = checked_product(output_batch_shape, "MatMul output batch")?;
        let lhs_matrix_len = m
            .checked_mul(k)
            .context("MatMul left matrix size overflows usize")?;
        let rhs_matrix_len = k
            .checked_mul(n)
            .context("MatMul right matrix size overflows usize")?;
        let output_matrix_len = m
            .checked_mul(n)
            .context("MatMul output matrix size overflows usize")?;
        anyhow::ensure!(
            batch_count.checked_mul(output_matrix_len) == Some(output_len),
            "MatMul output shape has inconsistent element count"
        );

        let mut out = self.take_buffer(output_len);
        if batch_count == 0 || output_matrix_len == 0 {
            return Ok(out);
        }

        let launch_len = u32::try_from(output_matrix_len)
            .context("MatMul matrix output is too large for a CUDA launch")?;
        let output_batch_strides = rowmajor_strides_u64(output_batch_shape, "MatMul output batch")?
            .into_iter()
            .map(|stride| usize::try_from(stride).context("MatMul batch stride exceeds usize"))
            .collect::<Result<Vec<_>>>()?;
        let lhs_batch_strides = rowmajor_strides_u64(lhs_batch_shape, "MatMul left batch")?
            .into_iter()
            .map(|stride| usize::try_from(stride).context("MatMul left batch stride exceeds usize"))
            .collect::<Result<Vec<_>>>()?;
        let rhs_batch_strides = rowmajor_strides_u64(rhs_batch_shape, "MatMul right batch")?
            .into_iter()
            .map(|stride| {
                usize::try_from(stride).context("MatMul right batch stride exceeds usize")
            })
            .collect::<Result<Vec<_>>>()?;

        let stream = self.device.default_stream();
        let f = self
            .module
            .load_function("matmul")
            .context("Failed to load CUDA matmul kernel")?;
        let cfg = LaunchConfig::for_num_elems(launch_len);
        let lhs_len_u64 = u64::try_from(lhs_matrix_len)
            .context("MatMul left matrix length does not fit in u64")?;
        let rhs_len_u64 = u64::try_from(rhs_matrix_len)
            .context("MatMul right matrix length does not fit in u64")?;
        let m_u64 = m as u64;
        let n_u64 = n as u64;
        let k_u64 = k as u64;
        for batch in 0..batch_count {
            let lhs_batch = broadcast_batch_index(
                batch,
                output_batch_shape,
                &output_batch_strides,
                lhs_batch_shape,
                &lhs_batch_strides,
            );
            let rhs_batch = broadcast_batch_index(
                batch,
                output_batch_shape,
                &output_batch_strides,
                rhs_batch_shape,
                &rhs_batch_strides,
            );
            let lhs_start = lhs_batch
                .checked_mul(lhs_matrix_len)
                .context("MatMul left batch offset overflows usize")?;
            let rhs_start = rhs_batch
                .checked_mul(rhs_matrix_len)
                .context("MatMul right batch offset overflows usize")?;
            let output_start = batch
                .checked_mul(output_matrix_len)
                .context("MatMul output batch offset overflows usize")?;
            let lhs_view = lhs.slice(lhs_start..lhs_start + lhs_matrix_len);
            let rhs_view = rhs.slice(rhs_start..rhs_start + rhs_matrix_len);
            let mut output_view = out.slice_mut(output_start..output_start + output_matrix_len);
            let mut launcher = stream.launch_builder(&f);
            launcher.arg(&lhs_view);
            launcher.arg(&lhs_len_u64);
            launcher.arg(&rhs_view);
            launcher.arg(&rhs_len_u64);
            launcher.arg(&mut output_view);
            launcher.arg(&m_u64);
            launcher.arg(&n_u64);
            launcher.arg(&k_u64);
            unsafe { launcher.launch(cfg) }
                .with_context(|| format!("CUDA matmul kernel launch failed for batch {batch}"))?;
        }

        Ok(out)
    }

    fn transpose(
        &self,
        input: &CudaSlice<f32>,
        rows: usize,
        cols: usize,
    ) -> Result<CudaSlice<f32>> {
        let _span = trace_span!("transpose").entered();
        let len = input.len();
        let expected_len = rows
            .checked_mul(cols)
            .context("Transpose matrix size overflows usize")?;
        anyhow::ensure!(
            len == expected_len,
            "Transpose input buffer has {len} elements but shape [{rows}, {cols}] has {expected_len}"
        );
        let len_u64 = len as u64;
        let mut out = self.take_buffer(len);
        if len == 0 {
            return Ok(out);
        }
        let launch_len =
            u32::try_from(len).context("Transpose output is too large for a CUDA launch")?;
        let stream = self.device.default_stream();
        let f = self
            .module
            .load_function("transpose_2d")
            .context("Failed to load CUDA transpose_2d kernel")?;
        let cfg = LaunchConfig::for_num_elems(launch_len);
        let rows_u64 = rows as u64;
        let cols_u64 = cols as u64;
        let mut launcher = stream.launch_builder(&f);
        launcher.arg(input);
        launcher.arg(&len_u64);
        launcher.arg(&len_u64);
        launcher.arg(&mut out);
        launcher.arg(&rows_u64);
        launcher.arg(&cols_u64);
        unsafe { launcher.launch(cfg) }.context("CUDA transpose_2d kernel launch failed")?;
        Ok(out)
    }

    fn permute(
        &self,
        input: &CudaSlice<f32>,
        input_shape: &[usize],
        output_shape: &[usize],
        axes: &[usize],
    ) -> Result<CudaSlice<f32>> {
        anyhow::ensure!(
            axes.len() == input_shape.len() && output_shape.len() == input_shape.len(),
            "Permute rank mismatch: input is {}D, output is {}D, and has {} axes",
            input_shape.len(),
            output_shape.len(),
            axes.len()
        );
        let mut seen = vec![false; axes.len()];
        for (output_axis, &input_axis) in axes.iter().enumerate() {
            anyhow::ensure!(
                input_axis < axes.len(),
                "Permute axis {input_axis} is out of bounds for rank {}",
                axes.len()
            );
            anyhow::ensure!(
                !seen[input_axis],
                "Permute contains duplicate axis {input_axis}"
            );
            seen[input_axis] = true;
            anyhow::ensure!(
                output_shape[output_axis] == input_shape[input_axis],
                "Permute output shape {output_shape:?} does not match input shape {input_shape:?} and axes {axes:?}"
            );
        }

        let input_len = checked_product(input_shape, "Permute input")?;
        let output_len = checked_product(output_shape, "Permute output")?;
        anyhow::ensure!(
            input.len() == input_len,
            "Permute input buffer length {} does not match shape {input_shape:?} ({input_len} elements)",
            input.len()
        );
        anyhow::ensure!(
            output_len == input_len,
            "Permute changes element count from {input_len} to {output_len}"
        );

        let mut output = self.take_buffer(output_len);
        if output_len == 0 {
            return Ok(output);
        }

        let input_strides = rowmajor_strides_u64(input_shape, "Permute input")?;
        let mapped_input_strides: Vec<u64> = axes.iter().map(|&axis| input_strides[axis]).collect();
        let output_strides = rowmajor_strides_u64(output_shape, "Permute output")?;
        let stream = self.device.default_stream();
        let mapped_input_strides_device = if mapped_input_strides.is_empty() {
            stream
                .null::<u64>()
                .context("Failed to create empty CUDA permute input strides")?
        } else {
            stream
                .memcpy_stod(&mapped_input_strides)
                .context("Failed to copy permute input strides to CUDA device")?
        };
        let output_strides_device = if output_strides.is_empty() {
            stream
                .null::<u64>()
                .context("Failed to create empty CUDA permute output strides")?
        } else {
            stream
                .memcpy_stod(&output_strides)
                .context("Failed to copy permute output strides to CUDA device")?
        };
        let f = self
            .module
            .load_function("permute")
            .context("Failed to load CUDA permute kernel")?;
        let launch_len =
            u32::try_from(output_len).context("Permute output is too large for a CUDA launch")?;
        let cfg = LaunchConfig::for_num_elems(launch_len);
        let input_len_u64 = u64::try_from(input_len).context("Permute input length exceeds u64")?;
        let rank_u64 = u64::try_from(axes.len()).context("Permute rank exceeds u64")?;
        let output_strides_len_u64 = u64::try_from(output_strides.len())
            .context("Permute output strides length exceeds u64")?;
        let output_len_u64 =
            u64::try_from(output_len).context("Permute output length exceeds u64")?;
        let mut launcher = stream.launch_builder(&f);
        launcher.arg(input);
        launcher.arg(&input_len_u64);
        launcher.arg(&mut output);
        launcher.arg(&mapped_input_strides_device);
        launcher.arg(&rank_u64);
        launcher.arg(&output_strides_device);
        launcher.arg(&output_strides_len_u64);
        launcher.arg(&output_len_u64);
        unsafe { launcher.launch(cfg) }.context("CUDA permute kernel launch failed")?;
        Ok(output)
    }

    /// Get the value of a specific node from the executor's cache after execution
    pub fn get_value(&self, node_idx: petgraph::graph::NodeIndex) -> Option<Vec<f32>> {
        self.values.get(&node_idx).map(|cuda_slice| {
            let mut host_vec = vec![0.0f32; cuda_slice.len()];
            self.device
                .default_stream()
                .memcpy_dtoh(cuda_slice, &mut host_vec)
                .unwrap();
            host_vec
        })
    }

    /// Memory allocation statistics gathered during execution.
    ///
    /// Counters are cumulative across executions; call
    /// [`CudaExecutor::reset_stats`] to start fresh.
    pub fn stats(&self) -> &AllocStats {
        &self.stats
    }

    /// Reset allocation counters without discarding pooled device buffers.
    pub fn reset_stats(&mut self) {
        self.stats.reset();
        self.pool.get_mut().reset_stats();
    }

    /// Discard cached device buffers to release VRAM held by the executor.
    pub fn clear_pool(&mut self) {
        self.pool.get_mut().clear();
    }

    /// Release pinned gradient buffers before the next execution.
    pub fn release_gradients(&mut self, graph: &TensorGraph<f32, WithGrad>) {
        for grad_node in graph.gradient_metadata().param_to_grad.values() {
            if let Some(buf) = self.values.remove(grad_node) {
                self.pool.get_mut().give(buf);
            }
        }
    }

    fn conv_host_fallback<G>(
        &mut self,
        graph: &TensorGraph<f32, G>,
        node_idx: petgraph::graph::NodeIndex,
        stride: usize,
        padding: usize,
        kind: ConvKind,
    ) -> Result<CudaSlice<f32>> {
        let stream = self.device.default_stream();
        let inputs_idx = graph.inputs(node_idx);

        // Copy inputs from device to host
        let mut host_inputs: Vec<Vec<f32>> = Vec::new();
        for &idx in &inputs_idx {
            let dev = self
                .values
                .get(&idx)
                .with_context(|| format!("Missing input {:?} for conv host fallback", idx))?;
            let mut host = vec![0.0f32; dev.len()];
            stream
                .memcpy_dtoh(dev, &mut host)
                .context("Failed to copy conv input from device")?;
            host_inputs.push(host);
        }

        let out_shape = graph.graph[node_idx].shape();
        let out_len: usize = out_shape.iter().product();
        let mut out_host = vec![0.0f32; out_len];

        match kind {
            ConvKind::Forward => {
                let input = &host_inputs[0];
                let weight = &host_inputs[1];
                let in_shape = graph.graph[inputs_idx[0]].shape();
                let w_shape = graph.graph[inputs_idx[1]].shape();
                let n = in_shape[0];
                let c_in = in_shape[1];
                let h = in_shape[2];
                let w = in_shape[3];
                let c_out = w_shape[0];
                let kh = w_shape[2];
                let kw = w_shape[3];
                let h_out = out_shape[2];
                let w_out = out_shape[3];

                for on in 0..n {
                    for oc in 0..c_out {
                        for ohi in 0..h_out {
                            for owi in 0..w_out {
                                let mut sum = 0.0f32;
                                for ic in 0..c_in {
                                    for khi in 0..kh {
                                        let ih_raw = ohi * stride + khi;
                                        if ih_raw >= padding && ih_raw - padding < h {
                                            let ih = ih_raw - padding;
                                            for kwi in 0..kw {
                                                let iw_raw = owi * stride + kwi;
                                                if iw_raw >= padding && iw_raw - padding < w {
                                                    let iw = iw_raw - padding;
                                                    let in_val =
                                                        input[((on * c_in + ic) * h + ih) * w + iw];
                                                    let w_val = weight
                                                        [((oc * c_in + ic) * kh + khi) * kw + kwi];
                                                    sum += in_val * w_val;
                                                }
                                            }
                                        }
                                    }
                                }
                                out_host[((on * c_out + oc) * h_out + ohi) * w_out + owi] = sum;
                            }
                        }
                    }
                }
            }
            ConvKind::Transpose => {
                let grad = &host_inputs[0];
                let weight = &host_inputs[1];
                let grad_shape = graph.graph[inputs_idx[0]].shape();
                let w_shape = graph.graph[inputs_idx[1]].shape();
                let n = out_shape[0];
                let c_in = out_shape[1];
                let h = out_shape[2];
                let w = out_shape[3];
                let c_out = w_shape[0];
                let kh = w_shape[2];
                let kw = w_shape[3];
                let h_out = grad_shape[2];
                let w_out = grad_shape[3];

                for on in 0..n {
                    for oc in 0..c_out {
                        for ohi in 0..h_out {
                            for owi in 0..w_out {
                                let g = grad[((on * c_out + oc) * h_out + ohi) * w_out + owi];
                                for ic in 0..c_in {
                                    for khi in 0..kh {
                                        let ih_raw = ohi * stride + khi;
                                        if ih_raw >= padding && ih_raw - padding < h {
                                            let ih = ih_raw - padding;
                                            for kwi in 0..kw {
                                                let iw_raw = owi * stride + kwi;
                                                if iw_raw >= padding && iw_raw - padding < w {
                                                    let iw = iw_raw - padding;
                                                    let w_val = weight
                                                        [((oc * c_in + ic) * kh + khi) * kw + kwi];
                                                    out_host
                                                        [((on * c_in + ic) * h + ih) * w + iw] +=
                                                        g * w_val;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            ConvKind::BackwardWeight => {
                let input = &host_inputs[0];
                let grad = &host_inputs[1];
                let in_shape = graph.graph[inputs_idx[0]].shape();
                let grad_shape = graph.graph[inputs_idx[1]].shape();
                let n = in_shape[0];
                let c_in = in_shape[1];
                let h = in_shape[2];
                let w = in_shape[3];
                let c_out = out_shape[0];
                let kh = out_shape[2];
                let kw = out_shape[3];
                let h_out = grad_shape[2];
                let w_out = grad_shape[3];

                for on in 0..n {
                    for oc in 0..c_out {
                        for ohi in 0..h_out {
                            for owi in 0..w_out {
                                let g = grad[((on * c_out + oc) * h_out + ohi) * w_out + owi];
                                for ic in 0..c_in {
                                    for khi in 0..kh {
                                        let ih_raw = ohi * stride + khi;
                                        if ih_raw >= padding && ih_raw - padding < h {
                                            let ih = ih_raw - padding;
                                            for kwi in 0..kw {
                                                let iw_raw = owi * stride + kwi;
                                                if iw_raw >= padding && iw_raw - padding < w {
                                                    let iw = iw_raw - padding;
                                                    let in_val =
                                                        input[((on * c_in + ic) * h + ih) * w + iw];
                                                    out_host[((oc * c_in + ic) * kh + khi) * kw
                                                        + kwi] += g * in_val;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Copy result back to device
        let mut out = stream
            .alloc_zeros::<f32>(out_len)
            .context("Failed to allocate CUDA memory for conv host fallback output")?;
        stream
            .memcpy_htod(&out_host, &mut out)
            .context("Failed to copy conv output to device")?;
        Ok(out)
    }

    fn maxpool_host_fallback<G>(
        &mut self,
        graph: &TensorGraph<f32, G>,
        node_idx: petgraph::graph::NodeIndex,
        kernel_size: usize,
        stride: usize,
        kind: PoolKind,
    ) -> Result<CudaSlice<f32>> {
        let stream = self.device.default_stream();
        let inputs_idx = graph.inputs(node_idx);

        // Copy inputs from device to host
        let mut host_inputs: Vec<Vec<f32>> = Vec::new();
        for &idx in &inputs_idx {
            let dev = self
                .values
                .get(&idx)
                .with_context(|| format!("Missing input {:?} for maxpool host fallback", idx))?;
            let mut host = vec![0.0f32; dev.len()];
            stream
                .memcpy_dtoh(dev, &mut host)
                .context("Failed to copy maxpool input from device")?;
            host_inputs.push(host);
        }

        let out_shape = graph.graph[node_idx].shape();
        let out_len: usize = out_shape.iter().product();
        let mut out_host = vec![0.0f32; out_len];

        match kind {
            PoolKind::Forward => {
                let x = &host_inputs[0];
                let in_shape = graph.graph[inputs_idx[0]].shape();
                let n = in_shape[0];
                let c = in_shape[1];
                let h = in_shape[2];
                let w = in_shape[3];
                let h_out = out_shape[2];
                let w_out = out_shape[3];

                for on in 0..n {
                    for oc in 0..c {
                        for ohi in 0..h_out {
                            for owi in 0..w_out {
                                let mut max_val = f32::NEG_INFINITY;
                                for khi in 0..kernel_size {
                                    let ih = ohi * stride + khi;
                                    if ih < h {
                                        for kwi in 0..kernel_size {
                                            let iw = owi * stride + kwi;
                                            if iw < w {
                                                let val = x[((on * c + oc) * h + ih) * w + iw];
                                                if val > max_val {
                                                    max_val = val;
                                                }
                                            }
                                        }
                                    }
                                }
                                out_host[((on * c + oc) * h_out + ohi) * w_out + owi] = max_val;
                            }
                        }
                    }
                }
            }
            PoolKind::Backward => {
                let input = &host_inputs[0];
                let output = &host_inputs[1];
                let grad = &host_inputs[2];
                let in_shape = graph.graph[inputs_idx[0]].shape();
                let pool_out_shape = graph.graph[inputs_idx[1]].shape();
                let n = in_shape[0];
                let c = in_shape[1];
                let h = in_shape[2];
                let w = in_shape[3];
                let h_out = pool_out_shape[2];
                let w_out = pool_out_shape[3];

                for on in 0..n {
                    for oc in 0..c {
                        for ohi in 0..h_out {
                            for owi in 0..w_out {
                                let o_val = output[((on * c + oc) * h_out + ohi) * w_out + owi];
                                let g = grad[((on * c + oc) * h_out + ohi) * w_out + owi];
                                for khi in 0..kernel_size {
                                    let ih = ohi * stride + khi;
                                    if ih < h {
                                        for kwi in 0..kernel_size {
                                            let iw = owi * stride + kwi;
                                            if iw < w {
                                                let in_val =
                                                    input[((on * c + oc) * h + ih) * w + iw];
                                                if in_val == o_val {
                                                    out_host[((on * c + oc) * h + ih) * w + iw] +=
                                                        g;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Copy result back to device
        let mut out = stream
            .alloc_zeros::<f32>(out_len)
            .context("Failed to allocate CUDA memory for maxpool host fallback output")?;
        stream
            .memcpy_htod(&out_host, &mut out)
            .context("Failed to copy maxpool output to device")?;
        Ok(out)
    }
}

impl Executor<f32> for CudaExecutor {
    fn execute<G>(
        &mut self,
        graph: &TensorGraph<f32, G>,
        inputs: HashMap<String, Vec<f32>>,
    ) -> Result<Vec<f32>> {
        let order = graph.toposort();
        let liveness = liveness::analyze(graph, &order);
        for (_, value) in self.values.drain() {
            self.pool.get_mut().give(value);
        }
        let mut live_bytes = 0usize;
        for (pos, node_idx) in order.iter().enumerate() {
            let node = &graph[*node_idx];
            let result = match node {
                TensorGraphNode::Constant { data, .. } => {
                    let _span = trace_span!("constant", node = node_idx.index(), size = data.len())
                        .entered();
                    let stream = self.device.default_stream();
                    let mut device_data = self.take_buffer(data.len());
                    if !data.is_empty() {
                        stream
                            .memcpy_htod(data.as_slice(), &mut device_data)
                            .context("Failed to copy constant to CUDA device")?;
                    }
                    device_data
                }
                TensorGraphNode::Input { name, .. } => {
                    let _span =
                        trace_span!("input", node = node_idx.index(), name = name).entered();
                    let val = inputs
                        .get::<str>(name)
                        .with_context(|| format!("Input '{}' not found", name))?;
                    let stream = self.device.default_stream();
                    let mut device_data = self.take_buffer(val.len());
                    if !val.is_empty() {
                        stream
                            .memcpy_htod(val.as_slice(), &mut device_data)
                            .context("Failed to copy input to CUDA device")?;
                    }
                    device_data
                }
                TensorGraphNode::Unary { op, .. } => {
                    let inputs = graph.inputs(*node_idx);
                    let input = self
                        .values
                        .get(&inputs[0])
                        .context("Missing input value for unary operation")?;
                    match op {
                        tensor::UnaryOp::Neg => self.neg(input),
                        tensor::UnaryOp::Exp => self.exp(input),
                        tensor::UnaryOp::Log => self.log(input),
                        tensor::UnaryOp::Relu => self.relu(input),
                    }
                }
                #[cfg(feature = "fusion")]
                TensorGraphNode::FusedUnary { ops, .. } => {
                    let inputs = graph.inputs(*node_idx);
                    let input = self
                        .values
                        .get(&inputs[0])
                        .context("Missing input value for fused unary operation")?;
                    let mut result = None;
                    for op in ops {
                        let source = result.as_ref().unwrap_or(input);
                        let next = match op {
                            tensor::UnaryOp::Neg => self.neg(source),
                            tensor::UnaryOp::Exp => self.exp(source),
                            tensor::UnaryOp::Log => self.log(source),
                            tensor::UnaryOp::Relu => self.relu(source),
                        };
                        if let Some(previous) = result.replace(next) {
                            self.give_buffer(previous);
                        }
                    }
                    result.context("Fused unary operation has no operators")?
                }
                TensorGraphNode::Binary { op, .. } => {
                    let ins = graph.inputs(*node_idx);
                    let lhs = self
                        .values
                        .get(&ins[0])
                        .context("Missing left operand for binary operation")?;
                    let rhs = self
                        .values
                        .get(&ins[1])
                        .context("Missing right operand for binary operation")?;
                    match op {
                        tensor::BinaryOp::Add => self.add(lhs, rhs),
                        tensor::BinaryOp::Sub => self.sub(lhs, rhs),
                        tensor::BinaryOp::Mul => self.mul(lhs, rhs),
                        tensor::BinaryOp::Div => self.div(lhs, rhs),
                    }
                }
                TensorGraphNode::MatMul { .. } => {
                    let ins = graph.inputs(*node_idx);
                    let lhs = self
                        .values
                        .get(&ins[0])
                        .context("Missing left operand for matmul")?;
                    let rhs = self
                        .values
                        .get(&ins[1])
                        .context("Missing right operand for matmul")?;
                    let lhs_shape = graph.graph[ins[0]].shape();
                    let rhs_shape = graph.graph[ins[1]].shape();
                    let output_shape = graph.graph[*node_idx].shape();
                    self.matmul(lhs, rhs, lhs_shape, rhs_shape, output_shape)
                        .with_context(|| {
                            format!("Failed to execute CUDA MatMul at node {}", node_idx.index())
                        })?
                }
                TensorGraphNode::Parameter { data, .. } => {
                    let v = data
                        .lock()
                        .map_err(|e| anyhow!("Failed to lock parameter data: {}", e))?;
                    let _span =
                        trace_span!("parameter", node = node_idx.index(), size = v.len()).entered();
                    let stream = self.device.default_stream();
                    let mut device_data = self.take_buffer(v.len());
                    if !v.is_empty() {
                        stream
                            .memcpy_htod(v.as_slice(), &mut device_data)
                            .context("Failed to copy parameter to CUDA device")?;
                    }
                    device_data
                }
                TensorGraphNode::BroadcastAxis { axis, .. } => {
                    let stream = self.device.default_stream();
                    let in_idx = graph.inputs(*node_idx)[0];
                    let in_val = self
                        .values
                        .get(&in_idx)
                        .context("Missing input value for broadcast")?;
                    let in_shape = graph.graph[in_idx].shape();
                    let out_shape = graph.graph[*node_idx].shape();
                    let out_size: usize = out_shape.iter().product();

                    anyhow::ensure!(
                        in_shape.len() == out_shape.len(),
                        "BroadcastAxis requires matching rank, got input rank {} and output rank {}",
                        in_shape.len(),
                        out_shape.len()
                    );
                    anyhow::ensure!(
                        in_shape[*axis] == 1,
                        "BroadcastAxis requires input axis {} size to be 1, got {}",
                        axis,
                        in_shape[*axis]
                    );

                    // Compute output strides for row-major layout
                    let mut out_strides = vec![1usize; out_shape.len()];
                    for i in (0..out_shape.len().saturating_sub(1)).rev() {
                        out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
                    }

                    // Compute input strides (size 1 dimensions have stride 0)
                    let mut in_strides = vec![1usize; in_shape.len()];
                    for i in (0..in_shape.len().saturating_sub(1)).rev() {
                        in_strides[i] = if in_shape[i + 1] == 1 {
                            in_strides[i + 1]
                        } else {
                            in_strides[i + 1] * in_shape[i + 1]
                        };
                    }
                    // Set stride to 0 for broadcast dimensions
                    for i in 0..in_shape.len() {
                        if in_shape[i] == 1 {
                            in_strides[i] = 0;
                        }
                    }

                    let mut out = self.take_buffer(out_size);

                    // Upload shape and strides to device
                    let mut out_shape_device = stream
                        .alloc_zeros::<usize>(out_shape.len())
                        .context("Failed to allocate CUDA memory for output shape")?;
                    stream
                        .memcpy_htod(out_shape, &mut out_shape_device)
                        .context("Failed to copy output shape to device")?;

                    let mut out_strides_device = stream
                        .alloc_zeros::<usize>(out_strides.len())
                        .context("Failed to allocate CUDA memory for output strides")?;
                    stream
                        .memcpy_htod(&out_strides, &mut out_strides_device)
                        .context("Failed to copy output strides to device")?;

                    let mut in_strides_device = stream
                        .alloc_zeros::<usize>(in_strides.len())
                        .context("Failed to allocate CUDA memory for input strides")?;
                    stream
                        .memcpy_htod(&in_strides, &mut in_strides_device)
                        .context("Failed to copy input strides to device")?;

                    let f = self
                        .module
                        .load_function("broadcast_axis")
                        .context("Failed to load broadcast_axis kernel")?;
                    let cfg = LaunchConfig::for_num_elems(out_size as u32);
                    let axis_u64 = *axis as u64;
                    let out_size_u64 = out_size as u64;
                    let in_len_u64 = in_val.len() as u64;
                    let shape_len_u64 = out_shape.len() as u64;
                    let mut launcher = stream.launch_builder(&f);
                    launcher.arg(in_val);
                    launcher.arg(&in_len_u64);
                    launcher.arg(&mut out);
                    launcher.arg(&axis_u64);
                    launcher.arg(&out_shape_device);
                    launcher.arg(&shape_len_u64);
                    launcher.arg(&out_strides_device);
                    launcher.arg(&shape_len_u64);
                    launcher.arg(&in_strides_device);
                    launcher.arg(&shape_len_u64);
                    launcher.arg(&out_size_u64);
                    unsafe { launcher.launch(cfg) }
                        .context("CUDA broadcast_axis kernel launch failed")?;

                    out
                }
                TensorGraphNode::ReduceAxis { op, axis, .. } => {
                    let in_idx = graph.inputs(*node_idx)[0];
                    let x_device = self
                        .values
                        .get(&in_idx)
                        .context("Missing input value for reduce")?;
                    let in_shape = graph.graph[in_idx].shape();
                    let out_shape = graph.graph[*node_idx].shape();
                    let out_len: usize = out_shape.iter().product();

                    anyhow::ensure!(
                        in_shape.len() == out_shape.len(),
                        "ReduceAxis preserves rank: input has {}D but output has {}D",
                        in_shape.len(),
                        out_shape.len()
                    );
                    anyhow::ensure!(
                        out_shape[*axis] == 1,
                        "ReduceAxis output axis {} must be 1, got {}",
                        axis,
                        out_shape[*axis]
                    );

                    let stream = self.device.default_stream();

                    // Compute strides for row-major layout
                    let mut in_strides = vec![1usize; in_shape.len()];
                    for i in (0..in_shape.len().saturating_sub(1)).rev() {
                        in_strides[i] = in_strides[i + 1] * in_shape[i + 1];
                    }

                    let mut out_strides = vec![1usize; out_shape.len()];
                    for i in (0..out_shape.len().saturating_sub(1)).rev() {
                        out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
                    }

                    let mut out_device = self.take_buffer(out_len);

                    // Upload shapes and strides to device
                    let mut in_shape_device = stream
                        .alloc_zeros::<usize>(in_shape.len())
                        .context("Failed to allocate CUDA memory for input shape")?;
                    stream
                        .memcpy_htod(in_shape, &mut in_shape_device)
                        .context("Failed to copy input shape to device")?;

                    let mut in_strides_device = stream
                        .alloc_zeros::<usize>(in_strides.len())
                        .context("Failed to allocate CUDA memory for input strides")?;
                    stream
                        .memcpy_htod(&in_strides, &mut in_strides_device)
                        .context("Failed to copy input strides to device")?;

                    let mut out_strides_device = stream
                        .alloc_zeros::<usize>(out_strides.len())
                        .context("Failed to allocate CUDA memory for output strides")?;
                    stream
                        .memcpy_htod(&out_strides, &mut out_strides_device)
                        .context("Failed to copy output strides to device")?;

                    let kernel_name = match op {
                        tensor::ReduceOp::Sum => "reduce_sum_axis",
                        tensor::ReduceOp::Max => "reduce_max_axis",
                        tensor::ReduceOp::Mean => "reduce_mean_axis",
                    };
                    let f = self
                        .module
                        .load_function(kernel_name)
                        .with_context(|| format!("Failed to load {} kernel", kernel_name))?;
                    let cfg = LaunchConfig::for_num_elems(out_len as u32);
                    let axis_u64 = *axis as u64;
                    let in_len_u64 = x_device.len() as u64;
                    let in_shape_len_u64 = in_shape.len() as u64;
                    let in_strides_len_u64 = in_strides.len() as u64;
                    let out_strides_len_u64 = out_strides.len() as u64;
                    let out_len_u64 = out_len as u64;
                    let axis_len_u64 = in_shape[*axis] as u64;
                    let mut launcher = stream.launch_builder(&f);
                    launcher.arg(x_device);
                    launcher.arg(&in_len_u64);
                    launcher.arg(&mut out_device);
                    launcher.arg(&axis_u64);
                    launcher.arg(&in_shape_device);
                    launcher.arg(&in_shape_len_u64);
                    launcher.arg(&in_strides_device);
                    launcher.arg(&in_strides_len_u64);
                    launcher.arg(&out_strides_device);
                    launcher.arg(&out_strides_len_u64);
                    launcher.arg(&out_len_u64);
                    launcher.arg(&axis_len_u64);
                    unsafe { launcher.launch(cfg) }
                        .with_context(|| format!("CUDA {} kernel launch failed", kernel_name))?;

                    out_device
                }
                TensorGraphNode::Transpose { .. } => {
                    let in_idx = graph.inputs(*node_idx)[0];
                    let input = self
                        .values
                        .get(&in_idx)
                        .context("Missing input value for transpose")?;
                    let in_shape = graph.graph[in_idx].shape();
                    anyhow::ensure!(
                        in_shape.len() >= 2,
                        "Transpose requires rank >= 2, got {}D",
                        in_shape.len()
                    );
                    if in_shape.len() == 2 {
                        self.transpose(input, in_shape[0], in_shape[1])
                            .with_context(|| {
                                format!(
                                    "Failed to execute CUDA Transpose at node {}",
                                    node_idx.index()
                                )
                            })?
                    } else {
                        let mut axes: Vec<usize> = (0..in_shape.len()).collect();
                        let rank = axes.len();
                        axes.swap(rank - 2, rank - 1);
                        let output_shape = graph.graph[*node_idx].shape();
                        self.permute(input, in_shape, output_shape, &axes)
                            .with_context(|| {
                                format!(
                                    "Failed to execute CUDA Transpose at node {}",
                                    node_idx.index()
                                )
                            })?
                    }
                }
                TensorGraphNode::Reshape { .. } => {
                    let in_idx = graph.inputs(*node_idx)[0];
                    let input = self
                        .values
                        .get(&in_idx)
                        .context("Missing input value for reshape")?;
                    let input_shape = graph.graph[in_idx].shape();
                    let output_shape = graph.graph[*node_idx].shape();
                    let input_len = checked_product(input_shape, "Reshape input")?;
                    let output_len = checked_product(output_shape, "Reshape output")?;
                    anyhow::ensure!(
                        input_len == input.len(),
                        "Reshape input shape {input_shape:?} has {input_len} elements but buffer has {}",
                        input.len()
                    );
                    anyhow::ensure!(
                        output_len == input_len,
                        "Reshape changes element count from {input_len} to {output_len}"
                    );
                    let stream = self.device.default_stream();
                    let mut out = self.take_buffer(output_len);
                    if output_len != 0 {
                        stream
                            .memcpy_dtod(input, &mut out)
                            .context("Failed to copy reshape data device-to-device")?;
                    }
                    out
                }
                TensorGraphNode::Permute { axes, .. } => {
                    let in_idx = graph.inputs(*node_idx)[0];
                    let input = self
                        .values
                        .get(&in_idx)
                        .context("Missing input value for permute")?;
                    let input_shape = graph.graph[in_idx].shape();
                    let output_shape = graph.graph[*node_idx].shape();
                    self.permute(input, input_shape, output_shape, axes)
                        .with_context(|| {
                            format!(
                                "Failed to execute CUDA Permute at node {}",
                                node_idx.index()
                            )
                        })?
                }
                TensorGraphNode::Gt { .. } => {
                    let ins = graph.inputs(*node_idx);
                    let lhs = self
                        .values
                        .get(&ins[0])
                        .context("Missing left operand for Gt")?;
                    let rhs = self
                        .values
                        .get(&ins[1])
                        .context("Missing right operand for Gt")?;
                    self.gt(lhs, rhs)
                }
                TensorGraphNode::Mask { .. } => {
                    let ins = graph.inputs(*node_idx);
                    let values = self
                        .values
                        .get(&ins[0])
                        .context("Missing values for Mask")?;
                    let condition = self
                        .values
                        .get(&ins[1])
                        .context("Missing condition for Mask")?;
                    self.mask(values, condition)
                }
                TensorGraphNode::Conv2d {
                    stride, padding, ..
                } => {
                    self.conv_host_fallback(graph, *node_idx, *stride, *padding, ConvKind::Forward)?
                }
                TensorGraphNode::ConvTranspose2d {
                    stride, padding, ..
                } => self.conv_host_fallback(
                    graph,
                    *node_idx,
                    *stride,
                    *padding,
                    ConvKind::Transpose,
                )?,
                TensorGraphNode::Conv2dBackwardWeight {
                    stride, padding, ..
                } => self.conv_host_fallback(
                    graph,
                    *node_idx,
                    *stride,
                    *padding,
                    ConvKind::BackwardWeight,
                )?,
                TensorGraphNode::MaxPool2d {
                    kernel_size,
                    stride,
                    ..
                } => self.maxpool_host_fallback(
                    graph,
                    *node_idx,
                    *kernel_size,
                    *stride,
                    PoolKind::Forward,
                )?,
                TensorGraphNode::MaxPool2dBackward {
                    kernel_size,
                    stride,
                    ..
                } => self.maxpool_host_fallback(
                    graph,
                    *node_idx,
                    *kernel_size,
                    *stride,
                    PoolKind::Backward,
                )?,
                TensorGraphNode::Flatten { .. } => {
                    let in_idx = graph.inputs(*node_idx)[0];
                    let input = self
                        .values
                        .get(&in_idx)
                        .context("Missing input value for flatten")?;
                    let stream = self.device.default_stream();
                    let len = input.len();
                    let mut out = stream
                        .alloc_zeros::<f32>(len)
                        .context("Failed to allocate CUDA memory for flatten output")?;
                    stream
                        .memcpy_dtod(input, &mut out)
                        .context("Failed to copy flatten data device-to-device")?;
                    out
                }
            };
            let bytes = result.len() * std::mem::size_of::<f32>();
            self.values.insert(*node_idx, result);
            live_bytes += bytes;
            self.stats.record_live(live_bytes);

            for &dead in liveness.free_after(pos) {
                if let Some(value) = self.values.remove(&dead) {
                    live_bytes -= value.len() * std::mem::size_of::<f32>();
                    self.pool.get_mut().give(value);
                }
            }
        }
        self.stats = self.stats.with_pool_stats(self.pool.get_mut().stats());
        tracing::debug!(
            bytes_allocated = self.stats.bytes_allocated,
            buffers_allocated = self.stats.buffers_allocated,
            peak_live_bytes = self.stats.peak_live_bytes,
            pool_hits = self.stats.pool_hits,
            pool_misses = self.stats.pool_misses,
            "execute memory stats"
        );

        let last_node_idx = order.last().context("Graph is empty")?;
        let out_device = self
            .values
            .get(last_node_idx)
            .context("Output value not found after execution")?;
        let mut out_host = vec![0.0f32; out_device.len()];
        self.device
            .default_stream()
            .memcpy_dtoh(out_device, &mut out_host)
            .context("Failed to copy output from CUDA device to host")?;
        Ok(out_host)
    }

    fn get_gradients(&self, graph: &TensorGraph<f32, WithGrad>) -> HashMap<usize, Vec<f32>> {
        let metadata = graph.gradient_metadata();

        let mut grads = HashMap::new();
        for (param_id, grad_node) in &metadata.param_to_grad {
            let grad_value_gpu = self
                .values
                .get(grad_node)
                .expect("Gradient value not computed");

            // Copy gradient from GPU to host
            let mut grad_host = vec![0.0f32; grad_value_gpu.len()];
            self.device
                .default_stream()
                .memcpy_dtoh(grad_value_gpu, &mut grad_host)
                .expect("Failed to copy gradient from CUDA device to host");

            grads.insert(*param_id, grad_host);
        }
        grads
    }
}
