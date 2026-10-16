//! Explicit physical BF16 linear forward/backward island.
//!
//! This is a bounded building block for mixed storage, not a model autocast
//! policy or a replacement for TensorGraph autograd. Each call takes f32
//! boundary values, rounds operands/upstream contributions to BF16, computes
//! linear and its analytical backward with BF16 outputs, and promotes results.
//! Master weights and accumulated parameter gradients stay physically f32.
mod kernels;
mod program;
#[cfg(test)]
mod tests;
use crate::{
    graph::TensorGraph,
    ptx::{BF16, CudaDType, F32},
    tensor::TensorExpr,
    tile::{DType, MatMulPrecision, TileDType},
};
use anyhow::{Context, Result, ensure};
use cudarc::driver::{CudaContext, CudaGraph, CudaSlice, CudaStream, sys};
use half::bf16;
use program::{Kernel, Program, binding};
use serde::{Deserialize, Serialize};
use std::{any::Any, collections::HashMap, sync::Arc};

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub enum LinearPrecision {
    Bf16,
    Tf32x3,
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct LinearShape {
    pub rows: usize,
    pub inputs: usize,
    pub outputs: usize,
}
impl LinearShape {
    fn lengths(self) -> Result<[usize; 4]> {
        ensure!(
            self.rows > 0 && self.inputs > 0 && self.outputs > 0,
            "linear dimensions must be positive"
        );
        let product = |a: usize, b: usize| -> Result<usize> {
            let n = a.checked_mul(b).context("linear extent overflow")?;
            ensure!(
                u32::try_from(n).is_ok(),
                "linear extent exceeds CUDA launch limit"
            );
            Ok(n)
        };
        Ok([
            product(self.rows, self.inputs)?,
            product(self.outputs, self.inputs)?,
            self.outputs,
            product(self.rows, self.outputs)?,
        ])
    }
}
#[derive(Deserialize, Serialize)]
pub struct LinearInputs {
    pub x: Vec<f32>,
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
    pub upstream: Vec<Vec<f32>>,
}
#[derive(Deserialize, Serialize, PartialEq)]
pub struct LinearResult {
    pub output: Vec<f32>,
    pub input_gradient: Vec<f32>,
    pub weight_gradient: Vec<f32>,
    pub bias_gradient: Vec<f32>,
    pub accumulated_weight_gradient: Vec<f32>,
    pub accumulated_bias_gradient: Vec<f32>,
}
#[derive(Serialize)]
pub struct LinearInfo {
    pub shape: LinearShape,
    pub precision: LinearPrecision,
    pub branches: usize,
    pub kernels: usize,
    pub f32_buffer_bytes: usize,
    pub bf16_buffer_bytes: usize,
    pub program_temporary_bytes: usize,
    pub split_f32_weight_gradient: bool,
    pub plans: Vec<String>,
}
#[derive(Clone, Copy)]
struct Binding {
    pointer: u64,
    len: usize,
    dtype: DType,
}
enum Buffer {
    F32(CudaSlice<f32>),
    BF16(CudaSlice<bf16>),
}
struct Captured {
    graph: Option<CudaGraph>,
    stream: Arc<CudaStream>,
}
impl Drop for Captured {
    fn drop(&mut self) {
        if let Some(graph) = self.graph.take() {
            if self.stream.context().bind_to_thread().is_ok() {
                drop(graph)
            } else {
                std::mem::forget(graph)
            }
        }
    }
}
/// A fixed-shape, private-stream owner. All uploads/readbacks are explicit and
/// outside replay. One or two downstream contributions can seed the backward;
/// each is rounded before their BF16 sum. This does not model distinct linear
/// uses of one shared master weight, which need a larger cast-aware graph.
pub struct LinearTraining {
    // Destroy captures before every referenced allocation/function.
    captured: Option<Captured>,
    stream: Arc<CudaStream>,
    buffers: Vec<Buffer>,
    _programs: Vec<Box<dyn Any>>,
    kernels: Vec<Kernel>,
    masters: Vec<usize>,
    outputs: [usize; 4],
    accumulated: [usize; 2],
    uploaded: bool,
    has_run: bool,
    info: LinearInfo,
    ptx: Vec<String>,
}
impl Drop for LinearTraining {
    fn drop(&mut self) {
        let _ = self.stream.synchronize();
    }
}
impl LinearTraining {
    pub fn new(
        shape: LinearShape,
        precision: LinearPrecision,
        branches: usize,
        replay: bool,
    ) -> Result<Self> {
        shape.lengths()?;
        ensure!(
            (1..=2).contains(&branches),
            "linear supports one or two upstream contributions"
        );
        let device = CudaContext::new(0)?;
        let stream = device.new_stream()?;
        let mut result = Self {
            captured: None,
            stream,
            buffers: Vec::new(),
            _programs: Vec::new(),
            kernels: Vec::new(),
            masters: Vec::new(),
            outputs: [0; 4],
            accumulated: [0; 2],
            uploaded: false,
            has_run: false,
            info: LinearInfo {
                shape,
                precision,
                branches,
                kernels: 0,
                f32_buffer_bytes: 0,
                bf16_buffer_bytes: 0,
                program_temporary_bytes: 0,
                split_f32_weight_gradient: false,
                plans: Vec::new(),
            },
            ptx: Vec::new(),
        };
        match precision {
            LinearPrecision::Bf16 => result.prepare::<BF16>()?,
            LinearPrecision::Tf32x3 => result.prepare::<F32>()?,
        };
        result.info.kernels = result.kernels.len();
        // Warm once before capture: no allocation, upload or readback is recorded.
        for kernel in &result.kernels {
            kernel.run(&result.stream)?;
        }
        result.stream.synchronize()?;
        if replay {
            result
                .stream
                .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
            let recorded = result
                .kernels
                .iter()
                .try_for_each(|k| k.run(&result.stream));
            let ended = result.stream.end_capture(
                sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            );
            recorded?;
            result.captured = Some(Captured {
                graph: Some(ended?.context("empty linear capture")?),
                stream: result.stream.clone(),
            });
        }
        result.clear_gradients()?;
        Ok(result)
    }
    fn alloc(&mut self, len: usize, dtype: DType) -> Result<usize> {
        let index = self.buffers.len();
        self.buffers.push(match dtype {
            DType::F32 => {
                self.info.f32_buffer_bytes += len * 4;
                Buffer::F32(self.stream.alloc_zeros(len)?)
            }
            DType::BF16 => {
                self.info.bf16_buffer_bytes += len * 2;
                Buffer::BF16(self.stream.alloc_zeros(len)?)
            }
            _ => anyhow::bail!("unsupported linear buffer dtype"),
        });
        Ok(index)
    }
    fn slot(&self, index: usize) -> Binding {
        binding(&self.buffers[index], &self.stream)
    }
    fn add_program<D: CudaDType>(
        &mut self,
        expression: TensorExpr<D::HostType>,
        inputs: HashMap<&str, Binding>,
        root: Option<Binding>,
        precision: MatMulPrecision,
    ) -> Result<Binding> {
        let graph: TensorGraph<D::HostType> = expression.into();
        let program = Program::<D>::new(&self.stream, graph, inputs, root, precision)?;
        let output = program.output;
        self.kernels.extend(program.kernels.clone());
        self.info.program_temporary_bytes += program.bytes();
        if D::HostType::TILE_DTYPE == DType::BF16 {
            self.info.bf16_buffer_bytes += program.bytes();
        } else {
            self.info.f32_buffer_bytes += program.bytes();
        }
        self.info.plans.push(program.plan.clone());
        self.ptx.push(program.ptx.clone());
        self._programs.push(Box::new(program));
        Ok(output)
    }
    fn prepare<D: CudaDType>(&mut self) -> Result<()> {
        let s = self.info.shape;
        let lengths = s.lengths()?;
        let dtype = D::HostType::TILE_DTYPE;
        let precision = if dtype == DType::BF16 {
            MatMulPrecision::AllowTf32
        } else {
            MatMulPrecision::Tf32x3
        };
        let mut operands = Vec::new();
        for len in lengths
            .into_iter()
            .chain(std::iter::repeat_n(lengths[3], self.info.branches - 1))
        {
            let master = self.alloc(len, DType::F32)?;
            self.masters.push(master);
            let operand = if dtype == DType::BF16 {
                let target = self.alloc(len, dtype)?;
                self.kernels.push(kernels::convert(
                    &self.stream,
                    self.slot(master),
                    self.slot(target),
                    false,
                )?);
                target
            } else {
                master
            };
            operands.push(self.slot(operand));
        }
        let [x, w, b, g0] = <[Binding; 4]>::try_from(&operands[..4])
            .map_err(|_| anyhow::anyhow!("missing linear operands"))?;
        let g = if self.info.branches == 2 {
            let a = TensorExpr::<D::HostType>::input("g0", vec![s.rows, s.outputs]);
            let b = TensorExpr::input("g1", vec![s.rows, s.outputs]);
            self.add_program::<D>(
                a + b,
                HashMap::from([("g0", g0), ("g1", operands[4])]),
                None,
                precision,
            )?
        } else {
            g0
        };
        for (i, len) in [lengths[3], lengths[0], lengths[1], lengths[2]]
            .into_iter()
            .enumerate()
        {
            self.outputs[i] = self.alloc(len, DType::F32)?;
        }
        self.accumulated = [
            self.alloc(lengths[1], DType::F32)?,
            self.alloc(lengths[2], DType::F32)?,
        ];
        let xexpr = TensorExpr::<D::HostType>::input("x", vec![s.rows, s.inputs]);
        let wexpr = TensorExpr::input("w", vec![s.outputs, s.inputs]);
        let bexpr = TensorExpr::input("b", vec![s.outputs]);
        let gexpr = TensorExpr::input("g", vec![s.rows, s.outputs]);
        let y = xexpr.clone().matmul(wexpr.clone().transpose()) + bexpr;
        let dx = gexpr.clone().matmul(wexpr);
        let split = dtype == DType::F32
            && s.inputs >= 128
            && s.outputs >= 128
            && s.rows >= 2048
            && s.rows / s.inputs.max(s.outputs) >= 4
            && s.rows.is_multiple_of(4);
        self.info.split_f32_weight_gradient = split;
        let dw = if split {
            gexpr
                .clone()
                .transpose()
                .reshape(vec![s.outputs, 4, s.rows / 4])
                .permute(vec![1, 0, 2])
                .matmul(xexpr.reshape(vec![4, s.rows / 4, s.inputs]))
                .reduce_sum(0)
        } else {
            gexpr.clone().transpose().matmul(xexpr)
        };
        let db = gexpr.reduce_sum(0);
        for (index, expression) in [y, dx, dw, db].into_iter().enumerate() {
            let target = self.slot(self.outputs[index]);
            let output = self.add_program::<D>(
                expression,
                HashMap::from([("x", x), ("w", w), ("b", b), ("g", g)]),
                if dtype == DType::F32 {
                    Some(target)
                } else {
                    None
                },
                precision,
            )?;
            if output.pointer != target.pointer {
                self.kernels
                    .push(kernels::convert(&self.stream, output, target, false)?);
            }
        }
        for (output, accumulated) in self.outputs[2..].iter().zip(&self.accumulated) {
            self.kernels.push(kernels::convert(
                &self.stream,
                self.slot(*output),
                self.slot(*accumulated),
                true,
            )?);
        }
        Ok(())
    }
    /// Replace f32 boundary values, including master weight/bias. Captured
    /// addresses remain fixed. All extents are checked before any mutation.
    pub fn upload(&mut self, inputs: &LinearInputs) -> Result<()> {
        ensure!(
            inputs.upstream.len() == self.info.branches,
            "linear upstream branch count mismatch"
        );
        let sources = std::iter::once(&inputs.x)
            .chain(std::iter::once(&inputs.weight))
            .chain(std::iter::once(&inputs.bias))
            .chain(inputs.upstream.iter())
            .collect::<Vec<_>>();
        for (&index, data) in self.masters.iter().zip(&sources) {
            ensure!(
                self.slot(index).len == data.len(),
                "linear upload extent mismatch"
            );
        }
        self.stream.synchronize()?;
        self.uploaded = false;
        self.has_run = false;
        for (&index, data) in self.masters.iter().zip(sources) {
            let Buffer::F32(buffer) = &mut self.buffers[index] else {
                anyhow::bail!("linear master must be f32")
            };
            self.stream.memcpy_htod(data, buffer)?;
        }
        self.uploaded = true;
        Ok(())
    }
    /// Enqueue the complete forward/backward, promotions and f32 parameter
    /// gradient accumulation. No host transfers or allocations occur here.
    pub fn run(&mut self) -> Result<()> {
        ensure!(self.uploaded, "linear inputs have not been uploaded");
        if let Some(capture) = &self.captured {
            capture
                .graph
                .as_ref()
                .context("linear capture absent")?
                .launch()?;
        } else {
            for kernel in &self.kernels {
                kernel.run(&self.stream)?;
            }
        }
        self.has_run = true;
        Ok(())
    }
    pub fn clear_gradients(&mut self) -> Result<()> {
        for index in self.accumulated {
            let Buffer::F32(buffer) = &mut self.buffers[index] else {
                anyhow::bail!("linear accumulated gradient must be f32")
            };
            self.stream.memset_zeros(buffer)?;
        }
        Ok(())
    }
    pub fn readback(&self) -> Result<LinearResult> {
        ensure!(self.has_run, "linear has no results for the current upload");
        let read = |index: usize| -> Result<Vec<f32>> {
            let Buffer::F32(buffer) = &self.buffers[index] else {
                anyhow::bail!("linear boundary must be f32")
            };
            Ok(self.stream.memcpy_dtov(buffer)?)
        };
        Ok(LinearResult {
            output: read(self.outputs[0])?,
            input_gradient: read(self.outputs[1])?,
            weight_gradient: read(self.outputs[2])?,
            bias_gradient: read(self.outputs[3])?,
            accumulated_weight_gradient: read(self.accumulated[0])?,
            accumulated_bias_gradient: read(self.accumulated[1])?,
        })
    }
    /// CUDA-event time for complete resident calls; includes conversions and
    /// accumulation, excludes upload/readback. Caller controls sample ordering.
    pub fn time_runs(&mut self, repeats: usize) -> Result<f64> {
        ensure!(repeats > 0, "linear timing needs positive repeats");
        let begin = self
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        for _ in 0..repeats {
            self.run()?;
        }
        let end = self
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        Ok(f64::from(begin.elapsed_ms(&end)?) * 1000. / repeats as f64)
    }
    pub fn info(&self) -> &LinearInfo {
        &self.info
    }
    pub fn module_sources(&self) -> &[String] {
        &self.ptx
    }
}
