//! Private ordered CUDA storage for incremental inference. No K/V readbacks.
use super::{
    DecodeGraph, GptAttentionPrecision, GptDecoderOptions, GptPrefillMode,
    attention::{AttentionScan, MergeScan},
    prefill, stages,
};
use crate::graph::{NodeIndex, TensorGraph, TensorGraphNode};
use crate::ptx::{
    F32, PtxGraph, PtxTarget,
    linear::{
        Binding, Buffer,
        kernels::convert,
        program::{Kernel, Program},
    },
};
use crate::tensor::Parameter;
use crate::tile::{
    DType, Dim, Expr, MatMulPrecision, RegionInput, RegionValue, TileGraph, TileIRBuilder,
    VirtualTensor, region::input_element_index,
};
use anyhow::{Context, Result, ensure};
use cudarc::{
    driver::{CudaContext, CudaSlice, CudaStream, DevicePtr},
    nvrtc::Ptx,
};
use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
};

const PARTITIONS: usize = 16;
const SPLIT_MIN_KEYS: usize = 128;

struct Compiled {
    _programs: Vec<Program<F32>>,
    kernels: Vec<Kernel>,
    split: HashMap<usize, (Kernel, Kernel)>,
    _storage: Vec<Buffer>,
    length: CudaSlice<u32>,
    updates: Vec<Kernel>,
    read: Kernel,
    output: CudaSlice<f32>,
    feeds: HashMap<&'static str, CudaSlice<f32>>,
    _constants: Vec<CudaSlice<f32>>,
}

pub(super) struct CudaDecoder {
    stream: Arc<CudaStream>,
    weights: HashMap<usize, CudaSlice<f32>>,
    cache: HashMap<usize, Buffer>,
    options: GptDecoderOptions,
    revisions: HashMap<usize, u64>,
    compiled: BTreeMap<usize, Compiled>,
}

impl Drop for CudaDecoder {
    fn drop(&mut self) {
        // Programs and every raw kernel binding outlive queued work.
        let _ = self.stream.synchronize();
        self.compiled.clear();
    }
}

fn binding(buffer: &CudaSlice<f32>, stream: &Arc<CudaStream>) -> Binding {
    Binding {
        pointer: buffer.device_ptr(stream).0,
        len: buffer.len(),
        dtype: DType::F32,
    }
}

impl CudaDecoder {
    pub(super) fn new(
        cache: &[Parameter<f32>],
        parameters: &[Parameter<f32>],
        options: GptDecoderOptions,
    ) -> Result<Self> {
        let context = CudaContext::new(0).context("initialize CUDA decoder")?;
        ensure!(
            options.attention_precision != GptAttentionPrecision::Bf16
                || PtxTarget::from_context(&context)?
                    .matmul_capabilities()
                    .bf16,
            "BF16 decoder attention requires CUDA compute capability 8.0 or newer"
        );
        let mut owner = Self {
            stream: context.new_stream()?,
            weights: HashMap::new(),
            cache: HashMap::new(),
            options,
            revisions: HashMap::new(),
            compiled: BTreeMap::new(),
        };
        for p in cache {
            owner
                .cache
                .insert(p.id(), owner.allocate(p.shape().iter().product())?);
        }
        owner.refresh(parameters)?;
        Ok(owner)
    }

    fn dtype(&self) -> DType {
        match self.options.attention_precision {
            GptAttentionPrecision::F32 => DType::F32,
            GptAttentionPrecision::Bf16 => DType::BF16,
        }
    }

    fn scan_partitions(&self, chunk: usize) -> usize {
        if self.dtype() == DType::BF16 && chunk <= 4 {
            32
        } else {
            PARTITIONS
        }
    }

    fn allocate(&self, len: usize) -> Result<Buffer> {
        Ok(match self.dtype() {
            DType::BF16 => Buffer::BF16(self.stream.alloc_zeros(len)?),
            _ => Buffer::F32(self.stream.alloc_zeros(len)?),
        })
    }

    fn refresh(&mut self, parameters: &[Parameter<f32>]) -> Result<()> {
        for p in parameters {
            let revision = p.revision()?;
            if self.revisions.get(&p.id()) == Some(&revision) {
                continue;
            }
            let values = p.values()?;
            if let Some(buffer) = self.weights.get_mut(&p.id()) {
                self.stream.memcpy_htod(&values, buffer)?;
            } else {
                self.weights
                    .insert(p.id(), self.stream.memcpy_stod(&values)?);
            }
            self.revisions.insert(p.id(), revision);
        }
        Ok(())
    }

    pub(super) fn reset(&mut self, cache: &[Parameter<f32>]) -> Result<()> {
        for p in cache {
            match self
                .cache
                .get_mut(&p.id())
                .context("decoder K/V allocation missing")?
            {
                Buffer::F32(buffer) => self.stream.memset_zeros(buffer)?,
                Buffer::BF16(buffer) => self.stream.memset_zeros(buffer)?,
            }
        }
        self.stream.synchronize()?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode(
        &mut self,
        graph: &DecodeGraph,
        feeds: HashMap<String, Vec<f32>>,
        parameters: &[Parameter<f32>],
        cache: &[Parameter<f32>],
        position: usize,
        chunk: usize,
        batch: usize,
        capacity: usize,
        width: usize,
    ) -> Result<Vec<f32>> {
        self.refresh(parameters)?;
        if !self.compiled.contains_key(&chunk) {
            if self.compiled.len() == 8 {
                self.stream.synchronize()?;
                self.compiled.clear();
            }
            let compiled = self.compile(graph, cache, chunk, batch, capacity, width)?;
            self.compiled.insert(chunk, compiled);
        }
        let element_bytes = self.dtype().size_bytes();
        let partitions = self.scan_partitions(chunk);
        let compiled = self
            .compiled
            .get_mut(&chunk)
            .context("decoder program missing")?;
        for (name, buffer) in &mut compiled.feeds {
            self.stream
                .memcpy_htod(feeds.get(*name).context("decoder feed missing")?, buffer)?;
        }
        let split_context = position + chunk >= SPLIT_MIN_KEYS;
        let lengths = scan_lengths(position, chunk, partitions)?;
        self.stream.memcpy_htod(&lengths, &mut compiled.length)?;
        for (id, kernel) in compiled.kernels.iter().enumerate() {
            if let Some((partial, merge)) = compiled.split.get(&id).filter(|_| split_context) {
                partial.run(&self.stream)?;
                merge.run(&self.stream)?;
            } else {
                kernel.run(&self.stream)?;
            }
        }
        let offset = u64::try_from(
            position
                .checked_mul(width)
                .and_then(|n| n.checked_mul(element_bytes))
                .context("decoder cache byte offset overflow")?,
        )?;
        for kernel in &compiled.updates {
            let mut update = kernel.clone();
            update.pointers[1] = update.pointers[1]
                .checked_add(offset)
                .context("decoder cache address overflow")?;
            update.run(&self.stream)?;
        }
        compiled.read.run(&self.stream)?;
        let output = self.stream.memcpy_dtov(&compiled.output)?;
        self.stream.synchronize()?;
        Ok(output)
    }

    fn compile(
        &self,
        graph: &DecodeGraph,
        cache: &[Parameter<f32>],
        chunk: usize,
        batch: usize,
        capacity: usize,
        width: usize,
    ) -> Result<Compiled> {
        let mut inputs = HashMap::new();
        let mut feeds = HashMap::new();
        let mut constants = Vec::new();
        for node in graph.graph.graph.node_indices() {
            let value = match &graph.graph[node] {
                TensorGraphNode::Parameter { id, .. } => {
                    if let Some(buffer) = self.cache.get(id) {
                        crate::ptx::linear::program::binding(buffer, &self.stream)
                    } else {
                        binding(
                            self.weights
                                .get(id)
                                .context("decoder parameter binding missing")?,
                            &self.stream,
                        )
                    }
                }
                TensorGraphNode::Input { name: "mask", .. } => continue,
                TensorGraphNode::Input { name, shape } => {
                    feeds.insert(*name, self.stream.alloc_zeros(shape.iter().product())?);
                    binding(&feeds[name], &self.stream)
                }
                _ => continue,
            };
            inputs.insert(node, value);
        }
        let partitions = self.scan_partitions(chunk);
        let length = self.stream.alloc_zeros::<u32>(2 + chunk * partitions)?;
        let mut available = inputs;
        let mut programs = Vec::new();
        let mut kernels = Vec::new();
        let mut split = HashMap::new();
        let mut storage = Vec::new();
        let mut updates = Vec::new();
        for (layer, attention) in graph.attention.iter().enumerate() {
            let outputs = [
                attention.hidden,
                attention.query,
                attention.key,
                attention.value,
            ];
            let mut stage = stages::extract(&graph.graph, &outputs, &available)?;
            let program = self.stage_program(&mut stage, &mut constants)?;
            kernels.extend(program.kernels.iter().cloned());
            for (&original, &output) in outputs.iter().zip(&stage.outputs) {
                let value = *program
                    .values
                    .get(&output)
                    .context("decoder stage output missing")?;
                let contiguous = program.views.get(&output).is_none_or(|view| {
                    view.predicate.is_none()
                        && view.access.is_identity()
                        && stage.graph[view.source].shape() == graph.graph[original].shape()
                });
                let target = if contiguous {
                    value
                } else {
                    let buffer = self.stream.alloc_zeros(batch * chunk * width)?;
                    let target = binding(&buffer, &self.stream);
                    kernels.push(self.copy(&program, &stage.graph, output, target, None)?);
                    storage.push(Buffer::F32(buffer));
                    target
                };
                available.insert(original, target);
            }
            for (slot, &output) in stage.outputs[2..].iter().enumerate() {
                updates.push(self.copy(
                    &program,
                    &stage.graph,
                    output,
                    crate::ptx::linear::program::binding(
                        &self.cache[&cache[2 * layer + slot].id()],
                        &self.stream,
                    ),
                    Some((chunk * width, capacity * width)),
                )?);
            }
            for node in [attention.query, attention.key, attention.value] {
                if self.dtype() == DType::BF16 {
                    let buffer = self.allocate(batch * chunk * width)?;
                    let target = crate::ptx::linear::program::binding(&buffer, &self.stream);
                    kernels.push(convert(&self.stream, available[&node], target, false)?);
                    available.insert(node, target);
                    storage.push(buffer);
                }
            }
            let buffer = self.allocate(batch * chunk * width)?;
            let target = crate::ptx::linear::program::binding(&buffer, &self.stream);
            let scan = AttentionScan {
                batch,
                chunk,
                width,
                capacity,
                heads: graph.graph[attention.output].shape()[1],
                partitions: 1,
                dtype: self.dtype(),
            };
            let columns = width / scan.heads;
            let rows = batch * scan.heads * chunk;
            let pointers = vec![
                available[&attention.query].pointer,
                available[&attention.key].pointer,
                available[&attention.value].pointer,
                crate::ptx::linear::program::binding(
                    &self.cache[&cache[2 * layer].id()],
                    &self.stream,
                )
                .pointer,
                crate::ptx::linear::program::binding(
                    &self.cache[&cache[2 * layer + 1].id()],
                    &self.stream,
                )
                .pointer,
                length.device_ptr(&self.stream).0,
                target.pointer,
            ];
            let tiled = self.dtype() == DType::BF16
                && self.options.prefill == GptPrefillMode::Auto
                && chunk >= 16
                && columns <= 128;
            if tiled {
                let partitions = prefill::partitions(chunk);
                let partial_count = (batch * chunk * width)
                    .checked_mul(partitions)
                    .context("decoder tiled partial workspace overflow")?;
                let partials = self.stream.alloc_zeros::<f32>(partial_count)?;
                let normalizers = self
                    .stream
                    .alloc_zeros::<f32>(prefill::STATISTICS_WIDTH * rows * partitions)?;
                let mut partial_pointers = pointers;
                partial_pointers[6] = partials.device_ptr(&self.stream).0;
                partial_pointers.push(normalizers.device_ptr(&self.stream).0);
                let scan = AttentionScan { partitions, ..scan };
                kernels.push(self.generated(
                    prefill::ir(&scan)?,
                    partial_pointers,
                    prefill::launch(&scan)?,
                )?);
                kernels.push(
                    self.generated(
                        MergeScan {
                            rows,
                            columns,
                            partitions,
                            dtype: self.dtype(),
                            raw_partials: true,
                        }
                        .ir()?,
                        vec![
                            partials.device_ptr(&self.stream).0,
                            normalizers.device_ptr(&self.stream).0,
                            target.pointer,
                        ],
                        scan_launch(rows, columns)?,
                    )?,
                );
                storage.push(Buffer::F32(partials));
                storage.push(Buffer::F32(normalizers));
            } else {
                let scan_id = kernels.len();
                kernels.push(self.generated(
                    scan.ir()?,
                    pointers.clone(),
                    scan_launch(rows, columns)?,
                )?);
                let partial_count = (batch * chunk * width)
                    .checked_mul(partitions)
                    .context("decoder partial workspace overflow")?;
                let partials = self.stream.alloc_zeros::<f32>(partial_count)?;
                let normalizers = self.stream.alloc_zeros::<f32>(rows * partitions)?;
                let mut partial_pointers = pointers;
                partial_pointers[6] = partials.device_ptr(&self.stream).0;
                partial_pointers.push(normalizers.device_ptr(&self.stream).0);
                let partial = self.generated(
                    AttentionScan { partitions, ..scan }.ir()?,
                    partial_pointers,
                    scan_launch(rows * partitions, columns)?,
                )?;
                let merge = self.generated(
                    MergeScan {
                        rows,
                        columns,
                        partitions,
                        dtype: self.dtype(),
                        raw_partials: false,
                    }
                    .ir()?,
                    vec![
                        partials.device_ptr(&self.stream).0,
                        normalizers.device_ptr(&self.stream).0,
                        target.pointer,
                    ],
                    scan_launch(rows, columns)?,
                )?;
                split.insert(scan_id, (partial, merge));
                storage.push(Buffer::F32(partials));
                storage.push(Buffer::F32(normalizers));
            }
            let promoted = if self.dtype() == DType::BF16 {
                let output = self.stream.alloc_zeros(batch * chunk * width)?;
                let promoted = binding(&output, &self.stream);
                kernels.push(convert(&self.stream, target, promoted, false)?);
                storage.push(Buffer::F32(output));
                promoted
            } else {
                target
            };
            available.insert(attention.output, promoted);
            storage.push(buffer);
            programs.push(program);
        }
        let mut stage = stages::extract(&graph.graph, &[graph.logits], &available)?;
        let program = self.stage_program(&mut stage, &mut constants)?;
        kernels.extend(program.kernels.iter().cloned());
        let output = self
            .stream
            .alloc_zeros(batch * chunk * graph.graph[graph.logits].shape()[2])?;
        let read = self.copy(
            &program,
            &stage.graph,
            stage.outputs[0],
            binding(&output, &self.stream),
            None,
        )?;
        programs.push(program);
        Ok(Compiled {
            _programs: programs,
            kernels,
            split,
            _storage: storage,
            length,
            updates,
            read,
            output,
            feeds,
            _constants: constants,
        })
    }

    fn stage_program(
        &self,
        stage: &mut stages::Stage,
        constants: &mut Vec<CudaSlice<f32>>,
    ) -> Result<Program<F32>> {
        for node in stage.graph.graph.node_indices() {
            if let TensorGraphNode::Constant { data, .. } = &stage.graph[node] {
                let buffer = self.stream.memcpy_stod(data.as_slice())?;
                stage.inputs.insert(node, binding(&buffer, &self.stream));
                constants.push(buffer);
            }
        }
        Program::new_bound(
            &self.stream,
            stage.graph.clone(),
            stage.inputs.clone(),
            None,
            MatMulPrecision::StrictF32,
        )
    }

    fn generated(
        &self,
        mut ir: crate::tile::TileIR,
        pointers: Vec<u64>,
        config: cudarc::driver::LaunchConfig,
    ) -> Result<Kernel> {
        ir.validate_layouts()?;
        let label = match ir.kernel_name.as_str() {
            "decoder_attention" => "decoder generated scan",
            "decoder_prefill_bf16" => "decoder tensor-core prefill",
            "decoder_merge" => "decoder generated merge",
            _ => "decoder generated copy",
        };
        ir.optimize_indices()?;
        let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
        tiles
            .region_kernels
            .push(crate::tile::graph::TileRegionKernel { region_id: 0, ir });
        let ptx =
            PtxGraph::from(tiles).with_target(PtxTarget::from_context(self.stream.context())?);
        let module = self
            .stream
            .context()
            .load_module(Ptx::from_src(ptx.to_ptx()))?;
        Ok(Kernel {
            label: label.into(),
            function: module.load_function(
                ptx.region_kernel_name(0)
                    .context("decoder kernel missing")?,
            )?,
            config,
            pointers,
            zero_before: None,
        })
    }

    fn copy(
        &self,
        program: &Program<F32>,
        graph: &TensorGraph<f32>,
        node: NodeIndex,
        target: Binding,
        strides: Option<(usize, usize)>,
    ) -> Result<Kernel> {
        let shape = graph[node].shape();
        let len = shape.iter().product::<usize>();
        let input = *program
            .values
            .get(&node)
            .context("decoder retained output missing")?;
        let view = program
            .views
            .get(&node)
            .cloned()
            .unwrap_or_else(|| VirtualTensor::identity(node, shape.clone(), DType::F32));
        let source_shape = graph[view.source].shape().clone();
        ensure!(
            source_shape.iter().product::<usize>() == input.len,
            "decoder source storage extent mismatch"
        );
        if let Some((source_stride, target_stride)) = strides {
            ensure!(
                source_stride > 0
                    && len.is_multiple_of(source_stride)
                    && target_stride >= source_stride
                    && target.len == len / source_stride * target_stride,
                "decoder cache copy extent mismatch"
            );
        }
        let mut ir = TileIRBuilder::new();
        ir.start_kernel("decoder_cache_copy");
        ir.add_param("input", DType::F32, true);
        ir.add_param("output", target.dtype, false);
        ir.bounds_check(len);
        let linear = Expr::Mul(
            Box::new(Expr::BlockIdx(Dim::X)),
            Box::new(Expr::BlockDim(Dim::X)),
        ) + Expr::ThreadIdx(Dim::X);
        let source_index = input_element_index(
            &RegionInput {
                node,
                value: RegionValue(0),
                tensor: view,
                source_shape,
            },
            shape,
            &linear,
        )?;
        let value = ir.alloc_register(DType::F32, 1, 1);
        ir.load_global_to_shared(value, "input", source_index, Expr::Const(0));
        let index = if let Some((source_stride, target_stride)) = strides {
            Expr::FloorDiv(Box::new(linear.clone()), source_stride) * target_stride
                + Expr::Mod(Box::new(linear), source_stride)
        } else {
            linear
        };
        ir.store("output", value, index, Expr::Const(0));
        ensure!(
            strides.is_some() || target.len == len,
            "decoder output extent mismatch"
        );
        self.generated(
            ir.finish(),
            vec![input.pointer, target.pointer],
            crate::ptx::target::pointwise_launch_config(u32::try_from(len)?),
        )
    }
}

fn scan_lengths(position: usize, chunk: usize, partitions: usize) -> Result<Vec<u32>> {
    let span = (position + chunk).div_ceil(partitions);
    let mut lengths = Vec::with_capacity(2 + chunk * partitions);
    lengths.push(u32::try_from(position)?);
    lengths.push(u32::try_from(span)?);
    for query in 0..chunk {
        for part in 0..partitions {
            lengths.push(u32::try_from(
                (position + query + 1).saturating_sub(part * span).min(span),
            )?);
        }
    }
    Ok(lengths)
}

fn scan_launch(rows: usize, columns: usize) -> Result<cudarc::driver::LaunchConfig> {
    Ok(if columns <= 128 {
        cudarc::driver::LaunchConfig {
            grid_dim: (u32::try_from(rows)?, 1, 1),
            block_dim: (u32::try_from(columns.max(32).next_power_of_two())?, 1, 1),
            shared_mem_bytes: 0,
        }
    } else {
        crate::ptx::target::pointwise_launch_config(u32::try_from(rows * columns)?)
    })
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod bf16_tests;
