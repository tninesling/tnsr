//! Default-off numerical experiment. Four F32 contiguous-K partials and the
//! original ordered left-four reducer precede exactly one BF16 output store.
//! Never invoked recursively; the existing long Split-K path has priority.
use super::*;
use crate::tile::{Dim, Expr, MatMulRegion, MatMulSchedule, SmallGridCandidate, ir::MatrixLayout};
use cudarc::driver::CudaFunction;

struct Prepared {
    partial: CudaFunction,
    reducer: CudaFunction,
    schedule: MatMulSchedule,
}

impl MixedExecutor {
    pub(super) fn small_grid_split_bf16_matmul(
        &mut self,
        node: NodeIndex,
        op: &TensorGraphNode<f32>,
        operands: &[Value],
        precision: MatMulPrecision,
    ) -> Result<Option<Value>> {
        if !self.small_grid_split_k_bf16
            || !matches!(op, TensorGraphNode::MatMul { .. })
            || operands.len() != 2
            || operands.iter().any(|v| v.view.shape.len() != 2)
        {
            return Ok(None);
        }
        let (m, k, n) = (
            operands[0].view.shape[0],
            operands[0].view.shape[1],
            operands[1].view.shape[1],
        );
        if !SmallGridCandidate::eligible_shape(m, n, k)
            || op.shape() != &[m, n]
            || operands[1].view.shape[0] != k
        {
            return Ok(None);
        }
        let prepared =
            if self.split_k_shared_memory_reuse && precision != MatMulPrecision::StrictF32 {
                prepare(&self.stream, node, operands, m, n, k, precision).ok()
            } else {
                None
            };
        let Some(prepared) = prepared else {
            self.info.small_grid_split_k_fallback_programs += 1;
            return Ok(None);
        };
        // Admission and both compilations finish before committing any buffers,
        // pointers, launches or counters. Allocation failure remains a Result
        // error; temporary owners drop without half-installing a candidate.
        let len = m.checked_mul(n).context("small split output overflow")?;
        let partial_len = len.checked_mul(4).context("small split partial overflow")?;
        let partial = Buffer::F32(self.stream.alloc_zeros::<f32>(partial_len)?);
        let output = Buffer::BF16(self.stream.alloc_zeros::<half::bf16>(len)?);
        let partial_binding = binding(&partial, &self.stream);
        let output_binding = binding(&output, &self.stream);
        let label = format!(
            "node={} small_grid_split_k=true [{m},{k}]x[{k},{n}]",
            node.index()
        );
        let kernels = [
            Kernel {
                owner: None,
                label: format!("{label} F32 partials4 square64-k32"),
                function: prepared.partial,
                config: LaunchConfig {
                    grid_dim: (u32::try_from(n / 64)?, u32::try_from(m / 64)?, 4),
                    block_dim: prepared.schedule.block_threads,
                    shared_mem_bytes: 0,
                },
                pointers: vec![
                    operands[0].binding.pointer,
                    operands[1].binding.pointer,
                    partial_binding.pointer,
                ],
                zero_before: None,
            },
            Kernel {
                owner: None,
                label: format!("{label} left4 F32 sum to BF16RN"),
                function: prepared.reducer,
                config: LaunchConfig::for_num_elems(u32::try_from(len)?),
                pointers: vec![partial_binding.pointer, output_binding.pointer],
                zero_before: None,
            },
        ];
        self.buffers.extend([partial, output]);
        self.kernels.extend(kernels);
        self.info.f32_bytes += partial_len * 4;
        self.info.bf16_bytes += len * 2;
        self.info.small_grid_split_k_programs += 1;
        self.info.bf16_tensor_core_programs += 1;
        Ok(Some(Value {
            binding: output_binding,
            view: VirtualTensor::identity(node, vec![m, n], DType::BF16),
            source_shape: vec![m, n],
            chain: Vec::new(),
        }))
    }
}

fn prepare(
    stream: &Arc<CudaStream>,
    node: NodeIndex,
    operands: &[Value],
    m: usize,
    n: usize,
    total_k: usize,
    precision: MatMulPrecision,
) -> Result<Prepared> {
    for operand in operands {
        let volume = operand
            .source_shape
            .iter()
            .try_fold(1_usize, |v, &d| v.checked_mul(d))
            .context("small split source extent overflow")?;
        ensure!(
            operand.binding.dtype == DType::BF16
                && operand.view.dtype == DType::BF16
                && operand.binding.pointer != 0
                && operand.binding.pointer.is_multiple_of(16)
                && operand.binding.len == volume,
            "small split requires full aligned BF16 bindings"
        );
    }
    let target = PtxTarget::from_context(stream.context())?;
    ensure!(
        target.matmul_capabilities().bf16 && target.supports_async_copy(),
        "small split requires native BF16 and async copies"
    );
    let part_k = total_k / 4;
    let mut graph = TensorGraph::<half::bf16>::new();
    let a = graph.graph.add_node(TensorGraphNode::Input {
        name: "a",
        shape: vec![4, m, part_k],
    });
    let b = graph.graph.add_node(TensorGraphNode::Input {
        name: "b",
        shape: vec![4, part_k, n],
    });
    let root = graph.graph.add_node(TensorGraphNode::MatMul {
        shape: vec![4, m, n],
    });
    graph.graph.add_edge(a, root, 0);
    graph.graph.add_edge(b, root, 1);
    let mut region = MatMulRegion::from_graph(&graph, root, vec![], vec![root])?;
    region.schedule = MatMulSchedule::select_with_resources(
        m,
        n,
        part_k,
        DType::BF16,
        target.matmul_capabilities(),
        precision,
        target.matmul_resources,
    );
    region.inputs[0].tensor = operands[0]
        .view
        .reshape(vec![m, 4, part_k])?
        .permute(&[1, 0, 2])?;
    region.inputs[0].source_shape = operands[0].source_shape.clone();
    region.inputs[1].tensor = operands[1].view.reshape(vec![4, part_k, n])?;
    region.inputs[1].source_shape = operands[1].source_shape.clone();
    let candidate = SmallGridCandidate::candidate(&region, total_k, target.matmul_resources, true)
        .context("small split physical map or shape rejected")?;
    let (schedule, ir) = candidate
        .partial_ir(
            &region,
            target.matmul_resources,
            target.supports_async_copy(),
        )
        .context("small split final layout/pipeline/resource proof rejected")?;
    let mut chosen = region.clone();
    chosen.schedule = schedule;
    let identity = format!(
        "small-grid-splitk/parts4/square64-k32/F32/left4-bf16rn/v1 target={target:?} precision={precision:?} region={chosen:?}"
    );
    let tag = format!("node{}-{m}-{n}-{total_k}", node.index());
    let partial = compile_ir(stream, target, ir, &identity, &format!("{tag}-partial"))?;
    ensure!(
        partial.num_regs()? <= 128
            && partial.local_size_bytes()? == 0
            && usize::try_from(partial.shared_size_bytes()?)? == schedule.shared_mem_bytes()
            && partial.max_threads_per_block()? >= 256
            && partial.occupancy_max_active_blocks_per_multiprocessor(256, 0, None)? >= 2,
        "small split driver resource proof rejected"
    );
    let reducer = compile_ir(
        stream,
        target,
        reducer_ir(m, n)?,
        &format!("small-grid-splitk/left4-no-initial-zero/bf16rn/v1 m={m} n={n}"),
        &format!("{tag}-reducer"),
    )?;
    #[cfg(test)]
    if let Some(directory) = std::env::var_os("TNSR_SMALL_GRID_SPLIT_K_AUDIT") {
        let directory = std::path::PathBuf::from(directory);
        std::fs::create_dir_all(&directory)?;
        let resources = serde_json::json!({"node":node.index(),"m":m,"n":n,"total_k":total_k,
            "part_k":part_k,"registers":partial.num_regs()?,"shared":partial.shared_size_bytes()?,
            "local":partial.local_size_bytes()?,"block":schedule.block_threads,
            "api_max_active_blocks":partial.occupancy_max_active_blocks_per_multiprocessor(256,0,None)?,
            "scope":"actual driver JIT resources; capacity is not achieved occupancy"});
        std::fs::write(
            directory.join(format!("{tag}-resources.json")),
            serde_json::to_vec_pretty(&resources)?,
        )?;
    }
    Ok(Prepared {
        partial,
        reducer,
        schedule,
    })
}

fn compile_ir(
    stream: &Arc<CudaStream>,
    target: PtxTarget,
    ir: crate::tile::TileIR,
    identity: &str,
    artifact: &str,
) -> Result<CudaFunction> {
    let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
    tiles
        .region_kernels
        .push(crate::tile::graph::TileRegionKernel { region_id: 0, ir });
    tiles.optimize_indices()?;
    let ptx = PtxGraph::from(tiles).with_target(target);
    let name = ptx
        .region_kernel_name(0)
        .context("small split generated function missing")?;
    let mut source = ptx.to_ptx();
    source.push_str("\n// ");
    source.push_str(identity);
    source.push('\n');
    #[cfg(test)]
    if let Some(directory) = std::env::var_os("TNSR_SMALL_GRID_SPLIT_K_AUDIT") {
        let directory = std::path::PathBuf::from(directory);
        std::fs::create_dir_all(&directory)?;
        std::fs::write(directory.join(format!("{artifact}.ptx")), &source)?;
    }
    #[cfg(not(test))]
    let _ = artifact;
    let module = stream.context().load_module(Ptx::from_src(source))?;
    Ok(module.load_function(name)?)
}

/// Literal existing Split-K reducer order: p0, then +p1,+p2,+p3. No initial
/// +0 and no BF16 conversion before its single final BF16 global store.
fn reducer_ir(m: usize, n: usize) -> Result<crate::tile::TileIR> {
    let len = m.checked_mul(n).context("small split reduction overflow")?;
    let mut ir = TileIRBuilder::new();
    ir.start_kernel("small_grid_split_k_left4_bf16rn_v1");
    ir.add_param("partial", DType::F32, true);
    ir.add_param("output", DType::BF16, false);
    ir.bounds_check(len);
    let index = Expr::Mul(
        Box::new(Expr::BlockIdx(Dim::X)),
        Box::new(Expr::BlockDim(Dim::X)),
    ) + Expr::ThreadIdx(Dim::X);
    let layout = MatrixLayout {
        rows: 1,
        cols: 4 * len,
        row_stride: 4 * len,
    };
    let mut sum = None;
    for part in 0..4 {
        let x = ir.alloc_register(DType::F32, 1, 1);
        ir.load_global_predicated(
            x,
            "partial",
            index.clone() + Expr::Const(i64::try_from(part * len)?),
            Expr::Const(0),
            index.clone(),
            layout,
        );
        sum = Some(if let Some(old) = sum {
            let next = ir.alloc_register(DType::F32, 1, 1);
            ir.add(next, old, x);
            next
        } else {
            x
        });
    }
    ir.store(
        "output",
        sum.context("small split partial sum missing")?,
        index,
        Expr::Const(0),
    );
    Ok(ir.finish())
}

#[cfg(test)]
mod tests;
