//! Long BF16 contractions use f32 partials and exactly one BF16 output store.
use super::*;
use crate::tile::{Dim, Expr, MatMulRegion, MatMulSchedule, ir::MatrixLayout};

impl MixedExecutor {
    pub(super) fn split_bf16_matmul(
        &mut self,
        node: NodeIndex,
        op: &TensorGraphNode<f32>,
        operands: &[Value],
        precision: MatMulPrecision,
    ) -> Result<Option<Value>> {
        if !matches!(op, TensorGraphNode::MatMul { .. })
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
        if m < 128 || n < 128 || k < 2048 || k / m.max(n) < 4 || !k.is_multiple_of(4) {
            return Ok(None);
        }
        let target = PtxTarget::from_context(self.stream.context())?;
        if !target.matmul_capabilities().bf16 || precision == MatMulPrecision::StrictF32 {
            return Ok(None);
        }
        ensure!(
            operands[1].view.shape[0] == k,
            "split matmul inner dimension mismatch"
        );
        let mut operands = operands.to_vec();
        let mut graph = TensorGraph::<half::bf16>::new();
        let lhs_shape = vec![4, m, k / 4];
        let rhs_shape = vec![4, k / 4, n];
        let a = graph.graph.add_node(TensorGraphNode::Input {
            name: "a",
            shape: lhs_shape,
        });
        let b = graph.graph.add_node(TensorGraphNode::Input {
            name: "b",
            shape: rhs_shape,
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
            k / 4,
            DType::BF16,
            target.matmul_capabilities(),
            precision,
            target.matmul_resources,
        );
        if region.schedule.block_tile.m == 32 && region.schedule.block_tile.n == 32 {
            region.schedule.block_tile.k = 64;
            region.schedule = region.schedule.with_shared_layout(
                crate::tile::MatMulSharedLayout::Padded,
                target.matmul_resources,
            )?;
        }
        region.inputs[0].tensor = operands[0]
            .view
            .reshape(vec![m, 4, k / 4])?
            .permute(&[1, 0, 2])?;
        region.inputs[0].source_shape = operands[0].source_shape.clone();
        region.inputs[1].tensor = operands[1].view.reshape(vec![4, k / 4, n])?;
        region.inputs[1].source_shape = operands[1].source_shape.clone();
        region.configure_pipeline(
            crate::tile::MatMulSharedLayout::Padded,
            crate::tile::MatMulPipeline::Auto,
            target.supports_async_copy(),
            target.matmul_resources,
        )?;
        // A long contraction reuses each operand in many output tiles. Repack
        // an unaligned operand once so every tile can use aligned async copies.
        // The physical K dimension is contiguous in both packed buffers; the
        // RHS retains its logical [K, N] view through a transpose.
        let mut repacked = false;
        if region.schedule.pipeline_stages == 2 {
            for (index, aligned) in [
                region.schedule.async_operands.0,
                region.schedule.async_operands.1,
            ]
            .into_iter()
            .enumerate()
            {
                if aligned {
                    continue;
                }
                let mut source = operands[index].clone();
                if index == 1 {
                    source.view = source.view.permute(&[1, 0])?;
                }
                let (_, mut packed) = self.alloc(node, source.view.shape.clone(), DType::BF16)?;
                self.kernels
                    .push(materialize(&self.stream, &source, packed.binding)?);
                if index == 1 {
                    packed.view = packed.view.permute(&[1, 0])?;
                }
                operands[index] = packed;
                repacked = true;
            }
        }
        if repacked {
            region.inputs[0].tensor = operands[0]
                .view
                .reshape(vec![m, 4, k / 4])?
                .permute(&[1, 0, 2])?;
            region.inputs[0].source_shape = operands[0].source_shape.clone();
            region.inputs[1].tensor = operands[1].view.reshape(vec![4, k / 4, n])?;
            region.inputs[1].source_shape = operands[1].source_shape.clone();
            let mut wider = region.clone();
            wider.schedule.block_tile = crate::tile::MatMulTile {
                m: 64,
                n: 64,
                k: 32,
            };
            wider.schedule.warp_topology = (2, 2);
            wider.schedule.warp_fragments = (2, 2);
            wider.configure_pipeline(
                crate::tile::MatMulSharedLayout::Padded,
                crate::tile::MatMulPipeline::Auto,
                target.supports_async_copy(),
                target.matmul_resources,
            )?;
            if wider.schedule.pipeline_stages == 2 {
                region = wider;
            } else {
                region.configure_pipeline(
                    crate::tile::MatMulSharedLayout::Padded,
                    crate::tile::MatMulPipeline::Auto,
                    target.supports_async_copy(),
                    target.matmul_resources,
                )?;
            }
        }
        // Expand perpendicular to the scalar operand so each block reuses
        // those expensive loads for twice as many outputs. Use all eight warps
        // with one fragment each; retain the smaller tile if two stages do not fit.
        if region.schedule.pipeline_stages == 2
            && region.schedule.block_tile.m == 32
            && region.schedule.block_tile.n == 32
            && matches!(
                region.schedule.async_operands,
                (true, false) | (false, true)
            )
        {
            let mut wider = region.clone();
            if region.schedule.async_operands.0 {
                wider.schedule.block_tile.m = 64;
                wider.schedule.warp_topology = (4, 2);
            } else {
                wider.schedule.block_tile.n = 64;
                wider.schedule.warp_topology = (2, 4);
            }
            wider.configure_pipeline(
                crate::tile::MatMulSharedLayout::Padded,
                crate::tile::MatMulPipeline::Auto,
                target.supports_async_copy(),
                target.matmul_resources,
            )?;
            if wider.schedule.pipeline_stages == 2
                && wider.schedule.async_operands == region.schedule.async_operands
            {
                region = wider;
            }
        }
        let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
        tiles.add_matmul_regions(&[region.clone()])?;
        // The MMA accumulator and epilogue registers are already f32. Only the
        // destination storage changes; input addressing/loads remain BF16.
        for param in &mut tiles.matmul_region_kernels[0].ir.params {
            if !param.is_input {
                param.dtype = DType::F32;
            }
        }
        tiles.optimize_indices()?;
        let ptx = PtxGraph::from(tiles).with_target(target);
        let module = self
            .stream
            .context()
            .load_module(Ptx::from_src(ptx.to_ptx()))?;
        let name = ptx
            .matmul_region_kernel_name(0)
            .context("split matmul kernel missing")?;
        let (_, partial) = self.alloc(node, vec![4, m, n], DType::F32)?;
        self.kernels.push(Kernel {
            label: format!(
                "node={} BF16 split-K partial [{m},{k}]x[{k},{n}]",
                node.index()
            ),
            function: module.load_function(name)?,
            config: LaunchConfig {
                grid_dim: (
                    u32::try_from(n.div_ceil(region.schedule.block_tile.n))?,
                    u32::try_from(m.div_ceil(region.schedule.block_tile.m))?,
                    4,
                ),
                block_dim: region.schedule.block_threads,
                shared_mem_bytes: 0,
            },
            pointers: vec![
                operands[0].binding.pointer,
                operands[1].binding.pointer,
                partial.binding.pointer,
            ],
            zero_before: None,
        });
        let (_, output) = self.alloc(node, vec![m, n], DType::BF16)?;
        let len = m.checked_mul(n).context("split matmul output overflow")?;
        let mut ir = TileIRBuilder::new();
        ir.start_kernel("split_bf16_sum");
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
            sum.context("split partial sum missing")?,
            index,
            Expr::Const(0),
        );
        let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
        tiles
            .region_kernels
            .push(crate::tile::graph::TileRegionKernel {
                region_id: 0,
                ir: ir.finish(),
            });
        tiles.optimize_indices()?;
        let ptx = PtxGraph::from(tiles).with_target(target);
        let module = self
            .stream
            .context()
            .load_module(Ptx::from_src(ptx.to_ptx()))?;
        self.kernels.push(Kernel {
            label: format!("node={} f32 split-K sum to BF16 [{m},{n}]", node.index()),
            function: module.load_function(
                ptx.region_kernel_name(0)
                    .context("split reduction kernel missing")?,
            )?,
            config: LaunchConfig::for_num_elems(u32::try_from(len)?),
            pointers: vec![partial.binding.pointer, output.binding.pointer],
            zero_before: None,
        });
        self.info.bf16_tensor_core_programs += 1;
        Ok(Some(output))
    }
}
