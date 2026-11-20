//! Long BF16 contractions use f32 partials and exactly one BF16 output store.
mod rectangle;
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
        // Keep all original packing and wider fallback decisions above. The
        // experiment may replace only this finalized, already-bound partial.
        let rectangle_requested =
            self.split_k_bf16_tile_policy == crate::tile::SplitKBf16TilePolicy::Rectangle128x64;
        let original_region = rectangle_requested.then(|| region.clone());
        let canonical_policy = if rectangle_requested {
            crate::tile::SplitKBf16TilePolicy::Square64
        } else {
            self.split_k_bf16_tile_policy
        };
        let candidate = canonical_policy
            .candidate(
                &region,
                4,
                k,
                target.matmul_resources,
                self.split_k_shared_memory_reuse,
            )
            .and_then(|c| {
                c.partial_ir(
                    &region,
                    target.matmul_resources,
                    target.supports_async_copy(),
                )
            });
        let mut selected = false;
        let mut function = None;
        if let Some((schedule, ir)) = candidate {
            let mut probe = region.clone();
            probe.schedule = schedule;
            // The full generated source is the driver module identity. Include
            // the policy ABI, target, precision, exact input maps/dtypes and
            // chosen schedule as explicit diagnostics in that same source.
            let identity = format!(
                "policy={} target={target:?} precision={precision:?} region={probe:?} output=F32 reducer=left4-bf16rn-v1",
                String::from_utf8_lossy(canonical_policy.codegen_tag()),
            );
            if let Ok(proposed) =
                compile_partial(&self.stream, target, &probe, Some(ir), Some(&identity))
            {
                let resources_fit = (|| -> Result<bool> {
                    Ok(proposed.num_regs()? <= 128
                        && proposed.shared_size_bytes()? == 18432
                        && proposed.local_size_bytes()? == 0
                        && proposed.max_threads_per_block()? >= 256
                        && proposed.occupancy_max_active_blocks_per_multiprocessor(256, 0, None)?
                            >= 2)
                })()
                .unwrap_or(false);
                if resources_fit {
                    region = probe;
                    function = Some(proposed);
                    selected = true;
                }
            }
        }
        // The existing region/function is an immutable complete fallback until
        // every late rectangle check succeeds; no old-function/new-grid mix.
        let mut rectangle_selected = false;
        if rectangle_requested {
            let diagnostic = format!("node={} [{m},{k}]x[{k},{n}]", node.index());
            let proposal = if selected {
                rectangle::propose(
                    &self.stream,
                    target,
                    precision,
                    original_region
                        .as_ref()
                        .context("rectangle source snapshot missing")?,
                    #[cfg(test)]
                    self.split_k_rectangle_test_resource_rejection,
                )
            } else {
                Err(anyhow::anyhow!(
                    "current Square64 fallback unavailable or outside exact physical family"
                ))
            };
            match proposal {
                Ok((proposed_region, proposed_function, resources)) => {
                    region = proposed_region;
                    function = Some(proposed_function);
                    rectangle_selected = true;
                    self.info.split_k_rectangle128x64_programs += 1;
                    self.info
                        .split_k_rectangle128x64_selections
                        .push(format!("{diagnostic} {resources}"));
                }
                Err(error) => {
                    self.info.split_k_rectangle128x64_fallback_programs += 1;
                    self.info
                        .split_k_rectangle128x64_rejections
                        .push(format!("{diagnostic}: {error:#}"));
                }
            }
        }
        let function = match function {
            Some(function) => function,
            None => compile_partial(&self.stream, target, &region, None, None)?,
        };
        if selected && !rectangle_selected {
            self.info.split_k_square64_programs += 1;
        } else if !selected
            && self.split_k_bf16_tile_policy != crate::tile::SplitKBf16TilePolicy::Off
        {
            self.info.split_k_square64_fallback_programs += 1;
        }
        let (_, partial) = self.alloc(node, vec![4, m, n], DType::F32)?;
        self.kernels.push(Kernel {
            owner: None,
            label: if rectangle_selected {
                format!(
                    "node={} BF16 split-K partial [{m},{k}]x[{k},{n}] rectangle128x64=true",
                    node.index()
                )
            } else if selected {
                format!(
                    "node={} BF16 split-K partial [{m},{k}]x[{k},{n}] square64=true",
                    node.index()
                )
            } else {
                format!(
                    "node={} BF16 split-K partial [{m},{k}]x[{k},{n}]",
                    node.index()
                )
            },
            function,
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
        let source = ptx.to_ptx();
        #[cfg(test)]
        if let Some(directory) = std::env::var_os("TNSR_SPLIT_K_SQUARE_PTX_DIR") {
            let directory = std::path::PathBuf::from(directory);
            std::fs::create_dir_all(&directory)?;
            let path = directory.join(format!("reducer-{m}-{n}.ptx"));
            if path.exists() {
                ensure!(
                    std::fs::read(&path)? == source.as_bytes(),
                    "preserve differing reducer PTX export"
                );
            } else {
                std::fs::write(&path, &source)?;
            }
        }
        let module = self.stream.context().load_module(Ptx::from_src(source))?;
        self.kernels.push(Kernel {
            owner: None,
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

/// Compile an immutable partial with the same typed operands and F32 output.
/// A failed opt-in compilation/resource query is discarded before installing
/// its schedule or function; the caller compiles the complete baseline instead.
fn compile_partial(
    stream: &Arc<CudaStream>,
    target: PtxTarget,
    region: &MatMulRegion,
    replacement: Option<crate::tile::TileIR>,
    identity: Option<&str>,
) -> Result<cudarc::driver::CudaFunction> {
    let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
    tiles.add_matmul_regions(std::slice::from_ref(region))?;
    if let Some(ir) = replacement {
        tiles.matmul_region_kernels[0].ir = ir;
    }
    // The MMA accumulator and epilogue registers are already F32. Only the
    // destination storage changes; input addressing and loads remain BF16.
    for param in &mut tiles.matmul_region_kernels[0].ir.params {
        if !param.is_input {
            param.dtype = DType::F32;
        }
    }
    tiles.optimize_indices()?;
    let ptx = PtxGraph::from(tiles).with_target(target);
    let mut source = ptx.to_ptx();
    if let Some(identity) = identity {
        // Debug metadata is single-line and trusted internal Rust descriptors.
        source.push_str("\n// ");
        source.push_str(identity);
        source.push('\n');
    }
    #[cfg(test)]
    let artifact = if let Some(directory) = std::env::var_os("TNSR_SPLIT_K_SQUARE_PTX_DIR") {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let directory = std::path::PathBuf::from(directory);
        std::fs::create_dir_all(&directory)?;
        let prefix = directory.join(format!(
            "compiled-{}-{}-{}-{}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed),
            region.m,
            region.n,
            region.k,
            if region.schedule.block_tile.m == 128 && identity.is_some() {
                "rectangle128x64"
            } else if identity.is_some() {
                "square64"
            } else {
                "baseline"
            }
        ));
        std::fs::write(prefix.with_extension("ptx"), &source)?;
        Some(prefix)
    } else {
        None
    };
    let module = stream.context().load_module(Ptx::from_src(source))?;
    let function = module.load_function(
        ptx.matmul_region_kernel_name(0)
            .context("split matmul kernel missing")?,
    )?;
    #[cfg(test)]
    if let Some(prefix) = artifact {
        let resources = serde_json::json!({"policy_identity":identity,"m":region.m,"n":region.n,"part_k":region.k,"registers":function.num_regs()?,"shared":function.shared_size_bytes()?,"local":function.local_size_bytes()?,"block":region.schedule.block_threads,"api_max_active_blocks_per_sm":function.occupancy_max_active_blocks_per_multiprocessor(256,0,None)?,"scope":"actual driver JIT resources; capacity is not achieved occupancy"});
        std::fs::write(
            prefix.with_extension("resources.json"),
            serde_json::to_vec_pretty(&resources)?,
        )?;
    }
    Ok(function)
}
