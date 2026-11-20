//! Experimental qualified B32 rectangle; canonical fallback and math stay unchanged.
use super::*;
use crate::graph::TensorGraph;
use crate::tile::{MatMulPipeline, MatMulSharedLayout, TileGraph, TileIR};
use anyhow::{Result, ensure};
impl SplitKSquareCandidate {
    pub(crate) fn rectangle128x64_partial_ir(
        &self,
        r: &MatMulRegion,
        budget: MatMulResources,
        asynchronous: bool,
    ) -> Result<(MatMulSchedule, TileIR)> {
        // Recheck the immutable current whole-source and existing Square64 proof.
        let c = self;
        ensure!(r.k == 4096, "rectangle requires qualified partK4096");
        ensure!(
            c.configure(r, budget, asynchronous).is_some(),
            "canonical source/resource proof rejected"
        );
        ensure!(
            r.m.is_multiple_of(128) && r.n.is_multiple_of(64),
            "partial rectangle"
        );
        let mut probe = r.clone();
        probe.schedule = c
            .schedule
            .bf16_rectangle_geometry(128, 64)
            .ok_or_else(|| anyhow::anyhow!("rectangle unsupported"))?;
        probe.configure_pipeline(
            MatMulSharedLayout::Padded,
            MatMulPipeline::Auto,
            asynchronous,
            budget,
        )?;
        let s = probe.schedule;
        ensure!(
            s.block_tile
                == MatMulTile {
                    m: 128,
                    n: 64,
                    k: 32
                }
                && s.block_threads == (32, 8, 1)
                && s.warp_topology == (4, 2)
                && s.warp_fragments == (2, 2)
                && s.pipeline_stages == 2
                && s.copy_bytes == 16
                && s.async_operands == (true, true)
                && s.reuse_shared_memory
                && s.operand_layouts.0.column_major
                && !s.operand_layouts.1.column_major
                && s.shared_mem_bytes() == 32768
                && s.conservative_two_block_fit(budget),
            "rectangle resource/layout/pipeline rejected"
        );
        let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
        tiles.add_matmul_regions(&[probe])?;
        let mut ir = tiles
            .matmul_region_kernels
            .pop()
            .ok_or_else(|| anyhow::anyhow!("missing partial"))?
            .ir;
        ir.kernel_name = "split_k_rectangle128x64_partial_f32_v1".into();
        ensure!(
            ir.params.len() == 3 && ir.params.iter().filter(|p| p.is_input).count() == 2,
            "partial ABI changed"
        );
        for p in &mut ir.params {
            if p.is_input {
                ensure!(p.dtype == DType::BF16, "input dtype changed");
            } else {
                p.dtype = DType::F32;
            }
        }
        ir.validate_layouts()?;
        Ok((s, ir))
    }
}
