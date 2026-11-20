//! Proven full-MN scalar BF16 output addresses. No arithmetic/launch changes.
use super::*;
use std::collections::HashSet;
mod proof;
#[cfg(test)]
mod tests;

#[cfg(test)]
thread_local! {static FORCE_REJECTION:Cell<bool>=const {Cell::new(false)};}
#[cfg(test)]
fn force_rejection<T>(run: impl FnOnce() -> T) -> T {
    FORCE_REJECTION.with(|flag| {
        struct Restore<'a>(&'a Cell<bool>, bool);
        impl Drop for Restore<'_> {
            fn drop(&mut self) {
                self.0.set(self.1);
            }
        }
        let _restore = Restore(flag, flag.replace(true));
        run()
    })
}
fn clone_tiles(t: &TileGraph) -> TileGraph {
    let regions = |v: &[crate::tile::graph::TileRegionKernel]| {
        v.iter()
            .map(|k| crate::tile::graph::TileRegionKernel {
                region_id: k.region_id,
                ir: k.ir.clone(),
            })
            .collect()
    };
    TileGraph {
        graph: t.graph.clone(),
        region_kernels: regions(&t.region_kernels),
        reduction_region_kernels: regions(&t.reduction_region_kernels),
        matmul_region_kernels: regions(&t.matmul_region_kernels),
        physical_nodes: t.physical_nodes.clone(),
    }
}
impl<D: CudaDType> PtxExecutor<D> {
    pub(super) fn compile_tile_variant_with_plan(
        &self,
        mut tiles: TileGraph,
        plan: &PtxExecutionPlan,
    ) -> Result<(PtxGraph, Arc<CudaModule>)> {
        if !self.experimental_kernel_options.plain_bf16_output {
            return self.compile_tile_variant(tiles);
        }
        let mut rewrites = Vec::new();
        let mut rejected = 0;
        for kernel in &tiles.matmul_region_kernels {
            let Some(region) = plan.matmul_regions().get(kernel.region_id) else {
                continue;
            };
            if proof::guard(region).is_err() {
                continue;
            }
            let mut ir = kernel.ir.clone();
            match proof::rewrite(
                &mut ir,
                proof::Scope {
                    m: region.m,
                    n: region.n,
                },
            ) {
                Ok(()) => rewrites.push((kernel.region_id, ir)),
                Err(_) => rejected += 1,
            }
        }
        if rewrites.is_empty() {
            let (mut graph, module) = self.compile_tile_variant(tiles)?;
            graph.set_plain_bf16_output_selection(HashSet::new(), rejected);
            return Ok((graph, module));
        }
        // Clone the complete optimized baseline only after at least one exact
        // proof succeeds. Existing CE candidate/fallback compilation is applied
        // independently to either module; its metadata stays tied to its body.
        let baseline = clone_tiles(&tiles);
        let selected = rewrites.iter().map(|(id, _)| *id).collect::<HashSet<_>>();
        for (id, ir) in rewrites {
            let kernel = tiles
                .matmul_region_kernels
                .iter_mut()
                .find(|k| k.region_id == id)
                .context("proved plain-output region disappeared")?;
            kernel.ir = ir;
        }
        let candidate = self.compile_tile_variant(tiles);
        let admitted = |graph: &PtxGraph, module: &Arc<CudaModule>| -> Result<bool> {
            #[cfg(test)]
            if FORCE_REJECTION.with(Cell::get) {
                return Ok(false);
            }
            for &id in &selected {
                let name = graph
                    .matmul_region_kernel_name(id)
                    .context("plain-output function absent")?;
                let f = module.load_function(name)?;
                if !super::square128_streamed::resources_fit(
                    f.num_regs()?,
                    f.local_size_bytes()?,
                    f.shared_size_bytes()?,
                    f.max_threads_per_block()?,
                    f.occupancy_max_active_blocks_per_multiprocessor(256, 0, None)?,
                    self.target.matmul_resources.shared_bytes_per_block,
                ) {
                    return Ok(false);
                }
            }
            Ok(true)
        };
        match candidate {
            Ok((mut graph, module)) if admitted(&graph, &module).unwrap_or(false) => {
                graph.set_plain_bf16_output_selection(selected, rejected);
                Ok((graph, module))
            }
            _ => {
                let (mut graph, module) = self
                    .compile_tile_variant(baseline)
                    .context("Plain BF16 output complete module fallback failed")?;
                graph.set_plain_bf16_output_selection(HashSet::new(), rejected + selected.len());
                Ok((graph, module))
            }
        }
    }
}
