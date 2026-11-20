//! Complete square-plan/module qualification; CE's existing same-launch
//! arithmetic and fallback are reused unchanged inside each plan variant.
use super::*;
impl<D: CudaDType> PtxExecutor<D> {
    pub(super) fn lower_planned_tiles<G>(
        &self,
        graph: &TensorGraph<D::HostType, G>,
        storage: &HashMap<petgraph::graph::NodeIndex, crate::tile::DType>,
        execution_plan: &mut PtxExecutionPlan,
    ) -> Result<(TileGraph, Duration)> {
        let mut tile_graph = TileGraph::from_with_matmul_schedules(
            graph,
            self.target.supports_tf32()
                && matches!(
                    self.matmul_precision,
                    crate::tile::MatMulPrecision::AllowTf32 | crate::tile::MatMulPrecision::Tf32x3
                ),
            execution_plan.matmul_schedules(),
        );
        tile_graph.add_fusion_regions(execution_plan.regions())?;
        execution_plan.set_column_reduction_policy(
            self.experimental_kernel_options.column_reduction_policy,
            self.target.matmul_resources.shared_bytes_per_block,
        );
        tile_graph.add_reduction_regions(execution_plan.reduction_regions())?;
        tile_graph.add_matmul_regions(execution_plan.matmul_regions())?;
        // Only emit kernels selected by the final plan; fused-away regions retain
        // diagnostic metadata but must not bloat the loaded module.
        tile_graph.region_kernels.retain(|k| {
            execution_plan
                .steps()
                .iter()
                .any(|s| s.action == PtxPlanAction::PointwiseRegion(k.region_id))
        });
        tile_graph.reduction_region_kernels.retain(|k| {
            execution_plan
                .steps()
                .iter()
                .any(|s| s.action == PtxPlanAction::ReductionRegion(k.region_id))
        });
        tile_graph.matmul_region_kernels.retain(|k| {
            execution_plan
                .steps()
                .iter()
                .any(|s| s.action == PtxPlanAction::MatMulRegion(k.region_id))
        });
        for (id, region) in execution_plan.online_regions().iter().enumerate() {
            tile_graph
                .region_kernels
                .push(crate::tile::graph::TileRegionKernel {
                    region_id: execution_plan.regions().len() + id,
                    ir: region.lower_to_tile_ir(id)?,
                });
        }
        for step in execution_plan.steps() {
            if let Some(view) = &step.materialized_view {
                tile_graph.set_materialized_view(step.node, view, graph[view.source].shape())?;
            }
        }
        tile_graph.set_physical_nodes(
            execution_plan
                .steps()
                .iter()
                .filter_map(|step| (step.action == PtxPlanAction::Kernel).then_some(step.node))
                .collect(),
        );
        if !storage.is_empty() {
            let mut dtypes = storage.clone();
            for step in execution_plan.steps() {
                if step.action == PtxPlanAction::VirtualView {
                    for (&output, view) in step.outputs.iter().zip(&step.virtual_outputs) {
                        if let Some(&dtype) = dtypes.get(&view.source) {
                            dtypes.insert(output, dtype);
                        }
                    }
                    continue;
                }
                let ir = match step.action {
                    PtxPlanAction::Kernel => tile_graph.graph.node_weight_mut(step.node),
                    PtxPlanAction::PointwiseRegion(id) => tile_graph
                        .region_kernels
                        .iter_mut()
                        .find(|k| k.region_id == id)
                        .map(|k| &mut k.ir),
                    PtxPlanAction::ReductionRegion(id) => tile_graph
                        .reduction_region_kernels
                        .iter_mut()
                        .find(|k| k.region_id == id)
                        .map(|k| &mut k.ir),
                    PtxPlanAction::MatMulRegion(id) => tile_graph
                        .matmul_region_kernels
                        .iter_mut()
                        .find(|k| k.region_id == id)
                        .map(|k| &mut k.ir),
                    _ => None,
                };
                if let Some(ir) = ir {
                    for (param, node) in ir
                        .params
                        .iter_mut()
                        .zip(step.inputs.iter().chain(&step.outputs))
                    {
                        if let Some(&dtype) = dtypes.get(node) {
                            param.dtype = dtype;
                        }
                    }
                }
            }
        }
        if self.reduction_mode != PtxReductionMode::Strict {
            tile_graph.schedule_conv_weight_reductions();
        }
        let index_started = Instant::now();
        tile_graph.optimize_indices()?;
        let index_optimization_time = index_started.elapsed();
        Ok((tile_graph, index_optimization_time))
    }
    pub(super) fn compile_tile_variant(
        &self,
        mut tile_graph: TileGraph,
    ) -> Result<(PtxGraph, Arc<CudaModule>)> {
        let forward = self
            .experimental_kernel_options
            .indexed_loss_cooperative_ordered_forward;
        let backward = self
            .experimental_kernel_options
            .indexed_loss_cooperative_ordered_backward;
        let fallback_tiles = if forward || backward {
            PtxGraph::enable_indexed_loss_cooperative_ordered(
                &mut tile_graph,
                self.target.matmul_resources.shared_bytes_per_block,
                forward,
                backward,
            )
        } else {
            None
        };
        let compile_tiles = |tiles: TileGraph| -> Result<_> {
            let graph = PtxGraph::from(tiles).with_target(self.target);
            let source = graph.module_source();
            let module = self.device.load_module(Ptx::from_src(&source))?;
            Ok((graph, module))
        };
        let candidate = compile_tiles(tile_graph);
        if let Some(baseline) = fallback_tiles {
            match candidate {
                Ok((graph,module)) if graph.indexed_loss_cooperative_ordered_backward_nodes().iter().chain(graph.indexed_loss_cooperative_ordered_forward_nodes().iter()).all(|&node| {
                    graph.kernel_name(node).and_then(|name|module.load_function(name).ok()).is_some_and(|f|
                        PtxGraph::indexed_loss_cooperative_ordered_backward_resources_fit(&f,self.target.matmul_resources.shared_bytes_per_block).unwrap_or(false))
                }) =>Ok((graph,module)),
                _=>compile_tiles(baseline).context("Ordered CE fallback PTX module failed"),
            }
        } else {
            candidate.context("Failed to load PTX module with cudarc")
        }
    }
    pub(super) fn square128_resources_fit(
        &self,
        plan: &PtxExecutionPlan,
        graph: &PtxGraph,
        module: &Arc<CudaModule>,
    ) -> Result<bool> {
        #[cfg(test)]
        if FORCE_REJECTION.with(|force| force.get()) {
            return Ok(false);
        }
        for step in plan.square128_streamed_steps() {
            let name = match step.action {
                PtxPlanAction::Kernel => graph.kernel_name(step.node),
                PtxPlanAction::MatMulRegion(id) => graph.matmul_region_kernel_name(id),
                _ => None,
            }
            .context("square function absent")?;
            let f = module.load_function(name)?;
            let (registers, local, shared, max_threads, capacity) = (
                f.num_regs()?,
                f.local_size_bytes()?,
                f.shared_size_bytes()?,
                f.max_threads_per_block()?,
                f.occupancy_max_active_blocks_per_multiprocessor(256, 0, None)?,
            );
            #[cfg(test)]
            eprintln!(
                "Square128Streamed {name}: registers={registers}, local={local}, shared={shared}, max_threads={max_threads}, static_block_capacity={capacity} (not measured occupancy)"
            );
            #[cfg(test)]
            if let Some(directory) = std::env::var_os("TNSR_SQUARE128_PTX_DIR") {
                std::fs::create_dir_all(&directory)?;
                let stem = format!("jit-{name}-shared{shared}-regs{registers}");
                let directory = std::path::PathBuf::from(directory);
                std::fs::write(directory.join(format!("{stem}.ptx")), graph.module_source())?;
                std::fs::write(
                    directory.join(format!("{stem}.json")),
                    serde_json::to_vec_pretty(
                        &serde_json::json!({"function":name,"registers":registers,"local_bytes":local,"shared_bytes":shared,"max_threads":max_threads,"static_block_capacity":capacity,"capacity_is_not_measured_occupancy":true,"scope":"actual loaded module before square admission or complete-plan fallback"}),
                    )?,
                )?;
            }
            if !resources_fit(
                registers,
                local,
                shared,
                max_threads,
                capacity,
                self.target.matmul_resources.shared_bytes_per_block,
            ) {
                return Ok(false);
            }
        }
        Ok(true)
    }
}
pub(super) fn resources_fit(
    registers: i32,
    local: i32,
    shared: i32,
    max_threads: i32,
    capacity: u32,
    shared_limit: usize,
) -> bool {
    registers > 0
        && registers <= 128
        && local == 0
        && shared > 0
        && usize::try_from(shared).is_ok_and(|n| n <= shared_limit)
        && max_threads >= 256
        && capacity >= 2
}
#[cfg(test)]
thread_local! {pub(super) static FORCE_REJECTION:Cell<bool>=const{Cell::new(false)};}
#[cfg(test)]
mod one_cta_diagnostic;
#[cfg(test)]
mod training_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ptx::BF16, tensor::TensorExpr, tile::LargeBf16TilePolicy as Policy};
    use half::bf16;
    struct Reject;
    impl Reject {
        fn new() -> Self {
            FORCE_REJECTION.with(|f| f.set(true));
            Self
        }
    }
    impl Drop for Reject {
        fn drop(&mut self) {
            FORCE_REJECTION.with(|f| f.set(false));
        }
    }
    fn executor() -> Result<PtxExecutor<BF16>> {
        let mut e = PtxExecutor::<BF16>::try_new_for_dtype()?;
        e.set_matmul_precision(crate::tile::MatMulPrecision::NativeHalf);
        e.set_matmul_pipeline(MatMulPipeline::DoubleBuffered);
        e.set_matmul_shared_layout(MatMulSharedLayout::Padded);
        e.set_experimental_shared_memory_reuse(true);
        Ok(e)
    }
    fn device() -> bool {
        if let Err(error) = CudaContext::new(0) {
            assert!(
                std::env::var("TNSR_REQUIRE_CUDA").as_deref() != Ok("1"),
                "{error:#}"
            );
            false
        } else {
            true
        }
    }
    fn feeds(phase: usize) -> HashMap<String, Vec<bf16>> {
        HashMap::from([
            (
                "a".into(),
                (0..2048 * 128)
                    .map(|i| bf16::from_f32(((i + phase) % 31) as f32 / 64. - 0.25))
                    .collect(),
            ),
            (
                "b".into(),
                (0..128 * 1024)
                    .map(|i| bf16::from_f32(((i + phase * 3) % 19) as f32 / 32. - 0.125))
                    .collect(),
            ),
        ])
    }
    fn compare(actual: &[bf16], expected: &[bf16]) {
        assert_eq!(actual.len(), expected.len());
        for (&a, &b) in actual.iter().zip(expected) {
            if b.is_nan() {
                assert!(a.is_nan());
            } else {
                assert_eq!(a.to_bits(), b.to_bits());
            }
        }
    }
    #[test]
    fn square128_streamed_actual_resource_gate_boundaries() {
        assert!(resources_fit(128, 0, 40 * 1024, 256, 2, 48 * 1024));
        for args in [
            (129, 0, 40 * 1024, 256, 2),
            (128, 4, 40 * 1024, 256, 2),
            (128, 0, 40 * 1024, 255, 2),
            (128, 0, 40 * 1024, 256, 1),
            (0, 0, 40 * 1024, 256, 2),
            (128, 0, 0, 256, 2),
        ] {
            assert!(!resources_fit(
                args.0,
                args.1,
                args.2,
                args.3,
                args.4,
                48 * 1024
            ));
        }
        assert!(!resources_fit(128, 0, 49 * 1024, 256, 2, 48 * 1024));
    }
    #[test]
    fn square128_streamed_rejected_module_restores_complete_plan_and_launch() -> Result<()> {
        if !device() {
            return Ok(());
        }
        let graph: TensorGraph<bf16> = TensorExpr::input("a", vec![2048, 128])
            .matmul(TensorExpr::input("b", vec![128, 1024]))
            .into();
        let mut e = executor()?;
        if e.target.matmul_resources.shared_bytes_per_sm < 80 * 1024 {
            return Ok(());
        }
        e.compile(&graph)?;
        let baseline_source = e.ptx_graph.as_ref().unwrap().module_source();
        let baseline_plan = e.execution_plan.as_ref().unwrap().clone();
        let expected = e.execute_compiled(&graph, feeds(0))?;
        e.set_experimental_large_bf16_tile_policy(Policy::Square128Streamed);
        {
            let _reject = Reject::new();
            e.compile(&graph)?;
        }
        assert_eq!(e.large_square128_streamed_counts(), (0, 1));
        assert_eq!(e.execution_plan().unwrap(), baseline_plan.as_ref());
        assert_eq!(
            e.execution_plan().unwrap().matmul_schedules(),
            baseline_plan.matmul_schedules()
        );
        assert_eq!(
            e.execution_plan().unwrap().matmul_regions(),
            baseline_plan.matmul_regions()
        );
        assert_eq!(
            e.ptx_graph.as_ref().unwrap().module_source(),
            baseline_source
        );
        compare(&e.execute_compiled(&graph, feeds(0))?, &expected);
        assert!(e.compilation_signature.is_some());
        Ok(())
    }
    #[test]
    fn square128_streamed_cache_toggle_orientations_changed_feeds_and_nonfinites() -> Result<()> {
        if !device() {
            return Ok(());
        }
        for (left, right) in [(false, false), (false, true), (true, false), (true, true)] {
            let a = TensorExpr::input(
                "a",
                if left {
                    vec![128, 2048]
                } else {
                    vec![2048, 128]
                },
            );
            let b = TensorExpr::input(
                "b",
                if right {
                    vec![1024, 128]
                } else {
                    vec![128, 1024]
                },
            );
            let graph: TensorGraph<bf16> = (if left { a.transpose() } else { a })
                .matmul(if right { b.transpose() } else { b })
                .into();
            let mut e = executor()?;
            if e.target.matmul_resources.shared_bytes_per_sm < 80 * 1024 {
                return Ok(());
            }
            let mut expected = Vec::new();
            let mut baseline_source = String::new();
            let mut baseline_plan = None;
            for (pass, policy) in [Policy::Off, Policy::Square128Streamed, Policy::Off]
                .into_iter()
                .enumerate()
            {
                e.set_experimental_large_bf16_tile_policy(policy);
                if pass > 0 {
                    assert!(e.execute_compiled(&graph, HashMap::new()).is_err());
                }
                e.compile(&graph)?;
                let counts = e.large_square128_streamed_counts();
                if pass == 1 {
                    assert!(
                        matches!(counts, (1, 0) | (0, 1)),
                        "orientation={left},{right}, counts={counts:?}"
                    );
                    if counts.1 == 1 {
                        assert_eq!(
                            e.execution_plan.as_ref().unwrap().as_ref(),
                            baseline_plan.as_ref().unwrap()
                        );
                        assert_eq!(
                            e.ptx_graph.as_ref().unwrap().module_source(),
                            baseline_source
                        );
                    }
                } else {
                    assert_eq!(counts, (0, 0));
                }
                if pass == 0 {
                    baseline_source = e.ptx_graph.as_ref().unwrap().module_source();
                    baseline_plan = Some(e.execution_plan.as_ref().unwrap().as_ref().clone());
                } else if pass == 2 {
                    assert_eq!(
                        e.ptx_graph.as_ref().unwrap().module_source(),
                        baseline_source
                    );
                }
                let mut observed = Vec::new();
                for phase in 0..2 {
                    observed.push(e.execute_compiled(&graph, feeds(phase))?);
                }
                for (a, b) in [
                    (1., f32::INFINITY),
                    (0., f32::INFINITY),
                    (-0., 1.),
                    (f32::NEG_INFINITY, 1.),
                ] {
                    observed.push(e.execute_compiled(
                        &graph,
                        HashMap::from([
                            ("a".into(), vec![bf16::from_f32(a); 2048 * 128]),
                            ("b".into(), vec![bf16::from_f32(b); 128 * 1024]),
                        ]),
                    )?);
                }
                if pass == 0 {
                    expected = observed;
                } else {
                    for (a, b) in observed.iter().zip(&expected) {
                        compare(a, b);
                    }
                }
                if pass == 1 {
                    let module = e.module.as_ref().unwrap();
                    let ptx = e.ptx_graph.as_ref().unwrap();
                    for step in e.execution_plan().unwrap().square128_streamed_steps() {
                        let name = match step.action {
                            PtxPlanAction::Kernel => ptx.kernel_name(step.node),
                            PtxPlanAction::MatMulRegion(id) => ptx.matmul_region_kernel_name(id),
                            _ => None,
                        }
                        .unwrap();
                        let f = module.load_function(name)?;
                        let r = serde_json::json!({"registers":f.num_regs()?,"shared_bytes":f.shared_size_bytes()?,"local_bytes":f.local_size_bytes()?,"max_threads":f.max_threads_per_block()?,"active_block_capacity":f.occupancy_max_active_blocks_per_multiprocessor(256,0,None)?,"capacity_is_not_measured_occupancy":true});
                        if let Some(dir) = std::env::var_os("TNSR_SQUARE128_PTX_DIR") {
                            std::fs::create_dir_all(&dir)?;
                            let path = std::path::PathBuf::from(dir);
                            std::fs::write(
                                path.join(format!("driver-{left}-{right}.ptx")),
                                ptx.module_source(),
                            )?;
                            std::fs::write(
                                path.join(format!("driver-{left}-{right}.json")),
                                serde_json::to_vec_pretty(&r)?,
                            )?;
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
