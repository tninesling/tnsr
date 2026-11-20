use super::*;
#[test]
fn nt128_bias_shape_scope_rejects_tails_and_small_grids() {
    assert!(shape_candidate(&[4096, 1408], &[4096, 512]));
    for (output, lhs) in [
        (vec![4095, 1408], vec![4095, 512]),
        (vec![4096, 1409], vec![4096, 512]),
        (vec![4096, 1408], vec![4096, 513]),
        (vec![128, 1408], vec![128, 512]),
    ] {
        assert!(!shape_candidate(&output, &lhs));
    }
}

use std::cell::RefCell;
thread_local! { static CENSUS: RefCell<Option<Vec<serde_json::Value>>> = const { RefCell::new(None) }; }
struct CensusGuard;
impl CensusGuard {
    fn install() -> Result<Self> {
        CENSUS.with(|s| {
            ensure!(s.borrow().is_none(), "nested NT census");
            *s.borrow_mut() = Some(Vec::new());
            Ok(Self)
        })
    }
    fn finish(self) -> Vec<serde_json::Value> {
        CENSUS.with(|s| s.borrow_mut().take().unwrap())
    }
}
impl Drop for CensusGuard {
    fn drop(&mut self) {
        CENSUS.with(|s| {
            s.borrow_mut().take();
        });
    }
}
pub(super) fn snapshot<D: CudaDType>(node: NodeIndex, p: &Program<D>) -> Option<serde_json::Value> {
    CENSUS.with(|s|s.borrow().is_some()).then(||{
        let regions=p.metadata.as_ref().map(|m|m.plan.matmul_regions().iter().map(|r|serde_json::json!({"m":r.m,"n":r.n,"k":r.k,"schedule":format!("{:?}",r.schedule),"inputs":r.inputs.iter().map(|i|serde_json::json!({"source_shape":i.source_shape,"logical_shape":i.tensor.shape,"map":format!("{:?}",i.tensor),"binding":p.values.get(&i.tensor.source).map(|b|serde_json::json!({"len":b.len,"dtype":format!("{:?}",b.dtype),"base_mod16":b.pointer%16}))})).collect::<Vec<_>>()})).collect::<Vec<_>>()).unwrap_or_default();
        serde_json::json!({"node":node.index(),"original_kernel_count":p.kernels.len(),"regions":regions,"original_launches":p.kernels.iter().map(|k|serde_json::json!({"grid":k.config.grid_dim,"block":k.config.block_dim,"arguments":k.pointers.len(),"registers":k.function.num_regs().ok(),"shared":k.function.shared_size_bytes().ok(),"local":k.function.local_size_bytes().ok()})).collect::<Vec<_>>()})
    })
}
pub(super) fn record<D: CudaDType>(
    before: Option<serde_json::Value>,
    outcome: &Outcome,
    p: &Program<D>,
) {
    if let Some(mut row) = before {
        let object = row.as_object_mut().unwrap();
        object.insert(
            "selected".into(),
            serde_json::json!(matches!(outcome, Outcome::Selected)),
        );
        object.insert(
            "rejection".into(),
            match outcome {
                Outcome::Selected => serde_json::Value::Null,
                Outcome::Retained(reason) => serde_json::json!(reason),
            },
        );
        object.insert("chosen".into(),serde_json::json!(p.kernels.iter().map(|k|serde_json::json!({"grid":k.config.grid_dim,"block":k.config.block_dim,"arguments":k.pointers.len(),"label":k.label,"registers":k.function.num_regs().ok(),"shared":k.function.shared_size_bytes().ok(),"local":k.function.local_size_bytes().ok(),"capacity256":k.function.occupancy_max_active_blocks_per_multiprocessor(256,0,None).ok()})).collect::<Vec<_>>()));
        CENSUS.with(|s| s.borrow_mut().as_mut().unwrap().push(row));
    }
}
fn options() -> MixedOptions {
    MixedOptions {
        fused_nt128_bias: true,
        online_attention_pair_numerical_screen: true,
        layer_norm_pair_fusion: true,
        attention_softmax_fusion: true,
        attention_softmax_backward_fusion: true,
        swiglu_pair_fusion: true,
        qkv_grad_pack_fusion: true,
        transposed_bf16_outputs: true,
        split_k_bf16_tile_policy: crate::tile::SplitKBf16TilePolicy::Square64,
        direct_master_bindings: true,
        incremental_allocation_reuse: true,
        physical_order_casts: true,
        shared_memory_reuse: true,
        narrow_bf16_tile: true,
        plain_bf16_output: true,
        medium_bf16_tile_policy: crate::tile::MediumBf16TilePolicy::SquareThenTall,
        column_reduction_policy: crate::tile::ColumnReductionPolicy::Four,
        batched_attention_bf16_tile_policy:
            crate::tile::BatchedAttentionBf16TilePolicy::TallScoresByGrid,
        indexed_loss_cooperative_ordered_forward: true,
        indexed_loss_cooperative_ordered_backward: true,
        ..MixedOptions::default()
    }
}
#[test]
fn nt128_bias_actual_b8_b16_b32_role_layout_counter_census() -> Result<()> {
    if !crate::ptx::mixed::tests::cuda_available() {
        return Ok(());
    }
    use crate::ptx::{
        mixed::test_decoder::{Config, synthetic},
        resident::{AdamGroup, ResidentAdam},
    };
    for batch in [8, 16, 32] {
        let model = synthetic(Config {
            batch,
            sequence: 512,
            width: 512,
            heads: 8,
            hidden: 1408,
            layers: 4,
            vocab: 4096,
            epsilon: 1e-5,
            learning_rate: 1e-4,
            max_grad_norm: 1.,
        });
        let (graph, _) = model.training_graph_with_options(crate::graph::GradientOptions {
            compact_softmax: true,
        })?;
        let names = model
            .parameters
            .iter()
            .map(|(name, p)| (p.parameter.id(), name.to_string()))
            .collect::<HashMap<_, _>>();
        let mut roles = HashMap::new();
        for node in graph
            .graph
            .node_indices()
            .filter(|&n| matches!(graph[n], TensorGraphNode::Linear { .. }))
        {
            let mut weight = graph.inputs(node)[1];
            loop {
                if let TensorGraphNode::Parameter { id, .. } = &graph[weight] {
                    roles.insert(node.index(), names[id].clone());
                    break;
                }
                let inputs = graph.inputs(weight);
                ensure!(inputs.len() == 1, "unexpected Linear weight path");
                weight = inputs[0];
            }
        }
        let mut owner = ResidentAdam::new_with_graph_replay(
            model
                .parameters
                .iter()
                .map(|(_, p)| (p.parameter.clone(), 0))
                .collect(),
            vec![AdamGroup::new(1e-4)],
        )?;
        owner.set_update_batch_size(8)?;
        owner.set_experimental_first_accumulation_overwrite(true)?;
        owner.set_experimental_gradient_norm_partials(true)?;
        owner.begin_step()?;
        let observer = CensusGuard::install()?;
        let executor = owner.mixed_executor_with_options(
            &graph,
            crate::tile::MatMulPrecision::Tf32x3,
            options(),
        )?;
        let mut rows = observer.finish();
        for row in &mut rows {
            let node = row["node"].as_u64().unwrap() as usize;
            row.as_object_mut()
                .unwrap()
                .insert("role".into(), serde_json::json!(roles[&node]));
        }
        let selected = rows.iter().filter(|r| r["selected"] == true).count();
        assert_eq!(selected, executor.info().fused_nt128_bias_programs);
        assert_eq!(
            rows.len() - selected,
            executor.info().fused_nt128_bias_fallback_programs
        );
        assert_eq!(
            executor.info().online_attention_pair_numerical_screen_pairs,
            4,
            "{:?}",
            executor.info()
        );
        assert_eq!(
            executor
                .info()
                .online_attention_pair_numerical_screen_async16_pairs,
            4
        );
        assert_eq!(executor.info().layer_norm_pair_forward_regions, 9);
        assert_eq!(executor.info().qkv_grad_pack_regions, 4);
        println!(
            "B{batch} actual NT selected={selected},fallback={},roles={rows:#?}",
            rows.len() - selected
        );
        if let Some(directory) = std::env::var_os("TNSR_NT128_BIAS_AUDIT") {
            std::fs::create_dir_all(&directory)?;
            std::fs::write(
                std::path::PathBuf::from(directory).join(format!("actual-b{batch}-census.json")),
                serde_json::to_vec_pretty(
                    &serde_json::json!({"batch":batch,"info":executor.info(),"roles":rows,"scope":"Actual rebased Programs and chosen driver kernels. No achieved occupancy or speed claim."}),
                )?,
            )?;
        }
    }
    Ok(())
}

// cudarc0.17 exposes the handle in derived Debug but has no public getter.
// Compare that exact handle field without accessing private layout or unsafe.
fn function_identity(function: &CudaFunction) -> String {
    let text = format!("{function:?}");
    assert!(text.starts_with("CudaFunction { cu_function:"));
    text.split(", module:").next().unwrap().to_string()
}
type KernelSnapshot = (
    String,
    Option<NodeIndex>,
    String,
    (u32, u32, u32),
    (u32, u32, u32),
    u32,
    Vec<u64>,
    Option<(u64, usize, DType)>,
);
#[derive(Debug, PartialEq, Eq)]
pub(super) struct ExecutableSnapshot {
    kernels: Vec<KernelSnapshot>,
    output: (u64, usize, DType),
    values: std::collections::BTreeMap<NodeIndex, (u64, usize, DType)>,
    views: HashMap<NodeIndex, VirtualTensor>,
    metadata: Option<String>,
    ptx: String,
    plan: String,
    counters: String,
    owned_bytes: usize,
}
impl ExecutableSnapshot {
    pub(super) fn capture<D: CudaDType>(p: &Program<D>) -> Self {
        Self {
            kernels: p
                .kernels
                .iter()
                .map(|k| {
                    (
                        function_identity(&k.function),
                        k.owner,
                        k.label.clone(),
                        k.config.grid_dim,
                        k.config.block_dim,
                        k.config.shared_mem_bytes,
                        k.pointers.clone(),
                        k.zero_before.map(|b| (b.pointer, b.len, b.dtype)),
                    )
                })
                .collect(),
            output: (p.output.pointer, p.output.len, p.output.dtype),
            values: p
                .values
                .iter()
                .map(|(&n, b)| (n, (b.pointer, b.len, b.dtype)))
                .collect(),
            views: p.views.clone(),
            metadata: p
                .metadata
                .as_ref()
                .map(|m| format!("{:?} {:?}", m.plan, m.kernel_steps)),
            ptx: p.ptx.clone(),
            plan: p.plan.clone(),
            counters: format!(
                "{:?} {:?}",
                (
                    p.root_fused_matmul,
                    p.ordinary_bf16_four_warp_count,
                    p.bf16_eight_compute_warps,
                    p.medium_bf16_tile_counts,
                    p.column_reduction_counts,
                    p.large_bf16_tile_counts
                ),
                (
                    p.large_square128_streamed_counts,
                    p.plain_bf16_output_counts,
                    p.batched_attention_bf16_tile_counts,
                    p.indexed_loss_cooperative_ordered_backward_programs,
                    p.indexed_loss_cooperative_ordered_forward_programs
                )
            ),
            owned_bytes: p.bytes(),
        }
    }
}
thread_local! { static FORCE_REJECT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }
pub(super) fn forced_rejection() -> bool {
    FORCE_REJECT.with(|v| v.get())
}
struct RejectGuard;
impl RejectGuard {
    fn install() -> Self {
        FORCE_REJECT.with(|v| {
            assert!(!v.replace(true));
        });
        Self
    }
}
impl Drop for RejectGuard {
    fn drop(&mut self) {
        FORCE_REJECT.with(|v| v.set(false));
    }
}
#[test]
fn nt128_bias_forced_resource_rejection_keeps_complete_program_and_output() -> Result<()> {
    if !crate::ptx::mixed::tests::cuda_available() {
        return Ok(());
    }
    use crate::tensor::TensorExpr;
    let expression = TensorExpr::<f32>::input("a", vec![4096, 512]).linear_bf16(
        TensorExpr::input("w", vec![1408, 512]).transpose(),
        TensorExpr::input("bias", vec![1408]),
    )?;
    let mut graph = TensorGraph::new();
    let output = graph.add_expr(&expression, false);
    let base = MixedOptions {
        physical_order_casts: true,
        shared_memory_reuse: true,
        ..MixedOptions::default()
    };
    let mut baseline =
        MixedExecutor::new_with_options(&graph, crate::tile::MatMulPrecision::Tf32x3, true, base)?;
    let guard = RejectGuard::install();
    let mut fallback = MixedExecutor::new_with_options(
        &graph,
        crate::tile::MatMulPrecision::Tf32x3,
        true,
        MixedOptions {
            fused_nt128_bias: true,
            ..base
        },
    )?;
    drop(guard);
    assert_eq!(fallback.info().fused_nt128_bias_programs, 0);
    assert_eq!(fallback.info().fused_nt128_bias_fallback_programs, 1);
    assert!(
        fallback.info().fused_nt128_bias_rejections[0]
            .contains("test-only forced rejection after successful resource qualification"),
        "{:?}",
        fallback.info()
    );
    let bodies = |e: &MixedExecutor| {
        e.programs
            .iter()
            .filter_map(|p| {
                p.downcast_ref::<crate::ptx::linear::program::Program<crate::ptx::BF16>>()
                    .map(|p| p.ptx.clone())
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(bodies(&baseline), bodies(&fallback));
    let feeds = HashMap::from([
        ("a".into(), vec![0.0625; 4096 * 512]),
        ("w".into(), vec![-0.015625; 1408 * 512]),
        ("bias".into(), vec![1.5; 1408]),
    ]);
    for executor in [&mut baseline, &mut fallback] {
        executor.upload(feeds.clone())?;
        executor.run()?;
    }
    let a = baseline.read(output)?;
    let b = fallback.read(output)?;
    assert_eq!(a.len(), b.len());
    for (a, b) in a.into_iter().zip(b) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    Ok(())
}
