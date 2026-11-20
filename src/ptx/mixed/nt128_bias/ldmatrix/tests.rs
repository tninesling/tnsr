use super::*;
use std::cell::Cell;
thread_local! { static REJECT: Cell<bool> = const { Cell::new(false) }; }
pub(super) fn forced_rejection() -> bool {
    REJECT.get()
}

fn fixture(n: usize) -> Result<(MatMulRegion, PtxTarget)> {
    use crate::{
        tensor::TensorExpr,
        tile::{MatMulPrecision, MatMulResources},
    };
    let mut target = PtxTarget::sm80();
    target.compute_capability = (8, 9);
    target.matmul_resources = MatMulResources {
        shared_bytes_per_sm: 100 * 1024,
        threads_per_sm: 1536,
        ..MatMulResources::default()
    };
    let m = 16384;
    let a = TensorExpr::<half::bf16>::input("a", vec![m, 512]);
    let w = TensorExpr::<half::bf16>::input("w", vec![n, 512]).transpose();
    let bias = TensorExpr::<half::bf16>::input("bias", vec![n])
        .reshape(vec![1, n])
        .broadcast(vec![m, n]);
    let graph: TensorGraph<half::bf16> = (a.matmul(w) + bias).into();
    let plan = crate::ptx::PtxExecutionPlan::build_with_reduction_mode(
        &graph,
        crate::ptx::PtxReductionMode::DeterministicTree,
    )?;
    let mut r = plan.matmul_regions()[0].clone();
    r.schedule = MatMulSchedule::select_with_resources(
        m,
        n,
        512,
        DType::BF16,
        target.matmul_capabilities(),
        MatMulPrecision::Tf32x3,
        target.matmul_resources,
    )
    .with_shared_memory_reuse(true);
    r.configure_pipeline(
        MatMulSharedLayout::Padded,
        MatMulPipeline::Auto,
        target.supports_async_copy(),
        target.matmul_resources,
    )?;
    Ok((r, target))
}

#[test]
fn production_ldmatrix_exact_two_shape_and_current_typed_range_proofs() -> Result<()> {
    let (r, _) = fixture(1408)?;
    assert!(exact_shape(&r));
    for (m, n, k) in [
        (4096, 1408, 512),
        (8192, 1408, 512),
        (16384, 4096, 512),
        (16384, 1408, 1024),
        (16384, 1409, 512),
    ] {
        let mut other = r.clone();
        other.m = m;
        other.n = n;
        other.k = k;
        assert!(!exact_shape(&other));
    }
    let owner = Binding {
        pointer: 4096,
        len: 2048,
        dtype: DType::BF16,
    };
    let b = Binding {
        pointer: 4096 + 128,
        len: 128,
        dtype: DType::BF16,
    };
    assert!(owned_range(b, &[owner]).is_ok());
    assert!(owned_range(b, &[]).is_err());
    assert!(owned_range(b, &[owner, owner]).is_err());
    assert!(owned_range(Binding { len: 2048, ..b }, &[owner]).is_err());
    assert!(
        owned_range(
            Binding {
                dtype: DType::F32,
                ..b
            },
            &[owner]
        )
        .is_err()
    );
    assert!(owned_range(Binding { pointer: 4097, ..b }, &[owner]).is_err());
    assert!(
        owned_range(
            Binding {
                pointer: u64::MAX - 15,
                len: 128,
                ..b
            },
            &[owner]
        )
        .is_err()
    );
    assert!(!MixedOptions::default().fused_nt128_bias_ldmatrix);
    Ok(())
}
#[test]
fn production_ldmatrix_two_qualified_complete_ptx_modules_are_identical() -> Result<()> {
    for (name, n) in [("ffn", 1408), ("qkv", 1536)] {
        let (r, target) = fixture(n)?;
        let body = ir::candidate(&r, target)?;
        let mut graph = TileGraph::from_graph(&TensorGraph::<f32>::new());
        graph.add_matmul_regions(&[r])?;
        graph.matmul_region_kernels[0].ir = body;
        graph.optimize_indices()?;
        let ptx = PtxGraph::from(graph).with_target(target).module_source();
        let path = format!(
            "target/paper-kernels/explicit-bf16-mma-direct-qualification/ldmatrix-v1/device-v1/{name}_coalesced/candidate.ptx"
        );
        assert_eq!(
            ptx,
            std::fs::read_to_string(path)?,
            "qualified ldmatrix module changed"
        );
    }
    Ok(())
}

fn options() -> MixedOptions {
    MixedOptions {
        fused_nt128_bias: true,
        fused_nt128_bias_ldmatrix: true,
        contiguous_bias_reshape_aliases: true,
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
#[ignore = "requires real CUDA; actual model constructor census"]
fn production_ldmatrix_actual_b8_b16_b32_final_role_ranges_resources_census() -> Result<()> {
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
        let e = owner.mixed_executor_with_options(
            &graph,
            crate::tile::MatMulPrecision::Tf32x3,
            options(),
        )?;
        let sources=e.nt128_ldmatrix_sources.iter().map(|(node,ptx,identity)|{
            let kernels=e.kernels.iter().filter(|k|k.owner==Some(*node) && function_identity(&k.function).is_ok_and(|id| id==*identity)).collect::<Vec<_>>();
            assert_eq!(kernels.len(),1);
            let k=kernels[0];
            assert_eq!(k.pointers[3]%128,0);
            assert!(ptx.contains("ldmatrix.sync.aligned.m8n8.x4.shared.b16"));
            assert!(ptx.contains("st.global.b32"));
            let old=e.programs.iter().filter_map(|p|p.downcast_ref::<Program<BF16>>()).find(|p|p.kernels.len()==1 && p.kernels[0].owner==Some(*node)).unwrap();
            assert_ne!(function_identity(&old.kernels[0].function)?,function_identity(&k.function)?);
            assert_eq!(function_identity(&k.function.clone())?,function_identity(&k.function)?);
            super::super::qualify(&k.function)?;
            Ok(serde_json::json!({"owner":node.index(),"op":graph[*node].name(),"shape":graph[*node].shape(),"pointers":k.pointers,"launch":format!("{:?}",k.config),"registers":k.function.num_regs()?,"shared":k.function.shared_size_bytes()?,"local":k.function.local_size_bytes()?,"APIcapacity256":k.function.occupancy_max_active_blocks_per_multiprocessor(256,0,None)?}))
        }).collect::<Result<Vec<_>>>()?;
        let directory = std::env::var_os("TNSR_NT128_LDMATRIX_AUDIT")
            .context("set fresh TNSR_NT128_LDMATRIX_AUDIT")?;
        std::fs::create_dir_all(&directory)?;
        std::fs::write(
            std::path::PathBuf::from(directory).join(format!("actual-b{batch}.json")),
            serde_json::to_vec_pretty(
                &serde_json::json!({"batch":batch,"info":e.info(),"selected":sources,"scope":"Actual final post-reuse constructor census. No model speed or achieved occupancy claim."}),
            )?,
        )?;
        assert_eq!(e.info().fused_nt128_bias_programs, 13, "{:?}", e.info());
        assert_eq!(e.info().fused_nt128_bias_fallback_programs, 8);
        assert_eq!(
            e.info()
                .online_attention_pair_numerical_screen_async16_pairs,
            4
        );
        assert_eq!(
            e.info().fused_nt128_bias_ldmatrix_programs,
            if batch == 32 { 12 } else { 0 },
            "{:?}",
            e.info()
        );
        assert_eq!(
            e.info().fused_nt128_bias_ldmatrix_fallback_programs,
            0,
            "{:?}",
            e.info()
        );
    }
    Ok(())
}

struct RejectGuard;
impl RejectGuard {
    fn install() -> Self {
        assert!(!REJECT.replace(true));
        Self
    }
}
impl Drop for RejectGuard {
    fn drop(&mut self) {
        REJECT.set(false);
    }
}
#[test]
#[ignore = "requires real CUDA; forced post-resource whole-WMMA fallback"]
fn production_ldmatrix_forced_postresource_rejection_retains_whole_wmma_and_bits() -> Result<()> {
    if !crate::ptx::mixed::tests::cuda_available() {
        return Ok(());
    }
    use crate::tensor::TensorExpr;
    let expression = TensorExpr::<f32>::input("a", vec![16384, 512]).linear_bf16(
        TensorExpr::input("w", vec![1408, 512]).transpose(),
        TensorExpr::input("bias", vec![1408]),
    )?;
    let mut graph = TensorGraph::new();
    let output = graph.add_expr(&expression, false);
    let opts = MixedOptions {
        fused_nt128_bias: true,
        physical_order_casts: true,
        shared_memory_reuse: true,
        ..MixedOptions::default()
    };
    let mut base =
        MixedExecutor::new_with_options(&graph, crate::tile::MatMulPrecision::Tf32x3, true, opts)?;
    // A valid whole logical BF16 output range with only16B alignment must
    // reject before loading a candidate; the original captured kernel is never
    // launched with this diagnostic suffix pointer.
    let (r, _) = fixture(1408)?;
    let index = base
        .kernels
        .iter()
        .position(|k| {
            k.pointers.len() == 4
                && k.config.block_dim == (32, 8, 1)
                && k.config.grid_dim == (11, 128, 1)
        })
        .context("selected WMMA absent")?;
    let original = base.kernels[index].pointers[3];
    let pending = Pending {
        owner: base.kernels[index].owner.unwrap(),
        region: r,
        old_function: base.kernels[index].function.clone(),
    };
    let scratch = base.stream.alloc_zeros::<half::bf16>(16384 * 1408 + 8)?;
    // This extra owner is used only for host range proof and remains alive.
    let raw = scratch.device_ptr(&base.stream).0;
    assert_eq!(raw % 128, 0);
    base.buffers.push(Buffer::BF16(scratch));
    let owners = base
        .buffers
        .iter()
        .map(|b| binding(b, &base.stream))
        .chain(
            base.programs
                .iter()
                .filter_map(|p| p.downcast_ref::<Program<BF16>>())
                .flat_map(|p| p.storage_bindings(&base.stream)),
        )
        .collect::<Vec<_>>();
    base.kernels[index].pointers[3] = raw + 16;
    let reason = current_kernel(&pending, &base.kernels, &owners)
        .err()
        .context("16B suffix unexpectedly selected")?;
    assert!(
        format!("{reason:#}").contains("not 128-byte aligned"),
        "{reason:#}"
    );
    base.kernels[index].pointers[3] = original;
    current_kernel(&pending, &base.kernels, &owners)?;
    let guard = RejectGuard::install();
    let mut rejected = MixedExecutor::new_with_options(
        &graph,
        crate::tile::MatMulPrecision::Tf32x3,
        true,
        MixedOptions {
            fused_nt128_bias_ldmatrix: true,
            ..opts
        },
    )?;
    drop(guard);
    assert_eq!(rejected.info().fused_nt128_bias_programs, 1);
    assert_eq!(rejected.info().fused_nt128_bias_ldmatrix_programs, 0);
    assert_eq!(
        rejected.info().fused_nt128_bias_ldmatrix_fallback_programs,
        1
    );
    assert!(
        rejected.info().fused_nt128_bias_ldmatrix_rejections[0]
            .contains("forced ldmatrix rejection after resource qualification"),
        "{:?}",
        rejected.info()
    );
    assert!(rejected.nt128_ldmatrix_sources.is_empty());
    let programs = |e: &MixedExecutor| {
        e.programs
            .iter()
            .filter_map(|p| p.downcast_ref::<Program<BF16>>())
            .map(|p| p.ptx.clone())
            .collect::<Vec<_>>()
    };
    assert_eq!(programs(&base), programs(&rejected));
    let feeds = HashMap::from([
        ("a".into(), vec![0.0625; 16384 * 512]),
        ("w".into(), vec![-0.125; 1408 * 512]),
        ("bias".into(), vec![0.5; 1408]),
    ]);
    base.upload(feeds.clone())?;
    rejected.upload(feeds)?;
    for _ in 0..2 {
        base.run()?;
        rejected.run()?;
        assert_eq!(
            base.read(output)?
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            rejected
                .read(output)?
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
    }
    Ok(())
}

#[test]
fn production_ldmatrix_opaque_function_identity_is_fail_closed() -> Result<()> {
    assert_eq!(
        parse_function_identity("CudaFunction { cu_function: 0x123ABC, module: mutable context }")?,
        parse_function_identity(
            "CudaFunction { cu_function: 0x123ABC, module: different context }"
        )?
    );
    assert_ne!(
        parse_function_identity("CudaFunction { cu_function: 0x123ABC, module: x }")?,
        parse_function_identity("CudaFunction { cu_function: 0x123ABD, module: x }")?
    );
    for text in [
        "CudaFunction { name: x }",
        "CudaFunction { cu_function: 0x0, module: x }",
        "CudaFunction { cu_function: 0x123g, module: x }",
        "CudaFunction { cu_function: 0x123, module: cu_function: 0x456 }",
    ] {
        assert!(parse_function_identity(text).is_err());
    }
    Ok(())
}

pub(super) fn executable_snapshot(e: &MixedExecutor) -> String {
    format!(
        "{:?}",
        (
            e.kernels
                .iter()
                .map(|k| (
                    k.owner,
                    k.label.clone(),
                    function_identity(&k.function).unwrap(),
                    k.config.grid_dim,
                    k.config.block_dim,
                    k.config.shared_mem_bytes,
                    k.pointers.clone(),
                    k.zero_before.map(|b| (b.pointer, b.len, b.dtype))
                ))
                .collect::<Vec<_>>(),
            e.programs
                .iter()
                .filter_map(|p| p.downcast_ref::<Program<BF16>>())
                .map(|p| (
                    &p.ptx,
                    &p.plan,
                    p.metadata.as_ref().map(|m| format!("{:?}", m.plan)),
                    p.output.pointer,
                    p.output.len,
                    p.output.dtype,
                    p.values
                        .iter()
                        .map(|(n, b)| (n.index(), (b.pointer, b.len, b.dtype)))
                        .collect::<std::collections::BTreeMap<_, _>>()
                ))
                .collect::<Vec<_>>()
        )
    )
}

#[test]
fn production_ldmatrix_local_owner_collision_cannot_establish_module_identity() {
    let global = NodeIndex::new(42);
    assert!(same_instance(
        Some(global),
        "CUfunction-WMMA",
        global,
        "CUfunction-WMMA"
    ));
    assert!(!same_instance(
        Some(global),
        "CUfunction-unrelated-F32-local42",
        global,
        "CUfunction-WMMA"
    ));
    assert!(!same_instance(
        Some(NodeIndex::new(43)),
        "CUfunction-WMMA",
        global,
        "CUfunction-WMMA"
    ));
    assert!(!same_instance(
        None,
        "CUfunction-WMMA",
        global,
        "CUfunction-WMMA"
    ));
}
