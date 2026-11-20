use super::tests::RejectGuard;
use super::*;
fn options() -> MixedOptions {
    MixedOptions {
        nn128x64_ldmatrix: true,
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
#[ignore = "requires real CUDA; actual final B8/B16/B32 NN census"]
fn production_nn128x64_actual_b8_b16_b32_final_role_ranges_resources_census() -> Result<()> {
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
        let rows=e.nn128x64_sources.iter().map(|(node,ptx,identity)|{
   let ks=e.kernels.iter().filter(|k|k.owner==Some(*node)&&nt128_bias::ldmatrix::function_identity(&k.function).is_ok_and(|id|id==*identity)).collect::<Vec<_>>();ensure!(ks.len()==1,"unique selected NNfunction");let k=ks[0];
   ensure!(matches!(&graph[*node],TensorGraphNode::MatMul{shape}if shape==&[M,N]),"actual exact NN output");
   ensure!(k.config.grid_dim==(8,128,1)&&k.config.block_dim==(32,8,1)&&k.pointers.len()==3&&k.pointers[2]%128==0,"actual NN launch/ABI/alignment");
   ensure!(ptx.contains("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16")&&!ptx.contains("add.rn.f32"),"collective plain no-bias source");
   let old=e.programs.iter().filter_map(|p|p.downcast_ref::<Program<BF16>>()).find(|p|p.kernels.len()==1&&p.kernels[0].owner==Some(*node)).context("retained original Program absent")?;
   ensure!(nt128_bias::ldmatrix::function_identity(&old.kernels[0].function)?!=*identity,"current kernel must differ from complete retained original");
   Ok(serde_json::json!({"owner":node.index(),"op":graph[*node].name(),"shape":graph[*node].shape(),"pointers":k.pointers,"launch":format!("{:?}",k.config),"registers":k.function.num_regs()?,"shared":k.function.shared_size_bytes()?,"local":k.function.local_size_bytes()?,"APIcapacity256":k.function.occupancy_max_active_blocks_per_multiprocessor(256,0,None)?}))
  }).collect::<Result<Vec<_>>>()?;
        let dir = std::path::PathBuf::from(
            std::env::var_os("TNSR_NN128X64_INTEGRATION_AUDIT")
                .context("set fresh integration audit")?,
        );
        std::fs::create_dir_all(&dir)?;
        std::fs::write(
            dir.join(format!("actual-b{batch}.json")),
            serde_json::to_vec_pretty(
                &serde_json::json!({"batch":batch,"info":e.info(),"selected":rows,"scope":"Actual final post-reuse constructor census; no model speed or achieved occupancy claim."}),
            )?,
        )?;
        assert_eq!(
            e.info().nn128x64_ldmatrix_programs,
            if batch == 32 { 8 } else { 0 },
            "{:?}",
            e.info()
        );
        assert_eq!(
            e.info().nn128x64_ldmatrix_fallback_programs,
            0,
            "{:?}",
            e.info()
        );
        assert_eq!(e.info().fused_nt128_bias_programs, 13);
        assert_eq!(e.info().fused_nt128_bias_fallback_programs, 8);
        assert_eq!(
            e.info().fused_nt128_bias_ldmatrix_programs,
            if batch == 32 { 12 } else { 0 }
        );
        assert_eq!(
            e.info()
                .online_attention_pair_numerical_screen_async16_pairs,
            4
        );
        assert_eq!(e.info().contiguous_bias_reshape_alias_steps, 20);
    }
    Ok(())
}
#[test]
#[ignore = "requires real CUDA; forced complete original NNProgram fallback"]
fn production_nn128x64_forced_postresource_rejection_retains_complete_original_and_bits()
-> Result<()> {
    if !crate::ptx::mixed::tests::cuda_available() {
        return Ok(());
    }
    use crate::tensor::TensorExpr;
    let expression = TensorExpr::<f32>::input("a", vec![32, 512, K])
        .cast(ValueDType::BF16)
        .reshape(vec![M, K])
        .matmul(TensorExpr::input("b", vec![K, N]).cast(ValueDType::BF16));
    let mut graph = TensorGraph::new();
    let output = graph.add_expr(&expression, false);
    let opts = MixedOptions {
        plain_bf16_output: true,
        shared_memory_reuse: true,
        physical_order_casts: true,
        ..MixedOptions::default()
    };
    let mut baseline =
        MixedExecutor::new_with_options(&graph, crate::tile::MatMulPrecision::Tf32x3, true, opts)?;
    let reject = RejectGuard::install();
    let mut candidate = MixedExecutor::new_with_options(
        &graph,
        crate::tile::MatMulPrecision::Tf32x3,
        true,
        MixedOptions {
            nn128x64_ldmatrix: true,
            ..opts
        },
    )?;
    drop(reject);
    assert_eq!(candidate.info().nn128x64_ldmatrix_programs, 0);
    assert_eq!(
        candidate.info().nn128x64_ldmatrix_fallback_programs,
        1,
        "{:?}",
        candidate.info()
    );
    assert!(
        candidate.info().nn128x64_ldmatrix_rejections[0]
            .contains("forced NN128x64 rejection after resource qualification"),
        "{:?}",
        candidate.info()
    );
    assert!(candidate.nn128x64_sources.is_empty());
    let programs = |e: &MixedExecutor| {
        e.programs
            .iter()
            .filter_map(|p| p.downcast_ref::<Program<BF16>>())
            .map(|p| (&p.ptx, &p.plan))
            .map(|(a, b)| (a.clone(), b.clone()))
            .collect::<Vec<_>>()
    };
    assert_eq!(programs(&baseline), programs(&candidate));
    let feeds = HashMap::from([
        ("a".into(), vec![0.0625; M * K]),
        ("b".into(), vec![-0.125; K * N]),
    ]);
    baseline.upload(feeds.clone())?;
    candidate.upload(feeds)?;
    for _ in 0..2 {
        baseline.run()?;
        candidate.run()?;
        assert_eq!(
            baseline
                .read(output)?
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            candidate
                .read(output)?
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
    }
    Ok(())
}
