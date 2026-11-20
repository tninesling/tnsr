//! Real decoder descriptors and sequence-major leaf qualification.
use super::super::input_layout::{prove, prove_triplet};
use super::*;

#[test]
fn training_attention_union_actual_decoder_runtime_binding_descriptors() -> Result<()> {
    use crate::ptx::mixed::test_decoder::{Config, synthetic};
    if !super::super::super::tests::cuda_available() {
        return Ok(());
    }
    let batch: usize = std::env::var("TNSR_ATTENTION_BINDING_BATCH")
        .ok()
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(8);
    ensure!(
        [8, 16, 32].contains(&batch),
        "binding inspection is restricted to established B8/B16/B32 cases"
    );
    let model = synthetic(Config {
        width: 512,
        layers: 4,
        heads: 8,
        hidden: 1408,
        sequence: 512,
        batch,
        vocab: 4096,
        epsilon: 1e-5,
        learning_rate: 1e-4,
        max_grad_norm: 1.,
    });
    let (graph, _) = model.training_graph_with_options(crate::graph::GradientOptions {
        compact_softmax: true,
    })?;
    let regions = super::super::recognize::recognize(&graph);
    ensure!(
        regions.len() == 4,
        "actual decoder must contain four attention unions"
    );
    let observer = super::super::binding_observer::install(
        regions.iter().flat_map(|region| [region.qk, region.output]),
    )?;
    let mut owner = crate::ptx::resident::ResidentAdam::new_with_graph_replay(
        model
            .parameters
            .iter()
            .map(|(_, p)| (p.parameter.clone(), 0))
            .collect(),
        vec![crate::ptx::resident::AdamGroup::new(1e-4)],
    )?;
    owner.set_experimental_first_accumulation_overwrite(true)?;
    owner.set_experimental_gradient_norm_partials(true)?;
    owner.set_experimental_adam_256_threads(true)?;
    owner.begin_step()?;
    let executor = owner.mixed_executor_with_options(
        &graph,
        MatMulPrecision::Tf32x3,
        MixedOptions {
            incremental_allocation_reuse: true,
            direct_master_bindings: true,
            layer_norm_pair_fusion: true,
            medium_bf16_tile_policy: crate::tile::MediumBf16TilePolicy::SquareThenTall,
            split_k_bf16_tile_policy: crate::tile::SplitKBf16TilePolicy::Square64,
            column_reduction_policy: crate::tile::ColumnReductionPolicy::Four,
            indexed_loss_cooperative_ordered_backward: true,
            physical_order_casts: true,
            transposed_bf16_outputs: true,
            shared_memory_reuse: true,
            narrow_bf16_tile: true,
            batched_attention_bf16_tile_policy:
                crate::tile::BatchedAttentionBf16TilePolicy::TallScoresByGrid,
            attention_softmax_fusion: true,
            attention_softmax_backward_fusion: true,
            qkv_grad_pack_fusion: true,
            swiglu_pair_fusion: true,
            ..Default::default()
        },
    )?;
    let snapshots = observer.finish();
    let mut records = Vec::new();
    let mut all_proven = true;
    for region in regions {
        let qk = snapshots
            .get(&region.qk)
            .context("actual QK operation snapshot absent")?;
        let pv = snapshots
            .get(&region.output)
            .context("actual PV operation snapshot absent")?;
        ensure!(
            qk.len() == 2 && pv.len() == 2,
            "attention operation ABI requires two operands"
        );
        let (q, k, v) = (&qk[0], &qk[1], &pv[1]);
        let layout = prove_triplet(q, k, v, batch, 8);
        all_proven &= layout == Some(InputLayout::SequenceMajor { heads: 8 });
        let mut operands = Vec::new();
        for (role, node, value, key) in [
            ("query", region.query, q, false),
            ("key", region.key, k, true),
            ("value", region.value, v, false),
        ] {
            operands.push(serde_json::json!({"role":role,"node":node.index(),"logical_shape":value.view.shape,"source_shape":value.source_shape,"binding_len":value.binding.len,"binding_dtype":format!("{:?}",value.binding.dtype),"virtual_source":value.view.source.index(),"access":format!("{:?}",value.view.access),"predicate":format!("{:?}",value.view.predicate),"out_of_bounds":format!("{:?}",value.view.out_of_bounds),"layout":format!("{:?}",prove(value,batch,8,key)),"chain":value.chain.iter().map(TensorGraphNode::name).collect::<Vec<_>>()}));
        }
        records.push(serde_json::json!({"qk":region.qk.index(),"pv":region.output.index(),"triplet_proof":format!("{layout:?}"),"operands":operands}));
    }
    ensure!(
        executor.info().reused_allocations > 0,
        "fixture must exercise real prefix/final reuse"
    );
    if let Some(path) = std::env::var_os("TNSR_ATTENTION_TRAINING_FORWARD_AUDIT") {
        std::fs::create_dir_all(&path)?;
        std::fs::write(
            std::path::Path::new(&path)
                .join(format!("actual-b{batch}-runtime-binding-descriptors.json")),
            serde_json::to_vec_pretty(
                &serde_json::json!({"config":{"width":512,"layers":4,"heads":8,"hidden":1408,"sequence":512,"batch":batch,"vocab":4096},"regions":records,"inspection":"actual operation operands during resident/incremental/direct preparation; metadata snapshots do not affect reuse or prove future union lifetimes","runtime_hook":false}),
            )?,
        )?;
    }
    ensure!(
        all_proven,
        "actual descriptors are not the proposed sequence-major ABI; inspect saved descriptor report"
    );
    Ok(())
}
