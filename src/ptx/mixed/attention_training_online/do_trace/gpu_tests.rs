//! Actual incoming-gradient descriptors only; no online kernel/model execution.
use super::super::leaf::{Descriptor, ReadLayout, Shape, prove};
use super::*;
use crate::ptx::mixed::{
    MixedOptions,
    attention_training_forward::binding_observer,
    test_decoder::{Config, synthetic},
};
use crate::ptx::resident::{AdamGroup, ResidentAdam};
use crate::tile::{
    BatchedAttentionBf16TilePolicy, ColumnReductionPolicy, MatMulPrecision, MediumBf16TilePolicy,
    SplitKBf16TilePolicy,
};

#[test]
fn online_training_actual_do_live_operation_descriptor() -> Result<()> {
    if !crate::ptx::mixed::tests::cuda_available() {
        return Ok(());
    }
    let batch: usize = std::env::var("TNSR_ONLINE_BINDING_BATCH")
        .ok()
        .map(|v| v.parse())
        .transpose()?
        .unwrap_or(8);
    ensure!(
        [8, 16, 32].contains(&batch),
        "actual dO inspection is restricted to B8/B16/B32"
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
    let traces = trace(&graph)?;
    ensure!(
        traces.len() == 4,
        "actual model must contain four complete old attention derivative pairs"
    );
    let observer = binding_observer::install(
        traces
            .iter()
            .flat_map(|t| [NodeIndex::new(t.old_dp), NodeIndex::new(t.old_dv)]),
    )?;
    let mut owner = ResidentAdam::new_with_graph_replay(
        model
            .parameters
            .iter()
            .map(|(_, p)| (p.parameter.clone(), 0))
            .collect(),
        vec![AdamGroup::new(1e-4)],
    )?;
    owner.set_experimental_first_accumulation_overwrite(true)?;
    owner.set_experimental_gradient_norm_partials(true)?;
    owner.set_experimental_adam_256_threads(true)?;
    owner.begin_step()?;
    let executor = owner.mixed_executor_with_options(
        &graph,
        MatMulPrecision::Tf32x3,
        MixedOptions {
            attention_training_forward_fusion: true,
            indexed_loss_cooperative_ordered_forward: true,
            indexed_loss_cooperative_ordered_backward: true,
            layer_norm_pair_fusion: true,
            split_k_bf16_tile_policy: SplitKBf16TilePolicy::Square64,
            swiglu_pair_fusion: true,
            qkv_grad_pack_fusion: true,
            direct_master_bindings: true,
            attention_softmax_backward_fusion: true,
            attention_softmax_fusion: true,
            physical_order_casts: true,
            transposed_bf16_outputs: true,
            shared_memory_reuse: true,
            narrow_bf16_tile: true,
            medium_bf16_tile_policy: MediumBf16TilePolicy::SquareThenTall,
            batched_attention_bf16_tile_policy: BatchedAttentionBf16TilePolicy::TallScoresByGrid,
            column_reduction_policy: ColumnReductionPolicy::Four,
            incremental_allocation_reuse: true,
            ..Default::default()
        },
    )?;
    let snapshots = observer.finish();
    let shape = Shape::new(batch, 8)?;
    let mut records = Vec::new();
    let mut all_proven = true;
    for trace in traces {
        let dp = snapshots
            .get(&NodeIndex::new(trace.old_dp))
            .context("live dP operands absent")?;
        let dv = snapshots
            .get(&NodeIndex::new(trace.old_dv))
            .context("live dV operands absent")?;
        ensure!(
            dp.len() == 2 && dv.len() == 2,
            "old dP/dV operation ABI differs"
        );
        let mut observations = Vec::new();
        for (role, value) in [("dP_lhs", &dp[0]), ("dV_rhs", &dv[1])] {
            let descriptor = Descriptor {
                pointer: value.binding.pointer,
                storage_len: value.binding.len,
                storage_dtype: value.binding.dtype,
                source_shape: value.source_shape.clone(),
                view: value.view.clone(),
            };
            let proof = prove(shape, &descriptor, false);
            all_proven &= proof == Some(ReadLayout::SequenceMajor);
            observations.push(serde_json::json!({"role":role,"dtype":format!("{:?}",value.binding.dtype),"binding_len":value.binding.len,"source_shape":value.source_shape,"logical_shape":value.view.shape,"access":format!("{:?}",value.view.access),"predicate":format!("{:?}",value.view.predicate),"out_of_bounds":format!("{:?}",value.view.out_of_bounds),"chain":value.chain.iter().map(TensorGraphNode::name).collect::<Vec<_>>(),"proof":format!("{proof:?}")}));
        }
        records.push(serde_json::json!({"pv":trace.pv,"incoming_do":trace.incoming_do,"dp":trace.old_dp,"dv":trace.old_dv,"observations":observations}));
    }
    ensure!(
        executor.info().reused_allocations > 0,
        "inspection must retain real allocation reuse"
    );
    if let Some(path) = std::env::var_os("TNSR_ONLINE_ATTENTION_AUDIT") {
        std::fs::create_dir_all(&path)?;
        std::fs::write(
            std::path::Path::new(&path).join(format!("actual-do-live-b{batch}.json")),
            serde_json::to_vec_pretty(
                &serde_json::json!({"batch":batch,"heads":8,"sequence":512,"width":64,"records":records,"checkpoint_forward_regions":executor.info().attention_training_forward_regions,"reused_allocations":executor.info().reused_allocations,"scope":"live operation metadata snapshots; no future pointer dereference, buffer ownership extension, online execution or new paired model hook"}),
            )?,
        )?;
    }
    ensure!(
        all_proven,
        "actual dO is not the proposed canonical sequence-major layout; inspect saved metadata"
    );
    Ok(())
}
