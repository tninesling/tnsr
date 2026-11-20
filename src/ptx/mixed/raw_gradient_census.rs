//! Test-only final-gradient storage census. No alternative Adam implementation.
use super::*;
use crate::{
    ptx::resident::{AdamGroup, ResidentAdam},
    tile::{BatchedAttentionBf16TilePolicy, MediumBf16TilePolicy, SplitKBf16TilePolicy},
};
use serde_json::json;
use std::{collections::HashMap, path::PathBuf};

fn volume(shape: &[usize]) -> Option<usize> {
    shape.iter().try_fold(1_usize, |n, d| n.checked_mul(*d))
}

/// Equality of complete normalized address maps, never sampled coordinates.
fn whole_contiguous(view: &VirtualTensor, source_shape: &[usize], len: usize) -> Result<bool> {
    if volume(&view.shape) != Some(len)
        || volume(source_shape) != Some(len)
        || view.predicate.is_some()
        || view.out_of_bounds.is_some()
    {
        return Ok(false);
    }
    let canonical = VirtualTensor::identity(view.source, source_shape.to_vec(), view.dtype)
        .reshape(view.shape.clone())?;
    Ok(view.access.clone().normalize_in_domain(&view.shape)?
        == canonical.access.normalize_in_domain(&view.shape)?)
}

fn accepted_options() -> MixedOptions {
    MixedOptions {
        physical_order_casts: true,
        shared_memory_reuse: true,
        incremental_allocation_reuse: true,
        direct_master_bindings: true,
        attention_softmax_fusion: true,
        attention_softmax_backward_fusion: true,
        narrow_bf16_tile: true,
        transposed_bf16_outputs: true,
        medium_bf16_tile_policy: MediumBf16TilePolicy::SquareThenTall,
        qkv_grad_pack_fusion: true,
        swiglu_pair_fusion: true,
        column_reduction_policy: crate::tile::ColumnReductionPolicy::Four,
        batched_attention_bf16_tile_policy: BatchedAttentionBf16TilePolicy::TallScoresByGrid,
        layer_norm_pair_fusion: true,
        split_k_bf16_tile_policy: SplitKBf16TilePolicy::Square64,
        indexed_loss_cooperative_ordered_backward: true,
        indexed_loss_cooperative_ordered_forward: true,
        contiguous_bias_reshape_aliases: true,
        ..MixedOptions::default()
    }
}

#[test]
fn raw_gradient_census_proves_whole_maps_without_sampling() -> Result<()> {
    let source = NodeIndex::new(0);
    let identity = VirtualTensor::identity(source, vec![3, 5], DType::BF16);
    assert!(whole_contiguous(&identity, &[3, 5], 15)?);
    assert!(whole_contiguous(&identity.reshape(vec![15])?, &[3, 5], 15)?);
    let transposed = identity.permute(&[1, 0])?;
    assert!(!whole_contiguous(&transposed, &[3, 5], 15)?);
    assert!(whole_contiguous(
        &transposed.permute(&[1, 0])?,
        &[3, 5],
        15
    )?);
    assert!(!whole_contiguous(&identity, &[4, 5], 15)?);
    assert!(!whole_contiguous(&identity, &[3, 5], 14)?);
    let broadcast =
        VirtualTensor::identity(source, vec![1, 5], DType::BF16).broadcast_axis(0, 3)?;
    assert!(!whole_contiguous(&broadcast, &[1, 5], 15)?);
    Ok(())
}

#[test]
fn raw_gradient_census_uses_v27_55m_policy_only() {
    let options = accepted_options();
    assert!(options.contiguous_bias_reshape_aliases);
    assert!(!options.online_attention_pair_numerical_screen);
    assert!(!options.attention_training_forward_fusion);
    assert!(!options.plain_bf16_output);
    assert!(!options.fused_nt128_bias);
    assert!(!options.ordinary_bf16_four_warp_block);
    assert!(!options.small_grid_split_k_bf16);
}

#[test]
#[ignore = "manual actual55M final-gradient metadata census plus one unchanged legacy update"]
fn raw_gradient_census_actual_55m_v27_bindings_and_executed_dispatch() -> Result<()> {
    if !super::tests::cuda_available() {
        return Ok(());
    }
    let output = PathBuf::from(
        std::env::var_os("TNSR_RAW_GRADIENT_CENSUS_OUTPUT")
            .context("set TNSR_RAW_GRADIENT_CENSUS_OUTPUT")?,
    );
    std::fs::create_dir_all(&output)?;
    ensure!(
        !output.join("census.json").exists(),
        "fresh census output required"
    );
    let archive = std::env::var_os("TNSR_RAW_GRADIENT_CENSUS_ARCHIVE")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(
                "target/transformer-training/parity-bias-reshape-w1024-s512/w1024_s512/archive",
            )
        });
    let model = super::test_decoder::Model::load(&archive)?;
    ensure!(
        model.config.width == 1024
            && model.config.sequence == 512
            && model.config.batch == 1
            && model.config.heads == 16
            && model.config.hidden == 2752
            && model.config.layers == 4
            && model.config.vocab == 4096,
        "census requires exact established55M config"
    );
    let (graph, _) = model.training_graph_with_options(crate::graph::GradientOptions {
        compact_softmax: true,
    })?;
    let mut owner = ResidentAdam::new_with_graph_replay(
        model
            .parameters
            .iter()
            .map(|(_, p)| (p.parameter.clone(), 0))
            .collect(),
        vec![AdamGroup::new(model.config.learning_rate)],
    )?;
    owner.set_update_batch_size(8)?;
    owner.set_experimental_first_accumulation_overwrite(true)?;
    owner.set_experimental_gradient_norm_partials(true)?;
    owner.begin_step()?;
    let mut executor =
        owner.mixed_executor_with_options(&graph, MatMulPrecision::Tf32x3, accepted_options())?;
    owner.accumulate_mixed(&mut executor, &graph, HashMap::new(), model.config.feeds()?)?;
    executor.synchronize()?;
    // All pointer observations occur while executor and its original owners are
    // alive. Only JSON metadata survives; no pointer is cached in ResidentAdam.
    let mut rows = Vec::new();
    let mut bf16_elements = 0_usize;
    let mut contiguous_bf16_elements = 0_usize;
    let mut total_elements = 0_usize;
    for (name, entry) in model.parameters.iter() {
        let id = entry.parameter.id();
        let node = *graph
            .gradient_metadata()
            .param_to_grad
            .get(&id)
            .context("parameter gradient absent")?;
        let (source, view, shape) = executor.device_binding(node)?;
        let record = model
            .records
            .iter()
            .find(|r| r.name == name)
            .context("parameter record absent")?;
        let len = record.count;
        ensure!(volume(&view.shape) == Some(len), "gradient volume mismatch");
        total_elements += len;
        let bytes = source
            .len
            .checked_mul(source.dtype.size_bytes())
            .context("source bytes overflow")?;
        let end = source
            .pointer
            .checked_add(u64::try_from(bytes)?)
            .context("source range overflow")?;
        let owners = executor.buffers.iter().enumerate().filter_map(|(index, buffer)| {
            let b = binding(buffer, &executor.stream);
            let bytes = b.len.checked_mul(b.dtype.size_bytes())?;
            let stop = b.pointer.checked_add(u64::try_from(bytes).ok()?)?;
            (source.dtype == b.dtype && source.pointer >= b.pointer && end <= stop).then(|| json!({
                "buffer_index":index,"dtype":format!("{:?}",b.dtype),
                "allocation_base":b.pointer,"allocation_end_exclusive":stop,
                "allocation_bytes":bytes,"source_offset_bytes":source.pointer-b.pointer,
                "source_spans_whole_allocation":source.pointer==b.pointer && end==stop,
                "ownership":"MixedExecutor.buffers owns CudaSlice directly; no Arc lease exported",
                "pool_handle_queried":false
            }))
        }).collect::<Vec<_>>();
        ensure!(
            owners.len() == 1,
            "gradient source not in exactly one live owned allocation"
        );
        let contiguous = whole_contiguous(&view, &shape, len)?;
        let whole_binding = source.len == len && volume(&shape) == Some(len);
        if source.dtype == DType::BF16 {
            bf16_elements += len;
            if contiguous && whole_binding {
                contiguous_bf16_elements += len;
            }
        }
        rows.push(json!({
            "name":name,"parameter_id":id,"elements":len,"parameter_shape":record.shape,
            "gradient_node":node.index(),"physical_dtype":format!("{:?}",source.dtype),
            "view_dtype":format!("{:?}",view.dtype),"view":format!("{view:?}"),
            "view_shape":view.shape,"source_shape":shape,"binding_elements":source.len,
            "pointer":source.pointer,"end_exclusive":end,
            "alignment":{"BF16":source.pointer.is_multiple_of(2),"16bytes":source.pointer.is_multiple_of(16),"128bytes":source.pointer.is_multiple_of(128)},
            "whole_binding_extent":whole_binding,"whole_contiguous_map_proved":contiguous,
            "final_node_is_graph_sink":graph.graph.neighbors_directed(node,petgraph::Direction::Outgoing).next().is_none(),
            "live_allocation_owners":owners
        }));
    }
    ensure!(
        rows.len() == 60 && total_elements == 55_373_312,
        "wrong actual parameter census"
    );
    // Materialize and execute the ORIGINAL owner dispatch, not a duplicated
    // predictor. This kernel path still reads the accumulated F32 gradient.
    let status = owner.step(model.config.max_grad_norm)?;
    executor.synchronize()?;
    let actual = owner.actual_adam_dispatch_for_census()?;
    let mut standalone_contiguous_bf16_elements = 0_usize;
    for row in &mut rows {
        let id = row["parameter_id"].as_u64().context("census id absent")?;
        let dispatch = actual
            .iter()
            .find(|d| u64::try_from(d.parameter_id).ok() == Some(id))
            .context("actual dispatch record absent")?;
        if !dispatch.batched
            && row["physical_dtype"] == "BF16"
            && row["whole_binding_extent"] == true
            && row["whole_contiguous_map_proved"] == true
        {
            standalone_contiguous_bf16_elements += dispatch.elements;
        }
        row["actual_legacy_adam_dispatch"] = serde_json::to_value(dispatch)?;
    }
    let report = json!({
        "scope":"Test-only actual55M V27 binding census after one real accumulation, before one unchanged F32-gradient legacy Adam update. No alternate Adam API, numerical qualification or performance claim.",
        "archive":archive,"config":model.config,"parameter_count":rows.len(),"parameter_elements":total_elements,
        "physical_bf16_elements":bf16_elements,"whole_contiguous_bf16_elements":contiguous_bf16_elements,
        "standalone_whole_contiguous_bf16_elements":standalone_contiguous_bf16_elements,
        "conditional_logical_gradient_read_reduction_bytes":standalone_contiguous_bf16_elements.checked_mul(2),
        "byte_scope":"Only hypothetical F32-to-BF16 Adam gradient-read width difference; no storage removed, no measured DRAM bytes or throughput prediction.",
        "actual_legacy_dispatch":actual,"actual_old_step_kernel_launches":status.kernel_launches,
        "norm_partials":owner.gradient_norm_partials_status(),"mixed_info":executor.info(),
        "rows":rows,"executor_alive_for_all_observations":true,"raw_gpu_lease_retained_by_owner":false,
        "operation_order":"Current accumulation G=+0+promote(raw), no owner micro_scale. Adam G*clip then unchanged decay/moment/variance/update arithmetic.",
        "lifetime_limit":"Graph sinks pinned only within live executor. Drop or re-execution invalidates raw-source assumptions. Existing step remains F32; any future borrowed API needs identity/run/upload generations and before-attempt invalidation."
    });
    std::fs::write(
        output.join("census.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    Ok(())
}
