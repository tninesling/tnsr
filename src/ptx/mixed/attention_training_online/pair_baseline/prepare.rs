//! Closed BF16 component feeds: keep real Mixed lowering and remove only proved
//! host-F32 identity narrowing setup kernels, not any attention computation.
use super::*;
use crate::{
    ptx::{
        linear::{Binding, program::Kernel},
        mixed::{MixedExecutor, MixedOptions},
    },
    tile::{
        BatchedAttentionBf16TilePolicy, ColumnReductionPolicy, DType, MatMulPrecision,
        MediumBf16TilePolicy, VirtualTensor,
    },
};

pub(super) struct Prepared {
    // Own every module/allocation underlying cloned kernels through capture.
    pub owner: MixedExecutor,
    pub kernels: Vec<Kernel>,
    pub metadata: serde_json::Value,
}

fn end(binding: Binding) -> Result<u64> {
    let bytes = binding
        .len
        .checked_mul(match binding.dtype {
            DType::F32 => 4,
            DType::BF16 => 2,
            _ => anyhow::bail!("unsupported component binding dtype"),
        })
        .context("component binding extent overflow")?;
    binding
        .pointer
        .checked_add(u64::try_from(bytes)?)
        .context("component pointer extent overflow")
}
fn overlaps(a: Binding, b: Binding) -> Result<bool> {
    Ok(a.pointer < end(b)? && b.pointer < end(a)?)
}
fn inside(pointer: u64, binding: Binding) -> Result<bool> {
    Ok(pointer >= binding.pointer && pointer < end(binding)?)
}

pub(super) fn prepare(
    shape: Shape,
    graph: Graph,
    exact_checkpoint_forward: bool,
    inputs: [Binding; 4],
    outputs: [Binding; 4],
) -> Result<Prepared> {
    let validation_only = graph.construction["validation_only"] == true;
    ensure!(
        !exact_checkpoint_forward || shape.bh() >= 128,
        "accepted four-checkpoint component baseline requires BH>=128"
    );
    for (index, value) in inputs.iter().chain(&outputs).enumerate() {
        ensure!(
            value.dtype == DType::BF16
                && value.len == shape.elements()
                && value.pointer != 0
                && value.pointer.is_multiple_of(2),
            "component external binding{index} is not a full aligned BF16 tensor"
        );
        let _ = end(*value)?;
    }
    let external: Vec<_> = inputs.iter().chain(&outputs).copied().collect();
    for (i, &a) in external.iter().enumerate() {
        for &b in &external[i + 1..] {
            ensure!(
                !overlaps(a, b)?,
                "component inputs/outputs must have disjoint byte ranges"
            );
        }
    }
    let mut owner = MixedExecutor::new_with_options(
        &graph.graph,
        MatMulPrecision::Tf32x3,
        false,
        MixedOptions {
            attention_training_forward_fusion: exact_checkpoint_forward,
            attention_softmax_fusion: true,
            attention_softmax_backward_fusion: true,
            shared_memory_reuse: true,
            narrow_bf16_tile: true,
            physical_order_casts: true,
            transposed_bf16_outputs: true,
            plain_bf16_output: true,
            medium_bf16_tile_policy: MediumBf16TilePolicy::SquareThenTall,
            batched_attention_bf16_tile_policy: BatchedAttentionBf16TilePolicy::TallScoresByGrid,
            column_reduction_policy: ColumnReductionPolicy::Four,
            ..Default::default()
        },
    )?;
    ensure!(
        owner.info().attention_training_forward_regions == usize::from(exact_checkpoint_forward),
        "requested old-native baseline must actually select its declared forward policy"
    );
    let mut original_inputs = Vec::new();
    let mut original_sources = Vec::new();
    let mut removed = Vec::new();
    for (slot, &node) in graph.inputs.iter().enumerate() {
        ensure!(
            matches!(
                graph.graph[node],
                TensorGraphNode::Cast {
                    dtype: ValueDType::BF16,
                    ..
                }
            ),
            "component feed must be the original identity BF16 cast boundary"
        );
        let [source]: [NodeIndex; 1] = graph
            .graph
            .inputs(node)
            .try_into()
            .map_err(|_| anyhow::anyhow!("component cast must have one source"))?;
        ensure!(
            matches!(graph.graph[source], TensorGraphNode::Input { .. }),
            "component feed cast must narrow a plain host-F32 input"
        );
        let value = owner
            .values
            .get(&node)
            .context("component cast value missing")?;
        let source_value = owner
            .values
            .get(&source)
            .context("component input source missing")?;
        let expected =
            VirtualTensor::identity(node, graph.graph[node].shape().clone(), DType::BF16);
        ensure!(
            value.binding.dtype == DType::BF16
                && value.binding.len == shape.elements()
                && value.source_shape == *graph.graph[node].shape()
                && value.view == expected
                && value.chain.is_empty(),
            "component input cast must be a full contiguous identity materialization"
        );
        ensure!(
            source_value.binding.dtype == DType::F32
                && source_value.binding.len == shape.elements(),
            "component input source extent/dtype mismatch"
        );
        ensure!(
            source_value.view
                == VirtualTensor::identity(source, graph.graph[source].shape().clone(), DType::F32)
                && source_value.chain.is_empty()
                && source_value.source_shape == *graph.graph[source].shape(),
            "component original F32 feed must have no hidden indexing/alias operation"
        );
        // Cast materialization currently carries owner=None, so identity is
        // proved from the closed graph operation plus both exact bindings and
        // the complete pointwise ABI/launch, never a label match.
        let owned: Vec<_> = owner
            .kernels
            .iter()
            .enumerate()
            .filter(|(_, k)| k.pointers == [source_value.binding.pointer, value.binding.pointer])
            .collect();
        ensure!(
            owned.len() == 1,
            "input conversion must be exactly one owned pointwise kernel"
        );
        let (position, kernel) = owned[0];
        let expected_launch =
            crate::ptx::target::pointwise_launch_config(u32::try_from(shape.elements())?);
        ensure!(
            kernel.owner.is_none_or(|owner| owner == node)
                && kernel.config.block_dim == expected_launch.block_dim
                && kernel.config.grid_dim == expected_launch.grid_dim
                && kernel.config.shared_mem_bytes == expected_launch.shared_mem_bytes
                && kernel.zero_before.is_none()
                && kernel.pointers.len() == 2
                && kernel.pointers[0] == source_value.binding.pointer
                && kernel.pointers[1] == value.binding.pointer,
            "input conversion ABI must be exactly source-F32 then identity-BF16 output"
        );
        if !validation_only {
            removed.push(position);
        }
        original_inputs.push((value.binding, inputs[slot]));
        original_sources.push(source_value.binding);
    }
    let mut remaps = if validation_only {
        Vec::new()
    } else {
        original_inputs.clone()
    };
    for (slot, &node) in graph.outputs.iter().enumerate() {
        let value = owner
            .values
            .get(&node)
            .context("component output value missing")?;
        ensure!(
            value.binding.dtype == DType::BF16
                && value.binding.len == shape.elements()
                && value.view
                    == VirtualTensor::identity(
                        node,
                        graph.graph[node].shape().clone(),
                        DType::BF16
                    )
                && value.source_shape == *graph.graph[node].shape(),
            "component output must preserve its original dense typed MatMul layout"
        );
        remaps.push((value.binding, outputs[slot]));
    }
    for (i, &(a, _)) in remaps.iter().enumerate() {
        for &(b, _) in &remaps[i + 1..] {
            ensure!(
                !overlaps(a, b)?,
                "original component boundary buffers unexpectedly alias"
            );
        }
    }
    let mut kernels = Vec::new();
    for (position, original) in owner.kernels.iter().enumerate() {
        if removed.contains(&position) {
            continue;
        }
        let mut kernel = original.clone();
        for &source in original_sources.iter().filter(|_| !validation_only) {
            ensure!(
                !kernel
                    .pointers
                    .iter()
                    .any(|&p| inside(p, source).unwrap_or(true)),
                "surviving attention kernel still reads original host-F32 feed storage"
            );
            if let Some(zero) = kernel.zero_before {
                ensure!(
                    !overlaps(zero, source)?,
                    "surviving zero targets removed F32 input storage"
                );
            }
        }
        if let Some(zero) = kernel.zero_before {
            for &(input, _) in &original_inputs {
                ensure!(
                    !overlaps(zero, input)?,
                    "baseline must never zero read-only BF16 input aliases"
                );
            }
        }
        for pointer in &mut kernel.pointers {
            for &(old, replacement) in &remaps {
                if inside(*pointer, old)? {
                    ensure!(
                        *pointer == old.pointer,
                        "only full-volume exact-base boundary aliases can remap"
                    );
                    *pointer = replacement.pointer;
                    break;
                }
            }
        }
        if let Some(zero) = &mut kernel.zero_before {
            for &(old, replacement) in &remaps {
                if overlaps(*zero, old)? {
                    ensure!(
                        zero.pointer == old.pointer
                            && zero.len == old.len
                            && zero.dtype == old.dtype,
                        "partial zero-output alias cannot remap"
                    );
                    *zero = replacement;
                    break;
                }
            }
        }
        kernels.push(kernel);
    }
    // Initialize immutable mask/scalar sources on the original stream outside
    // capture. Replaced input buffers are never read by retained kernels.
    owner.upload(
        ["q", "k", "v", "do"]
            .into_iter()
            .map(|name| (name.into(), vec![0.0_f32; shape.elements()]))
            .collect(),
    )?;
    owner.execution_stream().synchronize()?;
    let mut groups = Vec::new();
    for (&node, &role) in &graph.roles {
        let count = kernels
            .iter()
            .filter(|kernel| kernel.owner == Some(node))
            .count();
        if exact_checkpoint_forward && role == "QK" {
            ensure!(
                count == 0,
                "fused checkpoint baseline must remove the whole original QK group"
            );
        } else {
            ensure!(
                count > 0,
                "complete old baseline is missing required {role} kernel group"
            );
        }
        groups
            .push(serde_json::json!({"node":node.index(),"role":role,"actual_kernel_count":count}));
    }
    groups.sort_by_key(|group| group["node"].as_u64());
    let mut rows = Vec::new();
    for kernel in &kernels {
        let threads =
            kernel.config.block_dim.0 * kernel.config.block_dim.1 * kernel.config.block_dim.2;
        rows.push(serde_json::json!({"owner":kernel.owner.map(NodeIndex::index),
            "role":kernel.owner.and_then(|n| graph.roles.get(&n).copied()).unwrap_or("normalization/view/other-required"),
            "operation":kernel.owner.map(|n| graph.graph[n].name()),"label":kernel.label,
            "grid":kernel.config.grid_dim,"block":kernel.config.block_dim,
            "registers":kernel.function.num_regs()?,"static_shared":kernel.function.shared_size_bytes()?,
            "local":kernel.function.local_size_bytes()?,"dynamic_shared":kernel.config.shared_mem_bytes,
            "api_capacity":kernel.function.occupancy_max_active_blocks_per_multiprocessor(threads,usize::try_from(kernel.config.shared_mem_bytes)?,None)?}));
    }
    let metadata = serde_json::json!({"construction":graph.construction,"exact_checkpoint_forward":exact_checkpoint_forward,
        "removed_input_identity_cast_kernels":removed.len(),"kernel_count":kernels.len(),"kernels":rows,
        "actual_contraction_groups":groups,
        "input_casts_retained_for_independent_untimed_reference":validation_only,
        "owned_f32_bytes":owner.info().f32_bytes,"owned_bf16_bytes":owner.info().bf16_bytes,
        "owned_input_source_f32_bytes":4*shape.elements()*4,
        "bypassed_owned_input_source_f32_bytes":if validation_only {0}else{4*shape.elements()*4},
        "plain_bf16_output_programs":owner.info().plain_bf16_output_programs,
        "attention_softmax_regions":owner.info().attention_softmax_regions,
        "attention_softmax_backward_regions":owner.info().attention_softmax_backward_regions,
        "attention_training_forward_regions":owner.info().attention_training_forward_regions,
        "batched_scores":owner.info().batched_attention_score_programs,
        "batched_query_width":owner.info().batched_attention_query_width_programs,
        "batched_width_query":owner.info().batched_attention_width_query_programs,
        "ordinary_four_warp":owner.info().ordinary_bf16_four_warp_programs,
        "large_policy":"off","capture_scope":"all retained native kernels; no source conversion/loss/seed in pruned timing baseline",
        "allocator_scope":"nonresident component retains all original allocations, including bypassed source buffers; not a model-memory estimate"});
    Ok(Prepared {
        owner,
        kernels,
        metadata,
    })
}
