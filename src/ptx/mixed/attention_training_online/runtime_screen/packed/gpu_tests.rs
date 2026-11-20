//! Whole real-decoder semantic proofs only; no timing or memory-saving claim.
use super::super::gpu_tests::{
    changed_feeds, options, published, same_bits, same_published, state_fixture_with_vocab,
    state_owner, state_owner_with_policies,
};
use super::*;
use crate::ptx::mixed::MixedOptions;

fn policy(selected: bool, direct: bool) -> MixedOptions {
    let mut p = options();
    p.online_attention_packed_backward = selected;
    p.direct_master_bindings = direct;
    p.fused_nt128_bias = true;
    p.contiguous_bias_reshape_aliases = true;
    p
}
fn coverage(e: &MixedExecutor, selected: bool, forced: bool) {
    assert_eq!(
        e.info().online_attention_pair_numerical_screen_pairs,
        4,
        "{:?}",
        e.info()
    );
    assert_eq!(
        e.info()
            .online_attention_pair_numerical_screen_fallback_pairs,
        0
    );
    assert_eq!(
        e.info().online_attention_packed_backward_regions,
        if selected && !forced { 4 } else { 0 },
        "{:?}",
        e.info()
    );
    assert_eq!(
        e.info().online_attention_packed_backward_fallback_regions,
        if selected && forced { 4 } else { 0 },
        "{:?}",
        e.info()
    );
    assert_eq!(
        e.info()
            .online_attention_packed_backward_pack_kernels_removed,
        if selected && !forced { 4 } else { 0 }
    );
    assert_eq!(
        e.info().qkv_grad_pack_regions,
        if selected && !forced { 0 } else { 4 }
    );
    assert_eq!(
        e.kernels
            .iter()
            .filter(|k| k.label.starts_with("QKV gradient pack fused"))
            .count(),
        if selected && !forced { 0 } else { 4 }
    );
}
fn save(name: &str, value: &impl serde::Serialize) -> Result<()> {
    if let Some(path) = std::env::var_os("TNSR_ONLINE_PACKED_RUNTIME_AUDIT") {
        let path = std::path::PathBuf::from(path);
        std::fs::create_dir_all(&path)?;
        std::fs::write(path.join(name), serde_json::to_vec_pretty(value)?)?;
    }
    Ok(())
}
#[test]
#[ignore = "manual CUDA actual B8 packed constructor ownership and fallback census"]
fn packed_runtime_actual_b8_four_regions_live_owner_alias_proof() -> Result<()> {
    if !crate::ptx::mixed::tests::cuda_available() {
        return Ok(());
    }
    let f = state_fixture_with_vocab(8, 4096)?;
    let owner = state_owner(&f, true)?;
    let e = owner.mixed_executor_with_options(
        &f.graph,
        crate::tile::MatMulPrecision::Tf32x3,
        policy(true, true),
    )?;
    let mut groups = HashMap::<NodeIndex, Vec<&Kernel>>::new();
    for kernel in &e.kernels {
        if kernel.label.contains("packed Q+delta") || kernel.label.contains("packed K+V") {
            groups
                .entry(
                    kernel
                        .owner
                        .context("packed kernel semantic owner absent")?,
                )
                .or_default()
                .push(kernel);
        }
    }
    let mut records = Vec::new();
    for (node, kernels) in groups {
        ensure!(
            kernels.len() == 2,
            "each packed owner requires Q and joint K/V"
        );
        let value = e.values.get(&node).context("packed output Value missing")?;
        let storage = super::owner(&e, value.binding).map_err(|r| anyhow::anyhow!("{r}"))?;
        for k in kernels {
            ensure!(
                k.pointers.last() == Some(&value.binding.pointer),
                "packed kernel target mismatch"
            );
            ensure!(
                exact_resources(&e, &k.function),
                "packed strict resource mismatch"
            );
            records.push(serde_json::json!({"node":node.index(),"role":k.label,"pointer":value.binding.pointer,"logical_len":value.binding.len,"owner_len":storage.len,"registers":k.function.num_regs()?,"shared":k.function.shared_size_bytes()?,"local":k.function.local_size_bytes()?,"capacity":k.function.occupancy_max_active_blocks_per_multiprocessor(256,0,None)?}));
        }
    }
    save(
        "admission-b8.json",
        &serde_json::json!({"info":e.info(),"targets":records,"scope":"constructor only; no first-update/model/performance qualification"}),
    )?;
    coverage(&e, true, false);
    Ok(())
}

type PublishedState = (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>);

struct Snapshot {
    loss: Vec<f32>,
    norm: u64,
    scale: u32,
    step: usize,
    gradients: Vec<Vec<f32>>,
    published: PublishedState,
    optimizations: serde_json::Value,
}
fn actual_state(batch: usize, forced: bool) -> Result<()> {
    actual_state_with_composition(batch, forced, false)
}
fn actual_state_with_composition(batch: usize, forced: bool, composed: bool) -> Result<()> {
    if !crate::ptx::mixed::tests::cuda_available() {
        return Ok(());
    }
    // B32 uses unique-token V16384 for exact scatter-order-independent state.
    // B8/B16 retain V4096. The archived V4096 full-model fixed gates are separate.
    let vocab = if batch == 32 { 16384 } else { 4096 };
    let mut expected = Vec::<Snapshot>::new();
    let mut records = Vec::new();
    for selected in [false, true] {
        let f = state_fixture_with_vocab(batch, vocab)?;
        let mut owner = if composed {
            state_owner_with_policies(&f, selected, true, true)?
        } else {
            state_owner(&f, selected)?
        };
        let guard = if selected && forced {
            Some(ForcedReject::install())
        } else {
            None
        };
        let mut selected_policy = policy(selected, selected);
        if composed {
            selected_policy.fused_nt128_bias_ldmatrix = true;
        }
        let mut e = owner.mixed_executor_with_options(
            &f.graph,
            crate::tile::MatMulPrecision::Tf32x3,
            selected_policy,
        )?;
        drop(guard);
        save(
            &format!("pre-state-b{batch}-selected{selected}-forced{forced}.json"),
            e.info(),
        )?;
        coverage(&e, selected, forced);
        if composed {
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
        let mut private_copies = 0;
        for (interval, micros) in [1, 2, 1].into_iter().enumerate() {
            if interval > 0 {
                owner.begin_step()?;
            }
            let mut loss = Vec::new();
            for micro in 0..micros {
                owner.accumulate_mixed(
                    &mut e,
                    &f.graph,
                    HashMap::from([(
                        "loss_seed".into(),
                        vec![1. + (interval + micro) as f32 / 8.],
                    )]),
                    changed_feeds(&f, interval + micro),
                )?;
                // Ordinary host read happens BEFORE optimizer publication.
                loss = e.read(f.loss)?;
                if selected {
                    assert_eq!(e.transfers().parameter_copies, 0);
                } else {
                    private_copies += e.transfers().parameter_copies;
                }
            }
            let step = if composed {
                owner.step_mixed(&e, 1.)?
            } else {
                owner.step(1.)?
            };
            let optimizations = if composed {
                let norm = owner.small_accumulation_norm_batches_status();
                assert!(norm.enabled);
                assert_eq!(norm.last_accumulation_used, micros == 1);
                assert_eq!(norm.selected_members, if micros == 1 { 38 } else { 0 });
                assert_eq!(norm.batched_launches, if micros == 1 { 5 } else { 0 });
                assert_eq!(norm.actual_launches, if micros == 1 { 27 } else { 60 });
                let borrowed = owner.borrowed_bf16_adam_status();
                assert!(borrowed.enabled);
                assert_eq!(
                    borrowed.standalone_bf16_parameters,
                    if micros == 1 && batch == 8 { 16 } else { 0 }
                );
                serde_json::json!({"smallnorm":norm,"borrowed":borrowed})
            } else {
                serde_json::Value::Null
            };
            let gradients = owner.gradients()?;
            let gradients = f
                .parameters
                .iter()
                .map(|p| gradients[&p.id()].clone())
                .collect::<Vec<_>>();
            let state = published(&owner, &f)?;
            if selected {
                let old = &expected[interval];
                same_bits(&old.loss, &loss);
                assert_eq!(old.norm, step.gradient_norm.to_bits());
                assert_eq!(old.scale, step.gradient_scale.to_bits());
                assert_eq!(old.step, step.timestep);
                for (a, b) in old.gradients.iter().zip(&gradients) {
                    same_bits(a, b);
                }
                same_published(&old.published, &state);
                assert_eq!(old.optimizations, optimizations);
            } else {
                expected.push(Snapshot {
                    loss,
                    norm: step.gradient_norm.to_bits(),
                    scale: step.gradient_scale.to_bits(),
                    step: step.timestep,
                    gradients,
                    published: state,
                    optimizations,
                });
            }
            if interval == 1 {
                let before = published(&owner, &f)?;
                owner.begin_step()?;
                let _ = owner.accumulate_mixed(
                    &mut e,
                    &f.graph,
                    HashMap::from([("loss_seed".into(), vec![f32::NAN])]),
                    changed_feeds(&f, interval),
                );
                let failed = if composed {
                    owner.step_mixed(&e, 1.)
                } else {
                    owner.step(1.)
                };
                assert!(failed.is_err(), "NaN update must reject publication");
                same_published(&before, &published(&owner, &f)?);
            }
        }
        if selected {
            assert_eq!(e.info().direct_capture_variants, 2);
            assert_eq!(e.info().direct_capture_misses, 2);
            assert_eq!(e.transfers().parameter_copies, 0);
        } else {
            assert!(private_copies > 0);
        }
        records.push(serde_json::json!({"selected":selected,"info":e.info(),"composed_smallnorm":composed,"composed_borrowed_step":composed,"final_smallnorm_status":owner.small_accumulation_norm_batches_status(),"final_borrowed_status":owner.borrowed_bf16_adam_status(),"normal_loss_read_before_step":true,"intervals":[1,2,1],"nan_rollback_recovery":true,"loss_norm_scale_clipped_gradient_weights_moments_exact":true}));
    }
    save(
        &if composed {
            format!("state-composed-b{batch}.json")
        } else {
            format!("state-b{batch}-forced{forced}.json")
        },
        &serde_json::json!({"batch":batch,"vocab":vocab,"records":records,"scope":"old complete online pair+pack versus new packed policy; B32 unique-token fixture; no throughput or allocation saving claim"}),
    )?;
    Ok(())
}
#[test]
#[ignore = "manual CUDA actual B8 old pair+pack vs packed exact full state"]
fn packed_runtime_actual_b8_exact_state_replay_and_recovery() -> Result<()> {
    actual_state(8, false)
}
#[test]
#[ignore = "manual CUDA actual B16 old pair+pack vs packed exact full state"]
fn packed_runtime_actual_b16_exact_state_replay_and_recovery() -> Result<()> {
    actual_state(16, false)
}
#[test]
#[ignore = "manual CUDA B32 unique-token exact full state; separate archived gates required"]
fn packed_runtime_actual_b32_unique_tokens_exact_state_replay_and_recovery() -> Result<()> {
    actual_state(32, false)
}
#[test]
#[ignore = "manual CUDA forced packed rejection preserves complete old pair+pack"]
fn packed_runtime_actual_b8_forced_rejection_keeps_full_old_pair_and_pack() -> Result<()> {
    actual_state(8, true)
}

#[test]
#[ignore = "manual CUDA B8 current accepted smallnorm+borrowed composition, full old/packed state"]
fn packed_runtime_actual_b8_accepted_smallnorm_borrowed_composition_exact_state() -> Result<()> {
    actual_state_with_composition(8, false, true)
}
#[test]
#[ignore = "manual CUDA B32 unique-token accepted smallnorm+12LDM composition, full old/packed state"]
fn packed_runtime_actual_b32_accepted_smallnorm_ldmatrix_composition_exact_state() -> Result<()> {
    actual_state_with_composition(32, false, true)
}
