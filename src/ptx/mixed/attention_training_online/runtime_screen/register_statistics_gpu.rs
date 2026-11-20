//! Synthetic unique-token state proof, separate from archived V4096 model gates.
use super::*;

#[test]
#[ignore = "manual mandatory-CUDA B32 V16384 Global/Register/forced-Global full-state replay proof"]
fn online_register_statistics_actual_b32_three_modes_exact_state_recovery() -> Result<()> {
    anyhow::ensure!(
        std::env::var("TNSR_REQUIRE_CUDA").as_deref() == Ok("1"),
        "TNSR_REQUIRE_CUDA=1 is mandatory"
    );
    struct Interval {
        loss: Vec<f32>,
        norm: u64,
        scale: u32,
        timestep: usize,
        gradients: Vec<Vec<f32>>,
        published: PublishedState,
    }
    let mut expected: Vec<Interval> = Vec::new();
    let mut summary = Vec::new();
    let mut allocation = None;
    for mode in 0..3 {
        let f = state_fixture_with_vocab(32, 16384)?;
        for feed in f.feeds.values() {
            assert_eq!(feed.len(), 16384);
            assert_eq!(
                feed.iter().collect::<std::collections::HashSet<_>>().len(),
                16384
            );
        }
        let mut owner = state_owner_with_smallnorm(&f, true, true)?;
        let mut policy = options();
        policy.split_k_bf16_tile_policy = crate::tile::SplitKBf16TilePolicy::Rectangle128x64;
        policy.fused_nt128_bias = true;
        policy.fused_nt128_bias_ldmatrix = true;
        policy.contiguous_bias_reshape_aliases = true;
        policy.online_attention_packed_backward = true;
        policy.online_attention_register_statistics = mode != 0;
        policy.force_online_attention_register_statistics_rejection = mode == 2;
        policy.direct_master_bindings = true;
        if mode == 0 {
            let mut invalid = policy;
            invalid.online_attention_pair_numerical_screen = false;
            invalid.online_attention_register_statistics = true;
            assert!(
                owner
                    .mixed_executor_with_options(
                        &f.graph,
                        crate::tile::MatMulPrecision::Tf32x3,
                        invalid
                    )
                    .is_err(),
                "register statistics requires online pair"
            );
        }
        let mut e = owner.mixed_executor_with_options(
            &f.graph,
            crate::tile::MatMulPrecision::Tf32x3,
            policy,
        )?;
        assert_eq!(
            e.info().online_attention_pair_numerical_screen_pairs,
            4,
            "{:?}",
            e.info()
        );
        assert_eq!(
            e.info()
                .online_attention_pair_numerical_screen_async16_pairs,
            4
        );
        assert_eq!(e.info().online_attention_packed_backward_regions, 4);
        assert_eq!(e.info().split_k_rectangle128x64_programs, 16);
        assert_eq!(e.info().split_k_square64_programs, 0);
        assert_eq!(e.info().split_k_square64_fallback_programs, 4);
        assert_eq!(e.info().split_k_rectangle128x64_fallback_programs, 4);
        assert_eq!(
            e.info().online_attention_packed_backward_fallback_regions,
            0
        );
        assert_eq!(
            e.info().online_attention_register_statistics_pairs,
            if mode == 1 { 4 } else { 0 },
            "{:?}",
            e.info()
        );
        assert_eq!(
            e.info().online_attention_global_statistics_pairs,
            if mode == 1 { 0 } else { 4 }
        );
        assert_eq!(
            e.info().online_attention_register_statistics_fallback_pairs,
            if mode == 2 { 4 } else { 0 }
        );
        if mode == 2 {
            assert!(
                e.info()
                    .online_attention_register_statistics_rejections
                    .iter()
                    .all(|r| r.contains("Resources"))
            );
        }
        assert_eq!(
            e.info()
                .online_attention_pair_numerical_screen_workspace_bytes,
            4 * 256 * 512 * 3 * 4
        );
        let bytes = (
            e.info().f32_bytes,
            e.info().bf16_bytes,
            e.info()
                .online_attention_pair_numerical_screen_workspace_bytes,
        );
        if let Some(old) = allocation {
            assert_eq!(
                old, bytes,
                "statistics mode must not alter allocation accounting"
            );
        } else {
            allocation = Some(bytes);
        }
        let mut rows = Vec::new();
        let mut accepted_steps = 0usize;
        for (interval, microbatches) in [1, 2, 1, 0].into_iter().enumerate() {
            if interval > 0 {
                owner.begin_step()?;
            }
            let mut loss = Vec::new();
            for micro in 0..microbatches {
                owner.accumulate_mixed(
                    &mut e,
                    &f.graph,
                    HashMap::from([(
                        "loss_seed".into(),
                        vec![1. + (interval + micro) as f32 / 8.],
                    )]),
                    changed_feeds(&f, interval + micro),
                )?;
                loss = e.read(f.loss)?;
                assert_eq!(e.transfers().parameter_copies, 0);
            }
            let step = owner.step(1.)?;
            accepted_steps += 1;
            assert_eq!(step.timestep, accepted_steps);
            let gradients = owner.gradients()?;
            let state = Interval {
                loss,
                norm: step.gradient_norm.to_bits(),
                scale: step.gradient_scale.to_bits(),
                timestep: step.timestep,
                gradients: f
                    .parameters
                    .iter()
                    .map(|p| gradients[&p.id()].clone())
                    .collect(),
                published: published(&owner, &f)?,
            };
            if mode == 0 {
                expected.push(state);
            } else {
                let old = &expected[interval];
                same_bits(&old.loss, &state.loss);
                assert_eq!(
                    (old.norm, old.scale, old.timestep),
                    (state.norm, state.scale, state.timestep)
                );
                for (a, b) in old.gradients.iter().zip(&state.gradients) {
                    same_bits(a, b);
                }
                same_published(&old.published, &state.published);
            }
            if interval == 2 {
                assert_eq!(e.info().direct_capture_variants, 2);
                assert_eq!(e.info().direct_capture_misses, 2);
            }
            rows.push(serde_json::json!({"interval":interval,"microbatches":microbatches,"accepted_timestep":step.timestep,"loss_norm_scale_gradients_masters_both_moments_exact":true}));
            if interval == 1 {
                let before = published(&owner, &f)?;
                owner.begin_step()?;
                let _ = owner.accumulate_mixed(
                    &mut e,
                    &f.graph,
                    HashMap::from([("loss_seed".into(), vec![f32::NAN])]),
                    changed_feeds(&f, interval),
                );
                assert!(owner.step(1.).is_err());
                same_published(&before, &published(&owner, &f)?);
            }
        }
        assert_eq!(e.transfers().parameter_copies, 0);
        let mode_name = [
            "Global captured/direct",
            "Register captured/direct",
            "forced fallback Global captured/direct",
        ][mode];
        summary.push(serde_json::json!({"mode":mode_name,"info":e.info(),"intervals":rows}));
        e.synchronize()?;
        drop(e);
        drop(owner);
        drop(f);
    }
    if let Some(dir) = std::env::var_os("TNSR_ONLINE_SCREEN_AUDIT") {
        std::fs::create_dir_all(&dir)?;
        std::fs::write(
            std::path::PathBuf::from(dir)
                .join("exact-state-register-statistics-b32-unique-v16384.json"),
            serde_json::to_vec_pretty(
                &serde_json::json!({"scope":"Synthetic B32 V16384 unique-token Rect128x64 three-arm complete-state proof with empty update; distinct from archived V4096 fixed gates. Captured Global versus captured Register and forced-resource-fallback Global, unchanged packed backward4; no timing/performance/parity claim.","batch":32,"vocab":16384,"intervals":[1,2,1,0],"selected_rectangles_per_arm":16,"old_square64_fallbacks_per_arm":4,"nan_rollback_recovery":true,"statistics_allocation_bytes":4*256*512*3*4,"rows":summary}),
            )?,
        )?;
    }
    Ok(())
}
