//! Real decoder proofs reuse the existing unique-token fixture; no timing.
use super::*;
use crate::tile::SplitKBf16TilePolicy as Policy;
fn selected_options(rectangle: bool, forced: bool) -> MixedOptions {
    let mut p = options();
    p.split_k_bf16_tile_policy = if rectangle {
        Policy::Rectangle128x64
    } else {
        Policy::Square64
    };
    p.split_k_rectangle_test_resource_rejection = forced;
    p.online_attention_packed_backward = true;
    p.fused_nt128_bias = true;
    p.fused_nt128_bias_ldmatrix = true;
    p.contiguous_bias_reshape_aliases = true;
    p.direct_master_bindings = true;
    p
}
fn required_cuda() -> Result<()> {
    // Manual proofs must never succeed by silently returning without a device.
    ensure!(
        std::env::var_os("TNSR_REQUIRE_CUDA").is_some(),
        "set TNSR_REQUIRE_CUDA=1 for manual decoder proof"
    );
    cudarc::driver::CudaContext::new(0)?;
    Ok(())
}
fn save(name: &str, value: &impl serde::Serialize) -> Result<()> {
    let root = std::path::PathBuf::from(
        std::env::var_os("TNSR_SPLIT128_MODEL_DIR")
            .context("set fresh split128 model proof directory")?,
    );
    std::fs::create_dir_all(&root)?;
    let path = root.join(name);
    ensure!(!path.exists(), "preserve existing model proof artifact");
    std::fs::write(path, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}
#[test]
#[ignore = "manual requiredCUDA actual V4096 B32 census and forced wholeSquare64 fallback"]
fn split128_actual_b32_v4096_census_and_forced_resource_fallback() -> Result<()> {
    required_cuda()?;
    let mut records = vec![];
    let mut baseline_payload = None;
    for (rectangle, forced) in [(false, false), (true, false), (true, true)] {
        let f = state_fixture_with_vocab(32, 4096)?;
        let owner = state_owner_with_policies(&f, true, true, false)?;
        let e = owner.mixed_executor_with_options(
            &f.graph,
            crate::tile::MatMulPrecision::Tf32x3,
            selected_options(rectangle, forced),
        )?;
        let info = e.info();
        let expected_rect = if rectangle && !forced { 17 } else { 0 };
        assert_eq!(
            info.split_k_rectangle128x64_programs, expected_rect,
            "{info:?}"
        );
        assert_eq!(
            info.split_k_square64_programs,
            17 - expected_rect,
            "{info:?}"
        );
        assert_eq!(info.split_k_square64_fallback_programs, 4);
        assert_eq!(
            info.split_k_rectangle128x64_fallback_programs,
            if rectangle { 21 - expected_rect } else { 0 }
        );
        assert_eq!(
            info.split_k_rectangle128x64_rejections.len(),
            info.split_k_rectangle128x64_fallback_programs
        );
        if forced {
            assert_eq!(
                info.split_k_rectangle128x64_rejections
                    .iter()
                    .filter(|s| s.contains("forced rectangle JIT resource rejection"))
                    .count(),
                17
            );
        }
        let payload = (info.kernels, info.f32_bytes, info.bf16_bytes);
        if let Some(old) = baseline_payload {
            assert_eq!(old, payload, "no new globaltensorpayload/kernel count");
        } else {
            baseline_payload = Some(payload);
        }
        let selected=e.kernels.iter().filter(|k|k.label.contains("rectangle128x64=true")||k.label.contains("square64=true")).map(|k|->Result<_>{
            let rectangle=k.label.contains("rectangle128x64=true");assert_eq!(k.config.block_dim,(32,8,1));assert_eq!(k.pointers.len(),3);
            assert_eq!(k.function.shared_size_bytes()?,if rectangle{32768}else{18432});assert_eq!(k.function.local_size_bytes()?,0);assert!(k.function.num_regs()?>0 && k.function.num_regs()?<=128);assert!(k.function.max_threads_per_block()?>=256);assert!(k.function.occupancy_max_active_blocks_per_multiprocessor(256,0,None)?>=2);
            Ok(serde_json::json!({"label":k.label,"grid":k.config.grid_dim,"block":k.config.block_dim,"pointers":k.pointers,"registers":k.function.num_regs()?,"shared":k.function.shared_size_bytes()?,"local":k.function.local_size_bytes()?,"max_threads":k.function.max_threads_per_block()?,"capacity":k.function.occupancy_max_active_blocks_per_multiprocessor(256,0,None)?}))
        }).collect::<Result<Vec<_>>>()?;
        records.push(serde_json::json!({"requested_rectangle":rectangle,"forced":forced,"info":info,"selected":selected}));
        e.synchronize()?;
        drop(e);
        drop(owner);
        drop(f);
    }
    save(
        "actual-b32-v4096-census.json",
        &serde_json::json!({"qualified":true,"batch":32,"vocab":4096,"selected_rectangles":17,"scope":"Actual V4096 B32 constructor/census and forcedlate-resourcefallback only; no model update or timing proof","rows":records}),
    )
}
struct Snapshot {
    loss: Vec<f32>,
    norm: u64,
    scale: u32,
    step: usize,
    gradients: Vec<Vec<f32>>,
    published: PublishedState,
}
#[test]
#[ignore = "manual requiredCUDA B32 V16384 unique-token full state replay/empty/NaN/recovery"]
fn split128_actual_b32_unique_tokens_full_state_replay_empty_nan_and_recovery() -> Result<()> {
    required_cuda()?;
    let mut expected = Vec::<Snapshot>::new();
    let mut records = vec![];
    for rectangle in [false, true] {
        let f = state_fixture_with_vocab(32, 16384)?;
        for feed in f.feeds.values() {
            assert_eq!(feed.len(), 16384);
            assert_eq!(
                feed.iter().collect::<std::collections::HashSet<_>>().len(),
                16384
            );
        }
        let mut owner = state_owner_with_policies(&f, true, true, false)?;
        let mut e = owner.mixed_executor_with_options(
            &f.graph,
            crate::tile::MatMulPrecision::Tf32x3,
            selected_options(rectangle, false),
        )?;
        // V16384 removes the V4096 tied-head split geometry. This proof selects
        // 16 rectangle scopes; the separate V4096 census proves all17.
        assert_eq!(
            e.info().split_k_rectangle128x64_programs,
            if rectangle { 16 } else { 0 },
            "{:?}",
            e.info()
        );
        assert_eq!(
            e.info().split_k_square64_programs,
            if rectangle { 0 } else { 16 }
        );
        assert_eq!(e.info().split_k_square64_fallback_programs, 4);
        let mut rows = vec![];
        let mut accepted_steps = 0usize;
        for (interval, micros) in [1, 2, 1, 0].into_iter().enumerate() {
            if interval > 0 {
                owner.begin_step()?;
            }
            let mut loss = vec![];
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
                loss = e.read(f.loss)?;
                assert_eq!(e.transfers().parameter_copies, 0);
            }
            let step = owner.step(1.)?;
            accepted_steps += 1;
            assert_eq!(step.timestep, accepted_steps);
            let gradients = owner.gradients()?;
            let state = Snapshot {
                loss,
                norm: step.gradient_norm.to_bits(),
                scale: step.gradient_scale.to_bits(),
                step: step.timestep,
                gradients: f
                    .parameters
                    .iter()
                    .map(|p| gradients[&p.id()].clone())
                    .collect(),
                published: published(&owner, &f)?,
            };
            if rectangle {
                let old = &expected[interval];
                same_bits(&old.loss, &state.loss);
                assert_eq!(
                    (old.norm, old.scale, old.step),
                    (state.norm, state.scale, state.step)
                );
                for (a, b) in old.gradients.iter().zip(&state.gradients) {
                    same_bits(a, b);
                }
                same_published(&old.published, &state.published);
            } else {
                expected.push(state);
            }
            if interval == 2 {
                assert_eq!(e.info().direct_capture_variants, 2);
                assert_eq!(e.info().direct_capture_misses, 2);
            }
            rows.push(serde_json::json!({"interval":interval,"microbatches":micros,"loss_norm_scale_gradients_masters_moments_exact":true,"timestep":step.timestep,"capture_variants":e.info().direct_capture_variants,"capture_misses":e.info().direct_capture_misses}));
            if interval == 1 {
                let before = published(&owner, &f)?;
                owner.begin_step()?;
                let _ = owner.accumulate_mixed(
                    &mut e,
                    &f.graph,
                    HashMap::from([("loss_seed".into(), vec![f32::NAN])]),
                    changed_feeds(&f, interval),
                );
                assert!(owner.step(1.).is_err(), "NaNhealth must reject");
                same_published(&before, &published(&owner, &f)?);
            }
        }
        records.push(serde_json::json!({"rectangle":rectangle,"info":e.info(),"intervals":rows,"NaN_rollback_and_finite_recovery":true}));
        e.synchronize()?;
        drop(e);
        drop(owner);
        drop(f);
    }
    save(
        "actual-b32-v16384-unique-state.json",
        &serde_json::json!({"qualified":true,"batch":32,"vocab":16384,"selected_rectangles":16,"scope":"Synthetic actual decoder B32 V16384 unique-token full-state proof, separate from measured V4096 fixed numerical gates. Botharmscaptured/direct withacceptedpacked/NTLD/smallnorm; onlySplitKpolicy differs. No timing/adoption claim.","rows":records,"microbatches":[1,2,1,0],"borrowed_Adam":false}),
    )
}
