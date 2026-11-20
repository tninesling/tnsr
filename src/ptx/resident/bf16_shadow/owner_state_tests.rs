//! Actual evolving decoder state; no performance or archived first-update claims.
use super::*;
use cudarc::driver::DevicePtr;
use half::bf16;

struct Arm {
    fixture: Fixture,
    owner: ResidentAdam,
    executor: MixedExecutor,
}
fn arm(shadows: bool) -> Result<Arm> {
    let mut cfg = config(false);
    cfg.sequence = 128;
    cfg.batch = 2;
    let fixture = fixture(cfg)?;
    let options = MixedOptions {
        small_grid_split_k_bf16: true,
        ..options(false, true)
    };
    let mut owner = ResidentAdam::new_with_graph_replay(
        fixture
            .model
            .parameters
            .iter()
            .map(|(_, p)| (p.parameter.clone(), 0))
            .collect(),
        vec![AdamGroup::new(fixture.model.config.learning_rate)],
    )?;
    owner.set_update_batch_size(8)?;
    owner.set_experimental_first_accumulation_overwrite(true)?;
    owner.set_experimental_gradient_norm_partials(true)?;
    owner.set_experimental_borrowed_bf16_adam_source(true)?;
    owner.set_experimental_vector_bf16_first_adam(true)?;
    owner.set_experimental_small_accumulation_norm_batches(true)?;
    if shadows {
        owner.declare_experimental_bf16_master_shadows(&fixture.graph, options)?;
    }
    owner.begin_step()?;
    let executor =
        owner.mixed_executor_with_options(&fixture.graph, MatMulPrecision::Tf32x3, options)?;
    Ok(Arm {
        fixture,
        owner,
        executor,
    })
}
fn token_id(arm: &Arm) -> Result<usize> {
    Ok(arm
        .fixture
        .model
        .parameters
        .get("token.weight", &[4096, 1024])?
        .id())
}
fn prepare(a: &mut Arm, b: &mut Arm, micros: usize, revision: usize, scale: f32) -> Result<()> {
    a.owner.begin_step()?;
    b.owner.begin_step()?;
    for micro in 0..micros {
        let revision = revision + micro;
        let la = accumulate(
            &mut a.owner,
            &mut a.executor,
            &a.fixture,
            scale / micros as f32,
            revision,
        )?;
        let lb = accumulate(
            &mut b.owner,
            &mut b.executor,
            &b.fixture,
            scale / micros as f32,
            revision,
        )?;
        if scale.is_nan() {
            ensure!(
                la.iter().all(|v| v.is_nan()) && lb.iter().all(|v| v.is_nan()),
                "NaN feed not observed"
            );
        } else {
            exact(&la, &lb, "shadow actual loss")?;
        }
    }
    Ok(())
}
fn completed(a: &mut Arm, b: &mut Arm) -> Result<()> {
    let sa = a.owner.step_mixed(&a.executor, 1.)?;
    let sb = b.owner.step_mixed(&b.executor, 1.)?;
    ensure!(
        sa.gradient_norm.to_bits() == sb.gradient_norm.to_bits()
            && sa.gradient_scale.to_bits() == sb.gradient_scale.to_bits()
            && sa.timestep == sb.timestep,
        "shadow exact norm/scale/timestep changed"
    );
    let status = b.owner.bf16_master_shadow_status();
    ensure!(
        status.selected_parameters == 21
            && status.selected_elements == 54_788_096
            && status.submitted_parameters == 21
            && status.committed_parameters == 21
            && status.submitted_elements == 54_788_096
            && status.committed_elements == 54_788_096,
        "paired owner dispatch/publication census: {status:?}"
    );
    // Just-published current versions are not yet the executor's active sources.
    // Before diagnostics consume the lazy reader, it must hold both typed Arcs.
    let id = token_id(b)?;
    let parameter = b
        .owner
        .parameters
        .iter()
        .find(|p| p.id == id)
        .context("token owner absent")?;
    let shadow = &b.owner.bf16_master_shadows.parameters[&id].current;
    let master_refs = Arc::strong_count(&parameter.weight);
    let shadow_refs = Arc::strong_count(shadow);
    ensure!(
        master_refs >= 2 && shadow_refs >= 2,
        "lazy F32 reader did not retain matching pair"
    );
    let _ = b
        .fixture
        .model
        .parameters
        .get("token.weight", &[4096, 1024])?
        .values()?;
    ensure!(
        Arc::strong_count(&parameter.weight) + 1 == master_refs
            && Arc::strong_count(shadow) + 1 == shadow_refs,
        "lazy reader did not release both pair leases"
    );
    state_equal(&a.owner, &b.owner, &a.fixture, &b.fixture)?;
    ensure!(
        b.executor.transfers().parameter_copies == 0
            && b.executor.info().bf16_master_shadow_cast_aliases == 21
            && b.executor.info().bf16_master_shadow_alias_bytes == 109_576_192,
        "shadow input copies or removed-cast census changed"
    );
    Ok(())
}
fn shadow_contents(arm: &Arm) -> Result<HashMap<usize, Vec<u16>>> {
    let result = arm
        .owner
        .bf16_master_shadows
        .parameters
        .iter()
        .map(|(&id, shadow)| {
            let values = arm.owner.stream.memcpy_dtov(shadow.current.as_ref())?;
            Ok((id, values.into_iter().map(bf16::to_bits).collect()))
        })
        .collect::<Result<_>>()?;
    arm.owner.stream.synchronize()?;
    Ok(result)
}
#[test]
#[ignore = "manual sparse BF16 shadow actual closest decoder state/rollback/host/replay/old-reader proof"]
fn sparse_bf16_shadows_actual_w1024_s128_b2_preserve_updates_and_leases() -> Result<()> {
    if let Err(error) = CudaContext::new(0) {
        ensure!(
            std::env::var_os("TNSR_REQUIRE_CUDA").is_none(),
            "CUDA required: {error}"
        );
        return Ok(());
    }
    let root = PathBuf::from(
        std::env::var_os("TNSR_SHADOW_OWNER_STATE_DIR")
            .context("set fresh TNSR_SHADOW_OWNER_STATE_DIR")?,
    );
    std::fs::create_dir_all(&root)?;
    let out = root.join("actual-w1024-s128-b2.json");
    ensure!(!out.exists(), "fresh owner shadow state output required");
    let mut a = arm(false)?;
    let mut b = arm(true)?;
    ensure!(
        a.executor.info().physical_order_cast_materializations == 21
            && b.executor.info().physical_order_cast_materializations == 0
            && b.executor.info().bf16_master_shadow_cast_aliases == 21
            && a.executor.info().kernels == b.executor.info().kernels + 21,
        "actual sparse shadow casts did not replace exactly 21 materializers"
    );
    let id = token_id(&b)?;
    let initial_master = b
        .owner
        .parameters
        .iter()
        .find(|p| p.id == id)
        .context("initial token absent")?
        .weight
        .clone();
    let initial_shadow = b.owner.bf16_master_shadows.parameters[&id].current.clone();
    let initial_pointers = (
        initial_master.device_ptr(&b.owner.stream).0,
        initial_shadow.device_ptr(&b.owner.stream).0,
    );
    let weak_master = Arc::downgrade(&initial_master);
    let weak_shadow = Arc::downgrade(&initial_shadow);
    drop(initial_master);
    drop(initial_shadow);
    let mut rows = Vec::new();
    for (interval, micros) in [1usize, 2, 1, 0].into_iter().enumerate() {
        prepare(&mut a, &mut b, micros, interval * 3, 1.)?;
        completed(&mut a, &mut b)?;
        let status = b.owner.bf16_master_shadow_status();
        if micros > 0 {
            ensure!(
                status.maximum_versions_per_parameter == 2,
                "normal model-update retained more than two versions: {status:?}"
            );
        } else {
            // No model refresh releases the executor's old input pair here.
            // Current, still-leased input and next output can require 3 versions.
            ensure!(
                status.maximum_versions_per_parameter >= 2,
                "empty-interval versions not accounted"
            );
        }
        ensure!(
            status.paired_vector_parameters == if micros == 1 { 20 } else { 0 }
                && status.paired_f32_parameters == if micros == 1 { 1 } else { 21 },
            "paired dtype fallback census changed"
        );
        if interval == 1 {
            let current = b
                .owner
                .parameters
                .iter()
                .find(|p| p.id == id)
                .context("token absent after recycle")?;
            let shadow = &b.owner.bf16_master_shadows.parameters[&id].current;
            ensure!(
                (
                    current.weight.device_ptr(&b.owner.stream).0,
                    shadow.device_ptr(&b.owner.stream).0
                ) == initial_pointers
                    && weak_master.upgrade().is_none()
                    && weak_shadow.upgrade().is_none(),
                "joint recycling did not give same addresses fresh typed Arc identities"
            );
        }
        rows.push(
            json!({"interval":interval,"microbatches":micros,"state_exact":true,"shadow":status,
            "mixed_info":b.executor.info(),"borrowed":b.owner.borrowed_bf16_adam_status(),
            "vector":b.owner.vector_bf16_first_adam_status()}),
        );
    }
    ensure!(
        b.executor.info().direct_capture_variants == 2
            && b.executor.info().direct_capture_misses == 2,
        "normal paired ping-pong did not retain two capture variants"
    );
    // Gradient-NaN rejection happens before any Adam launch.
    let before = published(&b.owner, &b.fixture)?;
    let shadows = shadow_contents(&b)?;
    prepare(&mut a, &mut b, 1, 20, f32::NAN)?;
    ensure!(
        a.owner.step_mixed(&a.executor, 1.).is_err()
            && b.owner.step_mixed(&b.executor, 1.).is_err(),
        "NaN update accepted"
    );
    ensure!(
        b.owner.bf16_master_shadow_status().submitted_parameters == 0
            && b.owner.bf16_master_shadow_status().committed_parameters == 0,
        "NaN rejection submitted paired writes"
    );
    unchanged(&a.owner, &a.fixture, &before)?;
    unchanged(&b.owner, &b.fixture, &before)?;
    ensure!(
        shadow_contents(&b)? == shadows,
        "NaN rejection changed current BF16 shadows"
    );
    prepare(&mut a, &mut b, 1, 21, 1.)?;
    completed(&mut a, &mut b)?;
    // Finite gradients with overflowing Adam arithmetic write staging, then
    // reject via the unchanged F32 health scalar. BF16 never adds a health gate.
    let before = published(&b.owner, &b.fixture)?;
    let shadows = shadow_contents(&b)?;
    for owner in [&mut a.owner, &mut b.owner] {
        owner.groups[0].learning_rate = f32::MAX;
        owner.groups[0].weight_decay = f32::MAX;
    }
    prepare(&mut a, &mut b, 1, 22, 1.)?;
    ensure!(
        a.owner.step_mixed(&a.executor, 1.).is_err()
            && b.owner.step_mixed(&b.executor, 1.).is_err(),
        "staged overflow accepted"
    );
    let rejected = b.owner.bf16_master_shadow_status().clone();
    ensure!(
        rejected.submitted_parameters == 21 && rejected.committed_parameters == 0,
        "paired staged rejection accounting changed"
    );
    unchanged(&a.owner, &a.fixture, &before)?;
    unchanged(&b.owner, &b.fixture, &before)?;
    ensure!(
        shadow_contents(&b)? == shadows,
        "staged rejection changed current BF16 shadows"
    );
    for owner in [&mut a.owner, &mut b.owner] {
        owner.groups[0].learning_rate = 1e-4;
        owner.groups[0].weight_decay = 0.;
    }
    prepare(&mut a, &mut b, 1, 23, 1.)?;
    completed(&mut a, &mut b)?;
    // A missing optional vector function must choose a paired scalar BF16
    // function, never a stale shadow or an ordinary scalar function.
    b.owner
        .bf16_master_shadows
        .force_optional_vector_unavailable_for_test();
    prepare(&mut a, &mut b, 1, 24, 1.)?;
    completed(&mut a, &mut b)?;
    let scalar_fallback = b.owner.bf16_master_shadow_status().clone();
    ensure!(
        scalar_fallback.paired_vector_fallback_parameters == 20
            && scalar_fallback.paired_bf16_parameters == 20
            && scalar_fallback.paired_f32_parameters == 1
            && b.owner.vector_bf16_first_adam_status().committed_parameters == 0,
        "paired vector failure did not retain scalar BF16+F32 shadow publication"
    );
    // A real second run invalidates the borrowed-source stamp. Both owners use
    // their accumulated F32 gradients and still publish matching shadow stores.
    prepare(&mut a, &mut b, 1, 25, 1.)?;
    a.executor.run()?;
    b.executor.run()?;
    completed(&mut a, &mut b)?;
    ensure!(
        b.owner.bf16_master_shadow_status().paired_f32_parameters == 21
            && b.owner
                .borrowed_bf16_adam_status()
                .standalone_bf16_parameters
                == 0,
        "stale borrowed source was rescued or lost paired F32 writes"
    );
    // Explicit host replacement keeps shared F32 storage identity and imports
    // a fresh coherent pair before installing any new executor bindings.
    for arm in [&mut a, &mut b] {
        let parameter = arm
            .fixture
            .model
            .parameters
            .get("token.weight", &[4096, 1024])?;
        let mut values = parameter.values()?;
        values[0] += 0.03125;
        parameter.set_values(values)?;
    }
    prepare(&mut a, &mut b, 1, 26, 1.)?;
    completed(&mut a, &mut b)?;
    ensure!(
        b.owner
            .bf16_master_shadow_status()
            .host_mutation_initialization_launches
            == 1,
        "host mutation did not initialize exactly one fresh sparse shadow"
    );
    // A real external pair lease pins both old allocations. Later versions
    // must stay exact without overwriting the reader or pretending capacity=2.
    let id = token_id(&b)?;
    let held_master = b
        .owner
        .parameters
        .iter()
        .find(|p| p.id == id)
        .context("held token absent")?
        .weight
        .clone();
    let held_shadow = b.owner.bf16_master_shadows.parameters[&id].current.clone();
    let held_f32 = b.owner.stream.memcpy_dtov(held_master.as_ref())?;
    let held_bf16 = b.owner.stream.memcpy_dtov(held_shadow.as_ref())?;
    b.owner.stream.synchronize()?;
    let held_weak_master = Arc::downgrade(&held_master);
    let held_weak_shadow = Arc::downgrade(&held_shadow);
    for revision in [27usize, 28] {
        prepare(&mut a, &mut b, 1, revision, 1.)?;
        completed(&mut a, &mut b)?;
    }
    let after_f32 = b.owner.stream.memcpy_dtov(held_master.as_ref())?;
    let after_bf16 = b.owner.stream.memcpy_dtov(held_shadow.as_ref())?;
    b.owner.stream.synchronize()?;
    exact(&held_f32, &after_f32, "retained old F32 reader")?;
    ensure!(
        held_bf16
            .iter()
            .zip(after_bf16)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "retained old BF16 reader changed"
    );
    let retained = b.owner.bf16_master_shadow_status().clone();
    ensure!(
        retained.maximum_versions_per_parameter >= 3
            && retained.retained_shadow_bytes > 0
            && retained.current_shadow_versions
                + retained.staged_shadow_versions
                + retained.retained_shadow_versions
                > 2 * retained.selected_parameters,
        "external old reader did not disclose an extra live version beyond ordinary two-pair storage"
    );
    drop(held_master);
    drop(held_shadow);
    b.owner.reclaim_shadow_pairs()?;
    ensure!(
        held_weak_master.upgrade().is_none() && held_weak_shadow.upgrade().is_none(),
        "inactive captures retained strong old allocation leases"
    );
    prepare(&mut a, &mut b, 1, 29, 1.)?;
    completed(&mut a, &mut b)?;
    ensure!(
        b.executor.info().direct_capture_variants <= 2,
        "old-reader recapture cache is unbounded"
    );
    std::fs::write(
        out,
        serde_json::to_vec_pretty(&json!({
            "scope":"Actual closest decoder evolving state with extra microbatch/NaN scalar. No archived first-update or performance claim.",
            "config":b.fixture.model.config,"intervals":rows,"all_full_finite_state_bits_exact":true,
            "NaN_prelaunch_rejection":true,"staged_health_rejection":rejected,"paired_scalar_fallback":scalar_fallback,
            "host_mutation_fresh_pair":true,"stale_accumulation_F32_paired_fallback":true,
            "same_addresses_fresh_Arc_identity":true,"external_old_pair_immutable":true,"old_reader_status":retained,
            "inactive_captures_no_strong_pair_lease":true,"final_shadow":b.owner.bf16_master_shadow_status(),
            "final_mixed":b.executor.info(),"baseline_mixed":a.executor.info(),
        }))?,
    )?;
    Ok(())
}

#[test]
#[ignore = "manual integrated owner finite-F32/BF16-infinity shadow health boundary"]
fn sparse_bf16_shadow_finite_master_overflow_does_not_add_health_rejection() -> Result<()> {
    use crate::tensor::{Parameter, ValueDType};
    if let Err(error) = CudaContext::new(0) {
        ensure!(
            std::env::var_os("TNSR_REQUIRE_CUDA").is_none(),
            "CUDA required: {error}"
        );
        return Ok(());
    }
    let parameter = Parameter::new(vec![f32::MAX; 8192], vec![128, 64]);
    let expression = TensorExpr::from(parameter.clone())
        .transpose()
        .cast(ValueDType::BF16)
        .cast(ValueDType::F32)
        .mean_all();
    let mut graph = TensorGraph::new();
    let loss = graph.add_expr(&expression, false);
    let graph = graph.with_gradients(loss);
    let mut owner = ResidentAdam::new_with_graph_replay(
        vec![(parameter.clone(), 0)],
        vec![AdamGroup::new(1e-4)],
    )?;
    owner.declare_experimental_bf16_master_shadows(
        &graph,
        MixedOptions {
            physical_order_casts: true,
            direct_master_bindings: true,
            ..MixedOptions::default()
        },
    )?;
    ensure!(
        owner.bf16_master_shadow_status().selected_parameters == 1,
        "boundary matrix was not selected"
    );
    owner.begin_step()?;
    // No forward pass is needed: this tests zero-gradient Adam and the owner's
    // unchanged F32-only acceptance gate, despite the shadow containing infinity.
    let step = owner.step(0.)?;
    ensure!(
        step.timestep == 1 && step.gradient_norm == 0.,
        "finite master zero-gradient update rejected"
    );
    ensure!(
        parameter
            .values()?
            .iter()
            .all(|v| v.to_bits() == f32::MAX.to_bits()),
        "finite master changed"
    );
    let shadow = &owner.bf16_master_shadows.parameters[&parameter.id()].current;
    let values = owner.stream.memcpy_dtov(shadow.as_ref())?;
    owner.stream.synchronize()?;
    ensure!(
        values
            .iter()
            .all(|v| v.is_infinite() && !v.is_sign_negative()),
        "GPU shadow did not preserve finite-to-infinity rounding"
    );
    let status = owner.bf16_master_shadow_status();
    ensure!(
        status.paired_f32_parameters == 1
            && status.submitted_parameters == 1
            && status.committed_parameters == 1,
        "BF16 infinity added an owner health rejection"
    );
    Ok(())
}
