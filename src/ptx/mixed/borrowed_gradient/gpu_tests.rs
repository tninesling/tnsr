//! Manual semantic qualification; no performance claims or timers.
use super::*;
use crate::{
    ptx::resident::{AdamGroup, ResidentAdam},
    tensor::{Parameter, TensorExpr},
};
struct Fixture {
    graph: TensorGraph<f32, WithGrad>,
    parameters: Vec<Parameter<f32>>,
    loss: NodeIndex,
}
fn fixture(two: bool) -> Fixture {
    let mut loss = None;
    let mut parameters = Vec::new();
    for (branch, n) in if two {
        vec![33usize, 34]
    } else {
        vec![33usize]
    }
    .into_iter()
    .enumerate()
    {
        let p = Parameter::new(
            (0..32 * n).map(|i| ((i % 13) as f32 - 6.) / 64.).collect(),
            vec![32, n],
        );
        let w: TensorExpr<f32> = p.clone().into();
        let x = TensorExpr::<f32>::input(if branch == 0 { "x" } else { "x2" }, vec![32, 32]);
        let y = x
            .cast(ValueDType::BF16)
            .matmul(w.cast(ValueDType::BF16))
            .cast(ValueDType::F32);
        let term = (y * TensorExpr::input("seed", vec![]))
            .reduce_sum(0)
            .reduce_sum(1);
        loss = Some(match loss {
            None => term,
            Some(old) => old + term,
        });
        parameters.push(p);
    }
    let mut graph = TensorGraph::new();
    let root = graph.add_expr(&loss.unwrap(), false);
    Fixture {
        graph: graph.with_gradients(root),
        parameters,
        loss: root,
    }
}
fn feeds(two: bool, seed: f32, revision: usize) -> HashMap<String, Vec<f32>> {
    let data = (0..1024)
        .map(|i| ((i % 17) as f32 - 8. + revision as f32) / 32.)
        .collect::<Vec<_>>();
    let mut v = HashMap::from([("x".into(), data.clone()), ("seed".into(), vec![seed])]);
    if two {
        v.insert("x2".into(), data);
    }
    v
}
fn setup(
    f: &Fixture,
    borrowed: bool,
    batch: usize,
    compact: bool,
) -> Result<(ResidentAdam, MixedExecutor)> {
    setup_mode(f, borrowed, batch, compact, true, 0.0)
}
fn setup_mode(
    f: &Fixture,
    borrowed: bool,
    batch: usize,
    compact: bool,
    optimized: bool,
    decay: f32,
) -> Result<(ResidentAdam, MixedExecutor)> {
    let mut group = AdamGroup::new(1e-3);
    group.weight_decay = decay;
    let mut owner = ResidentAdam::new_with_graph_replay(
        f.parameters.iter().cloned().map(|p| (p, 0)).collect(),
        vec![group],
    )?;
    owner.set_update_batch_size(batch)?;
    if optimized {
        owner.set_experimental_first_accumulation_overwrite(true)?;
        owner.set_experimental_gradient_norm_partials(true)?;
    }
    owner.set_experimental_adam_256_threads(compact)?;
    owner.set_experimental_borrowed_bf16_adam_source(borrowed)?;
    owner.begin_step()?;
    let e = owner.mixed_executor_with_options(
        &f.graph,
        MatMulPrecision::NativeHalf,
        MixedOptions {
            direct_master_bindings: true,
            incremental_allocation_reuse: true,
            ..MixedOptions::default()
        },
    )?;
    assert!(
        e.borrowed_gradient_stamp().is_none(),
        "constructor is not a completed user run"
    );
    Ok((owner, e))
}
fn accumulate(
    owner: &mut ResidentAdam,
    e: &mut MixedExecutor,
    f: &Fixture,
    seed: f32,
    revision: usize,
) -> Result<()> {
    owner.accumulate_mixed(
        e,
        &f.graph,
        feeds(f.parameters.len() > 1, seed, revision),
        HashMap::new(),
    )
}
fn same(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (i, (a, b)) in a.iter().zip(b).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "state{i}: {a}/{b}");
    }
}
fn state_equal(a: &ResidentAdam, b: &ResidentAdam, fa: &Fixture, fb: &Fixture) -> Result<()> {
    let ga = a.gradients()?;
    let gb = b.gradients()?;
    let ma = a.moments()?;
    let mb = b.moments()?;
    for (pa, pb) in fa.parameters.iter().zip(&fb.parameters) {
        same(&pa.values()?, &pb.values()?);
        same(&ga[&pa.id()], &gb[&pb.id()]);
        same(&ma[&pa.id()].first, &mb[&pb.id()].first);
        same(&ma[&pa.id()].second, &mb[&pb.id()].second);
    }
    Ok(())
}
fn available() -> bool {
    let ok = CudaContext::new(0).is_ok();
    assert!(
        ok || std::env::var_os("TNSR_REQUIRE_CUDA").is_none(),
        "CUDA required"
    );
    ok
}

#[test]
#[ignore = "manual borrowed BF16 Adam dense/state/rollback/heterogeneous dispatch proof"]
fn borrowed_bf16_mixed_single_multi_empty_batched_and_recovery() -> Result<()> {
    if !available() {
        return Ok(());
    }
    for (two, batch, compact, optimized) in [
        (false, 1, false, true),
        (false, 1, true, true),
        (true, 8, true, true),
        (false, 1, false, false),
    ] {
        let fa = fixture(two);
        let fb = fixture(two);
        let (mut a, mut ea) = setup_mode(&fa, false, batch, compact, optimized, 0.)?;
        let (mut b, mut eb) = setup_mode(&fb, true, batch, compact, optimized, 0.)?;
        let mut proof_counts = None;
        for (interval, micros) in [1usize, 2, 1, 1, 0].into_iter().enumerate() {
            if interval > 0 {
                a.begin_step()?;
                b.begin_step()?;
            }
            for micro in 0..micros {
                accumulate(&mut a, &mut ea, &fa, 1. / micros as f32, interval + micro)?;
                accumulate(&mut b, &mut eb, &fb, 1. / micros as f32, interval + micro)?;
            }
            let sa = a.step(0.5)?;
            let sb = b.step_mixed(&eb, 0.5)?;
            assert_eq!(sa.gradient_norm.to_bits(), sb.gradient_norm.to_bits());
            assert_eq!(sa.gradient_scale.to_bits(), sb.gradient_scale.to_bits());
            state_equal(&a, &b, &fa, &fb)?;
            let current = (eb.borrowed_prefixes.1.get(), eb.borrowed_prefixes.2.get());
            if let Some(first) = proof_counts {
                assert_eq!(
                    current, first,
                    "normalization/owner scans repeated across direct-master variants"
                );
            } else {
                proof_counts = Some(current);
            }
            if interval >= 1 {
                assert_eq!(eb.info().direct_capture_variants, 2);
            }
            let expected = usize::from(micros == 1 && !two);
            assert_eq!(
                b.borrowed_bf16_adam_status().standalone_bf16_parameters,
                expected,
                "{:?}",
                b.borrowed_bf16_adam_status()
            );
            if two {
                assert_eq!(b.borrowed_bf16_adam_status().batched_f32_parameters, 2);
            }
            if micros > 0 {
                same(&ea.read(fa.loss)?, &eb.read(fb.loss)?);
            }
            if interval == 2 {
                let before = fb
                    .parameters
                    .iter()
                    .map(Parameter::values)
                    .collect::<Result<Vec<_>>>()?;
                b.begin_step()?;
                accumulate(&mut b, &mut eb, &fb, f32::NAN, 0)?;
                assert!(b.step_mixed(&eb, 0.5).is_err());
                for (p, old) in fb.parameters.iter().zip(before) {
                    same(&p.values()?, &old);
                }
                // Restore same successful interval count after invalid publication.
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "manual borrowed BF16 Adam stale mutation/identity/failure matrix"]
fn borrowed_bf16_mixed_stale_failed_mutations_and_wrong_executor_fallback() -> Result<()> {
    if !available() {
        return Ok(());
    }
    for mode in [
        "bad_upload",
        "upload",
        "rerun",
        "failed_rerun",
        "refresh_hit",
        "refresh_fail",
        "refresh_miss",
        "timing_zero",
        "overflow",
        "wrong_executor",
        "dropped_executor",
        "rate",
        "batch",
        "bad_geometry_setter",
        "disable_attempt",
        "descriptor",
        "host_edit",
    ] {
        let fa = fixture(false);
        let fb = fixture(false);
        let (mut a, mut ea) = setup(&fa, false, 1, true)?;
        let (mut b, mut eb) = setup(&fb, true, 1, true)?;
        accumulate(&mut a, &mut ea, &fa, 1., 0)?;
        accumulate(&mut b, &mut eb, &fb, 1., 0)?;
        let mut other = None;
        match mode {
            "bad_upload" => {
                assert!(eb.upload(HashMap::new()).is_err());
                assert!(eb.has_run);
            }
            "upload" => {
                eb.upload(feeds(false, 0.5, 1))?;
            }
            "rerun" => eb.run()?,
            "failed_rerun" => {
                eb.uploaded = false;
                assert!(eb.run().is_err());
                assert!(eb.has_run);
            }
            "refresh_hit" => {
                let p = eb
                    .bound_parameters
                    .iter()
                    .map(|(&n, v)| (n, v.upgrade().unwrap()))
                    .collect();
                eb.refresh_parameters(&p)?;
            }
            "refresh_fail" => {
                assert!(eb.refresh_parameters(&HashMap::new()).is_err());
            }
            "refresh_miss" => {
                let mut p = HashMap::new();
                for (&n, old) in &eb.bound_parameters {
                    let old = old.upgrade().unwrap();
                    let mut new = eb.stream.alloc_zeros::<f32>(old.len())?;
                    eb.stream.memcpy_dtod(old.as_ref(), &mut new)?;
                    p.insert(n, Arc::new(new));
                }
                eb.refresh_parameters(&p)?;
            }
            "timing_zero" => {
                assert!(eb.time_runs(0).is_err());
            }
            "overflow" => {
                eb.freshness.epoch = Some(u64::MAX);
                eb.run()?;
            }
            "wrong_executor" | "dropped_executor" => {
                other = Some(b.mixed_executor(&fb.graph, MatMulPrecision::NativeHalf)?);
                if mode == "dropped_executor" {
                    drop(eb);
                    eb = b.mixed_executor(&fb.graph, MatMulPrecision::NativeHalf)?;
                }
            }
            "rate" => {
                a.set_learning_rate(0, 1e-3)?;
                b.set_learning_rate(0, 1e-3)?;
            }
            "batch" => {
                a.set_update_batch_size(8)?;
                b.set_update_batch_size(8)?;
            }
            "bad_geometry_setter" => {
                assert!(b.set_experimental_adam_256_threads(true).is_err());
            }
            "disable_attempt" => {
                assert!(b.set_experimental_borrowed_bf16_adam_source(false).is_err());
            }
            "descriptor" => {
                let n = fb.graph.gradient_metadata().param_to_grad[&fb.parameters[0].id()];
                eb.values.get_mut(&n).unwrap().view.shape[0] -= 1;
            }
            "host_edit" => {
                let values = fb.parameters[0].values()?;
                fb.parameters[0].set_values(values)?;
            }
            _ => unreachable!(),
        }
        if mode == "host_edit" {
            assert!(b.step_mixed(&eb, 1.).is_err());
            continue;
        }
        a.step(1.)?;
        b.step_mixed(other.as_ref().unwrap_or(&eb), 1.)?;
        assert_eq!(
            b.borrowed_bf16_adam_status().standalone_bf16_parameters,
            0,
            "{mode}"
        );
        state_equal(&a, &b, &fa, &fb)?;
    }
    Ok(())
}

#[test]
#[ignore = "manual borrowed BF16 Adam failed second attempts and single-use retry"]
fn borrowed_bf16_mixed_failed_second_attempts_and_consumed_step_retry() -> Result<()> {
    if !available() {
        return Ok(());
    }
    for homogeneous in [false, true] {
        let f = fixture(false);
        let (mut owner, mut e) = setup(&f, true, 1, true)?;
        accumulate(&mut owner, &mut e, &f, 1., 0)?;
        let initial = f.parameters[0].values()?;
        if homogeneous {
            let mut other = owner.executor()?;
            assert!(
                owner
                    .accumulate(&mut other, &f.graph, HashMap::new())
                    .is_err()
            );
        } else {
            assert!(
                owner
                    .accumulate_mixed(&mut e, &f.graph, HashMap::new(), HashMap::new())
                    .is_err()
            );
        }
        assert!(owner.step_mixed(&e, 1.).is_err());
        same(&f.parameters[0].values()?, &initial);
        owner.begin_step()?;
        accumulate(&mut owner, &mut e, &f, 1., 1)?;
        assert!(owner.step_mixed(&e, -1.).is_err());
        owner.step_mixed(&e, 1.)?;
        assert_eq!(
            owner.borrowed_bf16_adam_status().standalone_bf16_parameters,
            0,
            "candidate must be single-use after failed step"
        );
    }
    Ok(())
}

#[test]
#[ignore = "manual borrowed gradient proof cache hit/rebind and pooled logical prefix"]
fn borrowed_bf16_live_prefix_cache_hits_rebind_and_invalid_offsets() -> Result<()> {
    if !available() {
        return Ok(());
    }
    let f = fixture(false);
    let (mut owner, mut e) = setup(&f, true, 1, true)?;
    accumulate(&mut owner, &mut e, &f, 1., 0)?;
    let node = f.graph.gradient_metadata().param_to_grad[&f.parameters[0].id()];
    let len = 32 * 33;
    let first = e
        .borrowed_bf16_gradient(node, len)
        .context("fresh BF16 gradient absent")?;
    let old = first.device_ptr(&e.stream).0;
    assert!(e.borrowed_prefixes.0.borrow().contains_key(&(node, len)));
    let again = e.borrowed_bf16_gradient(node, len).unwrap();
    assert_eq!(again.device_ptr(&e.stream).0, old);
    let mut replacement = e.stream.alloc_zeros::<bf16>(len * 2)?;
    e.stream
        .memcpy_dtod(&again, &mut replacement.slice_mut(0..len))?;
    e.stream.synchronize()?;
    let pointer = replacement.device_ptr(&e.stream).0;
    e.buffers.push(Buffer::BF16(replacement));
    e.values.get_mut(&node).unwrap().binding.pointer = pointer;
    e.freshness.dirty();
    let prefix = e
        .borrowed_bf16_gradient(node, len)
        .context("valid larger-owner logical prefix rejected")?;
    assert_eq!(prefix.len(), len);
    assert_eq!(prefix.device_ptr(&e.stream).0, pointer);
    assert_eq!(
        e.borrowed_prefixes.0.borrow()[&(node, len)].owner,
        Some(e.buffers.len() - 1)
    );
    e.values.get_mut(&node).unwrap().view = e.values[&node].view.clone().permute(&[1, 0])?;
    assert!(e.borrowed_bf16_gradient(node, len).is_none());
    let negative = (e.borrowed_prefixes.1.get(), e.borrowed_prefixes.2.get());
    assert!(e.borrowed_bf16_gradient(node, len).is_none());
    assert_eq!(
        negative,
        (e.borrowed_prefixes.1.get(), e.borrowed_prefixes.2.get()),
        "negative proof repeated expensive work"
    );
    e.values.get_mut(&node).unwrap().binding.pointer += 2;
    assert!(e.borrowed_bf16_gradient(node, len).is_none());
    owner.step_mixed(&e, 1.)?;
    assert_eq!(
        owner.borrowed_bf16_adam_status().standalone_bf16_parameters,
        0
    );
    Ok(())
}

#[test]
#[ignore = "manual borrowed BF16 Adam post-launch health rejection and cleanup"]
fn borrowed_bf16_health_rejection_keeps_public_state_and_retires_live_borrow() -> Result<()> {
    if !available() {
        return Ok(());
    }
    for borrowed in [false, true] {
        let f = fixture(false);
        f.parameters[0].set_values(vec![1e20; 32 * 33])?;
        let (mut owner, mut e) = setup_mode(&f, borrowed, 1, true, true, 1e30)?;
        let before = f.parameters[0].values()?;
        let moments = owner.moments()?;
        accumulate(&mut owner, &mut e, &f, 1., 0)?;
        assert!(
            if borrowed {
                owner.step_mixed(&e, 1.)
            } else {
                owner.step(1.)
            }
            .is_err()
        );
        if borrowed {
            assert_eq!(
                owner.borrowed_bf16_adam_status().standalone_bf16_parameters,
                1
            );
        }
        drop(e); // Caller can release all source storage immediately after Err.
        same(&f.parameters[0].values()?, &before);
        let after = owner.moments()?;
        let id = f.parameters[0].id();
        same(&moments[&id].first, &after[&id].first);
        same(&moments[&id].second, &after[&id].second);
    }
    Ok(())
}
