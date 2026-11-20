use super::*;
use crate::tensor::TensorExpr;
use cudarc::driver::DevicePtr;
use std::collections::HashSet;

fn owner(parameter: &Parameter<f32>, replay: bool) -> Option<ResidentAdam> {
    match ResidentAdam::initialize(
        vec![(parameter.clone(), 0)],
        vec![AdamGroup::new(0.01)],
        replay,
    ) {
        Ok(value) => Some(value),
        Err(error)
            if std::env::var("TNSR_REQUIRE_CUDA").as_deref() != Ok("1")
                && error
                    .to_string()
                    .starts_with("initialize resident training CUDA device") =>
        {
            None
        }
        Err(error) => panic!("{error:#}"),
    }
}

fn address(value: &CudaSlice<f32>, stream: &Arc<CudaStream>) -> u64 {
    value.device_ptr(stream).0
}

fn graph(p: &Parameter<f32>) -> TensorGraph<f32, WithGrad> {
    let value = TensorExpr::from(p.clone()) * TensorExpr::input("x", vec![3]);
    let graph: TensorGraph<f32> = value.mean_all().into();
    let loss = *graph.toposort().last().unwrap();
    graph.with_gradients(loss)
}

#[test]
fn adam_reuses_unpublished_state_and_preserves_padding() {
    let p = Parameter::new(vec![0.2, -0.4, 0.1], vec![3]);
    let Some(mut owner) = owner(&p, true) else {
        return;
    };
    let graph = graph(&p);
    let mut executor = owner.executor().unwrap();
    let stream = owner.stream.clone();
    let moments = HashSet::from([
        address(&owner.moment, &stream),
        address(&owner.staging.moment, &stream),
    ]);
    let variances = HashSet::from([
        address(&owner.variance, &stream),
        address(&owner.staging.variance, &stream),
    ]);
    let health = address(owner.staging.health.as_ref(), &stream);
    let settings = address(&owner.staging.settings[0], &stream);
    let mut weights = HashSet::from([address(owner.parameters[0].weight.as_ref(), &stream)]);
    for _ in 0..6 {
        owner.begin_step().unwrap();
        owner
            .accumulate(
                &mut executor,
                &graph,
                HashMap::from([("x".into(), vec![1.0, 2.0, 3.0])]),
            )
            .unwrap();
        owner.step(0.0).unwrap();
        assert_eq!(
            moments,
            HashSet::from([
                address(&owner.moment, &stream),
                address(&owner.staging.moment, &stream)
            ])
        );
        assert_eq!(
            variances,
            HashSet::from([
                address(&owner.variance, &stream),
                address(&owner.staging.variance, &stream)
            ])
        );
        assert_eq!(health, address(owner.staging.health.as_ref(), &stream));
        assert_eq!(settings, address(&owner.staging.settings[0], &stream));
        weights.insert(address(owner.parameters[0].weight.as_ref(), &stream));
    }
    // Reusing a physical weight needs a new Arc identity so weak replay source
    // identities continue to trigger a copy of the latest committed values.
    assert_eq!(weights.len(), 2);
    assert_eq!(executor.replay_metrics().captures, 1);
    for values in [
        &owner.moment,
        &owner.variance,
        &owner.staging.moment,
        &owner.staging.variance,
        owner.staging.health.as_ref(),
    ] {
        let host = stream.memcpy_dtov(values).unwrap();
        assert!(host[3..].iter().all(|v| v.to_bits() == 0));
    }
    for (a, e) in p.values().unwrap().iter().zip([0.14, -0.46, 0.04]) {
        assert!((a - e).abs() < 2e-6, "{a} != {e}");
    }
}

#[test]
fn adam_does_not_recycle_retained_published_versions() {
    for replay in [false, true] {
        let p = Parameter::new(vec![0.2, -0.4, 0.1], vec![3]);
        let Some(mut owner) = owner(&p, replay) else {
            return;
        };
        let graph = graph(&p);
        let mut executor = owner.executor().unwrap();
        let stream = owner.stream.clone();
        let mut versions = Vec::new();
        for _ in 0..5 {
            let version = owner.parameters[0].weight.clone();
            let before = stream.memcpy_dtov(version.as_ref()).unwrap();
            versions.push((version, before));
            owner.begin_step().unwrap();
            owner
                .accumulate(
                    &mut executor,
                    &graph,
                    HashMap::from([("x".into(), vec![1.0, 2.0, 3.0])]),
                )
                .unwrap();
            owner.step(0.0).unwrap();
        }
        drop(executor);
        drop(owner);
        for (version, before) in versions {
            assert_eq!(stream.memcpy_dtov(version.as_ref()).unwrap(), before);
        }
        // The last lazy publication also remains available after owner drop.
        for (a, e) in p.values().unwrap().iter().zip([0.15, -0.45, 0.05]) {
            assert!((a - e).abs() < 2e-6, "{a} != {e}");
        }
    }
}

fn health_owner(lengths: &[usize], replay: bool) -> Option<ResidentAdam> {
    let parameters = lengths
        .iter()
        .map(|&len| (Parameter::new(vec![0.0; len], vec![len]), 0))
        .collect();
    match ResidentAdam::initialize(parameters, vec![AdamGroup::new(0.01)], replay) {
        Ok(owner) => Some(owner),
        Err(error)
            if std::env::var("TNSR_REQUIRE_CUDA").as_deref() != Ok("1")
                && error
                    .to_string()
                    .starts_with("initialize resident training CUDA device") =>
        {
            None
        }
        Err(error) => panic!("{error:#}"),
    }
}
fn fill_health_gradients(owner: &mut ResidentAdam) -> Vec<f32> {
    owner.begin_step().unwrap();
    let mut gradients = vec![0.0; owner.padded_len];
    for parameter in &owner.parameters {
        for (i, value) in gradients[parameter.range.clone()].iter_mut().enumerate() {
            *value = [0.25, -0.5, -0.0, 1.0][i % 4];
        }
    }
    owner
        .stream
        .memcpy_htod(&gradients, Arc::get_mut(&mut owner.gradients).unwrap())
        .unwrap();
    gradients
}
#[test]
fn compact_health_handles_tails_batch_switches_and_keeps_gradient_diagnostics() {
    let lengths = [
        1, 3, 31, 33, 255, 256, 257, 1023, 1024, 1025, 4095, 4096, 4097,
    ];
    let Some(mut owner) = health_owner(&lengths, true) else {
        return;
    };
    let mut health_end = 0;
    for (parameter, len) in owner.parameters.iter().zip(lengths) {
        assert_eq!(
            parameter.health_range,
            health_end..health_end + len.div_ceil(kernels::HEALTH_GROUP_SIZE)
        );
        health_end = parameter.health_range.end;
    }
    assert_eq!(
        owner.staging.health.len(),
        health_end.next_multiple_of(1024)
    );
    let health_address = address(owner.staging.health.as_ref(), &owner.stream);
    for batch in [1, 8, 3, 1, 16] {
        owner.set_update_batch_size(batch).unwrap();
        let expected = fill_health_gradients(&mut owner);
        owner.step(0.0).unwrap();
        let actual = owner.gradients().unwrap();
        for p in &owner.parameters {
            assert_eq!(
                actual[&p.id]
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                expected[p.range.clone()]
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            );
        }
        assert_eq!(
            health_address,
            address(owner.staging.health.as_ref(), &owner.stream)
        );
        let health = owner
            .stream
            .memcpy_dtov(owner.staging.health.as_ref())
            .unwrap();
        assert!(health.iter().all(|v| v.to_bits() == 0));
    }
}

#[test]
fn compact_health_nonfinite_tail_rejection_and_recovery_preserve_committed_state() {
    let Some(mut owner) = health_owner(&[1, 17, 33, 257, 2049], true) else {
        return;
    };
    for (attempt, (target, element)) in [
        (4, 0),
        (4, 31),
        (4, 32),
        (4, 255),
        (4, 256),
        (4, 1023),
        (4, 1024),
        (4, 2048),
        (1, 16),
        (2, 32),
        (0, 0),
    ]
    .into_iter()
    .enumerate()
    {
        let failed_batch = if attempt % 2 == 0 { 8 } else { 1 };
        owner.set_update_batch_size(failed_batch).unwrap();
        for (index, p) in owner.parameters.iter().enumerate() {
            let mut values = p.storage.lock().unwrap();
            values.fill(0.0);
            if index == target {
                values[element] = f32::MAX;
            }
        }
        owner.groups[0].weight_decay = 2.0;
        owner.begin_step().unwrap();
        let before = owner
            .parameters
            .iter()
            .map(|p| owner.stream.memcpy_dtov(p.weight.as_ref()).unwrap())
            .collect::<Vec<_>>();
        let moment = owner.stream.memcpy_dtov(&owner.moment).unwrap();
        let variance = owner.stream.memcpy_dtov(&owner.variance).unwrap();
        let timestep = owner.timestep;
        let error = owner.step(0.0).unwrap_err();
        assert!(
            error.to_string().contains("Adam produced nonfinite"),
            "{error:#}"
        );
        assert_eq!(owner.timestep, timestep);
        assert_eq!(owner.stream.memcpy_dtov(&owner.moment).unwrap(), moment);
        assert_eq!(owner.stream.memcpy_dtov(&owner.variance).unwrap(), variance);
        for (p, expected) in owner.parameters.iter().zip(before) {
            assert_eq!(
                owner.stream.memcpy_dtov(p.weight.as_ref()).unwrap(),
                expected
            );
        }
        let health = owner
            .stream
            .memcpy_dtov(owner.staging.health.as_ref())
            .unwrap();
        let bad =
            owner.parameters[target].health_range.start + element / kernels::HEALTH_GROUP_SIZE;
        assert!(health[bad].is_nan(), "missing health failure at {element}");
        assert!(
            health[owner.parameters.last().unwrap().health_range.end..]
                .iter()
                .all(|v| v.to_bits() == 0)
        );
        // Switch launch geometry before recovery. Every valid partial must be
        // overwritten, including NaN slots from another parameter's long tail.
        owner
            .set_update_batch_size(if failed_batch == 1 { 8 } else { 1 })
            .unwrap();
        owner.groups[0].weight_decay = 0.0;
        for p in &owner.parameters {
            p.storage.lock().unwrap().fill(0.0);
        }
        owner.begin_step().unwrap();
        assert_eq!(owner.step(0.0).unwrap().timestep, timestep + 1);
        assert!(
            owner
                .stream
                .memcpy_dtov(owner.staging.health.as_ref())
                .unwrap()
                .iter()
                .all(|v| v.to_bits() == 0)
        );
    }
}

#[test]
fn compact_health_recovery_keeps_model_graph_replay_and_published_versions() {
    let p = Parameter::new(vec![0.2, -0.4, 0.1], vec![3]);
    let Some(mut owner) = owner(&p, true) else {
        return;
    };
    let graph = graph(&p);
    let mut executor = owner.executor().unwrap();
    let accumulate = |owner: &mut ResidentAdam, executor: &mut PtxExecutor| {
        owner.begin_step().unwrap();
        owner
            .accumulate(
                executor,
                &graph,
                HashMap::from([("x".into(), vec![1.0, 2.0, 3.0])]),
            )
            .unwrap();
    };
    accumulate(&mut owner, &mut executor);
    owner.step(0.0).unwrap();
    let retained = owner.parameters[0].weight.clone();
    let expected = owner.stream.memcpy_dtov(retained.as_ref()).unwrap();
    owner.groups[0].weight_decay = f32::MAX;
    accumulate(&mut owner, &mut executor);
    assert!(owner.step(0.0).is_err());
    assert_eq!(owner.timestep, 1);
    owner.groups[0].weight_decay = 0.0;
    accumulate(&mut owner, &mut executor);
    assert_eq!(owner.step(0.0).unwrap().timestep, 2);
    assert_eq!(executor.replay_metrics().captures, 1);
    assert_eq!(
        owner.stream.memcpy_dtov(retained.as_ref()).unwrap(),
        expected
    );
    assert!(
        owner
            .gradients()
            .unwrap()
            .values()
            .all(|g| g.iter().all(|v| v.is_finite()))
    );
}

#[test]
fn first_accumulation_overwrite_matches_clear_across_microbatches_and_recovery() {
    use crate::tensor::ValueDType;
    use crate::tile::MatMulPrecision;

    for mixed in [false, true] {
        let parameters = [
            Parameter::new(vec![0.2, -0.4, 0.1], vec![3]),
            Parameter::new(vec![0.2, -0.4, 0.1], vec![3]),
        ];
        let Some(baseline) = owner(&parameters[0], true) else {
            return;
        };
        let mut owners = [baseline, owner(&parameters[1], true).unwrap()];
        owners[1]
            .set_experimental_first_accumulation_overwrite(true)
            .unwrap();
        let graphs: Vec<_> = parameters
            .iter()
            .map(|p| {
                if !mixed {
                    return graph(p);
                }
                let value = (TensorExpr::from(p.clone()) * TensorExpr::input("x", vec![3]))
                    .cast(ValueDType::BF16)
                    .cast(ValueDType::F32);
                let g: TensorGraph<f32> = value.mean_all().into();
                let loss = *g.toposort().last().unwrap();
                g.with_gradients(loss)
            })
            .collect();
        let mut executors: Vec<_> = owners.iter().map(|o| o.executor().unwrap()).collect();
        for o in &mut owners {
            o.begin_step().unwrap();
        }
        let mut mixed_executors: Vec<_> = if mixed {
            owners
                .iter()
                .zip(&graphs)
                .map(|(o, g)| o.mixed_executor(g, MatMulPrecision::Tf32x3).unwrap())
                .collect()
        } else {
            Vec::new()
        };
        // Keep an immutable old device version through publication/reuse.
        let retained = owners[1].parameters[0].weight.clone();
        for step in 0..5 {
            for o in &mut owners {
                o.begin_step().unwrap();
                assert!(o.gradients().is_err());
                assert!(
                    o.set_experimental_first_accumulation_overwrite(false)
                        .is_err()
                );
            }
            if step == 2 {
                // Simulate an interrupted interval with a dirty parameter range.
                // A failed feed must forbid publication even if some writes exist.
                for (i, o) in owners.iter_mut().enumerate() {
                    let buffer = Arc::get_mut(&mut o.gradients).unwrap();
                    o.stream
                        .memcpy_htod(&[123.0_f32; 3], &mut buffer.slice_mut(0..3))
                        .unwrap();
                    let error = if mixed {
                        o.accumulate_mixed(
                            &mut mixed_executors[i],
                            &graphs[i],
                            HashMap::new(),
                            HashMap::new(),
                        )
                    } else {
                        o.accumulate(&mut executors[i], &graphs[i], HashMap::new())
                    };
                    assert!(error.is_err());
                    assert!(o.accumulation_failed);
                    assert!(o.step(0.5).is_err());
                    assert_eq!(o.timestep, step);
                    assert!(o.gradients().is_err());
                    o.begin_step().unwrap();
                }
            }
            // Step 3 deliberately has no accumulation: old gradients must clear.
            if step != 3 {
                for x in [vec![1.0, -0.0, -2.0], vec![-0.5, 0.0, 1.0]] {
                    for (i, o) in owners.iter_mut().enumerate() {
                        let inputs = HashMap::from([("x".into(), x.clone())]);
                        if mixed {
                            o.accumulate_mixed(
                                &mut mixed_executors[i],
                                &graphs[i],
                                inputs,
                                HashMap::new(),
                            )
                            .unwrap();
                        } else {
                            o.accumulate(&mut executors[i], &graphs[i], inputs).unwrap();
                        }
                        assert!(!o.pending_zero);
                    }
                }
            }
            let statuses: Vec<_> = owners.iter_mut().map(|o| o.step(0.5).unwrap()).collect();
            assert_eq!(
                statuses[0].gradient_norm.to_bits(),
                statuses[1].gradient_norm.to_bits()
            );
            let gradients: Vec<_> = owners
                .iter()
                .zip(&parameters)
                .map(|(o, p)| {
                    o.gradients().unwrap()[&p.id()]
                        .iter()
                        .map(|x| x.to_bits())
                        .collect::<Vec<_>>()
                })
                .collect();
            assert_eq!(gradients[0], gradients[1]);
            if step == 3 {
                assert!(gradients[1].iter().all(|x| *x == 0));
            }
            let weights: Vec<_> = parameters
                .iter()
                .map(|p| {
                    p.values()
                        .unwrap()
                        .iter()
                        .map(|x| x.to_bits())
                        .collect::<Vec<_>>()
                })
                .collect();
            assert_eq!(weights[0], weights[1]);
            let states: Vec<_> = owners
                .iter()
                .zip(&parameters)
                .map(|(o, p)| o.moments().unwrap().remove(&p.id()).unwrap())
                .collect();
            assert_eq!(states[0], states[1]);
            for o in &owners {
                let mut raw = vec![0.0_f32; o.padded_len];
                o.stream
                    .memcpy_dtoh(o.gradients.as_ref(), &mut raw)
                    .unwrap();
                assert!(raw[3..].iter().all(|x| x.to_bits() == 0));
            }
        }
        let mut old = vec![0.0; 3];
        owners[1]
            .stream
            .memcpy_dtoh(retained.as_ref(), &mut old)
            .unwrap();
        assert_eq!(old, vec![0.2, -0.4, 0.1]);
    }
}

#[test]
fn first_accumulation_overwrite_preserves_groups_and_unaligned_parameter_ranges() {
    let parameters = [
        [
            Parameter::new(vec![0.2; 3], vec![3]),
            Parameter::new(vec![-0.4; 257], vec![257]),
        ],
        [
            Parameter::new(vec![0.2; 3], vec![3]),
            Parameter::new(vec![-0.4; 257], vec![257]),
        ],
    ];
    // Probe device availability through the existing strict CUDA-test convention.
    let Some(_) = owner(&parameters[0][0], false) else {
        return;
    };
    let mut owners: Vec<_> = parameters
        .iter()
        .map(|p| {
            ResidentAdam::new_with_graph_replay(
                vec![(p[0].clone(), 0), (p[1].clone(), 1)],
                vec![AdamGroup::new(0.01), AdamGroup::new(0.02)],
            )
            .unwrap()
        })
        .collect();
    owners[1]
        .set_experimental_first_accumulation_overwrite(true)
        .unwrap();
    let graphs: Vec<_> = parameters
        .iter()
        .map(|p| {
            let loss = TensorExpr::from(p[0].clone()).mean_all()
                + (TensorExpr::from(p[1].clone()) * TensorExpr::from(2.0_f32)).mean_all();
            let g: TensorGraph<f32> = loss.into();
            let last = *g.toposort().last().unwrap();
            g.with_gradients(last)
        })
        .collect();
    let mut executors: Vec<_> = owners.iter().map(|o| o.executor().unwrap()).collect();
    for batch_size in [1, 8] {
        for (i, o) in owners.iter_mut().enumerate() {
            o.set_update_batch_size(batch_size).unwrap();
            o.begin_step().unwrap();
            for _ in 0..2 {
                o.accumulate(&mut executors[i], &graphs[i], HashMap::new())
                    .unwrap();
            }
            o.step(0.5).unwrap();
        }
        for index in 0..2 {
            assert_eq!(
                parameters[0][index].values().unwrap(),
                parameters[1][index].values().unwrap()
            );
            let left = owners[0]
                .gradients()
                .unwrap()
                .remove(&parameters[0][index].id())
                .unwrap();
            let right = owners[1]
                .gradients()
                .unwrap()
                .remove(&parameters[1][index].id())
                .unwrap();
            assert_eq!(left, right);
        }
    }
}

#[test]
fn adam_256_threads_matches_default_bits_groups_tails_and_transactional_recovery() {
    let lengths = [1, 17, 255, 256, 257, 1023, 1024, 1025, 4097, 8193];
    let Some(_) = health_owner(&[1], true) else {
        return;
    };
    let make = || {
        let parameters = lengths
            .iter()
            .enumerate()
            .map(|(index, &len)| {
                let values = (0..len).map(|i| (i % 11) as f32 / 16. - 0.25).collect();
                (Parameter::new(values, vec![len]), index % 2)
            })
            .collect();
        let mut second = AdamGroup::new(0.02);
        second.beta1 = 0.8;
        ResidentAdam::initialize(parameters, vec![AdamGroup::new(0.01), second], true).unwrap()
    };
    let mut owners = [make(), make()];
    assert!(!owners[0].adam_256_threads && !owners[1].adam_256_threads);
    owners[1].set_experimental_adam_256_threads(true).unwrap();
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    let retained = owners[1].parameters[8].weight.clone();
    let retained_values = owners[1].stream.memcpy_dtov(retained.as_ref()).unwrap();
    for batch in [1, 8, 1] {
        let mut statuses = Vec::new();
        for o in &mut owners {
            o.set_update_batch_size(batch).unwrap();
            fill_health_gradients(o);
            assert!(o.set_experimental_adam_256_threads(false).is_err());
            statuses.push(o.step(0.2).unwrap());
        }
        assert_eq!(statuses[0].timestep, statuses[1].timestep);
        assert_eq!(
            statuses[0].gradient_norm.to_bits(),
            statuses[1].gradient_norm.to_bits()
        );
        assert_eq!(
            statuses[0].gradient_scale.to_bits(),
            statuses[1].gradient_scale.to_bits()
        );
        assert_eq!(statuses[0].kernel_launches, statuses[1].kernel_launches);
        for (a, b) in owners[0].parameters.iter().zip(&owners[1].parameters) {
            assert_eq!(
                bits(&owners[0].stream.memcpy_dtov(a.weight.as_ref()).unwrap()),
                bits(&owners[1].stream.memcpy_dtov(b.weight.as_ref()).unwrap())
            );
            assert_eq!(
                bits(&owners[0].gradients().unwrap()[&a.id]),
                bits(&owners[1].gradients().unwrap()[&b.id])
            );
        }
        assert_eq!(
            bits(&owners[0].stream.memcpy_dtov(&owners[0].moment).unwrap()),
            bits(&owners[1].stream.memcpy_dtov(&owners[1].moment).unwrap())
        );
        assert_eq!(
            bits(&owners[0].stream.memcpy_dtov(&owners[0].variance).unwrap()),
            bits(&owners[1].stream.memcpy_dtov(&owners[1].variance).unwrap())
        );
        assert_eq!(
            bits(
                &owners[0]
                    .stream
                    .memcpy_dtov(owners[0].staging.health.as_ref())
                    .unwrap()
            ),
            bits(
                &owners[1]
                    .stream
                    .memcpy_dtov(owners[1].staging.health.as_ref())
                    .unwrap()
            )
        );
    }
    // Poison the last valid element beyond both launch widths and the batch
    // threshold. Reject the entire staged update without publishing any state.
    for o in &mut owners {
        o.set_update_batch_size(8).unwrap();
        let target = o.parameters.len() - 1;
        let parameter = &o.parameters[target];
        parameter.storage.lock().unwrap()[lengths[target] - 1] = f32::MAX;
        let group = parameter.group;
        o.groups[group].weight_decay = 2.0;
        o.begin_step().unwrap();
        let before: Vec<_> = o
            .parameters
            .iter()
            .map(|p| bits(&o.stream.memcpy_dtov(p.weight.as_ref()).unwrap()))
            .collect();
        let moment = bits(&o.stream.memcpy_dtov(&o.moment).unwrap());
        let variance = bits(&o.stream.memcpy_dtov(&o.variance).unwrap());
        let error = o.step(0.0).unwrap_err();
        assert!(
            error.to_string().contains("Adam produced nonfinite"),
            "{error:#}"
        );
        assert_eq!(o.timestep, 3);
        assert!(o.gradients().is_err());
        assert_eq!(moment, bits(&o.stream.memcpy_dtov(&o.moment).unwrap()));
        assert_eq!(variance, bits(&o.stream.memcpy_dtov(&o.variance).unwrap()));
        for (p, expected) in o.parameters.iter().zip(before) {
            assert_eq!(
                expected,
                bits(&o.stream.memcpy_dtov(p.weight.as_ref()).unwrap())
            );
        }
        let health = o.stream.memcpy_dtov(o.staging.health.as_ref()).unwrap();
        let bad = o.parameters[target].health_range.start
            + (lengths[target] - 1) / kernels::HEALTH_GROUP_SIZE;
        assert!(health[bad].is_nan());
        o.groups[group].weight_decay = 0.0;
        o.parameters[target].storage.lock().unwrap().fill(0.125);
        fill_health_gradients(o);
        assert_eq!(o.step(0.2).unwrap().timestep, 4);
        assert!(
            o.stream
                .memcpy_dtov(o.staging.health.as_ref())
                .unwrap()
                .iter()
                .all(|v| v.to_bits() == 0)
        );
    }
    for (a, b) in owners[0].parameters.iter().zip(&owners[1].parameters) {
        assert_eq!(
            bits(&owners[0].stream.memcpy_dtov(a.weight.as_ref()).unwrap()),
            bits(&owners[1].stream.memcpy_dtov(b.weight.as_ref()).unwrap())
        );
    }
    assert_eq!(
        bits(&owners[0].stream.memcpy_dtov(&owners[0].moment).unwrap()),
        bits(&owners[1].stream.memcpy_dtov(&owners[1].moment).unwrap())
    );
    assert_eq!(
        bits(&owners[0].stream.memcpy_dtov(&owners[0].variance).unwrap()),
        bits(&owners[1].stream.memcpy_dtov(&owners[1].variance).unwrap())
    );
    assert_eq!(
        bits(&retained_values),
        bits(&owners[1].stream.memcpy_dtov(retained.as_ref()).unwrap())
    );
    owners[1].set_experimental_adam_256_threads(false).unwrap();
    assert!(!owners[1].adam_256_threads);
}
