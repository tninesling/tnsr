use super::*;
use crate::ptx::mixed::MixedOptions;
use crate::tensor::{TensorExpr, ValueDType};
use crate::tile::MatMulPrecision;
use cudarc::driver::DevicePtr;

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|v| v.to_bits()).collect()
}
fn address(v: &CudaSlice<f32>, o: &ResidentAdam) -> u64 {
    v.device_ptr(&o.stream).0
}
fn model(
    p: &Parameter<f32>,
    b: &Parameter<f32>,
) -> (TensorGraph<f32, WithGrad>, crate::graph::NodeIndex) {
    let w = TensorExpr::from(p.clone()).transpose();
    let y = TensorExpr::input("x", vec![4, 2])
        .linear_bf16(w.clone(), b.clone().into())
        .unwrap()
        .cast(ValueDType::F32);
    let mut g = TensorGraph::new();
    let alias = g.add_expr(&w, false);
    let loss = g.add_expr(&y.mean_all(), false);
    (g.with_gradients(loss), alias)
}
#[test]
fn direct_masters_preserve_snapshots_reclaim_two_executors_and_bound_replay_versions() {
    for replay in [false, true] {
        check_direct_masters(replay);
    }
}
fn check_direct_masters(replay: bool) {
    let parameters = [
        (
            Parameter::new(vec![0.125, -0.25, 0.375, 0.5, -0.125, 0.25], vec![3, 2]),
            Parameter::new(vec![0.125, -0.25, 0.375], vec![3]),
        ),
        (
            Parameter::new(vec![0.125, -0.25, 0.375, 0.5, -0.125, 0.25], vec![3, 2]),
            Parameter::new(vec![0.125, -0.25, 0.375], vec![3]),
        ),
    ];
    let initialize = |index: usize| {
        ResidentAdam::initialize(
            vec![
                (parameters[index].0.clone(), 0),
                (parameters[index].1.clone(), 1),
            ],
            vec![AdamGroup::new(0.01), AdamGroup::new(0.02)],
            replay,
        )
    };
    let baseline = match initialize(0) {
        Ok(v) => v,
        Err(e)
            if std::env::var("TNSR_REQUIRE_CUDA").as_deref() != Ok("1")
                && e.to_string()
                    .starts_with("initialize resident training CUDA device") =>
        {
            return;
        }
        Err(e) => panic!("{e:#}"),
    };
    let mut owners = [baseline, initialize(1).unwrap()];
    let models = [
        model(&parameters[0].0, &parameters[0].1),
        model(&parameters[1].0, &parameters[1].1),
    ];
    for o in &mut owners {
        o.set_experimental_first_accumulation_overwrite(true)
            .unwrap();
        o.begin_step().unwrap();
    }
    let mut executors: Vec<Vec<_>> = owners
        .iter()
        .enumerate()
        .map(|(i, o)| {
            (0..2)
                .map(|_| {
                    o.mixed_executor_with_options(
                        &models[i].0,
                        MatMulPrecision::Tf32x3,
                        MixedOptions {
                            direct_master_bindings: i == 1,
                            incremental_allocation_reuse: true,
                            ..MixedOptions::default()
                        },
                    )
                    .unwrap()
                })
                .collect()
        })
        .collect();
    let master_bytes = (6 + 3) * 4;
    assert_eq!(executors[1][0].info().direct_master_bytes, master_bytes);
    assert_eq!(executors[1][0].info().direct_master_parameters, 2);
    assert_eq!(
        executors[0][0].info().f32_bytes - executors[1][0].info().f32_bytes,
        master_bytes
    );
    let mut old_version = None;
    let mut expected_sources = [Vec::new(), Vec::new()];
    for step in 0..6 {
        for o in &mut owners {
            o.begin_step().unwrap();
        }
        if step == 2 {
            let value = owners[1].parameters[0].weight.clone();
            let host = owners[1].stream.memcpy_dtov(value.as_ref()).unwrap();
            old_version = Some((value, host));
        }
        let current = owners[1]
            .stream
            .memcpy_dtov(owners[1].parameters[0].weight.as_ref())
            .unwrap();
        let transpose: Vec<_> = [0, 2, 4, 1, 3, 5].into_iter().map(|i| current[i]).collect();
        for (micro, expected_source) in expected_sources.iter_mut().enumerate() {
            // One executor intentionally keeps a stale uploaded source version.
            if step == 4 && micro == 1 {
                continue;
            }
            for i in 0..2 {
                owners[i]
                    .accumulate_mixed(
                        &mut executors[i][micro],
                        &models[i].0,
                        HashMap::from([(
                            "x".into(),
                            vec![0.25, -0.5, 0.75, 0.5, -0.25, 0.125, 0.625, -0.375],
                        )]),
                        HashMap::new(),
                    )
                    .unwrap();
            }
            *expected_source = transpose.clone();
        }
        if step == 1 {
            // Both executors released the initial active leases. Every retired
            // allocation can now become the next staged output without copying.
            assert!(
                owners[1]
                    .parameters
                    .iter()
                    .all(|p| p.staged_weight.is_some())
            );
        }
        if step == 3 {
            // The explicit old-version reader prevents recycling that master.
            assert!(owners[1].parameters[0].staged_weight.is_none());
        }
        let statuses: Vec<_> = owners.iter_mut().map(|o| o.step(0.3).unwrap()).collect();
        assert_eq!(
            statuses[0].gradient_norm.to_bits(),
            statuses[1].gradient_norm.to_bits()
        );
        assert_eq!(
            statuses[0].gradient_scale.to_bits(),
            statuses[1].gradient_scale.to_bits()
        );
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
        for (micro, executor) in executors[1].iter().enumerate() {
            // Publication must not change what a stale mixed executor reads.
            assert_eq!(executor.read(models[1].1).unwrap(), expected_sources[micro]);
            assert_eq!(executor.transfers().parameter_copies, 0);
            assert_eq!(executor.transfers().parameter_bytes, 0);
            assert!(executor.info().direct_capture_variants <= 2);
        }
    }
    let (retained, host) = old_version.unwrap();
    assert_eq!(
        bits(&host),
        bits(&owners[1].stream.memcpy_dtov(retained.as_ref()).unwrap())
    );
    if replay {
        assert!(
            executors[1]
                .iter()
                .all(|e| e.info().direct_capture_hits > 0 && e.info().direct_capture_misses >= 2)
        );
    } else {
        assert!(
            executors[1]
                .iter()
                .all(|e| e.info().direct_capture_variants == 0
                    && e.info().direct_capture_misses == 0)
        );
    }
    let current = address(owners[1].parameters[0].weight.as_ref(), &owners[1]);
    assert_ne!(current, address(retained.as_ref(), &owners[1]));
}

#[test]
fn direct_masters_keep_published_readers_and_replay_after_nonfinite_update_recovery() {
    let p = Parameter::new(vec![0.125, -0.25, 0.375, 0.5, -0.125, 0.25], vec![3, 2]);
    let b = Parameter::new(vec![0.125, -0.25, 0.375], vec![3]);
    let mut owner = match ResidentAdam::new_with_graph_replay(
        vec![(p.clone(), 0), (b.clone(), 0)],
        vec![AdamGroup::new(0.01)],
    ) {
        Ok(v) => v,
        Err(e)
            if std::env::var("TNSR_REQUIRE_CUDA").as_deref() != Ok("1")
                && e.to_string()
                    .starts_with("initialize resident training CUDA device") =>
        {
            return;
        }
        Err(e) => panic!("{e:#}"),
    };
    owner
        .set_experimental_first_accumulation_overwrite(true)
        .unwrap();
    owner.begin_step().unwrap();
    let (graph, alias) = model(&p, &b);
    let mut executor = owner
        .mixed_executor_with_options(
            &graph,
            MatMulPrecision::Tf32x3,
            MixedOptions {
                direct_master_bindings: true,
                ..MixedOptions::default()
            },
        )
        .unwrap();
    let accumulate = |o: &mut ResidentAdam, e: &mut crate::ptx::mixed::MixedExecutor| {
        o.begin_step().unwrap();
        o.accumulate_mixed(
            e,
            &graph,
            HashMap::from([(
                "x".into(),
                vec![0.25, -0.5, 0.75, 0.5, -0.25, 0.125, 0.625, -0.375],
            )]),
            HashMap::new(),
        )
        .unwrap();
    };
    accumulate(&mut owner, &mut executor);
    owner.step(0.3).unwrap();
    let retained = owner.parameters[0].weight.clone();
    let published = owner.stream.memcpy_dtov(retained.as_ref()).unwrap();
    let moment = owner.stream.memcpy_dtov(&owner.moment).unwrap();
    let variance = owner.stream.memcpy_dtov(&owner.variance).unwrap();
    let weights: Vec<_> = owner
        .parameters
        .iter()
        .map(|p| owner.stream.memcpy_dtov(p.weight.as_ref()).unwrap())
        .collect();
    owner.groups[0].weight_decay = f32::MAX;
    accumulate(&mut owner, &mut executor);
    let uploaded = executor.read(alias).unwrap();
    let misses = executor.info().direct_capture_misses;
    let error = owner.step(0.3).unwrap_err();
    assert!(
        error.to_string().contains("Adam produced nonfinite"),
        "{error:#}"
    );
    assert_eq!(owner.timestep, 1);
    assert_eq!(
        bits(&moment),
        bits(&owner.stream.memcpy_dtov(&owner.moment).unwrap())
    );
    assert_eq!(
        bits(&variance),
        bits(&owner.stream.memcpy_dtov(&owner.variance).unwrap())
    );
    for (p, expected) in owner.parameters.iter().zip(weights) {
        assert_eq!(
            bits(&expected),
            bits(&owner.stream.memcpy_dtov(p.weight.as_ref()).unwrap())
        );
    }
    assert_eq!(executor.read(alias).unwrap(), uploaded);
    owner.groups[0].weight_decay = 0.0;
    accumulate(&mut owner, &mut executor);
    assert_eq!(executor.info().direct_capture_misses, misses);
    assert_eq!(owner.step(0.3).unwrap().timestep, 2);
    assert_eq!(executor.read(alias).unwrap(), uploaded);
    assert_eq!(
        bits(&published),
        bits(&owner.stream.memcpy_dtov(retained.as_ref()).unwrap())
    );
    assert!(
        owner
            .gradients()
            .unwrap()
            .values()
            .flatten()
            .all(|v| v.is_finite())
    );
    assert_eq!(executor.transfers().parameter_copies, 0);
    assert_eq!(executor.transfers().parameter_bytes, 0);
}
