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
