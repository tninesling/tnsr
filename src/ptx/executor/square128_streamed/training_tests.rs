//! Resident publication/replay qualification of the initial plain-MatMul scope.
use super::*;
use crate::{
    ptx::{
        mixed::{MixedExecutor, MixedOptions},
        resident::{AdamGroup, ResidentAdam},
    },
    tensor::{Parameter, TensorExpr, ValueDType},
    tile::{LargeBf16TilePolicy as Policy, MatMulPrecision},
};

fn bits(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}
fn device() -> bool {
    if let Err(error) = CudaContext::new(0) {
        assert!(
            std::env::var("TNSR_REQUIRE_CUDA").as_deref() != Ok("1"),
            "{error:#}"
        );
        false
    } else {
        true
    }
}

#[test]
#[ignore = "Square resource admission not achieved; candidate state unqualified"]
fn square128_streamed_resident_microbatches_replay_full_state_and_recovery() -> Result<()> {
    if !device() {
        return Ok(());
    }
    let (rows, k, n) = (2048, 128, 1024);
    let mut parameters = Vec::new();
    let mut graphs = Vec::new();
    for _ in 0..2 {
        let x = Parameter::new(
            (0..rows * k)
                .map(|i| (i % 17) as f32 / 64. - 0.125)
                .collect(),
            vec![rows, k],
        );
        let w = Parameter::new(
            (0..k * n).map(|i| (i % 13) as f32 / 64. - 0.0625).collect(),
            vec![k, n],
        );
        let left = (TensorExpr::from(x.clone()) + TensorExpr::input("delta", vec![rows, k]))
            .cast(ValueDType::BF16);
        let y = left
            .matmul(TensorExpr::from(w.clone()).cast(ValueDType::BF16))
            .cast(ValueDType::F32);
        let mut graph = TensorGraph::new();
        let loss = graph.add_expr(
            &(y * TensorExpr::input("seed", vec![rows, n])).mean_all(),
            false,
        );
        graphs.push((graph.with_gradients(loss), loss));
        parameters.push([x, w]);
    }
    let mut owners = parameters
        .iter()
        .map(|p| {
            ResidentAdam::new_with_graph_replay(
                p.iter().cloned().map(|p| (p, 0)).collect(),
                vec![AdamGroup::new(1e-4)],
            )
        })
        .collect::<Result<Vec<_>>>()?;
    for owner in &mut owners {
        owner.set_update_batch_size(8)?;
        owner.set_experimental_first_accumulation_overwrite(true)?;
        owner.set_experimental_gradient_norm_partials(true)?;
        owner.begin_step()?;
    }
    let mut executors = owners
        .iter()
        .enumerate()
        .map(|(i, owner)| {
            owner.mixed_executor_with_options(
                &graphs[i].0,
                MatMulPrecision::Tf32x3,
                MixedOptions {
                    large_bf16_tile_policy: if i == 0 {
                        Policy::Off
                    } else {
                        Policy::Square128Streamed
                    },
                    shared_memory_reuse: true,
                    physical_order_casts: true,
                    incremental_allocation_reuse: true,
                    direct_master_bindings: true,
                    ..MixedOptions::default()
                },
            )
        })
        .collect::<Result<Vec<_>>>()?;
    assert_eq!(
        executors[0].info().large_bf16_square128_streamed_programs,
        0
    );
    assert_eq!(
        executors[1].info().large_bf16_square128_streamed_programs,
        1
    );
    assert_eq!(
        executors[1]
            .info()
            .large_bf16_square128_streamed_fallback_programs,
        0
    );
    for (phase, microbatches) in [1, 2, 1].into_iter().enumerate() {
        for i in 0..2 {
            if phase > 0 {
                owners[i].begin_step()?;
            }
            for micro in 0..microbatches {
                owners[i].accumulate_mixed(
                    &mut executors[i],
                    &graphs[i].0,
                    HashMap::from([
                        (
                            "delta".into(),
                            vec![(phase + micro) as f32 / 128.; rows * k],
                        ),
                        (
                            "seed".into(),
                            (0..rows * n)
                                .map(|j| ((j + phase + micro) % 5) as f32 / 16. - 0.125)
                                .collect(),
                        ),
                    ]),
                    HashMap::new(),
                )?;
            }
        }
        let a = owners[0].step(1.)?;
        let b = owners[1].step(1.)?;
        assert_eq!(a.timestep, b.timestep);
        assert_eq!(a.gradient_norm.to_bits(), b.gradient_norm.to_bits());
        assert_eq!(a.gradient_scale.to_bits(), b.gradient_scale.to_bits());
        bits(
            &executors[0].read(graphs[0].1)?,
            &executors[1].read(graphs[1].1)?,
        );
        let gradients = [owners[0].gradients()?, owners[1].gradients()?];
        let moments = [owners[0].moments()?, owners[1].moments()?];
        for j in 0..2 {
            let (a, b) = (&parameters[0][j], &parameters[1][j]);
            bits(&gradients[0][&a.id()], &gradients[1][&b.id()]);
            bits(&a.values()?, &b.values()?);
            bits(&moments[0][&a.id()].first, &moments[1][&b.id()].first);
            bits(&moments[0][&a.id()].second, &moments[1][&b.id()].second);
        }
        if phase == 1 {
            for i in 0..2 {
                let before = parameters[i]
                    .iter()
                    .map(|p| p.values())
                    .collect::<Result<Vec<_>>>()?;
                let moments = owners[i].moments()?;
                owners[i].begin_step()?;
                let _ = owners[i].accumulate_mixed(
                    &mut executors[i],
                    &graphs[i].0,
                    HashMap::from([
                        ("delta".into(), vec![0.; rows * k]),
                        ("seed".into(), vec![f32::NAN; rows * n]),
                    ]),
                    HashMap::new(),
                );
                assert!(owners[i].step(1.).is_err());
                let now = owners[i].moments()?;
                for (j, p) in parameters[i].iter().enumerate() {
                    bits(&before[j], &p.values()?);
                    bits(&moments[&p.id()].first, &now[&p.id()].first);
                    bits(&moments[&p.id()].second, &now[&p.id()].second);
                }
            }
        }
    }
    for e in &executors {
        assert!(e.info().replay);
        assert_eq!(e.info().direct_capture_variants, 2);
        assert_eq!(e.info().direct_capture_misses, 2);
        assert!(e.info().direct_capture_hits > 0);
        assert_eq!(e.transfers().parameter_copies, 0);
        assert_eq!(e.transfers().parameter_bytes, 0);
    }
    Ok(())
}

#[test]
fn square128_streamed_linear_bias_keeps_complete_original_kernel() -> Result<()> {
    if !device() {
        return Ok(());
    }
    let (m, k, n) = (2048, 128, 1024);
    let x = (0..m * k)
        .map(|i| if i % k == 0 { 1.0078125 } else { 0. })
        .collect();
    let w = (0..k * n)
        .map(|i| if i < n { 1.0078125 } else { 0. })
        .collect();
    let y = TensorExpr::constant(x, vec![m, k])
        .linear_bf16(
            TensorExpr::constant(w, vec![k, n]),
            TensorExpr::constant(vec![-1.015625; n], vec![n]),
        )?
        .cast(ValueDType::F32);
    let graph: TensorGraph<f32> = y.into();
    let root = graph.graph.node_indices().next_back().unwrap();
    for policy in [Policy::Off, Policy::Square128Streamed] {
        let mut e = MixedExecutor::new_with_options(
            &graph,
            MatMulPrecision::NativeHalf,
            false,
            MixedOptions {
                shared_memory_reuse: true,
                large_bf16_tile_policy: policy,
                ..MixedOptions::default()
            },
        )?;
        assert_eq!(e.info().large_bf16_square128_streamed_programs, 0);
        assert_eq!(e.info().large_bf16_square128_streamed_fallback_programs, 0);
        e.upload(HashMap::new())?;
        e.run()?;
        assert!(
            e.read(root)?
                .iter()
                .all(|v| v.to_bits() == (1.0f32 / 16384.).to_bits())
        );
    }
    Ok(())
}
