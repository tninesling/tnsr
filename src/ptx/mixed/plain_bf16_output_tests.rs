use super::*;
fn check(batch: usize) -> Result<()> {
    if !super::tests::cuda_available() {
        return Ok(());
    }
    super::batched_attention_tile_tests::check_actual_decoder_training_with_options(
        crate::tile::BatchedAttentionBf16TilePolicy::TallScoresByGrid,
        super::test_decoder::Config {
            width: 512,
            layers: 4,
            heads: 8,
            hidden: 1408,
            sequence: 512,
            batch,
            vocab: 4096,
            epsilon: 1e-5,
            learning_rate: 1e-4,
            max_grad_norm: 1.,
        },
        MixedOptions {
            plain_bf16_output: true,
            layer_norm_pair_fusion: true,
            indexed_loss_cooperative_ordered_backward: true,
            indexed_loss_cooperative_ordered_forward: true,
            attention_training_forward_fusion: batch >= 16,
            split_k_bf16_tile_policy: crate::tile::SplitKBf16TilePolicy::Square64,
            ..Default::default()
        },
    )
}
#[test]
fn plain_bf16_output_actual_b8_decoder_exact_three_updates() -> Result<()> {
    check(8)
}
#[test]
fn plain_bf16_output_actual_b16_decoder_ce_attention_exact_three_updates() -> Result<()> {
    check(16)
}

#[test]
fn plain_bf16_output_resident_nonfinite_rejection_and_replay_recovery() -> Result<()> {
    use crate::{
        ptx::resident::{AdamGroup, ResidentAdam},
        tensor::{Parameter, TensorExpr, ValueDType},
    };
    if !super::tests::cuda_available() {
        return Ok(());
    }
    let mut parameters = Vec::new();
    let mut graphs = Vec::new();
    let mut owners = Vec::new();
    let mut executors = Vec::new();
    for enabled in [false, true] {
        let p = Parameter::new(
            (0..512 * 512)
                .map(|i| (i % 17) as f32 / 64. - 0.125)
                .collect(),
            vec![512, 512],
        );
        let y = TensorExpr::input("x", vec![2048, 512])
            .cast(ValueDType::BF16)
            .matmul(TensorExpr::from(p.clone()).cast(ValueDType::BF16))
            .cast(ValueDType::F32)
            .mean_all();
        let mut graph = TensorGraph::new();
        let root = graph.add_expr(&y, false);
        let graph = graph.with_gradients(root);
        let mut owner =
            ResidentAdam::new_with_graph_replay(vec![(p.clone(), 0)], vec![AdamGroup::new(1e-4)])?;
        owner.set_experimental_first_accumulation_overwrite(true)?;
        owner.set_experimental_gradient_norm_partials(true)?;
        owner.begin_step()?;
        let e = owner.mixed_executor_with_options(
            &graph,
            MatMulPrecision::Tf32x3,
            MixedOptions {
                shared_memory_reuse: true,
                plain_bf16_output: enabled,
                physical_order_casts: true,
                direct_master_bindings: true,
                incremental_allocation_reuse: true,
                ..Default::default()
            },
        )?;
        assert_eq!(e.info().plain_bf16_output_programs, usize::from(enabled));
        parameters.push(p);
        graphs.push(graph);
        owners.push(owner);
        executors.push(e);
    }
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    let mut committed = None;
    for phase in 0..3 {
        let mut statuses = Vec::new();
        for i in 0..2 {
            if phase > 0 {
                owners[i].begin_step()?;
            }
            let x = if phase == 1 {
                vec![f32::NAN; 2048 * 512]
            } else {
                (0..2048 * 512)
                    .map(|j| ((j + phase) % 13) as f32 / 32. - 0.125)
                    .collect()
            };
            owners[i].accumulate_mixed(
                &mut executors[i],
                &graphs[i],
                HashMap::from([("x".into(), x)]),
                HashMap::new(),
            )?;
            statuses.push(owners[i].step(1.));
        }
        if phase == 1 {
            assert!(statuses.iter().all(Result::is_err));
            let (a, m, v) = committed.as_ref().unwrap();
            for i in 0..2 {
                assert_eq!(&bits(&parameters[i].values()?), a);
                let moments = owners[i].moments()?;
                assert_eq!(&bits(&moments[&parameters[i].id()].first), m);
                assert_eq!(&bits(&moments[&parameters[i].id()].second), v);
            }
        } else {
            let mut statuses = statuses.into_iter();
            let (a, b) = (statuses.next().unwrap()?, statuses.next().unwrap()?);
            assert_eq!(a.gradient_norm.to_bits(), b.gradient_norm.to_bits());
            assert_eq!(a.gradient_scale.to_bits(), b.gradient_scale.to_bits());
            let (g0, g1) = (owners[0].gradients()?, owners[1].gradients()?);
            assert_eq!(
                bits(&g0[&parameters[0].id()]),
                bits(&g1[&parameters[1].id()])
            );
            let (m0, m1) = (owners[0].moments()?, owners[1].moments()?);
            let (a, b) = (&m0[&parameters[0].id()], &m1[&parameters[1].id()]);
            assert_eq!(bits(&a.first), bits(&b.first));
            assert_eq!(bits(&a.second), bits(&b.second));
            assert_eq!(
                bits(&parameters[0].values()?),
                bits(&parameters[1].values()?)
            );
            committed = Some((
                bits(&parameters[0].values()?),
                bits(&a.first),
                bits(&a.second),
            ));
        }
    }
    assert_eq!(executors[1].transfers().parameter_copies, 0);
    assert_eq!(executors[1].transfers().parameter_bytes, 0);
    Ok(())
}
