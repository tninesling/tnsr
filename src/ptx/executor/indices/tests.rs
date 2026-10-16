use super::*;
use crate::tensor::{IndexTensor, Parameter, TensorExpr};
use crate::{Executor, SimpleExecutor};

fn cuda<D: CudaDType>() -> Option<PtxExecutor<D>> {
    match PtxExecutor::try_new_for_dtype() {
        Ok(executor) => Some(executor),
        Err(error)
            if std::env::var("TNSR_REQUIRE_CUDA").as_deref() != Ok("1")
                && error
                    .to_string()
                    .starts_with("Failed to initialize CUDA device 0") =>
        {
            None
        }
        Err(error) => panic!("{error:#}"),
    }
}

#[test]
fn exact_cuda_lookup_preserves_integer_above_f32_range() {
    let Some(mut executor) = cuda::<F32>() else {
        return;
    };
    let last = 16_777_217usize;
    let mut weight = vec![0.0f32; last + 1];
    weight[last - 1] = 3.0;
    weight[last] = 7.0;
    let graph: TensorGraph<f32> = TensorExpr::constant(weight, vec![last + 1, 1])
        .embedding_indices(
            IndexTensor::constant(vec![u32::try_from(last).unwrap()], vec![1]).unwrap(),
        )
        .into();
    assert_eq!(executor.execute(&graph, HashMap::new()).unwrap(), vec![7.0]);
    let ptx = executor.module_source().unwrap();
    assert!(ptx.contains("ld.global.u32"));
    assert!(!ptx.contains("cvt.rzi.u64.f32"));
}

fn check_half_lookup<D: CudaDType>()
where
    D::HostType: std::fmt::Debug,
{
    let Some(mut executor) = cuda::<D>() else {
        return;
    };
    let data = (0..683)
        .map(|i| <D::HostType as num_traits::NumCast>::from((i % 16) as f32).unwrap())
        .collect::<Vec<_>>();
    let weight = Parameter::new(data, vec![683, 1]);
    let y = TensorExpr::from(weight.clone()).embedding_indices(IndexTensor::input("ids", vec![3]));
    let graph: TensorGraph<D::HostType> = y.into();
    let feeds = HashMap::from([("ids".into(), vec![257, 365, 365])]);
    let actual = executor
        .execute_with_indices(&graph, HashMap::new(), feeds.clone())
        .unwrap();
    let expected: Vec<_> = [1.0f32, 13.0, 13.0]
        .iter()
        .map(|&x| <D::HostType as num_traits::NumCast>::from(x).unwrap())
        .collect();
    assert_eq!(actual, expected);
    // Half weights/output occupy 2 bytes, while exact indices always occupy 4.
    assert_eq!(executor.stats().bytes_allocated, 683 * 2 + 3 * 2 + 3 * 4);
    assert_eq!(executor.stats().buffers_allocated, 3);
    executor.reset_stats();
    executor
        .execute_with_indices(&graph, HashMap::new(), feeds.clone())
        .unwrap();
    assert_eq!(executor.stats().bytes_allocated, 0);
    assert_eq!(executor.stats().buffers_allocated, 0);
    let loss: TensorGraph<D::HostType> = TensorExpr::from(weight.clone())
        .embedding_indices(IndexTensor::input("ids", vec![3]))
        .reduce_sum(1)
        .reduce_sum(0)
        .into();
    let output = *loss.toposort().last().unwrap();
    let loss = loss.with_gradients(output);
    executor
        .execute_with_indices(&loss, HashMap::new(), feeds)
        .unwrap();
    let gradients = executor.get_gradients(&loss);
    assert_eq!(gradients[&weight.id()][257].to_f32(), Some(1.0));
    assert_eq!(gradients[&weight.id()][365].to_f32(), Some(2.0));
    assert_eq!(gradients[&weight.id()][364].to_f32(), Some(0.0));

    // More lookups than vocabulary rows must still write the entire result.
    let small: TensorGraph<D::HostType> = TensorExpr::constant(
        [1.0f32, 2.0]
            .into_iter()
            .map(|x| <D::HostType as num_traits::NumCast>::from(x).unwrap())
            .collect(),
        vec![2, 1],
    )
    .embedding_indices(IndexTensor::constant(vec![1, 0, 1, 1, 0], vec![5]).unwrap())
    .into();
    let output = executor.execute(&small, HashMap::new()).unwrap();
    assert_eq!(
        output.iter().map(ToPrimitive::to_f32).collect::<Vec<_>>(),
        vec![Some(2.0), Some(1.0), Some(2.0), Some(2.0), Some(1.0)]
    );

    let mut logits = vec![0.0f32; 683];
    logits[364] = -1.0;
    logits[365] = 1.0;
    let master = Parameter::new(
        logits
            .iter()
            .map(|&x| <D::HostType as num_traits::NumCast>::from(x).unwrap())
            .collect(),
        vec![1, 683],
    );
    let graph: TensorGraph<D::HostType> = TensorExpr::from(master.clone())
        .indexed_cross_entropy_indices(IndexTensor::constant(vec![365], vec![1]).unwrap())
        .mean_all()
        .into();
    let loss = *graph.toposort().last().unwrap();
    let graph = graph.with_gradients(loss);
    executor.execute(&graph, HashMap::new()).unwrap();
    let normalizer = 681.0f32 + (-1.0f32).exp() + 1.0f32.exp();
    let expected_loss = normalizer.ln() - 1.0;
    assert!((executor.get_value(loss).unwrap()[0].to_f32().unwrap() - expected_loss).abs() < 0.04);
    let gradients = executor.get_gradients(&graph);
    for (index, gradient) in gradients[&master.id()].iter().enumerate() {
        let reference = logits[index].exp() / normalizer - if index == 365 { 1.0 } else { 0.0 };
        assert!((gradient.to_f32().unwrap() - reference).abs() < 0.003);
    }
}

#[test]
fn exact_cuda_half_lookup_forward_and_backward() {
    check_half_lookup::<crate::ptx::types::F16>();
    check_half_lookup::<crate::ptx::types::BF16>();
}

#[test]
fn exact_cuda_targets_match_cpu_and_reject_invalid_ids() {
    let Some(mut executor) = cuda::<F32>() else {
        return;
    };
    let weight = Parameter::new(vec![0.0f32; 2 * 683], vec![2, 683]);
    let graph: TensorGraph<f32> = TensorExpr::from(weight.clone())
        .indexed_cross_entropy_indices(IndexTensor::input("targets", vec![2]))
        .mean_all()
        .into();
    let output = *graph.toposort().last().unwrap();
    let graph = graph.with_gradients(output);
    let feeds = HashMap::from([("targets".into(), vec![257, 365])]);
    let mut cpu = SimpleExecutor::new();
    cpu.execute_with_indices(&graph, HashMap::new(), feeds.clone())
        .unwrap();
    executor
        .execute_with_indices(&graph, HashMap::new(), feeds.clone())
        .unwrap();
    let expected = cpu.get_gradients(&graph);
    let actual = executor.get_gradients(&graph);
    for (a, b) in actual[&weight.id()].iter().zip(&expected[&weight.id()]) {
        assert!((a - b).abs() < 1e-6);
    }
    assert!(
        (executor.get_value(output).unwrap()[0] - cpu.get_value(output).unwrap()[0]).abs() < 1e-5
    );
    let error = executor
        .execute_with_indices(
            &graph,
            HashMap::new(),
            HashMap::from([("targets".into(), vec![257, u32::MAX])]),
        )
        .unwrap_err();
    assert!(error.to_string().contains("4294967295"));
    executor
        .execute_with_indices(&graph, HashMap::new(), feeds)
        .unwrap();
    let overflowing: TensorGraph<f32> = TensorExpr::constant(vec![1.0], vec![1, 1])
        .embedding_indices(IndexTensor::input("ids", vec![usize::MAX, 2]))
        .into();
    assert!(
        executor
            .compile(&overflowing)
            .unwrap_err()
            .to_string()
            .contains("overflows")
    );
}

#[test]
fn exact_cuda_replay_stages_changed_ids_and_constants_for_both_shapes() {
    use crate::ptx::resident::{AdamGroup, ResidentAdam};
    use crate::tensor::indices::IndexSource;
    let Some(_) = cuda::<F32>() else { return };
    let weight = Parameter::new(
        (0..683).map(|i| (i % 7) as f32 * 0.01).collect(),
        vec![683, 1],
    );
    let mut owner =
        ResidentAdam::new_with_graph_replay(vec![(weight.clone(), 0)], vec![AdamGroup::new(0.001)])
            .unwrap();
    let mut executors = [owner.executor().unwrap(), owner.executor().unwrap()];
    let mut graphs = [3, 1].map(|n| {
        let x =
            TensorExpr::from(weight.clone()).embedding_indices(IndexTensor::input("ids", vec![n]));
        let fixed = TensorExpr::from(weight.clone())
            .embedding_indices(IndexTensor::constant(vec![257], vec![1]).unwrap());
        let graph: TensorGraph<f32> = ((x.clone() * x).reduce_sum(1).reduce_sum(0)
            + (fixed.clone() * fixed).reduce_sum(1).reduce_sum(0))
        .into();
        let loss = *graph.toposort().last().unwrap();
        graph.with_gradients(loss)
    });
    let mut cpu = SimpleExecutor::new();
    for step in 0..6 {
        if step == 3 {
            for graph in &mut graphs {
                for node in graph.graph.node_weights_mut() {
                    if let TensorGraphNode::IndexSource { source } = node
                        && matches!(source.source(), IndexSource::Constant(_))
                    {
                        *source = IndexTensor::constant(vec![365], vec![1]).unwrap();
                    }
                }
            }
        }
        if step == 4 {
            for executor in &mut executors {
                executor.clear_pool();
            }
        }
        owner.begin_step().unwrap();
        let mut expected = vec![0.0f32; 683];
        for ((graph, executor), n) in graphs.iter().zip(&mut executors).zip([3, 1]) {
            if step == 2 {
                let before = executor.replay_metrics().graph_launches;
                let error = owner
                    .accumulate_with_indices(
                        executor,
                        graph,
                        HashMap::new(),
                        HashMap::from([("ids".into(), vec![u32::MAX; n])]),
                    )
                    .unwrap_err();
                assert!(error.to_string().contains("4294967295"));
                assert_eq!(executor.replay_metrics().graph_launches, before);
            }
            let ids = if step % 2 == 0 {
                [257, 365, 682]
            } else {
                [259, 367, 681]
            };
            let feeds = HashMap::from([("ids".into(), ids[..n].to_vec())]);
            cpu.execute_with_indices(graph, HashMap::new(), feeds.clone())
                .unwrap();
            let gradients = cpu.get_gradients(graph);
            for (sum, value) in expected.iter_mut().zip(&gradients[&weight.id()]) {
                *sum += value;
            }
            owner
                .accumulate_with_indices(executor, graph, HashMap::new(), feeds)
                .unwrap();
            assert!(
                executor.replay_metrics().active,
                "{:?}",
                executor.replay_metrics()
            );
            assert_eq!(executor.replay_metrics().fallbacks, 0);
            assert_eq!(
                executor.replay_metrics().captures,
                if step >= 4 { 2 } else { 1 }
            );
            assert_eq!(executor.compilation_count(), 1);
        }
        let status = owner.step(0.5).unwrap();
        assert_eq!(status.timestep, step + 1);
        let expected_norm = expected
            .iter()
            .map(|&x| f64::from(x).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!((status.gradient_norm - expected_norm).abs() < 1e-6);
        let actual = owner.gradients().unwrap();
        for (a, b) in actual[&weight.id()].iter().zip(&expected) {
            assert!((a - b * status.gradient_scale).abs() < 1e-6);
        }
    }
}
