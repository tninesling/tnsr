use super::*;
use crate::ptx::resident::{AdamGroup, ResidentAdam};
#[track_caller]
fn bits_equal(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (i, (&a, &b)) in a.iter().zip(b).enumerate() {
        assert!(
            (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits(),
            "{i}: {:08x} != {:08x}",
            a.to_bits(),
            b.to_bits()
        );
    }
}
fn options(enabled: bool) -> MixedOptions {
    MixedOptions {
        layer_norm_pair_fusion: enabled,
        physical_order_casts: true,
        shared_memory_reuse: true,
        narrow_bf16_tile: true,
        medium_bf16_tile_policy: crate::tile::MediumBf16TilePolicy::SquareThenTall,
        column_reduction_policy: crate::tile::ColumnReductionPolicy::Four,
        batched_attention_bf16_tile_policy: crate::tile::BatchedAttentionBf16TilePolicy::TallScores,
        attention_softmax_fusion: true,
        attention_softmax_backward_fusion: true,
        swiglu_pair_fusion: true,
        qkv_grad_pack_fusion: true,
        transposed_bf16_outputs: true,
        incremental_allocation_reuse: true,
        direct_master_bindings: true,
        ..MixedOptions::default()
    }
}
#[test]
fn actual_nine_pairs_replay_direct_masters_and_one_two_one_microbatches() {
    actual_model(false);
}
#[test]
fn escaped_saved_stat_keeps_other_eight_pairs_and_exact_training() {
    actual_model(true);
}
#[test]
fn actual_w1024_b1_pairs_replay_direct_and_evolving_microbatches() {
    actual_model_shape(1, 1024, 16, false);
}
#[test]
fn actual_w512_b2_pairs_replay_direct_and_evolving_microbatches() {
    actual_model_shape(2, 512, 8, false);
}
fn actual_model(escape_one: bool) {
    actual_model_shape(8, 512, 8, escape_one);
}
#[test]
fn actual_w512_s128_b2_pairs_preserve_evolving_training_state() {
    actual_model_geometry(2, 128, 512, 8, false);
}
#[test]
fn actual_w1024_s128_b2_pairs_preserve_evolving_training_state() {
    actual_model_geometry(2, 128, 1024, 16, false);
}
fn actual_model_shape(batch: usize, width: usize, heads: usize, escape_one: bool) {
    actual_model_geometry(batch, 512, width, heads, escape_one);
}
fn actual_model_geometry(
    batch: usize,
    sequence: usize,
    width: usize,
    heads: usize,
    escape_one: bool,
) {
    if !super::super::tests::cuda_available() {
        return;
    }
    // Unique token rows avoid nondeterministic atomic embedding collisions;
    // all LayerNorm topology, four layers, and changed feed permutations remain.
    let vocabulary = batch * sequence;
    let models = [
        super::tests::decoder_with_shape(batch, width, heads, sequence, vocabulary),
        super::tests::decoder_with_shape(batch, width, heads, sequence, vocabulary),
    ];
    for model in &models {
        for (name, parameter) in model.parameters.iter() {
            if name.ends_with("norm.weight") || name.ends_with("norm.bias") {
                let values = (0..width)
                    .map(|i| {
                        if name.ends_with("weight") {
                            1.0 + (i % 7) as f32 / 32.0
                        } else {
                            (i % 11) as f32 / 128.0 - 0.03125
                        }
                    })
                    .collect();
                parameter.parameter.set_values(values).unwrap();
            }
        }
    }
    let mut graphs = models
        .iter()
        .map(|m| {
            m.training_graph_with_options(crate::graph::GradientOptions {
                compact_softmax: true,
            })
        })
        .collect::<Result<Vec<_>>>()
        .unwrap();
    if escape_one {
        for (graph, _) in &mut graphs {
            let pair = recognize(graph).unwrap().remove(0);
            let shape = graph[pair.mean].shape().clone();
            let output = graph.graph.add_node(TensorGraphNode::Reshape { shape });
            graph.graph.add_edge(pair.mean, output, 0);
            assert_eq!(recognize(graph).unwrap().len(), 8);
        }
    }
    let mut owners = models
        .iter()
        .map(|m| {
            ResidentAdam::new_with_graph_replay(
                m.parameters
                    .iter()
                    .map(|(_, p)| (p.parameter.clone(), 0))
                    .collect(),
                vec![AdamGroup::new(1e-4)],
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    for owner in &mut owners {
        owner.set_update_batch_size(8).unwrap();
        owner
            .set_experimental_first_accumulation_overwrite(true)
            .unwrap();
        owner.set_experimental_gradient_norm_partials(true).unwrap();
        owner.begin_step().unwrap();
    }
    let mut executors = owners
        .iter_mut()
        .enumerate()
        .map(|(i, o)| {
            let mut policy = options(i == 1);
            if escape_one || batch < 8 {
                // Joint qualification of the independently admitted high-BH
                // ByGrid schedule and the eight remaining exact row pairs.
                policy.batched_attention_bf16_tile_policy =
                    crate::tile::BatchedAttentionBf16TilePolicy::TallScoresByGrid;
            }
            o.mixed_executor_with_options(&graphs[i].0, MatMulPrecision::Tf32x3, policy)
                .unwrap()
        })
        .collect::<Vec<_>>();
    eprintln!(
        "LN model kernels {} -> {}, regions {}/{}, fallback {}",
        executors[0].info().kernels,
        executors[1].info().kernels,
        executors[1].info().layer_norm_pair_forward_regions,
        executors[1].info().layer_norm_pair_backward_regions,
        executors[1].info().layer_norm_pair_fallback_regions
    );
    if escape_one || batch < 8 {
        for executor in &executors {
            assert_eq!(
                executor.info().batched_attention_score_programs,
                if sequence == 512 { 8 } else { 0 }
            );
            assert_eq!(
                executor.info().batched_attention_query_width_programs,
                if sequence == 128 || batch < 8 { 0 } else { 12 }
            );
            assert_eq!(
                executor.info().batched_attention_width_query_programs,
                if sequence == 128 || batch < 8 { 0 } else { 4 }
            );
        }
    }
    assert_eq!(executors[0].info().layer_norm_pair_forward_regions, 0);
    assert_eq!(
        executors[1].info().layer_norm_pair_forward_regions,
        if escape_one { 8 } else { 9 }
    );
    assert_eq!(
        executors[1].info().layer_norm_pair_backward_regions,
        if escape_one { 8 } else { 9 }
    );
    assert_eq!(executors[1].info().layer_norm_pair_fallback_regions, 0);
    assert!(executors[1].info().kernels < executors[0].info().kernels);
    for (interval, microbatches) in [1, 2, 1].into_iter().enumerate() {
        let mut steps = Vec::new();
        for i in 0..2 {
            if interval > 0 {
                owners[i].begin_step().unwrap();
            }
            for micro in 0..microbatches {
                let mut feeds = models[i].config.feeds().unwrap();
                for values in feeds.values_mut() {
                    for v in values {
                        *v = (*v + u32::try_from(interval + micro).unwrap())
                            % u32::try_from(models[i].config.vocab).unwrap();
                    }
                }
                owners[i]
                    .accumulate_mixed(&mut executors[i], &graphs[i].0, HashMap::new(), feeds)
                    .unwrap();
            }
            steps.push(owners[i].step(1.).unwrap());
            assert_eq!(executors[i].transfers().parameter_copies, 0);
            assert_eq!(executors[i].transfers().parameter_bytes, 0);
        }
        assert_eq!(
            steps[0].gradient_norm.to_bits(),
            steps[1].gradient_norm.to_bits()
        );
        assert_eq!(
            steps[0].gradient_scale.to_bits(),
            steps[1].gradient_scale.to_bits()
        );
        assert_eq!(steps[0].timestep, steps[1].timestep);
        bits_equal(
            &executors[0].read(graphs[0].1).unwrap(),
            &executors[1].read(graphs[1].1).unwrap(),
        );
        let grads = [
            owners[0].gradients().unwrap(),
            owners[1].gradients().unwrap(),
        ];
        let moments = [owners[0].moments().unwrap(), owners[1].moments().unwrap()];
        for ((name, a), (_, b)) in models[0].parameters.iter().zip(models[1].parameters.iter()) {
            let (a, b) = (&a.parameter, &b.parameter);
            eprintln!("LN interval {interval} parameter {name}");
            bits_equal(&grads[0][&a.id()], &grads[1][&b.id()]);
            bits_equal(&a.values().unwrap(), &b.values().unwrap());
            bits_equal(&moments[0][&a.id()].first, &moments[1][&b.id()].first);
            bits_equal(&moments[0][&a.id()].second, &moments[1][&b.id()].second);
        }
    }
    for executor in &executors {
        assert!(executor.info().replay);
        assert_eq!(executor.info().direct_capture_variants, 2);
        assert_eq!(executor.info().direct_capture_misses, 2);
        assert_eq!(executor.info().direct_capture_hits, 1);
    }
}
fn row_kernel(
    stream: &Arc<CudaStream>,
    ir: crate::tile::TileIR,
    pointers: Vec<u64>,
    rows: usize,
) -> Kernel {
    let mut tiles = TileGraph::from_graph(&TensorGraph::<f32>::new());
    tiles.region_kernels = vec![crate::tile::graph::TileRegionKernel { region_id: 0, ir }];
    tiles.optimize_indices().unwrap();
    let graph =
        PtxGraph::from(tiles).with_target(PtxTarget::from_context(stream.context()).unwrap());
    let name = graph.region_kernel_name(0).unwrap();
    let module = stream
        .context()
        .load_module(Ptx::from_src(graph.to_ptx()))
        .unwrap();
    let function = module.load_function(name).unwrap();
    assert!(function.max_threads_per_block().unwrap() >= 128);
    assert!(
        function
            .occupancy_max_active_blocks_per_multiprocessor(128, 0, None)
            .unwrap()
            > 0
    );
    Kernel {
        owner: None,
        label: "LN direct typed-storage qualification".into(),
        function,
        config: LaunchConfig {
            grid_dim: (u32::try_from(rows).unwrap(), 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        },
        pointers,
        zero_before: None,
    }
}
#[test]
fn typed_bf16_incoming_widens_exactly_for_all_bit_patterns() {
    if !super::super::tests::cuda_available() {
        return;
    }
    for width in [512, 1024] {
        typed_incoming_width(width);
    }
}
fn typed_incoming_width(width: usize) {
    let model = super::tests::decoder_with_dimensions(1, width, 8, 31);
    let (g, _) = model.training_graph().unwrap();
    let pair = recognize(&g).unwrap().remove(0);
    let stream = CudaContext::new(0).unwrap().new_stream().unwrap();
    let rows = 512;
    let total = rows * width;
    let z = stream
        .memcpy_stod(
            &(0..total)
                .map(|i| ((i % 513) as f32 - 256.) / 128.)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let gamma = stream
        .memcpy_stod(
            &(0..width)
                .map(|i| 1. + (i % 7) as f32 / 64.)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let beta = stream.memcpy_stod(&vec![0.01_f32; width]).unwrap();
    let y = stream.alloc_zeros::<f32>(total).unwrap();
    let n = stream.alloc_zeros::<f32>(total).unwrap();
    let m = stream.alloc_zeros::<f32>(rows).unwrap();
    let a = stream.alloc_zeros::<f32>(rows).unwrap();
    let r = stream.alloc_zeros::<f32>(rows).unwrap();
    let fp = vec![
        z.device_ptr(&stream).0,
        gamma.device_ptr(&stream).0,
        beta.device_ptr(&stream).0,
        y.device_ptr(&stream).0,
        n.device_ptr(&stream).0,
        m.device_ptr(&stream).0,
        a.device_ptr(&stream).0,
        r.device_ptr(&stream).0,
    ];
    row_kernel(&stream, kernel::forward(&pair).unwrap(), fp, rows)
        .run(&stream)
        .unwrap();
    let bf = stream
        .memcpy_stod(
            &(0..total)
                .map(|i| half::bf16::from_bits((i % 65536) as u16))
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let f = stream
        .memcpy_stod(
            &(0..total)
                .map(|i| f32::from_bits(((i % 65536) as u32) << 16))
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let left = stream.alloc_zeros::<f32>(total).unwrap();
    let right = stream.alloc_zeros::<f32>(total).unwrap();
    for tail in [
        kernel::TailBoundary::SeparateMeanTerm,
        kernel::TailBoundary::OriginalPointwiseOpportunity,
    ] {
        for (dtype, ptr, out) in [
            (
                DType::F32,
                f.device_ptr(&stream).0,
                left.device_ptr(&stream).0,
            ),
            (
                DType::BF16,
                bf.device_ptr(&stream).0,
                right.device_ptr(&stream).0,
            ),
        ] {
            let p = vec![
                z.device_ptr(&stream).0,
                gamma.device_ptr(&stream).0,
                ptr,
                m.device_ptr(&stream).0,
                a.device_ptr(&stream).0,
                r.device_ptr(&stream).0,
                out,
            ];
            row_kernel(
                &stream,
                kernel::backward(&pair, tail, dtype).unwrap(),
                p,
                rows,
            )
            .run(&stream)
            .unwrap();
        }
        stream.synchronize().unwrap();
        bits_equal(
            &stream.memcpy_dtov(&left).unwrap(),
            &stream.memcpy_dtov(&right).unwrap(),
        );
    }
}

#[test]
fn unsupported_original_partition_falls_back_and_preserves_recovery() {
    if !super::super::tests::cuda_available() {
        return;
    }
    let shape = vec![8, 512, 512];
    let x = Parameter::new(
        (0..shape.iter().product())
            .map(|i| (i % 257) as f32 / 128. - 1.)
            .collect(),
        shape.clone(),
    );
    let gamma = Parameter::new(
        (0..512).map(|i| 1. + (i % 7) as f32 / 32.).collect(),
        vec![512],
    );
    let beta = Parameter::new((0..512).map(|i| (i % 11) as f32 / 64.).collect(), vec![512]);
    let expression = crate::nn::layer_norm(
        TensorExpr::from(x.clone()).cast(ValueDType::F32),
        2,
        gamma.clone(),
        beta.clone(),
        1e-5,
    );
    let mut graph = TensorGraph::<f32>::new();
    let y = graph.add_expr(&expression, false);
    let loss = graph.add_expr(
        &(TensorExpr::node_ref(y, shape.clone()) * TensorExpr::input("incoming", shape.clone()))
            .reduce_axis_sum(2)
            .reduce_axis_sum(1)
            .reduce_axis_sum(0),
        false,
    );
    let graph = graph.with_gradients(loss);
    assert_eq!(recognize(&graph).unwrap().len(), 1);
    let parameters = [x, gamma, beta];
    let mut owner = ResidentAdam::new_with_graph_replay(
        parameters.iter().map(|p| (p.clone(), 0)).collect(),
        vec![AdamGroup::new(1e-4)],
    )
    .unwrap();
    owner
        .set_experimental_first_accumulation_overwrite(true)
        .unwrap();
    owner.begin_step().unwrap();
    let mut executor = owner
        .mixed_executor_with_options(&graph, MatMulPrecision::Tf32x3, options(true))
        .unwrap();
    assert_eq!(executor.info().layer_norm_pair_forward_regions, 0);
    assert_eq!(executor.info().layer_norm_pair_backward_regions, 0);
    assert_eq!(executor.info().layer_norm_pair_fallback_regions, 1);
    let total = shape.iter().product();
    let finite = (0..total)
        .map(|i| (i % 31) as f32 / 64. - 0.25)
        .collect::<Vec<_>>();
    let accumulate = |owner: &mut ResidentAdam, executor: &mut MixedExecutor, input: Vec<f32>| {
        owner
            .accumulate_mixed(
                executor,
                &graph,
                HashMap::from([("incoming".into(), input)]),
                HashMap::new(),
            )
            .unwrap();
    };
    accumulate(&mut owner, &mut executor, finite.clone());
    assert_eq!(owner.step(1.).unwrap().timestep, 1);
    let weights = parameters
        .iter()
        .map(|p| p.values().unwrap())
        .collect::<Vec<_>>();
    let moments = owner.moments().unwrap();
    owner.begin_step().unwrap();
    let mut bad = finite.clone();
    bad[total - 1] = f32::INFINITY;
    accumulate(&mut owner, &mut executor, bad);
    assert!(owner.step(1.).is_err());
    for (parameter, expected) in parameters.iter().zip(&weights) {
        bits_equal(&parameter.values().unwrap(), expected);
        let after = owner.moments().unwrap();
        bits_equal(
            &after[&parameter.id()].first,
            &moments[&parameter.id()].first,
        );
        bits_equal(
            &after[&parameter.id()].second,
            &moments[&parameter.id()].second,
        );
    }
    owner.begin_step().unwrap();
    accumulate(&mut owner, &mut executor, finite);
    assert_eq!(owner.step(1.).unwrap().timestep, 2);
    assert_eq!(executor.info().layer_norm_pair_forward_regions, 0);
    assert_eq!(executor.transfers().parameter_copies, 0);
}

#[test]
fn actual_pairs_nonfinite_failure_preserves_weights_moments_and_replays_recovery() {
    if !super::super::tests::cuda_available() {
        return;
    }
    let model = super::tests::decoder(8);
    let expression = model.loss().unwrap() * TensorExpr::input("loss_scale", vec![1]);
    let mut graph = TensorGraph::<f32>::new();
    let loss = graph.add_expr(&expression, false);
    let graph = graph.with_gradients_with_options(
        loss,
        crate::graph::GradientOptions {
            compact_softmax: true,
        },
    );
    assert_eq!(recognize(&graph).unwrap().len(), 9);
    let mut owner = ResidentAdam::new_with_graph_replay(
        model
            .parameters
            .iter()
            .map(|(_, p)| (p.parameter.clone(), 0))
            .collect(),
        vec![AdamGroup::new(1e-4)],
    )
    .unwrap();
    owner.set_update_batch_size(8).unwrap();
    owner
        .set_experimental_first_accumulation_overwrite(true)
        .unwrap();
    owner.set_experimental_gradient_norm_partials(true).unwrap();
    owner.begin_step().unwrap();
    let mut executor = owner
        .mixed_executor_with_options(&graph, MatMulPrecision::Tf32x3, options(true))
        .unwrap();
    assert_eq!(executor.info().layer_norm_pair_forward_regions, 9);
    assert_eq!(executor.info().layer_norm_pair_backward_regions, 9);
    let accumulate = |owner: &mut ResidentAdam, executor: &mut MixedExecutor, value| {
        owner
            .accumulate_mixed(
                executor,
                &graph,
                HashMap::from([("loss_scale".into(), vec![value])]),
                model.config.feeds().unwrap(),
            )
            .unwrap();
    };
    accumulate(&mut owner, &mut executor, 1.0);
    assert_eq!(owner.step(1.).unwrap().timestep, 1);
    let weights = model
        .parameters
        .iter()
        .map(|(_, p)| p.parameter.values().unwrap())
        .collect::<Vec<_>>();
    let moments = owner.moments().unwrap();
    owner.begin_step().unwrap();
    accumulate(&mut owner, &mut executor, f32::NAN);
    assert!(owner.step(1.).is_err());
    let after = owner.moments().unwrap();
    for ((_, parameter), expected) in model.parameters.iter().zip(&weights) {
        let parameter = &parameter.parameter;
        bits_equal(&parameter.values().unwrap(), expected);
        bits_equal(
            &after[&parameter.id()].first,
            &moments[&parameter.id()].first,
        );
        bits_equal(
            &after[&parameter.id()].second,
            &moments[&parameter.id()].second,
        );
    }
    owner.begin_step().unwrap();
    accumulate(&mut owner, &mut executor, 1.0);
    assert_eq!(owner.step(1.).unwrap().timestep, 2);
    assert_eq!(executor.info().layer_norm_pair_forward_regions, 9);
    assert_eq!(executor.transfers().parameter_copies, 0);
}
