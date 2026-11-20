//! Actual decoder fixtures for the opt-in batched attention policy hook.
//! This runs on actual decoder MatMul nodes and their view chains, not synthetic
//! identity operands. The graph-only fixture does not construct a CUDA context;
//! separate real-device tests qualify the complete training path.
use super::test_decoder::{Config, synthetic};
use super::*;
use crate::tile::{
    BatchedAttentionBf16TilePolicy as Policy, MatMulCapabilities, MatMulRegion, MatMulResources,
    MatMulSchedule, RegionInput, RegionValue,
};

fn physical_operand<G>(
    graph: &TensorGraph<f32, G>,
    dtypes: &HashMap<NodeIndex, ValueDType>,
    node: NodeIndex,
) -> Option<(VirtualTensor, Vec<usize>)> {
    let input = || graph.inputs(node).first().copied();
    let transform = |source: NodeIndex| physical_operand(graph, dtypes, source);
    match &graph[node] {
        TensorGraphNode::Reshape { shape } | TensorGraphNode::Flatten { shape } => {
            let (v, physical) = transform(input()?)?;
            Some((v.reshape(shape.clone()).ok()?, physical))
        }
        TensorGraphNode::Permute { axes, .. } => {
            let (v, physical) = transform(input()?)?;
            Some((v.permute(axes).ok()?, physical))
        }
        TensorGraphNode::Transpose { shape } => {
            let (v, physical) = transform(input()?)?;
            let mut axes: Vec<_> = (0..shape.len()).collect();
            let len = axes.len();
            if len < 2 {
                return None;
            }
            axes.swap(len - 2, len - 1);
            Some((v.permute(&axes).ok()?, physical))
        }
        TensorGraphNode::Cast { dtype, .. } => {
            if let Some((v, physical)) = transform(input()?) {
                // BF16->F32 promotion and identity re-narrow remain aliases.
                if v.dtype == DType::BF16 {
                    return Some((v, physical));
                }
            }
            if *dtype != ValueDType::BF16 {
                return None;
            }
            let shape = graph[node].shape().clone();
            Some((
                VirtualTensor::identity(node, shape.clone(), DType::BF16),
                shape,
            ))
        }
        TensorGraphNode::BroadcastAxis { .. } => None,
        _ if dtypes.get(&node) == Some(&ValueDType::BF16) => {
            let shape = graph[node].shape().clone();
            Some((
                VirtualTensor::identity(node, shape.clone(), DType::BF16),
                shape,
            ))
        }
        _ => None,
    }
}

#[test]
fn actual_decoder_batched_attention_high_bh_all_roles_and_low_bh_grid_scores() {
    use crate::tile::{MatMulPipeline, MatMulSharedLayout};
    let resources = MatMulResources {
        shared_bytes_per_block: 48 * 1024,
        shared_bytes_per_sm: 100 * 1024,
        registers_per_sm: 64 * 1024,
        threads_per_sm: 1536,
    };
    for (batch, width, heads, hidden) in [
        (8, 512, 8, 1408),
        (32, 512, 8, 1408),
        (1, 1024, 16, 2816),
        (2, 512, 8, 1408),
    ] {
        let model = synthetic(Config {
            width,
            layers: 4,
            heads,
            hidden,
            sequence: 512,
            batch,
            vocab: 4096,
            epsilon: 1e-5,
            learning_rate: 1e-4,
            max_grad_norm: 1.,
        });
        let (graph, _) = model
            .training_graph_with_options(crate::graph::GradientOptions {
                compact_softmax: true,
            })
            .unwrap();
        let dtypes = value_dtypes(&graph);
        let mut regions = Vec::new();
        for node in graph.toposort() {
            if !matches!(graph[node], TensorGraphNode::MatMul { .. })
                || dtypes.get(&node) != Some(&ValueDType::BF16)
            {
                continue;
            }
            let shape = graph[node].shape();
            if shape.len() != 4 || shape[..2] != [batch, heads] {
                continue;
            }
            let inputs = graph.inputs(node);
            let lhs = graph[inputs[0]].shape();
            let rhs = graph[inputs[1]].shape();
            let (m, n, k) = (shape[2], shape[3], lhs[lhs.len() - 1]);
            if !matches!((m, n, k), (512, 512, 64) | (512, 64, 512) | (64, 512, 512)) {
                continue;
            }
            let mut plain = TensorGraph::<half::bf16>::new();
            let a = plain.graph.add_node(TensorGraphNode::Input {
                name: "a",
                shape: lhs.clone(),
            });
            let b = plain.graph.add_node(TensorGraphNode::Input {
                name: "b",
                shape: rhs.clone(),
            });
            let output = plain.graph.add_node(TensorGraphNode::MatMul {
                shape: shape.clone(),
            });
            plain.graph.add_edge(a, output, 0);
            plain.graph.add_edge(b, output, 1);
            let mut r = MatMulRegion::from_graph(&plain, output, vec![], vec![output]).unwrap();
            r.schedule = MatMulSchedule::select_with_resources(
                m,
                n,
                k,
                DType::BF16,
                MatMulCapabilities {
                    tf32: true,
                    f16: true,
                    bf16: true,
                },
                MatMulPrecision::NativeHalf,
                resources,
            );
            for (i, input) in inputs.into_iter().enumerate() {
                let (tensor, source_shape) = physical_operand(&graph, &dtypes, input).unwrap();
                r.inputs[i] = RegionInput {
                    node: input,
                    value: RegionValue(i),
                    tensor,
                    source_shape,
                };
            }
            regions.push((node, r));
        }
        assert_eq!(regions.len(), 24);
        for policy in [
            Policy::TallScores,
            Policy::WideScores,
            Policy::TallScoresByGrid,
        ] {
            let mut roles = [0; 3];
            let mut geometries = [0; 2];
            for (node, region) in &regions {
                let should_admit = if batch * heads >= 64 {
                    true
                } else {
                    policy == Policy::TallScoresByGrid && (region.m, region.n) == (512, 512)
                };
                if !should_admit {
                    assert!(policy.candidate(region, resources, true).is_none());
                    continue;
                }
                let candidate = policy
                    .candidate(region, resources, true)
                    .unwrap_or_else(|| {
                        panic!(
                            "actual node{} {:?}x{:?} failed whole-map selection",
                            node.index(),
                            region.lhs_shape,
                            region.rhs_shape
                        )
                    });
                let s = candidate
                    .configure(
                        region,
                        MatMulSharedLayout::Auto,
                        MatMulPipeline::Auto,
                        true,
                        resources,
                    )
                    .unwrap_or_else(|| {
                        panic!(
                            "actual node{} {:?}x{:?} failed final async/resource proof",
                            node.index(),
                            region.lhs_shape,
                            region.rhs_shape
                        )
                    });
                roles[match (region.m, region.n, region.k) {
                    (512, 512, 64) => 0,
                    (512, 64, 512) => 1,
                    (64, 512, 512) => 2,
                    _ => unreachable!(),
                }] += 1;
                geometries[if s.block_tile.m == 128 { 0 } else { 1 }] += 1;
            }
            assert_eq!(
                roles,
                if batch * heads >= 64 {
                    [8, 12, 4]
                } else if policy == Policy::TallScoresByGrid {
                    [8, 0, 0]
                } else {
                    [0, 0, 0]
                }
            );
            assert_eq!(
                geometries,
                if batch * heads < 64 {
                    if policy == Policy::TallScoresByGrid {
                        [8, 0]
                    } else {
                        [0, 0]
                    }
                } else if policy == Policy::WideScores {
                    [12, 12]
                } else {
                    [20, 4]
                }
            );
        }
    }
}

fn same_bits(a: &[f32], b: &[f32], context: &str) {
    assert_eq!(a.len(), b.len());
    for (i, (&a, &b)) in a.iter().zip(b).enumerate() {
        assert!(
            (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits(),
            "{context} index{i}: {a:?}/{:08x} vs {b:?}/{:08x}",
            a.to_bits(),
            b.to_bits()
        );
    }
}

fn check_actual_decoder_training(policy: Policy, config: Config) -> Result<()> {
    check_actual_decoder_training_with_options(policy, config, MixedOptions::default())
}

pub(super) fn check_actual_decoder_training_with_options(
    policy: Policy,
    config: Config,
    extra: MixedOptions,
) -> Result<()> {
    use crate::ptx::resident::{AdamGroup, ResidentAdam};
    let ordinary_expected_split = match config.batch {
        8 => 0,
        16 => 16,
        _ => 0,
    };
    let low_bh = config.batch * config.heads < 64;
    let fixture_tag = format!("b{}-w{}-h{}", config.batch, config.width, config.heads);
    let models = [synthetic(config.clone()), synthetic(config)];
    let graphs = models
        .iter()
        .map(|m| {
            m.training_graph_with_options(crate::graph::GradientOptions {
                compact_softmax: true,
            })
            .unwrap()
        })
        .collect::<Vec<_>>();
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
        owner.set_experimental_first_accumulation_overwrite(true)?;
        owner.set_experimental_gradient_norm_partials(true)?;
        owner.begin_step()?;
    }
    let mut executors = owners
        .iter_mut()
        .enumerate()
        .map(|(i, owner)| {
            owner
                .mixed_executor_with_options(
                    &graphs[i].0,
                    MatMulPrecision::Tf32x3,
                    MixedOptions {
                        shared_memory_reuse: true,
                        plain_bf16_output: i == 1 && extra.plain_bf16_output,
                        indexed_loss_cooperative_ordered_backward: extra
                            .indexed_loss_cooperative_ordered_backward,
                        indexed_loss_cooperative_ordered_forward: extra
                            .indexed_loss_cooperative_ordered_forward,
                        attention_training_forward_fusion: extra.attention_training_forward_fusion,
                        ordinary_bf16_four_warp_block: i == 1
                            && extra.ordinary_bf16_four_warp_block,
                        layer_norm_pair_fusion: extra.layer_norm_pair_fusion,
                        split_k_bf16_tile_policy: extra.split_k_bf16_tile_policy,
                        physical_order_casts: true,
                        incremental_allocation_reuse: true,
                        direct_master_bindings: true,
                        attention_softmax_fusion: true,
                        attention_softmax_backward_fusion: true,
                        qkv_grad_pack_fusion: true,
                        swiglu_pair_fusion: true,
                        narrow_bf16_tile: true,
                        transposed_bf16_outputs: true,
                        medium_bf16_tile_policy: crate::tile::MediumBf16TilePolicy::SquareThenTall,
                        large_bf16_tile_policy: crate::tile::LargeBf16TilePolicy::Off,
                        column_reduction_policy: crate::tile::ColumnReductionPolicy::Four,
                        batched_attention_bf16_tile_policy: if i == 0
                            && !extra.ordinary_bf16_four_warp_block
                            && !extra.plain_bf16_output
                        {
                            Policy::Off
                        } else {
                            policy
                        },
                        ..Default::default()
                    },
                )
                .unwrap()
        })
        .collect::<Vec<_>>();
    for (i, e) in executors.iter().enumerate() {
        let info = e.info();
        assert!(info.replay);
        let removed = info.attention_training_forward_regions;
        assert!(removed <= 4);
        assert_eq!(
            [
                info.batched_attention_score_programs,
                info.batched_attention_query_width_programs,
                info.batched_attention_width_query_programs
            ],
            if i == 0 && !extra.ordinary_bf16_four_warp_block && !extra.plain_bf16_output {
                [0, 0, 0]
            } else if low_bh {
                [8 - removed, 0, 0]
            } else {
                [8 - removed, 12 - removed, 4]
            }
        );
        assert_eq!(
            [
                info.batched_attention_tall_programs,
                info.batched_attention_wide_programs
            ],
            if i == 0 && !extra.ordinary_bf16_four_warp_block && !extra.plain_bf16_output {
                [0, 0]
            } else if low_bh {
                [8 - removed, 0]
            } else if policy == Policy::WideScores {
                [12 - removed, 12 - removed]
            } else {
                [20 - 2 * removed, 4]
            }
        );
    }
    assert_eq!(executors[0].info().kernels, executors[1].info().kernels);
    if extra.plain_bf16_output {
        for e in &executors {
            assert_eq!(e.info().layer_norm_pair_forward_regions, 9);
            assert_eq!(e.info().layer_norm_pair_backward_regions, 9);
            assert_eq!(
                e.info().attention_training_forward_regions,
                if extra.attention_training_forward_fusion {
                    4
                } else {
                    0
                }
            );
        }
        assert_eq!(
            executors[0].info().split_k_square64_programs,
            executors[1].info().split_k_square64_programs
        );
        assert_eq!(executors[0].info().plain_bf16_output_programs, 0);
        assert!(executors[1].info().plain_bf16_output_programs > 0);
        assert_eq!(executors[1].info().plain_bf16_output_fallback_programs, 0);
        let mut selected = 0;
        for p in &executors[1].programs {
            if let Some(p) = p.downcast_ref::<Program<BF16>>() {
                selected += p.plain_bf16_output_counts.0;
                assert_eq!(
                    p.kernels
                        .iter()
                        .filter(|k| k.label.contains("plain_bf16_output=true"))
                        .count(),
                    p.plain_bf16_output_counts.0
                );
            }
        }
        if let Ok(directory) = std::env::var("TNSR_PLAIN_BF16_OUTPUT_AUDIT") {
            let directory = std::path::Path::new(&directory);
            std::fs::create_dir_all(directory)?;
            let mut resources = Vec::new();
            for (id, p) in executors[1].programs.iter().enumerate() {
                let Some(p) = p.downcast_ref::<Program<BF16>>() else {
                    continue;
                };
                if p.plain_bf16_output_counts.0 == 0 {
                    continue;
                }
                std::fs::write(
                    directory.join(format!("candidate-{fixture_tag}-{id}.ptx")),
                    &p.ptx,
                )?;
                std::fs::write(
                    directory.join(format!("candidate-{fixture_tag}-{id}.plan")),
                    &p.plan,
                )?;
                let old = executors[0].programs[id]
                    .downcast_ref::<Program<BF16>>()
                    .unwrap();
                std::fs::write(
                    directory.join(format!("baseline-{fixture_tag}-{id}.ptx")),
                    &old.ptx,
                )?;
                for k in &p.kernels {
                    if !k.label.contains("plain_bf16_output=true") {
                        continue;
                    }
                    assert!(k.function.num_regs()? <= 128);
                    assert_eq!(k.function.local_size_bytes()?, 0);
                    let capacity = k
                        .function
                        .occupancy_max_active_blocks_per_multiprocessor(256, 0, None)?;
                    assert!(capacity >= 2);
                    assert_eq!(k.config.block_dim, (32, 8, 1));
                    resources.push(serde_json::json!({"label":k.label,"registers":k.function.num_regs()?,"shared":k.function.shared_size_bytes()?,"local":k.function.local_size_bytes()?,"api_max_active_blocks":capacity,"capacity_is_not_measured_occupancy":true,"block":k.config.block_dim,"grid":k.config.grid_dim}));
                }
            }
            assert_eq!(resources.len(), selected);
            std::fs::write(
                directory.join(format!("actual-{fixture_tag}-resources.json")),
                serde_json::to_vec_pretty(&resources)?,
            )?;
        }
        assert_eq!(selected, executors[1].info().plain_bf16_output_programs);
        assert!(
            executors[1]
                .info()
                .indexed_loss_cooperative_ordered_backward_programs
                > 0
        );
        assert!(
            executors[1]
                .info()
                .indexed_loss_cooperative_ordered_forward_programs
                > 0
        );
        assert_eq!(
            executors[0].info().attention_training_forward_regions,
            executors[1].info().attention_training_forward_regions
        );
    }
    if extra.ordinary_bf16_four_warp_block {
        assert_eq!(executors[0].info().ordinary_bf16_four_warp_programs, 0);
        assert!(executors[1].info().ordinary_bf16_four_warp_programs > 0);
        assert_eq!(
            executors[0].info().split_k_square64_programs,
            executors[1].info().split_k_square64_programs
        );
        // B8 retains the V18 SplitSquare option but its dW K=4096 fails
        // K/max(M,N)>=4. B16 admits the 16 layer dW contractions; its tied
        // head still fails that guard. Both arms must retain this admission.
        assert_eq!(
            executors[1].info().split_k_square64_programs,
            ordinary_expected_split
        );
        assert_eq!(executors[0].info().layer_norm_pair_forward_regions, 9);
        assert_eq!(executors[1].info().layer_norm_pair_forward_regions, 9);
    }
    if extra.ordinary_bf16_four_warp_block
        && let Ok(path) = std::env::var("TNSR_ORDINARY_FOUR_WARP_AUDIT")
    {
        let directory = std::path::Path::new(&path);
        std::fs::create_dir_all(directory)?;
        let mut resources = Vec::new();
        for (id, program) in executors[1].programs.iter().enumerate() {
            let Some(program) = program.downcast_ref::<Program<BF16>>() else {
                continue;
            };
            if program.ordinary_bf16_four_warp_count == 0 {
                continue;
            }
            std::fs::write(
                directory.join(format!("actual-ordinary-{fixture_tag}-{id}.ptx")),
                &program.ptx,
            )?;
            std::fs::write(
                directory.join(format!("actual-ordinary-{fixture_tag}-{id}.plan")),
                &program.plan,
            )?;
            let baseline = executors[0].programs[id]
                .downcast_ref::<Program<BF16>>()
                .unwrap();
            std::fs::write(
                directory.join(format!("baseline-ordinary-{fixture_tag}-{id}.ptx")),
                &baseline.ptx,
            )?;
            for kernel in &program.kernels {
                if !kernel.label.contains("ordinary_bf16_four_warp_block=true") {
                    continue;
                }
                assert_eq!(kernel.config.block_dim, (32, 4, 1));
                assert!(kernel.function.num_regs()? <= 128);
                assert_eq!(kernel.function.local_size_bytes()?, 0);
                let capacity = kernel
                    .function
                    .occupancy_max_active_blocks_per_multiprocessor(128, 0, None)?;
                assert!(capacity >= 2);
                resources.push(serde_json::json!({"label":kernel.label,"registers":kernel.function.num_regs()?,"shared":kernel.function.shared_size_bytes()?,"local":kernel.function.local_size_bytes()?,"api_max_active_blocks":capacity,"block":kernel.config.block_dim,"grid":kernel.config.grid_dim}));
            }
        }
        assert_eq!(
            resources.len(),
            executors[1].info().ordinary_bf16_four_warp_programs
        );
        std::fs::write(
            directory.join(format!("actual-ordinary-{fixture_tag}-resources.json")),
            serde_json::to_vec_pretty(&resources)?,
        )?;
    }
    if let Ok(path) = std::env::var("TNSR_BATCHED_ATTENTION_AUDIT") {
        let directory = std::path::Path::new(&path);
        std::fs::create_dir_all(directory)?;
        let mut resource = Vec::new();
        for (id, p) in executors[1].programs.iter().enumerate() {
            if let Some(p) = p.downcast_ref::<Program<BF16>>() {
                let counts = p.batched_attention_bf16_tile_counts;
                if counts.scores + counts.query_width + counts.width_query == 0 {
                    continue;
                }
                std::fs::write(
                    directory.join(format!("actual-model-{fixture_tag}-{policy:?}-{id}.ptx")),
                    &p.ptx,
                )?;
                std::fs::write(
                    directory.join(format!("actual-model-{fixture_tag}-{policy:?}-{id}.plan")),
                    &p.plan,
                )?;
                for kernel in &p.kernels {
                    assert!(
                        kernel.function.num_regs()? <= 128,
                        "actual rectangle register budget"
                    );
                    assert_eq!(kernel.function.local_size_bytes()?, 0, "rectangle spills");
                    let blocks = kernel
                        .function
                        .occupancy_max_active_blocks_per_multiprocessor(256, 0, None)?;
                    assert!(blocks >= 2, "compiled rectangle occupancy budget");
                    resource.push(serde_json::json!({"label":kernel.label,"registers":kernel.function.num_regs()?,"shared":kernel.function.shared_size_bytes()?,"local":kernel.function.local_size_bytes()?,"occupancy_max_active_blocks":blocks,"block":kernel.config.block_dim,"grid":kernel.config.grid_dim}));
                }
            }
        }
        std::fs::write(
            directory.join(format!(
                "actual-model-{fixture_tag}-{policy:?}-resources.json"
            )),
            serde_json::to_vec_pretty(&resource)?,
        )?;
    }
    for step in 0..3 {
        let mut statuses = Vec::new();
        for i in 0..2 {
            if step > 0 {
                owners[i].begin_step()?;
            }
            let microbatches = if step == 1 { 2 } else { 1 };
            for micro in 0..microbatches {
                let mut feeds = models[i].config.feeds()?;
                let offset = u32::try_from(step * 5 + micro * 17)?;
                let vocabulary = u32::try_from(models[i].config.vocab)?;
                for values in feeds.values_mut() {
                    for value in values {
                        *value = (*value + offset) % vocabulary;
                    }
                }
                owners[i].accumulate_mixed(
                    &mut executors[i],
                    &graphs[i].0,
                    HashMap::new(),
                    feeds,
                )?;
            }
            statuses.push(owners[i].step(1.0)?);
            assert_eq!(
                owners[i].gradient_norm_partials_status().last_step_used,
                microbatches == 1
            );
            assert_eq!(executors[i].transfers().parameter_copies, 0);
        }
        assert_eq!(
            statuses[0].gradient_norm.to_bits(),
            statuses[1].gradient_norm.to_bits(),
            "{policy:?} step{step} norm"
        );
        assert_eq!(
            statuses[0].gradient_scale.to_bits(),
            statuses[1].gradient_scale.to_bits(),
            "{policy:?} step{step} scale"
        );
        same_bits(
            &executors[0].read(graphs[0].1)?,
            &executors[1].read(graphs[1].1)?,
            "loss",
        );
        let gradients = [owners[0].gradients()?, owners[1].gradients()?];
        let moments = [owners[0].moments()?, owners[1].moments()?];
        for ((name, a), (_, b)) in models[0].parameters.iter().zip(models[1].parameters.iter()) {
            let (a, b) = (&a.parameter, &b.parameter);
            same_bits(
                &gradients[0][&a.id()],
                &gradients[1][&b.id()],
                &format!("{policy:?} step{step} {name} gradient"),
            );
            same_bits(
                &a.values()?,
                &b.values()?,
                &format!("{policy:?} step{step} {name} master"),
            );
            same_bits(
                &moments[0][&a.id()].first,
                &moments[1][&b.id()].first,
                "first moment",
            );
            same_bits(
                &moments[0][&a.id()].second,
                &moments[1][&b.id()].second,
                "second moment",
            );
        }
    }
    for executor in &executors {
        let info = executor.info();
        // Two master-address transitions capture the alternating variant and
        // reactivate the initial variant once. The second microbatch replays
        // the already active variant; direct cache hits count reactivations,
        // not all model launches (DirectBindings::refresh_using's same-key exit).
        assert_eq!(info.direct_capture_variants, 2);
        assert_eq!(info.direct_capture_misses, 2);
        assert_eq!(info.direct_capture_hits, 1);
    }
    Ok(())
}

#[test]
fn batched_attention_actual_b8_decoder_replay_incremental_direct_partials_three_updates()
-> Result<()> {
    if !super::tests::cuda_available() {
        return Ok(());
    }
    for policy in [Policy::TallScores, Policy::WideScores] {
        check_actual_decoder_training(
            policy,
            Config {
                width: 512,
                layers: 4,
                heads: 8,
                hidden: 1408,
                sequence: 512,
                batch: 8,
                vocab: 4096,
                epsilon: 1e-5,
                learning_rate: 1e-4,
                max_grad_norm: 1.,
            },
        )?;
    }
    Ok(())
}

#[test]
fn batched_attention_by_grid_actual_b1_h16_b2_h8_replay_and_training_bits() -> Result<()> {
    if !super::tests::cuda_available() {
        return Ok(());
    }
    for (batch, width, heads, hidden) in [(1, 1024, 16, 2816), (2, 512, 8, 1408)] {
        check_actual_decoder_training(
            Policy::TallScoresByGrid,
            Config {
                width,
                layers: 4,
                heads,
                hidden,
                sequence: 512,
                batch,
                vocab: 4096,
                epsilon: 1e-5,
                learning_rate: 1e-4,
                max_grad_norm: 1.,
            },
        )?;
    }
    Ok(())
}

#[test]
fn batched_attention_by_grid_actual_high_bh_keeps_all_twenty_four_programs() -> Result<()> {
    if !super::tests::cuda_available() {
        return Ok(());
    }
    check_actual_decoder_training(
        Policy::TallScoresByGrid,
        Config {
            width: 512,
            layers: 4,
            heads: 8,
            hidden: 1408,
            sequence: 512,
            batch: 8,
            vocab: 4096,
            epsilon: 1e-5,
            learning_rate: 1e-4,
            max_grad_norm: 1.,
        },
    )
}
