use super::*;
use crate::{
    tensor::{Parameter, TensorExpr},
    tile::LargeBf16TilePolicy as Policy,
};

fn same(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        if b.is_nan() {
            assert!(a.is_nan());
        } else {
            assert_eq!(a.to_bits(), b.to_bits(), "{a:?} vs {b:?}");
        }
    }
}

#[test]
fn large_rectangles_mixed_forward_all_gradients_bias_rounding_replay_and_views() -> Result<()> {
    if !super::tests::cuda_available() {
        return Ok(());
    }
    for transpose in [false, true] {
        let rows = 2048;
        let k = 128;
        let n = 512;
        let x = Parameter::new(
            (0..rows * k)
                .map(|i| (i % 17) as f32 / 64. - 0.125)
                .collect(),
            vec![rows, k],
        );
        let w = Parameter::new(
            (0..k * n).map(|i| (i % 13) as f32 / 64. - 0.0625).collect(),
            if transpose { vec![n, k] } else { vec![k, n] },
        );
        let bias = Parameter::new(vec![0.03125; n], vec![n]);
        let weight = if transpose {
            TensorExpr::from(w.clone()).transpose()
        } else {
            w.clone().into()
        };
        let output = (TensorExpr::from(x.clone()) + TensorExpr::input("delta", vec![rows, k]))
            .linear_bf16(weight, bias.clone().into())?
            .cast(crate::tensor::ValueDType::F32);
        let mut graph = TensorGraph::new();
        let forward = graph.add_expr(&output, false);
        let loss = graph.add_expr(
            &(output * TensorExpr::input("seed", vec![rows, n])).mean_all(),
            false,
        );
        let graph = graph.with_gradients(loss);
        for policy in [Policy::Tall, Policy::Wide, Policy::TallThenWide] {
            let options = MixedOptions {
                shared_memory_reuse: true,
                physical_order_casts: true,
                ..MixedOptions::default()
            };
            let mut old = MixedExecutor::new_with_options(
                &graph,
                MatMulPrecision::NativeHalf,
                true,
                options,
            )?;
            let mut new = MixedExecutor::new_with_options(
                &graph,
                MatMulPrecision::NativeHalf,
                true,
                MixedOptions {
                    large_bf16_tile_policy: policy,
                    ..options
                },
            )?;
            let counts = (
                new.info().large_bf16_tall_programs,
                new.info().large_bf16_wide_programs,
            );
            assert!(
                if policy == Policy::Wide {
                    counts.1 > 0 && counts.0 == 0
                } else {
                    counts.0 > 0 && counts.1 == 0
                },
                "actualselected {counts:?}"
            );
            assert_eq!(old.info().kernels, new.info().kernels);
            assert_eq!(old.info().f32_bytes, new.info().f32_bytes);
            assert_eq!(old.info().bf16_bytes, new.info().bf16_bytes);
            for program in new
                .programs
                .iter()
                .filter_map(|p| p.downcast_ref::<Program<BF16>>())
                .filter(|p| p.large_bf16_tile_counts != (0, 0))
            {
                assert!(program.ptx.contains(".shared .align 32 .b8 shared[32768];"));
                for kernel in program
                    .kernels
                    .iter()
                    .filter(|k| k.label.contains("large_bf16_tile="))
                {
                    assert_eq!(kernel.function.shared_size_bytes()?, 32768);
                    assert!(
                        kernel.function.num_regs()? <= 128,
                        "qualification must honor the conservative register reserve"
                    );
                    assert_eq!(
                        kernel.function.local_size_bytes()?,
                        0,
                        "qualified fixture should not spill"
                    );
                    assert!(
                        kernel
                            .function
                            .occupancy_max_active_blocks_per_multiprocessor(256, 0, None)?
                            >= 2,
                        "static CUDA capacity query, not measured occupancy"
                    );
                }
                if let Some(directory) = std::env::var_os("TNSR_LARGE_TILE_PTX_DIR") {
                    std::fs::create_dir_all(&directory)?;
                    let index = new
                        .programs
                        .iter()
                        .position(|p| {
                            p.downcast_ref::<Program<BF16>>()
                                .is_some_and(|p| std::ptr::eq(p, program))
                        })
                        .unwrap();
                    std::fs::write(
                        std::path::PathBuf::from(&directory)
                            .join(format!("mixed-{policy:?}-transpose{transpose}-{index}.ptx")),
                        &program.ptx,
                    )?;
                    let baseline = old.programs[index]
                        .downcast_ref::<Program<BF16>>()
                        .expect("corresponding baseline program");
                    std::fs::write(
                        std::path::PathBuf::from(&directory).join(format!(
                            "mixed-{policy:?}-transpose{transpose}-{index}-old.ptx"
                        )),
                        &baseline.ptx,
                    )?;
                    let mut resource = Vec::new();
                    for (arm, p) in [("old", baseline), ("candidate", program)] {
                        for kernel in &p.kernels {
                            resource.push(serde_json::json!({"arm":arm,"label":kernel.label,"block":kernel.config.block_dim,"grid":kernel.config.grid_dim,"registers":kernel.function.num_regs()?,"shared":kernel.function.shared_size_bytes()?,"local":kernel.function.local_size_bytes()?,"cuda_occupancy_api_max_active_blocks":kernel.function.occupancy_max_active_blocks_per_multiprocessor(256,0,None)?}));
                        }
                    }
                    std::fs::write(
                        std::path::PathBuf::from(directory).join(format!(
                            "mixed-{policy:?}-transpose{transpose}-{index}-resources.json"
                        )),
                        serde_json::to_vec_pretty(&resource)?,
                    )?;
                }
            }
            for phase in 0..3 {
                let feeds = HashMap::from([
                    (
                        "delta".into(),
                        (0..rows * k)
                            .map(|i| ((i + phase) % 7) as f32 / 128.)
                            .collect(),
                    ),
                    (
                        "seed".into(),
                        (0..rows * n)
                            .map(|i| ((i + phase) % 5) as f32 / 16. - 0.125)
                            .collect(),
                    ),
                ]);
                old.upload(feeds.clone())?;
                new.upload(feeds)?;
                old.run()?;
                new.run()?;
                for node in [forward, loss] {
                    let expected = old.read(node)?;
                    assert!(expected.iter().all(|v| v.is_finite()));
                    same(&new.read(node)?, &expected);
                }
                let a = old.gradients(&graph)?;
                let b = new.gradients(&graph)?;
                for id in [x.id(), w.id(), bias.id()] {
                    same(&b[&id], &a[&id]);
                }
            }
            assert!(new.captured.is_some(), "mixedgraphmustreplay");
        }
    }
    Ok(())
}

#[test]
fn large_policy_rejects_split_k_batch_tails_and_missing_reuse() -> Result<()> {
    if !super::tests::cuda_available() {
        return Ok(());
    }
    for (m, n, k, batch, reuse) in [
        (2049, 512, 128, false, true),
        (2048, 513, 128, false, true),
        (2048, 512, 129, false, true),
        (2048, 512, 128, false, false),
        (2048, 512, 128, true, true),
        (2048, 512, 8192, false, true),
    ] {
        let a = if batch { vec![1, m, k] } else { vec![m, k] };
        let b = if batch { vec![1, k, n] } else { vec![k, n] };
        let output = TensorExpr::input("a", a)
            .cast(crate::tensor::ValueDType::BF16)
            .matmul(TensorExpr::input("b", b).cast(crate::tensor::ValueDType::BF16));
        let graph: TensorGraph<f32> = output.cast(crate::tensor::ValueDType::F32).into();
        let executor = MixedExecutor::new_with_options(
            &graph,
            MatMulPrecision::NativeHalf,
            false,
            MixedOptions {
                shared_memory_reuse: reuse,
                large_bf16_tile_policy: Policy::TallThenWide,
                ..MixedOptions::default()
            },
        )?;
        assert_eq!(
            (
                executor.info().large_bf16_tall_programs,
                executor.info().large_bf16_wide_programs
            ),
            (0, 0)
        );
    }
    Ok(())
}

#[test]
fn large_rectangles_linear_bias_precedes_the_only_bf16_output_round() -> Result<()> {
    if !super::tests::cuda_available() {
        return Ok(());
    }
    let (m, k, n) = (2048, 128, 512);
    let x: Vec<_> = (0..m * k)
        .map(|i| if i % k == 0 { 1.0078125 } else { 0. })
        .collect();
    let w: Vec<_> = (0..k * n)
        .map(|i| if i < n { 1.0078125 } else { 0. })
        .collect();
    let output = TensorExpr::constant(x, vec![m, k])
        .linear_bf16(
            TensorExpr::constant(w, vec![k, n]),
            TensorExpr::constant(vec![-1.015625; n], vec![n]),
        )?
        .cast(crate::tensor::ValueDType::F32);
    let graph: TensorGraph<f32> = output.into();
    let root = graph.graph.node_indices().next_back().unwrap();
    for policy in [
        Policy::Off,
        Policy::Tall,
        Policy::Wide,
        Policy::TallThenWide,
    ] {
        let mut executor = MixedExecutor::new_with_options(
            &graph,
            MatMulPrecision::NativeHalf,
            false,
            MixedOptions {
                shared_memory_reuse: true,
                large_bf16_tile_policy: policy,
                ..MixedOptions::default()
            },
        )?;
        executor.upload(HashMap::new())?;
        executor.run()?;
        let actual = executor.read(root)?;
        assert!(
            actual
                .iter()
                .all(|x| x.to_bits() == (1.0f32 / 16384.).to_bits()),
            "prematureBF16matmulroundwouldzero this residual"
        );
    }
    Ok(())
}

#[test]
fn large_policy_actual_four_layer_transposed_head_activation() -> Result<()> {
    use super::test_decoder::{Config, synthetic};
    if !super::tests::cuda_available() {
        return Ok(());
    }
    let model = synthetic(Config {
        width: 512,
        layers: 4,
        heads: 8,
        hidden: 1408,
        sequence: 128,
        batch: 1,
        vocab: 4096,
        epsilon: 1e-5,
        learning_rate: 1e-4,
        max_grad_norm: 1.,
    });
    let (graph, _) = model.training_graph_with_options(crate::graph::GradientOptions {
        compact_softmax: true,
    })?;
    for policy in [Policy::Tall, Policy::Wide, Policy::TallThenWide] {
        let executor = MixedExecutor::new_with_options(
            &graph,
            MatMulPrecision::NativeHalf,
            false,
            MixedOptions {
                large_bf16_tile_policy: policy,
                shared_memory_reuse: true,
                physical_order_casts: true,
                transposed_bf16_outputs: true,
                narrow_bf16_tile: true,
                medium_bf16_tile_policy: crate::tile::MediumBf16TilePolicy::SquareThenTall,
                attention_softmax_fusion: true,
                attention_softmax_backward_fusion: true,
                qkv_grad_pack_fusion: true,
                swiglu_pair_fusion: true,
                column_reduction_policy: crate::tile::ColumnReductionPolicy::Four,
                ..MixedOptions::default()
            },
        )?;
        let counts = (
            executor.info().large_bf16_tall_programs,
            executor.info().large_bf16_wide_programs,
        );
        eprintln!(
            "actual four-layer W512 B1 S128 {policy:?} selected tall/wide programs={counts:?}"
        );
        assert!(
            if policy == Policy::Wide {
                counts.1 > 0 && counts.0 == 0
            } else {
                counts.0 > 0 && counts.1 == 0
            },
            "policy must not silently miss real transposed-head geometry"
        );
    }
    Ok(())
}

/// Real decoder graph + actual transpose matcher/split-K exclusions, evaluated
/// at isolated row-major physical GEMM boundaries without allocating activation
/// storage. Full compiled model copy-map/resource activation is a campaign gate.
#[test]
fn large_policy_real_model_boundary_candidates_b8_b32_and_211m_without_gpu() -> Result<()> {
    use super::test_decoder::{Config, synthetic};
    use crate::{
        tensor::ValueDType,
        tile::{MatMulPipeline, MatMulSchedule, MatMulSharedLayout},
    };
    let target = PtxTarget::sm80();
    for (width, hidden, sequence, batch, expected) in [
        (512, 1408, 512, 8, [(43, 0), (0, 43), (43, 0)]),
        (512, 1408, 512, 32, [(42, 0), (0, 42), (42, 0)]),
        (1024, 2752, 512, 1, [(5, 0), (0, 13), (5, 8)]),
        (2048, 5504, 128, 1, [(21, 0), (0, 21), (21, 0)]),
    ] {
        let model = synthetic(Config {
            width,
            layers: 4,
            heads: width / 64,
            hidden,
            sequence,
            batch,
            vocab: 4096,
            epsilon: 1e-5,
            learning_rate: 1e-4,
            max_grad_norm: 1.,
        });
        let (graph, _) = model.training_graph_with_options(crate::graph::GradientOptions {
            compact_softmax: true,
        })?;
        let dtypes = value_dtypes(&graph);
        for (index, policy) in [Policy::Tall, Policy::Wide, Policy::TallThenWide]
            .into_iter()
            .enumerate()
        {
            let mut counts = (0, 0);
            let mut transposed = 0;
            for node in graph.graph.node_indices() {
                let shape = match &graph[node] {
                    TensorGraphNode::MatMul { shape } | TensorGraphNode::Linear { shape }
                        if shape.len() == 2 =>
                    {
                        shape
                    }
                    _ => continue,
                };
                if dtypes.get(&node) != Some(&ValueDType::BF16) {
                    continue;
                }
                let inputs = graph.inputs(node);
                let left_shape = graph[inputs[0]].shape();
                let k = left_shape[left_shape.len() - 1];
                let (mut m, mut n) = (shape[0], shape[1]);
                if matches!(graph[node], TensorGraphNode::MatMul { .. })
                    && m >= 128
                    && n >= 128
                    && k >= 2048
                    && k / m.max(n) >= 4
                    && k.is_multiple_of(4)
                {
                    continue;
                }
                let physical_transpose = super::transpose_output::eligible(&graph, node, &dtypes);
                if physical_transpose {
                    std::mem::swap(&mut m, &mut n);
                }
                let old = MatMulSchedule::select_with_resources(
                    m,
                    n,
                    k,
                    DType::BF16,
                    target.matmul_capabilities(),
                    MatMulPrecision::NativeHalf,
                    target.matmul_resources,
                );
                for candidate in policy.candidates(old, &[m, n], true) {
                    let configured = candidate
                        .with_shared_layout(MatMulSharedLayout::Auto, target.matmul_resources)?
                        .with_pipeline(MatMulPipeline::Auto, true, true, target.matmul_resources);
                    if Policy::configured_candidate_fits(configured, target.matmul_resources) {
                        if configured.block_tile.m == 128 {
                            counts.0 += 1;
                        } else {
                            counts.1 += 1;
                        }
                        transposed += usize::from(physical_transpose);
                        break;
                    }
                }
            }
            eprintln!(
                "actual4layer W{width} B{batch} S{sequence} {policy:?} boundaryshape tall/wide={counts:?},transposed={transposed};notcompiledcopy/resourceactivation"
            );
            assert_eq!(counts, expected[index]);
        }
    }
    Ok(())
}

/// Actual copied maps/selected plan/JIT resources, distinct from shape-only
/// opportunities. Initial plain-MatMul scope excludes forward Linear+bias.
#[test]
#[ignore = "Square resource admission not achieved; decoder candidate unqualified"]
fn square128_streamed_actual_decoder_program_counter_and_driver_proof() -> Result<()> {
    use super::test_decoder::{Config, synthetic};
    use crate::ptx::resident::{AdamGroup, ResidentAdam};
    if !super::tests::cuda_available() {
        return Ok(());
    }
    let model = synthetic(Config {
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
    });
    let (graph, _) = model.training_graph_with_options(crate::graph::GradientOptions {
        compact_softmax: true,
    })?;
    let mut owner = ResidentAdam::new_with_graph_replay(
        model
            .parameters
            .iter()
            .map(|(_, p)| (p.parameter.clone(), 0))
            .collect(),
        vec![AdamGroup::new(1e-4)],
    )?;
    owner.begin_step()?;
    let e = owner.mixed_executor_with_options(
        &graph,
        MatMulPrecision::Tf32x3,
        MixedOptions {
            large_bf16_tile_policy: Policy::Square128Streamed,
            shared_memory_reuse: true,
            physical_order_casts: true,
            transposed_bf16_outputs: true,
            narrow_bf16_tile: true,
            medium_bf16_tile_policy: crate::tile::MediumBf16TilePolicy::SquareThenTall,
            batched_attention_bf16_tile_policy:
                crate::tile::BatchedAttentionBf16TilePolicy::TallScoresByGrid,
            split_k_bf16_tile_policy: crate::tile::SplitKBf16TilePolicy::Square64,
            attention_softmax_fusion: true,
            attention_softmax_backward_fusion: true,
            qkv_grad_pack_fusion: true,
            swiglu_pair_fusion: true,
            layer_norm_pair_fusion: true,
            indexed_loss_cooperative_ordered_backward: true,
            column_reduction_policy: crate::tile::ColumnReductionPolicy::Four,
            incremental_allocation_reuse: true,
            direct_master_bindings: true,
            ..MixedOptions::default()
        },
    )?;
    let mut totals = (0, 0);
    let mut rows = Vec::new();
    for (index, program) in e
        .programs
        .iter()
        .filter_map(|p| p.downcast_ref::<Program<BF16>>())
        .enumerate()
    {
        let counts = program.large_square128_streamed_counts;
        totals.0 += counts.0;
        totals.1 += counts.1;
        if counts == (0, 0) {
            continue;
        }
        let selected = program
            .kernels
            .iter()
            .filter(|k| k.label.contains("square128_streamed=true"))
            .count();
        assert_eq!(selected, counts.0);
        for kernel in &program.kernels {
            if !kernel.label.contains("square128_streamed=true") {
                continue;
            }
            let regs = kernel.function.num_regs()?;
            let shared = kernel.function.shared_size_bytes()?;
            let local = kernel.function.local_size_bytes()?;
            let capacity = kernel
                .function
                .occupancy_max_active_blocks_per_multiprocessor(256, 0, None)?;
            assert!((1..=128).contains(&regs));
            assert_eq!(local, 0);
            assert!(capacity >= 2);
            assert_eq!(kernel.config.block_dim, (32, 8, 1));
            assert!((34 * 1024..=40 * 1024).contains(&shared));
            rows.push(serde_json::json!({"program":index,"label":kernel.label,"grid":kernel.config.grid_dim,"block":kernel.config.block_dim,"registers":regs,"shared_bytes":shared,"local_bytes":local,"static_block_capacity":capacity,"capacity_is_not_measured_occupancy":true}));
        }
        if let Some(dir) = std::env::var_os("TNSR_SQUARE128_PTX_DIR") {
            std::fs::create_dir_all(&dir)?;
            std::fs::write(
                std::path::PathBuf::from(dir).join(format!(
                    "actual-decoder-program-{index}-selected{}-fallback{}.ptx",
                    counts.0, counts.1
                )),
                &program.ptx,
            )?;
        }
    }
    assert!(
        totals.0 > 0,
        "no actual square kernel survived copy/resource qualification; fallbacks={}",
        totals.1
    );
    assert_eq!(
        totals,
        (
            e.info().large_bf16_square128_streamed_programs,
            e.info().large_bf16_square128_streamed_fallback_programs
        )
    );
    assert_eq!(e.info().large_bf16_tall_programs, 0);
    assert_eq!(e.info().large_bf16_wide_programs, 0);
    eprintln!(
        "Square128Streamed actual B8 decoder selected/fallback={totals:?}; plain-MatMul only, not projected forward Linear coverage"
    );
    if let Some(dir) = std::env::var_os("TNSR_SQUARE128_PTX_DIR") {
        std::fs::write(
            std::path::PathBuf::from(dir).join("actual-decoder-resources.json"),
            serde_json::to_vec_pretty(
                &serde_json::json!({"selected":totals.0,"fallback":totals.1,"kernels":rows,"scope":"actual compiled B8 four-layer decoder; no timing or whole-model equivalence claim"}),
            )?,
        )?;
    }
    Ok(())
}
