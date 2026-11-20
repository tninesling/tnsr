use super::*;

#[test]
fn ordinary_four_warp_actual_decoder_v18_options_replay_and_three_updates() -> Result<()> {
    if !super::tests::cuda_available() {
        return Ok(());
    }
    check_actual_decoder(8)
}

#[test]
fn ordinary_four_warp_actual_decoder_b16_joint_split_replay_and_three_updates() -> Result<()> {
    if !super::tests::cuda_available() {
        return Ok(());
    }
    check_actual_decoder(16)
}

fn check_actual_decoder(batch: usize) -> Result<()> {
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
            ordinary_bf16_four_warp_block: true,
            layer_norm_pair_fusion: true,
            split_k_bf16_tile_policy: crate::tile::SplitKBf16TilePolicy::Square64,
            ..Default::default()
        },
    )
}

#[test]
fn ordinary_four_warp_linear_preserves_bias_before_final_bf16_round() -> Result<()> {
    use crate::tensor::{Parameter, TensorExpr};
    if !super::tests::cuda_available() {
        return Ok(());
    }
    let (rows, cols, k) = (2048, 256, 64);
    let w = Parameter::new(
        (0..cols * k)
            .map(|i| if i % k == 0 { 1.0078125 } else { 0. })
            .collect(),
        vec![cols, k],
    );
    let bias = Parameter::new(vec![-1.015625; cols], vec![cols]);
    let y = TensorExpr::input("x", vec![rows, k])
        .linear_bf16(TensorExpr::from(w).transpose(), bias.into())?
        .cast(ValueDType::F32);
    let graph: TensorGraph<f32> = y.into();
    let out = *graph.toposort().last().unwrap();
    for enabled in [false, true] {
        let mut executor = MixedExecutor::new_with_options(
            &graph,
            MatMulPrecision::NativeHalf,
            true,
            MixedOptions {
                physical_order_casts: true,
                shared_memory_reuse: true,
                ordinary_bf16_four_warp_block: enabled,
                ..Default::default()
            },
        )?;
        assert_eq!(
            executor.info().ordinary_bf16_four_warp_programs,
            usize::from(enabled)
        );
        executor.upload(HashMap::from([(
            "x".into(),
            (0..rows * k)
                .map(|i| if i % k == 0 { 1.0078125 } else { 0. })
                .collect(),
        )]))?;
        executor.run()?;
        // F32 product1.01568603515625 minus bias1.015625 is2^-14. An
        // intermediate BF16 MatMul round would incorrectly turn it into zero.
        assert_eq!(executor.read(out)?, vec![1. / 16384.; rows * cols]);
    }
    Ok(())
}
