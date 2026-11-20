//! Untimed mixed-storage leaf: exact flat writes, partial rows and repaired norm.
use super::*;
use crate::{ptx::linear::Buffer, tensor::Parameter};
use half::bf16;
#[test]
#[ignore = "manual small accumulation norm batching mixed BF16/F32/allpatterns/tails/repair proof"]
fn small_accumulation_norm_leaf_exact_mixed_patterns_and_repair() -> Result<()> {
    let available = CudaContext::new(0).is_ok();
    ensure!(
        available || std::env::var_os("TNSR_REQUIRE_CUDA").is_none(),
        "CUDA required"
    );
    if !available {
        return Ok(());
    }
    let small = [
        1, 3, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049,
        4095, 4096,
    ];
    let mut lens = Vec::new();
    for l in small {
        lens.extend([l, 4096]);
    }
    lens.extend([4097, 8193]);
    let parameters = lens
        .iter()
        .map(|&n| (Parameter::new(vec![0.; n], vec![n]), 0))
        .collect();
    let mut owner = ResidentAdam::new_with_graph_replay(parameters, vec![AdamGroup::new(1e-3)])?;
    owner.set_experimental_first_accumulation_overwrite(true)?;
    owner.set_experimental_gradient_norm_partials(true)?;
    owner.set_experimental_small_accumulation_norm_batches(true)?;
    owner.begin_step()?;
    let ranges = owner
        .parameters
        .iter()
        .map(|p| p.range.clone())
        .collect::<Vec<_>>();
    let padded = owner.padded_len;
    let mut sources = lens
        .iter()
        .enumerate()
        .map(|(i, &len)| {
            if i < 38 && i % 2 == 1 {
                owner.stream.alloc_zeros::<bf16>(len).map(Buffer::BF16)
            } else {
                owner.stream.alloc_zeros::<f32>(len).map(Buffer::F32)
            }
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let values = sources
        .iter()
        .enumerate()
        .map(|(i, b)| {
            let binding = crate::ptx::linear::program::binding(b, &owner.stream);
            (
                binding,
                view(i, binding.len, binding.dtype),
                vec![binding.len],
            )
        })
        .collect::<Vec<_>>();
    let legacy = kernels::Kernels::compile(
        &owner.device,
        values
            .iter()
            .zip(&ranges)
            .map(|((_, v, s), r)| norm_partials::accumulate_ir(v, s, r.clone()))
            .collect::<Result<_>>()?,
    )?;
    let mut old = owner.stream.alloc_zeros::<f32>(padded)?;
    let mut oldworkspace = norm_partials::Workspace::new(
        &owner.stream,
        owner.stream_gate.clone(),
        norm_partials::PartialLayout::new(ranges.clone(), padded)?,
    )?;
    for phase in 0..3 {
        let mut bf16offset = 0;
        for (i, source) in sources.iter_mut().enumerate() {
            match source {
                Buffer::BF16(b) => {
                    let x = (0..b.len())
                        .map(|j| {
                            if phase == 1 {
                                bf16::from_bits((bf16offset + j) as u16)
                            } else {
                                bf16::from_f32(if j % 19 == 0 {
                                    -0.
                                } else {
                                    ((i + j + phase) % 17) as f32 / 64. - 0.125
                                })
                            }
                        })
                        .collect::<Vec<_>>();
                    bf16offset += b.len();
                    owner.stream.memcpy_htod(&x, b)?;
                }
                Buffer::F32(b) => {
                    let x = (0..b.len())
                        .map(|j| {
                            if j % 19 == 0 {
                                -0.
                            } else {
                                ((i + j + phase) % 17) as f32 / 64. - 0.125
                            }
                        })
                        .collect::<Vec<_>>();
                    owner.stream.memcpy_htod(&x, b)?;
                }
            }
        }
        owner.stream.memset_zeros(&mut old)?;
        owner
            .stream
            .memset_zeros(Arc::get_mut(&mut owner.gradients).context("gradient aliases")?)?;
        for (i, ((binding, _, _), r)) in values.iter().zip(&ranges).enumerate() {
            let function = legacy.module.load_function(&legacy.names[i])?;
            let mut target = old.slice_mut(r.clone());
            let config = oldworkspace.layout.accumulation_config(i)?;
            let mut launch = owner.stream.launch_builder(&function);
            launch
                .arg(&binding.pointer)
                .arg(&mut target)
                .arg(oldworkspace.partials_mut()?);
            // SAFETY: unchanged legacy ABI; sources/output extents are retained.
            unsafe { launch.launch(config) }?;
        }
        assert!(
            owner.try_small_accumulation_norm(&values)?,
            "{:?}",
            owner.small_accumulation_norm_batches_status()
        );
        assert_eq!(
            owner
                .small_accumulation_norm_batches_status()
                .selected_members,
            38
        );
        assert_eq!(
            owner
                .small_accumulation_norm_batches_status()
                .batched_launches,
            5
        );
        assert_eq!(
            owner
                .small_accumulation_norm_batches_status()
                .standalone_launches,
            2
        );
        assert_eq!(
            owner
                .small_accumulation_norm_batches_status()
                .actual_launches,
            7
        );
        let a = oldworkspace.run(&owner.stream, &old)?;
        let workspace = owner.norm_partial_workspace.as_mut().unwrap();
        let b = workspace.run(&owner.stream, &owner.gradients)?;
        assert!(a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()));
        let mut px = vec![0f32; oldworkspace.layout.rows];
        let mut py = vec![0f32; workspace.layout.rows];
        owner
            .stream
            .memcpy_dtoh(oldworkspace.partials_mut()?, &mut px)?;
        owner
            .stream
            .memcpy_dtoh(workspace.partials_mut()?, &mut py)?;
        owner.stream.synchronize()?;
        for (i, (a, b)) in px.iter().zip(&py).enumerate() {
            assert!(
                a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()),
                "partial{phase}/{i}: {a}/{b}"
            );
        }
        let mut x = vec![0.; padded];
        let mut y = vec![0.; padded];
        owner.stream.memcpy_dtoh(&old, &mut x)?;
        owner.stream.memcpy_dtoh(owner.gradients.as_ref(), &mut y)?;
        owner.stream.synchronize()?;
        for (i, (a, b)) in x.iter().zip(&y).enumerate() {
            assert!(
                a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()),
                "flat{phase}/{i}: {a}/{b}"
            );
        }
    }
    Ok(())
}
