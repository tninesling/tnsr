#[cfg(test)]
mod bound_storage_tests {
    use crate::graph::{NodeIndex, TensorGraph, TensorGraphNode};
    use crate::ptx::linear::Buffer;
    use crate::ptx::linear::program::{Program, binding};
    use crate::ptx::{F32, PtxExecutor, PtxReductionMode, PtxTarget};
    use crate::tile::{Conv2dGeometry, DType, MatMulPrecision};
    use cudarc::driver::{CudaContext, CudaStream, LaunchConfig, PushKernelArg};
    use cudarc::nvrtc::Ptx;
    use half::bf16;
    use std::collections::HashMap;
    use std::sync::Arc;

    fn stream_if_available() -> Option<Arc<CudaStream>> {
        let device = match CudaContext::new(0) {
            Ok(device) => device,
            Err(error) => {
                assert_ne!(
                    std::env::var("TNSR_REQUIRE_CUDA").as_deref(),
                    Ok("1"),
                    "{error}"
                );
                return None;
            }
        };
        let supported = PtxTarget::from_context(&device)
            .unwrap()
            .compute_capability
            .0
            >= 8;
        if !supported {
            assert_ne!(
                std::env::var("TNSR_REQUIRE_CUDA").as_deref(),
                Ok("1"),
                "BF16 tests require SM80+"
            );
            return None;
        }
        Some(device.default_stream())
    }

    fn geometry(kh: usize, kw: usize) -> Conv2dGeometry {
        let (stride, padding) = if kh == 1 { (1, 0) } else { (2, 1) };
        Conv2dGeometry {
            batch: 2,
            input_channels: 2,
            input_height: 5,
            input_width: 7,
            output_channels: 3,
            output_height: (5 + 2 * padding - kh) / stride + 1,
            output_width: (7 + 2 * padding - kw) / stride + 1,
            kernel_height: kh,
            kernel_width: kw,
            stride,
            padding,
        }
    }
    fn graph(g: Conv2dGeometry, transpose: bool) -> (TensorGraph<f32>, [NodeIndex; 3]) {
        let mut graph = TensorGraph::<f32>::new();
        let source_shape = if transpose {
            vec![g.batch, g.output_channels, g.output_height, g.output_width]
        } else {
            vec![g.batch, g.input_channels, g.input_height, g.input_width]
        };
        let output_shape = if transpose {
            vec![g.batch, g.input_channels, g.input_height, g.input_width]
        } else {
            vec![g.batch, g.output_channels, g.output_height, g.output_width]
        };
        let x = graph.graph.add_node(TensorGraphNode::Input {
            name: "x",
            shape: source_shape,
        });
        let w = graph.graph.add_node(TensorGraphNode::Input {
            name: "w",
            shape: vec![
                g.output_channels,
                g.input_channels,
                g.kernel_height,
                g.kernel_width,
            ],
        });
        let y = graph.graph.add_node(if transpose {
            TensorGraphNode::ConvTranspose2d {
                shape: output_shape,
                stride: g.stride,
                padding: g.padding,
            }
        } else {
            TensorGraphNode::Conv2d {
                shape: output_shape,
                stride: g.stride,
                padding: g.padding,
            }
        });
        graph.graph.add_edge(x, y, 0);
        graph.graph.add_edge(w, y, 1);
        (graph, [x, w, y])
    }
    fn data(len: usize, seed: usize) -> Vec<f32> {
        (0..len)
            .map(|i| {
                // Both physical dtypes contain these same already-rounded values.
                // Moderate BF16 products are exactly representable in f32, so an
                // optional JIT FMA contraction cannot invalidate this exact oracle.
                let raw = (((i * 7 + seed) % 29) as f32 - 14.0) / 19.0;
                bf16::from_f32(raw).to_f32()
            })
            .collect()
    }
    fn buffer(stream: &Arc<CudaStream>, dtype: DType, values: &[f32]) -> Buffer {
        match dtype {
            DType::F32 => Buffer::F32(stream.memcpy_stod(values).unwrap()),
            DType::BF16 => Buffer::BF16(
                stream
                    .memcpy_stod(
                        &values
                            .iter()
                            .copied()
                            .map(bf16::from_f32)
                            .collect::<Vec<_>>(),
                    )
                    .unwrap(),
            ),
            _ => unreachable!(),
        }
    }
    fn update(stream: &Arc<CudaStream>, buffer: &mut Buffer, values: &[f32]) {
        match buffer {
            Buffer::F32(v) => stream.memcpy_htod(values, v).unwrap(),
            Buffer::BF16(v) => stream
                .memcpy_htod(
                    &values
                        .iter()
                        .copied()
                        .map(bf16::from_f32)
                        .collect::<Vec<_>>(),
                    v,
                )
                .unwrap(),
        }
    }
    fn check(stream: &Arc<CudaStream>, actual: &Buffer, expected: &[f32], context: &str) {
        stream.synchronize().unwrap();
        match actual {
            Buffer::F32(v) => {
                let got = stream.memcpy_dtov(v).unwrap();
                assert_eq!(got.len(), expected.len(), "{context}");
                for (i, (&a, &e)) in got.iter().zip(expected).enumerate() {
                    assert_eq!(a.to_bits(), e.to_bits(), "{context} element{i}: {a} vs {e}");
                }
            }
            Buffer::BF16(v) => {
                let got = stream.memcpy_dtov(v).unwrap();
                assert_eq!(got.len(), expected.len(), "{context}");
                for (i, (&a, &e)) in got.iter().zip(expected).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        bf16::from_f32(e).to_bits(),
                        "{context} element{i}"
                    );
                }
            }
        }
    }
    fn reference(g: Conv2dGeometry, transpose: bool, x: &[f32], w: &[f32]) -> Vec<f32> {
        let (channels, height, width, reduction_channels) = if transpose {
            (
                g.input_channels,
                g.input_height,
                g.input_width,
                g.output_channels,
            )
        } else {
            (
                g.output_channels,
                g.output_height,
                g.output_width,
                g.input_channels,
            )
        };
        let mut result = vec![0.0; g.batch * channels * height * width];
        for n in 0..g.batch {
            for c in 0..channels {
                for y in 0..height {
                    for col in 0..width {
                        let mut sum = 0.0f32;
                        for r in 0..reduction_channels {
                            for ky in 0..g.kernel_height {
                                for kx in 0..g.kernel_width {
                                    let (source, weight) = if transpose {
                                        let py = y + g.padding;
                                        let px = col + g.padding;
                                        if py < ky || px < kx {
                                            continue;
                                        }
                                        let ny = py - ky;
                                        let nx = px - kx;
                                        if !ny.is_multiple_of(g.stride)
                                            || !nx.is_multiple_of(g.stride)
                                        {
                                            continue;
                                        }
                                        let oy = ny / g.stride;
                                        let ox = nx / g.stride;
                                        if oy >= g.output_height || ox >= g.output_width {
                                            continue;
                                        }
                                        (
                                            ((n * g.output_channels + r) * g.output_height + oy)
                                                * g.output_width
                                                + ox,
                                            ((r * g.input_channels + c) * g.kernel_height + ky)
                                                * g.kernel_width
                                                + kx,
                                        )
                                    } else {
                                        let py = y * g.stride + ky;
                                        let px = col * g.stride + kx;
                                        if py < g.padding || px < g.padding {
                                            continue;
                                        }
                                        let iy = py - g.padding;
                                        let ix = px - g.padding;
                                        if iy >= g.input_height || ix >= g.input_width {
                                            continue;
                                        }
                                        (
                                            ((n * g.input_channels + r) * g.input_height + iy)
                                                * g.input_width
                                                + ix,
                                            ((c * g.input_channels + r) * g.kernel_height + ky)
                                                * g.kernel_width
                                                + kx,
                                        )
                                    };
                                    let product: f32 = x[source] * w[weight];
                                    sum += product;
                                }
                            }
                        }
                        result[((n * channels + c) * height + y) * width + col] = sum;
                    }
                }
            }
        }
        result
    }

    #[test]
    fn bound_bf16_convolution_program_launch_preserves_f32_output() {
        let Some(stream) = stream_if_available() else {
            return;
        };
        for (kh, kw) in [(1, 1), (3, 2)] {
            for transpose in [false, true] {
                let g = geometry(kh, kw);
                for (xd, wd) in [
                    (DType::BF16, DType::F32),
                    (DType::F32, DType::BF16),
                    (DType::BF16, DType::BF16),
                ] {
                    let (graph, [x, w, y]) = graph(g, transpose);
                    let xl = graph[x].shape().iter().product();
                    let wl = graph[w].shape().iter().product();
                    let yl = graph[y].shape().iter().product();
                    let mut xb = buffer(&stream, xd, &data(xl, 1));
                    let mut wb = buffer(&stream, wd, &data(wl, 11));
                    let out = Buffer::F32(stream.alloc_zeros::<f32>(yl).unwrap());
                    let program = Program::<F32>::new_bound(
                        &stream,
                        graph,
                        HashMap::from([(x, binding(&xb, &stream)), (w, binding(&wb, &stream))]),
                        Some(binding(&out, &stream)),
                        MatMulPrecision::StrictF32,
                    )
                    .unwrap();
                    assert_eq!(program.kernels.len(), 1);
                    assert_eq!(program.kernels[0].config.block_dim, (256, 1, 1));
                    assert!(program.ptx.contains("ld.global.b16"));
                    assert!(!program.ptx.contains("st.global.b16"));
                    for seed in [1, 19] {
                        let xv = data(xl, seed);
                        let wv = data(wl, seed + 10);
                        update(&stream, &mut xb, &xv);
                        update(&stream, &mut wb, &wv);
                        for kernel in &program.kernels {
                            kernel.run(&stream).unwrap();
                        }
                        let expected = reference(g, transpose, &xv, &wv);
                        assert!(
                            expected
                                .iter()
                                .any(|&v| bf16::from_f32(v).to_f32().to_bits() != v.to_bits()),
                            "fixture must detect unintended BF16 output rounding"
                        );
                        check(
                            &stream,
                            &out,
                            &expected,
                            &format!(
                                "Program k{kh}x{kw} transpose{transpose} {xd:?}/{wd:?} seed{seed}"
                            ),
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn bound_bf16_convolution_output_rounds_once_in_f32_graph() {
        let Some(stream) = stream_if_available() else {
            return;
        };
        for (kh, kw) in [(1, 1), (3, 2)] {
            for transpose in [false, true] {
                let g = geometry(kh, kw);
                for (xd, wd) in [
                    (DType::F32, DType::F32),
                    (DType::BF16, DType::F32),
                    (DType::F32, DType::BF16),
                    (DType::BF16, DType::BF16),
                ] {
                    let (graph, [x, w, y]) = graph(g, transpose);
                    let xv = data(graph[x].shape().iter().product(), 3);
                    let wv = data(graph[w].shape().iter().product(), 17);
                    let expected = reference(g, transpose, &xv, &wv);
                    let xb = buffer(&stream, xd, &xv);
                    let wb = buffer(&stream, wd, &wv);
                    let mut out = Buffer::BF16(stream.alloc_zeros::<bf16>(expected.len()).unwrap());
                    let mut physical = HashMap::from([(y, DType::BF16)]);
                    if xd == DType::BF16 {
                        physical.insert(x, xd);
                    }
                    if wd == DType::BF16 {
                        physical.insert(w, wd);
                    }
                    let mut executor =
                        PtxExecutor::<F32>::try_new_with_reduction_mode(PtxReductionMode::Strict)
                            .unwrap();
                    executor.compile_bound_storage(&graph, &physical).unwrap();
                    let ptx = executor.module_source().unwrap();
                    assert!(ptx.contains("cvt.rn.bf16.f32"));
                    assert!(ptx.contains("st.global.b16"));
                    let module = stream.context().load_module(Ptx::from_src(&ptx)).unwrap();
                    let function = module
                        .load_function(&format!(
                            "{}_{}",
                            if transpose {
                                "conv_transpose2d"
                            } else {
                                "conv2d"
                            },
                            y.index()
                        ))
                        .unwrap();
                    let xp = binding(&xb, &stream).pointer;
                    let wp = binding(&wb, &stream).pointer;
                    let op = binding(&out, &stream).pointer;
                    let config =
                        LaunchConfig::for_num_elems(u32::try_from(expected.len()).unwrap());
                    // SAFETY: compile_bound_storage ABI exactly matches all three
                    // typed, correctly-sized live buffers on this same stream/context.
                    // F32 executor execute_compiled is intentionally not used with
                    // physical BF16 buffers; its homogeneous upload path is inapplicable.
                    unsafe {
                        stream
                            .launch_builder(&function)
                            .arg(&xp)
                            .arg(&wp)
                            .arg(&op)
                            .launch(config)
                            .unwrap();
                    }
                    check(
                        &stream,
                        &out,
                        &expected,
                        &format!(
                            "physical BF16 output k{kh}x{kw} transpose{transpose} {xd:?}/{wd:?}"
                        ),
                    );
                    // A second launch into the same BF16 storage must overwrite,
                    // rather than accumulate or retain any prior f32 output layout.
                    match &mut out {
                        Buffer::BF16(v) => stream
                            .memcpy_htod(&vec![bf16::from_f32(-99.0); expected.len()], v)
                            .unwrap(),
                        _ => unreachable!(),
                    }
                    unsafe {
                        stream
                            .launch_builder(&function)
                            .arg(&xp)
                            .arg(&wp)
                            .arg(&op)
                            .launch(config)
                            .unwrap();
                    }
                    check(&stream, &out, &expected, "repeat bound BF16 output");
                }
            }
        }
    }
}
