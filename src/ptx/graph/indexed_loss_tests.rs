use super::*;
use crate::graph::{TensorGraph, TensorGraphNode};

fn graph(rows: usize, vocabulary: usize) -> (TensorGraph<f32>, [NodeIndex; 4]) {
    let mut g = TensorGraph::new();
    let x = g.graph.add_node(TensorGraphNode::Input {
        name: "x",
        shape: vec![rows, vocabulary],
    });
    let t = g.graph.add_node(TensorGraphNode::Input {
        name: "t",
        shape: vec![rows],
    });
    let dy = g.graph.add_node(TensorGraphNode::Input {
        name: "dy",
        shape: vec![rows],
    });
    let out = g
        .graph
        .add_node(TensorGraphNode::IndexedCrossEntropyBackward {
            shape: vec![rows, vocabulary],
        });
    for (slot, n) in [x, t, dy].into_iter().enumerate() {
        g.graph.add_edge(n, out, slot);
    }
    (g, [x, t, dy, out])
}

#[test]
fn row_launch_metadata_and_barriers_follow_physical_storage() {
    for (vocabulary, eligible) in [
        (255, false),
        (256, true),
        (257, true),
        (4096, true),
        (65537, false),
    ] {
        let (g, [_, _, _, out]) = graph(3, vocabulary);
        let tiles = TileGraph::from_graph(&g);
        let ptx = PtxGraph::from(tiles);
        let config = ptx.indexed_loss_row_launch(out, 3 * vocabulary).unwrap();
        assert_eq!(config.is_some(), eligible);
        assert_eq!(ptx.to_ptx().contains("bar.sync"), eligible);
        if let Some(config) = config {
            assert_eq!(config.grid_dim, (3, 1, 1));
            assert_eq!(config.block_dim, (256, 1, 1));
            assert!(
                ptx.indexed_loss_row_launch(out, 3 * vocabulary + 1)
                    .is_err()
            );
        }
        for name in ["logits", "grad_output", "output"] {
            let mut tiles = TileGraph::from_graph(&g);
            tiles.graph[out]
                .params
                .iter_mut()
                .find(|p| p.name == name)
                .unwrap()
                .dtype = DType::BF16;
            let ptx = PtxGraph::from(tiles);
            assert!(
                ptx.indexed_loss_row_launch(out, 3 * vocabulary)
                    .unwrap()
                    .is_none()
            );
            assert!(!ptx.to_ptx().contains("bar.sync"));
        }
    }
    let (g, [_, _, _, out]) = graph(0, 4096);
    assert!(
        PtxGraph::from(TileGraph::from_graph(&g))
            .indexed_loss_row_launch(out, 0)
            .unwrap()
            .is_none()
    );
}

#[test]
fn row_program_checks_f32_and_half_bound_launches_with_changed_indices() {
    use crate::ptx::F32 as StorageF32;
    use crate::ptx::linear::program::{Program, binding};
    use crate::ptx::linear::{Binding, Buffer};
    use cudarc::driver::{CudaContext, DevicePtr};
    let device = match CudaContext::new(0) {
        Ok(v) => v,
        Err(err) => {
            assert_ne!(
                std::env::var("TNSR_REQUIRE_CUDA").as_deref(),
                Ok("1"),
                "{err}"
            );
            return;
        }
    };
    let stream = device.default_stream();
    for half_bound in [false, true] {
        let (mut g, [x, t, dy, _]) = graph(3, 257);
        g.graph[t] = TensorGraphNode::IndexSource {
            source: crate::tensor::IndexTensor::input("t", vec![3]),
        };
        let logits = if half_bound {
            Buffer::BF16(
                stream
                    .memcpy_stod(&vec![half::bf16::ZERO; 3 * 257])
                    .unwrap(),
            )
        } else {
            Buffer::F32(stream.memcpy_stod(&vec![0.0_f32; 3 * 257]).unwrap())
        };
        let mut targets = stream.memcpy_stod(&[0_u32, 256, 3]).unwrap();
        let mut upstream = Buffer::F32(stream.memcpy_stod(&[1.0_f32, -1.0, 0.0]).unwrap());
        let output = Buffer::F32(stream.alloc_zeros::<f32>(3 * 257).unwrap());
        let target_binding = Binding {
            pointer: targets.device_ptr(&stream).0,
            len: 3,
            dtype: DType::U32,
        };
        let program = Program::<StorageF32>::new_bound(
            &stream,
            g,
            HashMap::from([
                (x, binding(&logits, &stream)),
                (t, target_binding),
                (dy, binding(&upstream, &stream)),
            ]),
            Some(binding(&output, &stream)),
            crate::tile::MatMulPrecision::StrictF32,
        )
        .unwrap();
        assert_eq!(program.kernels.len(), 1);
        assert_eq!(
            program.kernels[0].config.grid_dim,
            if half_bound { (4, 1, 1) } else { (3, 1, 1) }
        );
        assert_eq!(program.ptx.contains("bar.sync"), !half_bound);
        for feed in 0..2 {
            let t = if feed == 0 {
                [0_u32, 256, 3]
            } else {
                [256, 0, 42]
            };
            let u = if feed == 0 {
                [1.0_f32, -1.0, 0.0]
            } else {
                [-0.5, 0.25, 1.0]
            };
            stream.memcpy_htod(&t, &mut targets).unwrap();
            if let Buffer::F32(v) = &mut upstream {
                stream.memcpy_htod(&u, v).unwrap();
            }
            for kernel in &program.kernels {
                kernel.run(&stream).unwrap();
            }
            stream.synchronize().unwrap();
            if let Buffer::F32(v) = &output {
                let actual = stream.memcpy_dtov(v).unwrap();
                for (i, &a) in actual.iter().enumerate() {
                    let r = i / 257;
                    let c = i % 257;
                    let expected = (1.0 / 257.0 - f32::from(u8::from(c == t[r] as usize))) * u[r];
                    assert!(
                        (a - expected).abs() < 3e-7,
                        "half={half_bound},feed={feed},index={i}: {a} != {expected}"
                    );
                }
            }
        }
    }
}

#[test]
fn forward_row_launch_tracks_output_extent_and_physical_dtype() {
    for vocabulary in [255, 256, 257, 4096, 65537] {
        let mut g = TensorGraph::<f32>::new();
        let x = g.graph.add_node(TensorGraphNode::Input {
            name: "x",
            shape: vec![33, vocabulary],
        });
        let t = g.graph.add_node(TensorGraphNode::Input {
            name: "t",
            shape: vec![33],
        });
        let out = g
            .graph
            .add_node(TensorGraphNode::IndexedCrossEntropy { shape: vec![33] });
        g.graph.add_edge(x, out, 0);
        g.graph.add_edge(t, out, 1);
        let ptx = PtxGraph::from(TileGraph::from_graph(&g));
        let config = ptx.indexed_loss_row_launch(out, 33).unwrap();
        assert_eq!(config.is_some(), (256..=65536).contains(&vocabulary));
        assert!(!ptx.to_ptx().contains("bar.sync"));
        if let Some(c) = config {
            assert_eq!(c.grid_dim, (33, 1, 1));
            assert_eq!(c.block_dim, (32, 1, 1));
            assert!(ptx.indexed_loss_row_launch(out, 34).is_err());
        }
        for name in ["logits", "output"] {
            for dtype in [DType::BF16, DType::F16] {
                let mut tiles = TileGraph::from_graph(&g);
                tiles.graph[out]
                    .params
                    .iter_mut()
                    .find(|p| p.name == name)
                    .unwrap()
                    .dtype = dtype;
                assert!(
                    PtxGraph::from(tiles)
                        .indexed_loss_row_launch(out, 33)
                        .unwrap()
                        .is_none()
                );
            }
        }
    }
}

#[test]
fn forward_row_program_reuses_exact_targets_with_changed_feeds() {
    use crate::ptx::F32 as StorageF32;
    use crate::ptx::linear::program::{Program, binding};
    use crate::ptx::linear::{Binding, Buffer};
    use cudarc::driver::{CudaContext, DevicePtr};
    let device = match CudaContext::new(0) {
        Ok(v) => v,
        Err(error) => {
            assert_ne!(
                std::env::var("TNSR_REQUIRE_CUDA").as_deref(),
                Ok("1"),
                "{error}"
            );
            return;
        }
    };
    let stream = device.default_stream();
    let mut g = TensorGraph::<f32>::new();
    let x = g.graph.add_node(TensorGraphNode::Input {
        name: "x",
        shape: vec![3, 257],
    });
    let t = g.graph.add_node(TensorGraphNode::IndexSource {
        source: crate::tensor::IndexTensor::input("t", vec![3]),
    });
    let out = g
        .graph
        .add_node(TensorGraphNode::IndexedCrossEntropy { shape: vec![3] });
    g.graph.add_edge(x, out, 0);
    g.graph.add_edge(t, out, 1);
    let mut logits = Buffer::F32(stream.alloc_zeros::<f32>(3 * 257).unwrap());
    let mut targets = stream.memcpy_stod(&[0_u32, 256, 3]).unwrap();
    let output = Buffer::F32(stream.alloc_zeros::<f32>(3).unwrap());
    let program = Program::<StorageF32>::new_bound(
        &stream,
        g,
        HashMap::from([
            (x, binding(&logits, &stream)),
            (
                t,
                Binding {
                    pointer: targets.device_ptr(&stream).0,
                    len: 3,
                    dtype: DType::U32,
                },
            ),
        ]),
        Some(binding(&output, &stream)),
        crate::tile::MatMulPrecision::StrictF32,
    )
    .unwrap();
    assert_eq!(program.kernels.len(), 1);
    assert_eq!(program.kernels[0].config.grid_dim, (3, 1, 1));
    assert_eq!(program.kernels[0].config.block_dim, (32, 1, 1));
    for feed in 0..2 {
        let values: Vec<f32> = (0..3 * 257)
            .map(|i| ((i * 7 + feed * 13) % 257) as f32 / 64.0)
            .collect();
        let indices = if feed == 0 {
            [0_u32, 256, 3]
        } else {
            [256, 0, 42]
        };
        if let Buffer::F32(v) = &mut logits {
            stream.memcpy_htod(&values, v).unwrap();
        }
        stream.memcpy_htod(&indices, &mut targets).unwrap();
        for kernel in &program.kernels {
            kernel.run(&stream).unwrap();
        }
        stream.synchronize().unwrap();
        if let Buffer::F32(v) = &output {
            let actual = stream.memcpy_dtov(v).unwrap();
            for r in 0..3 {
                let row = &values[r * 257..(r + 1) * 257];
                let maximum = row
                    .iter()
                    .copied()
                    .map(f64::from)
                    .fold(f64::NEG_INFINITY, f64::max);
                let expected = maximum - f64::from(row[indices[r] as usize])
                    + row
                        .iter()
                        .map(|&v| (f64::from(v) - maximum).exp())
                        .sum::<f64>()
                        .ln();
                assert!((f64::from(actual[r]) - expected).abs() < 2e-5);
            }
        }
    }
}
