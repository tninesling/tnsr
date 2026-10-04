#![cfg(feature = "cuda")]
use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use tnsr::graph::TensorGraph;
use tnsr::ptx::PtxGraph;
use tnsr::tensor::{BinaryOp, TensorExpr};
use tnsr::tile::*;

#[test]
fn generic_scalar_folds_preserve_nested_updates_and_empty_identities() {
    let device = match CudaContext::new(0) {
        Ok(device) => device,
        Err(error) if std::env::var("TNSR_REQUIRE_CUDA").as_deref() != Ok("1") => {
            eprintln!("skipping CUDA: {error:#}");
            return;
        }
        Err(error) => panic!("{error:#}"),
    };
    let stream = device.default_stream();
    let input = stream
        .memcpy_stod(&[1.0f32, 2., 3., 4., 5., 2., 3., 4., 5., 6.])
        .unwrap();
    for count in [0, 3, 5] {
        let mut builder = ScalarBuilder::default();
        let sum = builder.var();
        let product = builder.var();
        builder
            .fold(
                "column".into(),
                Expr::Const(count),
                vec![
                    LoopCarry {
                        var: sum,
                        initial: ScalarExpr::Constant(0.0),
                    },
                    LoopCarry {
                        var: product,
                        initial: ScalarExpr::Constant(1.0),
                    },
                ],
                |builder, column| {
                    let value = builder.load(
                        ScalarMemory::Global("input".into()),
                        Expr::BlockIdx(Dim::X) * 5usize + column.clone(),
                    );
                    builder.set(
                        product,
                        ScalarExpr::binary(
                            ScalarBinaryOp::Arithmetic(BinaryOp::Mul),
                            product,
                            value,
                        ),
                    );
                    builder.when(
                        ScalarPredicate::IndexLt {
                            lhs: column,
                            rhs: Expr::Const(3),
                        },
                        |builder| {
                            builder.fold(
                                "repeat".into(),
                                Expr::Const(2),
                                Vec::new(),
                                |builder, _| {
                                    builder.set(
                                        sum,
                                        ScalarExpr::binary(
                                            ScalarBinaryOp::Arithmetic(BinaryOp::Add),
                                            sum,
                                            value,
                                        ),
                                    );
                                    Ok(())
                                },
                            )
                        },
                    )
                },
            )
            .unwrap();
        builder.store(
            ScalarMemory::Global("output".into()),
            Expr::BlockIdx(Dim::X) * 2usize,
            sum,
        );
        builder.store(
            ScalarMemory::Global("output".into()),
            Expr::BlockIdx(Dim::X) * 2usize + Expr::Const(1),
            product,
        );
        let mut ir = TileIR {
            kernel_name: "recurrence".into(),
            params: vec![
                KernelParam {
                    name: "input".into(),
                    dtype: DType::F32,
                    is_input: true,
                },
                KernelParam {
                    name: "output".into(),
                    dtype: DType::F32,
                    is_input: false,
                },
            ],
            body: Block {
                stmts: builder.stmts,
            },
            shared_mem_bytes: 0,
        };
        ir.validate_layouts().unwrap();
        ir.optimize_indices().unwrap();
        ir.validate_layouts().unwrap();
        let graph: TensorGraph<f32> = TensorExpr::constant(vec![0.0], vec![1]).into();
        let mut tile_graph = TileGraph::from(graph);
        tile_graph.graph.clear();
        tile_graph.graph.add_node(ir);
        let source = PtxGraph::from(tile_graph).to_ptx();
        let module = device.load_module(Ptx::from_src(source)).unwrap();
        let function = module.load_function("recurrence_0").unwrap();
        let mut output = stream.alloc_zeros::<f32>(4).unwrap();
        // SAFETY: Two blocks each read their five-element row and write two
        // outputs; buffers remain live on the same stream through readback.
        unsafe {
            stream
                .launch_builder(&function)
                .arg(&input)
                .arg(&mut output)
                .launch(LaunchConfig {
                    grid_dim: (2, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .unwrap();
        let actual = stream.memcpy_dtov(&output).unwrap();
        let expected = match count {
            0 => [0., 1., 0., 1.],
            3 => [12., 6., 18., 24.],
            _ => [12., 120., 18., 720.],
        };
        assert_eq!(actual, expected);
    }
}

#[test]
fn counted_loops_keep_wide_indices_and_shadowed_bounds_in_the_parent_scope() {
    let device = match CudaContext::new(0) {
        Ok(device) => device,
        Err(error) if std::env::var("TNSR_REQUIRE_CUDA").as_deref() != Ok("1") => {
            eprintln!("skipping CUDA: {error:#}");
            return;
        }
        Err(error) => panic!("{error:#}"),
    };
    let mut builder = ScalarBuilder::default();
    let sum = builder.var();
    let start = i64::from(i32::MAX) + 7;
    builder
        .fold(
            "column".into(),
            Expr::Const(start + 3),
            vec![LoopCarry {
                var: sum,
                initial: ScalarExpr::Constant(0.0),
            }],
            |builder, column| {
                let offset = Expr::BlockIdx(Dim::X) * 5usize
                    + Expr::Sub(Box::new(column.clone()), Box::new(Expr::Const(start)));
                let value = builder.load(ScalarMemory::Global("input".into()), offset.clone());
                let body = builder.block(|builder| {
                    builder.set(
                        sum,
                        ScalarExpr::binary(ScalarBinaryOp::Arithmetic(BinaryOp::Add), sum, value),
                    );
                    Ok(())
                })?;
                builder.stmts.push(Stmt::ForLoop {
                    loop_var: "column".into(),
                    start: column.clone() + Expr::Const(1),
                    end: column + Expr::Const(3),
                    carries: Vec::new(),
                    body,
                });
                // This access must see the restored outer counter after the inner loop.
                let value = builder.load(ScalarMemory::Global("input".into()), offset);
                builder.set(
                    sum,
                    ScalarExpr::binary(ScalarBinaryOp::Arithmetic(BinaryOp::Add), sum, value),
                );
                Ok(())
            },
        )
        .unwrap();
    let Stmt::ForLoop {
        start: loop_start, ..
    } = &mut builder.stmts[0]
    else {
        panic!("expected outer loop")
    };
    *loop_start = Expr::Const(start);
    builder.store(
        ScalarMemory::Global("output".into()),
        Expr::BlockIdx(Dim::X),
        sum,
    );
    let mut ir = TileIR {
        kernel_name: "wide_shadow".into(),
        params: vec![
            KernelParam {
                name: "input".into(),
                dtype: DType::F32,
                is_input: true,
            },
            KernelParam {
                name: "output".into(),
                dtype: DType::F32,
                is_input: false,
            },
        ],
        body: Block {
            stmts: builder.stmts,
        },
        shared_mem_bytes: 0,
    };
    ir.validate_layouts().unwrap();
    ir.optimize_indices().unwrap();
    let graph: TensorGraph<f32> = TensorExpr::constant(vec![0.0], vec![1]).into();
    let mut tile_graph = TileGraph::from(graph);
    tile_graph.graph.clear();
    tile_graph.graph.add_node(ir);
    let module = device
        .load_module(Ptx::from_src(PtxGraph::from(tile_graph).to_ptx()))
        .unwrap();
    let function = module.load_function("wide_shadow_0").unwrap();
    let stream = device.default_stream();
    let input = stream
        .memcpy_stod(&[1.0f32, 2., 3., 4., 5., 2., 3., 4., 5., 6.])
        .unwrap();
    let mut output = stream.alloc_zeros::<f32>(2).unwrap();
    // SAFETY: Each block reads its five-element row and writes one output;
    // the nested loop reads only registers, and buffers stay live through readback.
    unsafe {
        stream
            .launch_builder(&function)
            .arg(&input)
            .arg(&mut output)
            .launch(LaunchConfig {
                grid_dim: (2, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            })
    }
    .unwrap();
    assert_eq!(stream.memcpy_dtov(&output).unwrap(), vec![18., 27.]);
}
