//! Bound spatial specialization without changing channel/tap accumulation order.
use super::*;
use crate::tile::Conv2dGeometry;

const MAX_SPECIALIZED_AREA: usize = 121;

#[derive(Clone, Copy)]
enum Direction {
    Forward,
    InputGradient,
}

struct Position<'a> {
    row: Operand<'a, U64>,
    col: Operand<'a, U64>,
}

struct Planes<'a> {
    source: Operand<'a, U64>,
    weight: Operand<'a, U64>,
}

pub(super) fn lower_conv2d<'a>(
    func: &mut Function<'a>,
    ctx: &mut LoweringContext<'a>,
    geometry: Conv2dGeometry,
) {
    if geometry.kernel_height.saturating_mul(geometry.kernel_width) > MAX_SPECIALIZED_AREA {
        super::lower_conv2d_serial(func, ctx, geometry);
    } else {
        lower(func, ctx, geometry, Direction::Forward);
    }
}

pub(super) fn lower_conv_transpose2d<'a>(
    func: &mut Function<'a>,
    ctx: &mut LoweringContext<'a>,
    geometry: Conv2dGeometry,
) {
    if geometry.kernel_height.saturating_mul(geometry.kernel_width) > MAX_SPECIALIZED_AREA {
        super::lower_conv_transpose2d_serial(func, ctx, geometry);
    } else {
        lower(func, ctx, geometry, Direction::InputGradient);
    }
}

fn add<'a>(
    func: &mut Function<'a>,
    left: Operand<'a, U64>,
    right: Operand<'a, U64>,
) -> Operand<'a, U64> {
    let result = func.add_u64_register();
    func.add_inst(Inst::add_u64(result.clone(), left, right));
    result
}

fn lower<'a>(
    func: &mut Function<'a>,
    ctx: &mut LoweringContext<'a>,
    g: Conv2dGeometry,
    direction: Direction,
) {
    let tid = global_linear_tid(func);
    let (channels, height, width, source_channels, source_height, source_width) = match direction {
        Direction::Forward => (
            g.output_channels,
            g.output_height,
            g.output_width,
            g.input_channels,
            g.input_height,
            g.input_width,
        ),
        Direction::InputGradient => (
            g.input_channels,
            g.input_height,
            g.input_width,
            g.output_channels,
            g.output_height,
            g.output_width,
        ),
    };
    let (batch, channel, row, col) = decode_nchw(func, tid.clone(), channels, height, width);
    let position = Position { row, col };
    let source_batch = multiply_u64(func, batch, source_channels * source_height * source_width);
    let area = g.kernel_height * g.kernel_width;
    let fixed_weight = multiply_u64(
        func,
        channel,
        match direction {
            Direction::Forward => g.input_channels * area,
            Direction::InputGradient => area,
        },
    );
    let accumulator = func.add_f32_register();
    func.add_inst(Inst::mov_f32(accumulator.clone(), Operand::imm_f32(0.0)));
    // Channel, kernel row, then kernel column retain the original sum order.
    // Hoist plane offsets rather than decoding a flattened reduction per tap.
    let channel_loop = begin_counted_loop(func, ctx, "conv_channel", source_channels);
    let source_channel = multiply_u64(
        func,
        channel_loop.counter.clone(),
        source_height * source_width,
    );
    let weight_channel = multiply_u64(
        func,
        channel_loop.counter.clone(),
        match direction {
            Direction::Forward => area,
            Direction::InputGradient => g.input_channels * area,
        },
    );
    let planes = Planes {
        source: add(func, source_batch, source_channel),
        weight: add(func, fixed_weight, weight_channel),
    };
    // Fixed loop bounds keep generated PTX small; padding skips still avoid
    // loads/products entirely, including invalid taps containing NaN or Inf.
    let row_loop = begin_counted_loop(func, ctx, "conv_row", g.kernel_height);
    let next_row = next_label(ctx, "conv_row_continue");
    let source_row = coordinate(
        func,
        &g,
        direction,
        position.row.clone(),
        row_loop.counter.clone(),
        true,
        next_row,
    );
    let row_offset = multiply_u64(func, source_row, source_width);
    let source_row = add(func, planes.source.clone(), row_offset);
    let weight_row_offset = multiply_u64(func, row_loop.counter.clone(), g.kernel_width);
    let weight_row = add(func, planes.weight.clone(), weight_row_offset);
    let col_loop = begin_counted_loop(func, ctx, "conv_col", g.kernel_width);
    emit_tap(
        func,
        ctx,
        &g,
        direction,
        &position,
        &Planes {
            source: source_row,
            weight: weight_row,
        },
        col_loop.counter.clone(),
        accumulator.clone(),
    );
    end_counted_loop(func, col_loop);
    func.add_inst(Inst::Label(next_row));
    end_counted_loop(func, row_loop);

    end_counted_loop(func, channel_loop);
    store_param_f32_at(func, ctx, "output", tid, accumulator);
}

fn coordinate<'a>(
    func: &mut Function<'a>,
    g: &Conv2dGeometry,
    direction: Direction,
    position: Operand<'a, U64>,
    tap: Operand<'a, U64>,
    row: bool,
    skip: &'a str,
) -> Operand<'a, U64> {
    match direction {
        Direction::Forward => padded_input_coordinate(
            func,
            position,
            tap,
            g.stride,
            g.padding,
            if row { g.input_height } else { g.input_width },
            skip,
        ),
        Direction::InputGradient => transposed_output_coordinate(
            func,
            position,
            tap,
            g.stride,
            g.padding,
            if row { g.output_height } else { g.output_width },
            skip,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_tap<'a>(
    func: &mut Function<'a>,
    ctx: &mut LoweringContext<'a>,
    g: &Conv2dGeometry,
    direction: Direction,
    position: &Position<'a>,
    planes: &Planes<'a>,
    kx: Operand<'a, U64>,
    accumulator: Operand<'a, F32>,
) {
    let next = next_label(ctx, "conv_tap_continue");
    let col = coordinate(
        func,
        g,
        direction,
        position.col.clone(),
        kx.clone(),
        false,
        next,
    );
    let source = add(func, planes.source.clone(), col);
    let weight = add(func, planes.weight.clone(), kx);
    let input = load_param_f32_at(
        func,
        ctx,
        match direction {
            Direction::Forward => "input",
            Direction::InputGradient => "grad_output",
        },
        source,
    );
    let weight = load_param_f32_at(func, ctx, "weight", weight);
    accumulate_product(func, accumulator, input, weight);
    func.add_inst(Inst::Label(next));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{TensorGraph, TensorGraphNode};
    use crate::tile::TileGraph;
    #[test]
    fn bounded_spatial_specialization_keeps_large_kernels_serial() {
        for (kh, kw, specialized) in [
            (1, 1, true),
            (3, 2, true),
            (5, 5, true),
            (2, 13, true),
            (11, 11, true),
            (15, 15, false),
        ] {
            for transpose in [false, true] {
                let mut graph = TensorGraph::<f32>::new();
                let input = graph.graph.add_node(TensorGraphNode::Input {
                    name: "x",
                    shape: if transpose {
                        vec![1, 5, 41 - kh, 41 - kw]
                    } else {
                        vec![1, 3, 40, 40]
                    },
                });
                let weight = graph.graph.add_node(TensorGraphNode::Input {
                    name: "w",
                    shape: vec![5, 3, kh, kw],
                });
                let output = graph.graph.add_node(if transpose {
                    TensorGraphNode::ConvTranspose2d {
                        shape: vec![1, 3, 40, 40],
                        stride: 1,
                        padding: 0,
                    }
                } else {
                    TensorGraphNode::Conv2d {
                        shape: vec![1, 5, 41 - kh, 41 - kw],
                        stride: 1,
                        padding: 0,
                    }
                });
                graph.graph.add_edge(input, output, 0);
                graph.graph.add_edge(weight, output, 1);
                let text = PtxGraph::from(TileGraph::from_graph(&graph)).to_ptx();
                assert_eq!(
                    text.contains("conv_channel_loop"),
                    specialized,
                    "kh={kh} kw={kw} transpose={transpose}"
                );
                assert_eq!(
                    text.matches("mul.f32").count(),
                    1,
                    "bounded emitted code size"
                );
                assert!(!text.contains("atom."));
                assert!(!text.contains("bar.sync"));
            }
        }
    }
}

#[cfg(test)]
mod binding_tests;
