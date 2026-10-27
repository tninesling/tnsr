//! Hoist serial weight-gradient address decoding without reassociating sums.
use super::*;

pub(in crate::ptx::graph) fn lower_weight_gradient<'a>(
    func: &mut Function<'a>,
    ctx: &mut LoweringContext<'a>,
    g: Conv2dGeometry,
) {
    let tid = global_linear_tid(func);
    let (output_channel, input_channel, kernel_row, kernel_col) = decode_nchw(
        func,
        tid.clone(),
        g.input_channels,
        g.kernel_height,
        g.kernel_width,
    );
    let source_channel = multiply_u64(func, input_channel, g.input_height * g.input_width);
    let grad_channel = multiply_u64(func, output_channel, g.output_height * g.output_width);
    let accumulator = func.add_f32_register();
    func.add_inst(Inst::mov_f32(accumulator.clone(), Operand::imm_f32(0.0)));

    // The original flattened reduction visits n -> oy -> ox. These counted
    // loops keep that order and the same accumulator, products and stores.
    let batch = begin_counted_loop(func, ctx, "conv_weight_batch", g.batch);
    let source_batch = multiply_u64(
        func,
        batch.counter.clone(),
        g.input_channels * g.input_height * g.input_width,
    );
    let grad_batch = multiply_u64(
        func,
        batch.counter.clone(),
        g.output_channels * g.output_height * g.output_width,
    );
    let source_plane = add(func, source_batch, source_channel);
    let grad_plane = add(func, grad_batch, grad_channel);
    let row = begin_counted_loop(func, ctx, "conv_weight_row", g.output_height);
    let next_row = next_label(ctx, "conv_weight_row_continue");
    // Invalid padding skips all products in the row, including NaN/Inf taps.
    let input_row = padded_input_coordinate(
        func,
        row.counter.clone(),
        kernel_row,
        g.stride,
        g.padding,
        g.input_height,
        next_row,
    );
    let source_row_offset = multiply_u64(func, input_row, g.input_width);
    let source_row = add(func, source_plane, source_row_offset);
    let grad_row_offset = multiply_u64(func, row.counter.clone(), g.output_width);
    let grad_row = add(func, grad_plane, grad_row_offset);
    let col = begin_counted_loop(func, ctx, "conv_weight_col", g.output_width);
    let next_col = next_label(ctx, "conv_weight_col_continue");
    let input_col = padded_input_coordinate(
        func,
        col.counter.clone(),
        kernel_col,
        g.stride,
        g.padding,
        g.input_width,
        next_col,
    );
    let source_offset = add(func, source_row, input_col);
    let grad_offset = add(func, grad_row, col.counter.clone());
    let input = load_param_f32_at(func, ctx, "input", source_offset);
    let grad = load_param_f32_at(func, ctx, "grad_output", grad_offset);
    accumulate_product(func, accumulator.clone(), input, grad);
    func.add_inst(Inst::Label(next_col));
    end_counted_loop(func, col);
    func.add_inst(Inst::Label(next_row));
    end_counted_loop(func, row);
    end_counted_loop(func, batch);
    store_param_f32_at(func, ctx, "output", tid, accumulator);
}
