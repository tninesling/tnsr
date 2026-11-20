//! Exact full-tile BF16 transport only. No arithmetic or scheduling policy.
use super::*;
use crate::tile::ir::MatrixLayout;

pub(in crate::ptx::mixed::attention_training_online) struct Stage<'a> {
    pub prefix: &'a str,
    pub source: &'a str,
    pub row_base: Expr,
    pub dest: TileVar,
    pub column_major: bool,
}

pub(in crate::ptx::mixed::attention_training_online) fn stage_with_transport(
    b: &mut ScalarBuilder,
    shape: Shape,
    layout: ReadLayout,
    copy: Stage<'_>,
    transport: Transport,
) -> Result<()> {
    if transport == Transport::Scalar {
        // Exactly the old builder calls, including loop names and variable IDs.
        return stage(
            b,
            copy.prefix,
            copy.source,
            shape,
            layout,
            copy.row_base,
            copy.dest,
        );
    }
    let name = b.loop_name(&format!("{}_async16", copy.prefix));
    // 512 groups ×8 BF16 elements =4096 elements. All256 threads issue exactly
    // two complete16B groups; no disabled/partial copy or zero-fill is needed.
    b.fold(name, ix(TILE * TILE / (THREADS * 8)), vec![], |b, round| {
        let group = round * THREADS + Expr::ThreadIdx(Dim::Y) * 32usize + Expr::ThreadIdx(Dim::X);
        let element = group * 8usize;
        let local_row = div(element.clone(), TILE);
        let feature = rem(element, TILE);
        let source_row = copy.row_base.clone() + local_row.clone();
        // Logical column-major K/V views still hold the physical sequence row
        // ×64+feature byte order. Swap coordinates, not the global source map.
        let (tile_row, tile_col) = if copy.column_major {
            (feature.clone(), local_row)
        } else {
            (local_row, feature.clone())
        };
        b.stmts.push(Stmt::AsyncCopy {
            dest: copy.dest,
            src_param: copy.source.into(),
            element_index: input_index(shape, layout, source_row.clone(), feature.clone()),
            row: source_row,
            col: feature,
            tile_row,
            tile_col,
            layout: MatrixLayout {
                rows: SEQUENCE,
                cols: TILE,
                row_stride: if layout == ReadLayout::SequenceMajor {
                    shape.heads * TILE
                } else {
                    TILE
                },
            },
            copy_bytes: 16,
        });
        Ok(())
    })
}

/// The caller's existing publication barrier remains immediately after this.
/// Backward groups all four operand tiles in one commit, then waits for all.
pub(in crate::ptx::mixed::attention_training_online) fn finish_staging(
    b: &mut ScalarBuilder,
    transport: Transport,
) {
    if transport == Transport::Async16 {
        b.stmts.push(Stmt::AsyncCommit);
        b.stmts.push(Stmt::AsyncWait);
    }
}
