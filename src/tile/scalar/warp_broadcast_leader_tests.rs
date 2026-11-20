use super::*;
use crate::tile::{Block, Stmt, TileIR};
fn ir(builder: ScalarBuilder) -> TileIR {
    TileIR {
        kernel_name: "leader_broadcast_validation".into(),
        params: Vec::new(),
        body: Block {
            stmts: builder.stmts,
        },
        shared_mem_bytes: 0,
    }
}
#[test]
fn warp_broadcast_leader_requires_defined_f32_and_preserves_distinct_destination() {
    let mut builder = ScalarBuilder::default();
    let missing = builder.var();
    builder.warp_broadcast_leader(missing);
    assert!(ir(builder).validate_layouts().is_err());
    let mut builder = ScalarBuilder::default();
    let src = builder.value(ScalarExpr::Constant(-0.));
    let dest = builder.warp_broadcast_leader(src);
    assert_ne!(src, dest);
    assert!(
        matches!(builder.stmts.last(),Some(Stmt::WarpBroadcastLeader {src:s,dest:d}) if *s==src&&*d==dest)
    );
    ir(builder).validate_layouts().unwrap();
}
