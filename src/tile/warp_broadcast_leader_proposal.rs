//! UNDECLARED IR-hook proposal. Do not declare before coordinated integration.
//!
//! Required existing-file changes, deliberately NOT made by this preparation:
//! 1. Add Stmt::WarpBroadcastLeader {dest:TileVar,src:TileVar} to tile/ir.rs.
//!    It copies lane0 f32 register BITS to all32 active lanes. No arithmetic,
//!    identity term, reduction, normalization, shared memory or block barrier.
//! 2. Declare this builder implementation (or move it into scalar.rs).
//! 3. In tile/layout.rs validate src as defined f32 scalar and define dest as
//!    f32 scalar, same variable rules as WarpReduce. Reject undefined sources.
//! 4. Route the new statement through graph.rs scalar dispatch and lower it
//!    with the separate PTX proposal. Existing WarpReduce broadcast=true stays
//!    the existing butterfly; broadcast=false stays canonical down16..1.
//! 5. Audit index visitors/IR formatters/matches and test interpreters for new
//!    statement exhaustiveness; variable dependencies are src read/dest write.
//!    There are no index expressions to rewrite. Bump the coordinated compiled
//!    signature only with actual opt-in integration, not for undeclared files.
//!
//! Contract: whole active32-lane warp, full0xffffffff member mask. Builders
//! must not put it inside divergent lane predicates or after partial-warp
//! returns. Initial warp-attention geometry uses exact (32,8,1) blocks, S512,
//! BH>=64 and rowcount divisible8; no inactive row or key tails exist.
use super::{ScalarBuilder,Stmt,TileVar};

impl ScalarBuilder {
    /// Bitwise lane0 broadcast over one full active warp. This is NOT an
    /// all-reduction; only the source in lane0 has semantic significance.
    pub fn warp_broadcast_leader(&mut self,src:TileVar)->TileVar {
        let dest=self.var();
        self.stmts.push(Stmt::WarpBroadcastLeader {dest,src});
        dest
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tile::{Block,ScalarExpr,TileIR};
    #[test]
    fn leader_broadcast_requires_a_defined_f32_source() {
        let mut builder=ScalarBuilder::default();
        let undefined=builder.var();builder.warp_broadcast_leader(undefined);
        let invalid=TileIR {kernel_name:"undefined_leader".into(),params:Vec::new(),body:Block {stmts:builder.stmts},shared_mem_bytes:0};
        assert!(invalid.validate_layouts().is_err());
        let mut builder=ScalarBuilder::default();let src=builder.value(ScalarExpr::Constant(-0.));
        let dest=builder.warp_broadcast_leader(src);assert_ne!(src,dest);
        assert!(matches!(builder.stmts.last(),Some(Stmt::WarpBroadcastLeader {src:s,dest:d}) if *s==src&&*d==dest));
        let valid=TileIR {kernel_name:"defined_leader".into(),params:Vec::new(),body:Block {stmts:builder.stmts},shared_mem_bytes:0};
        valid.validate_layouts().unwrap();
    }
}
