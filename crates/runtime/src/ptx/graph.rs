use super::Function;
use crate::tile::TileGraph;
use crate::tile::TileIR;
use petgraph::Graph;

pub struct PtxGraph {
    pub graph: Graph<Function, usize>,
}

impl From<TileGraph> for PtxGraph {
    fn from(tile_graph: TileGraph) -> Self {
        Self {
            graph: tile_graph
                .graph
                .map_owned(|_, func| func.into(), |_, weight| weight),
        }
    }
}

impl From<TileIR> for Function {
    fn from(tile_ir: TileIR) -> Self {
        let mut func = Function::new(&tile_ir.kernel_name);

        // Add parameters
        for param in &tile_ir.params {
            let ptx_type = dtype_to_ptx_type(param.dtype);
            func.params.push((param.name.clone(), ptx_type));
        }

        // Allocate shared memory if needed
        if tile_ir.shared_mem_bytes > 0 {
            func.add_shared_memory("shared", tile_ir.shared_mem_bytes);
        }

        // Convert body statements to PTX instructions
        lower_block(&mut func, &tile_ir.body);

        // Add return instruction
        func.add_inst(super::instructions::Inst::Ret);

        func
    }
}

fn dtype_to_ptx_type(dtype: crate::tile::DType) -> super::types::Type {
    match dtype {
        crate::tile::DType::F16 => super::types::Type::F16,
        crate::tile::DType::BF16 => super::types::Type::BF16,
        crate::tile::DType::F32 => super::types::Type::F32,
    }
}

fn lower_block(func: &mut Function, block: &crate::tile::Block) {
    for stmt in &block.stmts {
        lower_stmt(func, stmt);
    }
}

fn lower_stmt(func: &mut Function, stmt: &crate::tile::Stmt) {
    use crate::tile::Stmt;

    match stmt {
        Stmt::AllocTile { .. } => {
            // Register/shared allocations are tracked but don't emit instructions
            // This would be handled by the Function's register allocation
            // TODO: Integrate allocator here
        }
        Stmt::Load { .. } => {
            // TODO: Implement load from global memory
        }
        Stmt::Store { .. } => {
            // TODO: Implement store to global memory
        }
        Stmt::LoadSharedToReg { .. } => {
            // TODO: Implement load from shared to register
        }
        Stmt::Zero { .. } => {
            // TODO: Implement zero initialization
        }
        Stmt::MatMul { .. } => {
            // TODO: Implement matrix multiply
        }
        Stmt::Add { .. } => {
            // TODO: Implement add
        }
        Stmt::Sub { .. } => {
            // TODO: Implement sub
        }
        Stmt::Mul { .. } => {
            // TODO: Implement mul
        }
        Stmt::Div { .. } => {
            // TODO: Implement div
        }
        Stmt::Neg { .. } => {
            // TODO: Implement neg
        }
        Stmt::Exp { .. } => {
            // TODO: Implement exp
        }
        Stmt::Log { .. } => {
            // TODO: Implement log
        }
        Stmt::Relu { .. } => {
            // TODO: Implement relu
        }
        Stmt::Transpose { .. } => {
            // TODO: Implement transpose
        }
        Stmt::BroadcastAxis { .. } => {
            // TODO: Implement broadcast
        }
        Stmt::ReduceAxis { .. } => {
            // TODO: Implement reduce
        }
        Stmt::Gt { .. } => {
            // TODO: Implement greater than
        }
        Stmt::Mask { .. } => {
            // TODO: Implement mask
        }
        Stmt::Barrier => {
            func.add_inst(super::instructions::Inst::BarSync { barrier_id: 0 });
        }
        Stmt::ForLoop {
            loop_var: _,
            start: _,
            end: _,
            body,
        } => {
            // TODO: Implement for loop with labels and branches
            lower_block(func, body);
        }
    }
}
