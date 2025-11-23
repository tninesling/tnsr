use std::collections::HashMap;

use super::Function;
use super::instructions::{Inst, Operand};
use super::types::{F32, I32, U64};
use crate::tile::{Expr, TileGraph, TileIR, TileVar};
use petgraph::Graph;

pub struct PtxGraph {
    pub graph: Graph<Function, usize>,
}

impl From<TileGraph> for PtxGraph {
    fn from(tile_graph: TileGraph) -> Self {
        Self {
            graph: tile_graph.graph.map_owned(
                |node_idx, tile_ir| {
                    // Convert TileIR to Function
                    let mut func: Function = tile_ir.into();
                    // Make the kernel name unique by appending the node index
                    func.name = format!("{}_{}", func.name, node_idx.index());
                    func
                },
                |_, weight| weight,
            ),
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

        // Create lowering context to track tile variable mappings
        let mut ctx = LoweringContext::new();

        // Convert body statements to PTX instructions
        lower_block(&mut func, &mut ctx, &tile_ir.body);

        // Add return instruction
        func.add_inst(super::instructions::Inst::Ret);

        func
    }
}

struct LoweringContext {
    /// Maps TileVar to PTX register operands (for scalar/simple cases)
    tile_to_reg: HashMap<TileVar, Operand<F32>>,
    /// Tracks allocated shared memory base pointers
    shared_mem_ptrs: HashMap<TileVar, Operand<U64>>,
}

impl LoweringContext {
    fn new() -> Self {
        Self {
            tile_to_reg: HashMap::new(),
            shared_mem_ptrs: HashMap::new(),
        }
    }

    fn get_or_alloc_reg(&mut self, func: &mut Function, var: TileVar) -> Operand<F32> {
        if let Some(reg) = self.tile_to_reg.get(&var) {
            reg.clone()
        } else {
            let reg = func.add_f32_register();
            self.tile_to_reg.insert(var, reg.clone());
            reg
        }
    }
}

fn dtype_to_ptx_type(dtype: crate::tile::DType) -> super::types::Type {
    match dtype {
        crate::tile::DType::F16 => super::types::Type::F16,
        crate::tile::DType::BF16 => super::types::Type::BF16,
        crate::tile::DType::F32 => super::types::Type::F32,
    }
}

fn lower_block(func: &mut Function, ctx: &mut LoweringContext, block: &crate::tile::Block) {
    for stmt in &block.stmts {
        lower_stmt(func, ctx, stmt);
    }
}

fn lower_stmt(func: &mut Function, ctx: &mut LoweringContext, stmt: &crate::tile::Stmt) {
    use crate::tile::Stmt;

    match stmt {
        Stmt::AllocTile {
            var,
            space,
            dtype: _,
            rows: _,
            cols: _,
        } => {
            use crate::tile::MemorySpace;
            match space {
                MemorySpace::Register => {
                    // Allocate register for this tile variable
                    ctx.get_or_alloc_reg(func, *var);
                }
                MemorySpace::Shared => {
                    // Track shared memory allocation
                    // In a real implementation, we'd track offsets into the shared memory buffer
                    let ptr = func.add_u64_register();
                    ctx.shared_mem_ptrs.insert(*var, ptr);
                }
                MemorySpace::Global => {
                    // Global memory is parameter-based, not allocated
                }
            }
        }
        Stmt::Load {
            dest,
            src_param,
            row_offset,
            col_offset: _,
        } => {
            // Load from global memory to register/shared
            // Simplified: load a single f32 value
            let dest_reg = ctx.get_or_alloc_reg(func, *dest);

            // Get parameter pointer
            let param_ptr = func.add_global_ptr_param(src_param);

            // Calculate offset from row/col
            let offset_reg = lower_expr(func, row_offset);

            // Add offset to base pointer
            let addr = func.add_u64_register();
            func.add_inst(Inst::AddU64(super::instructions::AddInst::new(
                addr.clone(),
                param_ptr,
                offset_reg,
            )));

            // Load from global memory
            func.add_inst(Inst::load_global_scalar_f32(dest_reg, addr));
        }
        Stmt::Store {
            dest_param,
            src,
            row_offset,
            col_offset: _,
        } => {
            // Store from register/shared to global memory
            let src_reg = ctx.get_or_alloc_reg(func, *src);

            // Get parameter pointer (would need to be pre-loaded)
            let param_ptr = func.add_global_ptr_param(dest_param);

            // Calculate offset
            let offset_reg = lower_expr(func, row_offset);

            // Add offset to base pointer
            let addr = func.add_u64_register();
            func.add_inst(Inst::AddU64(super::instructions::AddInst::new(
                addr.clone(),
                param_ptr,
                offset_reg,
            )));

            // Store to global memory
            func.add_inst(Inst::store_global_scalar_f32(addr, src_reg));
        }
        Stmt::LoadSharedToReg { dest, src } => {
            // Load from shared memory to register
            let dest_reg = ctx.get_or_alloc_reg(func, *dest);

            if let Some(src_ptr) = ctx.shared_mem_ptrs.get(src) {
                func.add_inst(Inst::load_shared_scalar_f32(dest_reg, src_ptr.clone()));
            }
        }
        Stmt::Zero { tile } => {
            // Zero out a tile
            let reg = ctx.get_or_alloc_reg(func, *tile);
            let zero = Operand::imm_f32(0.0);
            func.add_inst(Inst::mov_f32(reg, zero));
        }
        Stmt::MatMul {
            dest,
            a,
            b,
            layout: _,
        } => {
            // Matrix multiply using tensor cores
            // This is a placeholder - real implementation would use WMMA or MMA instructions
            let dest_reg = ctx.get_or_alloc_reg(func, *dest);
            let a_reg = ctx.get_or_alloc_reg(func, *a);
            let b_reg = ctx.get_or_alloc_reg(func, *b);

            // Placeholder: just multiply and add (not a real matmul)
            let temp = func.add_f32_register();
            func.add_inst(Inst::mul_f32(temp.clone(), a_reg, b_reg));
            func.add_inst(Inst::add_f32(dest_reg.clone(), dest_reg, temp));
        }
        Stmt::Add { dest, a, b } => {
            let dest_reg = ctx.get_or_alloc_reg(func, *dest);
            let a_reg = ctx.get_or_alloc_reg(func, *a);
            let b_reg = ctx.get_or_alloc_reg(func, *b);
            func.add_inst(Inst::add_f32(dest_reg, a_reg, b_reg));
        }
        Stmt::Sub { dest, a, b } => {
            let dest_reg = ctx.get_or_alloc_reg(func, *dest);
            let a_reg = ctx.get_or_alloc_reg(func, *a);
            let b_reg = ctx.get_or_alloc_reg(func, *b);
            func.add_inst(Inst::sub_f32(dest_reg, a_reg, b_reg));
        }
        Stmt::Mul { dest, a, b } => {
            let dest_reg = ctx.get_or_alloc_reg(func, *dest);
            let a_reg = ctx.get_or_alloc_reg(func, *a);
            let b_reg = ctx.get_or_alloc_reg(func, *b);
            func.add_inst(Inst::mul_f32(dest_reg, a_reg, b_reg));
        }
        Stmt::Div { dest, a, b } => {
            let dest_reg = ctx.get_or_alloc_reg(func, *dest);
            let a_reg = ctx.get_or_alloc_reg(func, *a);
            let b_reg = ctx.get_or_alloc_reg(func, *b);
            func.add_inst(Inst::div_f32(dest_reg, a_reg, b_reg));
        }
        Stmt::Neg { dest, src } => {
            let dest_reg = ctx.get_or_alloc_reg(func, *dest);
            let src_reg = ctx.get_or_alloc_reg(func, *src);
            func.add_inst(Inst::neg_f32(dest_reg, src_reg));
        }
        Stmt::Exp { dest, src } => {
            let dest_reg = ctx.get_or_alloc_reg(func, *dest);
            let src_reg = ctx.get_or_alloc_reg(func, *src);

            // exp(x) = 2^(x * log2(e))
            // First compute x * log2(e)
            let log2e = Operand::imm_f32(std::f32::consts::LOG2_E);
            let scaled = func.add_f32_register();
            func.add_inst(Inst::mul_f32(scaled.clone(), src_reg, log2e));

            // Then compute 2^(scaled)
            func.add_inst(Inst::ex2_f32(dest_reg, scaled));
        }
        Stmt::Log { dest, src } => {
            let dest_reg = ctx.get_or_alloc_reg(func, *dest);
            let src_reg = ctx.get_or_alloc_reg(func, *src);

            // log(x) = log2(x) * ln(2)
            // First compute log2(x)
            let log2_result = func.add_f32_register();
            func.add_inst(Inst::lg2_f32(log2_result.clone(), src_reg));

            // Then multiply by ln(2)
            let ln2 = Operand::imm_f32(std::f32::consts::LN_2);
            func.add_inst(Inst::mul_f32(dest_reg, log2_result, ln2));
        }
        Stmt::Relu { dest, src } => {
            let dest_reg = ctx.get_or_alloc_reg(func, *dest);
            let src_reg = ctx.get_or_alloc_reg(func, *src);

            // relu(x) = max(0, x)
            let zero = Operand::imm_f32(0.0);
            func.add_inst(Inst::max_f32(dest_reg, src_reg, zero));
        }
        Stmt::Transpose { dest: _, src: _ } => {
            // TODO: Implement transpose
            // This requires more complex addressing logic
        }
        Stmt::BroadcastAxis {
            dest: _,
            src: _,
            axis: _,
        } => {
            // TODO: Implement broadcast
            // This requires loop unrolling or index manipulation
        }
        Stmt::ReduceAxis {
            dest: _,
            src: _,
            op: _,
            axis: _,
        } => {
            // TODO: Implement reduce
            // This requires accumulation loops and potentially warp-level primitives
        }
        Stmt::Gt { dest, a, b } => {
            let dest_reg = ctx.get_or_alloc_reg(func, *dest);
            let a_reg = ctx.get_or_alloc_reg(func, *a);
            let b_reg = ctx.get_or_alloc_reg(func, *b);

            // Compare a > b, store result as 1.0 or 0.0
            let pred = func.add_predicate_register();
            func.add_inst(Inst::setp_gt_f32(pred.clone(), a_reg, b_reg));

            let one = Operand::imm_f32(1.0);
            let zero = Operand::imm_f32(0.0);
            func.add_inst(Inst::selp_f32(dest_reg, one, zero, pred));
        }
        Stmt::Mask {
            dest,
            values,
            condition,
        } => {
            let dest_reg = ctx.get_or_alloc_reg(func, *dest);
            let values_reg = ctx.get_or_alloc_reg(func, *values);
            let cond_reg = ctx.get_or_alloc_reg(func, *condition);

            // Mask: select values[i] if condition[i] != 0, else 0
            let pred = func.add_predicate_register();
            let zero = Operand::imm_f32(0.0);

            // Check if condition != 0
            func.add_inst(Inst::setp_ne_f32(pred.clone(), cond_reg, zero.clone()));

            // Select values or zero based on predicate
            func.add_inst(Inst::selp_f32(dest_reg, values_reg, zero, pred));
        }
        Stmt::Barrier => {
            func.add_inst(Inst::BarSync { barrier_id: 0 });
        }
        Stmt::ForLoop {
            loop_var,
            start,
            end,
            body,
        } => {
            // Implement for loop with labels and branches
            // TODO: Need proper loop variable tracking in context
            let loop_counter = func.add_i32_register();

            // Initialize loop counter
            let start_val = lower_expr_i32(func, start);
            // Using add with 0 as a workaround for mov
            let temp = func.add_i32_register();
            func.add_inst(Inst::add_i32(temp.clone(), start_val, Operand::imm_i32(0)));
            func.add_inst(Inst::add_i32(
                loop_counter.clone(),
                temp,
                Operand::imm_i32(0),
            ));

            // Create labels
            let loop_start_label = format!("loop_start_{}", loop_var);
            let loop_body_label = format!("loop_body_{}", loop_var);
            let loop_end_label = format!("loop_end_{}", loop_var);

            // Loop start label
            func.add_inst(Inst::Label(loop_start_label.clone()));

            // Check loop condition: if counter < end, continue to body
            let end_val = lower_expr_i32(func, end);
            let pred = func.add_predicate_register();
            func.add_inst(Inst::setp_lt_i32(
                pred.clone(),
                loop_counter.clone(),
                end_val,
            ));

            // Branch to body if counter < end
            func.add_inst(Inst::Bra {
                condition: pred,
                target: loop_body_label.clone(),
            });

            // Otherwise fall through to end
            func.add_inst(Inst::BraUni {
                target: loop_end_label.clone(),
            });

            // Loop body label
            func.add_inst(Inst::Label(loop_body_label));

            // Loop body
            lower_block(func, ctx, body);

            // Increment loop counter
            let one = Operand::imm_i32(1);
            let incremented = func.add_i32_register();
            func.add_inst(Inst::add_i32(
                incremented.clone(),
                loop_counter.clone(),
                one,
            ));
            func.add_inst(Inst::add_i32(
                loop_counter.clone(),
                incremented,
                Operand::imm_i32(0),
            ));

            // Branch back to loop start
            func.add_inst(Inst::BraUni {
                target: loop_start_label,
            });

            // Loop end label
            func.add_inst(Inst::Label(loop_end_label));
        }
    }
}

/// Lower an expression to a U64 operand (for address calculations)
fn lower_expr(func: &mut Function, expr: &Expr) -> Operand<U64> {
    match expr {
        Expr::Const(val) => Operand::imm_u64(*val as u64),
        Expr::Var(name) => {
            // Variable reference - would need to be tracked in context
            Operand::reg(&format!("%{}", name))
        }
        Expr::BlockIdx(dim) => {
            use crate::tile::Dim;
            let block_idx = match dim {
                Dim::X => Operand::reg("%ctaid.x"),
                Dim::Y => Operand::reg("%ctaid.y"),
                Dim::Z => Operand::reg("%ctaid.z"),
            };
            // Convert from u32 to u64
            let result = func.add_u64_register();
            func.add_inst(Inst::convert_u64_u32(result.clone(), block_idx));
            result
        }
        Expr::ThreadIdx(dim) => {
            use crate::tile::Dim;
            let thread_idx = match dim {
                Dim::X => Operand::reg("%tid.x"),
                Dim::Y => Operand::reg("%tid.y"),
                Dim::Z => Operand::reg("%tid.z"),
            };
            // Convert from u32 to u64
            let result = func.add_u64_register();
            func.add_inst(Inst::convert_u64_u32(result.clone(), thread_idx));
            result
        }
        Expr::Mul(a, b) => {
            let a_val = lower_expr(func, a);
            let b_val = lower_expr(func, b);
            let result = func.add_u64_register();
            func.add_inst(Inst::mul_u64(result.clone(), a_val, b_val));
            result
        }
        Expr::Add(a, b) => {
            let a_val = lower_expr(func, a);
            let b_val = lower_expr(func, b);
            let result = func.add_u64_register();
            func.add_inst(Inst::add_u64(result.clone(), a_val, b_val));
            result
        }
    }
}

/// Lower an expression to an I32 operand (for loop counters)
fn lower_expr_i32(func: &mut Function, expr: &Expr) -> Operand<I32> {
    match expr {
        Expr::Const(val) => Operand::imm_i32(*val as i32),
        Expr::Var(name) => Operand::reg(&format!("%{}", name)),
        _ => {
            // For complex expressions, compute as u64 then convert
            let u64_val = lower_expr(func, expr);
            let result = func.add_i32_register();
            func.add_inst(Inst::convert_i32_u64(result.clone(), u64_val));
            result
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tile::{Block, DType, KernelParam, MemorySpace, Stmt, TileIR};

    #[test]
    fn test_basic_add_lowering() {
        // Create a simple TileIR with an Add operation
        let tile_ir = TileIR {
            kernel_name: "test_add".to_string(),
            params: vec![],
            shared_mem_bytes: 0,
            body: Block {
                stmts: vec![
                    Stmt::AllocTile {
                        var: TileVar(0),
                        space: MemorySpace::Register,
                        dtype: DType::F32,
                        rows: 1,
                        cols: 1,
                    },
                    Stmt::AllocTile {
                        var: TileVar(1),
                        space: MemorySpace::Register,
                        dtype: DType::F32,
                        rows: 1,
                        cols: 1,
                    },
                    Stmt::AllocTile {
                        var: TileVar(2),
                        space: MemorySpace::Register,
                        dtype: DType::F32,
                        rows: 1,
                        cols: 1,
                    },
                    Stmt::Add {
                        dest: TileVar(2),
                        a: TileVar(0),
                        b: TileVar(1),
                    },
                ],
            },
        };

        // Convert to PTX
        let func: Function = tile_ir.into();

        // Check that we have the right number of f32 registers
        assert_eq!(func.f32_registers.len(), 3);

        // Check that we have an Add instruction
        let has_add = func.body.iter().any(|inst| matches!(inst, Inst::AddF32(_)));
        assert!(has_add, "Should have generated an AddF32 instruction");

        // Check that function ends with Ret
        assert!(matches!(func.body.last(), Some(Inst::Ret)));
    }

    #[test]
    fn test_multiple_arithmetic_ops() {
        // Test: (a + b) * c - d
        let tile_ir = TileIR {
            kernel_name: "test_arithmetic".to_string(),
            params: vec![],
            shared_mem_bytes: 0,
            body: Block {
                stmts: vec![
                    Stmt::AllocTile {
                        var: TileVar(0),
                        space: MemorySpace::Register,
                        dtype: DType::F32,
                        rows: 1,
                        cols: 1,
                    },
                    Stmt::AllocTile {
                        var: TileVar(1),
                        space: MemorySpace::Register,
                        dtype: DType::F32,
                        rows: 1,
                        cols: 1,
                    },
                    Stmt::AllocTile {
                        var: TileVar(2),
                        space: MemorySpace::Register,
                        dtype: DType::F32,
                        rows: 1,
                        cols: 1,
                    },
                    Stmt::AllocTile {
                        var: TileVar(3),
                        space: MemorySpace::Register,
                        dtype: DType::F32,
                        rows: 1,
                        cols: 1,
                    },
                    Stmt::AllocTile {
                        var: TileVar(4),
                        space: MemorySpace::Register,
                        dtype: DType::F32,
                        rows: 1,
                        cols: 1,
                    },
                    Stmt::AllocTile {
                        var: TileVar(5),
                        space: MemorySpace::Register,
                        dtype: DType::F32,
                        rows: 1,
                        cols: 1,
                    },
                    Stmt::AllocTile {
                        var: TileVar(6),
                        space: MemorySpace::Register,
                        dtype: DType::F32,
                        rows: 1,
                        cols: 1,
                    },
                    Stmt::Add {
                        dest: TileVar(4),
                        a: TileVar(0),
                        b: TileVar(1),
                    },
                    Stmt::Mul {
                        dest: TileVar(5),
                        a: TileVar(4),
                        b: TileVar(2),
                    },
                    Stmt::Sub {
                        dest: TileVar(6),
                        a: TileVar(5),
                        b: TileVar(3),
                    },
                ],
            },
        };

        let func: Function = tile_ir.into();

        // Count instruction types
        let add_count = func
            .body
            .iter()
            .filter(|inst| matches!(inst, Inst::AddF32(_)))
            .count();
        let mul_count = func
            .body
            .iter()
            .filter(|inst| matches!(inst, Inst::MulF32(_)))
            .count();
        let sub_count = func
            .body
            .iter()
            .filter(|inst| matches!(inst, Inst::SubF32(_)))
            .count();

        assert_eq!(add_count, 1, "Should have 1 add instruction");
        assert_eq!(mul_count, 1, "Should have 1 mul instruction");
        assert_eq!(sub_count, 1, "Should have 1 sub instruction");
    }

    #[test]
    fn test_relu_lowering() {
        let tile_ir = TileIR {
            kernel_name: "test_relu".to_string(),
            params: vec![],
            shared_mem_bytes: 0,
            body: Block {
                stmts: vec![
                    Stmt::AllocTile {
                        var: TileVar(0),
                        space: MemorySpace::Register,
                        dtype: DType::F32,
                        rows: 1,
                        cols: 1,
                    },
                    Stmt::AllocTile {
                        var: TileVar(1),
                        space: MemorySpace::Register,
                        dtype: DType::F32,
                        rows: 1,
                        cols: 1,
                    },
                    Stmt::Relu {
                        dest: TileVar(1),
                        src: TileVar(0),
                    },
                ],
            },
        };

        let func: Function = tile_ir.into();

        // Relu is implemented as max(0, x)
        let has_max = func.body.iter().any(|inst| matches!(inst, Inst::MaxF32(_)));
        assert!(has_max, "Relu should use max instruction");
    }

    #[test]
    fn test_exp_log_lowering() {
        let tile_ir = TileIR {
            kernel_name: "test_exp_log".to_string(),
            params: vec![],
            shared_mem_bytes: 0,
            body: Block {
                stmts: vec![
                    Stmt::AllocTile {
                        var: TileVar(0),
                        space: MemorySpace::Register,
                        dtype: DType::F32,
                        rows: 1,
                        cols: 1,
                    },
                    Stmt::AllocTile {
                        var: TileVar(1),
                        space: MemorySpace::Register,
                        dtype: DType::F32,
                        rows: 1,
                        cols: 1,
                    },
                    Stmt::AllocTile {
                        var: TileVar(2),
                        space: MemorySpace::Register,
                        dtype: DType::F32,
                        rows: 1,
                        cols: 1,
                    },
                    Stmt::Exp {
                        dest: TileVar(1),
                        src: TileVar(0),
                    },
                    Stmt::Log {
                        dest: TileVar(2),
                        src: TileVar(1),
                    },
                ],
            },
        };

        let func: Function = tile_ir.into();

        // Exp uses ex2 (2^x), Log uses lg2 (log2(x))
        let has_ex2 = func.body.iter().any(|inst| matches!(inst, Inst::Ex2F32(_)));
        let has_lg2 = func.body.iter().any(|inst| matches!(inst, Inst::Lg2F32(_)));

        assert!(has_ex2, "Exp should use ex2 instruction");
        assert!(has_lg2, "Log should use lg2 instruction");
    }

    #[test]
    fn test_shared_memory_allocation() {
        let tile_ir = TileIR {
            kernel_name: "test_shared".to_string(),
            params: vec![],
            shared_mem_bytes: 4096,
            body: Block {
                stmts: vec![Stmt::AllocTile {
                    var: TileVar(0),
                    space: MemorySpace::Shared,
                    dtype: DType::F32,
                    rows: 16,
                    cols: 16,
                }],
            },
        };

        let func: Function = tile_ir.into();

        // Check that shared memory was allocated
        assert_eq!(func.shared_memory.len(), 1);
        assert_eq!(func.shared_memory[0].0, "shared");
        assert_eq!(func.shared_memory[0].1, 4096);

        // Check that a u64 register was allocated for the pointer
        assert_eq!(func.i64_registers.len(), 1);
    }

    #[test]
    fn test_barrier_lowering() {
        let tile_ir = TileIR {
            kernel_name: "test_barrier".to_string(),
            params: vec![],
            shared_mem_bytes: 0,
            body: Block {
                stmts: vec![Stmt::Barrier],
            },
        };

        let func: Function = tile_ir.into();

        let has_barrier = func
            .body
            .iter()
            .any(|inst| matches!(inst, Inst::BarSync { barrier_id: 0 }));
        assert!(has_barrier, "Should have barrier sync instruction");
    }

    #[test]
    fn test_comparison_and_mask() {
        let tile_ir = TileIR {
            kernel_name: "test_comparison".to_string(),
            params: vec![],
            shared_mem_bytes: 0,
            body: Block {
                stmts: vec![
                    Stmt::AllocTile {
                        var: TileVar(0),
                        space: MemorySpace::Register,
                        dtype: DType::F32,
                        rows: 1,
                        cols: 1,
                    },
                    Stmt::AllocTile {
                        var: TileVar(1),
                        space: MemorySpace::Register,
                        dtype: DType::F32,
                        rows: 1,
                        cols: 1,
                    },
                    Stmt::AllocTile {
                        var: TileVar(2),
                        space: MemorySpace::Register,
                        dtype: DType::F32,
                        rows: 1,
                        cols: 1,
                    },
                    Stmt::Gt {
                        dest: TileVar(2),
                        a: TileVar(0),
                        b: TileVar(1),
                    },
                ],
            },
        };

        let func: Function = tile_ir.into();

        // Gt uses setp and selp
        let has_setp = func.body.iter().any(|inst| matches!(inst, Inst::SetpF32(_)));
        let has_selp = func.body.iter().any(|inst| matches!(inst, Inst::SelpF32(_)));

        assert!(has_setp, "Gt should use setp instruction");
        assert!(has_selp, "Gt should use selp instruction");

        // Should allocate a predicate register
        assert_eq!(func.predicate_registers.len(), 1);
    }

    #[test]
    fn test_ptx_output_format() {
        // Test that the generated PTX can be formatted without panicking
        let tile_ir = TileIR {
            kernel_name: "simple_kernel".to_string(),
            params: vec![KernelParam {
                name: "input".to_string(),
                dtype: DType::F32,
                is_input: true,
            }],
            shared_mem_bytes: 0,
            body: Block {
                stmts: vec![
                    Stmt::AllocTile {
                        var: TileVar(0),
                        space: MemorySpace::Register,
                        dtype: DType::F32,
                        rows: 1,
                        cols: 1,
                    },
                    Stmt::Zero { tile: TileVar(0) },
                ],
            },
        };

        let func: Function = tile_ir.into();

        // This should not panic
        let ptx_str = format!("{}", func);

        // Basic sanity checks on the output
        assert!(ptx_str.contains(".visible .entry simple_kernel"));
        assert!(ptx_str.contains("ret;"));
        assert!(ptx_str.contains(".param .f32 input"));
        
        // Print the generated PTX for inspection
        println!("\n========== Generated PTX ==========\n{}\n===================================", ptx_str);
    }

    #[test]
    fn test_realistic_kernel_generation() {
        // Create a more realistic kernel: y = relu(a * x + b)
        let tile_ir = TileIR {
            kernel_name: "linear_relu".to_string(),
            params: vec![
                KernelParam {
                    name: "x".to_string(),
                    dtype: DType::F32,
                    is_input: true,
                },
                KernelParam {
                    name: "a".to_string(),
                    dtype: DType::F32,
                    is_input: true,
                },
                KernelParam {
                    name: "b".to_string(),
                    dtype: DType::F32,
                    is_input: true,
                },
                KernelParam {
                    name: "y".to_string(),
                    dtype: DType::F32,
                    is_input: false,
                },
            ],
            shared_mem_bytes: 0,
            body: Block {
                stmts: vec![
                    // Allocate registers
                    Stmt::AllocTile {
                        var: TileVar(0), // x_val
                        space: MemorySpace::Register,
                        dtype: DType::F32,
                        rows: 1,
                        cols: 1,
                    },
                    Stmt::AllocTile {
                        var: TileVar(1), // a_val
                        space: MemorySpace::Register,
                        dtype: DType::F32,
                        rows: 1,
                        cols: 1,
                    },
                    Stmt::AllocTile {
                        var: TileVar(2), // b_val
                        space: MemorySpace::Register,
                        dtype: DType::F32,
                        rows: 1,
                        cols: 1,
                    },
                    Stmt::AllocTile {
                        var: TileVar(3), // temp = a * x
                        space: MemorySpace::Register,
                        dtype: DType::F32,
                        rows: 1,
                        cols: 1,
                    },
                    Stmt::AllocTile {
                        var: TileVar(4), // temp2 = temp + b
                        space: MemorySpace::Register,
                        dtype: DType::F32,
                        rows: 1,
                        cols: 1,
                    },
                    Stmt::AllocTile {
                        var: TileVar(5), // result = relu(temp2)
                        space: MemorySpace::Register,
                        dtype: DType::F32,
                        rows: 1,
                        cols: 1,
                    },
                    // Compute: temp = a * x
                    Stmt::Mul {
                        dest: TileVar(3),
                        a: TileVar(1),
                        b: TileVar(0),
                    },
                    // Compute: temp2 = temp + b
                    Stmt::Add {
                        dest: TileVar(4),
                        a: TileVar(3),
                        b: TileVar(2),
                    },
                    // Compute: result = relu(temp2)
                    Stmt::Relu {
                        dest: TileVar(5),
                        src: TileVar(4),
                    },
                ],
            },
        };

        let func: Function = tile_ir.into();

        // Verify the function has all expected components
        assert_eq!(func.params.len(), 4);
        assert_eq!(func.f32_registers.len(), 6);

        // Count operations
        let mul_count = func
            .body
            .iter()
            .filter(|inst| matches!(inst, Inst::MulF32(_)))
            .count();
        let add_count = func
            .body
            .iter()
            .filter(|inst| matches!(inst, Inst::AddF32(_)))
            .count();
        let max_count = func
            .body
            .iter()
            .filter(|inst| matches!(inst, Inst::MaxF32(_)))
            .count();

        assert_eq!(mul_count, 1, "Should have 1 multiplication");
        assert_eq!(add_count, 1, "Should have 1 addition");
        assert_eq!(max_count, 1, "Should have 1 max (relu)");

        // Print the generated PTX
        let ptx_str = format!("{}", func);
        println!(
            "\n========== Realistic Kernel PTX ==========\n{}\n==========================================",
            ptx_str
        );
    }
}
