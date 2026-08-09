use std::collections::HashMap;

use super::instructions::{Inst, Operand};
use super::types::{B32, F32, I32, U64};
use super::{Function, Module};
use crate::tile::{Expr, TileGraph, TileIR, TileVar};
use petgraph::{Graph, graph::NodeIndex};

pub(crate) struct PtxGraph {
    // SAFETY INVARIANT: Functions contain references into `arena` whose lifetimes are
    // forged to 'static. `graph` is private, is declared before `arena` so it drops
    // first, and no API may return an owned Function or a borrow not tied to `&self`.
    graph: Graph<Function<'static>, usize>,
    #[allow(dead_code)]
    arena: Box<bumpalo::Bump>,
}

impl PtxGraph {
    pub(crate) fn module_source(&self) -> String {
        let mut module = Module::new();
        for function in self.graph.node_weights() {
            module.add_function(function.clone());
        }
        module.to_string()
    }

    pub(crate) fn kernel_name(&self, node: NodeIndex) -> Option<&str> {
        self.graph.node_weight(node).map(|function| function.name)
    }
}

impl From<TileGraph> for PtxGraph {
    fn from(tile_graph: TileGraph) -> Self {
        // Create a single arena for all functions in the graph
        let arena = Box::new(bumpalo::Bump::new());

        let graph = tile_graph.graph.map_owned(
            |node_idx, tile_ir| {
                // SAFETY: `arena` is boxed at a stable address and is owned by the
                // resulting PtxGraph. Its private graph is dropped before the arena,
                // and the impl above never lets an owned Function escape.
                let arena_ref: &'static bumpalo::Bump =
                    unsafe { &*(arena.as_ref() as *const bumpalo::Bump) };

                tile_ir_to_function(tile_ir, arena_ref, node_idx.index())
            },
            |_, weight| weight,
        );

        Self { graph, arena }
    }
}

fn tile_ir_to_function<'a>(
    tile_ir: TileIR,
    arena: &'a bumpalo::Bump,
    node_index: usize,
) -> Function<'a> {
    // Allocate kernel name in arena with node index suffix for uniqueness
    let kernel_name = bumpalo::format!(in arena, "{}_{}", tile_ir.kernel_name, node_index);
    let kernel_name_str = kernel_name.into_bump_str();

    let mut func = Function::new(kernel_name_str, arena);

    // Create lowering context to track tile variable mappings
    let mut ctx = LoweringContext::new(arena);

    // Add parameters as u64 pointers and store their loaded addresses
    for param in &tile_ir.params {
        // Allocate parameter name in arena
        let param_name = bumpalo::format!(in arena, "{}", param.name).into_bump_str();

        // All parameters are pointers, so they're u64 in PTX
        let ptr = func.add_global_ptr_param(param_name);
        ctx.param_ptrs.insert(param.name.clone(), ptr);
    }

    // Allocate shared memory if needed
    if tile_ir.shared_mem_bytes > 0 {
        let shared_name = arena.alloc_str("shared");
        func.add_shared_memory(shared_name, tile_ir.shared_mem_bytes);
    }

    // Convert body statements to PTX instructions
    lower_block(&mut func, &mut ctx, &tile_ir.body);

    func.add_inst(Inst::Label(ctx.return_label));
    func.add_inst(super::instructions::Inst::Ret);

    func
}

struct LoweringContext<'a> {
    /// Arena for allocating strings (labels, etc.)
    arena: &'a bumpalo::Bump,
    /// Maps TileVar to PTX register operands (for scalar/simple cases)
    tile_to_reg: HashMap<TileVar, Operand<'a, F32>>,
    /// Tracks allocated shared memory base pointers
    shared_mem_ptrs: HashMap<TileVar, Operand<'a, U64>>,
    /// Tracks loaded parameter pointers (global addresses)
    param_ptrs: HashMap<String, Operand<'a, U64>>,
    /// Maps loop variable names to their i32 register operands
    loop_vars: HashMap<String, Operand<'a, I32>>,
    /// Current offset into shared memory for allocation
    shared_mem_offset: usize,
    /// Maps TileVar to its dimensions (rows, cols)
    tile_dims: HashMap<TileVar, (usize, usize)>,
    tile_dtypes: HashMap<TileVar, crate::tile::DType>,
    fragment_regs: HashMap<TileVar, Vec<Operand<'a, F32>>>,
    /// Maps register tile vars to their shared memory source (for LoadSharedToReg)
    reg_to_shared: HashMap<TileVar, TileVar>,
    /// Counter for generating unique labels
    label_counter: usize,
    return_label: &'a str,
}

impl<'a> LoweringContext<'a> {
    fn new(arena: &'a bumpalo::Bump) -> Self {
        let return_label = arena.alloc_str("kernel_return");
        Self {
            arena,
            tile_to_reg: HashMap::new(),
            shared_mem_ptrs: HashMap::new(),
            param_ptrs: HashMap::new(),
            loop_vars: HashMap::new(),
            shared_mem_offset: 0,
            tile_dims: HashMap::new(),
            tile_dtypes: HashMap::new(),
            fragment_regs: HashMap::new(),
            reg_to_shared: HashMap::new(),
            label_counter: 0,
            return_label,
        }
    }

    fn get_or_alloc_reg(&mut self, func: &mut Function<'a>, var: TileVar) -> Operand<'a, F32> {
        if let Some(reg) = self.tile_to_reg.get(&var) {
            reg.clone()
        } else {
            let reg = func.add_f32_register();
            self.tile_to_reg.insert(var, reg.clone());
            reg
        }
    }
}

#[allow(dead_code)]
fn dtype_to_ptx_type(dtype: crate::tile::DType) -> super::types::Type {
    match dtype {
        crate::tile::DType::F16 => super::types::Type::F16,
        crate::tile::DType::BF16 => super::types::Type::BF16,
        crate::tile::DType::TF32 => super::types::Type::TF32,
        crate::tile::DType::F32 => super::types::Type::F32,
    }
}

fn lower_block<'a>(
    func: &mut Function<'a>,
    ctx: &mut LoweringContext<'a>,
    block: &crate::tile::Block,
) {
    for stmt in &block.stmts {
        lower_stmt(func, ctx, stmt);
    }
}

fn lower_stmt<'a>(
    func: &mut Function<'a>,
    ctx: &mut LoweringContext<'a>,
    stmt: &crate::tile::Stmt,
) {
    use crate::tile::Stmt;

    match stmt {
        Stmt::BoundsCheck { extent } => {
            let tid = global_linear_tid(func);
            let outside = func.add_predicate_register();
            func.add_inst(Inst::setp_ge_u64(
                outside.clone(),
                tid,
                Operand::imm_u64(*extent as u64),
            ));
            func.add_inst(Inst::Bra {
                condition: outside,
                target: ctx.return_label,
            });
        }
        Stmt::AllocTile {
            var,
            space,
            dtype,
            rows,
            cols,
        } => {
            // Track tile dimensions
            ctx.tile_dims.insert(*var, (*rows, *cols));
            ctx.tile_dtypes.insert(*var, *dtype);

            use crate::tile::MemorySpace;
            match space {
                MemorySpace::Register => {
                    // Allocate register for this tile variable
                    ctx.get_or_alloc_reg(func, *var);
                }
                MemorySpace::Fragment => {
                    let registers = (0..8).map(|_| func.add_f32_register()).collect();
                    ctx.fragment_regs.insert(*var, registers);
                }
                MemorySpace::Shared => {
                    // Calculate size in bytes for this tile
                    let tile_size_bytes = rows * cols * dtype.size_bytes();

                    // Get base address of shared memory array
                    let base_ptr = func.add_u64_register();
                    func.add_inst(Inst::MovU64 {
                        dst: base_ptr.clone(),
                        src: Operand::symbol("shared"),
                    });

                    // Add current offset to get this tile's pointer
                    let tile_ptr = func.add_u64_register();
                    func.add_inst(Inst::AddU64(super::instructions::AddInst::new(
                        tile_ptr.clone(),
                        base_ptr,
                        Operand::imm_u64(ctx.shared_mem_offset as u64),
                    )));

                    // Store the pointer and update offset
                    ctx.shared_mem_ptrs.insert(*var, tile_ptr);
                    ctx.shared_mem_offset += tile_size_bytes;
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
            col_offset,
        } => {
            // Load from global memory to register or shared memory
            // Get parameter pointer from cache (already loaded during initialization)
            let param_ptr = ctx
                .param_ptrs
                .get(src_param)
                .expect("Parameter not found in context")
                .clone();

            // Calculate element offset: row_offset + col_offset
            let row_offset_reg = lower_expr(func, ctx, row_offset);
            let col_offset_reg = lower_expr(func, ctx, col_offset);

            // Element offset = row_offset + col_offset
            let elem_offset = func.add_u64_register();
            func.add_inst(Inst::add_u64(
                elem_offset.clone(),
                row_offset_reg,
                col_offset_reg,
            ));

            // Byte offset = element offset * sizeof(f32) = element offset * 4
            let byte_offset = func.add_u64_register();
            func.add_inst(Inst::mul_u64(
                byte_offset.clone(),
                elem_offset,
                Operand::imm_u64(4),
            ));

            // Add offset to base pointer
            let addr = func.add_u64_register();
            func.add_inst(Inst::AddU64(super::instructions::AddInst::new(
                addr.clone(),
                param_ptr,
                byte_offset,
            )));

            // Check if destination is shared memory or register
            if let Some(shared_ptr) = ctx.shared_mem_ptrs.get(dest).cloned() {
                // Destination is shared memory: load to temp register, then store to shared
                let temp_reg = func.add_f32_register();
                func.add_inst(Inst::load_global_scalar_f32(temp_reg.clone(), addr));

                // Calculate offset within shared memory tile: threadIdx.y * cols + threadIdx.x
                let (_, cols) = ctx.tile_dims.get(dest).expect("Tile dimensions not found");

                // Convert tid.y and tid.x from u32 to i32 for arithmetic
                let tid_y_i32 = func.add_i32_register();
                func.add_inst(Inst::mov_i32(
                    tid_y_i32.clone(),
                    super::instructions::THREAD_ID.y.clone(),
                ));

                let tid_x_i32 = func.add_i32_register();
                func.add_inst(Inst::mov_i32(
                    tid_x_i32.clone(),
                    super::instructions::THREAD_ID.x.clone(),
                ));

                // row_offset = tid_y * cols
                let row_offset = func.add_i32_register();
                func.add_inst(Inst::mul_i32(
                    row_offset.clone(),
                    tid_y_i32,
                    Operand::imm_i32(*cols as i32),
                ));

                // elem_idx = row_offset + tid_x
                let elem_idx = func.add_i32_register();
                func.add_inst(Inst::add_i32(elem_idx.clone(), row_offset, tid_x_i32));

                // Convert to u64
                let elem_idx_u64 = func.add_u64_register();
                func.add_inst(Inst::convert_u64_i32(elem_idx_u64.clone(), elem_idx));

                // byte_offset = elem_idx * 4
                let byte_offset = func.add_u64_register();
                func.add_inst(Inst::mul_u64(
                    byte_offset.clone(),
                    elem_idx_u64,
                    Operand::imm_u64(4),
                ));

                // Add offset to base pointer
                let final_addr = func.add_u64_register();
                func.add_inst(Inst::add_u64(final_addr.clone(), shared_ptr, byte_offset));

                func.add_inst(Inst::StSharedF32 {
                    addr: final_addr,
                    src: vec![temp_reg],
                    vec: super::instructions::VecWidth::Scalar,
                });
            } else {
                // Destination is register: load directly
                let dest_reg = ctx.get_or_alloc_reg(func, *dest);
                func.add_inst(Inst::load_global_scalar_f32(dest_reg, addr));
            }
        }
        Stmt::LoadGlobalToSharedPredicated {
            dest,
            src_param,
            row,
            col,
            layout,
        } => {
            let row = lower_expr(func, ctx, row);
            let col = lower_expr(func, ctx, col);
            let value = func.add_f32_register();
            func.add_inst(Inst::mov_f32(value.clone(), Operand::imm_f32(0.0)));
            let skip_load =
                bumpalo::format!(in ctx.arena, "matmul_load_skip_{}", ctx.label_counter)
                    .into_bump_str();
            ctx.label_counter += 1;
            let row_outside = func.add_predicate_register();
            func.add_inst(Inst::setp_ge_u64(
                row_outside.clone(),
                row.clone(),
                Operand::imm_u64(layout.rows as u64),
            ));
            func.add_inst(Inst::Bra {
                condition: row_outside,
                target: skip_load,
            });
            let col_outside = func.add_predicate_register();
            func.add_inst(Inst::setp_ge_u64(
                col_outside.clone(),
                col.clone(),
                Operand::imm_u64(layout.cols as u64),
            ));
            func.add_inst(Inst::Bra {
                condition: col_outside,
                target: skip_load,
            });
            let row_offset = func.add_u64_register();
            func.add_inst(Inst::mul_u64(
                row_offset.clone(),
                row,
                Operand::imm_u64(layout.row_stride as u64),
            ));
            let element_offset = func.add_u64_register();
            func.add_inst(Inst::add_u64(element_offset.clone(), row_offset, col));
            let source = ctx
                .param_ptrs
                .get(src_param)
                .expect("Parameter not found in context")
                .clone();
            let address = global_f32_address(func, source, element_offset);
            func.add_inst(Inst::load_global_scalar_f32(value.clone(), address));
            func.add_inst(Inst::Label(skip_load));

            let shared_address = shared_thread_address(func, ctx, *dest);
            if ctx.tile_dtypes.get(dest) == Some(&crate::tile::DType::TF32) {
                let tf32 = func.add_b32_register();
                func.add_inst(Inst::convert_tf32_f32(tf32.clone(), value));
                func.add_inst(Inst::StSharedB32 {
                    addr: shared_address,
                    src: tf32,
                });
            } else {
                func.add_inst(Inst::StSharedF32 {
                    addr: shared_address,
                    src: vec![value],
                    vec: super::instructions::VecWidth::Scalar,
                });
            }
        }
        Stmt::Store {
            dest_param,
            src,
            row_offset,
            col_offset,
        } => {
            // Store from register/shared to global memory
            let src_reg = ctx.get_or_alloc_reg(func, *src);

            // Get parameter pointer from cache (already loaded during initialization)
            let param_ptr = ctx
                .param_ptrs
                .get(dest_param)
                .expect("Parameter not found in context")
                .clone();

            // Calculate element offset: row * N + col (need to know matrix width N)
            // For now, just use row_offset which should already include the full offset
            // TODO: This assumes row_offset already accounts for row-major layout
            let row_offset_reg = lower_expr(func, ctx, row_offset);
            let col_offset_reg = lower_expr(func, ctx, col_offset);

            // Element offset = row_offset + col_offset
            let elem_offset = func.add_u64_register();
            func.add_inst(Inst::add_u64(
                elem_offset.clone(),
                row_offset_reg,
                col_offset_reg,
            ));

            // Byte offset = element offset * sizeof(f32) = element offset * 4
            let byte_offset = func.add_u64_register();
            func.add_inst(Inst::mul_u64(
                byte_offset.clone(),
                elem_offset,
                Operand::imm_u64(4),
            ));

            // Add offset to base pointer
            let addr = func.add_u64_register();
            func.add_inst(Inst::AddU64(super::instructions::AddInst::new(
                addr.clone(),
                param_ptr,
                byte_offset,
            )));

            // Store to global memory
            func.add_inst(Inst::store_global_scalar_f32(addr, src_reg));
        }
        Stmt::StoreGlobalPredicated {
            dest_param,
            src,
            row,
            col,
            layout,
        } => {
            let row = lower_expr(func, ctx, row);
            let col = lower_expr(func, ctx, col);
            let skip_store =
                bumpalo::format!(in ctx.arena, "matmul_store_skip_{}", ctx.label_counter)
                    .into_bump_str();
            ctx.label_counter += 1;
            let row_outside = func.add_predicate_register();
            func.add_inst(Inst::setp_ge_u64(
                row_outside.clone(),
                row.clone(),
                Operand::imm_u64(layout.rows as u64),
            ));
            func.add_inst(Inst::Bra {
                condition: row_outside,
                target: skip_store,
            });
            let col_outside = func.add_predicate_register();
            func.add_inst(Inst::setp_ge_u64(
                col_outside.clone(),
                col.clone(),
                Operand::imm_u64(layout.cols as u64),
            ));
            func.add_inst(Inst::Bra {
                condition: col_outside,
                target: skip_store,
            });
            let row_offset = func.add_u64_register();
            func.add_inst(Inst::mul_u64(
                row_offset.clone(),
                row,
                Operand::imm_u64(layout.row_stride as u64),
            ));
            let element_offset = func.add_u64_register();
            func.add_inst(Inst::add_u64(element_offset.clone(), row_offset, col));
            let destination = ctx
                .param_ptrs
                .get(dest_param)
                .expect("Parameter not found in context")
                .clone();
            let address = global_f32_address(func, destination, element_offset);
            let value = if ctx.fragment_regs.contains_key(src) {
                let shared = ctx
                    .reg_to_shared
                    .get(src)
                    .expect("WMMA accumulator has no shared-memory backing");
                let address = shared_thread_address(func, ctx, *shared);
                let value = func.add_f32_register();
                func.add_inst(Inst::load_shared_scalar_f32(value.clone(), address));
                value
            } else {
                ctx.get_or_alloc_reg(func, *src)
            };
            func.add_inst(Inst::store_global_scalar_f32(address, value));
            func.add_inst(Inst::Label(skip_store));
        }
        Stmt::LoadSharedToReg { dest, src } => {
            // Track that this register tile is backed by shared memory
            // We don't need to emit any PTX here - MatMul will access shared memory directly
            ctx.reg_to_shared.insert(*dest, *src);

            // Also copy the tile dimensions
            if let Some(&dims) = ctx.tile_dims.get(src) {
                ctx.tile_dims.insert(*dest, dims);
            }
        }
        Stmt::Zero { tile } => {
            let zero = Operand::imm_f32(0.0);
            if let Some(registers) = ctx.fragment_regs.get(tile) {
                for register in registers.clone() {
                    func.add_inst(Inst::mov_f32(register, zero.clone()));
                }
            } else {
                let reg = ctx.get_or_alloc_reg(func, *tile);
                func.add_inst(Inst::mov_f32(reg, zero));
            }
        }
        Stmt::MatMul {
            dest,
            a,
            b,
            layout: _,
            plan,
        } => {
            if *plan == crate::tile::MatMulPlan::TensorCoreTf32 {
                lower_tf32_matmul(func, ctx, *dest, *a, *b);
                return;
            }
            // Matrix multiply: each thread computes one output element
            // Thread at (ty, tx) computes C[ty][tx] = sum over k of A[ty][k] * B[k][tx]

            let dest_reg = ctx.get_or_alloc_reg(func, *dest);

            // Resolve register tiles to their shared memory sources
            let a_smem = ctx.reg_to_shared.get(a).copied().unwrap_or(*a);
            let b_smem = ctx.reg_to_shared.get(b).copied().unwrap_or(*b);

            // Get shared memory pointers for A and B tiles
            let a_ptr = ctx
                .shared_mem_ptrs
                .get(&a_smem)
                .expect("MatMul operand A not in shared memory")
                .clone();
            let b_ptr = ctx
                .shared_mem_ptrs
                .get(&b_smem)
                .expect("MatMul operand B not in shared memory")
                .clone();

            // Get tile dimensions
            let (_a_rows, a_cols) = ctx
                .tile_dims
                .get(&a_smem)
                .expect("Tile dimensions not found for A");
            let (_b_rows, b_cols) = ctx
                .tile_dims
                .get(&b_smem)
                .expect("Tile dimensions not found for B");
            let tile_k = *a_cols; // K dimension of the tile

            // Get threadIdx.y and threadIdx.x (this thread's position in the output tile)
            let tid_y = func.add_u32_register();
            let tid_x = func.add_u32_register();
            func.add_inst(Inst::mov_u32(
                tid_y.clone(),
                super::instructions::THREAD_ID.y.clone(),
            ));
            func.add_inst(Inst::mov_u32(
                tid_x.clone(),
                super::instructions::THREAD_ID.x.clone(),
            ));

            // Hoist invariant computations outside the loop
            // Convert thread indices to u64 once
            let tid_y_u64 = func.add_u64_register();
            let tid_x_u64 = func.add_u64_register();
            func.add_inst(Inst::convert_u64_u32(tid_y_u64.clone(), tid_y.clone()));
            func.add_inst(Inst::convert_u64_u32(tid_x_u64.clone(), tid_x.clone()));

            // Compute A row base offset: threadIdx.y * a_cols * 4 (in bytes)
            let a_row_base = func.add_u64_register();
            func.add_inst(Inst::mul_u64(
                a_row_base.clone(),
                tid_y_u64.clone(),
                Operand::imm_u64(*a_cols as u64),
            ));
            func.add_inst(Inst::mul_u64(
                a_row_base.clone(),
                a_row_base.clone(),
                Operand::imm_u64(4),
            ));
            let a_row_ptr = func.add_u64_register();
            func.add_inst(Inst::add_u64(a_row_ptr.clone(), a_ptr.clone(), a_row_base));

            // Compute B column base offset: threadIdx.x * 4 (in bytes)
            let b_col_base = func.add_u64_register();
            func.add_inst(Inst::mul_u64(
                b_col_base.clone(),
                tid_x_u64.clone(),
                Operand::imm_u64(4),
            ));
            let b_col_ptr = func.add_u64_register();
            func.add_inst(Inst::add_u64(b_col_ptr.clone(), b_ptr.clone(), b_col_base));

            // Reusable registers for the loop
            let a_addr = func.add_u64_register();
            let b_addr = func.add_u64_register();
            let a_val = func.add_f32_register();
            let b_val = func.add_f32_register();
            let prod = func.add_f32_register();

            // Loop over k dimension using PTX control flow instead of unrolling
            // for (k = 0; k < tile_k; k++)
            let k_idx = func.add_i32_register();
            let k_limit = func.add_i32_register();

            // Initialize k = 0
            func.add_inst(Inst::mov_i32(k_idx.clone(), Operand::imm_u64(0)));
            // Load limit into register
            func.add_inst(Inst::mov_i32(
                k_limit.clone(),
                Operand::imm_u64(tile_k as u64),
            ));

            // Loop start label
            let loop_start =
                bumpalo::format!(in ctx.arena, "matmul_loop_start_{}", ctx.label_counter)
                    .into_bump_str();
            let loop_end = bumpalo::format!(in ctx.arena, "matmul_loop_end_{}", ctx.label_counter)
                .into_bump_str();
            ctx.label_counter += 1;

            func.add_inst(Inst::Label(loop_start));

            // Check loop condition: setp.ge sets pred when k >= tile_k (exit condition)
            let exit_pred = func.add_predicate_register();
            func.add_inst(Inst::SetpI32(super::instructions::SetpInst::new(
                exit_pred.clone(),
                k_idx.clone(),
                k_limit.clone(),
                super::instructions::CompareOp::Ge,
            )));
            func.add_inst(Inst::Bra {
                condition: exit_pred,
                target: loop_end,
            });

            // Loop body:
            // Convert k to u64 for address calculations
            let k_u64 = func.add_u64_register();
            func.add_inst(Inst::convert_u64_i32(k_u64.clone(), k_idx.clone()));

            // Load A[threadIdx.y][k] from shared memory
            // Address = a_row_ptr + k * 4
            let k_offset_a = func.add_u64_register();
            func.add_inst(Inst::mul_u64(
                k_offset_a.clone(),
                k_u64.clone(),
                Operand::imm_u64(4),
            ));
            func.add_inst(Inst::add_u64(a_addr.clone(), a_row_ptr.clone(), k_offset_a));
            func.add_inst(Inst::load_shared_scalar_f32(a_val.clone(), a_addr.clone()));

            // Load B[k][threadIdx.x] from shared memory
            // Address = b_col_ptr + k * b_cols * 4
            let k_offset_b = func.add_u64_register();
            func.add_inst(Inst::mul_u64(
                k_offset_b.clone(),
                k_u64.clone(),
                Operand::imm_u64(*b_cols as u64),
            ));
            func.add_inst(Inst::mul_u64(
                k_offset_b.clone(),
                k_offset_b.clone(),
                Operand::imm_u64(4),
            ));
            func.add_inst(Inst::add_u64(b_addr.clone(), b_col_ptr.clone(), k_offset_b));
            func.add_inst(Inst::load_shared_scalar_f32(b_val.clone(), b_addr.clone()));

            // Multiply and accumulate: dest += a_val * b_val
            func.add_inst(Inst::mul_f32(prod.clone(), a_val.clone(), b_val.clone()));
            func.add_inst(Inst::add_f32(
                dest_reg.clone(),
                dest_reg.clone(),
                prod.clone(),
            ));

            // Increment k
            func.add_inst(Inst::add_i32(
                k_idx.clone(),
                k_idx.clone(),
                Operand::imm_i32(1),
            ));

            // Branch back to loop start
            func.add_inst(Inst::BraUni { target: loop_start });

            // Loop end label
            func.add_inst(Inst::Label(loop_end));
        }
        Stmt::Embedding {
            vocabulary: _,
            width,
            index_count: _,
        } => {
            let tid = global_linear_tid(func);
            let width = (*width).max(1);
            let index_position = func.add_u64_register();
            func.add_inst(Inst::div_u64(
                index_position.clone(),
                tid.clone(),
                Operand::imm_u64(width as u64),
            ));
            let row_start = func.add_u64_register();
            func.add_inst(Inst::mul_u64(
                row_start.clone(),
                index_position.clone(),
                Operand::imm_u64(width as u64),
            ));
            let column = func.add_u64_register();
            func.add_inst(Inst::sub_u64(column.clone(), tid, row_start));

            let index_value = load_param_f32_at(func, ctx, "indices", index_position);
            let index = func.add_u64_register();
            func.add_inst(Inst::convert_u64_f32(index.clone(), index_value));
            let source_row = func.add_u64_register();
            func.add_inst(Inst::mul_u64(
                source_row.clone(),
                index,
                Operand::imm_u64(width as u64),
            ));
            let source_offset = func.add_u64_register();
            func.add_inst(Inst::add_u64(source_offset.clone(), source_row, column));
            let value = load_param_f32_at(func, ctx, "weight", source_offset);
            let output_offset = global_linear_tid(func);
            store_param_f32_at(func, ctx, "output", output_offset, value);
        }
        Stmt::EmbeddingBackward {
            vocabulary: _,
            width,
            index_count: _,
        } => {
            let tid = global_linear_tid(func);
            let width = (*width).max(1);
            let index_position = func.add_u64_register();
            func.add_inst(Inst::div_u64(
                index_position.clone(),
                tid.clone(),
                Operand::imm_u64(width as u64),
            ));
            let row_start = func.add_u64_register();
            func.add_inst(Inst::mul_u64(
                row_start.clone(),
                index_position.clone(),
                Operand::imm_u64(width as u64),
            ));
            let column = func.add_u64_register();
            func.add_inst(Inst::sub_u64(column.clone(), tid.clone(), row_start));
            let index_value = load_param_f32_at(func, ctx, "indices", index_position);
            let index = func.add_u64_register();
            func.add_inst(Inst::convert_u64_f32(index.clone(), index_value));
            let destination_row = func.add_u64_register();
            func.add_inst(Inst::mul_u64(
                destination_row.clone(),
                index,
                Operand::imm_u64(width as u64),
            ));
            let destination_offset = func.add_u64_register();
            func.add_inst(Inst::add_u64(
                destination_offset.clone(),
                destination_row,
                column,
            ));
            let gradient = load_param_f32_at(func, ctx, "grad_output", tid);
            let output_ptr = ctx
                .param_ptrs
                .get("output")
                .expect("Output parameter not found")
                .clone();
            let destination = global_f32_address(func, output_ptr, destination_offset);
            let discarded = func.add_f32_register();
            func.add_inst(Inst::atomic_add_global_f32(
                discarded,
                destination,
                gradient,
            ));
        }
        Stmt::IndexedCrossEntropy {
            vocabulary,
            row_count: _,
        } => {
            let row = global_linear_tid(func);
            let row_start = func.add_u64_register();
            func.add_inst(Inst::mul_u64(
                row_start.clone(),
                row.clone(),
                Operand::imm_u64(*vocabulary as u64),
            ));
            let (maximum, exponential_sum) =
                emit_row_softmax_stats(func, ctx, row_start.clone(), *vocabulary);
            let target_value = load_param_f32_at(func, ctx, "targets", row.clone());
            let target = func.add_u64_register();
            func.add_inst(Inst::convert_u64_f32(target.clone(), target_value));
            let target_offset = func.add_u64_register();
            func.add_inst(Inst::add_u64(target_offset.clone(), row_start, target));
            let target_logit = load_param_f32_at(func, ctx, "logits", target_offset);
            let shifted = func.add_f32_register();
            func.add_inst(Inst::sub_f32(shifted.clone(), maximum, target_logit));
            let log_sum = emit_log(func, exponential_sum);
            let loss = func.add_f32_register();
            func.add_inst(Inst::add_f32(loss.clone(), shifted, log_sum));
            store_param_f32_at(func, ctx, "output", row, loss);
        }
        Stmt::IndexedCrossEntropyBackward {
            vocabulary,
            row_count: _,
        } => {
            let tid = global_linear_tid(func);
            let row = func.add_u64_register();
            func.add_inst(Inst::div_u64(
                row.clone(),
                tid.clone(),
                Operand::imm_u64(*vocabulary as u64),
            ));
            let row_start = func.add_u64_register();
            func.add_inst(Inst::mul_u64(
                row_start.clone(),
                row.clone(),
                Operand::imm_u64(*vocabulary as u64),
            ));
            let column = func.add_u64_register();
            func.add_inst(Inst::sub_u64(
                column.clone(),
                tid.clone(),
                row_start.clone(),
            ));
            let (maximum, exponential_sum) =
                emit_row_softmax_stats(func, ctx, row_start, *vocabulary);
            let logit = load_param_f32_at(func, ctx, "logits", tid.clone());
            let shifted = func.add_f32_register();
            func.add_inst(Inst::sub_f32(shifted.clone(), logit, maximum));
            let exponential = emit_exp(func, shifted);
            let probability = func.add_f32_register();
            func.add_inst(Inst::div_f32(
                probability.clone(),
                exponential,
                exponential_sum,
            ));
            let target_value = load_param_f32_at(func, ctx, "targets", row.clone());
            let target = func.add_u64_register();
            func.add_inst(Inst::convert_u64_f32(target.clone(), target_value));
            let is_target = func.add_predicate_register();
            func.add_inst(Inst::setp_eq_u64(is_target.clone(), column, target));
            let indicator = func.add_f32_register();
            func.add_inst(Inst::selp_f32(
                indicator.clone(),
                Operand::imm_f32(1.0),
                Operand::imm_f32(0.0),
                is_target,
            ));
            let difference = func.add_f32_register();
            func.add_inst(Inst::sub_f32(difference.clone(), probability, indicator));
            let upstream = load_param_f32_at(func, ctx, "grad_output", row);
            let gradient = func.add_f32_register();
            func.add_inst(Inst::mul_f32(gradient.clone(), difference, upstream));
            store_param_f32_at(func, ctx, "output", tid, gradient);
        }
        Stmt::Conv2d { geometry } => lower_conv2d(func, ctx, *geometry),
        Stmt::ConvTranspose2d { geometry } => lower_conv_transpose2d(func, ctx, *geometry),
        Stmt::Conv2dBackwardWeight { geometry } => {
            lower_conv2d_backward_weight(func, ctx, *geometry)
        }
        Stmt::MaxPool2d { geometry } => lower_max_pool2d(func, ctx, *geometry),
        Stmt::MaxPool2dBackward { geometry } => lower_max_pool2d_backward(func, ctx, *geometry),
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
        Stmt::Reindex {
            dest,
            src: _,
            input_shape,
            output_shape,
            axes,
        } => {
            let dest_reg = ctx.get_or_alloc_reg(func, *dest);
            let src_ptr = ctx
                .param_ptrs
                .get("input")
                .expect("Input parameter not found")
                .clone();
            let tid = global_linear_tid(func);
            let input_strides = contiguous_strides(input_shape);
            let output_strides = contiguous_strides(output_shape);
            let input_offset = func.add_u64_register();
            func.add_inst(Inst::mov_u64(input_offset.clone(), Operand::imm_u64(0)));
            for output_axis in 0..output_shape.len() {
                let coordinate = decode_coordinate(
                    func,
                    tid.clone(),
                    output_strides[output_axis],
                    output_shape[output_axis],
                );
                let contribution = func.add_u64_register();
                func.add_inst(Inst::mul_u64(
                    contribution.clone(),
                    coordinate,
                    Operand::imm_u64(input_strides[axes[output_axis]] as u64),
                ));
                func.add_inst(Inst::add_u64(
                    input_offset.clone(),
                    input_offset.clone(),
                    contribution,
                ));
            }
            load_f32_at(func, dest_reg, src_ptr, input_offset);
        }
        Stmt::BroadcastAxis {
            dest,
            src: _,
            axis,
            input_shape,
            output_shape,
        } => {
            let dest_reg = ctx.get_or_alloc_reg(func, *dest);
            let src_ptr = ctx
                .param_ptrs
                .get("input")
                .expect("Input parameter not found")
                .clone();

            let tid = global_linear_tid(func);
            let input_strides = contiguous_strides(input_shape);
            let output_strides = contiguous_strides(output_shape);
            let input_offset = func.add_u64_register();
            func.add_inst(Inst::mov_u64(input_offset.clone(), Operand::imm_u64(0)));
            for dimension in 0..output_shape.len() {
                if dimension == *axis {
                    continue;
                }
                let coordinate = decode_coordinate(
                    func,
                    tid.clone(),
                    output_strides[dimension],
                    output_shape[dimension],
                );
                let contribution = func.add_u64_register();
                func.add_inst(Inst::mul_u64(
                    contribution.clone(),
                    coordinate,
                    Operand::imm_u64(input_strides[dimension] as u64),
                ));
                func.add_inst(Inst::add_u64(
                    input_offset.clone(),
                    input_offset.clone(),
                    contribution,
                ));
            }
            load_f32_at(func, dest_reg, src_ptr, input_offset);
        }
        Stmt::ReduceAxis {
            dest,
            src: _,
            op,
            axis,
            input_shape,
            output_shape,
        } => {
            // Each thread computes one output element by reducing along the specified axis
            // Example: input shape [2, 3], axis=1, output shape [2, 1]
            //   Thread 0 computes output[0] = reduce(input[0, :])
            //   Thread 1 computes output[1] = reduce(input[1, :])

            let dest_reg = ctx.get_or_alloc_reg(func, *dest);

            // Get parameter pointers for direct access
            let src_ptr = ctx
                .param_ptrs
                .get("input")
                .expect("Input parameter not found")
                .clone();
            let input_strides = contiguous_strides(input_shape);
            let output_strides = contiguous_strides(output_shape);
            let tid = global_linear_tid(func);
            let base_offset = func.add_u64_register();
            func.add_inst(Inst::mov_u64(base_offset.clone(), Operand::imm_u64(0)));
            for dimension in 0..output_shape.len() {
                if dimension == *axis {
                    continue;
                }
                let coordinate = decode_coordinate(
                    func,
                    tid.clone(),
                    output_strides[dimension],
                    output_shape[dimension],
                );
                let contribution = func.add_u64_register();
                func.add_inst(Inst::mul_u64(
                    contribution.clone(),
                    coordinate,
                    Operand::imm_u64(input_strides[dimension] as u64),
                ));
                func.add_inst(Inst::add_u64(
                    base_offset.clone(),
                    base_offset.clone(),
                    contribution,
                ));
            }

            // Initialize accumulator based on reduce operation
            let accumulator = func.add_f32_register();
            match op {
                crate::tile::ReduceOp::Sum | crate::tile::ReduceOp::Mean => {
                    func.add_inst(Inst::mov_f32(accumulator.clone(), Operand::imm_f32(0.0)));
                }
                crate::tile::ReduceOp::Max => {
                    func.add_inst(Inst::mov_f32(
                        accumulator.clone(),
                        Operand::imm_f32(f32::NEG_INFINITY),
                    ));
                }
            }

            // Calculate the size of the reduction axis
            let reduce_size = input_shape[*axis];

            // Loop over the reduction axis
            for k in 0..reduce_size {
                let input_offset = func.add_u64_register();
                let k_contribution = func.add_u64_register();
                func.add_inst(Inst::mul_u64(
                    k_contribution.clone(),
                    Operand::imm_u64(k as u64),
                    Operand::imm_u64(input_strides[*axis] as u64),
                ));
                func.add_inst(Inst::add_u64(
                    input_offset.clone(),
                    base_offset.clone(),
                    k_contribution,
                ));

                // Convert element offset to byte offset
                let byte_offset = func.add_u64_register();
                func.add_inst(Inst::mul_u64(
                    byte_offset.clone(),
                    input_offset,
                    Operand::imm_u64(4),
                ));

                // Load input element
                let addr = func.add_u64_register();
                func.add_inst(Inst::add_u64(addr.clone(), src_ptr.clone(), byte_offset));

                let value = func.add_f32_register();
                func.add_inst(Inst::load_global_scalar_f32(value.clone(), addr));

                // Accumulate based on operation
                match op {
                    crate::tile::ReduceOp::Sum | crate::tile::ReduceOp::Mean => {
                        let new_acc = func.add_f32_register();
                        func.add_inst(Inst::add_f32(new_acc.clone(), accumulator.clone(), value));
                        func.add_inst(Inst::mov_f32(accumulator.clone(), new_acc));
                    }
                    crate::tile::ReduceOp::Max => {
                        let new_acc = func.add_f32_register();
                        func.add_inst(Inst::max_f32(new_acc.clone(), accumulator.clone(), value));
                        func.add_inst(Inst::mov_f32(accumulator.clone(), new_acc));
                    }
                }
            }

            // For Mean operation, divide by the reduction size
            if matches!(op, crate::tile::ReduceOp::Mean) {
                let divisor = Operand::imm_f32(reduce_size as f32);
                let result = func.add_f32_register();
                func.add_inst(Inst::div_f32(result.clone(), accumulator, divisor));
                func.add_inst(Inst::mov_f32(dest_reg, result));
            } else {
                func.add_inst(Inst::mov_f32(dest_reg, accumulator));
            }
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
            let loop_counter = func.add_i32_register();

            // Initialize loop counter
            let start_val = lower_expr_i32(func, ctx, start);
            // Using add with 0 as a workaround for mov
            let temp = func.add_i32_register();
            func.add_inst(Inst::add_i32(temp.clone(), start_val, Operand::imm_i32(0)));
            func.add_inst(Inst::add_i32(
                loop_counter.clone(),
                temp,
                Operand::imm_i32(0),
            ));

            // Track the loop variable in context
            ctx.loop_vars.insert(loop_var.clone(), loop_counter.clone());

            // Create labels
            let loop_start_label =
                bumpalo::format!(in ctx.arena, "loop_start_{}", loop_var).into_bump_str();
            let loop_body_label =
                bumpalo::format!(in ctx.arena, "loop_body_{}", loop_var).into_bump_str();
            let loop_end_label =
                bumpalo::format!(in ctx.arena, "loop_end_{}", loop_var).into_bump_str();

            // Loop start label
            func.add_inst(Inst::Label(loop_start_label));

            // Check loop condition: if counter < end, continue to body
            let end_val = lower_expr_i32(func, ctx, end);
            let pred = func.add_predicate_register();
            func.add_inst(Inst::setp_lt_i32(
                pred.clone(),
                loop_counter.clone(),
                end_val,
            ));

            // Branch to body if counter < end
            func.add_inst(Inst::Bra {
                condition: pred,
                target: loop_body_label,
            });

            // Otherwise fall through to end
            func.add_inst(Inst::BraUni {
                target: loop_end_label,
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

            // Remove loop variable from context
            ctx.loop_vars.remove(loop_var);
        }
    }
}

fn lower_conv2d<'a>(
    func: &mut Function<'a>,
    ctx: &mut LoweringContext<'a>,
    geometry: crate::tile::Conv2dGeometry,
) {
    let tid = global_linear_tid(func);
    let (batch, output_channel, output_row, output_col) = decode_nchw(
        func,
        tid.clone(),
        geometry.output_channels,
        geometry.output_height,
        geometry.output_width,
    );
    let accumulator = func.add_f32_register();
    func.add_inst(Inst::mov_f32(accumulator.clone(), Operand::imm_f32(0.0)));
    let reduction_size = geometry.input_channels * geometry.kernel_height * geometry.kernel_width;
    let reduction = begin_counted_loop(func, ctx, "conv", reduction_size);
    let input_channel = decode_coordinate(
        func,
        reduction.counter.clone(),
        geometry.kernel_height * geometry.kernel_width,
        geometry.input_channels,
    );
    let kernel_row = decode_coordinate(
        func,
        reduction.counter.clone(),
        geometry.kernel_width,
        geometry.kernel_height,
    );
    let kernel_col = decode_coordinate(func, reduction.counter.clone(), 1, geometry.kernel_width);
    let continue_label = next_label(ctx, "conv_continue");
    let input_row = padded_input_coordinate(
        func,
        output_row,
        kernel_row.clone(),
        geometry.stride,
        geometry.padding,
        geometry.input_height,
        continue_label,
    );
    let input_col = padded_input_coordinate(
        func,
        output_col,
        kernel_col.clone(),
        geometry.stride,
        geometry.padding,
        geometry.input_width,
        continue_label,
    );
    let input_offset = flatten_nchw(
        func,
        batch,
        input_channel.clone(),
        input_row,
        input_col,
        geometry.input_channels,
        geometry.input_height,
        geometry.input_width,
    );
    let weight_offset = flatten_nchw(
        func,
        output_channel,
        input_channel,
        kernel_row,
        kernel_col,
        geometry.input_channels,
        geometry.kernel_height,
        geometry.kernel_width,
    );
    let input = load_param_f32_at(func, ctx, "input", input_offset);
    let weight = load_param_f32_at(func, ctx, "weight", weight_offset);
    accumulate_product(func, accumulator.clone(), input, weight);
    func.add_inst(Inst::Label(continue_label));
    end_counted_loop(func, reduction);
    store_param_f32_at(func, ctx, "output", tid, accumulator);
}

fn lower_conv_transpose2d<'a>(
    func: &mut Function<'a>,
    ctx: &mut LoweringContext<'a>,
    geometry: crate::tile::Conv2dGeometry,
) {
    let tid = global_linear_tid(func);
    let (batch, input_channel, input_row, input_col) = decode_nchw(
        func,
        tid.clone(),
        geometry.input_channels,
        geometry.input_height,
        geometry.input_width,
    );
    let accumulator = func.add_f32_register();
    func.add_inst(Inst::mov_f32(accumulator.clone(), Operand::imm_f32(0.0)));
    let reduction_size = geometry.output_channels * geometry.kernel_height * geometry.kernel_width;
    let reduction = begin_counted_loop(func, ctx, "conv_transpose", reduction_size);
    let output_channel = decode_coordinate(
        func,
        reduction.counter.clone(),
        geometry.kernel_height * geometry.kernel_width,
        geometry.output_channels,
    );
    let kernel_row = decode_coordinate(
        func,
        reduction.counter.clone(),
        geometry.kernel_width,
        geometry.kernel_height,
    );
    let kernel_col = decode_coordinate(func, reduction.counter.clone(), 1, geometry.kernel_width);
    let continue_label = next_label(ctx, "conv_transpose_continue");
    let output_row = transposed_output_coordinate(
        func,
        input_row,
        kernel_row.clone(),
        geometry.stride,
        geometry.padding,
        geometry.output_height,
        continue_label,
    );
    let output_col = transposed_output_coordinate(
        func,
        input_col,
        kernel_col.clone(),
        geometry.stride,
        geometry.padding,
        geometry.output_width,
        continue_label,
    );
    let grad_offset = flatten_nchw(
        func,
        batch,
        output_channel.clone(),
        output_row,
        output_col,
        geometry.output_channels,
        geometry.output_height,
        geometry.output_width,
    );
    let weight_offset = flatten_nchw(
        func,
        output_channel,
        input_channel,
        kernel_row,
        kernel_col,
        geometry.input_channels,
        geometry.kernel_height,
        geometry.kernel_width,
    );
    let grad = load_param_f32_at(func, ctx, "grad_output", grad_offset);
    let weight = load_param_f32_at(func, ctx, "weight", weight_offset);
    accumulate_product(func, accumulator.clone(), grad, weight);
    func.add_inst(Inst::Label(continue_label));
    end_counted_loop(func, reduction);
    store_param_f32_at(func, ctx, "output", tid, accumulator);
}

fn lower_conv2d_backward_weight<'a>(
    func: &mut Function<'a>,
    ctx: &mut LoweringContext<'a>,
    geometry: crate::tile::Conv2dGeometry,
) {
    let tid = global_linear_tid(func);
    let (output_channel, input_channel, kernel_row, kernel_col) = decode_nchw(
        func,
        tid.clone(),
        geometry.input_channels,
        geometry.kernel_height,
        geometry.kernel_width,
    );
    let accumulator = func.add_f32_register();
    func.add_inst(Inst::mov_f32(accumulator.clone(), Operand::imm_f32(0.0)));
    let reduction_size = geometry.batch * geometry.output_height * geometry.output_width;
    let reduction = begin_counted_loop(func, ctx, "conv_weight", reduction_size);
    let batch = decode_coordinate(
        func,
        reduction.counter.clone(),
        geometry.output_height * geometry.output_width,
        geometry.batch,
    );
    let output_row = decode_coordinate(
        func,
        reduction.counter.clone(),
        geometry.output_width,
        geometry.output_height,
    );
    let output_col = decode_coordinate(func, reduction.counter.clone(), 1, geometry.output_width);
    let continue_label = next_label(ctx, "conv_weight_continue");
    let input_row = padded_input_coordinate(
        func,
        output_row.clone(),
        kernel_row.clone(),
        geometry.stride,
        geometry.padding,
        geometry.input_height,
        continue_label,
    );
    let input_col = padded_input_coordinate(
        func,
        output_col.clone(),
        kernel_col,
        geometry.stride,
        geometry.padding,
        geometry.input_width,
        continue_label,
    );
    let input_offset = flatten_nchw(
        func,
        batch.clone(),
        input_channel,
        input_row,
        input_col,
        geometry.input_channels,
        geometry.input_height,
        geometry.input_width,
    );
    let grad_offset = flatten_nchw(
        func,
        batch,
        output_channel,
        output_row,
        output_col,
        geometry.output_channels,
        geometry.output_height,
        geometry.output_width,
    );
    let input = load_param_f32_at(func, ctx, "input", input_offset);
    let grad = load_param_f32_at(func, ctx, "grad_output", grad_offset);
    accumulate_product(func, accumulator.clone(), input, grad);
    func.add_inst(Inst::Label(continue_label));
    end_counted_loop(func, reduction);
    store_param_f32_at(func, ctx, "output", tid, accumulator);
}

fn lower_max_pool2d<'a>(
    func: &mut Function<'a>,
    ctx: &mut LoweringContext<'a>,
    geometry: crate::tile::MaxPool2dGeometry,
) {
    let tid = global_linear_tid(func);
    let (batch, channel, output_row, output_col) = decode_nchw(
        func,
        tid.clone(),
        geometry.channels,
        geometry.output_height,
        geometry.output_width,
    );
    let maximum = func.add_f32_register();
    func.add_inst(Inst::mov_f32(
        maximum.clone(),
        Operand::imm_f32(f32::NEG_INFINITY),
    ));
    let reduction = begin_counted_loop(
        func,
        ctx,
        "max_pool",
        geometry.kernel_size * geometry.kernel_size,
    );
    let kernel_row = decode_coordinate(
        func,
        reduction.counter.clone(),
        geometry.kernel_size,
        geometry.kernel_size,
    );
    let kernel_col = decode_coordinate(func, reduction.counter.clone(), 1, geometry.kernel_size);
    let input_row = scaled_coordinate(func, output_row, geometry.stride, kernel_row);
    let input_col = scaled_coordinate(func, output_col, geometry.stride, kernel_col);
    let input_offset = flatten_nchw(
        func,
        batch,
        channel,
        input_row,
        input_col,
        geometry.channels,
        geometry.input_height,
        geometry.input_width,
    );
    let value = load_param_f32_at(func, ctx, "input", input_offset);
    func.add_inst(Inst::max_f32(maximum.clone(), maximum.clone(), value));
    end_counted_loop(func, reduction);
    store_param_f32_at(func, ctx, "output", tid, maximum);
}

fn lower_max_pool2d_backward<'a>(
    func: &mut Function<'a>,
    ctx: &mut LoweringContext<'a>,
    geometry: crate::tile::MaxPool2dGeometry,
) {
    let tid = global_linear_tid(func);
    let (batch, channel, input_row, input_col) = decode_nchw(
        func,
        tid.clone(),
        geometry.channels,
        geometry.input_height,
        geometry.input_width,
    );
    let input_value = load_param_f32_at(func, ctx, "input", tid.clone());
    let accumulator = func.add_f32_register();
    func.add_inst(Inst::mov_f32(accumulator.clone(), Operand::imm_f32(0.0)));
    let reduction = begin_counted_loop(
        func,
        ctx,
        "max_pool_backward",
        geometry.output_height * geometry.output_width,
    );
    let output_row = decode_coordinate(
        func,
        reduction.counter.clone(),
        geometry.output_width,
        geometry.output_height,
    );
    let output_col = decode_coordinate(func, reduction.counter.clone(), 1, geometry.output_width);
    let continue_label = next_label(ctx, "max_pool_backward_continue");
    branch_unless_window_contains(
        func,
        input_row,
        output_row.clone(),
        geometry.stride,
        geometry.kernel_size,
        continue_label,
    );
    branch_unless_window_contains(
        func,
        input_col,
        output_col.clone(),
        geometry.stride,
        geometry.kernel_size,
        continue_label,
    );
    let output_offset = flatten_nchw(
        func,
        batch,
        channel,
        output_row,
        output_col,
        geometry.channels,
        geometry.output_height,
        geometry.output_width,
    );
    let pooled = load_param_f32_at(func, ctx, "pooled", output_offset.clone());
    let not_maximum = func.add_predicate_register();
    func.add_inst(Inst::setp_ne_f32(
        not_maximum.clone(),
        input_value.clone(),
        pooled,
    ));
    func.add_inst(Inst::Bra {
        condition: not_maximum,
        target: continue_label,
    });
    let grad = load_param_f32_at(func, ctx, "grad_output", output_offset);
    func.add_inst(Inst::add_f32(
        accumulator.clone(),
        accumulator.clone(),
        grad,
    ));
    func.add_inst(Inst::Label(continue_label));
    end_counted_loop(func, reduction);
    store_param_f32_at(func, ctx, "output", tid, accumulator);
}

struct CountedLoop<'a> {
    counter: Operand<'a, U64>,
    start: &'a str,
    end: &'a str,
}

fn begin_counted_loop<'a>(
    func: &mut Function<'a>,
    ctx: &mut LoweringContext<'a>,
    prefix: &str,
    limit: usize,
) -> CountedLoop<'a> {
    let counter = func.add_u64_register();
    func.add_inst(Inst::mov_u64(counter.clone(), Operand::imm_u64(0)));
    let start = next_label(ctx, &format!("{prefix}_loop"));
    let end = next_label(ctx, &format!("{prefix}_end"));
    func.add_inst(Inst::Label(start));
    let done = func.add_predicate_register();
    func.add_inst(Inst::setp_ge_u64(
        done.clone(),
        counter.clone(),
        Operand::imm_u64(limit as u64),
    ));
    func.add_inst(Inst::Bra {
        condition: done,
        target: end,
    });
    CountedLoop {
        counter,
        start,
        end,
    }
}

fn end_counted_loop<'a>(func: &mut Function<'a>, state: CountedLoop<'a>) {
    func.add_inst(Inst::add_u64(
        state.counter.clone(),
        state.counter,
        Operand::imm_u64(1),
    ));
    func.add_inst(Inst::BraUni {
        target: state.start,
    });
    func.add_inst(Inst::Label(state.end));
}

fn next_label<'a>(ctx: &mut LoweringContext<'a>, prefix: &str) -> &'a str {
    let label = bumpalo::format!(in ctx.arena, "{}_{}", prefix, ctx.label_counter).into_bump_str();
    ctx.label_counter += 1;
    label
}

fn decode_nchw<'a>(
    func: &mut Function<'a>,
    linear: Operand<'a, U64>,
    channels: usize,
    height: usize,
    width: usize,
) -> (
    Operand<'a, U64>,
    Operand<'a, U64>,
    Operand<'a, U64>,
    Operand<'a, U64>,
) {
    let batch_stride = channels * height * width;
    (
        quotient_u64(func, linear.clone(), batch_stride),
        decode_coordinate(func, linear.clone(), height * width, channels),
        decode_coordinate(func, linear.clone(), width, height),
        decode_coordinate(func, linear, 1, width),
    )
}

#[allow(clippy::too_many_arguments)]
fn flatten_nchw<'a>(
    func: &mut Function<'a>,
    batch: Operand<'a, U64>,
    channel: Operand<'a, U64>,
    row: Operand<'a, U64>,
    col: Operand<'a, U64>,
    channels: usize,
    height: usize,
    width: usize,
) -> Operand<'a, U64> {
    let batch_offset = multiply_u64(func, batch, channels * height * width);
    let channel_offset = multiply_u64(func, channel, height * width);
    let row_offset = multiply_u64(func, row, width);
    let offset = func.add_u64_register();
    func.add_inst(Inst::add_u64(offset.clone(), batch_offset, channel_offset));
    func.add_inst(Inst::add_u64(offset.clone(), offset.clone(), row_offset));
    func.add_inst(Inst::add_u64(offset.clone(), offset.clone(), col));
    offset
}

fn padded_input_coordinate<'a>(
    func: &mut Function<'a>,
    output: Operand<'a, U64>,
    kernel: Operand<'a, U64>,
    stride: usize,
    padding: usize,
    extent: usize,
    continue_label: &'a str,
) -> Operand<'a, U64> {
    let raw = scaled_coordinate(func, output, stride, kernel);
    branch_if_lt(func, raw.clone(), padding, continue_label);
    let coordinate = func.add_u64_register();
    func.add_inst(Inst::sub_u64(
        coordinate.clone(),
        raw,
        Operand::imm_u64(padding as u64),
    ));
    branch_if_ge(func, coordinate.clone(), extent, continue_label);
    coordinate
}

fn transposed_output_coordinate<'a>(
    func: &mut Function<'a>,
    input: Operand<'a, U64>,
    kernel: Operand<'a, U64>,
    stride: usize,
    padding: usize,
    extent: usize,
    continue_label: &'a str,
) -> Operand<'a, U64> {
    let padded = func.add_u64_register();
    func.add_inst(Inst::add_u64(
        padded.clone(),
        input,
        Operand::imm_u64(padding as u64),
    ));
    let before_kernel = func.add_predicate_register();
    func.add_inst(Inst::setp_lt_u64(
        before_kernel.clone(),
        padded.clone(),
        kernel.clone(),
    ));
    func.add_inst(Inst::Bra {
        condition: before_kernel,
        target: continue_label,
    });
    let numerator = func.add_u64_register();
    func.add_inst(Inst::sub_u64(numerator.clone(), padded, kernel));
    let coordinate = quotient_u64(func, numerator.clone(), stride);
    let reconstructed = multiply_u64(func, coordinate.clone(), stride);
    let not_divisible = func.add_predicate_register();
    func.add_inst(Inst::setp_ne_u64(
        not_divisible.clone(),
        reconstructed,
        numerator,
    ));
    func.add_inst(Inst::Bra {
        condition: not_divisible,
        target: continue_label,
    });
    branch_if_ge(func, coordinate.clone(), extent, continue_label);
    coordinate
}

fn branch_unless_window_contains<'a>(
    func: &mut Function<'a>,
    input: Operand<'a, U64>,
    output: Operand<'a, U64>,
    stride: usize,
    kernel_size: usize,
    continue_label: &'a str,
) {
    let start = multiply_u64(func, output, stride);
    let before = func.add_predicate_register();
    func.add_inst(Inst::setp_lt_u64(
        before.clone(),
        input.clone(),
        start.clone(),
    ));
    func.add_inst(Inst::Bra {
        condition: before,
        target: continue_label,
    });
    let end = func.add_u64_register();
    func.add_inst(Inst::add_u64(
        end.clone(),
        start,
        Operand::imm_u64(kernel_size as u64),
    ));
    let after = func.add_predicate_register();
    func.add_inst(Inst::setp_ge_u64(after.clone(), input, end));
    func.add_inst(Inst::Bra {
        condition: after,
        target: continue_label,
    });
}

fn scaled_coordinate<'a>(
    func: &mut Function<'a>,
    coordinate: Operand<'a, U64>,
    scale: usize,
    offset: Operand<'a, U64>,
) -> Operand<'a, U64> {
    let scaled = multiply_u64(func, coordinate, scale);
    let result = func.add_u64_register();
    func.add_inst(Inst::add_u64(result.clone(), scaled, offset));
    result
}

fn multiply_u64<'a>(
    func: &mut Function<'a>,
    value: Operand<'a, U64>,
    multiplier: usize,
) -> Operand<'a, U64> {
    let result = func.add_u64_register();
    func.add_inst(Inst::mul_u64(
        result.clone(),
        value,
        Operand::imm_u64(multiplier as u64),
    ));
    result
}

fn quotient_u64<'a>(
    func: &mut Function<'a>,
    value: Operand<'a, U64>,
    divisor: usize,
) -> Operand<'a, U64> {
    let result = func.add_u64_register();
    func.add_inst(Inst::div_u64(
        result.clone(),
        value,
        Operand::imm_u64(divisor as u64),
    ));
    result
}

fn branch_if_lt<'a>(
    func: &mut Function<'a>,
    value: Operand<'a, U64>,
    limit: usize,
    target: &'a str,
) {
    let predicate = func.add_predicate_register();
    func.add_inst(Inst::setp_lt_u64(
        predicate.clone(),
        value,
        Operand::imm_u64(limit as u64),
    ));
    func.add_inst(Inst::Bra {
        condition: predicate,
        target,
    });
}

fn branch_if_ge<'a>(
    func: &mut Function<'a>,
    value: Operand<'a, U64>,
    limit: usize,
    target: &'a str,
) {
    let predicate = func.add_predicate_register();
    func.add_inst(Inst::setp_ge_u64(
        predicate.clone(),
        value,
        Operand::imm_u64(limit as u64),
    ));
    func.add_inst(Inst::Bra {
        condition: predicate,
        target,
    });
}

fn accumulate_product<'a>(
    func: &mut Function<'a>,
    accumulator: Operand<'a, F32>,
    left: Operand<'a, F32>,
    right: Operand<'a, F32>,
) {
    let product = func.add_f32_register();
    func.add_inst(Inst::mul_f32(product.clone(), left, right));
    func.add_inst(Inst::add_f32(accumulator.clone(), accumulator, product));
}

fn lower_tf32_matmul<'a>(
    func: &mut Function<'a>,
    ctx: &mut LoweringContext<'a>,
    dest: TileVar,
    a: TileVar,
    b: TileVar,
) {
    let accumulators = ctx
        .fragment_regs
        .get(&dest)
        .expect("TF32 MatMul destination is not a fragment")
        .clone();
    let a_shared = ctx.reg_to_shared.get(&a).copied().unwrap_or(a);
    let b_shared = ctx.reg_to_shared.get(&b).copied().unwrap_or(b);
    let a_ptr = ctx
        .shared_mem_ptrs
        .get(&a_shared)
        .expect("TF32 MatMul operand A is not in shared memory")
        .clone();
    let b_ptr = ctx
        .shared_mem_ptrs
        .get(&b_shared)
        .expect("TF32 MatMul operand B is not in shared memory")
        .clone();
    let c_shared = ctx
        .reg_to_shared
        .get(&dest)
        .expect("TF32 MatMul destination has no shared-memory backing");
    let c_ptr = ctx
        .shared_mem_ptrs
        .get(c_shared)
        .expect("TF32 MatMul output is not in shared memory")
        .clone();

    // The 16x16 staging block contains eight warps. Warp zero owns the complete
    // output fragment while every thread still participates in staging/barriers.
    let thread_y = func.add_u64_register();
    func.add_inst(Inst::convert_u64_u32(
        thread_y.clone(),
        super::instructions::THREAD_ID.y.clone(),
    ));
    let inactive = func.add_predicate_register();
    func.add_inst(Inst::setp_ge_u64(
        inactive.clone(),
        thread_y,
        Operand::imm_u64(2),
    ));
    let done =
        bumpalo::format!(in ctx.arena, "tf32_matmul_done_{}", ctx.label_counter).into_bump_str();
    ctx.label_counter += 1;
    func.add_inst(Inst::Bra {
        condition: inactive,
        target: done,
    });

    for k_offset in [0_u64, 8] {
        let a_address = shared_address_with_byte_offset(func, a_ptr.clone(), k_offset * 4);
        let b_address = shared_address_with_byte_offset(func, b_ptr.clone(), k_offset * 16 * 4);
        let a_fragments: Vec<Operand<'a, B32>> = (0..4).map(|_| func.add_b32_register()).collect();
        let b_fragments: Vec<Operand<'a, B32>> = (0..4).map(|_| func.add_b32_register()).collect();
        func.add_inst(Inst::wmma_load_a(
            a_fragments.clone(),
            a_address,
            Operand::imm_i32(16),
        ));
        func.add_inst(Inst::wmma_load_b(
            b_fragments.clone(),
            b_address,
            Operand::imm_i32(16),
        ));
        func.add_inst(Inst::wmma_mma(
            accumulators.clone(),
            a_fragments,
            b_fragments,
            accumulators.clone(),
        ));
    }
    func.add_inst(Inst::wmma_store(c_ptr, accumulators, Operand::imm_i32(16)));
    func.add_inst(Inst::Label(done));
}

fn shared_address_with_byte_offset<'a>(
    func: &mut Function<'a>,
    base: Operand<'a, U64>,
    byte_offset: u64,
) -> Operand<'a, U64> {
    if byte_offset == 0 {
        return base;
    }
    let address = func.add_u64_register();
    func.add_inst(Inst::add_u64(
        address.clone(),
        base,
        Operand::imm_u64(byte_offset),
    ));
    address
}

fn global_linear_tid<'a>(func: &mut Function<'a>) -> Operand<'a, U64> {
    let block = func.add_u64_register();
    let block_dim = func.add_u64_register();
    let thread = func.add_u64_register();
    func.add_inst(Inst::convert_u64_u32(
        block.clone(),
        super::instructions::BLOCK_ID.x.clone(),
    ));
    func.add_inst(Inst::convert_u64_u32(
        block_dim.clone(),
        super::instructions::BLOCK_DIM.x.clone(),
    ));
    func.add_inst(Inst::convert_u64_u32(
        thread.clone(),
        super::instructions::THREAD_ID.x.clone(),
    ));
    let block_offset = func.add_u64_register();
    func.add_inst(Inst::mul_u64(block_offset.clone(), block, block_dim));
    let tid = func.add_u64_register();
    func.add_inst(Inst::add_u64(tid.clone(), block_offset, thread));
    tid
}

fn shared_thread_address<'a>(
    func: &mut Function<'a>,
    ctx: &LoweringContext<'a>,
    tile: TileVar,
) -> Operand<'a, U64> {
    let row = func.add_u64_register();
    let column = func.add_u64_register();
    func.add_inst(Inst::convert_u64_u32(
        row.clone(),
        super::instructions::THREAD_ID.y.clone(),
    ));
    func.add_inst(Inst::convert_u64_u32(
        column.clone(),
        super::instructions::THREAD_ID.x.clone(),
    ));
    let shared = ctx
        .shared_mem_ptrs
        .get(&tile)
        .expect("Shared tile pointer not found")
        .clone();
    let (_, columns) = ctx
        .tile_dims
        .get(&tile)
        .expect("Shared tile dimensions not found");
    let row_offset = func.add_u64_register();
    func.add_inst(Inst::mul_u64(
        row_offset.clone(),
        row,
        Operand::imm_u64(*columns as u64),
    ));
    let element_offset = func.add_u64_register();
    func.add_inst(Inst::add_u64(element_offset.clone(), row_offset, column));
    let byte_offset = func.add_u64_register();
    func.add_inst(Inst::mul_u64(
        byte_offset.clone(),
        element_offset,
        Operand::imm_u64(4),
    ));
    let address = func.add_u64_register();
    func.add_inst(Inst::add_u64(address.clone(), shared, byte_offset));
    address
}

fn global_f32_address<'a>(
    func: &mut Function<'a>,
    base: Operand<'a, U64>,
    element_offset: Operand<'a, U64>,
) -> Operand<'a, U64> {
    let byte_offset = func.add_u64_register();
    func.add_inst(Inst::mul_u64(
        byte_offset.clone(),
        element_offset,
        Operand::imm_u64(4),
    ));
    let address = func.add_u64_register();
    func.add_inst(Inst::add_u64(address.clone(), base, byte_offset));
    address
}

fn load_param_f32_at<'a>(
    func: &mut Function<'a>,
    ctx: &LoweringContext<'a>,
    parameter: &str,
    element_offset: Operand<'a, U64>,
) -> Operand<'a, F32> {
    let base = ctx
        .param_ptrs
        .get(parameter)
        .unwrap_or_else(|| panic!("Parameter {parameter} not found"))
        .clone();
    let address = global_f32_address(func, base, element_offset);
    let value = func.add_f32_register();
    func.add_inst(Inst::load_global_scalar_f32(value.clone(), address));
    value
}

fn store_param_f32_at<'a>(
    func: &mut Function<'a>,
    ctx: &LoweringContext<'a>,
    parameter: &str,
    element_offset: Operand<'a, U64>,
    value: Operand<'a, F32>,
) {
    let base = ctx
        .param_ptrs
        .get(parameter)
        .unwrap_or_else(|| panic!("Parameter {parameter} not found"))
        .clone();
    let address = global_f32_address(func, base, element_offset);
    func.add_inst(Inst::store_global_scalar_f32(address, value));
}

fn emit_exp<'a>(func: &mut Function<'a>, value: Operand<'a, F32>) -> Operand<'a, F32> {
    let scaled = func.add_f32_register();
    func.add_inst(Inst::mul_f32(
        scaled.clone(),
        value,
        Operand::imm_f32(std::f32::consts::LOG2_E),
    ));
    let exponential = func.add_f32_register();
    func.add_inst(Inst::ex2_f32(exponential.clone(), scaled));
    exponential
}

fn emit_log<'a>(func: &mut Function<'a>, value: Operand<'a, F32>) -> Operand<'a, F32> {
    let logarithm_base_two = func.add_f32_register();
    func.add_inst(Inst::lg2_f32(logarithm_base_two.clone(), value));
    let logarithm = func.add_f32_register();
    func.add_inst(Inst::mul_f32(
        logarithm.clone(),
        logarithm_base_two,
        Operand::imm_f32(std::f32::consts::LN_2),
    ));
    logarithm
}

fn emit_row_softmax_stats<'a>(
    func: &mut Function<'a>,
    ctx: &mut LoweringContext<'a>,
    row_start: Operand<'a, U64>,
    vocabulary: usize,
) -> (Operand<'a, F32>, Operand<'a, F32>) {
    let maximum = func.add_f32_register();
    func.add_inst(Inst::mov_f32(
        maximum.clone(),
        Operand::imm_f32(f32::NEG_INFINITY),
    ));
    let maximum_column = func.add_u64_register();
    func.add_inst(Inst::mov_u64(maximum_column.clone(), Operand::imm_u64(0)));
    let maximum_loop =
        bumpalo::format!(in ctx.arena, "ce_max_{}", ctx.label_counter).into_bump_str();
    let maximum_end =
        bumpalo::format!(in ctx.arena, "ce_max_end_{}", ctx.label_counter).into_bump_str();
    ctx.label_counter += 1;
    func.add_inst(Inst::Label(maximum_loop));
    let maximum_done = func.add_predicate_register();
    func.add_inst(Inst::setp_ge_u64(
        maximum_done.clone(),
        maximum_column.clone(),
        Operand::imm_u64(vocabulary as u64),
    ));
    func.add_inst(Inst::Bra {
        condition: maximum_done,
        target: maximum_end,
    });
    let offset = func.add_u64_register();
    func.add_inst(Inst::add_u64(
        offset.clone(),
        row_start.clone(),
        maximum_column.clone(),
    ));
    let value = load_param_f32_at(func, ctx, "logits", offset);
    func.add_inst(Inst::max_f32(maximum.clone(), maximum.clone(), value));
    func.add_inst(Inst::add_u64(
        maximum_column.clone(),
        maximum_column.clone(),
        Operand::imm_u64(1),
    ));
    func.add_inst(Inst::BraUni {
        target: maximum_loop,
    });
    func.add_inst(Inst::Label(maximum_end));

    let exponential_sum = func.add_f32_register();
    func.add_inst(Inst::mov_f32(
        exponential_sum.clone(),
        Operand::imm_f32(0.0),
    ));
    let sum_column = func.add_u64_register();
    func.add_inst(Inst::mov_u64(sum_column.clone(), Operand::imm_u64(0)));
    let sum_loop = bumpalo::format!(in ctx.arena, "ce_sum_{}", ctx.label_counter).into_bump_str();
    let sum_end =
        bumpalo::format!(in ctx.arena, "ce_sum_end_{}", ctx.label_counter).into_bump_str();
    ctx.label_counter += 1;
    func.add_inst(Inst::Label(sum_loop));
    let sum_done = func.add_predicate_register();
    func.add_inst(Inst::setp_ge_u64(
        sum_done.clone(),
        sum_column.clone(),
        Operand::imm_u64(vocabulary as u64),
    ));
    func.add_inst(Inst::Bra {
        condition: sum_done,
        target: sum_end,
    });
    let offset = func.add_u64_register();
    func.add_inst(Inst::add_u64(offset.clone(), row_start, sum_column.clone()));
    let value = load_param_f32_at(func, ctx, "logits", offset);
    let shifted = func.add_f32_register();
    func.add_inst(Inst::sub_f32(shifted.clone(), value, maximum.clone()));
    let exponential = emit_exp(func, shifted);
    func.add_inst(Inst::add_f32(
        exponential_sum.clone(),
        exponential_sum.clone(),
        exponential,
    ));
    func.add_inst(Inst::add_u64(
        sum_column.clone(),
        sum_column,
        Operand::imm_u64(1),
    ));
    func.add_inst(Inst::BraUni { target: sum_loop });
    func.add_inst(Inst::Label(sum_end));
    (maximum, exponential_sum)
}

fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for dimension in (0..shape.len().saturating_sub(1)).rev() {
        strides[dimension] = strides[dimension + 1] * shape[dimension + 1];
    }
    strides
}

fn decode_coordinate<'a>(
    func: &mut Function<'a>,
    linear: Operand<'a, U64>,
    stride: usize,
    dimension: usize,
) -> Operand<'a, U64> {
    let quotient = func.add_u64_register();
    func.add_inst(Inst::div_u64(
        quotient.clone(),
        linear,
        Operand::imm_u64(stride as u64),
    ));
    let groups = func.add_u64_register();
    func.add_inst(Inst::div_u64(
        groups.clone(),
        quotient.clone(),
        Operand::imm_u64(dimension as u64),
    ));
    let consumed = func.add_u64_register();
    func.add_inst(Inst::mul_u64(
        consumed.clone(),
        groups,
        Operand::imm_u64(dimension as u64),
    ));
    let coordinate = func.add_u64_register();
    func.add_inst(Inst::sub_u64(coordinate.clone(), quotient, consumed));
    coordinate
}

fn load_f32_at<'a>(
    func: &mut Function<'a>,
    destination: Operand<'a, F32>,
    base: Operand<'a, U64>,
    element_offset: Operand<'a, U64>,
) {
    let byte_offset = func.add_u64_register();
    func.add_inst(Inst::mul_u64(
        byte_offset.clone(),
        element_offset,
        Operand::imm_u64(4),
    ));
    let address = func.add_u64_register();
    func.add_inst(Inst::add_u64(address.clone(), base, byte_offset));
    func.add_inst(Inst::load_global_scalar_f32(destination, address));
}

/// Lower an expression to a U64 operand (for address calculations)
fn lower_expr<'a>(
    func: &mut Function<'a>,
    ctx: &LoweringContext<'a>,
    expr: &Expr,
) -> Operand<'a, U64> {
    match expr {
        Expr::Const(val) => Operand::imm_u64(*val as u64),
        Expr::Var(name) => {
            // Check if it's a loop variable
            if let Some(loop_var_i32) = ctx.loop_vars.get(name) {
                // Convert from i32 to u64
                let result = func.add_u64_register();
                func.add_inst(Inst::ConvertU64I32(super::instructions::ConvertInst::new(
                    result.clone(),
                    loop_var_i32.clone(),
                )));
                result
            } else {
                // Variable reference - would need to be tracked in context
                let reg_name = bumpalo::format!(in ctx.arena, "%{}", name).into_bump_str();
                Operand::reg(reg_name)
            }
        }
        Expr::BlockIdx(dim) => {
            use crate::tile::Dim;
            let block_idx = match dim {
                Dim::X => super::instructions::BLOCK_ID.x.clone(),
                Dim::Y => super::instructions::BLOCK_ID.y.clone(),
                Dim::Z => super::instructions::BLOCK_ID.z.clone(),
            };
            // Convert from u32 to u64
            let result = func.add_u64_register();
            func.add_inst(Inst::convert_u64_u32(result.clone(), block_idx));
            result
        }
        Expr::ThreadIdx(dim) => {
            use crate::tile::Dim;
            let thread_idx = match dim {
                Dim::X => super::instructions::THREAD_ID.x.clone(),
                Dim::Y => super::instructions::THREAD_ID.y.clone(),
                Dim::Z => super::instructions::THREAD_ID.z.clone(),
            };
            // Convert from u32 to u64
            let result = func.add_u64_register();
            func.add_inst(Inst::convert_u64_u32(result.clone(), thread_idx));
            result
        }
        Expr::BlockDim(dim) => {
            use crate::tile::Dim;
            let block_dim = match dim {
                Dim::X => super::instructions::BLOCK_DIM.x.clone(),
                Dim::Y => super::instructions::BLOCK_DIM.y.clone(),
                Dim::Z => super::instructions::BLOCK_DIM.z.clone(),
            };
            // Convert from u32 to u64
            let result = func.add_u64_register();
            func.add_inst(Inst::convert_u64_u32(result.clone(), block_dim));
            result
        }
        Expr::Mul(a, b) => {
            let a_val = lower_expr(func, ctx, a);
            let b_val = lower_expr(func, ctx, b);
            let result = func.add_u64_register();
            func.add_inst(Inst::mul_u64(result.clone(), a_val, b_val));
            result
        }
        Expr::Add(a, b) => {
            let a_val = lower_expr(func, ctx, a);
            let b_val = lower_expr(func, ctx, b);
            let result = func.add_u64_register();
            func.add_inst(Inst::add_u64(result.clone(), a_val, b_val));
            result
        }
    }
}

/// Lower an expression to an I32 operand (for loop counters)
fn lower_expr_i32<'a>(
    func: &mut Function<'a>,
    ctx: &LoweringContext<'a>,
    expr: &Expr,
) -> Operand<'a, I32> {
    match expr {
        Expr::Const(val) => Operand::imm_i32(*val as i32),
        Expr::Var(name) => {
            // Check if it's a loop variable
            if let Some(loop_var) = ctx.loop_vars.get(name) {
                loop_var.clone()
            } else {
                let reg_name = bumpalo::format!(in ctx.arena, "%{}", name).into_bump_str();
                Operand::reg(reg_name)
            }
        }
        _ => {
            // For complex expressions, compute as u64 then convert
            let u64_val = lower_expr(func, ctx, expr);
            let result = func.add_i32_register();
            func.add_inst(Inst::convert_i32_u64(result.clone(), u64_val));
            result
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::TensorGraph;
    use crate::tensor::TensorExpr;
    use crate::tile::{Block, DType, KernelParam, MemorySpace, Stmt, TileIR};

    // Helper function to convert TileIR to Function for testing
    // Returns arena wrapped in Box to ensure stable address
    fn tile_ir_into_function(tile_ir: TileIR) -> (Box<bumpalo::Bump>, Function<'static>) {
        let arena = Box::new(bumpalo::Bump::new());
        // SAFETY: We're using 'static lifetime and keeping the arena alive via Box
        // The arena address is stable since it's heap-allocated
        let arena_ref: &'static bumpalo::Bump =
            unsafe { &*(arena.as_ref() as *const bumpalo::Bump) };
        let func = tile_ir_to_function(tile_ir, arena_ref, 0);
        (arena, func)
    }

    #[test]
    fn production_matmul_lowers_to_tf32_wmma() {
        let a = TensorExpr::constant(vec![0.25; 32 * 32], vec![32, 32]);
        let b = TensorExpr::constant(vec![0.5; 32 * 32], vec![32, 32]);
        let graph: TensorGraph<f32> = a.matmul(b).into();
        let ptx_graph: PtxGraph = TileGraph::from(graph).into();
        let source = ptx_graph.module_source();

        assert!(source.contains("cvt.rna.tf32.f32"));
        assert!(source.contains("wmma.mma.sync.aligned.m16n16k8"));
        assert!(source.contains("wmma.store.d.sync.aligned.m16n16k8"));
    }

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
        let (_arena, func) = tile_ir_into_function(tile_ir);

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

        let (_arena, func) = tile_ir_into_function(tile_ir);

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

        let (_arena, func) = tile_ir_into_function(tile_ir);

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

        let (_arena, func) = tile_ir_into_function(tile_ir);

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

        let (_arena, func) = tile_ir_into_function(tile_ir);

        // Check that shared memory was allocated
        assert_eq!(func.shared_memory.len(), 1);
        assert_eq!(func.shared_memory[0].0, "shared");
        assert_eq!(func.shared_memory[0].1, 4096);

        // Check that u64 registers were allocated (base_ptr and tile_ptr)
        assert_eq!(func.i64_registers.len(), 2);
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

        let (_arena, func) = tile_ir_into_function(tile_ir);

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

        let (_arena, func) = tile_ir_into_function(tile_ir);

        // Gt uses setp and selp
        let has_setp = func
            .body
            .iter()
            .any(|inst| matches!(inst, Inst::SetpF32(_)));
        let has_selp = func
            .body
            .iter()
            .any(|inst| matches!(inst, Inst::SelpF32(_)));

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

        let (_arena, func) = tile_ir_into_function(tile_ir);

        // This should not panic
        let ptx_str = format!("{}", func);

        // Basic sanity checks on the output
        assert!(ptx_str.contains(".visible .entry simple_kernel"));
        assert!(ptx_str.contains("ret;"));
        assert!(ptx_str.contains(".param .u64 input")); // Parameters are pointers in PTX

        // Print the generated PTX for inspection
        println!(
            "\n========== Generated PTX ==========\n{}\n===================================",
            ptx_str
        );
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

        let (_arena, func) = tile_ir_into_function(tile_ir);

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
