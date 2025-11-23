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

        // Create lowering context to track tile variable mappings
        let mut ctx = LoweringContext::new();

        // Add parameters as u64 pointers and store their loaded addresses
        for param in &tile_ir.params {
            // All parameters are pointers, so they're u64 in PTX
            let ptr = func.add_global_ptr_param(&param.name);
            ctx.param_ptrs.insert(param.name.clone(), ptr);
        }

        // Allocate shared memory if needed
        if tile_ir.shared_mem_bytes > 0 {
            func.add_shared_memory("shared", tile_ir.shared_mem_bytes);
        }

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
    /// Tracks loaded parameter pointers (global addresses)
    param_ptrs: HashMap<String, Operand<U64>>,
    /// Maps loop variable names to their i32 register operands
    loop_vars: HashMap<String, Operand<I32>>,
    /// Current offset into shared memory for allocation
    shared_mem_offset: usize,
    /// Maps TileVar to its dimensions (rows, cols)
    tile_dims: HashMap<TileVar, (usize, usize)>,
    /// Maps register tile vars to their shared memory source (for LoadSharedToReg)
    reg_to_shared: HashMap<TileVar, TileVar>,
}

impl LoweringContext {
    fn new() -> Self {
        Self {
            tile_to_reg: HashMap::new(),
            shared_mem_ptrs: HashMap::new(),
            param_ptrs: HashMap::new(),
            loop_vars: HashMap::new(),
            shared_mem_offset: 0,
            tile_dims: HashMap::new(),
            reg_to_shared: HashMap::new(),
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
            dtype,
            rows,
            cols,
        } => {
            // Track tile dimensions
            ctx.tile_dims.insert(*var, (*rows, *cols));
            
            use crate::tile::MemorySpace;
            match space {
                MemorySpace::Register => {
                    // Allocate register for this tile variable
                    ctx.get_or_alloc_reg(func, *var);
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
            func.add_inst(Inst::add_u64(elem_offset.clone(), row_offset_reg, col_offset_reg));
            
            // Byte offset = element offset * sizeof(f32) = element offset * 4
            let byte_offset = func.add_u64_register();
            func.add_inst(Inst::mul_u64(byte_offset.clone(), elem_offset, Operand::imm_u64(4)));

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
                func.add_inst(Inst::add_i32(
                    elem_idx.clone(),
                    row_offset,
                    tid_x_i32,
                ));
                
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
                func.add_inst(Inst::add_u64(
                    final_addr.clone(),
                    shared_ptr,
                    byte_offset,
                ));
                
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
            func.add_inst(Inst::add_u64(elem_offset.clone(), row_offset_reg, col_offset_reg));
            
            // Byte offset = element offset * sizeof(f32) = element offset * 4
            let byte_offset = func.add_u64_register();
            func.add_inst(Inst::mul_u64(byte_offset.clone(), elem_offset, Operand::imm_u64(4)));

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
            // Matrix multiply: each thread computes one output element
            // Thread at (ty, tx) computes C[ty][tx] = sum over k of A[ty][k] * B[k][tx]
            
            let dest_reg = ctx.get_or_alloc_reg(func, *dest);
            
            // Resolve register tiles to their shared memory sources
            let a_smem = ctx.reg_to_shared.get(a).copied().unwrap_or(*a);
            let b_smem = ctx.reg_to_shared.get(b).copied().unwrap_or(*b);
            
            // Get shared memory pointers for A and B tiles
            let a_ptr = ctx.shared_mem_ptrs.get(&a_smem)
                .expect("MatMul operand A not in shared memory").clone();
            let b_ptr = ctx.shared_mem_ptrs.get(&b_smem)
                .expect("MatMul operand B not in shared memory").clone();
            
            // Get tile dimensions
            let (_a_rows, a_cols) = ctx.tile_dims.get(&a_smem)
                .expect("Tile dimensions not found for A");
            let (_b_rows, b_cols) = ctx.tile_dims.get(&b_smem)
                .expect("Tile dimensions not found for B");
            let tile_k = *a_cols; // K dimension of the tile
            
            // Get threadIdx.y and threadIdx.x (this thread's position in the output tile)
            let tid_y = func.add_u32_register();
            let tid_x = func.add_u32_register();
            func.add_inst(Inst::mov_u32(tid_y.clone(), super::instructions::THREAD_ID.y.clone()));
            func.add_inst(Inst::mov_u32(tid_x.clone(), super::instructions::THREAD_ID.x.clone()));
            
            // Loop over k dimension: for (k = 0; k < tile_k; k++)
            for k in 0..tile_k {
                // Load A[threadIdx.y][k] from shared memory
                // Offset in elements = threadIdx.y * a_cols + k
                let a_offset = func.add_u64_register();
                let ty_u64 = func.add_u64_register();
                func.add_inst(Inst::convert_u64_u32(ty_u64.clone(), tid_y.clone()));
                func.add_inst(Inst::mul_u64(a_offset.clone(), ty_u64, Operand::imm_u64(*a_cols as u64)));
                let a_k_offset = func.add_u64_register();
                func.add_inst(Inst::add_u64(a_k_offset.clone(), a_offset, Operand::imm_u64(k as u64)));
                
                // Byte offset = element offset * 4
                let a_byte_offset = func.add_u64_register();
                func.add_inst(Inst::mul_u64(a_byte_offset.clone(), a_k_offset, Operand::imm_u64(4)));
                
                // Address = base + byte_offset
                let a_addr = func.add_u64_register();
                func.add_inst(Inst::add_u64(a_addr.clone(), a_ptr.clone(), a_byte_offset));
                
                // Load A element
                let a_val = func.add_f32_register();
                func.add_inst(Inst::load_shared_scalar_f32(a_val.clone(), a_addr));
                
                // Load B[k][threadIdx.x] from shared memory
                // Offset in elements = k * b_cols + threadIdx.x
                let b_offset = func.add_u64_register();
                func.add_inst(Inst::mul_u64(b_offset.clone(), Operand::imm_u64(k as u64), Operand::imm_u64(*b_cols as u64)));
                let tx_u64 = func.add_u64_register();
                func.add_inst(Inst::convert_u64_u32(tx_u64.clone(), tid_x.clone()));
                let b_kx_offset = func.add_u64_register();
                func.add_inst(Inst::add_u64(b_kx_offset.clone(), b_offset, tx_u64));
                
                // Byte offset = element offset * 4
                let b_byte_offset = func.add_u64_register();
                func.add_inst(Inst::mul_u64(b_byte_offset.clone(), b_kx_offset, Operand::imm_u64(4)));
                
                // Address = base + byte_offset
                let b_addr = func.add_u64_register();
                func.add_inst(Inst::add_u64(b_addr.clone(), b_ptr.clone(), b_byte_offset));
                
                // Load B element
                let b_val = func.add_f32_register();
                func.add_inst(Inst::load_shared_scalar_f32(b_val.clone(), b_addr));
                
                // Multiply and accumulate: dest += a_val * b_val
                let prod = func.add_f32_register();
                func.add_inst(Inst::mul_f32(prod.clone(), a_val, b_val));
                func.add_inst(Inst::add_f32(dest_reg.clone(), dest_reg.clone(), prod));
            }
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
        Stmt::Transpose { dest, src: _, input_shape, output_shape } => {
            // Implement 2D transpose: output[i,j] = input[j,i]
            // For input shape [rows, cols] -> output shape [cols, rows]
            // Each thread computes one output element
            
            assert_eq!(input_shape.len(), 2, "Transpose only supports 2D tensors");
            assert_eq!(output_shape.len(), 2, "Transpose only supports 2D tensors");
            
            let rows = input_shape[0]; // Input rows
            let cols = input_shape[1]; // Input cols
            
            assert_eq!(output_shape[0], cols, "Output rows should equal input cols");
            assert_eq!(output_shape[1], rows, "Output cols should equal input rows");
            
            let dest_reg = ctx.get_or_alloc_reg(func, *dest);
            
            // Get parameter pointers for direct memory access
            let src_ptr = ctx.param_ptrs.get("input")
                .expect("Input parameter not found").clone();
            
            // Get thread index (which output element this thread computes)
            let tid = func.add_u64_register();
            func.add_inst(Inst::convert_u64_u32(tid.clone(), super::instructions::THREAD_ID.x.clone()));
            
            // Calculate output position (out_row, out_col) from thread id
            // tid = out_row * output_cols + out_col
            // out_row = tid / output_cols
            // out_col = tid % output_cols
            
            let out_row = func.add_u64_register();
            func.add_inst(Inst::div_u64(
                out_row.clone(),
                tid.clone(),
                Operand::imm_u64(rows as u64), // output_cols = input_rows
            ));
            
            let temp = func.add_u64_register();
            func.add_inst(Inst::mul_u64(
                temp.clone(),
                out_row.clone(),
                Operand::imm_u64(rows as u64),
            ));
            
            let out_col = func.add_u64_register();
            func.add_inst(Inst::sub_u64(
                out_col.clone(),
                tid.clone(),
                temp,
            ));
            
            // Map to input position: input[out_col, out_row]
            // input_offset = out_col * input_cols + out_row
            
            let input_row_offset = func.add_u64_register();
            func.add_inst(Inst::mul_u64(
                input_row_offset.clone(),
                out_col.clone(),
                Operand::imm_u64(cols as u64), // input_cols
            ));
            
            let input_offset = func.add_u64_register();
            func.add_inst(Inst::add_u64(
                input_offset.clone(),
                input_row_offset,
                out_row,
            ));
            
            // Load from input[input_offset]
            let byte_offset = func.add_u64_register();
            func.add_inst(Inst::mul_u64(
                byte_offset.clone(),
                input_offset,
                Operand::imm_u64(4), // sizeof(f32)
            ));
            
            let input_addr = func.add_u64_register();
            func.add_inst(Inst::add_u64(
                input_addr.clone(),
                src_ptr,
                byte_offset,
            ));
            
            func.add_inst(Inst::load_global_scalar_f32(dest_reg, input_addr));
        }
        Stmt::BroadcastAxis {
            dest,
            src,
            axis,
            input_shape,
            output_shape,
        } => {
            // Each thread handles one output element
            // Broadcast means copying values from input to multiple output positions
            // Example: input [2,1], output [2,3], axis=1
            //   output[i,j] = input[i, 0] for all j
            
            let dest_reg = ctx.get_or_alloc_reg(func, *dest);
            
            // Get parameter pointers for direct access
            let src_ptr = ctx.param_ptrs.get("input")
                .expect("Input parameter not found").clone();
            
            // Get thread index (which output element this thread computes)
            let tid = func.add_u64_register();
            func.add_inst(Inst::convert_u64_u32(tid.clone(), super::instructions::THREAD_ID.x.clone()));
            
            // Calculate strides for input tensor
            let input_stride: Vec<usize> = {
                let mut strides = vec![1; input_shape.len()];
                for i in (0..input_shape.len()-1).rev() {
                    strides[i] = strides[i + 1] * input_shape[i + 1];
                }
                strides
            };
            
            // Calculate input offset based on output thread index
            // For broadcasting along axis 1 with shapes [2,1] -> [2,3]:
            //   output[i,j] maps to input[i, 0]
            //   tid represents output position in row-major order
            //   row = tid / output_cols
            
            let input_offset = if output_shape.len() == 2 && *axis == 1 {
                // Common case: 2D broadcast along last axis
                // tid = row * output_cols + col
                // input_offset = row * input_stride[0] (since input has size 1 along axis 1)
                
                let row = func.add_u64_register();
                func.add_inst(Inst::div_u64(
                    row.clone(),
                    tid.clone(),
                    Operand::imm_u64(output_shape[1] as u64),
                ));
                
                let offset = func.add_u64_register();
                func.add_inst(Inst::mul_u64(
                    offset.clone(),
                    row,
                    Operand::imm_u64(input_stride[0] as u64),
                ));
                
                offset
            } else if output_shape.len() == 2 && *axis == 0 {
                // 2D broadcast along first axis
                // output[i,j] maps to input[0, j]
                // col = tid % output_shape[1] = tid - (tid / output_shape[1]) * output_shape[1]
                
                let div_result = func.add_u64_register();
                func.add_inst(Inst::div_u64(
                    div_result.clone(),
                    tid.clone(),
                    Operand::imm_u64(output_shape[1] as u64),
                ));
                
                let mul_result = func.add_u64_register();
                func.add_inst(Inst::mul_u64(
                    mul_result.clone(),
                    div_result,
                    Operand::imm_u64(output_shape[1] as u64),
                ));
                
                let col = func.add_u64_register();
                func.add_inst(Inst::sub_u64(
                    col.clone(),
                    tid.clone(),
                    mul_result,
                ));
                
                col
            } else if output_shape.len() == 1 && *axis == 0 {
                // 1D broadcast along axis 0: [1] -> [N]
                // All output elements map to input[0]
                let offset = func.add_u64_register();
                func.add_inst(Inst::mov_u64(offset.clone(), Operand::imm_u64(0)));
                offset
            } else {
                // Fallback: assume simple 1:1 mapping
                let offset = func.add_u64_register();
                func.add_inst(Inst::mov_u64(offset.clone(), tid.clone()));
                offset
            };
            
            // Convert element offset to byte offset
            let byte_offset = func.add_u64_register();
            func.add_inst(Inst::mul_u64(byte_offset.clone(), input_offset, Operand::imm_u64(4)));
            
            // Load from input
            let addr = func.add_u64_register();
            func.add_inst(Inst::add_u64(addr.clone(), src_ptr.clone(), byte_offset));
            
            func.add_inst(Inst::load_global_scalar_f32(dest_reg, addr));
        }
        Stmt::ReduceAxis {
            dest,
            src,
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
            let src_ptr = ctx.param_ptrs.get("input")
                .expect("Input parameter not found").clone();
            let dest_ptr = ctx.param_ptrs.get("output")
                .expect("Output parameter not found").clone();
            
            // Calculate strides for input tensor
            let input_stride: Vec<usize> = {
                let mut strides = vec![1; input_shape.len()];
                for i in (0..input_shape.len()-1).rev() {
                    strides[i] = strides[i + 1] * input_shape[i + 1];
                }
                strides
            };
            
            // Get thread index (which output element this thread computes)
            let tid = func.add_u64_register();
            func.add_inst(Inst::convert_u64_u32(tid.clone(), super::instructions::THREAD_ID.x.clone()));
            
            // Initialize accumulator based on reduce operation
            let accumulator = func.add_f32_register();
            match op {
                crate::tile::ReduceOp::Sum | crate::tile::ReduceOp::Mean => {
                    func.add_inst(Inst::mov_f32(accumulator.clone(), Operand::imm_f32(0.0)));
                }
                crate::tile::ReduceOp::Max => {
                    func.add_inst(Inst::mov_f32(accumulator.clone(), Operand::imm_f32(f32::NEG_INFINITY)));
                }
            }
            
            // Calculate the size of the reduction axis
            let reduce_size = input_shape[*axis];
            
            // Loop over the reduction axis
            for k in 0..reduce_size {
                // Calculate input index based on output thread index and reduction position
                // For axis=1, input_shape=[2,3]: thread i accesses input[i, k]
                // offset = i * input_stride[0] + k * input_stride[1]
                
                let mut input_offset = func.add_u64_register();
                
                // Start with tid * stride[axis-1] (or 0 if axis==0)
                if *axis == 0 {
                    // Reducing along first axis: all threads access input[k, tid, ...]
                    // offset = k * stride[0] + tid * stride[1]
                    let k_contrib = func.add_u64_register();
                    func.add_inst(Inst::mul_u64(
                        k_contrib.clone(),
                        Operand::imm_u64(k as u64),
                        Operand::imm_u64(input_stride[0] as u64),
                    ));
                    
                    let tid_contrib = func.add_u64_register();
                    if input_shape.len() > 1 {
                        func.add_inst(Inst::mul_u64(
                            tid_contrib.clone(),
                            tid.clone(),
                            Operand::imm_u64(input_stride[1] as u64),
                        ));
                    } else {
                        func.add_inst(Inst::mov_u64(tid_contrib.clone(), Operand::imm_u64(0)));
                    }
                    
                    func.add_inst(Inst::add_u64(input_offset.clone(), k_contrib, tid_contrib));
                } else {
                    // Reducing along axis > 0: thread i accesses input[i, k] (for 2D)
                    // offset = tid * stride[axis-1] + k * stride[axis]
                    let tid_contrib = func.add_u64_register();
                    func.add_inst(Inst::mul_u64(
                        tid_contrib.clone(),
                        tid.clone(),
                        Operand::imm_u64(input_stride[axis - 1] as u64),
                    ));
                    
                    let k_contrib = func.add_u64_register();
                    func.add_inst(Inst::mul_u64(
                        k_contrib.clone(),
                        Operand::imm_u64(k as u64),
                        Operand::imm_u64(input_stride[*axis] as u64),
                    ));
                    
                    func.add_inst(Inst::add_u64(input_offset.clone(), tid_contrib, k_contrib));
                }
                
                // Convert element offset to byte offset
                let byte_offset = func.add_u64_register();
                func.add_inst(Inst::mul_u64(byte_offset.clone(), input_offset, Operand::imm_u64(4)));
                
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
            let loop_start_label = format!("loop_start_{}", loop_var);
            let loop_body_label = format!("loop_body_{}", loop_var);
            let loop_end_label = format!("loop_end_{}", loop_var);

            // Loop start label
            func.add_inst(Inst::Label(loop_start_label.clone()));

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

            // Remove loop variable from context
            ctx.loop_vars.remove(loop_var);
        }
    }
}

/// Lower an expression to a U64 operand (for address calculations)
fn lower_expr(func: &mut Function, ctx: &LoweringContext, expr: &Expr) -> Operand<U64> {
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
                Operand::reg(&format!("%{}", name))
            }
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
fn lower_expr_i32(func: &mut Function, ctx: &LoweringContext, expr: &Expr) -> Operand<I32> {
    match expr {
        Expr::Const(val) => Operand::imm_i32(*val as i32),
        Expr::Var(name) => {
            // Check if it's a loop variable
            if let Some(loop_var) = ctx.loop_vars.get(name) {
                loop_var.clone()
            } else {
                Operand::reg(&format!("%{}", name))
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
        assert!(ptx_str.contains(".param .u64 input"));  // Parameters are pointers in PTX
        
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
