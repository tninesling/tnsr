mod online;
use std::collections::{HashMap, HashSet};

use super::instructions::{AndInst, Inst, Operand};
use super::target::PtxTarget;
use super::types::{B32, F32, I32, U64};
use super::{Function, Module};
use crate::tile::{
    DType, Dim, Expr, MemorySpace, RegionOpKind, SharedLayout, Stmt, TileGraph, TileIR, TileLayout,
    TileVar,
};
use petgraph::{Graph, graph::NodeIndex};

pub struct PtxGraph {
    // SAFETY INVARIANT: Functions contain references into `arena` whose lifetimes are
    // forged to 'static. `graph` is private, is declared before `arena` so it drops
    // first, and no API may return an owned Function or a borrow not tied to `&self`.
    graph: Graph<Function<'static>, usize>,
    region_functions: Vec<(usize, Function<'static>)>,
    reduction_region_functions: Vec<(usize, Function<'static>)>,
    matmul_region_functions: Vec<(usize, Function<'static>)>,
    physical_nodes: Option<HashSet<NodeIndex>>,
    target: PtxTarget,
    #[allow(dead_code)]
    arena: Box<bumpalo::Bump>,
}

impl PtxGraph {
    pub(crate) fn module_source(&self) -> String {
        let mut module = Module::new_with_target(self.target.compute_capability);
        for node in self.graph.node_indices() {
            if self
                .physical_nodes
                .as_ref()
                .is_none_or(|physical| physical.contains(&node))
            {
                module.add_function(self.graph[node].clone());
            }
        }
        for (_, function) in &self.region_functions {
            module.add_function(function.clone());
        }
        for (_, function) in &self.reduction_region_functions {
            module.add_function(function.clone());
        }
        for (_, function) in &self.matmul_region_functions {
            module.add_function(function.clone());
        }
        module.to_string()
    }

    /// Render this graph as a complete PTX module.
    pub fn to_ptx(&self) -> String {
        self.module_source()
    }

    pub(crate) fn kernel_name(&self, node: NodeIndex) -> Option<&str> {
        self.graph.node_weight(node).map(|function| function.name)
    }

    pub(crate) fn region_kernel_name(&self, region_id: usize) -> Option<&str> {
        self.region_functions
            .iter()
            .find(|(id, _)| *id == region_id)
            .map(|(_, function)| function.name)
    }

    pub(crate) fn reduction_region_kernel_name(&self, region_id: usize) -> Option<&str> {
        self.reduction_region_functions
            .iter()
            .find(|(id, _)| *id == region_id)
            .map(|(_, function)| function.name)
    }

    pub(crate) fn matmul_region_kernel_name(&self, region_id: usize) -> Option<&str> {
        self.matmul_region_functions
            .iter()
            .find(|(id, _)| *id == region_id)
            .map(|(_, function)| function.name)
    }

    pub(crate) fn with_target(mut self, target: PtxTarget) -> Self {
        self.target = target;
        self
    }
}

impl From<TileGraph> for PtxGraph {
    fn from(tile_graph: TileGraph) -> Self {
        // Create a single arena for all functions in the graph
        // Keep the arena at a stable address. Bump-backed vectors retain a
        // pointer to the allocator, so moving a stack-allocated Bump here
        // would invalidate cloned functions.
        let arena = Box::new(bumpalo::Bump::new());

        let crate::tile::TileGraph {
            graph: tile_nodes,
            region_kernels,
            reduction_region_kernels,
            matmul_region_kernels,
            physical_nodes,
        } = tile_graph;
        let graph = tile_nodes.map_owned(
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
        let region_offset = graph.node_count();
        let region_functions: Vec<(usize, Function<'static>)> = region_kernels
            .into_iter()
            .map(|kernel| {
                // SAFETY: Same arena ownership and drop-order invariant as graph functions.
                let arena_ref: &'static bumpalo::Bump =
                    unsafe { &*(arena.as_ref() as *const bumpalo::Bump) };
                (
                    kernel.region_id,
                    tile_ir_to_function(kernel.ir, arena_ref, region_offset + kernel.region_id),
                )
            })
            .collect();
        let reduction_region_offset = region_offset + region_functions.len();
        let reduction_region_functions = reduction_region_kernels
            .into_iter()
            .map(|kernel| {
                // SAFETY: Same arena ownership and drop-order invariant as graph functions.
                let arena_ref: &'static bumpalo::Bump =
                    unsafe { &*(arena.as_ref() as *const bumpalo::Bump) };
                (
                    kernel.region_id,
                    tile_ir_to_function(
                        kernel.ir,
                        arena_ref,
                        reduction_region_offset + kernel.region_id,
                    ),
                )
            })
            .collect::<Vec<_>>();
        let matmul_region_offset = reduction_region_offset + reduction_region_functions.len();
        let matmul_region_functions = matmul_region_kernels
            .into_iter()
            .map(|kernel| {
                // SAFETY: Same arena ownership and drop-order invariant as graph functions.
                let arena_ref: &'static bumpalo::Bump =
                    unsafe { &*(arena.as_ref() as *const bumpalo::Bump) };
                (
                    kernel.region_id,
                    tile_ir_to_function(
                        kernel.ir,
                        arena_ref,
                        matmul_region_offset + kernel.region_id,
                    ),
                )
            })
            .collect();

        Self {
            graph,
            region_functions,
            reduction_region_functions,
            matmul_region_functions,
            physical_nodes,
            target: PtxTarget::sm80(),
            arena,
        }
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
        ctx.param_dtypes.insert(param.name.clone(), param.dtype);
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
    /// Storage dtype for global-memory parameters.
    param_dtypes: HashMap<String, crate::tile::DType>,
    /// Maps loop variable names to their i32 register operands
    loop_vars: HashMap<String, Operand<'a, I32>>,
    index_vars: HashMap<String, Operand<'a, U64>>,
    /// Current offset into shared memory for allocation
    shared_mem_offset: usize,
    /// Maps TileVar to its dimensions (rows, cols)
    tile_dims: HashMap<TileVar, (usize, usize)>,
    tile_dtypes: HashMap<TileVar, crate::tile::DType>,
    tile_layouts: HashMap<TileVar, TileLayout>,
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
            tile_dtypes: HashMap::new(),
            shared_mem_ptrs: HashMap::new(),
            param_ptrs: HashMap::new(),
            param_dtypes: HashMap::new(),
            loop_vars: HashMap::new(),
            index_vars: HashMap::new(),
            shared_mem_offset: 0,
            tile_dims: HashMap::new(),
            tile_layouts: HashMap::new(),
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

fn load_global_as_f32<'a>(
    func: &mut Function<'a>,
    dtype: crate::tile::DType,
    dst: Operand<'a, F32>,
    addr: Operand<'a, U64>,
) {
    match dtype {
        crate::tile::DType::F32 | crate::tile::DType::TF32 => {
            func.add_inst(Inst::load_global_scalar_f32(dst, addr));
        }
        crate::tile::DType::F16 => {
            let value = func.add_f16_register();
            func.add_inst(Inst::LdGlobalF16 {
                dst: value.clone(),
                addr,
            });
            func.add_inst(Inst::convert_f32_f16(dst, value));
        }
        crate::tile::DType::BF16 => {
            let value = func.add_bf16_register();
            func.add_inst(Inst::LdGlobalBF16 {
                dst: value.clone(),
                addr,
            });
            func.add_inst(Inst::convert_f32_bf16(dst, value));
        }
    }
}

fn store_global_from_f32<'a>(
    func: &mut Function<'a>,
    dtype: crate::tile::DType,
    addr: Operand<'a, U64>,
    src: Operand<'a, F32>,
) {
    match dtype {
        crate::tile::DType::F32 | crate::tile::DType::TF32 => {
            func.add_inst(Inst::store_global_scalar_f32(addr, src));
        }
        crate::tile::DType::F16 => {
            let value = func.add_f16_register();
            func.add_inst(Inst::convert_f16_f32(value.clone(), src));
            func.add_inst(Inst::StGlobalF16 { addr, src: value });
        }
        crate::tile::DType::BF16 => {
            let value = func.add_bf16_register();
            func.add_inst(Inst::convert_bf16_f32(value.clone(), src));
            func.add_inst(Inst::StGlobalBF16 { addr, src: value });
        }
    }
}

fn load_shared_as_f32<'a>(
    func: &mut Function<'a>,
    dtype: crate::tile::DType,
    dst: Operand<'a, F32>,
    addr: Operand<'a, U64>,
) {
    match dtype {
        crate::tile::DType::F32 | crate::tile::DType::TF32 => {
            func.add_inst(Inst::load_shared_scalar_f32(dst, addr));
        }
        crate::tile::DType::F16 => {
            let value = func.add_f16_register();
            func.add_inst(Inst::LdSharedF16 {
                dst: value.clone(),
                addr,
            });
            func.add_inst(Inst::convert_f32_f16(dst, value));
        }
        crate::tile::DType::BF16 => {
            let value = func.add_bf16_register();
            func.add_inst(Inst::LdSharedBF16 {
                dst: value.clone(),
                addr,
            });
            func.add_inst(Inst::convert_f32_bf16(dst, value));
        }
    }
}

fn store_shared_from_f32<'a>(
    func: &mut Function<'a>,
    dtype: crate::tile::DType,
    addr: Operand<'a, U64>,
    src: Operand<'a, F32>,
) {
    match dtype {
        crate::tile::DType::F32 | crate::tile::DType::TF32 => {
            func.add_inst(Inst::StSharedF32 {
                addr,
                src: vec![src],
                vec: super::instructions::VecWidth::Scalar,
            });
        }
        crate::tile::DType::F16 => {
            let value = func.add_f16_register();
            func.add_inst(Inst::convert_f16_f32(value.clone(), src));
            func.add_inst(Inst::StSharedF16 { addr, src: value });
        }
        crate::tile::DType::BF16 => {
            let value = func.add_bf16_register();
            func.add_inst(Inst::convert_bf16_f32(value.clone(), src));
            func.add_inst(Inst::StSharedBF16 { addr, src: value });
        }
    }
}

fn lower_block<'a>(
    func: &mut Function<'a>,
    ctx: &mut LoweringContext<'a>,
    block: &crate::tile::Block,
) {
    let outer_bindings = ctx.index_vars.clone();
    for stmt in &block.stmts {
        lower_stmt(func, ctx, stmt);
    }
    ctx.index_vars = outer_bindings;
}

fn lower_stmt<'a>(func: &mut Function<'a>, ctx: &mut LoweringContext<'a>, stmt: &Stmt) {
    match stmt {
        Stmt::LetIndex { name, value } => {
            let value = lower_expr(func, ctx, value);
            ctx.index_vars.insert(name.clone(), value);
        }
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
            layout,
            dtype,
            rows,
            cols,
        } => {
            ctx.tile_layouts.insert(*var, *layout);
            // Track tile dimensions
            ctx.tile_dims.insert(*var, (*rows, *cols));
            ctx.tile_dtypes.insert(*var, *dtype);

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
                    let tile_size_bytes = shared_layout(ctx, *var).allocation_bytes(*rows, *dtype);

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
            let dtype = *ctx
                .param_dtypes
                .get(src_param)
                .expect("Parameter dtype not found in context");
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

            // Byte offset uses the global parameter's storage dtype.
            let byte_offset = func.add_u64_register();
            func.add_inst(Inst::mul_u64(
                byte_offset.clone(),
                elem_offset,
                Operand::imm_u64(dtype.size_bytes() as u64),
            ));

            // Add offset to base pointer
            let addr = func.add_u64_register();
            func.add_inst(Inst::AddU64(super::instructions::AddInst::new(
                addr.clone(),
                param_ptr,
                byte_offset,
            )));

            // Check if destination is shared memory or register
            if ctx.shared_mem_ptrs.contains_key(dest) {
                // Destination is shared memory: load to temp register, then store to shared
                let temp_reg = func.add_f32_register();
                load_global_as_f32(func, dtype, temp_reg.clone(), addr);

                let dest_dtype = ctx.tile_dtypes[dest];
                let final_addr = shared_thread_address(func, ctx, *dest);
                store_shared_from_f32(func, dest_dtype, final_addr, temp_reg);
            } else {
                // Destination is register: load directly
                let dest_reg = ctx.get_or_alloc_reg(func, *dest);
                load_global_as_f32(func, dtype, dest_reg, addr);
            }
        }
        Stmt::LoadGlobalToSharedPredicated {
            dest,
            src_param,
            element_index,
            row,
            col,
            tile_row,
            tile_col,
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
            let element_offset = lower_expr(func, ctx, element_index);
            let source = ctx
                .param_ptrs
                .get(src_param)
                .expect("Parameter not found in context")
                .clone();
            let address =
                element_address(func, source, element_offset, ctx.param_dtypes[src_param]);
            load_global_as_f32(func, ctx.param_dtypes[src_param], value.clone(), address);
            func.add_inst(Inst::Label(skip_load));

            let shared_address = shared_coordinate_address(func, ctx, *dest, tile_row, tile_col);
            if ctx.tile_dtypes.get(dest) == Some(&crate::tile::DType::TF32) {
                let tf32 = func.add_b32_register();
                func.add_inst(Inst::convert_tf32_f32(tf32.clone(), value));
                func.add_inst(Inst::StSharedB32 {
                    addr: shared_address,
                    src: tf32,
                });
            } else {
                store_shared_from_f32(func, ctx.tile_dtypes[dest], shared_address, value);
            }
        }
        Stmt::AsyncCopy {
            dest,
            src_param,
            element_index,
            row,
            col,
            tile_row,
            tile_col,
            layout,
            copy_bytes,
        } => {
            let done = bumpalo::format!(in ctx.arena, "async_copy_done_{}", ctx.label_counter)
                .into_bump_str();
            let zero = bumpalo::format!(in ctx.arena, "async_copy_zero_{}", ctx.label_counter)
                .into_bump_str();
            ctx.label_counter += 1;
            // Copy groups are assigned to complete threads, with inactive threads
            // excluded before addressing shared memory. All threads commit and wait.
            for (coordinate, extent) in [
                (tile_row, ctx.tile_dims[dest].0),
                (tile_col, ctx.tile_dims[dest].1),
            ] {
                let coordinate = lower_expr(func, ctx, coordinate);
                let outside = func.add_predicate_register();
                func.add_inst(Inst::setp_ge_u64(
                    outside.clone(),
                    coordinate,
                    Operand::imm_u64(extent as u64),
                ));
                func.add_inst(Inst::Bra {
                    condition: outside,
                    target: done,
                });
            }
            let bytes = func.add_i32_register();
            func.add_inst(Inst::mov_i32(bytes.clone(), Operand::imm_i32(0)));
            let address = func.add_u64_register();
            func.add_inst(Inst::mov_u64(
                address.clone(),
                ctx.param_ptrs[src_param].clone(),
            ));
            for (coordinate, extent) in [(row, layout.rows), (col, layout.cols)] {
                let coordinate = lower_expr(func, ctx, coordinate);
                let outside = func.add_predicate_register();
                func.add_inst(Inst::setp_ge_u64(
                    outside.clone(),
                    coordinate,
                    Operand::imm_u64(extent as u64),
                ));
                func.add_inst(Inst::Bra {
                    condition: outside,
                    target: zero,
                });
            }
            let index = lower_expr(func, ctx, element_index);
            let source = element_address(
                func,
                ctx.param_ptrs[src_param].clone(),
                index,
                ctx.param_dtypes[src_param],
            );
            func.add_inst(Inst::mov_u64(address.clone(), source));
            func.add_inst(Inst::mov_i32(
                bytes.clone(),
                Operand::imm_i32(*copy_bytes as i32),
            ));
            func.add_inst(Inst::Label(zero));
            let destination = shared_coordinate_address(func, ctx, *dest, tile_row, tile_col);
            func.add_inst(Inst::AsyncCopy {
                dst: destination,
                src: address,
                source_bytes: bytes,
                copy_bytes: *copy_bytes,
            });
            func.add_inst(Inst::Label(done));
        }
        Stmt::SelectSharedStage {
            dest,
            first,
            second,
            stage,
        } => {
            let stage = lower_expr(func, ctx, stage);
            let predicate = func.add_predicate_register();
            func.add_inst(Inst::setp_ge_u64(
                predicate.clone(),
                stage,
                Operand::imm_u64(1),
            ));
            let pointer = func.add_u64_register();
            func.add_inst(Inst::selp_u64(
                pointer.clone(),
                ctx.shared_mem_ptrs[second].clone(),
                ctx.shared_mem_ptrs[first].clone(),
                predicate,
            ));
            ctx.shared_mem_ptrs.insert(*dest, pointer);
            ctx.tile_layouts.insert(*dest, ctx.tile_layouts[first]);
            ctx.tile_dims.insert(*dest, ctx.tile_dims[first]);
            ctx.tile_dtypes.insert(*dest, ctx.tile_dtypes[first]);
        }
        Stmt::AsyncCommit => func.add_inst(Inst::AsyncCommit),
        Stmt::AsyncWait => func.add_inst(Inst::AsyncWait),
        Stmt::LoadGlobalPredicated {
            dest,
            src_param,
            element_index,
            row,
            col,
            layout,
        } => {
            let row = lower_expr(func, ctx, row);
            let col = lower_expr(func, ctx, col);
            let value = ctx.get_or_alloc_reg(func, *dest);
            func.add_inst(Inst::mov_f32(value.clone(), Operand::imm_f32(0.0)));
            let skip_load =
                bumpalo::format!(in ctx.arena, "matmul_epilogue_load_skip_{}", ctx.label_counter)
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
            let element_offset = lower_expr(func, ctx, element_index);
            let source = ctx
                .param_ptrs
                .get(src_param)
                .expect("Parameter not found in context")
                .clone();
            let address =
                element_address(func, source, element_offset, ctx.param_dtypes[src_param]);
            load_global_as_f32(func, ctx.param_dtypes[src_param], value, address);
            func.add_inst(Inst::Label(skip_load));
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

            let dtype = *ctx
                .param_dtypes
                .get(dest_param)
                .expect("Parameter dtype not found in context");
            // Byte offset uses the global parameter's storage dtype.
            let byte_offset = func.add_u64_register();
            func.add_inst(Inst::mul_u64(
                byte_offset.clone(),
                elem_offset,
                Operand::imm_u64(dtype.size_bytes() as u64),
            ));

            // Add offset to base pointer
            let addr = func.add_u64_register();
            func.add_inst(Inst::AddU64(super::instructions::AddInst::new(
                addr.clone(),
                param_ptr,
                byte_offset,
            )));

            // Store to global memory
            store_global_from_f32(func, dtype, addr, src_reg);
        }
        Stmt::StoreGlobalPredicated {
            dest_param,
            src,
            element_index,
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
            let element_offset = lower_expr(func, ctx, element_index);
            let destination = ctx
                .param_ptrs
                .get(dest_param)
                .expect("Parameter not found in context")
                .clone();
            let address = element_address(
                func,
                destination,
                element_offset,
                ctx.param_dtypes[dest_param],
            );
            let value = ctx.get_or_alloc_reg(func, *src);
            store_global_from_f32(func, ctx.param_dtypes[dest_param], address, value);
            func.add_inst(Inst::Label(skip_store));
        }
        Stmt::LoadSharedToReg { dest, src } => {
            // Track that this register tile is backed by shared memory
            // We don't need to emit any PTX here - MatMul will access shared memory directly
            ctx.reg_to_shared.insert(*dest, *src);

            ctx.tile_layouts.insert(*dest, ctx.tile_layouts[src]);
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
            schedule,
        } => {
            if schedule.plan != crate::tile::MatMulPlan::ScalarF32 {
                lower_tensor_core_matmul(func, ctx, *dest, *a, *b, schedule);
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
            let (_b_rows, _b_cols) = ctx
                .tile_dims
                .get(&b_smem)
                .expect("Tile dimensions not found for B");
            let tile_k = *a_cols; // K dimension of the tile
            let a_dtype = *ctx
                .tile_dtypes
                .get(&a_smem)
                .expect("Tile dtype not found for A");
            let b_dtype = *ctx
                .tile_dtypes
                .get(&b_smem)
                .expect("Tile dtype not found for B");

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

            // Compute A row base offset in bytes.
            let a_row_base = func.add_u64_register();
            func.add_inst(Inst::mul_u64(
                a_row_base.clone(),
                tid_y_u64.clone(),
                Operand::imm_u64(shared_row_stride(ctx, a_smem) as u64),
            ));
            func.add_inst(Inst::mul_u64(
                a_row_base.clone(),
                a_row_base.clone(),
                Operand::imm_u64(a_dtype.size_bytes() as u64),
            ));
            let a_row_ptr = func.add_u64_register();
            func.add_inst(Inst::add_u64(a_row_ptr.clone(), a_ptr.clone(), a_row_base));

            // Compute B column base offset in bytes.
            let b_col_base = func.add_u64_register();
            func.add_inst(Inst::mul_u64(
                b_col_base.clone(),
                tid_x_u64.clone(),
                Operand::imm_u64(b_dtype.size_bytes() as u64),
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

            if shared_layout(ctx, a_smem).xor_mask == 0 {
                let k_offset_a = multiply_u64(func, k_u64.clone(), a_dtype.size_bytes());
                func.add_inst(Inst::add_u64(a_addr.clone(), a_row_ptr.clone(), k_offset_a));
            } else {
                let address = shared_address(func, ctx, a_smem, tid_y_u64.clone(), k_u64.clone());
                func.add_inst(Inst::mov_u64(a_addr.clone(), address));
            }
            load_shared_as_f32(func, a_dtype, a_val.clone(), a_addr.clone());

            if shared_layout(ctx, b_smem).xor_mask == 0 {
                let k_offset_b = multiply_u64(func, k_u64.clone(), shared_row_stride(ctx, b_smem));
                let k_offset_b = multiply_u64(func, k_offset_b, b_dtype.size_bytes());
                func.add_inst(Inst::add_u64(b_addr.clone(), b_col_ptr.clone(), k_offset_b));
            } else {
                let address = shared_address(func, ctx, b_smem, k_u64, tid_x_u64.clone());
                func.add_inst(Inst::mov_u64(b_addr.clone(), address));
            }
            load_shared_as_f32(func, b_dtype, b_val.clone(), b_addr.clone());

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
        Stmt::ConvertLayout {
            dest,
            src,
            coordinates,
        } => match (ctx.tile_layouts[src], ctx.tile_layouts[dest]) {
            (
                TileLayout::WarpAccumulator {
                    operand_dtype,
                    block_width,
                    warp_topology,
                },
                TileLayout::Shared(layout),
            ) => {
                let done =
                    bumpalo::format!(in ctx.arena, "fragment_store_done_{}", ctx.label_counter)
                        .into_bump_str();
                ctx.label_counter += 1;
                skip_noncomputing_warps(func, block_width, warp_topology.0 * warp_topology.1, done);
                let (row, col) = warp_coordinates(block_width, warp_topology);
                let address = if warp_topology == (1, 1) && layout.xor_mask == 0 {
                    ctx.shared_mem_ptrs[dest].clone()
                } else {
                    shared_coordinate_address(func, ctx, *dest, &row, &col)
                };
                func.add_inst(Inst::WmmaStore {
                    dtype: operand_dtype,
                    addr: address,
                    frags: ctx.fragment_regs[src].clone(),
                    stride: Operand::imm_i32(layout.row_stride as i32),
                });
                func.add_inst(Inst::Label(done));
            }
            (TileLayout::Shared(_), TileLayout::ThreadScalar) => {
                let address = if let Some((row, col)) = coordinates {
                    shared_coordinate_address(func, ctx, *src, row, col)
                } else {
                    shared_thread_address(func, ctx, *src)
                };
                let output = ctx.get_or_alloc_reg(func, *dest);
                load_shared_as_f32(func, ctx.tile_dtypes[src], output, address);
            }
            (TileLayout::ThreadScalar, TileLayout::ThreadScalar) => {
                let value = ctx.get_or_alloc_reg(func, *src);
                let output = ctx.get_or_alloc_reg(func, *dest);
                func.add_inst(Inst::mov_f32(output, value));
            }
            _ => unreachable!("tile layout conversion must be validated before lowering"),
        },
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
            index_count,
        } => {
            if ctx.param_dtypes["output"] != crate::tile::DType::F32 {
                lower_half_embedding_backward(func, ctx, *width, *index_count);
                return;
            }
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
            let destination = element_address(
                func,
                output_ptr,
                destination_offset,
                crate::tile::DType::F32,
            );
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
            let dtype = *ctx
                .param_dtypes
                .get("input")
                .expect("Input parameter dtype not found");
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
            load_at(func, dtype, dest_reg, src_ptr, input_offset);
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
            let dtype = *ctx
                .param_dtypes
                .get("input")
                .expect("Input parameter dtype not found");

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
            load_at(func, dtype, dest_reg, src_ptr, input_offset);
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
            let dtype = *ctx
                .param_dtypes
                .get("input")
                .expect("Input parameter dtype not found");
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
                    Operand::imm_u64(dtype.size_bytes() as u64),
                ));

                // Load input element
                let addr = func.add_u64_register();
                func.add_inst(Inst::add_u64(addr.clone(), src_ptr.clone(), byte_offset));

                let value = func.add_f32_register();
                load_global_as_f32(func, dtype, value.clone(), addr);

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
        Stmt::OnlineRegion { region } => online::lower_online_region(func, ctx, region),
        Stmt::ReductionRegion { region } => {
            lower_reduction_region(func, ctx, region);
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
            let outer_loop_var = ctx.loop_vars.insert(loop_var.clone(), loop_counter.clone());

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
            if let Some(outer) = outer_loop_var {
                ctx.loop_vars.insert(loop_var.clone(), outer);
            } else {
                ctx.loop_vars.remove(loop_var);
            }
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

/// One thread owns each output element, accumulating repeated indices in f32.
/// This avoids f32 atomics on b16 storage and rounds only after the complete sum.
fn lower_half_embedding_backward<'a>(
    func: &mut Function<'a>,
    ctx: &mut LoweringContext<'a>,
    width: usize,
    index_count: usize,
) {
    let tid = global_linear_tid(func);
    let row = quotient_u64(func, tid.clone(), width.max(1));
    let column = decode_coordinate(func, tid.clone(), 1, width.max(1));
    let sum = func.add_f32_register();
    func.add_inst(Inst::mov_f32(sum.clone(), Operand::imm_f32(0.0)));
    let state = begin_counted_loop(func, ctx, "half_embedding_backward", index_count);
    let index_value = load_param_f32_at(func, ctx, "indices", state.counter.clone());
    let index = func.add_u64_register();
    func.add_inst(Inst::convert_u64_f32(index.clone(), index_value));
    let skip = next_label(ctx, "half_embedding_skip");
    let different = func.add_predicate_register();
    func.add_inst(Inst::setp_ne_u64(different.clone(), index, row));
    func.add_inst(Inst::Bra {
        condition: different,
        target: skip,
    });
    let offset = multiply_u64(func, state.counter.clone(), width);
    func.add_inst(Inst::add_u64(offset.clone(), offset.clone(), column));
    let gradient = load_param_f32_at(func, ctx, "grad_output", offset);
    func.add_inst(Inst::add_f32(sum.clone(), sum.clone(), gradient));
    func.add_inst(Inst::Label(skip));
    end_counted_loop(func, state);
    store_param_f32_at(func, ctx, "output", tid, sum);
}

fn lower_reduction_region<'a>(
    func: &mut Function<'a>,
    ctx: &mut LoweringContext<'a>,
    region: &crate::tile::ReductionRegion,
) {
    match region.schedule {
        crate::tile::ReductionSchedule::Serial => lower_serial_reduction_region(func, ctx, region),
        crate::tile::ReductionSchedule::Subgroup { width } => {
            lower_cooperative_reduction_region(func, ctx, region, width, false)
        }
        crate::tile::ReductionSchedule::Block { threads } => {
            lower_cooperative_reduction_region(func, ctx, region, threads, true)
        }
    }
}

fn lower_serial_reduction_region<'a>(
    func: &mut Function<'a>,
    ctx: &mut LoweringContext<'a>,
    region: &crate::tile::ReductionRegion,
) {
    use crate::tile::ReductionInputDomain;

    let tid = global_linear_tid(func);
    let input_strides = contiguous_strides(&region.input_shape);
    let output_strides = contiguous_strides(&region.output_shape);
    let base_offset = func.add_u64_register();
    func.add_inst(Inst::mov_u64(base_offset.clone(), Operand::imm_u64(0)));
    for dimension in 0..region.output_shape.len() {
        if dimension == region.axis {
            continue;
        }
        let coordinate = decode_coordinate(
            func,
            tid.clone(),
            output_strides[dimension],
            region.output_shape[dimension],
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

    let accumulator = func.add_f32_register();
    match region.op {
        crate::tensor::ReduceOp::Sum | crate::tensor::ReduceOp::Mean => {
            func.add_inst(Inst::mov_f32(accumulator.clone(), Operand::imm_f32(0.0)));
        }
        crate::tensor::ReduceOp::Max => {
            func.add_inst(Inst::mov_f32(
                accumulator.clone(),
                Operand::imm_f32(f32::NEG_INFINITY),
            ));
        }
    }

    let reduction = begin_counted_loop(
        func,
        ctx,
        "reduction_region",
        region.input_shape[region.axis],
    );
    let contribution = func.add_u64_register();
    func.add_inst(Inst::mul_u64(
        contribution.clone(),
        reduction.counter.clone(),
        Operand::imm_u64(input_strides[region.axis] as u64),
    ));
    let input_offset = func.add_u64_register();
    func.add_inst(Inst::add_u64(
        input_offset.clone(),
        base_offset.clone(),
        contribution,
    ));
    let mut values = HashMap::new();
    for (index, input) in region.inputs.iter().enumerate() {
        if input.domain == ReductionInputDomain::Full {
            let offset = reduction_region_input_offset(
                func,
                input,
                &region.input_shape,
                input_offset.clone(),
            );
            let value = load_param_f32_at(func, ctx, &format!("input_{index}"), offset);
            values.insert(input.value, value);
        }
    }
    for operation in &region.producer_operations {
        emit_region_op(func, operation, &mut values);
    }
    for (index, (output, shape)) in region.outputs.iter().zip(&region.output_shapes).enumerate() {
        if shape == &region.input_shape && values.contains_key(&output.value) {
            let value = values[&output.value].clone();
            store_param_f32_at(
                func,
                ctx,
                &format!("output_{index}"),
                input_offset.clone(),
                value,
            );
        }
    }
    let reduction_value = values
        .get(&region.reduction_input)
        .expect("reduction region input SSA value is unavailable")
        .clone();
    match region.op {
        crate::tensor::ReduceOp::Sum | crate::tensor::ReduceOp::Mean => {
            func.add_inst(Inst::add_f32(
                accumulator.clone(),
                accumulator.clone(),
                reduction_value,
            ));
        }
        crate::tensor::ReduceOp::Max => {
            func.add_inst(Inst::max_f32(
                accumulator.clone(),
                accumulator.clone(),
                reduction_value,
            ));
        }
    }
    end_counted_loop(func, reduction);

    let reduced =
        if region.op == crate::tensor::ReduceOp::Mean && region.input_shape[region.axis] != 0 {
            let value = func.add_f32_register();
            func.add_inst(Inst::div_f32(
                value.clone(),
                accumulator,
                Operand::imm_f32(region.input_shape[region.axis] as f32),
            ));
            value
        } else {
            // Match the reference executor: an empty mean retains the initialized
            // zero accumulator because there are no per-element divisions.
            accumulator
        };
    values.clear();
    values.insert(region.reduced_value, reduced.clone());
    for (index, input) in region.inputs.iter().enumerate() {
        if input.domain == ReductionInputDomain::Reduced {
            let offset =
                reduction_region_input_offset(func, input, &region.output_shape, tid.clone());
            let value = load_param_f32_at(func, ctx, &format!("input_{index}"), offset);
            values.insert(input.value, value);
        }
    }
    for operation in &region.epilogue_operations {
        emit_region_op(func, operation, &mut values);
    }
    let reduced_values = values.clone();
    for (index, (output, shape)) in region.outputs.iter().zip(&region.output_shapes).enumerate() {
        if shape == &region.input_shape {
            continue;
        }
        let value = values
            .get(&output.value)
            .expect("reduction region output SSA value is unavailable")
            .clone();
        store_param_f32_at(func, ctx, &format!("output_{index}"), tid.clone(), value);
    }

    if !region.full_epilogue_operations.is_empty() {
        let full = begin_counted_loop(
            func,
            ctx,
            "reduction_full_epilogue",
            region.input_shape[region.axis],
        );
        let contribution = func.add_u64_register();
        func.add_inst(Inst::mul_u64(
            contribution.clone(),
            full.counter.clone(),
            Operand::imm_u64(input_strides[region.axis] as u64),
        ));
        let full_offset = func.add_u64_register();
        func.add_inst(Inst::add_u64(
            full_offset.clone(),
            base_offset,
            contribution,
        ));
        values.clear();
        values.extend(reduced_values);
        for (index, input) in region.inputs.iter().enumerate() {
            if input.domain == ReductionInputDomain::Full {
                let offset = reduction_region_input_offset(
                    func,
                    input,
                    &region.input_shape,
                    full_offset.clone(),
                );
                let value = load_param_f32_at(func, ctx, &format!("input_{index}"), offset);
                values.insert(input.value, value);
            }
        }
        for operation in &region.producer_operations {
            emit_region_op(func, operation, &mut values);
        }
        for operation in &region.full_epilogue_operations {
            emit_region_op(func, operation, &mut values);
        }
        for (index, (output, shape)) in region.outputs.iter().zip(&region.output_shapes).enumerate()
        {
            if shape == &region.input_shape {
                let value = values
                    .get(&output.value)
                    .expect("full reduction epilogue output SSA value is unavailable")
                    .clone();
                store_param_f32_at(
                    func,
                    ctx,
                    &format!("output_{index}"),
                    full_offset.clone(),
                    value,
                );
            }
        }
        end_counted_loop(func, full);
    }
}

fn lower_cooperative_reduction_region<'a>(
    func: &mut Function<'a>,
    ctx: &mut LoweringContext<'a>,
    region: &crate::tile::ReductionRegion,
    thread_count: u32,
    tree_reduce: bool,
) {
    use crate::tile::ReductionInputDomain;

    let thread_count = u64::from(thread_count);

    let global_tid = global_linear_tid(func);
    let fiber = func.add_u64_register();
    func.add_inst(Inst::div_u64(
        fiber.clone(),
        global_tid.clone(),
        Operand::imm_u64(thread_count),
    ));
    let outside = func.add_predicate_register();
    func.add_inst(Inst::setp_ge_u64(
        outside.clone(),
        fiber.clone(),
        Operand::imm_u64(region.output_shape.iter().product::<usize>() as u64),
    ));
    func.add_inst(Inst::Bra {
        condition: outside,
        target: ctx.return_label,
    });

    let warp_base = func.add_u64_register();
    func.add_inst(Inst::mul_u64(
        warp_base.clone(),
        fiber.clone(),
        Operand::imm_u64(thread_count),
    ));
    let lane = func.add_u64_register();
    func.add_inst(Inst::sub_u64(lane.clone(), global_tid, warp_base));

    let input_strides = contiguous_strides(&region.input_shape);
    let output_strides = contiguous_strides(&region.output_shape);
    let base_offset = func.add_u64_register();
    func.add_inst(Inst::mov_u64(base_offset.clone(), Operand::imm_u64(0)));
    for dimension in 0..region.output_shape.len() {
        if dimension == region.axis {
            continue;
        }
        let coordinate = decode_coordinate(
            func,
            fiber.clone(),
            output_strides[dimension],
            region.output_shape[dimension],
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

    let accumulator = func.add_f32_register();
    let identity = region.op.identity_f32();
    func.add_inst(Inst::mov_f32(
        accumulator.clone(),
        Operand::imm_f32(identity),
    ));

    let counter = func.add_u64_register();
    func.add_inst(Inst::mov_u64(counter.clone(), lane.clone()));
    let loop_start = next_label(ctx, "warp_reduction_loop");
    let loop_end = next_label(ctx, "warp_reduction_end");
    func.add_inst(Inst::Label(loop_start));
    let done = func.add_predicate_register();
    func.add_inst(Inst::setp_ge_u64(
        done.clone(),
        counter.clone(),
        Operand::imm_u64(region.input_shape[region.axis] as u64),
    ));
    func.add_inst(Inst::Bra {
        condition: done,
        target: loop_end,
    });

    let contribution = func.add_u64_register();
    func.add_inst(Inst::mul_u64(
        contribution.clone(),
        counter.clone(),
        Operand::imm_u64(input_strides[region.axis] as u64),
    ));
    let input_offset = func.add_u64_register();
    func.add_inst(Inst::add_u64(
        input_offset.clone(),
        base_offset.clone(),
        contribution,
    ));
    let mut values = HashMap::new();
    for (index, input) in region.inputs.iter().enumerate() {
        if input.domain == ReductionInputDomain::Full {
            let offset = reduction_region_input_offset(
                func,
                input,
                &region.input_shape,
                input_offset.clone(),
            );
            let value = load_param_f32_at(func, ctx, &format!("input_{index}"), offset);
            values.insert(input.value, value);
        }
    }
    for operation in &region.producer_operations {
        emit_cooperative_reduction_region_op(func, operation, &mut values);
    }
    for (index, (output, shape)) in region.outputs.iter().zip(&region.output_shapes).enumerate() {
        if shape == &region.input_shape && values.contains_key(&output.value) {
            store_param_f32_at(
                func,
                ctx,
                &format!("output_{index}"),
                input_offset.clone(),
                values[&output.value].clone(),
            );
        }
    }
    let reduction_value = values
        .get(&region.reduction_input)
        .expect("reduction region input SSA value is unavailable")
        .clone();
    match region.op {
        crate::tensor::ReduceOp::Sum | crate::tensor::ReduceOp::Mean => func.add_inst(
            Inst::add_f32(accumulator.clone(), accumulator.clone(), reduction_value),
        ),
        crate::tensor::ReduceOp::Max => func.add_inst(Inst::max_f32(
            accumulator.clone(),
            accumulator.clone(),
            reduction_value,
        )),
    }
    func.add_inst(Inst::add_u64(
        counter.clone(),
        counter,
        Operand::imm_u64(thread_count),
    ));
    func.add_inst(Inst::BraUni { target: loop_start });
    func.add_inst(Inst::Label(loop_end));

    let shared_name = ctx.arena.alloc_str("reduction_partials");
    let shared_value_count = if tree_reduce { thread_count } else { 1 };
    func.add_shared_memory(
        shared_name,
        shared_value_count as usize * std::mem::size_of::<f32>(),
    );
    let shared_base = func.add_u64_register();
    func.add_inst(Inst::mov_u64(
        shared_base.clone(),
        Operand::symbol(shared_name),
    ));

    if tree_reduce {
        let shared_address = shared_address_for_lane(func, shared_base.clone(), lane.clone());
        func.add_inst(Inst::StSharedF32 {
            addr: shared_address,
            src: vec![accumulator.clone()],
            vec: super::instructions::VecWidth::Scalar,
        });
        func.add_inst(Inst::BarSync { barrier_id: 0 });

        let mut stride = thread_count / 2;
        while stride != 0 {
            let skip = next_label(ctx, "block_reduction_skip");
            let inactive = func.add_predicate_register();
            func.add_inst(Inst::setp_ge_u64(
                inactive.clone(),
                lane.clone(),
                Operand::imm_u64(stride),
            ));
            func.add_inst(Inst::Bra {
                condition: inactive,
                target: skip,
            });
            let peer_lane = func.add_u64_register();
            func.add_inst(Inst::add_u64(
                peer_lane.clone(),
                lane.clone(),
                Operand::imm_u64(stride),
            ));
            let peer_byte_offset = func.add_u64_register();
            func.add_inst(Inst::mul_u64(
                peer_byte_offset.clone(),
                peer_lane,
                Operand::imm_u64(std::mem::size_of::<f32>() as u64),
            ));
            let peer_address = func.add_u64_register();
            func.add_inst(Inst::add_u64(
                peer_address.clone(),
                shared_base.clone(),
                peer_byte_offset,
            ));
            let peer = func.add_f32_register();
            func.add_inst(Inst::LdSharedF32 {
                dst: vec![peer.clone()],
                addr: peer_address,
                vec: super::instructions::VecWidth::Scalar,
            });
            match region.op {
                crate::tensor::ReduceOp::Sum | crate::tensor::ReduceOp::Mean => func.add_inst(
                    Inst::add_f32(accumulator.clone(), accumulator.clone(), peer),
                ),
                crate::tensor::ReduceOp::Max => func.add_inst(Inst::max_f32(
                    accumulator.clone(),
                    accumulator.clone(),
                    peer,
                )),
            }
            let lane_address = shared_address_for_lane(func, shared_base.clone(), lane.clone());
            func.add_inst(Inst::StSharedF32 {
                addr: lane_address,
                src: vec![accumulator.clone()],
                vec: super::instructions::VecWidth::Scalar,
            });
            func.add_inst(Inst::Label(skip));
            func.add_inst(Inst::BarSync { barrier_id: 0 });
            stride /= 2;
        }
    } else {
        assert_eq!(
            thread_count, 32,
            "PTX subgroup reductions require one CUDA warp"
        );
        for offset in [1, 2, 4, 8, 16] {
            let accumulator_bits = func.add_b32_register();
            func.add_inst(Inst::mov_b32(accumulator_bits.clone(), accumulator.clone()));
            let shuffled_bits = func.add_b32_register();
            let valid = func.add_predicate_register();
            func.add_inst(Inst::shfl_sync_down_b32(
                shuffled_bits.clone(),
                valid.clone(),
                accumulator_bits,
                offset,
            ));
            let shuffled = func.add_f32_register();
            func.add_inst(Inst::mov_f32_b32(shuffled.clone(), shuffled_bits));
            let peer = func.add_f32_register();
            func.add_inst(Inst::selp_f32(
                peer.clone(),
                shuffled,
                Operand::imm_f32(identity),
                valid,
            ));
            match region.op {
                crate::tensor::ReduceOp::Sum | crate::tensor::ReduceOp::Mean => func.add_inst(
                    Inst::add_f32(accumulator.clone(), accumulator.clone(), peer),
                ),
                crate::tensor::ReduceOp::Max => func.add_inst(Inst::max_f32(
                    accumulator.clone(),
                    accumulator.clone(),
                    peer,
                )),
            }
        }

        let shuffle_done = next_label(ctx, "subgroup_reduction_shuffle_done");
        let not_leader = func.add_predicate_register();
        func.add_inst(Inst::setp_ne_u64(
            not_leader.clone(),
            lane.clone(),
            Operand::imm_u64(0),
        ));
        func.add_inst(Inst::Bra {
            condition: not_leader,
            target: shuffle_done,
        });
        func.add_inst(Inst::StSharedF32 {
            addr: shared_base.clone(),
            src: vec![accumulator.clone()],
            vec: super::instructions::VecWidth::Scalar,
        });
        func.add_inst(Inst::Label(shuffle_done));
        func.add_inst(Inst::BarSync { barrier_id: 0 });
    }

    let reduced_accumulator = func.add_f32_register();
    func.add_inst(Inst::LdSharedF32 {
        dst: vec![reduced_accumulator.clone()],
        addr: shared_base,
        vec: super::instructions::VecWidth::Scalar,
    });

    let reduced =
        if region.op == crate::tensor::ReduceOp::Mean && region.input_shape[region.axis] != 0 {
            let value = func.add_f32_register();
            func.add_inst(Inst::div_f32(
                value.clone(),
                reduced_accumulator,
                Operand::imm_f32(region.input_shape[region.axis] as f32),
            ));
            value
        } else {
            reduced_accumulator
        };
    values.clear();
    values.insert(region.reduced_value, reduced);
    for (index, input) in region.inputs.iter().enumerate() {
        if input.domain == ReductionInputDomain::Reduced {
            let offset =
                reduction_region_input_offset(func, input, &region.output_shape, fiber.clone());
            let value = load_param_f32_at(func, ctx, &format!("input_{index}"), offset);
            values.insert(input.value, value);
        }
    }
    for operation in &region.epilogue_operations {
        emit_cooperative_reduction_region_op(func, operation, &mut values);
    }
    let reduced_values = values.clone();

    let skip_store = next_label(ctx, "warp_reduction_skip_store");
    let not_leader = func.add_predicate_register();
    func.add_inst(Inst::setp_ne_u64(
        not_leader.clone(),
        lane.clone(),
        Operand::imm_u64(0),
    ));
    func.add_inst(Inst::Bra {
        condition: not_leader,
        target: skip_store,
    });
    for (index, (output, shape)) in region.outputs.iter().zip(&region.output_shapes).enumerate() {
        if shape != &region.input_shape {
            let value = values
                .get(&output.value)
                .expect("reduction region output SSA value is unavailable")
                .clone();
            store_param_f32_at(func, ctx, &format!("output_{index}"), fiber.clone(), value);
        }
    }
    func.add_inst(Inst::Label(skip_store));

    if !region.full_epilogue_operations.is_empty() {
        let full_counter = func.add_u64_register();
        func.add_inst(Inst::mov_u64(full_counter.clone(), lane));
        let full_start = next_label(ctx, "cooperative_full_epilogue_loop");
        let full_end = next_label(ctx, "cooperative_full_epilogue_end");
        func.add_inst(Inst::Label(full_start));
        let done = func.add_predicate_register();
        func.add_inst(Inst::setp_ge_u64(
            done.clone(),
            full_counter.clone(),
            Operand::imm_u64(region.input_shape[region.axis] as u64),
        ));
        func.add_inst(Inst::Bra {
            condition: done,
            target: full_end,
        });
        let contribution = func.add_u64_register();
        func.add_inst(Inst::mul_u64(
            contribution.clone(),
            full_counter.clone(),
            Operand::imm_u64(input_strides[region.axis] as u64),
        ));
        let full_offset = func.add_u64_register();
        func.add_inst(Inst::add_u64(
            full_offset.clone(),
            base_offset,
            contribution,
        ));
        values.clear();
        values.extend(reduced_values);
        for (index, input) in region.inputs.iter().enumerate() {
            if input.domain == ReductionInputDomain::Full {
                let offset = reduction_region_input_offset(
                    func,
                    input,
                    &region.input_shape,
                    full_offset.clone(),
                );
                let value = load_param_f32_at(func, ctx, &format!("input_{index}"), offset);
                values.insert(input.value, value);
            }
        }
        for operation in &region.producer_operations {
            emit_cooperative_reduction_region_op(func, operation, &mut values);
        }
        for operation in &region.full_epilogue_operations {
            emit_cooperative_reduction_region_op(func, operation, &mut values);
        }
        for (index, (output, shape)) in region.outputs.iter().zip(&region.output_shapes).enumerate()
        {
            if shape == &region.input_shape {
                let value = values
                    .get(&output.value)
                    .expect("full reduction epilogue output SSA value is unavailable")
                    .clone();
                store_param_f32_at(
                    func,
                    ctx,
                    &format!("output_{index}"),
                    full_offset.clone(),
                    value,
                );
            }
        }
        func.add_inst(Inst::add_u64(
            full_counter.clone(),
            full_counter,
            Operand::imm_u64(thread_count),
        ));
        func.add_inst(Inst::BraUni { target: full_start });
        func.add_inst(Inst::Label(full_end));
    }

    fn emit_cooperative_reduction_region_op<'a>(
        func: &mut Function<'a>,
        operation: &crate::tile::RegionOp,
        values: &mut HashMap<crate::tile::RegionValue, Operand<'a, F32>>,
    ) {
        let get = |value: crate::tile::RegionValue| {
            values
                .get(&value)
                .expect("reduction region operand SSA value is unavailable")
                .clone()
        };
        let output = func.add_f32_register();
        match operation.kind {
            RegionOpKind::Unary { op, input } => match op {
                crate::tensor::UnaryOp::Neg => {
                    func.add_inst(Inst::neg_f32(output.clone(), get(input)))
                }
                crate::tensor::UnaryOp::Exp => {
                    let scaled = func.add_f32_register();
                    func.add_inst(Inst::mul_f32(
                        scaled.clone(),
                        get(input),
                        Operand::imm_f32(std::f32::consts::LOG2_E),
                    ));
                    func.add_inst(Inst::ex2_f32(output.clone(), scaled));
                }
                crate::tensor::UnaryOp::Log => {
                    let logarithm = func.add_f32_register();
                    func.add_inst(Inst::lg2_f32(logarithm.clone(), get(input)));
                    func.add_inst(Inst::mul_f32(
                        output.clone(),
                        logarithm,
                        Operand::imm_f32(std::f32::consts::LN_2),
                    ));
                }
                crate::tensor::UnaryOp::Relu => func.add_inst(Inst::max_f32(
                    output.clone(),
                    get(input),
                    Operand::imm_f32(0.0),
                )),
            },
            RegionOpKind::Binary { op, lhs, rhs } => match op {
                crate::tensor::BinaryOp::Add => {
                    func.add_inst(Inst::add_f32(output.clone(), get(lhs), get(rhs)))
                }
                crate::tensor::BinaryOp::Sub => {
                    func.add_inst(Inst::sub_f32(output.clone(), get(lhs), get(rhs)))
                }
                crate::tensor::BinaryOp::Mul => {
                    func.add_inst(Inst::mul_f32(output.clone(), get(lhs), get(rhs)))
                }
                crate::tensor::BinaryOp::Div => {
                    func.add_inst(Inst::div_f32(output.clone(), get(lhs), get(rhs)))
                }
            },
            RegionOpKind::Gt { lhs, rhs } => {
                let predicate = func.add_predicate_register();
                func.add_inst(Inst::setp_gt_f32(predicate.clone(), get(lhs), get(rhs)));
                func.add_inst(Inst::selp_f32(
                    output.clone(),
                    Operand::imm_f32(1.0),
                    Operand::imm_f32(0.0),
                    predicate,
                ));
            }
            RegionOpKind::Mask {
                values: input,
                condition,
            } => {
                let predicate = func.add_predicate_register();
                func.add_inst(Inst::setp_ne_f32(
                    predicate.clone(),
                    get(condition),
                    Operand::imm_f32(0.0),
                ));
                func.add_inst(Inst::selp_f32(
                    output.clone(),
                    get(input),
                    Operand::imm_f32(0.0),
                    predicate,
                ));
            }
        }
        values.insert(operation.output, output);
    }
}

fn shared_address_for_lane<'a>(
    func: &mut Function<'a>,
    shared_base: Operand<'a, U64>,
    lane: Operand<'a, U64>,
) -> Operand<'a, U64> {
    let byte_offset = func.add_u64_register();
    func.add_inst(Inst::mul_u64(
        byte_offset.clone(),
        lane,
        Operand::imm_u64(std::mem::size_of::<f32>() as u64),
    ));
    let address = func.add_u64_register();
    func.add_inst(Inst::add_u64(address.clone(), shared_base, byte_offset));
    address
}

fn reduction_region_input_offset<'a>(
    func: &mut Function<'a>,
    input: &crate::tile::ReductionInput,
    logical_shape: &[usize],
    logical_linear: Operand<'a, U64>,
) -> Operand<'a, U64> {
    use crate::tile::{IndexExpr, IndexMap};

    if input.tensor.shape == input.source_shape
        && input.tensor.access == IndexMap::identity(input.tensor.shape.len())
    {
        return logical_linear;
    }
    assert_eq!(input.tensor.shape, logical_shape);
    let logical_strides = contiguous_strides(logical_shape);
    let coordinates: Vec<_> = logical_shape
        .iter()
        .enumerate()
        .map(|(dimension, &extent)| {
            decode_coordinate(
                func,
                logical_linear.clone(),
                logical_strides[dimension].max(1),
                extent.max(1),
            )
        })
        .collect();
    let source_strides = contiguous_strides(&input.source_shape);
    let offset = func.add_u64_register();
    func.add_inst(Inst::mov_u64(offset.clone(), Operand::imm_u64(0)));
    for (expression, &stride) in input
        .tensor
        .access
        .results
        .iter()
        .zip(source_strides.iter())
    {
        let coordinate = lower_index(expression, func, &coordinates);
        let contribution = func.add_u64_register();
        func.add_inst(Inst::mul_u64(
            contribution.clone(),
            coordinate,
            Operand::imm_u64(stride as u64),
        ));
        func.add_inst(Inst::add_u64(offset.clone(), offset.clone(), contribution));
    }
    return offset;

    fn lower_index<'a>(
        expression: &IndexExpr,
        func: &mut Function<'a>,
        coordinates: &[Operand<'a, U64>],
    ) -> Operand<'a, U64> {
        match expression {
            IndexExpr::IterDim(dimension) => coordinates[*dimension].clone(),
            IndexExpr::Symbol(symbol) => panic!("unsupported virtual index symbol {symbol}"),
            IndexExpr::Const(value) => Operand::imm_u64(*value as u64),
            IndexExpr::Add(lhs, rhs) => binary(func, lhs, rhs, coordinates, Inst::add_u64),
            IndexExpr::Sub(lhs, rhs) => binary(func, lhs, rhs, coordinates, Inst::sub_u64),
            IndexExpr::Mul(lhs, rhs) => binary(func, lhs, rhs, coordinates, Inst::mul_u64),
            IndexExpr::FloorDiv(value, divisor) => {
                let value = lower_index(value, func, coordinates);
                let result = func.add_u64_register();
                func.add_inst(Inst::div_u64(
                    result.clone(),
                    value,
                    Operand::imm_u64(*divisor as u64),
                ));
                result
            }
            IndexExpr::Mod(value, modulus) => {
                let value = lower_index(value, func, coordinates);
                let quotient = func.add_u64_register();
                func.add_inst(Inst::div_u64(
                    quotient.clone(),
                    value.clone(),
                    Operand::imm_u64(*modulus as u64),
                ));
                let consumed = func.add_u64_register();
                func.add_inst(Inst::mul_u64(
                    consumed.clone(),
                    quotient,
                    Operand::imm_u64(*modulus as u64),
                ));
                let result = func.add_u64_register();
                func.add_inst(Inst::sub_u64(result.clone(), value, consumed));
                result
            }
        }
    }

    fn binary<'a>(
        func: &mut Function<'a>,
        lhs: &crate::tile::IndexExpr,
        rhs: &crate::tile::IndexExpr,
        coordinates: &[Operand<'a, U64>],
        operation: impl FnOnce(Operand<'a, U64>, Operand<'a, U64>, Operand<'a, U64>) -> Inst<'a>,
    ) -> Operand<'a, U64> {
        let lhs = lower_index(lhs, func, coordinates);
        let rhs = lower_index(rhs, func, coordinates);
        let result = func.add_u64_register();
        func.add_inst(operation(result.clone(), lhs, rhs));
        result
    }
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

fn lower_tensor_core_matmul<'a>(
    func: &mut Function<'a>,
    ctx: &mut LoweringContext<'a>,
    dest: TileVar,
    a: TileVar,
    b: TileVar,
    schedule: &crate::tile::MatMulSchedule,
) {
    let accumulators = ctx
        .fragment_regs
        .get(&dest)
        .expect("TF32 MatMul destination is not a fragment")
        .clone();
    let a_shared = ctx.reg_to_shared.get(&a).copied().unwrap_or(a);
    let b_shared = ctx.reg_to_shared.get(&b).copied().unwrap_or(b);
    // Each computing warp owns one 16x16 fragment. All block threads stage
    // operands and participate in the surrounding barriers.
    let done =
        bumpalo::format!(in ctx.arena, "tf32_matmul_done_{}", ctx.label_counter).into_bump_str();
    ctx.label_counter += 1;
    skip_noncomputing_warps(
        func,
        schedule.block_threads.0,
        schedule.warp_topology.0 * schedule.warp_topology.1,
        done,
    );
    let (warp_row, warp_col) = warp_coordinates(schedule.block_threads.0, schedule.warp_topology);

    for k_offset in (0..schedule.block_tile.k).step_by(schedule.instruction_tile.k) {
        let (a_address, b_address) = if schedule.warp_topology == (1, 1)
            && shared_layout(ctx, a_shared).xor_mask == 0
            && shared_layout(ctx, b_shared).xor_mask == 0
        {
            (
                shared_address_with_byte_offset(
                    func,
                    ctx.shared_mem_ptrs[&a_shared].clone(),
                    (k_offset * schedule.operand_dtype.size_bytes()) as u64,
                ),
                shared_address_with_byte_offset(
                    func,
                    ctx.shared_mem_ptrs[&b_shared].clone(),
                    (k_offset
                        * shared_row_stride(ctx, b_shared)
                        * schedule.operand_dtype.size_bytes()) as u64,
                ),
            )
        } else {
            (
                shared_coordinate_address(
                    func,
                    ctx,
                    a_shared,
                    &warp_row,
                    &Expr::Const(k_offset as i64),
                ),
                shared_coordinate_address(
                    func,
                    ctx,
                    b_shared,
                    &Expr::Const(k_offset as i64),
                    &warp_col,
                ),
            )
        };
        let a_fragments: Vec<Operand<'a, B32>> =
            (0..if schedule.operand_dtype == crate::tile::DType::F16 {
                8
            } else {
                4
            })
                .map(|_| func.add_b32_register())
                .collect();
        let b_fragments: Vec<Operand<'a, B32>> =
            (0..if schedule.operand_dtype == crate::tile::DType::F16 {
                8
            } else {
                4
            })
                .map(|_| func.add_b32_register())
                .collect();
        func.add_inst(Inst::WmmaLoadA {
            dtype: schedule.operand_dtype,
            frags: a_fragments.clone(),
            addr: a_address,
            stride: Operand::imm_i32(shared_row_stride(ctx, a_shared) as i32),
        });
        func.add_inst(Inst::WmmaLoadB {
            dtype: schedule.operand_dtype,
            frags: b_fragments.clone(),
            addr: b_address,
            stride: Operand::imm_i32(shared_row_stride(ctx, b_shared) as i32),
        });
        if schedule.pipeline_stages == 2 && schedule.operand_dtype == DType::TF32 {
            // Async copies move storage bits. Round only the fragments consumed
            // by MMA, after the completed copy has been loaded from shared memory.
            for fragment in a_fragments.iter().chain(&b_fragments) {
                func.add_inst(Inst::ConvertTf32Bits {
                    dst: fragment.clone(),
                    src: fragment.clone(),
                });
            }
        }
        func.add_inst(Inst::WmmaMma {
            dtype: schedule.operand_dtype,
            d_frags: accumulators.clone(),
            a_frags: a_fragments,
            b_frags: b_fragments,
            c_frags: accumulators.clone(),
        });
    }
    func.add_inst(Inst::Label(done));
}

fn shared_layout(ctx: &LoweringContext<'_>, tile: TileVar) -> SharedLayout {
    match ctx.tile_layouts[&tile] {
        TileLayout::Shared(layout) => layout,
        _ => unreachable!("expected validated shared-memory tile layout"),
    }
}

fn shared_row_stride(ctx: &LoweringContext<'_>, tile: TileVar) -> usize {
    shared_layout(ctx, tile).row_stride
}

fn skip_noncomputing_warps<'a>(
    func: &mut Function<'a>,
    block_width: u32,
    warps: usize,
    done: &'a str,
) {
    let thread_y = func.add_u64_register();
    func.add_inst(Inst::convert_u64_u32(
        thread_y.clone(),
        super::instructions::THREAD_ID.y.clone(),
    ));
    let inactive = func.add_predicate_register();
    func.add_inst(Inst::setp_ge_u64(
        inactive.clone(),
        thread_y,
        Operand::imm_u64(warps as u64 * 32 / u64::from(block_width)),
    ));
    func.add_inst(Inst::Bra {
        condition: inactive,
        target: done,
    });
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

fn warp_coordinates(block_width: u32, topology: (usize, usize)) -> (Expr, Expr) {
    let warp = if block_width == 32 {
        Expr::ThreadIdx(Dim::Y)
    } else {
        Expr::FloorDiv(Box::new(Expr::ThreadIdx(Dim::Y)), 32 / block_width as usize)
    };
    let row = if topology.0 == 1 {
        Expr::Const(0)
    } else {
        Expr::ShiftRight(
            Box::new(warp.clone()),
            u64::from(topology.1.trailing_zeros()),
        ) * 16usize
    };
    let col = if topology.1 == 1 {
        Expr::Const(0)
    } else {
        Expr::BitAnd(Box::new(warp), (topology.1 - 1) as u64) * 16usize
    };
    (row, col)
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
    shared_coordinate_address(
        func,
        ctx,
        tile,
        &Expr::ThreadIdx(Dim::Y),
        &Expr::ThreadIdx(Dim::X),
    )
}

fn shared_coordinate_address<'a>(
    func: &mut Function<'a>,
    ctx: &LoweringContext<'a>,
    tile: TileVar,
    row: &Expr,
    column: &Expr,
) -> Operand<'a, U64> {
    let row = lower_expr(func, ctx, row);
    let column = lower_expr(func, ctx, column);
    shared_address(func, ctx, tile, row, column)
}

fn shared_address<'a>(
    func: &mut Function<'a>,
    ctx: &LoweringContext<'a>,
    tile: TileVar,
    row: Operand<'a, U64>,
    column: Operand<'a, U64>,
) -> Operand<'a, U64> {
    let layout = shared_layout(ctx, tile);
    let column = if layout.xor_mask == 0 {
        column
    } else {
        let bits = func.add_u64_register();
        func.add_inst(Inst::AndU64(AndInst::new(
            bits.clone(),
            row.clone(),
            Operand::imm_u64(layout.xor_mask as u64),
        )));
        let swizzled = func.add_u64_register();
        func.add_inst(Inst::XorU64 {
            dst: swizzled.clone(),
            a: column,
            b: bits,
        });
        swizzled
    };
    let shared = ctx
        .shared_mem_ptrs
        .get(&tile)
        .expect("Shared tile pointer not found")
        .clone();
    let columns = shared_row_stride(ctx, tile);
    let row_offset = func.add_u64_register();
    func.add_inst(Inst::mul_u64(
        row_offset.clone(),
        row,
        Operand::imm_u64(columns as u64),
    ));
    let element_offset = func.add_u64_register();
    func.add_inst(Inst::add_u64(element_offset.clone(), row_offset, column));
    let byte_offset = func.add_u64_register();
    func.add_inst(Inst::mul_u64(
        byte_offset.clone(),
        element_offset,
        Operand::imm_u64(ctx.tile_dtypes[&tile].size_bytes() as u64),
    ));
    let address = func.add_u64_register();
    func.add_inst(Inst::add_u64(address.clone(), shared, byte_offset));
    address
}

fn element_address<'a>(
    func: &mut Function<'a>,
    base: Operand<'a, U64>,
    element_offset: Operand<'a, U64>,
    dtype: crate::tile::DType,
) -> Operand<'a, U64> {
    let byte_offset = func.add_u64_register();
    func.add_inst(Inst::mul_u64(
        byte_offset.clone(),
        element_offset,
        Operand::imm_u64(dtype.size_bytes() as u64),
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
    let address = element_address(func, base, element_offset, ctx.param_dtypes[parameter]);
    let value = func.add_f32_register();
    load_global_as_f32(func, ctx.param_dtypes[parameter], value.clone(), address);
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
    let address = element_address(func, base, element_offset, ctx.param_dtypes[parameter]);
    store_global_from_f32(func, ctx.param_dtypes[parameter], address, value);
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

fn load_at<'a>(
    func: &mut Function<'a>,
    dtype: crate::tile::DType,
    destination: Operand<'a, F32>,
    base: Operand<'a, U64>,
    element_offset: Operand<'a, U64>,
) {
    let byte_offset = func.add_u64_register();
    func.add_inst(Inst::mul_u64(
        byte_offset.clone(),
        element_offset,
        Operand::imm_u64(dtype.size_bytes() as u64),
    ));
    let address = func.add_u64_register();
    func.add_inst(Inst::add_u64(address.clone(), base, byte_offset));
    load_global_as_f32(func, dtype, destination, address);
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
            if let Some(value) = ctx.index_vars.get(name) {
                return value.clone();
            }
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
        Expr::Sub(a, b) => {
            let lhs = lower_expr(func, ctx, a);
            let rhs = lower_expr(func, ctx, b);
            let result = func.add_u64_register();
            func.add_inst(Inst::sub_u64(result.clone(), lhs, rhs));
            result
        }
        Expr::FloorDiv(value, divisor) => {
            // The tile index pass normalizes and places expressions upstream.
            let value = lower_expr(func, ctx, value);
            let result = func.add_u64_register();
            func.add_inst(Inst::div_u64(
                result.clone(),
                value,
                Operand::imm_u64(*divisor as u64),
            ));
            result
        }
        Expr::ShiftRight(value, shift) => {
            let value = lower_expr(func, ctx, value);
            let result = func.add_u64_register();
            func.add_inst(Inst::ShiftRightU64(
                super::instructions::ShiftRightInst::new(
                    result.clone(),
                    value,
                    Operand::imm_i32(*shift as i32),
                ),
            ));
            result
        }
        Expr::BitAnd(value, mask) => {
            let value = lower_expr(func, ctx, value);
            let result = func.add_u64_register();
            func.add_inst(Inst::AndU64(AndInst::new(
                result.clone(),
                value,
                Operand::imm_u64(*mask),
            )));
            result
        }
        Expr::Mod(value, modulus) => {
            let value = lower_expr(func, ctx, value);
            let quotient = func.add_u64_register();
            func.add_inst(Inst::div_u64(
                quotient.clone(),
                value.clone(),
                Operand::imm_u64(*modulus as u64),
            ));
            let consumed = func.add_u64_register();
            func.add_inst(Inst::mul_u64(
                consumed.clone(),
                quotient,
                Operand::imm_u64(*modulus as u64),
            ));
            let result = func.add_u64_register();
            func.add_inst(Inst::sub_u64(result.clone(), value, consumed));
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
            if let Some(value) = ctx.index_vars.get(name) {
                let result = func.add_i32_register();
                func.add_inst(Inst::convert_i32_u64(result.clone(), value.clone()));
                return result;
            }
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

fn emit_region_op<'a>(
    func: &mut Function<'a>,
    operation: &crate::tile::RegionOp,
    values: &mut HashMap<crate::tile::RegionValue, Operand<'a, F32>>,
) {
    use crate::tile::RegionOpKind;
    let get = |value: crate::tile::RegionValue| {
        values
            .get(&value)
            .expect("reduction region operand SSA value is unavailable")
            .clone()
    };
    let output = func.add_f32_register();
    match operation.kind {
        RegionOpKind::Unary { op, input } => match op {
            crate::tensor::UnaryOp::Neg => func.add_inst(Inst::neg_f32(output.clone(), get(input))),
            crate::tensor::UnaryOp::Exp => {
                let scaled = func.add_f32_register();
                func.add_inst(Inst::mul_f32(
                    scaled.clone(),
                    get(input),
                    Operand::imm_f32(std::f32::consts::LOG2_E),
                ));
                func.add_inst(Inst::ex2_f32(output.clone(), scaled));
            }
            crate::tensor::UnaryOp::Log => {
                let logarithm = func.add_f32_register();
                func.add_inst(Inst::lg2_f32(logarithm.clone(), get(input)));
                func.add_inst(Inst::mul_f32(
                    output.clone(),
                    logarithm,
                    Operand::imm_f32(std::f32::consts::LN_2),
                ));
            }
            crate::tensor::UnaryOp::Relu => func.add_inst(Inst::max_f32(
                output.clone(),
                get(input),
                Operand::imm_f32(0.0),
            )),
        },
        RegionOpKind::Binary { op, lhs, rhs } => match op {
            crate::tensor::BinaryOp::Add => {
                func.add_inst(Inst::add_f32(output.clone(), get(lhs), get(rhs)))
            }
            crate::tensor::BinaryOp::Sub => {
                func.add_inst(Inst::sub_f32(output.clone(), get(lhs), get(rhs)))
            }
            crate::tensor::BinaryOp::Mul => {
                func.add_inst(Inst::mul_f32(output.clone(), get(lhs), get(rhs)))
            }
            crate::tensor::BinaryOp::Div => {
                func.add_inst(Inst::div_f32(output.clone(), get(lhs), get(rhs)))
            }
        },
        RegionOpKind::Gt { lhs, rhs } => {
            let predicate = func.add_predicate_register();
            func.add_inst(Inst::setp_gt_f32(predicate.clone(), get(lhs), get(rhs)));
            func.add_inst(Inst::selp_f32(
                output.clone(),
                Operand::imm_f32(1.0),
                Operand::imm_f32(0.0),
                predicate,
            ));
        }
        RegionOpKind::Mask {
            values: input,
            condition,
        } => {
            let predicate = func.add_predicate_register();
            func.add_inst(Inst::setp_ne_f32(
                predicate.clone(),
                get(condition),
                Operand::imm_f32(0.0),
            ));
            func.add_inst(Inst::selp_f32(
                output.clone(),
                get(input),
                Operand::imm_f32(0.0),
                predicate,
            ));
        }
    }
    values.insert(operation.output, output);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::TensorGraph;
    use crate::ptx::{PtxExecutionPlan, PtxPlanAction, PtxReductionMode};
    use crate::tensor::TensorExpr;
    use crate::tile::{
        Block, DType, KernelParam, MatMulPipeline, MatMulPrecision, MatMulSharedLayout,
        MemorySpace, Stmt, TileIR,
    };

    struct TestFunction {
        // Drop the arena after the function and its bump-backed collections.
        function: Function<'static>,
        _arena: Box<bumpalo::Bump>,
    }

    impl std::ops::Deref for TestFunction {
        type Target = Function<'static>;

        fn deref(&self) -> &Self::Target {
            &self.function
        }
    }

    impl std::fmt::Display for TestFunction {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            self.function.fmt(formatter)
        }
    }

    fn tile_ir_into_function(tile_ir: TileIR) -> TestFunction {
        let arena = Box::new(bumpalo::Bump::new());
        // SAFETY: The boxed arena has a stable address and TestFunction drops the
        // function before the arena.
        let arena_ref: &'static bumpalo::Bump =
            unsafe { &*(arena.as_ref() as *const bumpalo::Bump) };
        let function = tile_ir_to_function(tile_ir, arena_ref, 0);
        TestFunction {
            function,
            _arena: arena,
        }
    }

    #[test]
    fn pipeline_codegen_orders_wait_publish_prefetch_compute_and_retire() {
        for sm in [(7, 5), (8, 0)] {
            let target = PtxTarget {
                multiprocessor_count: 1,
                compute_capability: sm,
                matmul_resources: Default::default(),
            };
            for policy in [MatMulPipeline::Synchronous, MatMulPipeline::DoubleBuffered] {
                let a = TensorExpr::constant(vec![0.25; 33 * 65], vec![33, 65]);
                let b = TensorExpr::constant(vec![0.5; 65 * 35], vec![65, 35]);
                let graph: TensorGraph<f32> = a.matmul(b).relu().into();
                let mut plan =
                    PtxExecutionPlan::build_with_reduction_mode(&graph, PtxReductionMode::Strict)
                        .unwrap();
                plan.schedule_matmuls(
                    target.matmul_capabilities(),
                    MatMulPrecision::AllowTf32,
                    target.matmul_resources,
                    MatMulSharedLayout::Auto,
                    policy,
                    target.supports_async_copy(),
                )
                .unwrap();
                let mut tiles = TileGraph::from_with_matmul_schedules(
                    &graph,
                    target.supports_tf32(),
                    plan.matmul_schedules(),
                );
                tiles.add_matmul_regions(plan.matmul_regions()).unwrap();
                tiles.set_physical_nodes(
                    plan.steps()
                        .iter()
                        .filter_map(|step| {
                            (step.action == PtxPlanAction::Kernel).then_some(step.node)
                        })
                        .collect(),
                );
                tiles.optimize_indices().unwrap();
                let source = PtxGraph::from(tiles).with_target(target).module_source();
                let pipelined =
                    target.supports_async_copy() && policy == MatMulPipeline::DoubleBuffered;
                assert_eq!(source.contains("cp.async"), pipelined);
                if !pipelined {
                    continue;
                }
                let start = source.find("loop_body_k_tile:").unwrap();
                let end = source[start..].find("loop_end_k_tile:").unwrap() + start;
                let body = &source[start..end];
                let wait = body.find("cp.async.wait_group 0;").unwrap();
                let publish = body[wait..].find("bar.sync 0;").unwrap() + wait;
                let copy = body[publish..].find("cp.async.ca.shared.global").unwrap() + publish;
                let commit = body[copy..].find("cp.async.commit_group;").unwrap() + copy;
                let load = body[commit..].find("wmma.load.a.").unwrap() + commit;
                let rounding = body[load..].find("cvt.rna.tf32.f32").unwrap() + load;
                let compute = body[rounding..].find("wmma.mma.").unwrap() + rounding;
                assert!(!body[compute..].contains("bar.sync 0;"));
                assert_eq!(body.matches("bar.sync 0;").count(), 1);
                let drain = &source[end..];
                assert!(
                    drain.find("cp.async.wait_group 0;").unwrap()
                        < drain.find("bar.sync 0;").unwrap()
                );
                assert!(
                    drain.find("cp.async.wait_group 0;").unwrap()
                        < drain.find("wmma.store.d.").unwrap()
                );
                assert_eq!(plan.matmul_regions()[0].schedule.pipeline_stages, 2);
            }
        }
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
    fn matmul_plan_and_ptx_header_follow_target_capability() {
        for (target, expect_tf32) in [
            (
                PtxTarget {
                    multiprocessor_count: 1,
                    matmul_resources: Default::default(),
                    compute_capability: (7, 5),
                },
                false,
            ),
            (
                PtxTarget {
                    multiprocessor_count: 1,
                    matmul_resources: Default::default(),
                    compute_capability: (8, 0),
                },
                true,
            ),
        ] {
            let lhs = TensorExpr::constant(vec![0.25; 32 * 32], vec![32, 32]);
            let rhs = TensorExpr::constant(vec![0.5; 32 * 32], vec![32, 32]);
            let graph: TensorGraph<f32> = lhs.matmul(rhs).into();
            let plan =
                PtxExecutionPlan::build_with_reduction_mode(&graph, PtxReductionMode::Strict)
                    .unwrap();
            let mut tile_graph = TileGraph::from_with_tf32_support(graph, target.supports_tf32());
            tile_graph.set_physical_nodes(
                plan.steps()
                    .iter()
                    .filter_map(|step| {
                        (step.action == crate::ptx::PtxPlanAction::Kernel).then_some(step.node)
                    })
                    .collect(),
            );
            let source = PtxGraph::from(tile_graph)
                .with_target(target)
                .module_source();

            assert!(source.contains(&format!(
                ".target sm_{}{}",
                target.compute_capability.0, target.compute_capability.1
            )));
            assert_eq!(source.contains("wmma.mma.sync"), expect_tf32);
            assert_eq!(source.contains("cvt.rna.tf32.f32"), expect_tf32);
        }
    }

    #[test]
    fn fused_batched_matmul_codegen_uses_target_schedule_and_one_entry() {
        use crate::tile::{MatMulPlan, MatMulPrecision};
        for compute_capability in [(6, 1), (7, 5), (8, 0), (8, 9)] {
            for precision in [MatMulPrecision::AllowTf32, MatMulPrecision::StrictF32] {
                let target = super::super::target::PtxTarget {
                    multiprocessor_count: 1,
                    compute_capability,
                    matmul_resources: Default::default(),
                };
                let lhs = TensorExpr::constant(vec![0.25; 2 * 32 * 32], vec![2, 32, 32]);
                let rhs = TensorExpr::constant(vec![0.5; 32 * 32], vec![32, 32]);
                let bias = TensorExpr::constant(vec![0.1; 32], vec![32]);
                let graph: TensorGraph<f32> = (lhs.matmul(rhs) + bias).relu().into();
                let mut plan =
                    PtxExecutionPlan::build_with_reduction_mode(&graph, PtxReductionMode::Strict)
                        .unwrap();
                plan.schedule_matmuls(
                    target.matmul_capabilities(),
                    precision,
                    target.matmul_resources,
                    MatMulSharedLayout::Auto,
                    MatMulPipeline::Auto,
                    target.supports_async_copy(),
                )
                .unwrap();
                assert_eq!(plan.matmul_regions().len(), 1);
                let schedule = plan.matmul_regions()[0].schedule;
                let expect_tf32 = target.supports_tf32() && precision == MatMulPrecision::AllowTf32;
                assert_eq!(schedule.plan == MatMulPlan::TensorCoreTf32, expect_tf32);
                let mut tile_graph = TileGraph::from_with_matmul_schedules(
                    &graph,
                    expect_tf32,
                    plan.matmul_schedules(),
                );
                tile_graph
                    .add_matmul_regions(plan.matmul_regions())
                    .unwrap();
                tile_graph.set_physical_nodes(
                    plan.steps()
                        .iter()
                        .filter_map(|step| {
                            (step.action == crate::ptx::PtxPlanAction::Kernel).then_some(step.node)
                        })
                        .collect(),
                );
                let source = PtxGraph::from(tile_graph)
                    .with_target(target)
                    .module_source();
                assert_eq!(source.matches(".visible .entry").count(), 1);
                assert_eq!(
                    source.contains("wmma.mma.sync.aligned.m16n16k8"),
                    expect_tf32
                );
                assert_eq!(source.contains("cvt.rna.tf32.f32"), expect_tf32);
                assert!(source.contains("%ctaid.z"));
                assert!(source.contains("add.f32"));
                assert!(source.contains("max.f32"));
            }
        }
    }

    #[test]
    fn shuffle_down_renders_predicate_and_full_warp_mask() {
        let arena = bumpalo::Bump::new();
        let mut function = Function::new("shuffle", &arena);
        let src = function.add_b32_register();
        let dst = function.add_b32_register();
        let predicate = function.add_predicate_register();
        function.add_inst(Inst::shfl_sync_down_b32(dst, predicate, src, 16));

        assert!(
            format!("{function}")
                .contains("shfl.sync.down.b32 %r1|%p0, %r0, 16, 0x1f, 0xffffffff;")
        );
    }

    #[test]
    fn subgroup_reduction_emits_shuffle_tree() {
        let input = TensorExpr::constant(vec![0.25; 2 * 127], vec![2, 127]);
        let graph: TensorGraph<f32> = input.exp().reduce_sum(1).into();
        let plan = PtxExecutionPlan::build_with_reduction_mode(
            &graph,
            PtxReductionMode::DeterministicTree,
        )
        .unwrap();
        assert_eq!(
            plan.reduction_regions()[0].schedule,
            crate::tile::ReductionSchedule::Subgroup { width: 32 }
        );
        let mut tile_graph = TileGraph::from(graph);
        tile_graph
            .add_reduction_regions(plan.reduction_regions())
            .unwrap();
        let source = PtxGraph::from(tile_graph).module_source();

        assert_eq!(source.matches("shfl.sync.down.b32").count(), 5);
        for offset in [1, 2, 4, 8, 16] {
            assert!(
                source.lines().any(|line| {
                    line.contains("shfl.sync.down.b32")
                        && line.contains(&format!(", {offset}, 0x1f, 0xffffffff;"))
                }),
                "missing shuffle stage for offset {offset}"
            );
        }
        assert!(source.contains(".shared .align 32 .b8 reduction_partials[4];"));
        assert_eq!(source.matches("bar.sync 0;").count(), 1);
    }

    #[test]
    fn block_reduction_emits_shared_tree() {
        let input = TensorExpr::constant(vec![0.25; 2 * 1024], vec![2, 1024]);
        let graph: TensorGraph<f32> = input.exp().reduce_sum(1).into();
        let plan = PtxExecutionPlan::build_with_reduction_mode(
            &graph,
            PtxReductionMode::DeterministicTree,
        )
        .unwrap();
        let mut tile_graph = TileGraph::from(graph);
        tile_graph
            .add_reduction_regions(plan.reduction_regions())
            .unwrap();
        let source = PtxGraph::from(tile_graph).module_source();

        assert!(source.contains(".shared .align 32 .b8 reduction_partials[512];"));
        assert_eq!(source.matches("bar.sync 0;").count(), 8);
        assert!(source.contains("ld.shared.f32"));
        assert!(!source.contains("shfl.sync"));
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
                        layout: TileLayout::ThreadScalar,
                        dtype: DType::F32,
                        rows: 1,
                        cols: 1,
                    },
                    Stmt::AllocTile {
                        var: TileVar(1),
                        space: MemorySpace::Register,
                        layout: TileLayout::ThreadScalar,
                        dtype: DType::F32,
                        rows: 1,
                        cols: 1,
                    },
                    Stmt::AllocTile {
                        var: TileVar(2),
                        space: MemorySpace::Register,
                        layout: TileLayout::ThreadScalar,
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
        let func = tile_ir_into_function(tile_ir);

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
                        layout: TileLayout::ThreadScalar,
                        dtype: DType::F32,
                        rows: 1,
                        cols: 1,
                    },
                    Stmt::AllocTile {
                        var: TileVar(1),
                        space: MemorySpace::Register,
                        layout: TileLayout::ThreadScalar,
                        dtype: DType::F32,
                        rows: 1,
                        cols: 1,
                    },
                    Stmt::AllocTile {
                        var: TileVar(2),
                        space: MemorySpace::Register,
                        layout: TileLayout::ThreadScalar,
                        dtype: DType::F32,
                        rows: 1,
                        cols: 1,
                    },
                    Stmt::AllocTile {
                        var: TileVar(3),
                        space: MemorySpace::Register,
                        layout: TileLayout::ThreadScalar,
                        dtype: DType::F32,
                        rows: 1,
                        cols: 1,
                    },
                    Stmt::AllocTile {
                        var: TileVar(4),
                        space: MemorySpace::Register,
                        layout: TileLayout::ThreadScalar,
                        dtype: DType::F32,
                        rows: 1,
                        cols: 1,
                    },
                    Stmt::AllocTile {
                        var: TileVar(5),
                        space: MemorySpace::Register,
                        layout: TileLayout::ThreadScalar,
                        dtype: DType::F32,
                        rows: 1,
                        cols: 1,
                    },
                    Stmt::AllocTile {
                        var: TileVar(6),
                        space: MemorySpace::Register,
                        layout: TileLayout::ThreadScalar,
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

        let func = tile_ir_into_function(tile_ir);

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
                        layout: TileLayout::ThreadScalar,
                        dtype: DType::F32,
                        rows: 1,
                        cols: 1,
                    },
                    Stmt::AllocTile {
                        var: TileVar(1),
                        space: MemorySpace::Register,
                        layout: TileLayout::ThreadScalar,
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

        let func = tile_ir_into_function(tile_ir);

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
                        layout: TileLayout::ThreadScalar,
                        dtype: DType::F32,
                        rows: 1,
                        cols: 1,
                    },
                    Stmt::AllocTile {
                        var: TileVar(1),
                        space: MemorySpace::Register,
                        layout: TileLayout::ThreadScalar,
                        dtype: DType::F32,
                        rows: 1,
                        cols: 1,
                    },
                    Stmt::AllocTile {
                        var: TileVar(2),
                        space: MemorySpace::Register,
                        layout: TileLayout::ThreadScalar,
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

        let func = tile_ir_into_function(tile_ir);

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
                    layout: TileLayout::Shared(SharedLayout::row_major(16)),
                    dtype: DType::F32,
                    rows: 16,
                    cols: 16,
                }],
            },
        };

        let func = tile_ir_into_function(tile_ir);

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

        let func = tile_ir_into_function(tile_ir);

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
                        layout: TileLayout::ThreadScalar,
                        dtype: DType::F32,
                        rows: 1,
                        cols: 1,
                    },
                    Stmt::AllocTile {
                        var: TileVar(1),
                        space: MemorySpace::Register,
                        layout: TileLayout::ThreadScalar,
                        dtype: DType::F32,
                        rows: 1,
                        cols: 1,
                    },
                    Stmt::AllocTile {
                        var: TileVar(2),
                        space: MemorySpace::Register,
                        layout: TileLayout::ThreadScalar,
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

        let func = tile_ir_into_function(tile_ir);

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
                        layout: TileLayout::ThreadScalar,
                        dtype: DType::F32,
                        rows: 1,
                        cols: 1,
                    },
                    Stmt::Zero { tile: TileVar(0) },
                ],
            },
        };

        let func = tile_ir_into_function(tile_ir);

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
                        layout: TileLayout::ThreadScalar,
                        dtype: DType::F32,
                        rows: 1,
                        cols: 1,
                    },
                    Stmt::AllocTile {
                        var: TileVar(1), // a_val
                        space: MemorySpace::Register,
                        layout: TileLayout::ThreadScalar,
                        dtype: DType::F32,
                        rows: 1,
                        cols: 1,
                    },
                    Stmt::AllocTile {
                        var: TileVar(2), // b_val
                        space: MemorySpace::Register,
                        layout: TileLayout::ThreadScalar,
                        dtype: DType::F32,
                        rows: 1,
                        cols: 1,
                    },
                    Stmt::AllocTile {
                        var: TileVar(3), // temp = a * x
                        space: MemorySpace::Register,
                        layout: TileLayout::ThreadScalar,
                        dtype: DType::F32,
                        rows: 1,
                        cols: 1,
                    },
                    Stmt::AllocTile {
                        var: TileVar(4), // temp2 = temp + b
                        space: MemorySpace::Register,
                        layout: TileLayout::ThreadScalar,
                        dtype: DType::F32,
                        rows: 1,
                        cols: 1,
                    },
                    Stmt::AllocTile {
                        var: TileVar(5), // result = relu(temp2)
                        space: MemorySpace::Register,
                        layout: TileLayout::ThreadScalar,
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

        let func = tile_ir_into_function(tile_ir);

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
