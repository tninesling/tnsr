use super::ir::{
    Block, DType, Expr, KernelParam, MatMulLayout, MemorySpace, ReduceOp, Stmt, TileIR, TileVar,
};

#[allow(dead_code)]
#[derive(Default)]
pub struct TileIRBuilder {
    kernel_name: String,
    params: Vec<KernelParam>,
    stmts: Vec<Stmt>,
    next_var: usize,
    shared_mem_bytes: usize,
}

impl TileIRBuilder {
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self::default()
    }

    #[allow(dead_code)]
    pub fn start_kernel(&mut self, name: &str) {
        self.kernel_name = name.to_string();
    }

    #[allow(dead_code)]
    pub fn add_param(&mut self, name: &str, dtype: DType, is_input: bool) {
        self.params.push(KernelParam {
            name: name.to_string(),
            dtype,
            is_input,
        });
    }

    #[allow(dead_code)]
    pub fn alloc_shared(&mut self, dtype: DType, rows: usize, cols: usize) -> TileVar {
        let var = TileVar(self.next_var);
        self.next_var += 1;

        self.stmts.push(Stmt::AllocTile {
            var,
            space: MemorySpace::Shared,
            dtype,
            rows,
            cols,
        });

        // Track shared memory usage
        self.shared_mem_bytes += rows * cols * dtype.size_bytes();

        var
    }

    #[allow(dead_code)]
    pub fn alloc_register(&mut self, dtype: DType, rows: usize, cols: usize) -> TileVar {
        let var = TileVar(self.next_var);
        self.next_var += 1;

        self.stmts.push(Stmt::AllocTile {
            var,
            space: MemorySpace::Register,
            dtype,
            rows,
            cols,
        });

        var
    }

    #[allow(dead_code)]
    pub fn zero(&mut self, tile: TileVar) {
        self.stmts.push(Stmt::Zero { tile });
    }

    #[allow(dead_code)]
    pub fn matmul(&mut self, dest: TileVar, a: TileVar, b: TileVar, layout: MatMulLayout) {
        self.stmts.push(Stmt::MatMul { dest, a, b, layout });
    }

    #[allow(dead_code)]
    pub fn barrier(&mut self) {
        self.stmts.push(Stmt::Barrier);
    }

    #[allow(dead_code)]
    pub fn load_global_to_shared(
        &mut self,
        dest: TileVar,
        src_param: &str,
        row_offset: Expr,
        col_offset: Expr,
    ) {
        self.stmts.push(Stmt::Load {
            dest,
            src_param: src_param.to_string(),
            row_offset,
            col_offset,
        });
    }

    #[allow(dead_code)]
    pub fn load_shared_to_register(&mut self, dest: TileVar, src: TileVar) {
        self.stmts.push(Stmt::LoadSharedToReg { dest, src });
    }

    #[allow(dead_code)]
    pub fn store(&mut self, dest_param: &str, src: TileVar, row_offset: Expr, col_offset: Expr) {
        self.stmts.push(Stmt::Store {
            dest_param: dest_param.to_string(),
            src,
            row_offset,
            col_offset,
        });
    }

    #[allow(dead_code)]
    pub fn add(&mut self, dest: TileVar, a: TileVar, b: TileVar) {
        self.stmts.push(Stmt::Add { dest, a, b });
    }

    #[allow(dead_code)]
    pub fn sub(&mut self, dest: TileVar, a: TileVar, b: TileVar) {
        self.stmts.push(Stmt::Sub { dest, a, b });
    }

    #[allow(dead_code)]
    pub fn mul(&mut self, dest: TileVar, a: TileVar, b: TileVar) {
        self.stmts.push(Stmt::Mul { dest, a, b });
    }

    #[allow(dead_code)]
    pub fn div(&mut self, dest: TileVar, a: TileVar, b: TileVar) {
        self.stmts.push(Stmt::Div { dest, a, b });
    }

    #[allow(dead_code)]
    pub fn neg(&mut self, dest: TileVar, src: TileVar) {
        self.stmts.push(Stmt::Neg { dest, src });
    }

    #[allow(dead_code)]
    pub fn exp(&mut self, dest: TileVar, src: TileVar) {
        self.stmts.push(Stmt::Exp { dest, src });
    }

    #[allow(dead_code)]
    pub fn log(&mut self, dest: TileVar, src: TileVar) {
        self.stmts.push(Stmt::Log { dest, src });
    }

    #[allow(dead_code)]
    pub fn relu(&mut self, dest: TileVar, src: TileVar) {
        self.stmts.push(Stmt::Relu { dest, src });
    }

    #[allow(dead_code)]
    pub fn transpose(
        &mut self,
        dest: TileVar,
        src: TileVar,
        input_shape: Vec<usize>,
        output_shape: Vec<usize>,
    ) {
        self.stmts.push(Stmt::Transpose {
            dest,
            src,
            input_shape,
            output_shape,
        });
    }

    #[allow(dead_code)]
    pub fn broadcast_axis(
        &mut self,
        dest: TileVar,
        src: TileVar,
        axis: usize,
        input_shape: Vec<usize>,
        output_shape: Vec<usize>,
    ) {
        self.stmts.push(Stmt::BroadcastAxis {
            dest,
            src,
            axis,
            input_shape,
            output_shape,
        });
    }

    #[allow(dead_code)]
    pub fn reduce_axis(
        &mut self,
        dest: TileVar,
        src: TileVar,
        op: ReduceOp,
        axis: usize,
        input_shape: Vec<usize>,
        output_shape: Vec<usize>,
    ) {
        self.stmts.push(Stmt::ReduceAxis {
            dest,
            src,
            op,
            axis,
            input_shape,
            output_shape,
        });
    }

    #[allow(dead_code)]
    pub fn gt(&mut self, dest: TileVar, a: TileVar, b: TileVar) {
        self.stmts.push(Stmt::Gt { dest, a, b });
    }

    #[allow(dead_code)]
    pub fn mask(&mut self, dest: TileVar, values: TileVar, condition: TileVar) {
        self.stmts.push(Stmt::Mask {
            dest,
            values,
            condition,
        });
    }

    #[allow(dead_code)]
    pub fn for_loop<F>(&mut self, var_name: &str, start: i64, end: i64, body_fn: F)
    where
        F: FnOnce(&mut Self, String),
    {
        let var = var_name.to_string();

        // Build body in temporary builder
        let mut body_builder = TileIRBuilder::new();
        body_builder.next_var = self.next_var;
        body_fn(&mut body_builder, var.clone());
        self.next_var = body_builder.next_var;

        self.stmts.push(Stmt::ForLoop {
            loop_var: var,
            start: Expr::Const(start),
            end: Expr::Const(end),
            body: Block {
                stmts: body_builder.stmts,
            },
        });
    }

    #[allow(dead_code)]
    pub fn finish(self) -> TileIR {
        TileIR {
            kernel_name: self.kernel_name,
            params: self.params,
            body: Block { stmts: self.stmts },
            shared_mem_bytes: self.shared_mem_bytes,
        }
    }
}
