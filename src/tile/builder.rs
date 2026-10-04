use super::MatMulSchedule;
use super::ir::{
    Block, DType, Dim, Expr, KernelParam, MatMulLayout, MatrixLayout, MemorySpace, ReduceOp, Stmt,
    TileIR, TileLayout, TileVar,
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
            layout: TileLayout::SharedRowMajor { row_stride: cols },
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
            layout: TileLayout::ThreadScalar,
            dtype,
            rows,
            cols,
        });

        var
    }

    pub fn alloc_fragment(
        &mut self,
        dtype: DType,
        rows: usize,
        cols: usize,
        schedule: MatMulSchedule,
    ) -> TileVar {
        let var = TileVar(self.next_var);
        self.next_var += 1;
        self.stmts.push(Stmt::AllocTile {
            var,
            space: MemorySpace::Fragment,
            layout: schedule.accumulator_tile_layout(),
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
    pub fn matmul(
        &mut self,
        dest: TileVar,
        a: TileVar,
        b: TileVar,
        layout: MatMulLayout,
        schedule: MatMulSchedule,
    ) {
        self.stmts.push(Stmt::MatMul {
            dest,
            a,
            b,
            layout,
            schedule,
        });
    }

    pub fn convert_layout(&mut self, dest: TileVar, src: TileVar) {
        self.stmts.push(Stmt::ConvertLayout {
            dest,
            src,
            coordinates: None,
        });
    }

    pub fn extract_scalar(&mut self, dest: TileVar, src: TileVar, coordinates: (Expr, Expr)) {
        self.stmts.push(Stmt::ConvertLayout {
            dest,
            src,
            coordinates: Some(coordinates),
        });
    }

    pub fn load_global_predicated(
        &mut self,
        dest: TileVar,
        src_param: &str,
        element_index: Expr,
        row: Expr,
        col: Expr,
        layout: MatrixLayout,
    ) {
        self.stmts.push(Stmt::LoadGlobalPredicated {
            dest,
            src_param: src_param.to_string(),
            element_index,
            row,
            col,
            layout,
        });
    }

    pub fn embedding(&mut self, vocabulary: usize, width: usize, index_count: usize) {
        self.stmts.push(Stmt::Embedding {
            vocabulary,
            width,
            index_count,
        });
    }

    pub fn embedding_backward(&mut self, vocabulary: usize, width: usize, index_count: usize) {
        self.stmts.push(Stmt::EmbeddingBackward {
            vocabulary,
            width,
            index_count,
        });
    }

    pub fn indexed_cross_entropy(&mut self, vocabulary: usize, row_count: usize) {
        self.stmts.push(Stmt::IndexedCrossEntropy {
            vocabulary,
            row_count,
        });
    }

    pub fn indexed_cross_entropy_backward(&mut self, vocabulary: usize, row_count: usize) {
        self.stmts.push(Stmt::IndexedCrossEntropyBackward {
            vocabulary,
            row_count,
        });
    }

    pub fn conv2d(&mut self, geometry: super::ir::Conv2dGeometry) {
        self.stmts.push(Stmt::Conv2d { geometry });
    }

    pub fn conv_transpose2d(&mut self, geometry: super::ir::Conv2dGeometry) {
        self.stmts.push(Stmt::ConvTranspose2d { geometry });
    }

    pub fn conv2d_backward_weight(&mut self, geometry: super::ir::Conv2dGeometry) {
        self.stmts.push(Stmt::Conv2dBackwardWeight { geometry });
    }

    pub fn max_pool2d(&mut self, geometry: super::ir::MaxPool2dGeometry) {
        self.stmts.push(Stmt::MaxPool2d { geometry });
    }

    pub fn max_pool2d_backward(&mut self, geometry: super::ir::MaxPool2dGeometry) {
        self.stmts.push(Stmt::MaxPool2dBackward { geometry });
    }

    #[allow(dead_code)]
    pub fn barrier(&mut self) {
        self.stmts.push(Stmt::Barrier);
    }

    pub fn bounds_check(&mut self, extent: usize) {
        self.stmts.push(Stmt::BoundsCheck { extent });
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

    pub fn load_global_to_shared_predicated(
        &mut self,
        dest: TileVar,
        src_param: &str,
        row: Expr,
        col: Expr,
        layout: MatrixLayout,
    ) {
        self.stmts.push(Stmt::LoadGlobalToSharedPredicated {
            dest,
            src_param: src_param.to_string(),
            element_index: row.clone() * layout.row_stride + col.clone(),
            row,
            col,
            tile_row: Expr::ThreadIdx(Dim::Y),
            tile_col: Expr::ThreadIdx(Dim::X),
            layout,
        });
    }

    pub fn load_global_to_shared_indexed(
        &mut self,
        dest: TileVar,
        src_param: &str,
        element_index: Expr,
        bounds: (Expr, Expr),
        tile_coordinates: (Expr, Expr),
        layout: MatrixLayout,
    ) {
        self.stmts.push(Stmt::LoadGlobalToSharedPredicated {
            dest,
            src_param: src_param.to_string(),
            element_index,
            row: bounds.0,
            col: bounds.1,
            tile_row: tile_coordinates.0,
            tile_col: tile_coordinates.1,
            layout,
        });
    }

    pub fn store_global_indexed(
        &mut self,
        dest_param: &str,
        src: TileVar,
        element_index: Expr,
        bounds: (Expr, Expr),
        layout: MatrixLayout,
    ) {
        self.stmts.push(Stmt::StoreGlobalPredicated {
            dest_param: dest_param.to_string(),
            src,
            element_index,
            row: bounds.0,
            col: bounds.1,
            layout,
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

    pub fn store_global_predicated(
        &mut self,
        dest_param: &str,
        src: TileVar,
        row: Expr,
        col: Expr,
        layout: MatrixLayout,
    ) {
        self.stmts.push(Stmt::StoreGlobalPredicated {
            dest_param: dest_param.to_string(),
            src,
            element_index: row.clone() * layout.row_stride + col.clone(),
            row,
            col,
            layout,
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
    pub fn reindex(
        &mut self,
        dest: TileVar,
        src: TileVar,
        input_shape: Vec<usize>,
        output_shape: Vec<usize>,
        axes: Vec<usize>,
    ) {
        self.stmts.push(Stmt::Reindex {
            dest,
            src,
            input_shape,
            output_shape,
            axes,
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

    pub fn reduction_region(&mut self, region: super::ReductionRegion) {
        self.stmts.push(Stmt::ReductionRegion {
            region: Box::new(region),
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
