pub struct TileIR {
    pub kernel_name: String,
    pub params: Vec<KernelParam>,
    pub body: Block,
    pub shared_mem_bytes: usize,
}

pub struct KernelParam {
    pub name: String,
    pub dtype: DType,
    pub is_input: bool,
}

pub struct Block {
    pub stmts: Vec<Stmt>,
}

pub enum Stmt {
    /// Return threads whose global linear index is outside the logical extent.
    BoundsCheck {
        extent: usize,
    },

    /// Allocate a tile variable
    AllocTile {
        var: TileVar,
        space: MemorySpace,
        dtype: DType,
        rows: usize,
        cols: usize,
    },

    /// Load from global memory to shared/register
    Load {
        dest: TileVar,
        src_param: String, // Parameter name
        row_offset: Expr,
        col_offset: Expr,
    },

    /// Load one logical matrix element into shared memory, or zero outside bounds.
    LoadGlobalToSharedPredicated {
        dest: TileVar,
        src_param: String,
        row: Expr,
        col: Expr,
        layout: MatrixLayout,
    },

    /// Load one logical matrix element into a scalar register, or zero outside bounds.
    LoadGlobalPredicated {
        dest: TileVar,
        src_param: String,
        element_index: Expr,
        row: Expr,
        col: Expr,
        layout: MatrixLayout,
    },

    /// Store from register/shared to global
    Store {
        dest_param: String, // Parameter name
        src: TileVar,
        row_offset: Expr,
        col_offset: Expr,
    },

    /// Store one logical matrix element when its row and column are in bounds.
    StoreGlobalPredicated {
        dest_param: String,
        src: TileVar,
        row: Expr,
        col: Expr,
        layout: MatrixLayout,
    },

    /// Load from shared memory to register tile (for tensor cores)
    /// Maps to wmma::load_matrix_sync or PTX ldmatrix
    LoadSharedToReg {
        dest: TileVar,
        src: TileVar,
    },

    /// Zero out a tile
    Zero {
        tile: TileVar,
    },

    /// Matrix multiply with accumulation: dest = dest + (a @ b)
    MatMul {
        dest: TileVar, // Accumulator (f32)
        a: TileVar,
        b: TileVar,
        layout: MatMulLayout, // NN, NT, TN, TT
        plan: MatMulPlan,
    },

    /// Expose the calling thread's scalar from a register or fragment tile.
    LoadTileElement {
        dest: TileVar,
        src: TileVar,
    },

    /// Gather rows from a contiguous [vocabulary, width] table.
    Embedding {
        vocabulary: usize,
        width: usize,
        index_count: usize,
    },

    /// Scatter-add embedding gradients, including repeated indices.
    EmbeddingBackward {
        vocabulary: usize,
        width: usize,
        index_count: usize,
    },

    /// Stable cross-entropy loss for each contiguous logits row.
    IndexedCrossEntropy {
        vocabulary: usize,
        row_count: usize,
    },

    /// Gradient of indexed cross entropy with respect to contiguous logits rows.
    IndexedCrossEntropyBackward {
        vocabulary: usize,
        row_count: usize,
    },

    Conv2d {
        geometry: Conv2dGeometry,
    },
    ConvTranspose2d {
        geometry: Conv2dGeometry,
    },
    Conv2dBackwardWeight {
        geometry: Conv2dGeometry,
    },
    MaxPool2d {
        geometry: MaxPool2dGeometry,
    },
    MaxPool2dBackward {
        geometry: MaxPool2dGeometry,
    },

    /// Element-wise binary operations
    Add {
        dest: TileVar,
        a: TileVar,
        b: TileVar,
    },
    Sub {
        dest: TileVar,
        a: TileVar,
        b: TileVar,
    },
    Mul {
        dest: TileVar,
        a: TileVar,
        b: TileVar,
    },
    Div {
        dest: TileVar,
        a: TileVar,
        b: TileVar,
    },

    /// Element-wise unary operations
    Neg {
        dest: TileVar,
        src: TileVar,
    },
    Exp {
        dest: TileVar,
        src: TileVar,
    },
    Log {
        dest: TileVar,
        src: TileVar,
    },
    Relu {
        dest: TileVar,
        src: TileVar,
    },

    /// Materialize a contiguous row-major permutation.
    Reindex {
        dest: TileVar,
        src: TileVar,
        input_shape: Vec<usize>,
        output_shape: Vec<usize>,
        axes: Vec<usize>,
    },

    /// Broadcast along axis
    BroadcastAxis {
        dest: TileVar,
        src: TileVar,
        axis: usize,
        input_shape: Vec<usize>,
        output_shape: Vec<usize>,
    },

    /// Reduce along axis
    ReduceAxis {
        dest: TileVar,
        src: TileVar,
        op: ReduceOp,
        axis: usize,
        input_shape: Vec<usize>,
        output_shape: Vec<usize>,
    },

    ReductionRegion {
        region: Box<super::ReductionRegion>,
    },

    /// Greater than comparison
    Gt {
        dest: TileVar,
        a: TileVar,
        b: TileVar,
    },

    /// Mask operation (select values based on condition)
    Mask {
        dest: TileVar,
        values: TileVar,
        condition: TileVar,
    },

    /// Synchronization barrier
    Barrier,

    /// For loop
    ForLoop {
        loop_var: String,
        start: Expr,
        end: Expr,
        body: Block,
    },
}

/// Tile variable reference (like SSA values)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileVar(pub usize);

#[derive(Debug, Clone, Copy)]
pub struct MatrixLayout {
    pub rows: usize,
    pub cols: usize,
    pub row_stride: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct Conv2dGeometry {
    pub batch: usize,
    pub input_channels: usize,
    pub input_height: usize,
    pub input_width: usize,
    pub output_channels: usize,
    pub output_height: usize,
    pub output_width: usize,
    pub kernel_height: usize,
    pub kernel_width: usize,
    pub stride: usize,
    pub padding: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct MaxPool2dGeometry {
    pub batch: usize,
    pub channels: usize,
    pub input_height: usize,
    pub input_width: usize,
    pub output_height: usize,
    pub output_width: usize,
    pub kernel_size: usize,
    pub stride: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemorySpace {
    Register,
    Fragment,
    Shared,
    Global,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatMulPlan {
    ScalarF32,
    TensorCoreTf32,
}

impl MatMulPlan {
    pub fn for_shape(m: usize, n: usize, k: usize) -> Self {
        const MIN_TENSOR_CORE_WORK: usize = 32 * 32 * 32;
        if m >= 16
            && n >= 16
            && k >= 8
            && m.saturating_mul(n).saturating_mul(k) >= MIN_TENSOR_CORE_WORK
        {
            Self::TensorCoreTf32
        } else {
            Self::ScalarF32
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F16,
    BF16,
    /// TensorFloat-32 compute representation stored in 32 bits.
    TF32,
    F32,
}

impl DType {
    #[allow(dead_code)]
    pub fn size_bytes(&self) -> usize {
        match self {
            DType::F16 => 2,
            DType::BF16 => 2,
            DType::TF32 => 4,
            DType::F32 => 4,
        }
    }
}

/// Matrix multiply layout
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub enum MatMulLayout {
    NN, // A @ B
    NT, // A @ B^T
    TN, // A^T @ B
    TT, // A^T @ B^T
}

/// Reduction operations
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub enum ReduceOp {
    Sum,
    Max,
    Mean,
}

/// Simple expressions for indices and loop bounds
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum Expr {
    Const(i64),
    Var(String),
    BlockIdx(Dim),
    BlockDim(Dim),
    ThreadIdx(Dim),
    Mul(Box<Expr>, Box<Expr>),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    FloorDiv(Box<Expr>, usize),
    Mod(Box<Expr>, usize),
}

impl std::ops::Mul<usize> for Expr {
    type Output = Expr;

    fn mul(self, rhs: usize) -> Self::Output {
        Expr::Mul(Box::new(self), Box::new(Expr::Const(rhs as i64)))
    }
}

impl std::ops::Mul<i64> for Expr {
    type Output = Expr;

    fn mul(self, rhs: i64) -> Self::Output {
        Expr::Mul(Box::new(self), Box::new(Expr::Const(rhs)))
    }
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub enum Dim {
    X,
    Y,
    Z,
}
