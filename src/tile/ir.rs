use super::SharedLayout;

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

    /// Bind a pure wrapping-u64 index expression in its dependency scope.
    LetIndex {
        name: String,
        value: Expr,
    },

    /// Allocate a tile variable
    AllocTile {
        var: TileVar,
        space: MemorySpace,
        layout: TileLayout,
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
        element_index: Expr,
        row: Expr,
        col: Expr,
        /// Local logical coordinates in the destination shared tile.
        tile_row: Expr,
        tile_col: Expr,
        layout: MatrixLayout,
    },

    /// Copy disjoint, aligned 4- or 16-byte storage groups into a pipeline stage.
    /// Out-of-bounds groups are zero-filled; partial groups use synchronous staging.
    AsyncCopy {
        dest: TileVar,
        src_param: String,
        element_index: Expr,
        row: Expr,
        col: Expr,
        tile_row: Expr,
        tile_col: Expr,
        layout: MatrixLayout,
        copy_bytes: usize,
    },
    /// Bind a shared tile to one of two identically laid-out pipeline stages.
    SelectSharedStage {
        dest: TileVar,
        first: TileVar,
        second: TileVar,
        stage: Expr,
    },
    AsyncCommit,
    /// Complete this thread's committed copies before a block-wide barrier.
    AsyncWait,

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
        element_index: Expr,
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
        schedule: super::MatMulSchedule,
    },

    /// Convert between accumulator fragments, shared tiles, and thread scalars.
    /// Synchronization is represented by separate barriers in the surrounding block.
    ConvertLayout {
        dest: TileVar,
        src: TileVar,
        /// Shared coordinates when extracting a thread scalar.
        coordinates: Option<(Expr, Expr)>,
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

    /// Assign an f32 scalar expression, including loop-state updates.
    SetScalar {
        dest: TileVar,
        value: super::ScalarExpr,
    },
    LoadScalar {
        dest: TileVar,
        source: super::ScalarMemory,
        index: Expr,
    },
    StoreScalar {
        target: super::ScalarMemory,
        index: Expr,
        value: TileVar,
    },
    /// The reduction result is complete in lane zero of each full active warp.
    WarpReduce {
        dest: TileVar,
        src: TileVar,
        op: crate::tensor::ReduceOp,
    },
    If {
        condition: super::ScalarPredicate,
        body: Block,
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

    /// Counted loop over wrapping-u64 index expressions.
    /// Bounds are evaluated in the enclosing scope before binding the counter.
    /// Carries are initialized before the loop, then updated by body assignments.
    /// They remain available after the loop, including when it has zero iterations.
    ForLoop {
        loop_var: String,
        start: Expr,
        end: Expr,
        carries: Vec<super::LoopCarry>,
        body: Block,
    },
}

/// Tile variable reference (like SSA values)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileVar(pub usize);

/// Physical tile representation, independent of the logical tensor's index map.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TileLayout {
    ThreadScalar,
    Shared(SharedLayout),
    WarpAccumulator {
        operand_dtype: DType,
        block_width: u32,
        warp_topology: (usize, usize),
    },
}

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
    TensorCoreF16,
    TensorCoreBF16,
}

impl MatMulPlan {
    pub fn for_shape(m: usize, n: usize, k: usize) -> Self {
        Self::for_shape_with_tf32(m, n, k, true)
    }

    pub fn for_shape_with_tf32(m: usize, n: usize, k: usize, supports_tf32: bool) -> Self {
        const MIN_TENSOR_CORE_WORK: usize = 32 * 32 * 32;
        if supports_tf32
            && m >= 16
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

/// Host scalar types that can be represented by the tile IR.
///
/// `tensor::DType` is a blanket `Clone` marker; this lowering capability adds
/// an explicit IR-format mapping. Host graph types such as `f64` can satisfy
/// the marker without being supported by tile lowering. TF32 is an IR compute
/// representation, not a separate Rust host scalar type.
pub trait TileDType: crate::tensor::DType + 'static {
    const TILE_DTYPE: DType;
}

impl TileDType for f32 {
    const TILE_DTYPE: DType = DType::F32;
}

impl TileDType for half::f16 {
    const TILE_DTYPE: DType = DType::F16;
}

impl TileDType for half::bf16 {
    const TILE_DTYPE: DType = DType::BF16;
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
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
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
    ShiftRight(Box<Expr>, u64),
    BitAnd(Box<Expr>, u64),
}

impl std::ops::Add for Expr {
    type Output = Expr;

    fn add(self, rhs: Self) -> Self::Output {
        Expr::Add(Box::new(self), Box::new(rhs))
    }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[allow(dead_code)]
pub enum Dim {
    X,
    Y,
    Z,
}
