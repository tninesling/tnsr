// TILE IR DEFINITIONS

/// The complete Tile IR for a kernel
#[allow(dead_code)]
pub struct TileIR {
    pub kernel_name: String,
    pub params: Vec<KernelParam>,
    pub body: Block,
    pub shared_mem_bytes: usize,
}

/// Kernel parameters (inputs/outputs)
#[allow(dead_code)]
pub struct KernelParam {
    pub name: String,
    pub dtype: DType,
    pub is_input: bool,
}

/// A block of statements (like a basic block)
#[allow(dead_code)]
pub struct Block {
    pub stmts: Vec<Stmt>,
}

/// Operations on tiles
#[allow(dead_code)]
pub enum Stmt {
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

    /// Store from register/shared to global
    Store {
        dest_param: String, // Parameter name
        src: TileVar,
        row_offset: Expr,
        col_offset: Expr,
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
        dest: TileVar,        // Accumulator (f32)
        a: TileVar,           // Operand (f16)
        b: TileVar,           // Operand (f16)
        layout: MatMulLayout, // NN, NT, TN, TT
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

    /// Transpose operation
    Transpose {
        dest: TileVar,
        src: TileVar,
        input_shape: Vec<usize>,
        output_shape: Vec<usize>,
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

/// Memory spaces
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum MemorySpace {
    Register,
    Shared,
    Global,
}

/// Data types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum DType {
    F16,
    BF16,
    F32,
}

impl DType {
    #[allow(dead_code)]
    pub fn size_bytes(&self) -> usize {
        match self {
            DType::F16 => 2,
            DType::BF16 => 2,
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
    ThreadIdx(Dim),
    Mul(Box<Expr>, Box<Expr>),
    Add(Box<Expr>, Box<Expr>),
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
