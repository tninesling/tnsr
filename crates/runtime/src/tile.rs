use petgraph::Graph;
use tensor::graph::TensorGraph;
use tensor::graph::TensorGraphNode;

/////////////////////////////////////
/// TILE IR DEFINITIONS
/////////////////////////////////////

/// The complete Tile IR for a kernel
pub struct TileIR {
    pub kernel_name: String,
    pub params: Vec<KernelParam>,
    pub body: Block,
    pub shared_mem_bytes: usize,
}

/// Kernel parameters (inputs/outputs)
pub struct KernelParam {
    pub name: String,
    pub dtype: DType,
    pub is_input: bool,
}

/// A block of statements (like a basic block)
pub struct Block {
    pub stmts: Vec<Stmt>,
}

/// Operations on tiles
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
    Zero { tile: TileVar },

    /// Matrix multiply with accumulation: dest = dest + (a @ b)
    MatMul {
        dest: TileVar,        // Accumulator (f32)
        a: TileVar,           // Operand (f16)
        b: TileVar,           // Operand (f16)
        layout: MatMulLayout, // NN, NT, TN, TT
    },

    /// Element-wise operations
    Add {
        dest: TileVar,
        a: TileVar,
        b: TileVar,
    },
    Mul {
        dest: TileVar,
        a: TileVar,
        b: TileVar,
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
pub enum MemorySpace {
    Register,
    Shared,
    Global,
}

/// Data types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F16,
    BF16,
    F32,
}

impl DType {
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
pub enum MatMulLayout {
    NN, // A @ B
    NT, // A @ B^T
    TN, // A^T @ B
    TT, // A^T @ B^T
}

/// Simple expressions for indices and loop bounds
#[derive(Debug, Clone)]
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
pub enum Dim {
    X,
    Y,
    Z,
}

/////////////////////////////////////
/// TILE IR BUILDER
/////////////////////////////////////

pub struct TileIRBuilder {
    kernel_name: String,
    params: Vec<KernelParam>,
    stmts: Vec<Stmt>,
    next_var: usize,
    shared_mem_bytes: usize,
}

impl TileIRBuilder {
    pub fn new() -> Self {
        Self {
            kernel_name: String::new(),
            params: Vec::new(),
            stmts: Vec::new(),
            next_var: 0,
            shared_mem_bytes: 0,
        }
    }

    pub fn start_kernel(&mut self, name: &str) {
        self.kernel_name = name.to_string();
    }

    pub fn add_param(&mut self, name: &str, dtype: DType, is_input: bool) {
        self.params.push(KernelParam {
            name: name.to_string(),
            dtype,
            is_input,
        });
    }

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

    pub fn zero(&mut self, tile: TileVar) {
        self.stmts.push(Stmt::Zero { tile });
    }

    pub fn matmul(&mut self, dest: TileVar, a: TileVar, b: TileVar, layout: MatMulLayout) {
        self.stmts.push(Stmt::MatMul { dest, a, b, layout });
    }

    pub fn barrier(&mut self) {
        self.stmts.push(Stmt::Barrier);
    }

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

    pub fn load_shared_to_register(&mut self, dest: TileVar, src: TileVar) {
        self.stmts.push(Stmt::LoadSharedToReg { dest, src });
    }

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

    pub fn finish(self) -> TileIR {
        TileIR {
            kernel_name: self.kernel_name,
            params: self.params,
            body: Block { stmts: self.stmts },
            shared_mem_bytes: self.shared_mem_bytes,
        }
    }
}

/////////////////////////////////////
/// TENSOR TO TILE IR LOWERING
/////////////////////////////////////

pub struct TileGraph {
    pub graph: Graph<TileIR, usize>,
}

impl<G> From<TensorGraph<f32, G>> for TileGraph {
    fn from(tensor_graph: TensorGraph<f32, G>) -> Self {
        Self {
            graph: tensor_graph.graph.map_owned(
                |idx, node| {
                    let shape = tensor_graph.shapes.get(&idx).unwrap();
                    Self::lower_node(node, shape)
                },
                |_, e| e,
            ),
        }
    }
}

impl TileGraph {
    fn lower_node(node: TensorGraphNode<f32>, shape: &[usize]) -> TileIR {
        match node {
            TensorGraphNode::Constant { data } => todo!(),
            TensorGraphNode::Input { name } => todo!(),
            TensorGraphNode::Parameter { id, data } => todo!(),
            TensorGraphNode::Unary { op } => todo!(),
            TensorGraphNode::Binary { op } => todo!(),
            TensorGraphNode::MatMul => Self::lower_matmul(&node, shape, false, false),
            TensorGraphNode::Transpose => todo!(),
            TensorGraphNode::BroadcastAxis { axis } => todo!(),
            TensorGraphNode::ReduceAxis { op, axis } => todo!(),
            TensorGraphNode::Gt => todo!(),
            TensorGraphNode::Mask => todo!(),
        }
    }

    fn lower_matmul(
        _node: &TensorGraphNode<f32>,
        shape: &[usize], // [M, N, K]
        transpose_a: bool,
        transpose_b: bool,
    ) -> TileIR {
        let mut builder = TileIRBuilder::new();

        // Fixed tile sizes for MVP
        const M: usize = 16;
        const N: usize = 16;
        const K: usize = 16;

        // Add parameters for inputs and outputs
        builder.add_param("A", DType::F32, true);
        builder.add_param("B", DType::F16, true);
        builder.add_param("C", DType::F32, false);

        // Allocate shared memory for input tiles
        let a_smem = builder.alloc_shared(DType::F32, M, K);
        let b_smem = builder.alloc_shared(DType::F32, K, N);

        // Allocate register tiles
        let a_reg = builder.alloc_register(DType::F32, M, K);
        let b_reg = builder.alloc_register(DType::F16, K, N);
        let c_reg = builder.alloc_register(DType::F32, M, N);

        // Initialize accumulator
        builder.zero(c_reg);

        // Get input dimensions
        let k_tiles = shape[2] / K; // Assuming shape is [M, N, K]

        // Main tiling loop
        builder.for_loop("k", 0, k_tiles as i64, |builder, k_var| {
            // Load A tile to shared memory
            builder.load_global_to_shared(
                a_smem,
                "A",
                Expr::BlockIdx(Dim::X) * M,
                Expr::Var(k_var.clone()) * K,
            );

            // Load B tile to shared memory
            builder.load_global_to_shared(
                b_smem,
                "B",
                Expr::Var(k_var) * K,
                Expr::BlockIdx(Dim::Y) * N,
            );

            builder.barrier();

            // Load to registers (simplified - assume warp load)
            builder.load_shared_to_register(a_reg, a_smem);
            builder.load_shared_to_register(b_reg, b_smem);

            // Compute
            let layout = match (transpose_a, transpose_b) {
                (false, false) => MatMulLayout::NN,
                (false, true) => MatMulLayout::NT,
                (true, false) => MatMulLayout::TN,
                (true, true) => MatMulLayout::TT,
            };

            builder.matmul(c_reg, a_reg, b_reg, layout);

            builder.barrier();
        });

        builder.finish()
    }
}
