#[derive(Clone, Debug)]
pub struct Function {
    pub name: String,
    pub inputs: Vec<TensorView>,
    pub outputs: Vec<TensorView>,
    pub grid: GridShape,
    pub body: Vec<Op>,
    pub shared_mem_bytes: usize,
}

#[derive(Clone, Debug)]
pub struct TensorView {
    pub shape: Vec<usize>,
    pub space: MemorySpace,
}

#[derive(Clone, Debug)]
pub enum MemorySpace {
    Global,
    Shared,
    Register,
}

#[derive(Clone, Debug)]
pub struct GridShape {
    pub block: usize,
    pub threads: usize,
}

#[derive(Clone, Debug)]
pub enum Op {
    Add {
        lhs: String,
        rhs: String,
        out: String,
        elements_per_thread: usize,
    },
}
