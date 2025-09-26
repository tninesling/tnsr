use kernels_core::Gemm;
use std::sync::Arc;
use std::sync::Mutex;

lazy_static::lazy_static! {
    static ref GEMM_REGISTRY: Mutex<Vec<Arc<dyn Gemm + Send + Sync>>> = Mutex::new(Vec::new());
}

pub fn register_gemm(g: Arc<dyn Gemm + Send + Sync>) {
    GEMM_REGISTRY.lock().unwrap().push(g);
}

pub fn get_preferred_gemm() -> Arc<dyn Gemm + Send + Sync> {
    GEMM_REGISTRY
        .lock()
        .unwrap()
        .last()
        .expect("No GEMM registered")
        .clone()
}
