use cuda_std::kernel;
use cuda_std::thread;

#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn naive_gemm(
    mat_a: &[f32],
    mat_b: &[f32],
    mat_c: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
) {
    let row = (thread::block_dim_x() * thread::block_idx_x() + thread::thread_idx_x()) as usize;
    let col = (thread::block_dim_y() * thread::block_idx_y() + thread::thread_idx_y()) as usize;

    if row < m && col < n {
        let mut sum = 0.0f32;
        for i in 0..k {
            sum += mat_a[row * k + i] * mat_b[i * n + col];
        }
        let elem = unsafe { &mut *mat_c.add((row * n + col) as usize) };
        *elem = alpha * sum + beta * *elem;
    }
}
