use cuda_std::{kernel, thread};

#[inline(always)]
fn grid_stride_start_stride() -> (usize, usize) {
    let idx = (thread::block_idx_x() * thread::block_dim_x() + thread::thread_idx_x()) as usize;
    let stride = (thread::grid_dim_x() * thread::block_dim_x()) as usize;
    (idx, stride)
}

#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn neg(input: &[f32], out: *mut f32, len: usize) {
    let (mut i, stride) = grid_stride_start_stride();
    while i < len {
        let x = input[i];
        *out.add(i) = -x;
        i += stride;
    }
}

#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn exp(input: &[f32], out: *mut f32, len: usize) {
    let (mut i, stride) = grid_stride_start_stride();
    while i < len {
        let x = input[i];
        *out.add(i) = libm::expf(x);
        i += stride;
    }
}

#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn log(input: &[f32], out: *mut f32, len: usize) {
    let (mut i, stride) = grid_stride_start_stride();
    while i < len {
        let x = input[i];
        *out.add(i) = libm::logf(x);
        i += stride;
    }
}

#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn relu(input: &[f32], out: *mut f32, len: usize) {
    let (mut i, stride) = grid_stride_start_stride();
    while i < len {
        let x = input[i];
        *out.add(i) = if x > 0.0 { x } else { 0.0 };
        i += stride;
    }
}

#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn add(a: &[f32], b: &[f32], out: *mut f32, len: usize) {
    let (mut i, stride) = grid_stride_start_stride();
    while i < len {
        *out.add(i) = a[i] + b[i];
        i += stride;
    }
}

#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn sub(a: &[f32], b: &[f32], out: *mut f32, len: usize) {
    let (mut i, stride) = grid_stride_start_stride();
    while i < len {
        *out.add(i) = a[i] - b[i];
        i += stride;
    }
}

#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn mul(a: &[f32], b: &[f32], out: *mut f32, len: usize) {
    let (mut i, stride) = grid_stride_start_stride();
    while i < len {
        *out.add(i) = a[i] * b[i];
        i += stride;
    }
}

#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn div(a: &[f32], b: &[f32], out: *mut f32, len: usize) {
    let (mut i, stride) = grid_stride_start_stride();
    while i < len {
        *out.add(i) = a[i] / b[i];
        i += stride;
    }
}
