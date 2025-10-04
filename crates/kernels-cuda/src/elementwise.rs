use cust::{error::CudaResult, function::Function, launch, memory::DeviceBuffer, module::Module, stream::Stream};

fn launch_unary(
    module: &Module,
    stream: &Stream,
    func_name: &str,
    input: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    len: usize,
) -> CudaResult<()> {
    debug_assert!(input.len() >= len);
    debug_assert!(output.len() >= len);

    let func: Function = module.get_function(func_name)?;

    let block: u32 = 256;
    let grid: u32 = ((len as u32) + block - 1) / block;

    unsafe {
        // Kernel signature: (input: &[f32], out: *mut f32, len: usize)
        // For &[f32], we must pass (ptr, len) in that order.
        launch!(func<<<grid, block, 0, stream>>>(
            input.as_device_ptr(),
            input.len(),
            output.as_device_ptr(),
            len
        ))?
    };

    Ok(())
}

fn launch_binary(
    module: &Module,
    stream: &Stream,
    func_name: &str,
    a: &DeviceBuffer<f32>,
    b: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    len: usize,
) -> CudaResult<()> {
    debug_assert!(a.len() >= len);
    debug_assert!(b.len() >= len);
    debug_assert!(output.len() >= len);

    let func: Function = module.get_function(func_name)?;

    let block: u32 = 256;
    let grid: u32 = ((len as u32) + block - 1) / block;

    unsafe {
        // Kernel signature: (a: &[f32], b: &[f32], out: *mut f32, len: usize)
        // For each &[f32], pass (ptr, len) pair.
        launch!(func<<<grid, block, 0, stream>>>(
            a.as_device_ptr(),
            a.len(),
            b.as_device_ptr(),
            b.len(),
            output.as_device_ptr(),
            len
        ))?
    };

    Ok(())
}

pub fn neg(module: &Module, stream: &Stream, input: &DeviceBuffer<f32>, output: &mut DeviceBuffer<f32>, len: usize) -> CudaResult<()> {
    launch_unary(module, stream, "neg", input, output, len)
}

pub fn exp(module: &Module, stream: &Stream, input: &DeviceBuffer<f32>, output: &mut DeviceBuffer<f32>, len: usize) -> CudaResult<()> {
    launch_unary(module, stream, "exp", input, output, len)
}

pub fn log(module: &Module, stream: &Stream, input: &DeviceBuffer<f32>, output: &mut DeviceBuffer<f32>, len: usize) -> CudaResult<()> {
    launch_unary(module, stream, "log", input, output, len)
}

pub fn relu(module: &Module, stream: &Stream, input: &DeviceBuffer<f32>, output: &mut DeviceBuffer<f32>, len: usize) -> CudaResult<()> {
    launch_unary(module, stream, "relu", input, output, len)
}

pub fn add(module: &Module, stream: &Stream, a: &DeviceBuffer<f32>, b: &DeviceBuffer<f32>, output: &mut DeviceBuffer<f32>, len: usize) -> CudaResult<()> {
    launch_binary(module, stream, "add", a, b, output, len)
}

pub fn sub(module: &Module, stream: &Stream, a: &DeviceBuffer<f32>, b: &DeviceBuffer<f32>, output: &mut DeviceBuffer<f32>, len: usize) -> CudaResult<()> {
    launch_binary(module, stream, "sub", a, b, output, len)
}

pub fn mul(module: &Module, stream: &Stream, a: &DeviceBuffer<f32>, b: &DeviceBuffer<f32>, output: &mut DeviceBuffer<f32>, len: usize) -> CudaResult<()> {
    launch_binary(module, stream, "mul", a, b, output, len)
}

pub fn div(module: &Module, stream: &Stream, a: &DeviceBuffer<f32>, b: &DeviceBuffer<f32>, output: &mut DeviceBuffer<f32>, len: usize) -> CudaResult<()> {
    launch_binary(module, stream, "div", a, b, output, len)
}
