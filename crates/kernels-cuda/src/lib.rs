pub mod elementwise;

pub static PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));

pub fn load_kernel_ptx() -> String {
    let path = std::path::Path::new(
        std::env::var("OUT_DIR")
            .expect("Must have build output")
            .as_str(),
    )
    .join("kernels.ptx");
    std::fs::read_to_string(path).expect("Failed to read PTX file")
}
