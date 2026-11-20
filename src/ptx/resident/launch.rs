//! Scheduling-only experimental standalone Adam launch geometry.
use anyhow::{Context, Result, ensure};
use cudarc::driver::LaunchConfig;

pub(super) fn standalone_adam(len: usize, use_256_threads: bool) -> Result<LaunchConfig> {
    ensure!(len > 0, "standalone Adam length must be positive");
    let len = u32::try_from(len).context("standalone Adam length exceeds launch range")?;
    Ok(if use_256_threads {
        LaunchConfig {
            grid_dim: (len.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        }
    } else {
        LaunchConfig::for_num_elems(len)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn standalone_adam_launch_keeps_default_and_covers_each_element_once() {
        for len in [1_usize, 17, 255, 256, 257, 1023, 1024, 1025, 4097, 8193] {
            let legacy = LaunchConfig::for_num_elems(u32::try_from(len).unwrap());
            let default = standalone_adam(len, false).unwrap();
            assert_eq!(default.grid_dim, legacy.grid_dim);
            assert_eq!(default.block_dim, legacy.block_dim);
            assert_eq!(default.shared_mem_bytes, legacy.shared_mem_bytes);
            for enabled in [false, true] {
                let config = standalone_adam(len, enabled).unwrap();
                assert_eq!(config.block_dim, (if enabled { 256 } else { 1024 }, 1, 1));
                let mut visits = vec![0_u8; len];
                for block in 0..config.grid_dim.0 {
                    for thread in 0..config.block_dim.0 {
                        let index = usize::try_from(block * config.block_dim.0 + thread).unwrap();
                        if index < len {
                            visits[index] += 1;
                        }
                    }
                }
                assert!(visits.iter().all(|n| *n == 1));
                assert_eq!(config.shared_mem_bytes, 0);
            }
        }
        assert!(standalone_adam(0, true).is_err());
        if usize::BITS > 32 {
            assert!(standalone_adam(usize::MAX, true).is_err());
        }
    }
}
