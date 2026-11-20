//! Late rectangle proposal. The caller retains the complete canonical tuple
//! until this module returns a fully compiled, resource-qualified replacement.
use super::*;
#[derive(Clone, Copy, Debug)]
struct Resources {
    registers: i32,
    shared: i32,
    local: i32,
    max_threads: i32,
    capacity: u32,
}
impl Resources {
    fn validate(self) -> Result<()> {
        ensure!(
            self.registers <= 128
                && self.registers > 0
                && self.shared == 32768
                && self.local == 0
                && self.max_threads >= 256
                && self.capacity >= 2,
            "rectangle JIT resource rejection: {self:?}"
        );
        Ok(())
    }
}
pub(super) fn propose(
    stream: &Arc<CudaStream>,
    target: PtxTarget,
    precision: crate::tile::MatMulPrecision,
    original: &MatMulRegion,
    #[cfg(test)] force_resource_rejection: bool,
) -> Result<(MatMulRegion, cudarc::driver::CudaFunction, String)> {
    let c = crate::tile::SplitKBf16TilePolicy::Square64
        .candidate(
            original,
            4,
            original
                .k
                .checked_mul(4)
                .context("rectangle K extent overflow")?,
            target.matmul_resources,
            true,
        )
        .context("rectangle outside exact current Square64 physical family")?;
    let (schedule, ir) = c.rectangle128x64_partial_ir(
        original,
        target.matmul_resources,
        target.supports_async_copy(),
    )?;
    let mut region = original.clone();
    region.schedule = schedule;
    let identity = format!(
        "policy={} target={target:?} precision={precision:?} region={region:?} output=F32 reducer=left4-bf16rn-v1",
        String::from_utf8_lossy(crate::tile::SplitKBf16TilePolicy::Rectangle128x64.codegen_tag())
    );
    let function = compile_partial(stream, target, &region, Some(ir), Some(&identity))?;
    let resources = Resources {
        registers: function.num_regs()?,
        shared: function.shared_size_bytes()?,
        local: function.local_size_bytes()?,
        max_threads: function.max_threads_per_block()?,
        capacity: function.occupancy_max_active_blocks_per_multiprocessor(256, 0, None)?,
    };
    #[cfg(test)]
    if force_resource_rejection {
        anyhow::bail!("forced rectangle JIT resource rejection: {resources:?}");
    }
    resources.validate()?;
    Ok((region, function, format!("{resources:?}")))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rectangle_resource_gate_rejects_each_unsafe_dimension() {
        let good = Resources {
            registers: 98,
            shared: 32768,
            local: 0,
            max_threads: 1024,
            capacity: 2,
        };
        assert!(good.validate().is_ok());
        for bad in [
            Resources {
                registers: 129,
                ..good
            },
            Resources {
                registers: 0,
                ..good
            },
            Resources {
                shared: 18432,
                ..good
            },
            Resources { local: 1, ..good },
            Resources {
                max_threads: 255,
                ..good
            },
            Resources {
                capacity: 1,
                ..good
            },
        ] {
            assert!(bad.validate().is_err());
        }
    }
}
