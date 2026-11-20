//! Capture/compare legacy whole modules before source-type refactoring.
use super::*;
fn module(ir: crate::tile::TileIR) -> Result<String> {
    let mut tiles = crate::tile::TileGraph::from_graph(&TensorGraph::<f32>::new());
    tiles
        .region_kernels
        .push(crate::tile::graph::TileRegionKernel { region_id: 0, ir });
    tiles.optimize_indices()?;
    let mut target = crate::ptx::PtxTarget::sm80();
    target.compute_capability = (8, 9);
    Ok(crate::ptx::PtxGraph::from(tiles)
        .with_target(target)
        .module_source())
}
#[test]
#[ignore = "manual prechange PTX capture/comparison requires retained artifact directory"]
fn borrowed_bf16_legacy_adam_complete_module_goldens() -> Result<()> {
    let directory = std::path::PathBuf::from(
        std::env::var_os("TNSR_BORROWED_BF16_GOLDEN_DIR").context("set golden directory")?,
    );
    let capture = std::env::var_os("TNSR_BORROWED_BF16_CAPTURE_GOLDENS").is_some();
    std::fs::create_dir_all(&directory)?;
    let mut modules = Vec::new();
    for len in [1usize, 255, 256, 257, 1025] {
        // Both 256/1024 launch policies consume this exact same module; their
        // actual host geometry is a separate checked policy, not a PTX change.
        modules.push((
            format!("standalone-{len}.ptx"),
            module(kernels::adam(len)?)?,
        ));
    }
    modules.push((
        "batch-heterogeneous.ptx".into(),
        module(kernels::adam_batch(
            &[(0, 3, 0), (8, 257, 1), (272, 17, 3)],
            1024,
            1024,
        )?)?,
    ));
    for (name, text) in modules {
        let path = directory.join(name);
        if capture {
            ensure!(
                !path.exists(),
                "preserve existing golden {}",
                path.display()
            );
            std::fs::write(path, text)?;
        } else {
            assert_eq!(std::fs::read_to_string(path)?, text);
        }
    }
    for len in [1usize, 255, 256, 257, 1025] {
        for compact in [false, true] {
            let config = launch::standalone_adam(len, compact)?;
            assert_eq!(config.block_dim, (if compact { 256 } else { 1024 }, 1, 1));
        }
    }
    Ok(())
}
