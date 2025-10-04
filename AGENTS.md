# Agents Guide

- Build (CPU): `cargo build --workspace`.
- Build (CUDA): use `./build-cuda.sh` once, then `./run-cuda.sh`; inside container: `cargo build -p kernels-cuda && cargo build -p runtime --features cuda`.
- Test (all): `cargo nextest run`. Test filter: `cargo nextest run -p runtime elementwise_binary_f32`.
- Bench: CPU `cargo bench -p runtime --bench cpu`; GPU `cargo bench -p runtime --bench gpu --features cuda` (criterion dev-dep; CUDA toolchain required).
- Lint/Format: `cargo fmt --all --check` and `cargo clippy --workspace --all-targets --all-features -D warnings`.
- Toolchain: pinned nightly (see `rust-toolchain.toml`); AVX2 enabled via `.cargo/config.toml` (`-C target-cpu=native`).
- Imports: group as std, third-party, then crate/local; avoid glob imports; sort within groups; separate groups with blank lines.
- Formatting: use rustfmt defaults; keep modules small; re-export with `pub use` for stable APIs where appropriate.
- Types: prefer explicit types; share buffers with `Arc`; tensor generics use `D: DType`; `Shape = Vec<usize>`; avoid implicit widening/narrowing casts.
- Naming: types/traits CamelCase; modules/files/functions/vars snake_case; constants SCREAMING_SNAKE_CASE.
- Errors: library code should return `Result`; reserve `unwrap/expect` for tests, benches, build scripts; add context in messages.
- Unsafe: minimize and isolate; gate SIMD with `#[cfg(target_arch = "x86_64")]` and `is_x86_feature_detected!("avx2")`; document safety invariants.
- Features: CUDA is optional; workspace excludes `kernels-cuda` by default—enable via `--features cuda` for GPU paths/benches.
- Tests style: colocate under `#[cfg(test)]`; use epsilon comparisons for floats (see `runtime` tests).
- CUDA kernels: PTX built via `kernels-cuda/build.rs`; loaded at runtime in benches.
- Cursor/Copilot: no repository-specific Cursor or Copilot rules detected.
