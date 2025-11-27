# Agents Guide

- Build (CPU): `cargo check`.
- Build (CUDA): `cargo check --features cuda`.
- Test (all): `cargo nextest run`. Test filter: `cargo nextest run elementwise_binary_f32`.
- Test (CUDA): `cargo nextest run --features cuda`.
- Bench: CPU `cargo bench --bench cpu`; GPU `cargo bench --bench gpu --features cuda` (criterion dev-dep; CUDA toolchain required).
- Lint/Format: `cargo fmt --all --check` and `cargo clippy --all-targets --all-features -D warnings`.
- Formatting: use rustfmt defaults; keep modules small; re-export with `pub use` for stable APIs where appropriate.
- Types: prefer explicit types; share buffers with `Arc`; tensor generics use `D: DType`; `Shape = Vec<usize>`; avoid implicit widening/narrowing casts.
- Naming: types/traits CamelCase; modules/files/functions/vars snake_case; constants SCREAMING_SNAKE_CASE.
- Errors: library code should return `Result`; reserve `unwrap/expect` for tests, benches, build scripts; add context in messages.
- Unsafe: minimize and isolate; gate SIMD with `#[cfg(target_arch = "x86_64")]` and `is_x86_feature_detected!("avx2")`; document safety invariants.
- Features: CUDA is optional; enable via `--features cuda` for GPU paths/benches.
- Tests style: colocate under `#[cfg(test)]`; use epsilon comparisons for floats.
- CUDA kernels: static PTX in `src/cuda/kernels.ptx`; loaded at runtime via `src/cuda/mod.rs`.
