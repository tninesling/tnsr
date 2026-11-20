#![recursion_limit = "256"]

//! Full decoder forward/backward/clipping/Adam benchmark from an explicit archive.
#[cfg(feature = "cuda")]
#[path = "transformer_training_bench/memory.rs"]
mod memory;
#[cfg(any(feature = "cuda", test))]
#[path = "transformer_training_bench/model.rs"]
mod model;
#[cfg(feature = "cuda")]
#[path = "transformer_training_bench/training.rs"]
mod training;

#[cfg(feature = "cuda")]
fn main() -> anyhow::Result<()> {
    training::run()
}
#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("enable --features cuda for the full transformer training benchmark");
}
