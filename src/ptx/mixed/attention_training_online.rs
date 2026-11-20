//! Explicit default-off complete online attention numerical screen.
//!
//! This numerical variant is not the checkpoint-preserving forward fusion.
//! The independent F64 oracle differentiates ideal finite scaled causal attention,
//! not BF16 rounding. Runtime qualification and fixed complete-training gates
//! are separate from leaf/component evidence; old quadratic owners are retained.
#[cfg(test)]
mod do_trace;
mod leaf;
#[cfg(test)]
mod oracle;
#[cfg(test)]
mod pair_baseline;
mod pair_recognize;
pub(super) mod runtime_screen;

#[cfg(test)]
mod async_transport;
mod backward;
mod common;
#[cfg(test)]
mod export;
mod forward;
#[cfg(test)]
mod tests;
