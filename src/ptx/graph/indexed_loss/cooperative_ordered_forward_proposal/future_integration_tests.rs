//! UNDECLARED FUTURE-API tests. Do not register until forward plumbing is
//! separately authorized and implemented. These names intentionally refer to
//! the approved future option/counter and do not exist in current production.
//! Intended eventual placement: a child of ptx::mixed, where kernels/captured
//! are accessible. This is a disconnected integration-test sketch here; do NOT
//! declare it under indexed_loss, whose sibling privacy forbids those reads.
use crate::{graph::TensorGraph,ptx::mixed::{MixedExecutor,MixedOptions},tensor::{IndexTensor,Parameter,TensorExpr,ValueDType},tile::MatMulPrecision};
use anyhow::Result;
use std::collections::HashMap;
fn same(a:&[f32],b:&[f32]) {assert_eq!(a.len(),b.len());for (&a,&b) in a.iter().zip(b){assert!((a.is_nan()&&b.is_nan())||a.to_bits()==b.to_bits());}}
#[test]
fn future_forward_backward_modes_replay_and_actual_launch_metadata()->Result<()> {
    if let Err(error)=cudarc::driver::CudaContext::new(0){assert!(std::env::var_os("TNSR_REQUIRE_CUDA").is_none(),"{error}");return Ok(());}
    for vocabulary in [4095usize,4096] {for half in [false,true] {
        let p=Parameter::new((0..3*vocabulary).map(|i|(i%31)as f32/16.-0.75).collect(),vec![3,vocabulary]);
        let mut logits=TensorExpr::from(p.clone())+TensorExpr::input("delta",vec![3,vocabulary]);if half {logits=logits.cast(ValueDType::BF16).cast(ValueDType::F32);}
        let row_loss=logits.indexed_cross_entropy_indices(IndexTensor::input("targets",vec![3]));let mut graph=TensorGraph::new();let forward=graph.add_expr(&row_loss,false);let root=graph.add_expr(&row_loss.mean_all(),false);let graph=graph.with_gradients(root);
        let mut expected:Option<Vec<(Vec<f32>,Vec<f32>,Vec<f32>)>>=None;
        for (fwd,bwd) in [(false,false),(true,false),(false,true),(true,true)] {
            let mut e=MixedExecutor::new_with_options(&graph,MatMulPrecision::StrictF32,true,MixedOptions{indexed_loss_cooperative_ordered_forward:fwd,indexed_loss_cooperative_ordered_backward:bwd,..MixedOptions::default()})?;
            assert_eq!(e.info().indexed_loss_cooperative_ordered_forward_programs,usize::from(fwd&&vocabulary==4096));assert_eq!(e.info().indexed_loss_cooperative_ordered_backward_programs,usize::from(bwd&&vocabulary==4096));
            let forward_kernel=e.kernels.iter().find(|k|k.label.contains("IndexedCrossEntropy Kernel")).expect("actual forward kernel");assert_eq!(forward_kernel.config.block_dim,(if fwd&&vocabulary==4096 {256}else{32},1,1));assert_eq!(forward_kernel.config.grid_dim,(3,1,1));
            let mut outputs=Vec::new();
            for revision in 0..3 {e.upload_with_indices(HashMap::from([("delta".into(),vec![revision as f32/32.;3*vocabulary])]),HashMap::from([("targets".into(),vec![0,(vocabulary-1)as u32,revision*17])]))?;e.run()?;outputs.push((e.read(forward)?,e.read(root)?,e.gradients(&graph)?[&p.id()].clone()));}
            if let Some(old)=&expected {for (a,b) in outputs.iter().zip(old){same(&a.0,&b.0);same(&a.1,&b.1);same(&a.2,&b.2);}}else{expected=Some(outputs);}
            assert!(e.captured.is_some());
            assert!(e.upload_with_indices(HashMap::from([("delta".into(),vec![0.;3*vocabulary])]),HashMap::from([("targets".into(),vec![vocabulary as u32;3])])).is_err());
        }
    }}
    Ok(())
}
// Still mandatory at real integration: force a JIT-resource rejection on an
// otherwise eligible V4096 forward; inspect the restored PtxGraph metadata and
// compiled launch (32 threads, rows output), not only a requested option. With
// both flags requested, define atomic full-module fallback and test both actual
// counters. Also run Off/On/Off capture/signature tests and evolving resident
// [1,2,1] state, moments, invalid-update rollback and recovery, then full gates.
