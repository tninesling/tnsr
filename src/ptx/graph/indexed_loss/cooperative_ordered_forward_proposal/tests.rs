//! UNRUN proposal tests. This entire parent module is undeclared.
use super::*;
use crate::graph::{TensorGraph, TensorGraphNode};
fn fixture(rows: usize, vocabulary: usize) -> (TileGraph, NodeIndex) {
    let mut g = TensorGraph::<f32>::new();
    let x = g.graph.add_node(TensorGraphNode::Input { name: "x", shape: vec![rows, vocabulary] });
    let t = g.graph.add_node(TensorGraphNode::IndexSource { source: crate::tensor::IndexTensor::input("targets", vec![rows]) });
    let out = g.graph.add_node(TensorGraphNode::IndexedCrossEntropy { shape: vec![rows] });
    g.graph.add_edge(x,out,0); g.graph.add_edge(t,out,1);
    (TileGraph::from_graph(&g),out)
}
fn render(cooperative: bool, dtype: DType, target_dtype: DType, target: crate::ptx::PtxTarget) -> String {
    let arena = bumpalo::Bump::new();
    let name = if cooperative { "ce_forward_cooperative_proposal" } else { "ce_forward_ordered_control" };
    let mut func = Function::new(name, &arena);
    let mut ctx = LoweringContext::new(&arena);
    ctx.return_label = arena.alloc_str("cooperative_forward_test_return");
    for (name,dtype) in [("logits",dtype),("targets",target_dtype),("output",DType::F32)] {
        let pointer = func.add_global_ptr_param(arena.alloc_str(name));
        ctx.param_ptrs.insert(name.into(),pointer); ctx.param_dtypes.insert(name.into(),dtype);
    }
    if cooperative { lower_forward(&mut func,&mut ctx); }
    else {
        let row = forward_row(&mut func,&mut ctx);
        let row_start = multiply_u64(&mut func,row.clone(),VOCABULARY);
        let (maximum,sum) = emit_row_softmax_stats(&mut func,&mut ctx,row_start.clone(),VOCABULARY);
        let target = load_index_at(&mut func,&ctx,"targets",row.clone());
        let offset = func.add_u64_register(); func.add_inst(Inst::add_u64(offset.clone(),row_start,target));
        let value = load_param_f32_at(&mut func,&ctx,"logits",offset);
        let shifted = func.add_f32_register(); func.add_inst(Inst::sub_f32(shifted.clone(),maximum,value));
        let logarithm = emit_log(&mut func,sum);
        let loss = func.add_f32_register(); func.add_inst(Inst::add_f32(loss.clone(),shifted,logarithm));
        store_param_f32_at(&mut func,&ctx,"output",row,loss);
    }
    func.add_inst(Inst::Label(ctx.return_label)); func.add_inst(Inst::Ret);
    let mut module = crate::ptx::Module::new_with_target(target.compute_capability); module.add_function(func); module.to_string()
}
#[test]
fn proposed_forward_eligibility_and_original_metadata_remain_explicit() {
    for (rows,vocabulary,expected) in [(1,4096,true),(17,4096,true),(0,4096,false),(1,2048,false),(1,4095,false),(1,4097,false)] {
        let (tiles,out) = fixture(rows,vocabulary);
        assert_eq!(eligible_original_forward(&tiles.graph[out],49152),expected);
        if expected { let actual = schedule_for_ir(&tiles.graph[out]).unwrap(); assert_eq!(actual.threads,32); assert_eq!(actual.output_len,rows); }
    }
    for (name,dtype,expected) in [("logits",DType::BF16,true),("logits",DType::F16,false),("targets",DType::F32,true),("targets",DType::U32,true),("output",DType::BF16,false)] {
        let (mut tiles,out) = fixture(3,4096); tiles.graph[out].params.iter_mut().find(|p|p.name==name).unwrap().dtype=dtype;
        assert_eq!(eligible_original_forward(&tiles.graph[out],49152),expected);
    }
    let (tiles,out) = fixture(3,4096); assert!(!eligible_original_forward(&tiles.graph[out],17411));
}
#[test]
fn proposed_forward_follower_return_follows_every_cooperative_barrier() {
    let target = crate::ptx::PtxTarget::sm80();
    for dtype in [DType::F32,DType::BF16] {
        let source = render(true,dtype,DType::U32,target);
        assert_eq!(source.matches("bar.sync").count(),11);
        let lines = source.lines().collect::<Vec<_>>();
        let last_barrier = lines.iter().rposition(|s|s.contains("bar.sync")).unwrap();
        let follower_return = lines.iter().position(|s|s.contains("bra")&&s.contains("cooperative_forward_test_return")).unwrap();
        assert!(follower_return>last_barrier,"no follower may leave before MAX/EXP/SUM completes");
        assert!(source.contains("[17412]")); assert!(source.contains("ce_cooperative_ordered_column")); assert!(!source.contains("ce_parallel_sum"));
        assert_eq!(source.matches("lg2.approx").count(),1);
        let old = render(false,dtype,DType::U32,target); assert!(!old.contains("bar.sync"));
    }
}
// Mandatory AFTER future real plumbing exists: force descriptor/JIT fallback
// and inspect the actual PtxGraph launch metadata (old forward threads32,
// output_len=rows), not a mock copy of a candidate descriptor. Test all four
// independent forward/backward bool combinations and Off/On/Off cache identity.
#[cfg(test)]
mod gpu;
