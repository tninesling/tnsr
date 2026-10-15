"""Restore native weights as a forty-five PyTorch checkpoint and optional serving ONNX."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from export import load_archive, model_config, load_model


def restore(archive: Path, root: Path, checkpoint: Path, onnx: Path | None = None):
    import torch
    torch.set_num_threads(2)
    model=load_archive(archive,root)
    args=model_config(model)
    args['phase_heads']=args.pop('use_phase_heads')
    # Weight-only handoff: the original trainer starts fresh Adam/RNG state.
    with checkpoint.open('xb') as stream:
        torch.save(dict(model_state_dict=model.state_dict(),args=args,
                        tnsr_weight_only=True,source_archive=str(archive.resolve())),stream)
    rebuilt=load_model(checkpoint,root)
    for name,value in model.state_dict().items():
        torch.testing.assert_close(rebuilt.state_dict()[name],value,atol=0,rtol=0)
    if onnx is not None:
        from forty_five_ai.export_onnx import export_policy_to_onnx
        if onnx.exists():raise FileExistsError(onnx)
        export_policy_to_onnx(rebuilt,onnx,observation_dim=0,action_dim=46)
    return rebuilt


def validate_serving(model, root: Path, onnx: Path):
    import numpy as np
    import onnxruntime as ort
    from validate import observations, infer_reference, compare
    batch,masks=observations(root)
    logits,values=infer_reference(model,batch)
    options=ort.SessionOptions();options.intra_op_num_threads=2
    session=ort.InferenceSession(str(onnx),sess_options=options,providers=["CPUExecutionProvider"])
    a,b=session.run(None,{key:np.array(value,dtype=np.int64) for key,value in batch.items()})
    return compare(dict(logits=a.tolist(),values=b.tolist()),logits,values,masks)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive',type=Path)
    parser.add_argument('checkpoint',type=Path)
    parser.add_argument('--forty-five-root',type=Path,required=True)
    parser.add_argument('--onnx',type=Path)
    args=parser.parse_args()
    model=restore(args.archive,args.forty_five_root,args.checkpoint,args.onnx)
    if args.onnx:
        result=validate_serving(model,args.forty_five_root,args.onnx)
        args.checkpoint.with_suffix(".validation.json").write_text(json.dumps(result,indent=2))
        print(json.dumps(result))
if __name__=='__main__':main()
