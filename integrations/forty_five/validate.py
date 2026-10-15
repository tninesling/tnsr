"""Compare a real forty-five model with tnsr and verify checkpoint interchange."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from bridge import Worker
from export import export_model, load_model, load_archive


def observations(root: Path, max_tokens=64):
    sys.path.insert(0, str(root / "python"))
    from forty_five_ai.env import FortyFiveRLAdapter
    env = FortyFiveRLAdapter(seed=7)
    state = env.reset(seed=7)
    seen = {}
    for step in range(300):
        seen.setdefault(state.phase, state)
        legal = [i for i, valid in enumerate(state.action_mask) if valid]
        if not legal:
            state = env.reset(seed=8 + step)
            continue
        bids = [a for a in legal if 1 <= a <= 4]
        action = max(bids) if state.phase == 1 and bids else legal[0]
        state, _, done, truncated = env.step(action)
        if done or truncated:
            state = env.reset(seed=8 + step)
        if set(seen) >= {1, 2, 3, 4}:
            break
    if not set(seen) >= {1, 2, 3, 4}:
        raise AssertionError(f"missing playable game phases: {set(seen)}")
    rows = [seen[phase] for phase in sorted(seen)]
    ids = [row.token_ids for row in rows]
    types = [row.token_type_ids for row in rows]
    phases = [row.phase for row in rows]
    masks = [row.action_mask for row in rows]
    # Exercise unused/terminal embedding IDs as well as every playable phase.
    for phase in (0, 5):
        ids.append(ids[0].copy()); types.append(types[0].copy()); phases.append(phase); masks.append([True] * 46)
    ids.append([0] * max_tokens); types.append([0] * max_tokens); phases.append(0); masks.append([True] * 46)
    return dict(token_ids=ids, token_type_ids=types, phase_ids=phases), masks


def infer_reference(model, batch):
    import torch
    with torch.no_grad():
        logits, values = model(**{key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()})
    return logits, values


def compare(actual, logits, values, masks):
    import torch
    observed_logits = torch.tensor(actual["logits"])
    observed_values = torch.tensor(actual["values"])
    torch.testing.assert_close(observed_logits, logits, atol=5e-5, rtol=5e-4)
    torch.testing.assert_close(observed_values, values, atol=5e-5, rtol=5e-4)
    mask = torch.tensor(masks, dtype=torch.bool)
    expected_probs = torch.softmax(logits.masked_fill(~mask, float("-inf")), -1)
    actual_probs = torch.softmax(observed_logits.masked_fill(~mask, float("-inf")), -1)
    torch.testing.assert_close(actual_probs, expected_probs, atol=2e-5, rtol=5e-4)
    return dict(logits_max_absolute_error=(observed_logits-logits).abs().max().item(),
                values_max_absolute_error=(observed_values-values).abs().max().item(),
                probabilities_max_absolute_error=(actual_probs-expected_probs).abs().max().item())


def validate(source: Path, root: Path, executable: Path, output: Path, backends):
    import torch
    output.mkdir(parents=True, exist_ok=False)
    model = load_model(source, root)
    archive = output / "archive"
    export_model(model, archive)
    batch, masks = observations(root)
    logits, values = infer_reference(model, batch)
    report = dict(source=str(source.resolve()), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  torch_version=torch.__version__, batch_size=len(batch["phase_ids"]), backends={})
    if source.suffix == ".onnx":
        import numpy as np
        import onnxruntime as ort
        options = ort.SessionOptions(); options.intra_op_num_threads=2
        session=ort.InferenceSession(str(source),sess_options=options,providers=["CPUExecutionProvider"])
        a,b=session.run(None,{k:np.array(v,dtype=np.int64) for k,v in batch.items()})
        report["recovered_pytorch_vs_onnx"] = compare(dict(logits=a.tolist(),values=b.tolist()),logits,values,masks)
    for backend in backends:
        start=time.monotonic()
        with Worker(executable,archive,backend) as worker:
            actual=worker.request("infer",batch=batch)
            result=compare(actual,logits,values,masks)
            repeat=worker.request("infer",batch=batch)
            if repeat != actual:
                raise AssertionError("repeat inference is not deterministic")
            saved=output / f"saved-{backend}"
            worker.request("save",directory=str(saved))
        restored=load_archive(saved,root)
        restored_logits,restored_values=infer_reference(restored,batch)
        torch.testing.assert_close(restored_logits,logits,atol=0,rtol=0)
        torch.testing.assert_close(restored_values,values,atol=0,rtol=0)
        with Worker(executable,saved,backend) as worker:
            compare(worker.request("infer",batch=batch),logits,values,masks)
        result["elapsed_seconds"]=time.monotonic()-start
        result["checkpoint_roundtrip"]=True
        report["backends"][backend]=result
        print(backend,json.dumps(result),flush=True)
    (output/"report.json").write_text(json.dumps(report,indent=2))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source",type=Path)
    parser.add_argument("--forty-five-root",type=Path,required=True)
    parser.add_argument("--worker",type=Path,default=Path("target/release/examples/forty_five_worker"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--backends",nargs="+",choices=["cpu","cuda"],default=["cpu","cuda"])
    args=parser.parse_args()
    validate(args.source,args.forty_five_root,args.worker,args.output,args.backends)
if __name__=="__main__": main()
