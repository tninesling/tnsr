"""Export a forty-five PyTorch checkpoint or v8.4 ONNX weights to tnsr f32 archives."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys


def import_project(root: Path):
    sys.path.insert(0, str(root / "python"))
    from forty_five_ai.model import PolicyValueNet
    from forty_five_ai.export_onnx import build_model_from_checkpoint
    return PolicyValueNet, build_model_from_checkpoint


def from_onnx(path: Path, root: Path):
    import numpy as np
    import onnx
    import torch
    PolicyValueNet, _ = import_project(root)
    model = PolicyValueNet(action_dim=46, use_swiglu=True, use_card_conditioned_actions=True)
    graph = onnx.load(str(path)).graph
    arrays = {tensor.name: onnx.numpy_helper.to_array(tensor).copy() for tensor in graph.initializer}
    aliases = {node.output[0]: node.input[0] for node in graph.node if node.op_type == "Identity"}
    def resolve(name):
        while name in aliases:
            name = aliases[name]
        return arrays.get(name)
    weights = {}
    for name in model.state_dict():
        value = resolve("model." + name)
        if value is not None:
            weights[name] = torch.from_numpy(value.copy())
    for node in graph.node:
        if node.op_type != "MatMul" or len(node.input) != 2:
            continue
        value = resolve(node.input[1])
        if value is None:
            continue
        segments = node.name.split("/")[2:-1]
        parts = []
        for segment in segments:
            if parts and segment.startswith(parts[-1] + "."):
                parts[-1] = segment
            else:
                parts.append(segment)
        name = ".".join(parts) + ".weight"
        if name in model.state_dict():
            weights[name] = torch.from_numpy(value.T.copy())
    for node in graph.node:
        if node.name == "/model/value_pool/Expand":
            weights["value_pool.query"] = torch.from_numpy(resolve(node.input[0]).reshape(-1).copy())
        if node.name == "/model/card_action_heads/Expand":
            values = resolve(node.input[0]).reshape(32, 256)
            rows = [next(i for i in range(32) if i.bit_count() == n) for n in range(6)]
            weights["card_action_heads.discard_count_embedding.weight"] = torch.from_numpy(values[rows].copy())
    model.load_state_dict(weights, strict=True)
    model.eval()
    return model


def load_model(path: Path, root: Path):
    import torch
    torch.set_num_threads(2)
    if path.suffix == ".onnx":
        return from_onnx(path, root)
    _, build = import_project(root)
    # This is an explicitly supplied local training checkpoint, not an arbitrary download.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return build(checkpoint, observation_dim=0, action_dim=46)


def model_config(model):
    encoder = model.token_encoder
    return dict(token_dim=model.token_dim, token_heads=encoder.encoder.layers[0].attention.num_heads,
                token_layers=len(encoder.encoder.layers), token_max_tokens=model.token_max_tokens,
                token_vocab_size=encoder.token_embedding.num_embeddings,
                token_type_vocab_size=encoder.token_type_embedding.num_embeddings,
                phase_embed_dim=model.phase_embedding.embedding_dim,
                num_phases=model.phase_embedding.num_embeddings,
                use_phase_heads=model.use_phase_heads,
                use_card_conditioned_actions=model.use_card_conditioned_actions,
                use_separate_critic=model.use_separate_critic,
                use_privileged_critic=model.use_privileged_critic,
                use_rope=model.use_rope, use_swiglu=model.use_swiglu)


def export_model(model, directory: Path):
    import numpy as np
    directory.mkdir(parents=True, exist_ok=False)
    policy = {id(parameter) for parameter in model.policy_parameters()}
    value = {id(parameter) for parameter in model.value_parameters()}
    assert not policy & value
    records = []
    offset = 0
    with (directory / "weights.f32").open("wb") as stream:
        for name, parameter in sorted(model.named_parameters()):
            data = parameter.detach().cpu().numpy().astype("<f4", copy=False)
            if not np.isfinite(data).all():
                raise ValueError(f"nonfinite parameter: {name}")
            if id(parameter) not in policy | value:
                raise ValueError(f"ungrouped parameter: {name}")
            records.append(dict(name=name, shape=list(data.shape), offset=offset,
                                count=data.size, group="policy" if id(parameter) in policy else "value"))
            stream.write(data.tobytes(order="C"))
            offset += data.size
    manifest = dict(version=1, dtype="f32-le", data_file="weights.f32", config=model_config(model), tensors=records)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--forty-five-root", type=Path, required=True)
    args = parser.parse_args()
    model = load_model(args.source, args.forty_five_root)
    manifest = export_model(model, args.output)
    print(json.dumps(dict(parameters=len(manifest["tensors"]), elements=sum(x["count"] for x in manifest["tensors"]))))


def load_archive(directory: Path, root: Path):
    """Restore native weights to the original architecture for serving/export."""
    import numpy as np
    import torch
    PolicyValueNet, _ = import_project(root)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["version"] != 1 or manifest["dtype"] != "f32-le":
        raise ValueError("unsupported tnsr archive")
    model = PolicyValueNet(action_dim=46, **manifest["config"])
    data = np.fromfile(directory / manifest["data_file"], dtype="<f4")
    state = {record["name"]: torch.from_numpy(data[record["offset"]:record["offset"] + record["count"]]
             .copy().reshape(record["shape"])) for record in manifest["tensors"]}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


if __name__ == "__main__":
    main()
