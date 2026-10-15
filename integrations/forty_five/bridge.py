"""Persistent native tnsr worker and callable adapter for forty-five rollouts."""
from __future__ import annotations
import json
from pathlib import Path
import subprocess


class Worker:
    def __init__(self, executable: Path, archive: Path, backend: str):
        self.process = subprocess.Popen([str(executable.resolve()), str(archive.resolve()), backend],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        text=True, bufsize=1)

    def request(self, op: str, **fields):
        if self.process.poll() is not None:
            raise RuntimeError(f"tnsr worker exited: {self.process.returncode}")
        self.process.stdin.write(json.dumps(dict(op=op, **fields), allow_nan=False) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"tnsr worker closed its output (exit {self.process.poll()})")
        result = json.loads(line)
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    def close(self):
        if self.process.stdin:
            self.process.stdin.close()
        try:
            code = self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=10)
            raise RuntimeError("tnsr worker did not shut down")
        if code:
            raise RuntimeError(f"tnsr worker exited with status {code}")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class Policy:
    """Model-call adapter; sampling and game collection stay in forty-five."""
    def __init__(self, worker: Worker):
        self.worker = worker

    def eval(self):
        return self

    def __call__(self, *, token_ids, token_type_ids, phase_ids, privileged_card_state=None):
        import torch
        if privileged_card_state is not None:
            raise ValueError("privileged critic inputs are not supported by this port")
        result = self.worker.request("infer", batch=dict(token_ids=token_ids.tolist(),
                                     token_type_ids=token_type_ids.tolist(), phase_ids=phase_ids.tolist()))
        return torch.tensor(result["logits"], dtype=torch.float32), torch.tensor(result["values"], dtype=torch.float32)
