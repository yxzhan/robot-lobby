"""Exports a checkpoint's actor to ONNX: observation -> action, the observation
normalizer baked in, input "obs" and output "actions" like rsl_rl's exports
(as the viewer's Microduck policies take it).

    python scripts/export_onnx.py logs/duck-walk/<run>/model_6000.pt ../../pretrained_policies/microduck_supermimic/walk.onnx
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from smlab.core.nets import Actor  # noqa: E402


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("checkpoint")
  ap.add_argument("out")
  args = ap.parse_args()
  ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
  sd = ckpt["actor"]
  obs_dim, act_dim = sd["norm.mean"].shape[0], [v for k, v in sd.items() if k.endswith("weight")][-1].shape[0]
  actor = Actor(obs_dim, tuple(ckpt["cfg"]["actor_hidden"]), act_dim)
  actor.load_state_dict(sd)
  actor.eval()
  torch.onnx.export(actor, torch.zeros(1, obs_dim), args.out, input_names=["obs"],
                    output_names=["actions"], dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
                    opset_version=17, dynamo=False)
  print(f"{args.out}: {obs_dim} -> {act_dim} (iteration {ckpt.get('iteration')})")


if __name__ == "__main__":
  main()
