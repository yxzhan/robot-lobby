"""Train a task with SuperMimic. Every TrainCfg field (smlab/core/trainer.py) is a flag:

    python scripts/train.py --task duck-walk --run-name walk
    python scripts/train.py --task duck-walk --num-envs 2048 --imag-batch 2048   # less GPU memory
"""

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from smlab.core.trainer import TrainCfg, Trainer  # noqa: E402
from smlab.tasks import make_task  # noqa: E402


def main() -> None:
  ap = argparse.ArgumentParser()
  for name, val in asdict(TrainCfg()).items():
    flag = f"--{name.replace('_', '-')}"
    if isinstance(val, tuple):
      ap.add_argument(flag, type=int, nargs="+", default=list(val))
    elif isinstance(val, bool):  # type=bool would read "false" as True
      ap.add_argument(flag, type=lambda x: x.lower() in ("1", "true"), default=val)
    else:
      ap.add_argument(flag, type=type(val), default=val)
  args = vars(ap.parse_args())
  cfg = TrainCfg(**{k: tuple(v) if isinstance(v, list) else v for k, v in args.items()})
  torch.set_float32_matmul_precision("high")
  Trainer(cfg, make_task(cfg.task)).run()


if __name__ == "__main__":
  main()
