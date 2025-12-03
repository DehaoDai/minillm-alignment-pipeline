# aligner/utils.py
import random
import yaml
import torch
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML config file into a Python dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


@dataclass
class SimpleLogger:
    step: int = 0

    def log(self, metrics: Dict[str, Any]) -> None:
        """Print metrics in a simple structured format."""
        msg_parts = []
        for k, v in metrics.items():
            if isinstance(v, (float, int)):
                msg_parts.append(f"{k}: {v:.4f}")
            else:
                msg_parts.append(f"{k}: {v}")
        msg = f"[step={self.step}] " + " | ".join(msg_parts)
        print(msg)

    def set_step(self, step: int) -> None:
        self.step = step


import json
from datetime import datetime
from pathlib import Path


def log_experiment(
    kind: str,
    config: Dict[str, Any],
    output_dir: str,
    extra: Dict[str, Any] | None = None,
    exp_root: str = "experiments",
) -> str:
    """
    Save a JSON file that records an experiment run.

    kind: "sft" or "dpo" (or any string)
    config: the training config (YAML-loaded dict)
    output_dir: where the model checkpoint is saved
    extra: any extra metrics you want to record (e.g. final_loss)
    exp_root: root folder for experiment logs
    """
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_dir = Path(exp_root)
    exp_dir.mkdir(parents=True, exist_ok=True)

    record = {
        "kind": kind,
        "timestamp": ts,
        "output_dir": output_dir,
        "config": config,
    }
    if extra is not None:
        record["extra"] = extra

    # file name like: sft-20251202-103000.json
    fname = f"{kind}-{ts}.json"
    fpath = exp_dir / fname
    with fpath.open("w") as f:
        json.dump(record, f, indent=2)

    print(f"[experiment] logged to {fpath}")
    return str(fpath)
