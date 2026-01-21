import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import glob
import yaml
from copy import deepcopy

from ml_pipeline import run_pipeline  

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def deep_merge(a, b):
    out = deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def main():
    base = load_yaml("configs/base.yaml")
    subset_files = sorted(glob.glob("configs/subsets/*.yaml"))

    for f in subset_files:
        subset_cfg = load_yaml(f)
        cfg = deep_merge(base, subset_cfg)
        print(f"=== Running subset: {cfg['subset']['name']} ===")

        subset = cfg.get("subset", {})

        subset_name = subset.get("name", "unknown")

        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        run_id = f"{stamp}__{subset_name}"

        cfg.setdefault("run", {})
        cfg["run"]["id"] = run_id
        cfg["run"]["subset_name"] = subset_name

        run_pipeline(cfg)

if __name__ == "__main__":
    main()
