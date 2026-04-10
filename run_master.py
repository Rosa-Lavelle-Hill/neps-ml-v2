# python run_master.py > ml_pipeline_logs 2>&1
import sys
import glob
import yaml
import os
import subprocess
from datetime import datetime
from pathlib import Path
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed


from ml_pipeline import run_pipeline  

def save_run_config(cfg: dict, run_id: str):
    from pathlib import Path
    import yaml
    import json

    run_dir = Path("Results") / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # YAML (human-readable)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # JSON (machine-friendly)
    with open(run_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

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

def git_commit_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def main():
    base = load_yaml("configs/base.yaml")
    subset_files = sorted(glob.glob("configs/subsets/*.yaml"))

    jobs = []

    for f in subset_files:
        subset_cfg = load_yaml(f)
        cfg = deep_merge(base, subset_cfg)

        subset = cfg.get("subset", {})
        subset_name = subset.get("name", "unknown")

        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        run_id = f"{stamp}__{subset_name}"

        cfg.setdefault("run", {})
        cfg["run"]["id"] = run_id
        cfg["run"]["subset_name"] = subset_name

        save_run_config(cfg, run_id)

        jobs.append(cfg)

    # --- PARALLEL OR SERIAL ---
    parallel_cfg = base.get("parallel", {})
    parallel_enabled = parallel_cfg.get("enabled", False)
    max_workers = parallel_cfg.get("max_workers", os.cpu_count())

    if parallel_enabled:
        print(f"Running {len(jobs)} subsets in PARALLEL (max_workers={max_workers})")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_one_subset, cfg) for cfg in jobs]

            for future in as_completed(futures):
                future.result()  # raises error if any run fails
    else:
        print(f"Running {len(jobs)} subsets SERIAL")

        for cfg in jobs:
            run_one_subset(cfg)

def main_no_parallel():
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

        cfg.setdefault("meta", {})
        cfg["meta"]["timestamp"] = datetime.now().isoformat()
        cfg["meta"]["git_commit"] = git_commit_hash()

        save_run_config(cfg, run_id)

        run_pipeline(cfg)

def run_one_subset(cfg):
    from ml_pipeline import run_pipeline
    run_pipeline(cfg)

if __name__ == "__main__":
    main()
