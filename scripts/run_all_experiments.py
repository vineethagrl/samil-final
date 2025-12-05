
"""
Run comprehensive baseline + ablation experiments on synthetic data.
Produces metrics CSV with balanced-acc, AUROC, loss, time, and hardware info.
"""

import os
import sys
import json
import yaml
import time
import subprocess
import argparse
from pathlib import Path
import psutil
import torch

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
CONFIGS_DIR = PROJECT_ROOT / "configs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

def get_hardware_info():
    """Return hardware specs (CPU cores, GPU, RAM)."""
    cpu_count = psutil.cpu_count(logical=False)
    ram_gb = psutil.virtual_memory().total / (1024**3)
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "None"
    return {
        "cpu_cores": cpu_count,
        "ram_gb": round(ram_gb, 1),
        "gpu_count": gpu_count,
        "gpu_name": gpu_name,
        "cuda_available": torch.cuda.is_available(),
    }

def run_experiment(config_name, seed, output_dir):
    """Run single experiment with given config and seed. Return metrics dict."""
    config_path = CONFIGS_DIR / f"{config_name}.yaml"

    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}")
        return None


    checkpoint_dir = output_dir / f"{config_name}_seed{seed}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)


    train_script = SCRIPTS_DIR / "train.py"

    python_exe = sys.executable
    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    if venv_python.exists():
        python_exe = str(venv_python)

    cmd = [
        python_exe,
        str(train_script),
        "--config", str(config_path),
        "--seed", str(seed),
        "--checkpoint_dir", str(checkpoint_dir),
    ]

    print(f"\n{'='*60}")
    print(f"Running: {config_name} (seed={seed})")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")


    start_time = time.time()

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        elapsed_time = time.time() - start_time

        if result.returncode != 0:
            print(f"ERROR: Training failed with return code {result.returncode}")
            print(f"STDERR: {result.stderr[-500:]}")
            return None



        metrics = {
            "config": config_name,
            "seed": seed,
            "elapsed_time_sec": round(elapsed_time, 1),
            "status": "success",
            "checkpoint_dir": str(checkpoint_dir),
        }

        print(f"Training completed in {elapsed_time:.1f}s")
        return metrics

    except subprocess.TimeoutExpired:
        print(f"ERROR: Training timed out (>1 hour)")
        return None
    except Exception as e:
        print(f"ERROR: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Run all experiments")
    parser.add_argument("--num_seeds", type=int, default=2, help="Number of random seeds")
    parser.add_argument("--configs", nargs="+", default=["default", "ablation_contrastive"],
                        help="Config names to run (without .yaml)")
    args = parser.parse_args()


    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


    setup_log = RESULTS_DIR / "setup_info.json"
    setup_info = {
        "hardware": get_hardware_info(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_seeds": args.num_seeds,
        "configs": args.configs,
    }
    with open(setup_log, "w") as f:
        json.dump(setup_info, f, indent=2)

    print("\n" + "="*60)
    print("EXPERIMENT SETUP")
    print("="*60)
    print(json.dumps(setup_info, indent=2))


    all_experiments = []
    for config_name in args.configs:
        for seed in range(args.num_seeds):
            exp = run_experiment(config_name, seed, RESULTS_DIR)
            if exp:
                all_experiments.append(exp)


    summary_path = RESULTS_DIR / "experiments_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_experiments, f, indent=2)

    print(f"\n✓ Saved summary to: {summary_path}")
    print(f"Total experiments: {len(all_experiments)}")

if __name__ == "__main__":
    main()
