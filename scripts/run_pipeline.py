"""
Pipeline Runner
Runs all evaluation and generation scripts in order.
"""
import argparse
import subprocess
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def run_script(script_name, overwrite: bool = False):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting: {script_name}...")
    start_time = time.time()
    cmd = [sys.executable, str(ROOT / "scripts" / script_name)]
    if overwrite:
        cmd.append("--overwrite")
    try:
        subprocess.run(
            cmd,
            cwd=str(ROOT),
            check=True,
            capture_output=False
        )
        elapsed = time.time() - start_time
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Finished: {script_name} in {elapsed:.2f}s\n")
    except subprocess.CalledProcessError as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ERROR in {script_name}: {e}\n")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run evaluation and figure generation pipeline.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite precomputed outputs in results/ instead of writing to results/rerun/.",
    )
    args = parser.parse_args()

    print("Running evaluation pipeline...")
    
    scripts = [
        "dataset_descriptive_stats.py",
        "run_sota_comparison.py",
        "run_ablation_study.py",
        "run_shap_analysis.py",
        "run_classification_metrics.py",
        "benchmark_runtime.py",
        "generate_plots.py",
        "generate_dataset_plots.py",
        "generate_ewta_demo.py",
        "generate_confusion_matrices.py",
    ]
    
    for s in scripts:
        run_script(s, overwrite=args.overwrite)
        
    print("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
