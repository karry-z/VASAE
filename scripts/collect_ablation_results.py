"""Collect ablation results from 002_F_AblationSoft experiments.

Parses experiment names to extract experiment type, model, layer, and config,
then outputs per-experiment CSVs.

Naming conventions:
  002F_gpt2_lambda_L5_a1e-4    — lambda sweep
  002F_gpt2_mode_L5_logsumexp  — mode comparison
  002F_gpt2_k_L5_k64           — k sweep
  002F_llama_freq_L15_every50   — frequency sweep

Usage:
    uv run python scripts/collect_ablation_results.py \
        --results-dir /scratch/b5bq/pu22650.b5bq/VASAE_out/002_F_AblationSoft \
        --output-dir exp/002_F_AblationSoft
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def parse_exp_name(exp_name: str) -> dict:
    """Parse ablation experiment name into structured fields."""
    parts = exp_name.split("_")
    # parts[0] = '002F', parts[1] = model, parts[2] = exp_type, ...
    model = parts[1]
    exp_type = parts[2]

    if exp_type == "lambda":
        # 002F_gpt2_lambda_L5_a1e-4
        layer = int(parts[3].lstrip("L"))
        lambda_val = parts[4].lstrip("a")
        return {
            "model": model, "exp_type": exp_type, "layer": layer,
            "lambda": lambda_val, "mode": "hard", "k": 32,
        }
    elif exp_type == "mode":
        # 002F_gpt2_mode_L5_logsumexp
        layer = int(parts[3].lstrip("L"))
        mode = parts[4]
        return {
            "model": model, "exp_type": exp_type, "layer": layer,
            "lambda": "1e-4", "mode": mode, "k": 32,
        }
    elif exp_type == "k":
        # 002F_gpt2_k_L5_k64
        layer = int(parts[3].lstrip("L"))
        k = int(parts[4].lstrip("k"))
        return {
            "model": model, "exp_type": exp_type, "layer": layer,
            "lambda": "1e-4", "mode": "hard", "k": k,
        }
    elif exp_type == "freq":
        # 002F_llama_freq_L15_every50
        layer = int(parts[3].lstrip("L"))
        freq = int(parts[4].replace("every", ""))
        return {
            "model": model, "exp_type": exp_type, "layer": layer,
            "lambda": "1e-4", "mode": "hard", "k": 32,
            "anchor_every": freq,
        }
    else:
        return {"model": model, "exp_type": exp_type}


def collect_results(results_dir: Path) -> pd.DataFrame:
    rows = []
    for results_file in sorted(results_dir.glob("*/results.json")):
        with open(results_file) as f:
            data = json.load(f)

        exp_name = data["config"]["exp_name"]
        parsed = parse_exp_name(exp_name)
        test = data.get("test", {})

        row = {
            **parsed,
            "stopped_epoch": data.get("stopped_epoch"),
            "ve": test.get("variance_explained"),
            "ce_recovery": test.get("loss_recovered"),
            "dead_rate": data.get("dead_feature_rate"),
            "l0": data.get("l0"),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Collect ablation results")
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = collect_results(results_dir)
    if df.empty:
        print("No results found.")
        return

    # Save full results
    full_path = output_dir / "results_all.csv"
    df.to_csv(full_path, index=False)
    print(f"All results saved to {full_path}")

    # Print per-experiment summaries
    for exp_type, group in df.groupby("exp_type"):
        print(f"\n=== {exp_type} ===")
        group_sorted = group.sort_values(["model", "layer"]).reset_index(drop=True)
        print(group_sorted.to_string(index=False))

        # Save per-experiment CSV
        csv_path = output_dir / f"results_{exp_type}.csv"
        group_sorted.to_csv(csv_path, index=False)
        print(f"  → saved to {csv_path}")


if __name__ == "__main__":
    main()
