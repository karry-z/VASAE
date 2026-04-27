"""Collect benchmarking results from 001_F_Benchmarking experiments.

Scans results.json files, parses model/layer/variant from exp_name,
and outputs per-layer CSV + summary statistics.

Usage:
    uv run python scripts/aggregate/collect_benchmarking_results.py \
        --results-dir /scratch/b5bq/pu22650.b5bq/VASAE_out/001_F_Benchmarking \
        --output-dir exp/001_F_Benchmarking
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_exp_name(exp_name: str) -> dict:
    """Parse '001F_gpt2_L5_plain' → {model: 'gpt2', layer: 5, variant: 'plain'}."""
    parts = exp_name.split("_")
    # Format: 001F_<model>_L<layer>_<variant>
    model = parts[1]
    layer = int(parts[2].lstrip("L"))
    variant = parts[3]
    return {"model": model, "layer": layer, "variant": variant}


def collect_results(results_dir: Path) -> pd.DataFrame:
    rows = []
    for results_file in sorted(results_dir.glob("*/results.json")):
        with open(results_file) as f:
            data = json.load(f)

        eval_results_file = results_file.with_name("results_eval.json")
        if eval_results_file.exists():
            with open(eval_results_file) as f:
                eval_data = json.load(f)
        else:
            eval_data = data

        exp_name = data["config"]["exp_name"]
        parsed = parse_exp_name(exp_name)

        test = eval_data.get("test", {})
        row = {
            **parsed,
            "stopped_epoch": data.get("stopped_epoch"),
            "mse": test.get("loss"),
            "variance_explained": test.get("variance_explained"),
            "logitlens_acc": test.get("logitlens_acc"),
            "ce_id": test.get("ce_id"),
            "ce_sae": test.get("ce_sae"),
            "ce_zero": test.get("ce_zero"),
            "loss_recovered": test.get("loss_recovered"),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Collect benchmarking results")
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory containing experiment subdirectories with results.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write output CSV and summary",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = collect_results(results_dir)
    if df.empty:
        print("No results found.")
        return

    # Sort by model, variant, layer
    df = df.sort_values(["model", "variant", "layer"]).reset_index(drop=True)

    # Save per-layer CSV
    csv_path = output_dir / "results_per_layer.csv"
    df.to_csv(csv_path, index=False)
    print(f"Per-layer results saved to {csv_path}")

    # Summary: mean ± std per model × variant
    metrics = ["mse", "variance_explained", "logitlens_acc", "loss_recovered", "ce_sae"]
    summary_rows = []
    for (model, variant), group in df.groupby(["model", "variant"]):
        row = {"model": model, "variant": variant, "n_layers": len(group)}
        for m in metrics:
            vals = group[m].dropna()
            if len(vals) > 0:
                row[f"{m}_mean"] = vals.mean()
                row[f"{m}_std"] = vals.std()
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "results_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to {summary_path}")

    # Print summary table
    print("\n=== Summary (mean ± std across layers) ===\n")
    for _, row in summary_df.iterrows():
        print(f"  {row['model']} / {row['variant']} ({int(row['n_layers'])} layers):")
        for m in metrics:
            mean_key = f"{m}_mean"
            std_key = f"{m}_std"
            if mean_key in row and pd.notna(row[mean_key]):
                print(f"    {m}: {row[mean_key]:.4f} ± {row[std_key]:.4f}")
        print()


if __name__ == "__main__":
    main()
