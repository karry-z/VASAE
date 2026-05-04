import argparse
import json
from pathlib import Path
from typing import Any

from vasae.data.corpus_windows import CORPORA, default_dataset_run_dir


METRIC_KEYS = (
    "loss_reconst",
    "variance_explained",
    "logitlens_acc",
    "loss_recovered",
    "dead_rate",
    "l0",
    "n_alive",
    "n_samples",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize dataset heldout evals")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument(
        "--corpora",
        nargs="+",
        choices=CORPORA,
        default=list(CORPORA),
        help="Heldout corpora to summarize.",
    )
    return parser.parse_args()


def load_eval(run_dir: Path, corpus: str) -> dict[str, Any]:
    path = run_dir / f"results_eval_{corpus}.json"
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def collect_metric_rows(evals: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    rows = {}
    for corpus, result in evals.items():
        test = result["test"]
        rows[corpus] = {key: test.get(key) for key in METRIC_KEYS}
    return rows


def collect_alive_sets(evals: dict[str, dict[str, Any]]) -> dict[str, set[int]]:
    return {
        corpus: set(result["test"].get("alive_features", []))
        for corpus, result in evals.items()
    }


def compute_alive_overlap(alive: dict[str, set[int]]) -> dict[str, dict[str, float]]:
    corpora = list(alive)
    overlap = {}
    for i, first in enumerate(corpora):
        for second in corpora[i + 1 :]:
            intersection = len(alive[first] & alive[second])
            union = len(alive[first] | alive[second])
            overlap[f"{first}_{second}"] = {
                "intersection": intersection,
                "union": union,
                "jaccard": intersection / union if union else 0.0,
            }

    if corpora:
        common = set.intersection(*(alive[corpus] for corpus in corpora))
    else:
        common = set()
    overlap["all"] = {"intersection": len(common)}
    if set(corpora) == set(CORPORA):
        overlap["all_three"] = {"intersection": len(common)}
    return overlap


def build_summary(
    run_dir: Path,
    evals: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    alive = collect_alive_sets(evals)
    return {
        "run_dir": str(run_dir),
        "metrics": collect_metric_rows(evals),
        "overlap": compute_alive_overlap(alive),
    }


def fmt(value: float | int | None) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}"


def render_markdown(summary: dict[str, Any], corpora: list[str]) -> str:
    lines = [
        "| corpus | MSE | VE | logitlens | CE recovered | dead_rate | L0 | n_alive |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for corpus in corpora:
        row = summary["metrics"][corpus]
        lines.append(
            "| {corpus} | {mse} | {ve} | {ll} | {ce} | {dead} | {l0} | {alive} |".format(
                corpus=corpus,
                mse=fmt(row["loss_reconst"]),
                ve=fmt(row["variance_explained"]),
                ll=fmt(row["logitlens_acc"]),
                ce=fmt(row["loss_recovered"]),
                dead=fmt(row["dead_rate"]),
                l0=fmt(row["l0"]),
                alive=fmt(row["n_alive"]),
            )
        )

    lines.extend(
        [
            "",
            "| feature sets | intersection | union | jaccard |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, item in summary["overlap"].items():
        if name in {"all", "all_three"} and name != "all_three":
            continue
        lines.append(
            f"| {name} | {item['intersection']} | {fmt(item.get('union'))} | {fmt(item.get('jaccard'))} |"
        )
    return "\n".join(lines) + "\n"


def write_summary(run_dir: Path, summary: dict[str, Any], markdown: str) -> None:
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    (run_dir / "summary.md").write_text(markdown, encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir or default_dataset_run_dir()
    corpora = list(args.corpora)
    evals = {corpus: load_eval(run_dir, corpus) for corpus in corpora}
    summary = build_summary(run_dir, evals)
    markdown = render_markdown(summary, corpora)
    write_summary(run_dir, summary, markdown)
    print(markdown)


if __name__ == "__main__":
    main()
