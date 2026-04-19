"""Dump top-N firing cards from an F002 results.json.

Usage:
    uv run python scripts/analyze/alignment/dump_top_firing_cards.py \
        --results exp/F002_AlignmentAnalysis/quick_active/L6/results.json \
        --top-n 20 \
        --sort-by rho_in --min-active 5 \
        --out exp/F002_AlignmentAnalysis/quick_active/L6/top_cards.md
"""

import argparse
import json
from pathlib import Path


def render_card(card: dict) -> str:
    lines = []
    header = (
        f"### fid={card['feature_id']}  "
        f"aligned=`{card['aligned_token']}`  "
        f"n_active={card['n_active']}  "
        f"category={card['category']}  "
        f"rho_in={card['rho_in']:+.3f}  "
        f"rho_out={card['rho_out']:+.3f}  "
        f"geo={card['geo_max_sim']:.3f}"
    )
    lines.append(header)
    for i, f in enumerate(card["top_firings"]):
        toks = list(f["context_tokens"])
        fp = f["fire_pos"]
        if 0 <= fp < len(toks):
            toks[fp] = f"»{toks[fp]}«"
        ctx = "".join(toks)
        lines.append(f"  [{i + 1:2d}] z={f['z']:.3f}  `{ctx}`")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=str, required=True)
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument(
        "--sort-by",
        type=str,
        default="n_active",
        choices=["n_active", "rho_in", "rho_out"],
        help="Field to sort cards by (descending)",
    )
    p.add_argument(
        "--min-active",
        type=int,
        default=0,
        help="Minimum n_active to include a feature",
    )
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    d = json.loads(Path(args.results).read_text())
    cards = d.get("firing_cards", [])
    if not cards:
        print("No firing_cards in results file.")
        return

    if args.min_active > 0:
        cards = [c for c in cards if c["n_active"] >= args.min_active]

    cards_sorted = sorted(cards, key=lambda c: c.get(args.sort_by, 0), reverse=True)[
        : args.top_n
    ]

    sort_label = args.sort_by
    if args.min_active > 0:
        sort_label += f" (n_active>={args.min_active})"
    lines = [
        f"# Top-{args.top_n} firing cards by {sort_label}",
        f"Source: `{args.results}`",
        f"Layer: {d.get('layer_idx')}  alive+aligned: {d.get('n_alive_aligned')}  total_positions: {d.get('total_positions')}",
        "",
    ]
    for c in cards_sorted:
        lines.append(render_card(c))
        lines.append("")

    text = "\n".join(lines)
    if args.out:
        Path(args.out).write_text(text)
        print(f"Wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
