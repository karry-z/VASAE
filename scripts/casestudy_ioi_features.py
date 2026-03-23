"""Case study: concrete causal effects of individual VASAE features on IOI.

For each (layer, feature_id), shows:
1. What vocab token this feature's decoder direction aligns to
2. Gender-stratified analysis: does the feature differentially affect
   female-correct vs male-correct IOI prompts?
3. Representative examples showing the causal chain:
   feature identity → name gender → asymmetric logit shift

Example:
    uv run python scripts/casestudy_ioi_features.py \
        --features 7:3733 0:5628 6:3042 0:783 \
        --sae-root /scratch/b5bq/pu22650.b5bq/VASAE_out/010_soft_align \
        --output-dir exp/IOI_casestudy/results \
        --n-prompts 100 --device cpu
"""

from __future__ import annotations

import logging
import os

os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
for _name in ("transformers", "huggingface_hub", "nnsight", "datasets"):
    logging.getLogger(_name).setLevel(logging.WARNING)

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from nnsight import NNsight

from easy_transformer.ioi_redwood_adapter import load_redwood_ioi_examples
from vasae.engine.intervention import extract_activations, patch_and_forward
from vasae.models.factory import load_model
from vasae.models.sae import SAEModel
from vasae.utils.log import get_logger
from vasae.utils.seed import set_seed

logger = get_logger("casestudy_ioi")


FEMALE_NAMES = {
    "Jessica", "Ashley", "Jennifer", "Amanda", "Sarah", "Stephanie", "Nicole",
    "Heather", "Elizabeth", "Megan", "Melissa", "Christina", "Rachel", "Laura",
    "Lauren", "Amber", "Brittany", "Danielle", "Kimberly", "Amy", "Crystal",
    "Michelle", "Tiffany", "Emily", "Rebecca", "Erin", "Samantha", "Sara",
    "Angela", "Katherine", "Andrea", "Erica", "Mary", "Lisa", "Lindsey",
    "Kristen", "Katie", "Lindsay", "Shannon", "Vanessa", "Courtney",
    "Christine", "Alicia", "Allison",
}
MALE_NAMES = {
    "Michael", "Christopher", "Matthew", "Joshua", "Daniel", "David", "James",
    "Robert", "John", "Joseph", "Andrew", "Ryan", "Brandon", "Jason", "Justin",
    "William", "Jonathan", "Brian", "Nicholas", "Anthony", "Eric", "Adam",
    "Kevin", "Steven", "Thomas", "Timothy", "Kyle", "Richard", "Jeffrey",
    "Jeremy", "Benjamin", "Mark", "Aaron", "Charles", "Jacob", "Stephen",
    "Patrick", "Sean", "Nathan", "Dustin", "Paul", "Tyler", "Scott", "Gregory",
    "Travis", "Kenneth", "Bryan", "Jose", "Alexander", "Jesse", "Cody",
    "Bradley", "Samuel",
}


def name_gender(name: str) -> str:
    """Return 'F', 'M', or '?' for a name (strip leading space)."""
    n = name.strip()
    if n in FEMALE_NAMES:
        return "F"
    if n in MALE_NAMES:
        return "M"
    return "?"


@dataclass
class IOIExample:
    clean_text: str
    corrupted_text: str
    correct: str
    wrong: str


def parse_feature_spec(spec: str) -> Tuple[int, int]:
    parts = spec.split(":")
    return int(parts[0]), int(parts[1])


def vocab_alignment(decoder_col: torch.Tensor, unembed: torch.Tensor,
                    tokenizer, top_k: int = 5) -> List[dict]:
    sims = F.cosine_similarity(decoder_col.unsqueeze(0), unembed, dim=1)
    topk = sims.topk(top_k)
    return [
        {"rank": i + 1,
         "token": tokenizer.decode([idx.item()]).replace("\n", "\\n"),
         "cosine": round(score.item(), 4)}
        for i, (score, idx) in enumerate(zip(topk.values, topk.indices))
    ]


def get_rank(logits_1d: torch.Tensor, token_id: int) -> int:
    """Get 1-based rank of a token in the logit distribution."""
    return int((logits_1d > logits_1d[token_id]).sum().item()) + 1


def top_prob_changes(logits_before: torch.Tensor, logits_after: torch.Tensor,
                     tokenizer, k: int = 5) -> Tuple[List[dict], List[dict]]:
    """Top-k tokens by probability increase and decrease."""
    p_before = F.softmax(logits_before, dim=-1)
    p_after = F.softmax(logits_after, dim=-1)
    delta = p_after - p_before

    # Biggest gains
    top_gain_idx = delta.topk(k).indices
    gains = [{"token": tokenizer.decode([i.item()]).replace("\n", "\\n"),
              "before": round(p_before[i].item(), 6),
              "after": round(p_after[i].item(), 6),
              "delta": round(delta[i].item(), 6)}
             for i in top_gain_idx]

    # Biggest drops
    top_drop_idx = (-delta).topk(k).indices
    drops = [{"token": tokenizer.decode([i.item()]).replace("\n", "\\n"),
              "before": round(p_before[i].item(), 6),
              "after": round(p_after[i].item(), 6),
              "delta": round(delta[i].item(), 6)}
             for i in top_drop_idx]

    return gains, drops


@torch.no_grad()
def run_casestudy(
    nn_model: NNsight,
    sae_model: SAEModel,
    tokenizer,
    layer_idx: int,
    feature_id: int,
    examples: List[IOIExample],
    device: torch.device,
    unembed: torch.Tensor,
) -> dict:
    decoder_col = sae_model.decoder.weight[:, feature_id]
    va = vocab_alignment(decoder_col, unembed, tokenizer)

    cases = []
    for ex_idx, ex in enumerate(examples):
        enc = tokenizer([ex.clean_text], return_tensors="pt", padding=True).to(device)
        input_ids = enc["input_ids"]
        attn_mask = enc["attention_mask"]
        last_pos = int(attn_mask.sum() - 1)

        h = extract_activations(nn_model, input_ids, layer_idx)
        _, z = sae_model.encode(h.squeeze(0))

        strength = z[last_pos, feature_id].item()
        if abs(strength) < 1e-8:
            continue

        correct_id = tokenizer.encode(ex.correct, add_special_tokens=False)[0]
        wrong_id = tokenizer.encode(ex.wrong, add_special_tokens=False)[0]

        # SAE reconstruction (before)
        h_recon = sae_model.decode(z).unsqueeze(0)
        logits_recon = patch_and_forward(
            nn_model, input_ids, attn_mask, layer_idx,
            lambda h, _r=h_recon: _r,
        )
        lr = logits_recon[0, last_pos]

        # Ablated (after: zero out this feature)
        z_abl = z.clone()
        z_abl[:, feature_id] = 0.0
        h_abl = sae_model.decode(z_abl).unsqueeze(0)
        logits_abl = patch_and_forward(
            nn_model, input_ids, attn_mask, layer_idx,
            lambda h, _a=h_abl: _a,
        )
        la = logits_abl[0, last_pos]

        logit_diff_before = (lr[correct_id] - lr[wrong_id]).item()
        logit_diff_after = (la[correct_id] - la[wrong_id]).item()

        cases.append({
            "prompt_idx": ex_idx,
            "prompt": ex.clean_text,
            "correct": ex.correct,
            "wrong": ex.wrong,
            "correct_gender": name_gender(ex.correct),
            "wrong_gender": name_gender(ex.wrong),
            "strength": round(strength, 4),
            "logit_correct_before": round(lr[correct_id].item(), 4),
            "logit_wrong_before": round(lr[wrong_id].item(), 4),
            "logit_diff_before": round(logit_diff_before, 4),
            "rank_correct_before": get_rank(lr, correct_id),
            "rank_wrong_before": get_rank(lr, wrong_id),
            "logit_correct_after": round(la[correct_id].item(), 4),
            "logit_wrong_after": round(la[wrong_id].item(), 4),
            "logit_diff_after": round(logit_diff_after, 4),
            "rank_correct_after": get_rank(la, correct_id),
            "rank_wrong_after": get_rank(la, wrong_id),
        })

    return {
        "layer": layer_idx,
        "feature_id": feature_id,
        "aligned_token": va[0]["token"],
        "vocab_alignment": va,
        "intervention": f"zero out feature {feature_id} at all positions in layer {layer_idx}",
        "n_active": len(cases),
        "n_total": len(examples),
        "cases": cases,
    }


def plot_logit_diff_shift(result: dict, output_dir: Path):
    """Bar chart: logit diff before/after for each active prompt."""
    cases = result["cases"]
    if not cases:
        return
    layer, fid = result["layer"], result["feature_id"]
    n = len(cases)

    fig, ax = plt.subplots(figsize=(max(7, n * 1.2), 5))
    x = np.arange(n)
    w = 0.35

    ld_before = [c["logit_diff_before"] for c in cases]
    ld_after = [c["logit_diff_after"] for c in cases]

    ax.bar(x - w / 2, ld_before, w, label="Before (SAE recon)", color="#2196F3", alpha=0.85)
    ax.bar(x + w / 2, ld_after, w, label="After (feature zeroed)", color="#F44336", alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.5)

    xlabels = [f"#{c['prompt_idx']}\ns={c['strength']:.1f}" for c in cases]
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=7)
    ax.set_ylabel("Logit Diff (correct − wrong)")
    ax.legend(fontsize=9)
    ax.set_title(
        f"L{layer} Feature {fid} (aligned: \"{result['aligned_token']}\")\n"
        f"Logit difference before/after zeroing feature",
        fontsize=11,
    )
    fig.tight_layout()
    path = output_dir / f"L{layer}_F{fid}_logitdiff.pdf"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_rank_shift(result: dict, output_dir: Path):
    """Scatter: rank of correct/wrong name before → after."""
    cases = result["cases"]
    if not cases:
        return
    layer, fid = result["layer"], result["feature_id"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for ax, key, label, color in [
        (ax1, "correct", "Correct name", "#2196F3"),
        (ax2, "wrong", "Wrong name", "#F44336"),
    ]:
        ranks_before = [c[f"rank_{key}_before"] for c in cases]
        ranks_after = [c[f"rank_{key}_after"] for c in cases]
        strengths = [c["strength"] for c in cases]

        ax.scatter(ranks_before, ranks_after, c=strengths, cmap="YlOrRd",
                   s=60, edgecolors="black", linewidth=0.5)
        max_rank = max(max(ranks_before), max(ranks_after)) * 1.1
        ax.plot([0, max_rank], [0, max_rank], "k--", alpha=0.3, linewidth=1)
        ax.set_xlabel(f"Rank before (SAE recon)")
        ax.set_ylabel(f"Rank after (feature zeroed)")
        ax.set_title(f"{label} rank shift")

        # Annotate with prompt index
        for c, rb, ra in zip(cases, ranks_before, ranks_after):
            ax.annotate(f"#{c['prompt_idx']}", (rb, ra), fontsize=6, alpha=0.7)

    fig.suptitle(
        f"L{layer} Feature {fid} (\"{result['aligned_token']}\") — Name rank shift\n"
        f"Below diagonal = rank worsened after ablation (color = feature strength)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = output_dir / f"L{layer}_F{fid}_ranks.pdf"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)


def _gender_label(g: str) -> str:
    return {"F": "女", "M": "男"}.get(g, "?")


def _select_representative_cases(cases: List[dict], n: int = 3) -> List[dict]:
    """Pick n cases that best illustrate the effect: largest |delta_ld|."""
    ranked = sorted(cases, key=lambda c: abs(c["logit_diff_after"] - c["logit_diff_before"]), reverse=True)
    return ranked[:n]


def generate_report(all_results: List[dict], output_path: Path):
    lines = [
        "---",
        "title: IOI Feature Case Study — 具体因果效果",
        "date: 2026-03-19",
        "---",
        "",
        "# 目的",
        "",
        "验证 VASAE 特征的 vocab alignment 与其因果效果之间的对应关系。",
        "具体地：如果一个 feature 的 decoder 方向对齐女性名字（如 `herself`, `Anne`, `Elizabeth`），",
        "那么消融该 feature 后，模型在 **correct name 为女性** 的 IOI prompt 上应该表现更差（logit diff 下降），",
        "而在 **correct name 为男性** 的 prompt 上影响应该较小。反之亦然。",
        "",
        "# 干预方式",
        "",
        "对每个目标 feature $f$ 在 layer $l$：",
        "",
        "1. 用 clean prompt 前向传播到 layer $l$，得到 hidden state $h$",
        "2. SAE encode：$h \\to z$（topk=32），其中 $z_f$ 为 feature 激活强度",
        "3. **干预**：$z_f \\leftarrow 0$（仅置零该 feature，其余不变）",
        "4. SAE decode 并 patch 回 layer $l$，继续前向传播",
        "5. 测量 $\\Delta u = \\text{logit}(\\text{correct}) - \\text{logit}(\\text{wrong})$ 的变化",
        "",
        "**关键指标**：消融后 correct name 的 logit 降幅 vs wrong name 的 logit 降幅。",
        "如果 feature 确实编码了特定性别的名字信息，那么匹配性别的 name logit 应该降得**更多**。",
        "",
    ]

    for res in all_results:
        layer, fid = res["layer"], res["feature_id"]
        va = res["vocab_alignment"]
        cases = res["cases"]

        lines.append(f"# L{layer} Feature {fid}")
        lines.append("")

        # Feature identity
        va_str = "、".join(f'`{v["token"]}`({v["cosine"]})' for v in va[:5])
        lines.append(f"**Vocab alignment**：{va_str}")
        lines.append("")

        if not cases:
            lines.append(f"**激活情况**：0/{res['n_total']} 个 prompt 激活，无数据。")
            lines.append("")
            continue

        # Gender-stratified summary
        f_correct = [c for c in cases if c["correct_gender"] == "F"]
        m_correct = [c for c in cases if c["correct_gender"] == "M"]

        def _group_stats(group: List[dict]) -> dict:
            if not group:
                return {"n": 0, "mean_delta_ld": 0, "mean_delta_correct": 0, "mean_delta_wrong": 0}
            delta_ld = [c["logit_diff_after"] - c["logit_diff_before"] for c in group]
            delta_correct = [c["logit_correct_after"] - c["logit_correct_before"] for c in group]
            delta_wrong = [c["logit_wrong_after"] - c["logit_wrong_before"] for c in group]
            return {
                "n": len(group),
                "mean_delta_ld": sum(delta_ld) / len(delta_ld),
                "mean_delta_correct": sum(delta_correct) / len(delta_correct),
                "mean_delta_wrong": sum(delta_wrong) / len(delta_wrong),
            }

        fs = _group_stats(f_correct)
        ms = _group_stats(m_correct)

        lines.append(f"**激活情况**：{res['n_active']}/{res['n_total']} 个 prompt "
                     f"（correct=女 {fs['n']} 个，correct=男 {ms['n']} 个）")
        lines.append("")

        # Summary table
        lines.append("## 性别分组汇总")
        lines.append("")
        lines.append("| correct name 性别 | prompt 数 | 平均 Δ(logit diff) | 平均 Δ(logit correct) | 平均 Δ(logit wrong) |")
        lines.append("| --- | --- | --- | --- | --- |")
        if fs["n"] > 0:
            lines.append(f"| 女 | {fs['n']} | **{fs['mean_delta_ld']:+.4f}** "
                        f"| {fs['mean_delta_correct']:+.4f} | {fs['mean_delta_wrong']:+.4f} |")
        if ms["n"] > 0:
            lines.append(f"| 男 | {ms['n']} | **{ms['mean_delta_ld']:+.4f}** "
                        f"| {ms['mean_delta_correct']:+.4f} | {ms['mean_delta_wrong']:+.4f} |")
        lines.append("")

        # Interpretation of the asymmetry
        lines.append("**解读**：")
        # Determine feature direction
        va_tokens_lower = " ".join(v["token"].strip().lower() for v in va[:5])
        is_female_feature = any(w in va_tokens_lower for w in
                                ["herself", "anne", "elizabeth", "nicole", "mary",
                                 "marie", "mae", "daughter", "woman", "margaret",
                                 "christina", "maid", "louise", "jane", "maria"])
        is_male_feature = any(w in va_tokens_lower for w in
                              ["himself", "james", "john", "francis", "henry",
                               "daniel", "paul", "thomas", "michael", "david",
                               "patrick", "ben", "lucas", "robert"])

        if is_female_feature:
            feat_direction = "女性名字/性别"
        elif is_male_feature:
            feat_direction = "男性名字"
        else:
            feat_direction = "未明确"

        lines.append(f"该 feature 的 decoder 方向对齐 **{feat_direction}** token。")

        if is_female_feature and fs["n"] > 0 and ms["n"] > 0:
            if fs["mean_delta_ld"] < ms["mean_delta_ld"] - 0.1:
                lines.append(f"消融后，correct=女性名的 prompt Δu 平均下降 {fs['mean_delta_ld']:.3f}，"
                            f"而 correct=男性名仅下降 {ms['mean_delta_ld']:.3f}。"
                            f"**符合预期**：女性名字 feature 选择性地影响女性 correct name。")
            else:
                lines.append(f"消融效果在两组间差异不大（女 {fs['mean_delta_ld']:.3f} vs 男 {ms['mean_delta_ld']:.3f}），"
                            "可能该 feature 编码的是更一般的信息。")
        elif is_male_feature and fs["n"] > 0 and ms["n"] > 0:
            if ms["mean_delta_ld"] < fs["mean_delta_ld"] - 0.1:
                lines.append(f"消融后，correct=男性名的 prompt Δu 平均下降 {ms['mean_delta_ld']:.3f}，"
                            f"而 correct=女性名仅下降 {fs['mean_delta_ld']:.3f}。"
                            f"**符合预期**：男性名字 feature 选择性地影响男性 correct name。")
            else:
                lines.append(f"消融效果在两组间差异不大（男 {ms['mean_delta_ld']:.3f} vs 女 {fs['mean_delta_ld']:.3f}），"
                            "可能该 feature 编码的是更一般的信息。")
        lines.append("")

        # Figures
        lines.append(f"![Logit Diff Shift](L{layer}_F{fid}_logitdiff.pdf)")
        lines.append("")
        lines.append(f"![Rank Shift](L{layer}_F{fid}_ranks.pdf)")
        lines.append("")

        # Representative examples (not all)
        lines.append("## 代表性案例")
        lines.append("")

        # Pick best examples: one female-correct, one male-correct if available
        show_cases = []
        if f_correct:
            best_f = _select_representative_cases(f_correct, n=2)
            show_cases.extend(best_f)
        if m_correct:
            best_m = _select_representative_cases(m_correct, n=2)
            show_cases.extend(best_m)
        if not show_cases:
            show_cases = _select_representative_cases(cases, n=3)

        for c in show_cases:
            ld_b = c["logit_diff_before"]
            ld_a = c["logit_diff_after"]
            delta_ld = ld_a - ld_b
            delta_correct = c["logit_correct_after"] - c["logit_correct_before"]
            delta_wrong = c["logit_wrong_after"] - c["logit_wrong_before"]
            cg = _gender_label(c["correct_gender"])
            wg = _gender_label(c["wrong_gender"])

            lines.append(f"### Prompt #{c['prompt_idx']}（correct={cg}，wrong={wg}）")
            lines.append("")
            lines.append(f"> {c['prompt']}")
            lines.append("")
            lines.append(f"- correct=`{c['correct']}`（{cg}），wrong=`{c['wrong']}`（{wg}），"
                        f"strength=**{c['strength']}**")
            lines.append("")

            lines.append("| | logit(correct) | logit(wrong) | $\\Delta u$ | rank(correct) | rank(wrong) |")
            lines.append("| --- | --- | --- | --- | --- | --- |")
            lines.append(f"| 干预前 | {c['logit_correct_before']:.4f} | {c['logit_wrong_before']:.4f} "
                        f"| **{ld_b:.4f}** | {c['rank_correct_before']} | {c['rank_wrong_before']} |")
            lines.append(f"| 干预后 | {c['logit_correct_after']:.4f} | {c['logit_wrong_after']:.4f} "
                        f"| **{ld_a:.4f}** | {c['rank_correct_after']} | {c['rank_wrong_after']} |")
            lines.append(f"| 变化 | {delta_correct:+.4f} | {delta_wrong:+.4f} "
                        f"| **{delta_ld:+.4f}** "
                        f"| {c['rank_correct_after']-c['rank_correct_before']:+d} "
                        f"| {c['rank_wrong_after']-c['rank_wrong_before']:+d} |")
            lines.append("")

            # Highlight the asymmetry
            if abs(delta_correct) > abs(delta_wrong) + 0.1:
                lines.append(f"**不对称性**：correct name logit 降幅（{delta_correct:+.2f}）"
                            f"大于 wrong name（{delta_wrong:+.2f}），"
                            f"差值 {abs(delta_correct) - abs(delta_wrong):.2f}。"
                            f"该 feature 不成比例地提升了 correct name 的 logit。")
            elif abs(delta_wrong) > abs(delta_correct) + 0.1:
                lines.append(f"**不对称性**：wrong name logit 降幅（{delta_wrong:+.2f}）"
                            f"大于 correct name（{delta_correct:+.2f}）。"
                            f"该 feature 主要影响 wrong name 方向。")
            else:
                lines.append(f"correct 和 wrong name logit 降幅接近"
                            f"（{delta_correct:+.2f} vs {delta_wrong:+.2f}），"
                            "该 feature 对两个名字的影响较对称。")
            lines.append("")

    # Overall summary
    lines.append("# 总结")
    lines.append("")
    lines.append("| Feature | Vocab 方向 | 类型 | correct=女 Δu | correct=男 Δu | 符合预期？ |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for res in all_results:
        layer, fid = res["layer"], res["feature_id"]
        cases = res["cases"]
        va = res["vocab_alignment"]
        va_top = "/".join(v["token"].strip() for v in va[:3])
        f_correct = [c for c in cases if c["correct_gender"] == "F"]
        m_correct = [c for c in cases if c["correct_gender"] == "M"]
        f_delta = (sum(c["logit_diff_after"] - c["logit_diff_before"] for c in f_correct)
                   / max(len(f_correct), 1))
        m_delta = (sum(c["logit_diff_after"] - c["logit_diff_before"] for c in m_correct)
                   / max(len(m_correct), 1))

        va_tokens_lower = " ".join(v["token"].strip().lower() for v in va[:5])
        is_fem = any(w in va_tokens_lower for w in
                     ["herself", "anne", "elizabeth", "nicole", "mary", "marie",
                      "mae", "daughter", "woman", "margaret", "christina", "maid"])
        is_mal = any(w in va_tokens_lower for w in
                     ["himself", "james", "john", "francis", "henry", "daniel",
                      "paul", "thomas", "michael", "david", "patrick"])
        feat_type = "女性" if is_fem else ("男性" if is_mal else "?")

        if is_fem:
            match = "是" if f_delta < m_delta - 0.1 else "部分"
        elif is_mal:
            match = "是" if m_delta < f_delta - 0.1 else "部分"
        else:
            match = "?"

        lines.append(f"| L{layer} F{fid} | {va_top} | {feat_type} "
                    f"| {f_delta:+.3f} | {m_delta:+.3f} | {match} |")
    lines.append("")

    output_path.write_text("\n".join(lines))
    logger.info("Saved report to %s", output_path)


def parse_args():
    p = argparse.ArgumentParser(description="IOI feature case study")
    p.add_argument("--features", type=str, nargs="+", required=True,
                   help="layer:feature_id (e.g. 1:1932 7:3733)")
    p.add_argument("--sae-root", type=str, required=True)
    p.add_argument("--model-name", type=str, default="gpt2")
    p.add_argument("--n-prompts", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    feature_specs = [parse_feature_spec(s) for s in args.features]
    logger.info("Features: %s", feature_specs)

    llm, tokenizer = load_model(args.model_name, device=str(device))
    nn_model = NNsight(llm)
    unembed = llm.lm_head.weight.detach()

    redwood_examples = load_redwood_ioi_examples(
        tokenizer=tokenizer, n_prompts=args.n_prompts, seed=args.seed, prompt_type="mixed",
    )
    examples = [
        IOIExample(clean_text=ex.clean_text, corrupted_text=ex.corrupted_text,
                   correct=ex.correct, wrong=ex.wrong)
        for ex in redwood_examples
    ]
    logger.info("Loaded %d IOI examples", len(examples))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    sae_cache = {}

    for layer_idx, feature_id in feature_specs:
        logger.info("=== L%d Feature %d ===", layer_idx, feature_id)
        if layer_idx not in sae_cache:
            sae_dir = Path(args.sae_root) / f"010_soft_gpt2_L{layer_idx}_k32_a1e-3"
            sae_model = SAEModel.from_pretrained(str(sae_dir.resolve())).to(device)
            sae_model.eval()
            sae_cache[layer_idx] = sae_model

        result = run_casestudy(
            nn_model, sae_cache[layer_idx], tokenizer,
            layer_idx, feature_id, examples, device, unembed,
        )
        all_results.append(result)
        logger.info("Active in %d/%d prompts", result["n_active"], result["n_total"])

        plot_logit_diff_shift(result, output_dir)
        plot_rank_shift(result, output_dir)

    with (output_dir / "casestudy.json").open("w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    generate_report(all_results, output_dir / "report.md")


if __name__ == "__main__":
    main()
