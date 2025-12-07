import json
import os
from functools import partial
from typing import Dict

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers import util as st_util
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from utils import get_logger


class CFG:
    save_predictions_path = "out/predictions_references.json"
    save_scores_path = "out/squad_gpt2_eval.json"
    batch_size = 384
    device = "cuda"


def save_results(results: Dict, save_path: str):
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)


def get_model(device="cpu"):
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    model.config.pad_token_id = model.config.eos_token_id
    return model, tokenizer


class SBERTScore:
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    def compute(self, predictions, references):
        pred_texts = [p["prediction_text"] for p in predictions]
        ref_texts = [r["answers"]["text"][0] for r in references]

        emb_preds = self.model.encode(
            pred_texts, convert_to_tensor=True, show_progress_bar=False
        )
        emb_refs = self.model.encode(
            ref_texts, convert_to_tensor=True, show_progress_bar=False
        )

        cosine_scores = st_util.pairwise_cos_sim(emb_preds, emb_refs).tolist()

        return cosine_scores


def build_dataset(example):
    return {
        "id": example["id"],
        "prompt": (
            "context: "
            + example["context"].strip()
            + "\nquestion: "
            + example["question"].strip()
            + "\nanswer: "
        ),
        "answer": example["answers"],
    }


def collate_fn_with_tokenizer(batch, tokenizer):
    prompts = [b["prompt"] for b in batch]
    ids = [b["id"] for b in batch]
    input_tokens = tokenizer(prompts, return_tensors="pt", padding=True)
    answers = [b["answer"] for b in batch]
    return ids, input_tokens, answers


def main():
    args = CFG()
    logger = get_logger(__file__)
    os.makedirs("out", exist_ok=True)

    # load metric
    metric_squad = evaluate.load("squad")
    metric_bertscore = evaluate.load("bertscore")
    metric_sbertscore = SBERTScore()
    logger.info("evaluation metric loaded")

    # load model
    model, tokenizer = get_model(args.device)
    logger.info("GPT-2 pretrained model loaded")

    # load dataset
    data = load_dataset("squad", split="validation")
    data = data.map(build_dataset)
    collate_fn = partial(collate_fn_with_tokenizer, tokenizer=tokenizer)
    loader = DataLoader(
        data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )
    logger.info("SQuAD validation split loaded")

    predictions = []
    references = []

    logger.info("evaluating GPT-2 on SQuAD start")

    for batch_i, (ids, input_ids, answers) in enumerate(loader):
        input_ids = {k: v.to(model.device) for k, v in input_ids.items()}
        # generate answer
        with torch.no_grad():
            outputs = model.generate(
                **input_ids,
                max_new_tokens=32,
                do_sample=False,  # greedy decoding for deterministic eval
                pad_token_id=model.config.eos_token_id,
            )

        input_len = input_ids["input_ids"].shape[1]
        generated_tokens = outputs[:, input_len:]
        generated_texts = tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )

        for example_id, answer, prediction in zip(ids, answers, generated_texts):
            predictions.append(
                {"id": example_id, "prediction_text": prediction.strip()}
            )
            references.append({"id": example_id, "answers": answer})

        save_results(
            {"predictions": predictions, "references": references},
            save_path=args.save_predictions_path,
        )
        logger.info(f"{batch_i+1}/{len(loader)} batch saved")

    logger.info(f"predictions finally saved to {args.save_predictions_path}")

    # compute metrics
    results_squad = metric_squad.compute(predictions=predictions, references=references)
    logger.info(f"EM={results_squad['exact_match']:.4f}, F1={results_squad['f1']:.4f}")

    results_bertscore = metric_bertscore.compute(
        predictions=[p["prediction_text"] for p in predictions],
        references=[r["answers"]["text"] for r in references],
        model_type="bert-base-uncased",
    )
    logger.info(
        f"bert_score f1: {np.mean(results_bertscore["f1"]):.4f} ± {np.std(results_bertscore["f1"]):.4f}"
    )
    logger.info(
        f"bert_score precision: {np.mean(results_bertscore["precision"]):.4f} ± {np.std(results_bertscore["precision"]):.4f}"
    )
    logger.info(
        f"bert_score recall: {np.mean(results_bertscore["recall"]):.4f} ± {np.std(results_bertscore["recall"]):.4f}"
    )

    results_sbertscore = metric_sbertscore.compute(predictions, references)
    logger.info(
        f"sbert_score: {np.mean(results_sbertscore):.4f} ± {np.std(results_sbertscore):.4f}"
    )

    # save results
    save_results(
        {
            "samples": len(data),
            "squad": results_squad,
            "bert_score": results_bertscore,
            "sbert_score": results_sbertscore,
        },
        save_path=args.save_scores_path,
    )

    logger.info(f"evaluation metrics saved to {args.save_scores_path}")


if __name__ == "__main__":
    main()
