import os
import json
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from utils import get_logger
import evaluate
import torch


def main():
    logger = get_logger(__file__)
    os.makedirs("out", exist_ok=True)

    # load dataset
    logger.info("loading SQuAD validation split")
    data = load_dataset("squad", split="validation")

    # load metric
    logger.info("loading SQuAD evaluation metric")
    metric = evaluate.load("squad")
    bertscore = evaluate.load("bertscore")

    # load model
    logger.info("loading GPT-2 pretrained model")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    model.config.pad_token_id = model.config.eos_token_id

    predictions = []
    references = []

    logger.info("start evaluating GPT-2 on SQuAD")

    for idx, ex in enumerate(data):
        context = ex["context"].strip()
        question = ex["question"].strip()
        gt_answers = ex["answers"]
        sample_id = ex["id"]

        # build prompt
        prompt = (
            "context: " + context +
            "\nquestion: " + question +
            "\nanswer:"
        )
        inputs = tokenizer(prompt, return_tensors="pt")

        # generate answer
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,  # greedy decoding for deterministic eval
                pad_token_id=model.config.eos_token_id
            )

        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()

        predictions.append({
            "id": sample_id,
            "prediction_text": generated
        })
        references.append({
            "id": sample_id,
            "answers": gt_answers
        })
        if idx % 10 == 0:
            logger.info(f"process {idx+1}/{len(data)}")

    logger.info("computing EM/F1 with official metric")
    results = metric.compute(predictions=predictions, references=references)

    bert_res = bertscore.compute(
        predictions=[predictions],
        references=[references],
        model_type="bert-base-uncased"
    )

    # save results
    out_path = "out/squad_gpt2_eval.json"
    with open(out_path, "w") as f:
        json.dump({
            "results": results,
            "bert_res": bert_res,
            "samples": len(data)
        }, f, indent=2)

    logger.info(f"evaluation done. saved to {out_path}")
    logger.info(f"EM={results['exact_match']}, F1={results['f1']}")
    logger.info(f"bert_res={bert_res}")

if __name__ == "__main__":
    main()