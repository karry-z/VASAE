
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json, math, torch

from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from utils import get_logger

logger = get_logger(__file__)

# load validation
logger.info("loading SQuAD validation")
val = load_dataset("squad", split="validation")

def build(e):
    ans = e["answers"]["text"][0] if e["answers"]["text"] else ""
    return {
        "text": (
            "context: " + e["context"].strip() +
            "\nquestion: " + e["question"].strip() +
            "\nanswer: " + ans.strip()
        )
    }

logger.info("building text examples")
val = val.map(build)

logger.info("loading GPT-2")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2").eval()

def perplexity(text):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        loss = model(**enc, labels=enc["input_ids"]).loss.item()
    return math.exp(loss)

results = []
logger.info("starting evaluation")
logger.info(f"total samples: {len(val)}")
for i, ex in enumerate(val):
    p = perplexity(ex["text"])
    results.append({"id": i, "perplexity": p})
    logger.info(f"{i} samples processed. perplexity: {p}")

with open("out/gpt2_squad_val_eval.json", "w") as f:
    json.dump(results, f, indent=2)

logger.info("finished, results saved")