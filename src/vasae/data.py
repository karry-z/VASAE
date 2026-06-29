from typing import Dict, Iterator

import torch
from nnsight import NNsight
from torch.utils.data import DataLoader

from vasae.engine import extract_activations


class OnlineActivationSource:
    """Extract activations on-the-fly from a language model with nnsight."""

    def __init__(
        self,
        model: NNsight,
        tokenizer,
        layer_idx: int,
        text_dataset,
        batch_size: int = 32,
        max_length: int = 128,
        text_column: str = "text",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.layer_idx = layer_idx
        self.text_dataset = text_dataset
        self.batch_size = batch_size
        self.max_length = max_length
        self.text_column = text_column

        self.dataloader = DataLoader(
            text_dataset, batch_size=batch_size, shuffle=True, collate_fn=self._collate
        )

    def _collate(self, batch):
        texts = [item[self.text_column] if isinstance(item, dict) else item for item in batch]
        return self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        for batch in self.dataloader:
            input_ids = batch["input_ids"].to(self.model.device)
            attention_mask = batch["attention_mask"].to(self.model.device)
            activations = extract_activations(self.model, input_ids, self.layer_idx)

            yield {
                "activations": activations,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

    def __len__(self):
        return len(self.dataloader)
