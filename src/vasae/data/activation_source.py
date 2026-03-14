from typing import Iterator, Dict

import torch
from torch.utils.data import DataLoader
from nnsight import NNsight

from vasae.engine.intervention import extract_activations


class OnlineActivationSource:
    """nnsight-based online activation data source.

    Extracts activations on-the-fly from a language model, replacing
    the need for pre-extracted memmap files.
    """

    def __init__(
        self,
        model: NNsight,
        tokenizer,
        layer_idx: int,
        text_dataset,
        batch_size: int = 32,
        max_length: int = 128,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.layer_idx = layer_idx
        self.text_dataset = text_dataset
        self.batch_size = batch_size
        self.max_length = max_length

        self.dataloader = DataLoader(
            text_dataset, batch_size=batch_size, shuffle=True, collate_fn=self._collate
        )

    def _collate(self, batch):
        texts = [item["text"] if isinstance(item, dict) else item for item in batch]
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        return encoded

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
