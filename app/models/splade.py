"""SPLADE encoder: MLM logits -> log-saturation -> masked max-pool over tokens."""

import torch
from torch import nn


class SpladeEncoder(nn.Module):
    def __init__(self, base_model: str):
        super().__init__()
        from transformers import AutoModelForMaskedLM

        self.backbone = AutoModelForMaskedLM.from_pretrained(base_model)
        self.vocab_size = self.backbone.config.vocab_size

    def forward(self, input_ids, attention_mask) -> torch.Tensor:
        logits = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits  # (B, L, V) — the memory peak with a 250k vocab
        weights = torch.log1p(logits.relu_())
        weights = weights * attention_mask.unsqueeze(-1)
        return weights.max(dim=1).values  # (B, V) sparse term weights


def flops_loss(reps: torch.Tensor) -> torch.Tensor:
    return (reps.mean(dim=0) ** 2).sum()
