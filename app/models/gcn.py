"""Model parts from the paper: shared text encoder (+LoRA) and the GCN head (Algorithm 1)."""

import torch
from torch import nn

from app.constants import train as T


class GCNHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.w_q = nn.Linear(dim, dim)
        self.w_p = nn.Linear(2 * dim, dim)

    def forward(
        self,
        h_p: torch.Tensor,          # (P, D) product CLS features
        h_neighbors: torch.Tensor,  # (P, N, D) neighbor-query CLS features
        neighbor_mask: torch.Tensor,  # (P, N) 1.0 where a neighbor exists
    ) -> torch.Tensor:
        h = torch.relu(self.w_q(h_neighbors)) * neighbor_mask.unsqueeze(-1)
        count = neighbor_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        h_agg = h.sum(dim=1) / count
        return torch.relu(self.w_p(torch.cat([h_p, h_agg], dim=-1)))


class TextEncoder(nn.Module):
    """Transformer backbone with LoRA adapters, CLS pooling. Shared by both towers."""

    def __init__(self, base_model: str = T.BASE_MODEL):
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModel

        backbone = AutoModel.from_pretrained(base_model)
        config = LoraConfig(
            r=T.LORA_R,
            lora_alpha=T.LORA_ALPHA,
            lora_dropout=T.LORA_DROPOUT,
            target_modules=list(T.LORA_TARGET_MODULES),
            bias="none",
        )
        self.backbone = get_peft_model(backbone, config)
        self.hidden_size = backbone.config.hidden_size

    def forward(self, input_ids, attention_mask) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0]
