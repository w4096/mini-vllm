import torch
from torch import nn


class Sampler(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor|None) -> torch.Tensor:
        logits = logits.div(temperatures.unsqueeze(-1))
        torch.softmax(logits, dim=-1, out=logits)
        return torch.multinomial(logits, num_samples=1).view(-1)
 