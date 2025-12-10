import torch
from torch import nn


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(var + self.eps)
        norm_x = norm_x * self.weight
        x = norm_x.to(input_dtype)
        return x
