import torch
from torch import nn
import torch.nn.functional as F


class Linear(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
        else:
            self.register_parameter("bias", None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)
