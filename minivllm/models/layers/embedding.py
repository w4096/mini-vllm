import torch
from torch import nn
import torch.nn.functional as F

from minivllm.executor.context import get_forward_context

class LMHead(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))

    def forward(self, x: torch.Tensor):
        context = get_forward_context()
        if context.prefill:
            last_indices = context.accum_seq_lens_q[1:] - 1
            x = x[last_indices].contiguous()
        logits = F.linear(x, self.weight)
        return logits
