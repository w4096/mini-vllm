import torch
from torch import nn
from flashinfer.sampling import top_k_top_p_sampling_from_logits

class Sampler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
                logits: torch.Tensor,
                temperatures: torch.Tensor|None,
                top_p: torch.Tensor|None,
                top_k: torch.Tensor|None) -> torch.Tensor:

        if temperatures is not None:
            logits = logits / temperatures.unsqueeze(-1)
        
        if top_p is not None or top_k is not None:
            # flashinfer has a bug when default device is cuda
            # see https://github.com/flashinfer-ai/flashinfer/issues/2333
            device = torch.get_default_device()
            torch.set_default_device("cpu")
            sampled_tokens = top_k_top_p_sampling_from_logits(
                logits,
                top_k,
                top_p,
            )
            torch.set_default_device(device)
            return sampled_tokens
        
        return torch.argmax(logits, dim=-1)
