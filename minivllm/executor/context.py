from dataclasses import dataclass
import torch


@dataclass
class Context:
    prefill: bool = False

    cu_seq_lens_q: torch.Tensor | None = None
    cu_seq_lens_k: torch.Tensor | None = None

    max_seq_len_q: int = 0
    max_seq_len_k: int = 0

    positions: torch.Tensor | None = None

    """
    mapping from token index to slot index in the cache
    """
    slot_mapping: torch.Tensor | None = None

    """
    The context length of each token (only for decode)
    The context length is the length of the prefix of the token that is used to compute the attention.
    """
    context_lens: torch.Tensor | None = None


    """
    The block table is a 2D tensor that maps each request to its corresponding block in the cache.
    [
        [block_0, block_1, ..., block_n],  # req 0
        [block_0, block_1, ..., -1     ],  # req 1
        ...,
        [block_0, block_1, ..., block_n],  # req m
    ]
    """
    block_table: torch.Tensor | None = None


__ctx: Context | None = None

def get_forward_context() -> Context:
    global __ctx
    return __ctx

def set_forward_context(context: Context):
    global __ctx
    __ctx = context
