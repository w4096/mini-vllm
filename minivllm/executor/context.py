from dataclasses import dataclass
import torch


@dataclass
class Context:
    """
    Context holds information for the current forward pass in the executor.
    """
    
    
    prefill: bool = False



    """
    cumulative sequence lengths for queries and keys
    Used for flash attention implementation.
    
    for seqeunce lengths: [L1, L2, L3], the cumulative sequence lengths is [0, L1, L1+L2, L1+L2+L3]
    """
    cu_seq_lens_q: torch.Tensor | None = None
    cu_seq_lens_k: torch.Tensor | None = None

    max_seq_len_q: int = 0
    max_seq_len_k: int = 0


    """
    positions of the tokens in the original sequence
    
    In continuous batching, all input sequences are concatenated into a single sequence.
    for example, we have 2 sequences:
    seq 0: [A B C D]
    seq 1: [E F G]
    
    after continuous batching, we have:
    [A B C D E F G]
    
    The positions tensor is:
    [0 1 2 3 0 1 2]
    
    this positions tensor keeps track of the original positions of the tokens in their original sequences.
    """
    positions: torch.Tensor | None = None



    """
    mapping from token index to slot index in the cache
    
      
    +-----------------+--------------------+
    | 0 | 1 | 2 | ... | 32 | 33 | 34 | ... |   ...
    +-----------------+--------------------+
        block 0             block 1

    Each token maps to a slot in the KV cache blocks.
    """
    slot_mapping: torch.Tensor | None = None



    """
    The context length of each token (only for decode)
    The context length is the length of the prefix of the token that is used to compute the attention.
    """
    context_lens: torch.Tensor | None = None


    """
    The block table is a 2D tensor that maps each request to its corresponding block in the cache.
    
    For example, req 0 uses blocks 0, 1, ..., n
                 req 1 uses blocks 0, 1, ..., m

    The block table is:
    
    [
        [0, 1, ..., n-1,   n],  # req 0
        [0, 1, ..., m,    -1],  # req 1
    ]
    """
    block_table: torch.Tensor | None = None

