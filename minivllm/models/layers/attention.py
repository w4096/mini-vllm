import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from minivllm.executor.context import Context


# this file is copied from nano-vllm with minor modifications

@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class FlashAttention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scaling,
        num_kv_heads,
        sliding_window = (-1, -1),
    ):
        """
        
        window_size: (left, right). If not (-1, -1), implements sliding window local attention. 
                     left and right indicate how many tokens to the left/right each query can attend to.
        """
        
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scaling = scaling
        self.num_kv_heads = num_kv_heads
        self.sliding_window = sliding_window
        
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, ctx: Context, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        
        k_cache, v_cache = self.k_cache, self.v_cache
        
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, ctx.slot_mapping)
            
        if ctx.prefill:
            # if ctx.block_table is not None, this means we are using prefix caching,
            # in this situation, we need to provide the full k_cache and v_cache and
            # the block_table to flash attention
            if ctx.block_table is not None:
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=ctx.max_seqlen_q, cu_seqlens_q=ctx.cu_seqlens_q,
                                       max_seqlen_k=ctx.max_seqlen_k, cu_seqlens_k=ctx.cu_seqlens_k,
                                       softmax_scale=self.scaling, causal=True, block_table=ctx.block_table,
                                       window_size=self.sliding_window)
        else:
            # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=ctx.context_lens, block_table=ctx.block_table,
                                        softmax_scale=self.scaling, causal=True,
                                        window_size=self.sliding_window)
        return o
