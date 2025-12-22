import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from minivllm.executor.context import get_forward_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    Dim: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, Dim)
    value_offsets = idx * value_stride + tl.arange(0, Dim)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * Dim + tl.arange(0, Dim)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    seqlen, num_heads, head_dim = key.shape
    hidden_size = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == hidden_size and v_cache.stride(1) == hidden_size
    assert slot_mapping.numel() == seqlen
    store_kvcache_kernel[(seqlen,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, hidden_size)


class FlashAttention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        
        context = get_forward_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
            
        if context.prefill:
            if context.block_table is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seq_len_q, cu_seqlens_q=context.cu_seq_lens_q,
                                       max_seqlen_k=context.max_seq_len_k, cu_seqlens_k=context.cu_seq_lens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_table)
        else:
            # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_table,
                                        softmax_scale=self.scale, causal=True)
        return o
