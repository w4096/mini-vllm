import torch
from torch import nn
import torch.nn.functional as F
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from minivllm.models.layers.norm import RMSNorm
from minivllm.models.layers.linear import Linear
from minivllm.models.layers.rope import RotaryEmbedding
from minivllm.models.layers.head import LMHead
from minivllm.models.layers.attention import FlashAttention


class Qwen3Attention(nn.Module):

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.config = config

        self.scaling = config.head_dim ** -0.5

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * config.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * config.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * config.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * config.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.rotary_embedding = RotaryEmbedding.create(
            config.head_dim,
            rotary_dim=config.head_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
        )
        self.q_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps)

        self.attn = FlashAttention(
            num_heads=self.config.num_attention_heads,
            head_dim=self.config.head_dim,
            scale=self.scaling,
            num_kv_heads=self.config.num_key_value_heads,
        )
        self.use_flash_attn = True

    def forward(
            self,
            hidden_states: torch.Tensor,
            positions: torch.Tensor,
            mask: torch.Tensor,
    ) -> torch.Tensor:
        # [seq_len, hidden_size]
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # view: [seq_len, num_heads, head_dim]
        q = q.view(-1, self.config.num_attention_heads, self.config.head_dim)
        k = k.view(-1, self.config.num_key_value_heads, self.config.head_dim)
        v = v.view(-1, self.config.num_key_value_heads, self.config.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q, k = self.rotary_embedding(positions, q, k)

        if self.use_flash_attn:
            o = self.attn(q, k, v)
        else:
            # [num_heads, seq_len, head_dim]
            q = q.transpose(0, 1)
            k = k.transpose(0, 1)
            v = v.transpose(0, 1)
            group_size = self.config.num_attention_heads // self.config.num_key_value_heads
            k = k.repeat_interleave(group_size, dim=0)
            v = v.repeat_interleave(group_size, dim=0)

            # [num_heads, num_heads, seq_len] @ [num_heads, seq_len, head_dim]
            # scores: [num_heads, seq_len, seq_len]
            scores = q @ k.transpose(-2, -1)
            scores = scores.masked_fill(mask, -torch.inf)
            weights = F.softmax(scores * self.scaling, dim=-1)

            # [num_heads, seq_len, head_dim]
            o = weights @ v
            
            # [num_heads, seq_len, head_dim] -> [seq_len, num_heads, head_dim]
            o = o.transpose(0, 1)

        # [seq_len, num_heads, head_dim] -> [seq_len, hidden_size]
        o = o.flatten(1, -1)

        # [seq_len, hidden_size]
        output = self.o_proj(o)
        return output


class MLP(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int,
    ) -> None:
        super().__init__()
        self.gate_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        x = F.silu(gate) * up
        x = self.down_proj(x)
        return x


class DecoderLayer(nn.Module):

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()

        self.self_attn = Qwen3Attention(config)
        self.mlp = MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
        )
        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
            self,
            x: torch.Tensor,
            positions: torch.Tensor,
            mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shortcut = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, positions, mask)
        x = x + shortcut

        shortcut = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = x + shortcut
        return x


class Qwen3Model(nn.Module):
    def __init__(
            self,
            config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
    ) -> torch.Tensor:
        x = self.embed_tokens(input_ids)

        seq_len = input_ids.shape[0]
        # [1, seq_len, seq_len]
        mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1,
        ).unsqueeze(0)

        for layer in self.layers:
            x = layer(x, positions, mask)

        x = self.norm(x)
        return x


class Qwen3ForCausalLM(nn.Module):
    def __init__(
            self,
            config: Qwen3Config
    ) -> None:
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = LMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
            self,
            hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)

