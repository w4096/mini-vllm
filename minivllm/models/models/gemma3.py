from typing import Iterable
import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.gemma3.modeling_gemma3 import Gemma3TextConfig

from minivllm.executor.context import Context
from minivllm.models.layers.attention import FlashAttention
from minivllm.models.layers.rotary import RotaryEmbedding


class Gemma3RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


class Gemma3Attention(nn.Module):
    
    def __init__(self,
                 config: Gemma3TextConfig,
                 rotary_embedding: RotaryEmbedding,
                 attention_type: str
    ) -> None:
        super().__init__()
        self.config = config
        self.scaling = config.query_pre_attn_scalar ** -0.5
        self.head_dim = config.head_dim
        
        self.rotary_embedding = rotary_embedding

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            bias=config.attention_bias,
        )

        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=config.attention_bias,
        )

        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=config.attention_bias,
        )

        self.o_proj = nn.Linear(
            config.num_attention_heads * config.head_dim, config.hidden_size, bias=config.attention_bias
        )
        
        self.q_norm = Gemma3RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma3RMSNorm(config.head_dim, eps=config.rms_norm_eps)

        if attention_type == "full_attention":
            sliding_window = (-1, -1)
        else:
            sliding_window = (config.sliding_window, -1)

        self.attn = FlashAttention(
            num_heads=self.config.num_attention_heads,
            head_dim=self.config.head_dim,
            scaling=self.scaling,
            num_kv_heads=self.config.num_key_value_heads,
            sliding_window=sliding_window,
        )


    def forward(
            self,
            ctx: Context,
            x: torch.Tensor,
            positions: torch.Tensor,
    ) -> torch.Tensor:
        # x: [seq_len, hidden_size]
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # view: [seq_len, num_heads, head_dim]
        q = q.view(-1, self.config.num_attention_heads, self.config.head_dim)
        k = k.view(-1, self.config.num_key_value_heads, self.config.head_dim)
        v = v.view(-1, self.config.num_key_value_heads, self.config.head_dim)
        
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        q, k = self.rotary_embedding(positions, q, k)

        x = self.attn(ctx, q, k, v)

        x = x.flatten(1, -1)

        # [seq_len, hidden_size]
        output = self.o_proj(x)
        
        return output

    
class Gemma3MLP(nn.Module):
    def __init__(
            self,
            config: Gemma3TextConfig,
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

        self.activation = lambda x: F.gelu(x, approximate='tanh')
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        x = self.activation(gate) * up
        x = self.down_proj(x)
        return x


class Gemma3DecoderLayer(nn.Module):
    def __init__(
            self,
            config: Gemma3TextConfig,
            rotary_embedding: RotaryEmbedding,
            attention_type,
    ) -> None:
        super().__init__()
        
        self.self_attn = Gemma3Attention(config, rotary_embedding=rotary_embedding, attention_type=attention_type)
        
        self.input_layernorm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.mlp = Gemma3MLP(config)
        
    def forward(
            self,
            ctx: Context,
            x: torch.Tensor,
            positions: torch.Tensor,
    ) -> torch.Tensor:
        residual = x
        
        x = self.input_layernorm(x)
        
        x = self.self_attn(ctx, x, positions)
        
        x = self.post_attention_layernorm(x)
        x = residual + x

        residual = x
        x = self.pre_feedforward_layernorm(x)
        x = self.mlp(x)
        x = self.post_feedforward_layernorm(x)
        x = residual + x
        
        return x        


class Gemma3TextScaledWordEmbedding(nn.Embedding):
    """
    This module overrides nn.Embeddings' forward by multiplying with embeddings scale.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, embed_scale: float = 1.0):
        super().__init__(num_embeddings, embedding_dim)
        self.register_buffer("embed_scale", torch.tensor(embed_scale), persistent=False)

    def forward(self, input_ids: torch.Tensor):
        return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)


class Gemma3TextModel(nn.Module):
    def __init__(self, config: Gemma3TextConfig):
        super().__init__()
        
        self.embed_tokens = Gemma3TextScaledWordEmbedding(config.vocab_size, config.hidden_size, embed_scale=config.hidden_size ** 0.5)
    
        self.norm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.rotary_emb = RotaryEmbedding(
            rotary_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
        )
        
        self.rotary_emb_local = RotaryEmbedding(
            rotary_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_local_base_freq,
        )
        
        layers = []
        for i in range(config.num_hidden_layers):
            attention_type = "sliding_attention" if bool((i + 1) % 6) else "full_attention"
            rotary_embedding = self.rotary_emb if attention_type == "full_attention" else self.rotary_emb_local
            layers.append(
                Gemma3DecoderLayer(
                    config=config,
                    rotary_embedding=rotary_embedding,
                    attention_type=attention_type,
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(
            self,
            ctx,
            x: torch.Tensor,
            positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.embed_tokens(x)
        
        for layer in self.layers:
            x = layer(ctx, x, positions)
            
        x = self.norm(x)
        
        return x
        
    
class Gemma3ForCausalLM(nn.Module):
    def __init__(
            self,
            config: Gemma3TextConfig,
    ) -> None:
        super().__init__()
        self.config = config
        
        self.model = Gemma3TextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
            self,
            ctx: Context,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.model(ctx, input_ids, positions)
        if ctx.prefill:
            last_indices = ctx.cu_seq_lens_q[1:] - 1
            hidden_states = hidden_states[last_indices]

        logits = self.lm_head(hidden_states)
        
        return logits
    
    def load_weights(self, weights: Iterable[torch.Tensor]):
        params = dict(self.named_parameters())
        for name, weight in weights:
            assert name in params, f"Unrecognized parameter name: {name}"
            assert weight.shape == params[name].shape
            
            param = params[name]
            param.data.copy_(weight)
