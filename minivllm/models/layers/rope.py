from functools import lru_cache
import torch
from torch import nn


def apply_rotary_embedding(
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(
            self,
            head_size: int,
            rotary_dim: int,
            max_position_embeddings: int,
            base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size

        # [rotary_dim / 2]
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        # [max_position_embeddings]
        t = torch.arange(max_position_embeddings, dtype=torch.float)

        # [max_position_embeddings, rotary_dim / 2]
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()

        # [max_position_embeddings, 1, rotary_dim]
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
            self,
            positions: torch.Tensor,
            query: torch.Tensor,
            key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # query: [seq_len, num_heads, head_dim]
        # key: [seq_len, num_kv_heads, head_dim]

        # [positions.shape[0], 1, rotary_dim]
        cos_sin = self.cos_sin_cache[positions]

        # [positions.shape[0], 1, rotary_dim / 2]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_embedding(query, cos, sin)
        key = apply_rotary_embedding(key, cos, sin)
        return query, key

    @classmethod
    @lru_cache(1)
    def create(
            cls,
            head_size: int,
            rotary_dim: int,
            max_position: int,
            base: float,
    ) -> "RotaryEmbedding":
        return cls(
            head_size=head_size,
            rotary_dim=rotary_dim,
            max_position_embeddings=max_position,
            base=base,
        )
