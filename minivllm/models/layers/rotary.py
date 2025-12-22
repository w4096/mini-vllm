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
            rope_theta: float,
    ) -> None:
        super().__init__()
        assert head_size == rotary_dim
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        
        cache = self._compute_cos_sin_cache()
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        # base^{-2i/d}
        # [rotary_dim/2]
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim))
        
        # [max_position_embeddings]
        t = torch.arange(self.max_position_embeddings, dtype=torch.float)
        
        # [max_position_embeddings, rotary_dim / 2]
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        
        # [max_position_embeddings, rotary_dim]
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    @torch.compile
    def forward(self,
            positions: torch.Tensor,
            query: torch.Tensor,
            key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert positions.size(0) == query.size(0)
        assert positions.size(0) == key.size(0)

        # [positions.shape[0], rotary_dim]        
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        
        if query.dim() == 3:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        if query.dim() == 4:
            cos = cos.unsqueeze(1).unsqueeze(1)
            sin = sin.unsqueeze(1).unsqueeze(1)
        
        query = apply_rotary_embedding(query, cos, sin)
        key = apply_rotary_embedding(key, cos, sin)
        return query, key

    @classmethod
    @lru_cache(1)
    def get(
            cls,
            head_size: int,
            rotary_dim: int,
            max_position: int,
            rope_theta: float,
    ) -> "RotaryEmbedding":
        return cls(
            head_size=head_size,
            rotary_dim=rotary_dim,
            max_position_embeddings=max_position,
            rope_theta=rope_theta,
        )
