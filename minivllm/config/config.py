from dataclasses import dataclass
from minivllm.config.cache import CacheConfig
from minivllm.config.model import ModelConfig


@dataclass()
class Config:
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    eos: int = -1
    cache: CacheConfig = None
    model: ModelConfig = None


    def __post_init__(self):
        assert self.max_num_batched_tokens >= self.max_model_len

        if self.cache is None:
            self.cache = CacheConfig()
