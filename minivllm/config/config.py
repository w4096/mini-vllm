from dataclasses import dataclass, field
from transformers import AutoConfig, PretrainedConfig


@dataclass()
class Config:
    model: str = "Qwen/Qwen3-0.6B"
    hf_config: PretrainedConfig = field(init=False)
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    eos: int = -1

    kvcache_num_blocks: int = 1024
    kvcache_block_size: int = 256


    def __post_init__(self):
        assert self.max_num_batched_tokens >= self.max_model_len
        self.hf_config = AutoConfig.from_pretrained(self.model)
        assert hasattr(self.hf_config, "eos_token_id")
        self.eos = self.hf_config.eos_token_id
