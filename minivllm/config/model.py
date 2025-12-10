from dataclasses import dataclass, field
from transformers import AutoConfig, PretrainedConfig


@dataclass
class ModelConfig:
    # model name or path
    model: str = "Qwen/Qwen3-0.6B"
    hf_config: PretrainedConfig = field(init=False)

    def __post_init__(self):
        self.hf_config = AutoConfig.from_pretrained(self.model)
