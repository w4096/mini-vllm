from dataclasses import dataclass, field
from transformers import AutoConfig, PretrainedConfig

@dataclass()
class Config:
    # ================= model config =================

    # the model name or path
    model: str = "Qwen/Qwen3-0.6B"

    # the huggingface config of the model
    hf_config: PretrainedConfig = field(init=False)

    # the max length of generated tokens
    max_model_len: int = 4096




    # ================= scheduler config =================

    # the max number of tokens in a batch when running prefill and decode
    max_num_batched_tokens: int = 16384

    # the max number of sequences in a batch when running prefill and decode
    max_num_batched_seqs: int = 512

    # the end of sequence token id
    eos: int = -1




    # ================= kv cache config =================
    # the number of kv cache blocks
    kv_cache_num_blocks: int = 256
    # the size of each kv cache block
    kv_cache_block_size: int = 256
    # the max utilization of the kv cache memory
    gpu_memory_utilization: float = 0.5
    
    
    
    # ================= executor config =================
    # whether to use cuda graph for decoding
    use_cuda_graph:  bool = True


    def __post_init__(self):
        self.hf_config = AutoConfig.from_pretrained(self.model)
        assert hasattr(self.hf_config, "eos_token_id")
        self.eos = self.hf_config.eos_token_id
