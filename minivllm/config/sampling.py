from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 512
    ignore_eos: bool = False
    
    # TODO: implement top_k and top_p sampling in the sampler
    top_k: int = 0
    top_p: float = 0.0
