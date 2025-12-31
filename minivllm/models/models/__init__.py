from .gemma3 import Gemma3ForCausalLM
from .qwen3 import Qwen3ForCausalLM

_MODELS = {
    # architecture        class
    "Qwen3ForCausalLM":  Qwen3ForCausalLM,
    "Gemma3ForCausalLM": Gemma3ForCausalLM,
}

def get_model_class(architecture: str) -> type | None:
    return _MODELS.get(architecture, None)
