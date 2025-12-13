import importlib


# Register new models here
_VLLM_MODELS = [
    # architecture       name     class name
    ("Qwen3ForCausalLM", "qwen3", "Qwen3ForCausalLM"),
]

def get_model_class(architecture: str) -> type | None:
    for arch, name, cls_name in _VLLM_MODELS:
        if arch == architecture:
            try:
                module = importlib.import_module(f".models.{name}", __package__)
                return getattr(module, cls_name)
            except ImportError as e:
                print("Failed to import the specified model:", e)
                raise e
    return None
