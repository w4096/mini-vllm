import os
from glob import glob
import torch
import safetensors
import importlib
from minivllm.config.model import ModelConfig

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

def _initialize_model(config: ModelConfig) -> torch.nn.Module:
    architecture = config.hf_config.architectures[0]
    cls = get_model_class(architecture)
    model = cls(config.hf_config)
    return model

def _get_weights_iterator(path: str):
    for file in glob(os.path.join(path, "*.safetensors")):
        with safetensors.safe_open(file, "pt", "cpu") as f:
            for name in f.keys():
                yield name, f.get_tensor(name)

def load_model(config: ModelConfig) -> torch.nn.Module:
    model = _initialize_model(config)
    model.load_weights(_get_weights_iterator(config.model))

    return model
