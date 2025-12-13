import os
from glob import glob
import torch
import safetensors
from minivllm.config.config import Config
from minivllm.models import get_model_class

def _initialize_model(config: Config) -> torch.nn.Module:
    architecture = config.hf_config.architectures[0]
    cls = get_model_class(architecture)
    model = cls(config.hf_config)
    return model

def _get_weights_iterator(path: str):
    for file in glob(os.path.join(path, "*.safetensors")):
        with safetensors.safe_open(file, "pt", "cpu") as f:
            for name in f.keys():
                yield name, f.get_tensor(name)


def load_model(config: Config) -> torch.nn.Module:
    model = _initialize_model(config)
    params = dict(model.named_parameters())
    for name, weight in _get_weights_iterator(config.model):
        assert name in params, f"Parameter {name} not found in model"
        param = params[name]
        param.data.copy_(weight)
    return model
