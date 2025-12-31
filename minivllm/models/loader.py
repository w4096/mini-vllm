import logging
import os
from glob import glob
import torch
import safetensors
from minivllm.config.config import Config
from minivllm.models.models import get_model_class

def _initialize_model(config: Config) -> torch.nn.Module:
    architecture = config.hf_config.architectures[0]
    cls = get_model_class(architecture)
    if cls is None:
        raise ValueError(f"Model architecture {architecture} is not supported.")
    model = cls(config.hf_config)
    return model

def _get_weights_iterator(path: str):
    for file in glob(os.path.join(path, "*.safetensors")):
        with safetensors.safe_open(file, "pt", "cpu") as f:
            for name in f.keys():
                yield name, f.get_tensor(name)


def load_model(config: Config) -> torch.nn.Module:
    logging.info("Loading model on device...")
    
    model = _initialize_model(config)
    model.load_weights(_get_weights_iterator(config.model))
    return model
