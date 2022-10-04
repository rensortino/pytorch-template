import torch
import random
import numpy as np
import torchvision
import importlib

def fix_seed(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model_output_size(model, x):
    for layer in model:
        x = layer(x)
    return x.shape

def test_model_size(model, x):
    for i, layer in enumerate(model):
        print(f'Module: {i}')
        print(f'Input: {x.shape}')
        print(f'Model: {layer}')
        x = layer(x)
        print(f'Output: {x.shape}\n')

def visualize_image_tensor(image):
    torchvision.transforms.functional.to_pil_image(image).save('tmp.png')

def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)