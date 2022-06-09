import torch
import random
import numpy as np
import torchvision

def eliminate_randomness(seed=42):
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

def count_parameters(model, trainable=True):
    return sum(par.numel() for par in model.parameters() if par.requires_grad)