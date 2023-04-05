import copy
import requests
import pickle 
import torch
import timm
import torchvision



def get_model(model_name, pretrained=True):
    model = timm.create_model(model_name, pretrained=pretrained)
    mean = model.default_cfg["mean"]
    std = model.default_cfg["std"]
    normalizer = torchvision.transforms.Normalize(mean=mean, std=std)

    model = torch.nn.Sequential(normalizer, model)
    model = model.eval()
    return model

