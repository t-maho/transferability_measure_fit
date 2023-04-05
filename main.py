import os
import torch
import numpy as np

from fit.fit import FiT

from torchvision import transforms as T
from torchvision.io import read_image

# Define sources and target

sources = ["convit_small", "mixnet_m", "tf_efficientnet_b0"]
target = "resnet50"


# Define FiT object

fit = FiT(
    device="cuda",
    batch_size=32,
    fbi_images=200,
    transq_method="1",
    bin_steps=10,
    attacks=["di"]
)


# Load Images attack

X = []
images_folder = "./data/images"
transform = T.Compose([T.Resize(256), T.CenterCrop(224)])
for file in os.listdir(images_folder):
    X.append(os.path.join(images_folder, file))
    img = read_image(os.path.join(images_folder, file))
    img = transform(img).float()
    X.append(img.unsqueeze(0) / 255)

X = torch.cat(X, dim=0)

# Predictions of all sources have been precomputed

predictions = np.load()

# Run FiT
transferable_direction = fit(X, sources, target)
