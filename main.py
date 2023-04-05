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
# Examples of images are in ./data/images. 
# It's a subset of 5 images from the set used in the paper's evaluation.

print("Load Images")
X = []
y = []
images_folder = "./data/images"
transform = T.Compose([T.Resize(256), T.CenterCrop(224)])
for file in os.listdir(images_folder):
    img = read_image(os.path.join(images_folder, file))
    img = transform(img).float()
    X.append(img.unsqueeze(0) / 255)

    label = file.split("-label-")[1].split(".")[0]
    y.append(int(label))
X = torch.cat(X, dim=0)
y = torch.tensor(y)

print(X.shape)
print(y.shape)

# Predictions of all sources have been precomputed
print("Load predictions precomputed")
truths = np.load(os.path.join("data/predictions-precomputed/labels.npy"))
predictions = {}
for m_i, m in enumerate(sources):
    p = np.load(os.path.join("./data/predictions-precomputed/", m + ".npy"))
    predictions[m] = p[:, 0]

# Run FiT
print("Run FiT")
transferable_direction = fit(X, sources, target, predictions=predictions, y=y)
