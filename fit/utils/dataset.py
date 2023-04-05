import random
import os
import json
import numpy as np

from torch.utils.data import Dataset
from torchvision.io import read_image

from torchvision import transforms as T

class InferenceDataset(Dataset):
    def __init__(self, path, n=None):
        self.img_dir = path
        self.truths = json.load(open(os.path.join(path, "labels.json", "r")))
                                
        self.images = [e for e in os.listdir(path) if e.lower().endswith(".jpg")]
        if n is not None:
            self.images = random.choices(self.images, k=n)

        self.transform = T.Compose([T.Resize(256), T.CenterCrop(224)])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = read_image(os.path.join(self.img_dir, self.images[idx]))
        img = self.transform(img).float()

        y = self.truths[self.images[idx][:-5]]
        return img / 255, y, idx