import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms.functional import convert_image_dtype


def create_sequence(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        feature = data.iloc[i: i + seq_length]
        target = data.iloc[i + seq_length]["Kp"]
        sequences.append((feature, target))
    return sequences


class ImageAndKpDataset(Dataset):
    def __init__(self, sequences, img_dir, img_transform=None):
        self.sequences = sequences
        self.img_transform = img_transform
        self.img_dir = img_dir

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        features, target = self.sequences[idx]
        images = []
        numerical_features = []
        for _, row in features.iterrows():
            img_filename = row.pop("Image_filename")
            img_path = os.path.join(self.img_dir, img_filename)
            image = read_image(img_path)
            image = convert_image_dtype(image, torch.float32)
            if self.img_transform:
                image = self.img_transform(image)
            images.append(image)
            numerical_features.append(row.values)
        images = torch.stack(images)
        numerical_features = torch.tensor(
            np.array(numerical_features).astype(float), dtype=torch.float32
        )
        target = torch.tensor(target, dtype=torch.float32).unsqueeze(-1)
        return images, numerical_features, target
