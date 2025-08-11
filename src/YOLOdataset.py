# custom_dataset.py
from ultralytics.data.dataset import YOLODataset
from custom_transforms import get_custom_transform
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import torch


class CustomYOLODataset(YOLODataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_transform = get_custom_transform()

    def __getitem__(self, index):
        # Use *args to handle variable number of return values
        results = super().__getitem__(index)
        img = results[0]  # First element is the image
        labels = results[1]  # Second element is the labels
        
        if isinstance(img, np.ndarray):
            if img.shape[0] in [1, 3]:  # CHW format
                img = img.transpose(1, 2, 0)  # to HWC
            img = Image.fromarray(img.astype('uint8'), 'RGB')
        
        img = self.custom_transform(img)
        
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)
            
        return img, labels  # Return only img and labels
