import sys
import os

# Get the directory of the current script (main.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from mlp_block import ModifiedYOLOv8Head, MLPBlock
from transform_block import YOLOv8HeadWithTransform, PointwiseConv  # Import the new head
from ultralytics import YOLO
import torch.nn as nn
from yolo_loss import CustomLoss
from CustomAugmentations import CustomAugmentations
import albumentations as A
import cv2
import torch
import numpy as np
import logging
import random

# Configure logging (optional)
logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Augmentation config
AUG_CONFIG = {
    'horizontal_flip': True,   # Safe and very useful
    'rotation': 15,            # Small rotation to avoid bbox issues
    'brightness': 0.2,         # Mild brightness change
    'blur': False,             # Disable at first
    'cutout': False,           # Disable at first
    'affine': False,           # Disable at first
    'clahe': False             # Disable at first
}

def inject_augmentations(trainer):
    original_dataset = trainer.train_loader.dataset
    augmentor = CustomAugmentations(AUG_CONFIG)
    original_getitem = original_dataset.__getitem__

    def augmented_getitem(index):
        img, labels, paths, shapes = original_getitem(index)
        if original_dataset.augment:
            img_np = img.permute(1, 2, 0).numpy()
            boxes = labels[:, 1:].numpy()
            class_ids = labels[:, 0].int().numpy()
            try:
                transformed = augmentor(image=img_np, bboxes=boxes, class_labels=class_ids)
                img = torch.from_numpy(transformed['image']).permute(2, 0, 1).float()
                labels = torch.cat([torch.from_numpy(transformed['class_labels']).unsqueeze(1).float(), torch.from_numpy(transformed['bboxes']).float()], dim=1)
            except Exception as e:
                print(f"Augmentation failed for {paths}: {str(e)}")
                return img, labels, paths, shapes
        return img, labels, paths, shapes

    original_dataset.__getitem__ = augmented_getitem

class ModelHandler:
    def __init__(self, use_mlp=False, use_transform_head=False, data="dataset.yaml", epochs=1, imgsz=640, batch=4, augment=False, plots=True, name="run"):
        self.use_mlp = use_mlp
        self.use_transform_head = use_transform_head
        self.data = data
        self.epochs = epochs
        self.imgsz = imgsz
        self.batch = batch
        self.augment = augment
        self.plots = plots
        self.name = name
        self.model_type = "default"
        if use_mlp:
            self.model_type = "mlp"
        elif use_transform_head:
            self.model_type = "transform"

        self.model = YOLO("yolov8n.pt")
        self._modify_head()
        self.model.loss = CustomLoss(self.model.model)
        self.model.add_callback("on_train_start", inject_augmentations)

    def _get_backbone_output_channels(self):
        modules = self.model.model.model
        for i in reversed(range(len(modules) - 1)):
            m = modules[i]
            if isinstance(m, nn.Conv2d):
                return m.out_channels
            elif hasattr(m, 'conv') and isinstance(m.conv, nn.Conv2d):
                return m.conv.out_channels
            elif isinstance(m, nn.Sequential):
                for sub_m in reversed(m):
                    if isinstance(sub_m, nn.Conv2d):
                        return sub_m.out_channels
                    elif hasattr(sub_m, 'conv') and isinstance(sub_m.conv, nn.Conv2d):
                        return sub_m.conv.out_channels
                if hasattr(self, 'backbone_out_channels'):
                    return self.backbone_out_channels
            if hasattr(self, 'backbone_out_channels'):
                return self.backbone_out_channels
        raise ValueError("Could not determine backbone output channels.")

    def _modify_head(self):
        backbone_out_channels = self._get_backbone_output_channels()
        detect_layer = self.model.model.model[-1]
        nc = detect_layer.nc
        anchors = detect_layer.anchors

        if self.use_mlp:
            custom_head = ModifiedYOLOv8Head(nc=nc, anchors=anchors, ch=[backbone_out_channels])
            self.model.model.model[-1] = custom_head
            print("Using Modified YOLOv8 Head with MLP")
            logging.info("Using Modified YOLOv8 Head with MLP")
        elif self.use_transform_head:
            from transform_block import YOLOv8HeadWithTransform  # Import here to avoid circular dependency if transform_block imports this
            custom_head = YOLOv8HeadWithTransform(nc=nc, anchors=anchors, ch=[backbone_out_channels])
            self.model.model.model[-1] = custom_head
            print("Using YOLOv8 Head with Transformation")
            logging.info("Using YOLOv8 Head with Transformation")
        else:
            print("Using Default YOLOv8 Head")
            logging.info("Using Default YOLOv8 Head")

    def train(self):
        self.model.train(data=self.data, epochs=self.epochs, imgsz=self.imgsz, batch=self.batch, augment=self.augment, plots=self.plots, name=self.name)

    def info(self):
        self.model.info()

if __name__ == "__main__":
    # Control which head to use
    use_mlp_head = True
    use_transform_head = False#  # Set to True to use the transformation head, False otherwise

    # Ensure only one custom head is active
    if use_mlp_head and use_transform_head:
        raise ValueError("Only one custom head (MLP or Transformation) can be active at a time.")

    # Training parameters
    training_params = {
        "data": "dataset.yaml",
        "epochs": 40,
        "imgsz": 640,
        "batch": 16,
        "augment": False, # Custom augmentation
        "plots": True,
        "name": f"run_{'mlp' if use_mlp_head else ('transform' if use_transform_head else 'default')}"
    }

    # Initialize the model handler
    model_handler = ModelHandler(use_mlp=use_mlp_head, use_transform_head=use_transform_head, **training_params)

    # Print model information
    model_handler.info()

    # Start training
    model_handler.train()
