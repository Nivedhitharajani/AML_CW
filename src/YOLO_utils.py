import torch
from torchvision import transforms
from PIL import Image

class MyYOLOUtils:
    def __init__(self, flip_prob=0.5, rotation_degrees=15, reshape_size=(640, 640)):
        self.flip_prob = flip_prob
        self.rotation_degrees = rotation_degrees
        self.reshape_size = reshape_size
        self.to_tensor = transforms.ToTensor()

    def apply_transforms(self, image):
        # Convert image to tensor
        image = self.to_tensor(image)

        # Random horizontal flip
        if torch.rand(1) < self.flip_prob:
            image = torch.flip(image, dims=[2])  # Flip along the width dimension

        # Random rotation
        angle = torch.FloatTensor(1).uniform_(-self.rotation_degrees, self.rotation_degrees).item()
        image = transforms.functional.rotate(image, angle, interpolation=transforms.InterpolationMode.BILINEAR)

        # Resize image
        image = transforms.Resize(self.reshape_size)(image)

        return image
