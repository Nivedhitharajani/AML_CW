import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations.dropout.coarse_dropout import CoarseDropout

class CustomAugmentations:
    def __init__(self, config):
        """
        Initialize with augmentation configuration.
        """
        self.config = config
        self.transform = self._build_pipeline()

    def _build_pipeline(self):
        """Build the augmentation pipeline based on config."""
        return A.Compose(
            transforms=self._get_transforms(),
            bbox_params=A.BboxParams(
                format='yolo',
                min_visibility=0.5,  # Stricter visibility filter
                label_fields=['class_labels']
            )
        )

    def _get_transforms(self):
        """Generate list of transforms based on config."""
        transforms = []

        # --- Geometric Transforms ---
        if self.config.get('horizontal_flip', False):
            transforms.append(A.HorizontalFlip(p=0.5))

        if self.config.get('rotation'):
            transforms.append(A.Rotate(limit=self.config['rotation'], p=0.3))  # reduced prob

        if self.config.get('affine', False):
            transforms.append(A.Affine(
                scale=(0.9, 1.1),  # Narrowed scaling to avoid excessive distortion
                shear=10,
                p=0.3
            ))

        # --- Color Transforms ---
        if self.config.get('brightness'):
            transforms.append(A.RandomBrightnessContrast(
                brightness_limit=self.config['brightness'],
                contrast_limit=0.1,
                p=0.4
            ))

        if self.config.get('clahe', False):
            transforms.append(A.CLAHE(p=0.3))  # reduced to avoid over-enhancement

        transforms.append(
            A.OneOf([
                A.HueSaturationValue(p=0.5),
                A.RGBShift(p=0.5),
            ], p=0.3)
        )

        # --- Blur ---
        if self.config.get('blur', False):
            transforms.append(A.GaussianBlur(
                blur_limit=(3, 7),
                p=0.2
            ))

        # --- Cutout ---
        if self.config.get('cutout', False):
            transforms.append(A.CoarseDropout(
                max_holes=4,         # reduced number
                max_height=32,
                max_width=32,
                min_holes=1,
                min_height=8,
                min_width=8,
                fill_value=0,
                p=0.5
            ))

        # --- Mandatory ---
        transforms.extend([
            A.Resize(640, 640),
            ToTensorV2()
        ])

        return transforms

    def __call__(self, image, bboxes, class_labels):
        """
        Apply augmentations to image and bounding boxes.
        """
        transformed = self.transform(
            image=image,
            bboxes=bboxes,
            class_labels=class_labels
        )
        return transformed['image'], transformed['bboxes'], transformed['class_labels']
