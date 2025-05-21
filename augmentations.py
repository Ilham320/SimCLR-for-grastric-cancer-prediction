import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import random

class MultiViewTransform:
    def __init__(self, config):
        global_config = config['global_crops']
        local_config = config['local_crops']
        
        # Define histology-specific color normalization
        normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Global view transform with histology-specific augmentations
        self.global_transform = T.Compose([
            T.Resize((global_config['size'], global_config['size'])),
            T.RandomResizedCrop(
                size=global_config['size'],
                scale=global_config['scale'],
                ratio=(0.75, 1.333)
            ),
            T.RandomApply([
                T.ColorJitter(
                    brightness=global_config['color_jitter']['brightness'],
                    contrast=global_config['color_jitter']['contrast'],
                    saturation=global_config['color_jitter']['saturation'],
                    hue=global_config['color_jitter']['hue']
                )
            ], p=0.8),
            T.RandomApply([
                T.GaussianBlur(
                    kernel_size=global_config['gaussian_blur']['kernel_size'],
                    sigma=global_config['gaussian_blur']['sigma']
                )
            ], p=0.5),
            T.RandomGrayscale(p=global_config['grayscale_prob']),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(p=0.3),  # Added vertical flip for medical images
            T.RandomAffine(  # Adding subtle rotation/translation for histology
                degrees=10,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                fill=0
            ),
            T.ToTensor(),
            normalize
        ])

        # Local view transform
        self.local_transform = T.Compose([
            T.Resize((local_config['size'], local_config['size'])),
            T.RandomResizedCrop(
                size=local_config['size'],
                scale=local_config['scale'],
                ratio=(0.75, 1.333)
            ),
            T.RandomApply([
                T.ColorJitter(
                    brightness=global_config['color_jitter']['brightness'] * 0.8,
                    contrast=global_config['color_jitter']['contrast'] * 1.2,  # Enhanced contrast for local views
                    saturation=global_config['color_jitter']['saturation'] * 0.8,
                    hue=global_config['color_jitter']['hue'] * 0.8
                )
            ], p=0.9),  # Higher probability for local views
            T.RandomApply([
                T.GaussianBlur(
                    kernel_size=global_config['gaussian_blur']['kernel_size'] - 6,  # Smaller kernel for local views
                    sigma=global_config['gaussian_blur']['sigma']
                )
            ], p=0.4),
            T.RandomGrayscale(p=global_config['grayscale_prob']),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(p=0.3),
            T.RandomAffine(
                degrees=15,  # Slightly higher rotation for local views
                translate=(0.12, 0.12),
                scale=(0.85, 1.15),
                fill=0
            ),
            T.ToTensor(),
            normalize
        ])

        self.num_local_views = local_config['num']

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            raise ValueError(f"Expected PIL Image but got {type(img)}")

        try:
            # Apply global transforms
            global_views = [
                self.global_transform(img)
                for _ in range(2)
            ]

            # Apply local transforms
            local_views = [
                self.local_transform(img)
                for _ in range(self.num_local_views)
            ]

            return global_views + local_views
        except Exception as e:
            print(f"Transform error: {str(e)}")
            # Return zero tensors as fallback
            global_size = 224  # Default global view size
            local_size = 96   # Default local view size
            zero_global_views = [torch.zeros(3, global_size, global_size) for _ in range(2)]
            zero_local_views = [torch.zeros(3, local_size, local_size) for _ in range(self.num_local_views)]
            return zero_global_views + zero_local_views

def get_eval_transform(size=224):
    return T.Compose([
        T.Resize(int(size * 1.15)),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])