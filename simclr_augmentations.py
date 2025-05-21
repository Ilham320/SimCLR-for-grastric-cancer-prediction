import torch
import torchvision.transforms as T
from PIL import Image, ImageFilter
import random
import numpy as np
from scipy import ndimage
import math

class HistologyTransform:
    """
    SimCLR augmentation strategy optimized for histology images.
    Creates two differently augmented versions of each image.
    """
    def __init__(self, config, eval_mode=False):
        self.config = config
        self.eval_mode = eval_mode
        
        # Define histology-specific color normalization
        normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # H&E staining specific transforms
        self.size = config.get('image_size', 224)
        
        # Get augmentation parameters from config
        aug_config = config.get('augmentation', {})
        color_jitter_strength = aug_config.get('color_jitter_strength', 0.5)
        gaussian_blur_prob = aug_config.get('gaussian_blur_prob', 0.5)
        grayscale_prob = aug_config.get('grayscale_prob', 0.2)
        crop_scale = aug_config.get('random_resized_crop_scale', [0.25, 1.0])
        
        # Use histology stain augmentation if specified
        use_stain_aug = aug_config.get('use_histology_stain_aug', False)
        use_cutmix = aug_config.get('use_cutmix', False)
        cutmix_prob = aug_config.get('cutmix_prob', 0.3)
        use_strong_color = aug_config.get('use_strong_color_transforms', False)
        
        if self.eval_mode:
            # Simple transform for evaluation mode
            self.transform = T.Compose([
                T.Resize((self.size, self.size)),
                T.ToTensor(),
                normalize,
            ])
        else:
            # Enhanced augmentation pipeline for training
            color_transforms = []
            
            # Basic color jitter
            color_transforms.append(T.ColorJitter(
                brightness=0.4 * color_jitter_strength,
                contrast=0.4 * color_jitter_strength,
                saturation=0.2 * color_jitter_strength,
                hue=0.05 * color_jitter_strength
            ))
            
            # Use stronger color transforms if specified
            if use_strong_color:
                color_transforms.append(AdvancedColorTransform())
            
            # Primary transformations for histology images
            transform_list = [
                T.RandomResizedCrop(
                    size=self.size,
                    scale=crop_scale,
                    ratio=(0.7, 1.4)
                ),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomApply(color_transforms, p=0.8),
                T.RandomApply([T.GaussianBlur(
                    kernel_size=23,  # Fixed kernel size (must be odd)
                    sigma=(0.1, 2.0)
                )], p=gaussian_blur_prob),
                T.RandomGrayscale(p=grayscale_prob)
            ]
            
            # Add histology-specific transforms
            if use_stain_aug:
                transform_list.append(AdvancedHistologyAugmentations(p=0.5))
            else:
                transform_list.append(SafeHistologyAugmentations(p=0.3))
                
            # Add CutMix for histology if specified
            if use_cutmix:
                transform_list.append(HistologyCutMix(p=cutmix_prob))
            
            # Final transforms
            transform_list.extend([
                T.ToTensor(),
                normalize,
            ])
            
            self.transform = T.Compose(transform_list)
        
    def __call__(self, x):
        """
        Generate two differently augmented versions of the image
        """
        try:
            # Ensure image is RGB and properly sized
            if not isinstance(x, Image.Image):
                raise TypeError(f"Expected PIL Image but got {type(x)}")
                
            x = x.convert('RGB')
                
            if self.eval_mode:
                # For evaluation, just return the image once
                return self.transform(x)
            else:
                # Apply transform twice for two views
                y1 = self.transform(x)
                y2 = self.transform(x)
                
                # Ensure tensors have correct shape
                if y1.shape[0] != 3 or y2.shape[0] != 3:
                    raise ValueError(f"Expected 3-channel images, got shapes {y1.shape} and {y2.shape}")
                    
                return y1, y2
            
        except Exception as e:
            print(f"Transform error: {str(e)}, returning zeros")
            # Return zero tensors as fallback with correct shapes
            if self.eval_mode:
                return torch.zeros(3, self.size, self.size)
            else:
                return torch.zeros(3, self.size, self.size), torch.zeros(3, self.size, self.size)


class SafeHistologyAugmentations:
    """
    Domain-specific augmentations for histology images with safety checks.
    """
    def __init__(self, p=0.3):
        self.p = p
    
    def __call__(self, img):
        """Apply random histology-specific augmentations"""
        if random.random() > self.p:
            return img
            
        try:
            if not isinstance(img, Image.Image):
                return img
                
            # Get image dimensions
            w, h = img.size
                
            # Convert to array for processing
            img_array = np.array(img).astype(np.float32) / 255.0
            
            # Apply one of several histology-specific augmentations
            aug_type = random.randint(0, 3)
            
            if aug_type == 0:
                # Simulate H&E staining variation (safer version)
                h_factor = np.random.uniform(0.85, 1.15)
                e_factor = np.random.uniform(0.85, 1.15)
                
                # Apply color shifts
                if img_array.shape[2] >= 3:  # Ensure RGB
                    img_array[:, :, 0] = np.clip(img_array[:, :, 0] * e_factor, 0, 1)
                    img_array[:, :, 2] = np.clip(img_array[:, :, 2] * h_factor, 0, 1)
                
            elif aug_type == 1:
                # Simulate tissue fold (safer version)
                fold_width = max(1, int(w * random.uniform(0.01, 0.03)))
                fold_intensity = random.uniform(0.7, 0.9)
                
                # Create a vertical line with random position
                x_pos = random.randint(fold_width, w - fold_width - 1)
                x_start = max(0, x_pos - fold_width // 2)
                x_end = min(w, x_pos + fold_width // 2)
                
                # Apply darkening to the vertical strip
                img_array[:, x_start:x_end, :] = img_array[:, x_start:x_end, :] * fold_intensity
                
            elif aug_type == 2:
                # Simulate intensity variation (safer version)
                intensity = np.random.uniform(0.85, 1.15)
                
                # Apply to a quarter of the image
                quad_size_h = h // 2
                quad_size_w = w // 2
                y_start = random.randint(0, h - quad_size_h)
                x_start = random.randint(0, w - quad_size_w)
                
                y_end = min(h, y_start + quad_size_h)
                x_end = min(w, x_start + quad_size_w)
                
                img_array[y_start:y_end, x_start:x_end, :] = np.clip(
                    img_array[y_start:y_end, x_start:x_end, :] * intensity, 0, 1
                )
                
            else:
                # Add slight Gaussian noise (safer version)
                noise_level = random.uniform(0.01, 0.03)
                noise = np.random.normal(0, noise_level, img_array.shape)
                img_array = np.clip(img_array + noise, 0, 1)
            
            # Convert back to PIL
            img_array = (img_array * 255).clip(0, 255).astype(np.uint8)
            return Image.fromarray(img_array)
            
        except Exception as e:
            print(f"Histology augmentation error: {str(e)}")
            return img  # Return original image if augmentation fails


class AdvancedHistologyAugmentations:
    """
    Advanced domain-specific augmentations for histology images to achieve higher accuracy.
    Memory-efficient implementation to avoid large array allocations.
    """
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, img):
        """Apply advanced histology-specific augmentations"""
        if random.random() > self.p:
            return img
            
        try:
            if not isinstance(img, Image.Image):
                return img
                
            # Get image dimensions
            w, h = img.size
                
            # Convert to array for processing
            img_array = np.array(img).astype(np.float32) / 255.0
            
            # Apply one of several advanced histology-specific augmentations
            aug_type = random.randint(0, 5)
            
            if aug_type == 0:
                # Advanced H&E staining normalization simulation
                # Simulate stain matrix variation
                h_channel = img_array[:,:,0] * 0.25 + img_array[:,:,2] * 0.75
                e_channel = img_array[:,:,0] * 0.65 + img_array[:,:,1] * 0.35
                
                # Apply stain normalization
                h_std = np.random.uniform(0.8, 1.2)
                e_std = np.random.uniform(0.8, 1.2)
                
                # Normalize and adjust stain components
                h_mean = h_channel.mean()
                h_std_val = h_channel.std() + 1e-8
                e_mean = e_channel.mean()
                e_std_val = e_channel.std() + 1e-8
                
                # Calculate normalized components without creating temporary arrays
                h_norm_factor = h_std / h_std_val
                e_norm_factor = e_std / e_std_val
                h_offset = random.uniform(0.9, 1.1) - h_mean * h_norm_factor
                e_offset = random.uniform(0.9, 1.1) - e_mean * e_norm_factor
                
                # Reconstruct image (approximately)
                if img_array.shape[2] >= 3:
                    # Mix normalized stains back into RGB channels
                    for i in range(h):
                        for j in range(w):
                            h_val = h_channel[i, j] * h_norm_factor + h_offset
                            e_val = e_channel[i, j] * e_norm_factor + e_offset
                            img_array[i, j, 0] = np.clip(e_val * 0.6 + h_val * 0.1, 0, 1)
                            img_array[i, j, 1] = np.clip(e_val * 0.4 + h_val * 0.2, 0, 1)
                            img_array[i, j, 2] = np.clip(h_val * 0.7, 0, 1)
                
            elif aug_type == 1:
                # Multiple tissue fold simulation
                folds = random.randint(1, 3)
                for _ in range(folds):
                    # Random fold orientation (vertical or horizontal)
                    is_vertical = random.random() > 0.5
                    
                    if is_vertical:
                        fold_width = max(1, int(w * random.uniform(0.01, 0.04)))
                        fold_intensity = random.uniform(0.6, 0.9)
                        
                        x_pos = random.randint(fold_width, w - fold_width - 1)
                        x_start = max(0, x_pos - fold_width // 2)
                        x_end = min(w, x_pos + fold_width // 2)
                        
                        # Create folder effect with gradient
                        for i in range(x_start, x_end):
                            # Calculate distance from center and intensity factor
                            dist_factor = 1.0 - abs(i - x_pos) / (fold_width / 2)
                            shadow_intensity = fold_intensity + (1.0 - fold_intensity) * (1.0 - dist_factor)
                            img_array[:, i, :] *= shadow_intensity
                    else:
                        fold_width = max(1, int(h * random.uniform(0.01, 0.04)))
                        fold_intensity = random.uniform(0.6, 0.9)
                        
                        y_pos = random.randint(fold_width, h - fold_width - 1)
                        y_start = max(0, y_pos - fold_width // 2)
                        y_end = min(h, y_pos + fold_width // 2)
                        
                        # Create folder effect with gradient
                        for i in range(y_start, y_end):
                            dist_factor = 1.0 - abs(i - y_pos) / (fold_width / 2)
                            shadow_intensity = fold_intensity + (1.0 - fold_intensity) * (1.0 - dist_factor)
                            img_array[i, :, :] *= shadow_intensity
                
            elif aug_type == 2:
                # Simulate uneven staining with memory-efficient implementation
                # Create intensity variation
                direction = random.randint(0, 3)  # 0: top-bottom, 1: left-right, 2: diagonal1, 3: diagonal2
                
                # Randomize intensity range with minimal memory usage
                intensity_range = random.uniform(0.1, 0.3)  # Max 30% variation
                ch_var = [random.uniform(0.8, 1.2), random.uniform(0.8, 1.2), random.uniform(0.8, 1.2)]
                
                # Apply gradient effect manually without creating large arrays
                for y in range(h):
                    for x in range(w):
                        # Calculate gradient value based on direction
                        if direction == 0:
                            gradient = y / float(h)
                        elif direction == 1:
                            gradient = x / float(w)
                        elif direction == 2:
                            gradient = (x + y) / float(w + h)
                        else:
                            gradient = (x + (h - y)) / float(w + h)
                        
                        # Randomize gradient direction
                        if random.random() > 0.5:
                            gradient = 1 - gradient
                            
                        # Calculate intensity adjustment
                        intensity_map = 1.0 - intensity_range + gradient * intensity_range
                        
                        # Apply to each channel
                        for c in range(3):
                            adjusted_map = intensity_map ** ch_var[c]
                            img_array[y, x, c] = np.clip(img_array[y, x, c] * adjusted_map, 0, 1)
                
            elif aug_type == 3:
                # Simulate slide scanning artifacts with memory-efficient implementation
                artifact_type = random.randint(0, 2)
                
                if artifact_type == 0:
                    # Simulate minor blur in one region
                    region_size = min(w, h) // random.randint(3, 6)
                    x_start = random.randint(0, w - region_size)
                    y_start = random.randint(0, h - region_size)
                    
                    # Extract region
                    region = img_array[y_start:y_start+region_size, x_start:x_start+region_size, :]
                    
                    # Apply gaussian blur
                    sigma = random.uniform(1.0, 3.0)
                    for c in range(3):
                        region[:,:,c] = ndimage.gaussian_filter(region[:,:,c], sigma=sigma)
                    
                    # Put back
                    img_array[y_start:y_start+region_size, x_start:x_start+region_size, :] = region
                    
                elif artifact_type == 1:
                    # Simulate scan line
                    is_horizontal = random.random() > 0.5
                    line_width = max(1, int(min(w, h) * 0.005))
                    line_intensity = random.uniform(0.7, 1.3)
                    
                    if is_horizontal:
                        y_pos = random.randint(0, h - 1)
                        y_start = max(0, y_pos - line_width // 2)
                        y_end = min(h, y_pos + line_width // 2 + 1)
                        img_array[y_start:y_end, :, :] = np.clip(img_array[y_start:y_end, :, :] * line_intensity, 0, 1)
                    else:
                        x_pos = random.randint(0, w - 1)
                        x_start = max(0, x_pos - line_width // 2)
                        x_end = min(w, x_pos + line_width // 2 + 1)
                        img_array[:, x_start:x_end, :] = np.clip(img_array[:, x_start:x_end, :] * line_intensity, 0, 1)
                        
                else:
                    # Simulate slight vignetting with memory-efficient implementation
                    center_x, center_y = w / 2, h / 2
                    max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
                    vignette_strength = random.uniform(0.05, 0.15)
                    
                    # Apply vignette effect manually
                    for y in range(h):
                        for x in range(w):
                            dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                            vignette_factor = 1.0 - np.clip(dist / max_dist * vignette_strength, 0, vignette_strength)
                            img_array[y, x, :] = np.clip(img_array[y, x, :] * vignette_factor, 0, 1)
                    
            elif aug_type == 4:
                # Simulate tissue heterogeneity with memory-efficient implementation
                num_patterns = random.randint(3, 8)
                pattern_strength = random.uniform(0.07, 0.15)
                
                for _ in range(num_patterns):
                    # Create random cellular-like pattern
                    pat_size = random.randint(20, 100)
                    pat_size = min(pat_size, min(w, h) - 1)  # Make sure pattern fits
                    x_pos = random.randint(0, w - pat_size)
                    y_pos = random.randint(0, h - pat_size)
                    
                    # Generate cellular-like pattern using noise
                    pattern = np.random.normal(0, 1, (pat_size, pat_size, 1)) * pattern_strength
                    # Smooth to make it more natural
                    pattern = ndimage.gaussian_filter(pattern, sigma=random.uniform(1.0, 3.0))
                    
                    # Adjust pattern to be both brighter and darker
                    if random.random() > 0.5:
                        pattern = -pattern
                    
                    # Apply pattern
                    y_end = min(y_pos + pat_size, h)
                    x_end = min(x_pos + pat_size, w)
                    p_h = y_end - y_pos
                    p_w = x_end - x_pos
                    
                    # Different effect on different channels
                    for c in range(3):
                        ch_factor = random.uniform(0.5, 1.5)
                        img_array[y_pos:y_end, x_pos:x_end, c] = np.clip(
                            img_array[y_pos:y_end, x_pos:x_end, c] + pattern[:p_h, :p_w, 0] * ch_factor, 
                            0, 1
                        )
                
            else:
                # Simulate staining inconsistency with memory-efficient approach
                # Create several random blobs
                num_blobs = random.randint(2, 6)
                blob_strength = random.uniform(0.1, 0.3)
                
                # Channel factors
                factors = [random.uniform(0.85, 1.15), random.uniform(0.85, 1.15), random.uniform(0.85, 1.15)]
                
                # Generate blob centers and sizes instead of full mask
                blobs = []
                for _ in range(num_blobs):
                    cx = random.randint(0, w-1)
                    cy = random.randint(0, h-1)
                    radius = random.randint(min(w, h) // 10, min(w, h) // 4)
                    blobs.append((cx, cy, radius))
                
                # Apply blob effects efficiently
                for y in range(h):
                    for x in range(w):
                        # Calculate max influence from all blobs
                        max_influence = 0
                        for cx, cy, radius in blobs:
                            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                            influence = max(0, 1.0 - dist / radius)
                            influence = influence ** 2  # Square to make it smoother
                            max_influence = max(max_influence, influence)
                            
                        # Skip if no influence
                        if max_influence <= 0:
                            continue
                            
                        # Apply mask with different factors per channel
                        adjustment = max_influence * blob_strength
                        for c in range(3):
                            intensity_factor = 1.0 + (factors[c] - 1.0) * adjustment
                            img_array[y, x, c] = np.clip(img_array[y, x, c] * intensity_factor, 0, 1)
            
            # Convert back to PIL
            img_array = (img_array * 255).clip(0, 255).astype(np.uint8)
            return Image.fromarray(img_array)
            
        except Exception as e:
            print(f"Advanced histology augmentation error: {str(e)}")
            return img  # Return original image if augmentation fails


class HistologyCutMix:
    """
    CutMix augmentation specialized for histology images.
    Mixes rectangular regions between the image and a rotated/transformed version of itself.
    """
    def __init__(self, p=0.3, alpha=1.0):
        self.p = p
        self.alpha = alpha
    
    def __call__(self, img):
        if random.random() > self.p:
            return img
            
        try:
            if not isinstance(img, Image.Image):
                return img
                
            # Create a transformed copy of the original image
            img2 = img.copy()
            
            # Apply random rotation to the second image
            angle = random.uniform(-180, 180)
            img2 = img2.rotate(angle, resample=Image.BICUBIC, expand=False)
            
            # Optional additional transform
            if random.random() > 0.5:
                # Apply a random affine transformation
                orig_width, orig_height = img.size
                
                # Scale params
                scale = random.uniform(0.8, 1.2)
                
                # Calculate shear
                shear = random.uniform(-10, 10)
                
                # Create an affine transform
                img2 = img2.transform(
                    (orig_width, orig_height), 
                    Image.AFFINE, 
                    (scale, 0, orig_width * (1-scale)/2,
                     0, scale, orig_height * (1-scale)/2),
                    resample=Image.BICUBIC
                )
            
            # Get image width and height
            width, height = img.size
            
            # Generate two random points for the rectangle
            lambda_param = np.random.beta(self.alpha, self.alpha)
            
            # Calculate rectangle area based on lambda
            cut_area = lambda_param * width * height
            
            # Calculate rectangle dimensions
            cut_ratio = np.sqrt(cut_area / (width * height))
            
            cut_w = int(width * cut_ratio)
            cut_h = int(height * cut_ratio)
            
            # Get random center point
            cx = random.randint(0, width)
            cy = random.randint(0, height)
            
            # Calculate rectangle boundaries
            left = max(0, cx - cut_w // 2)
            top = max(0, cy - cut_h // 2)
            right = min(width, cx + cut_w // 2)
            bottom = min(height, cy + cut_h // 2)
            
            # Convert to numpy arrays for processing
            img_array = np.array(img)
            img2_array = np.array(img2)
            
            # Create mixed image
            mixed_array = img_array.copy()
            mixed_array[top:bottom, left:right, :] = img2_array[top:bottom, left:right, :]
            
            # Return mixed image
            return Image.fromarray(mixed_array)
            
        except Exception as e:
            print(f"CutMix augmentation error: {str(e)}")
            return img  # Return original image if augmentation fails


class AdvancedColorTransform:
    """
    Advanced color transformations specially designed for histology images.
    """
    def __init__(self):
        pass
        
    def __call__(self, img):
        try:
            if not isinstance(img, Image.Image):
                return img
                
            # Randomly select a transformation
            transform_type = random.randint(0, 2)
            
            if transform_type == 0:
                # Channel rebalancing
                r, g, b = img.split()
                
                # Adjust channel levels
                r_factor = random.uniform(0.8, 1.2)
                g_factor = random.uniform(0.8, 1.2)
                b_factor = random.uniform(0.8, 1.2)
                
                # Apply factors using point operation
                r = r.point(lambda x: min(255, max(0, int(x * r_factor))))
                g = g.point(lambda x: min(255, max(0, int(x * g_factor))))
                b = b.point(lambda x: min(255, max(0, int(x * b_factor))))
                
                # Merge back
                img = Image.merge('RGB', (r, g, b))
                
            elif transform_type == 1:
                # Gamma correction
                gamma = random.uniform(0.7, 1.3)
                
                # Apply gamma correction
                img = img.point(lambda x: int(255 * ((x / 255) ** gamma)))
                
            else:
                # Contrast enhancement - simulates enhanced nuclear contrast often seen in H&E
                img = np.array(img).astype(np.float32)
                
                # Get mid-gray level
                mid = 128
                
                # Apply contrast adjustment
                contrast_factor = random.uniform(1.0, 1.5)
                img = np.clip((img - mid) * contrast_factor + mid, 0, 255).astype(np.uint8)
                
                # Convert back to PIL
                img = Image.fromarray(img)
            
            return img
            
        except Exception as e:
            print(f"Color transform error: {str(e)}")
            return img  # Return original image if transform fails