import numpy as np
from typing import Literal
import cv2

COLORMAP = cv2.applyColorMap(np.arange(0, 256).astype(np.uint8), cv2.COLORMAP_HSV)


def contrast_enhancement(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    hsv_image[:, :, 2] = clahe.apply(hsv_image[:, :, 2])  # The value channel is the intensity of the image (gray scale)
    enhanced_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return enhanced_image


def contrast_stretch(image: np.ndarray) -> np.ndarray:
    # Convert to float to avoid overflow during calculations
    image_float = image.astype(np.double)

    # Apply contrast stretching formula
    min_vals = np.percentile(image_float, 1.5, axis=(0, 1))
    max_vals = np.percentile(image_float, 98.5, axis=(0, 1))

    # If a channel has a uniform color, avoid division by 0
    if np.any(max_vals - min_vals == 0):
        index = max_vals - min_vals == 0
        max_vals[index] = min_vals[index] + 1

    # Stretch image and clip values
    stretched_image = (image_float - min_vals) / (max_vals - min_vals)
    return np.clip(255 * stretched_image, 0, 255).astype(np.uint8)


def superpixel_segmentation(image: np.ndarray, region_size: int = 40, ruler: float = 10.0, n_iters: int = 10) -> tuple[np.ndarray, np.ndarray]:
    # OpenCV Documentation: https://docs.opencv.org/3.4/df/d6c/group__ximgproc__superpixel.html#gacf29df20eaca242645a8d3a2d88e6c06

    # Create SLIC superpixels
    slic = cv2.ximgproc.createSuperpixelSLIC(image, region_size=region_size, ruler=ruler, algorithm=cv2.ximgproc.SLICO)
    slic.iterate(n_iters)  # Number of iterations

    # Get the labels and centroids of the segments
    labels = slic.getLabels()
    centroids = np.array([np.median(np.where(labels == l), axis=1)[::-1] for l in np.unique(labels)], dtype=int)

    return labels, centroids


def apply_mask(image: np.ndarray, mask: np.ndarray, mode: Literal['contours', 'highlight'] = 'highlight', border_thickness: int = 2, alpha: float = 0.3) -> np.ndarray:
    output_image = image.copy()

    valid_vals = np.unique(mask[mask >= 0])
    min_val, max_val = valid_vals[0], valid_vals[-1]
    for val in valid_vals:
        if min_val == max_val:
            color = COLORMAP[0]
        else:
            color = int(255 * (val - min_val) / (max_val - min_val))
            color = COLORMAP[color]

        if mode == 'contours':
            contours, _ = cv2.findContours(mask[mask == val], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(output_image, contours, -1, color, border_thickness)
        else:
            output_image[mask == val] = color

    if mode == 'highlight':
        output_image = cv2.addWeighted(image, 1 - alpha, output_image, alpha, 0)

    return output_image


def process_images(images: list[np.ndarray], correct_color: bool = True, stretch_contrast: bool = True, clip_limit: float = 40.0, tile_grid_size: tuple[int, int] = (8, 8)) -> tuple[np.ndarray, np.ndarray]:
    images_gray = []
    images_hsv = []
    
    for image in images:
        if correct_color:
            image = contrast_enhancement(image, clip_limit=clip_limit, tile_grid_size=tile_grid_size)

        if stretch_contrast:
            image = contrast_stretch(image)

        images_gray.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        images_hsv.append(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))

    return np.array(images_gray), np.array(images_hsv)

