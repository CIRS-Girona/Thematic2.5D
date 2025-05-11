import numpy as np
from typing import Literal, List, Tuple
import cv2

COLORMAP: np.ndarray = cv2.applyColorMap(np.arange(0, 256).astype(np.uint8), cv2.COLORMAP_RAINBOW)


def contrast_enhancement(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance image contrast.

    Enhances the value (V) channel of the input BGR image in HSV color space.

    Args:
        image (np.ndarray): The input BGR image (NumPy array).
        clip_limit (float): Threshold for contrast limiting.
        tile_grid_size (Tuple[int, int]): Size of the grid for histogram equalization.

    Returns:
        np.ndarray: The contrast-enhanced BGR image (NumPy array).
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    # The value channel is the intensity of the image (gray scale)
    hsv_image[:, :, 2] = clahe.apply(hsv_image[:, :, 2])
    enhanced_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return enhanced_image


def contrast_stretch(image: np.ndarray) -> np.ndarray:
    """
    Applies contrast stretching to the input image using percentile values.

    Stretches the intensity range of each channel based on the 1.5th and 98.5th percentiles
    to improve visibility.

    Args:
        image (np.ndarray): The input image (NumPy array). Expected to be BGR or grayscale.

    Returns:
        np.ndarray: The contrast-stretched image (NumPy array), with pixel values scaled to 0-255.
    """
    # Convert to float to avoid overflow during calculations
    image_float = image.astype(np.float32)

    # Apply contrast stretching formula
    # Calculate min/max values based on percentiles for each channel
    min_vals = np.percentile(image_float, 1.5, axis=(0, 1))
    max_vals = np.percentile(image_float, 98.5, axis=(0, 1))

    # If a channel has a uniform color (min == max), avoid division by 0
    zero_range_indices = max_vals - min_vals == 0
    max_vals[zero_range_indices] = min_vals[zero_range_indices] + 1 # Add a small epsilon

    # Stretch image and clip values to [0, 255]
    stretched_image = (image_float - min_vals) / (max_vals - min_vals)
    return np.clip(255 * stretched_image, 0, 255).astype(np.uint8)


def superpixel_segmentation(image: np.ndarray, region_size: int = 40, ruler: float = 10.0, n_iters: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs superpixel segmentation on an image using the SLIC algorithm.

    Args:
        image (np.ndarray): The input image (NumPy array).
        region_size (int): The average desired superpixel size in pixels.
        ruler (float): Affects the trade-off between compactness and boundary adherence.
                       Larger values result in more compact superpixels.
        n_iters (int): The number of iterations for the algorithm.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - labels (np.ndarray): An integer array of the same size as the image,
                                   where each element is the superpixel label for the corresponding pixel.
            - centroids (np.ndarray): An array of shape (n_superpixels, 2) containing the (x, y)
                                      coordinates of the centroid for each superpixel.
    """
    # OpenCV Documentation: https://docs.opencv.org/3.4/df/d6c/group__ximgproc__superpixel.html#gacf29df20eaca242645a8d3a2d88e6c06

    # Create SLIC superpixels
    slic = cv2.ximgproc.createSuperpixelSLIC(image, region_size=region_size, ruler=ruler, algorithm=cv2.ximgproc.SLICO)
    slic.iterate(n_iters)  # Number of iterations

    # Get the labels and centroids of the segments
    labels = slic.getLabels()
    # Calculate centroids. np.where returns (row_indices, col_indices, ...), median is applied per dim,
    # [::-1] is used to swap (y, x) to (x, y) coordinates.
    centroids = np.array([np.median(np.where(labels == l), axis=1)[::-1] for l in np.unique(labels)], dtype=int)

    return labels, centroids


def apply_mask(image: np.ndarray, mask: np.ndarray, min_val: int = 0, max_val: int = 1, mode: Literal['contours', 'highlight'] = 'highlight', border_thickness: int = 2, beta: float = 0.3) -> np.ndarray:
    """
    Applies a mask to an image, visualizing the mask values using a colormap.

    The mask can be visualized either as contours around regions of the same value
    or by highlighting the regions with colors from the colormap.

    Args:
        image (np.ndarray): The input BGR image (NumPy array).
        mask (np.ndarray): An integer array of the same spatial dimensions as the image,
                           containing integer values representing different regions or categories.
                           Negative values in the mask are ignored.
        min_val (int): The minimum possible value in the mask, used for colormap scaling.
        max_val (int): The maximum possible value in the mask, used for colormap scaling.
        mode (Literal['contours', 'highlight']): The visualization mode.
                                               'contours': Draw contours around mask regions.
                                               'highlight': Overlay colored regions on the image.
        border_thickness (int): Thickness of the contours if mode is 'contours'.
        beta (float): Transparency factor for the overlay if mode is 'highlight'.
                      The output is `image * (1 - beta) + color_overlay * beta`.

    Returns:
        np.ndarray: The output image (NumPy array) with the mask visualization applied.
    """
    output_image = image.copy()

    valid_vals = np.unique(mask[mask >= 0])

    for val in valid_vals:
        scaled_val = int(255 * (val - min_val) / (max_val - min_val if max_val != min_val else 1))
        scaled_val = np.clip(scaled_val, 0, 255)
        color = COLORMAP[scaled_val]

        if mode == 'contours':
            value_mask = (mask == val).astype(np.uint8) * 255
            contours, _ = cv2.findContours(value_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(output_image, contours, -1, color, border_thickness)
        elif mode == 'highlight':
            output_image[mask == val] = color

    if mode == 'highlight':
        # Blend the original image with the colored overlay
        output_image = cv2.addWeighted(image, 1 - beta, output_image, beta, 0)

    return output_image


def process_images(images: List[np.ndarray], enhance_contrast: bool = True, stretch_contrast: bool = True, clip_limit: float = 40.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies a sequence of image processing steps (contrast enhancement and stretching)
    to a list of images and returns the results in grayscale and HSV format.

    Args:
        images (List[np.ndarray]): A list of input BGR images (NumPy arrays).
        enhance_contrast (bool): Whether to apply contrast enhancement (CLAHE).
        stretch_contrast (bool): Whether to apply contrast stretching.
        clip_limit (float): Clip limit for CLAHE if enhance_contrast is True.
        tile_grid_size (Tuple[int, int]): Tile grid size for CLAHE if enhance_contrast is True.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing two NumPy arrays:
            - A stacked array of grayscale images.
            - A stacked array of HSV images.
        Both arrays have shape (n_images, height, width) or (n_images, height, width, 3).
    """
    images_gray: List[np.ndarray] = []
    images_hsv: List[np.ndarray] = []

    for image in images:
        processed_image = image.copy()

        if enhance_contrast:
            processed_image = contrast_enhancement(processed_image, clip_limit=clip_limit, tile_grid_size=tile_grid_size)

        if stretch_contrast:
            processed_image = contrast_stretch(processed_image)

        images_gray.append(cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY))
        images_hsv.append(cv2.cvtColor(processed_image, cv2.COLOR_BGR2HSV))

    return np.array(images_gray), np.array(images_hsv)