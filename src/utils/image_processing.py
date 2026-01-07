import numpy as np
from scipy import ndimage
import cv2

from typing import Literal, List, Tuple

if "AVX2" in np.__config__.CONFIG["SIMD Extensions"]["found"]:
    from fast_slic.avx2 import SlicAvx2 as Slic
else:
    from fast_slic import Slic

COLORMAP: np.ndarray = cv2.applyColorMap(np.arange(0, 256).astype(np.uint8), cv2.COLORMAP_RAINBOW)




def superpixel_segmentation(image: np.ndarray, num_components=600, compactness=10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs superpixel segmentation on an image using the SLIC algorithm.

    Args:
        image (np.ndarray): The input image (NumPy array).
        num_components (int): The desired number of superpixels.
        compactness (int): The compactness parameter for the SLIC algorithm. Larger values result in more compact superpixels.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - labels (np.ndarray): An integer array of the same size as the image,
                                   where each element is the superpixel label for the corresponding pixel.
            - centroids (np.ndarray): An array of shape (n_superpixels, 2) containing the (x, y)
                                      coordinates of the centroid for each superpixel.
    """
    # fast-slic documentation: https://github.com/m-krastev/fast-slic/tree/master

    # Create SLIC superpixels
    slic = Slic(num_components=num_components, compactness=compactness)

    # Get the labels and centroids of the segments
    labels = slic.iterate(image)

    # Calculate centroids. np.where returns (row_indices, col_indices, ...), median is applied per dim,
    # [::-1] is used to swap (y, x) to (x, y) coordinates.
    # centroids = np.array([np.median(np.where(labels == l), axis=1)[::-1] for l in np.unique(labels)], dtype=int)

    # index=unique_labels tells scipy which labels to calculate for
    # It returns a list of (row, col) tuples
    centroids = np.array(ndimage.center_of_mass(np.ones_like(labels), labels, index=np.unique(labels)), dtype=int)

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
