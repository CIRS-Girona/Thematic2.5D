import numpy as np
from scipy import ndimage
import cv2

from typing import Literal, Tuple

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

    # index=unique_labels tells scipy which labels to calculate for
    # It returns a list of (row, col) tuples
    centroids = np.array(ndimage.center_of_mass(np.ones_like(labels), labels, index=np.unique(labels)), dtype=int)

    return labels, centroids


def apply_mask(image: np.ndarray, mask: np.ndarray, max_val: int = 1, mode: Literal['contours', 'highlight'] = 'highlight', border_thickness: int = 2, beta: float = 0.3) -> np.ndarray:
    """
    Applies a mask to an image, visualizing the mask values using a colormap.

    The mask can be visualized either as contours around regions of the same value
    or by highlighting the regions with colors from the colormap.

    Args:
        image (np.ndarray): The input BGR image (NumPy array).
        mask (np.ndarray): An integer array of the same spatial dimensions as the image,
                           containing integer values representing different regions or categories.
                           Negative values in the mask are ignored.
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
        scaled_val = int(255 * val / (max_val if max_val != 0 else 1))
        color = COLORMAP[np.clip(scaled_val, 0, 255)]

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


def meanIoU(mask_gt: np.ndarray, mask: np.ndarray) -> float:
    """
    Calculate the mean Intersection over Union (IoU) score between two segmentation masks.

    Args:
        mask_gt (np.ndarray): Ground truth segmentation mask, where each element represents a class label.
        mask (np.ndarray): Predicted segmentation mask, with the same shape and class labels as `mask_gt`.

    Returns:
        float: The mean IoU score across all classes. Returns a value between 0.0 and 1.0, where 1.0 indicates perfect overlap and 0.0 indicates no overlap.
    """
    # Get unique classes in both masks
    classes = np.unique(np.concatenate([mask_gt, mask]))

    iou_scores = 0.0
    for cls in classes:
        # Create binary masks for the current class
        mask1_cls = mask_gt == cls
        mask2_cls = mask == cls

        # Calculate intersection and union
        intersection = np.sum(np.logical_and(mask1_cls, mask2_cls))
        union = np.sum(np.logical_or(mask1_cls, mask2_cls))

        # Calculate IoU for the class (handle case where union is 0)
        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = intersection / union

        iou_scores += iou

    return iou_scores / len(classes)
