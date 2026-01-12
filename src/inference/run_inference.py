import numpy as np
import cv2
import os, time, logging
from typing import Tuple

from ..features import extract_features
from ..utils import superpixel_segmentation, apply_mask
from ..classification import SVMModel

logger = logging.getLogger(__name__)


def get_window_bounds(centers: np.ndarray, radius: int, max_val: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized calculation of window bounds with boundary clamping.
    - If window goes < 0, shift start to 0.
    - If window goes >= max, shift end to max-1.
    - Maintains window size of 2*radius.
    """
    starts = centers - radius
    ends = centers + radius
    
    # 1. Handle left/top edge underflow
    underflow_mask = starts < 0
    starts[underflow_mask] = 0
    ends[underflow_mask] = 2 * radius
    
    # 2. Handle right/bottom edge overflow
    # Original logic used (range[1] - 1) as the hard stop for 'end'
    limit = max_val
    overflow_mask = ends >= limit
    ends[overflow_mask] = limit - 1
    starts[overflow_mask] = limit - 1 - (2 * radius)
    
    return starts, ends


def run_inference(
        image_path: str,
        depth_path: str,
        models_dir: str,
        results_dir: str,
        max_uxo_code: int,
        num_components: int = 600,
        compactness: int = 10,
        window_size: int = 400,
        patch_size: int = 128,
        subdivide_axis: int = 3,
        threshold: int = 3
    ) -> None:
    """
    Runs the inference process using trained SVM models on an image and its corresponding depth map.

    Performs superpixel segmentation, extracts patches around segment centroids,
    processes these patches, extracts 2D and 3D features, runs inference using
    SVM models found in the models directory, and generates prediction masks
    and highlighted inference images.

    Args:
        image_path: Path to the input image file.
        depth_path: Path to the input depth map file.
        models_dir: Directory containing the trained SVM models.
        results_dir: Directory to save the inference results (masks and highlighted images).
        max_uxo_code: The maximum integer code representing UXO classes in multi-class models.
        num_components: Number of superpixels to generate using SLIC for inference.
        compactness: How compact the superpixels are; higher values make them more square.
        window_size: Size of the window around centroids for patch extraction.
        patch_size: Desired size of the extracted and resized image/depth patches.
        subdivide_axis: Number of subdivisions along each axis within the window for patch extraction.
        threshold: Minimum number of patch predictions required to consider a superpixel region as positive.
    """
    img = cv2.imread(image_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    img_label = os.path.splitext(os.path.basename(image_path))[0]

    h, w = img.shape[:2]
    window_radius = window_size // 2

    s = time.perf_counter()

    # Superpixel Segmentation
    labels, centroids = superpixel_segmentation(img, num_components=num_components, compactness=compactness)

    # Vectorized Coordinate Generation
    c_x = centroids[:, 1]
    c_y = centroids[:, 0]
    
    # Get Parent Window Bounds (Vectorized)
    # Note: Passed (0, w) logic via max_val=w
    p_x_s, p_x_e = get_window_bounds(c_x, window_radius, w) # Parent X Start/End
    p_y_s, p_y_e = get_window_bounds(c_y, window_radius, h) # Parent Y Start/End

    # Generate Subdivision Grid
    # We want 'subdivide_axis + 1' points between start and end.
    # We use broadcasting to create a grid for every centroid at once.
    # shape: (num_centroids, num_steps)
    steps = subdivide_axis + 1

    # Linear interpolation for all windows at once
    # Creates array of shape (steps, num_centroids) -> transpose to (num_centroids, steps)
    t = np.linspace(0, 1, steps)
    sub_x = (p_x_s[None, :] * (1 - t[:, None]) + p_x_e[None, :] * t[:, None]).astype(int).T
    sub_y = (p_y_s[None, :] * (1 - t[:, None]) + p_y_e[None, :] * t[:, None]).astype(int).T

    # Flatten the grids to get every combination of X and Y per centroid
    # Repeat X for every Y, and tile Y for every X
    # Resulting size: num_centroids * steps * steps
    sub_x_flat = np.repeat(sub_x, steps, axis=1).flatten()
    sub_y_flat = np.tile(sub_y, (1, steps)).flatten()

    # Also replicate the superpixel labels to match the expanded coordinates
    patch_labels = np.repeat(labels[c_y, c_x], steps * steps)

    # Calculate final patch crops around these grid points
    # (re-using the vectorized function)
    crop_x_s, crop_x_e = get_window_bounds(sub_x_flat, window_radius, w)
    crop_y_s, crop_y_e = get_window_bounds(sub_y_flat, window_radius, h)

    logger.info(f"Superpixel segmentation and patch coordinate calculation took {time.perf_counter() - s:.2f} seconds.")

    s = time.perf_counter()

    # Batch Extraction & Normalization
    num_patches = len(crop_x_s)

    batch_imgs = [
        cv2.resize(
            img[crop_y_s[i]:crop_y_e[i], crop_x_s[i]:crop_x_e[i]],
            (patch_size, patch_size), interpolation=cv2.INTER_AREA
        ) for i in range(num_patches)
    ]
    batch_depths = [
        cv2.resize(
            depth[crop_y_s[i]:crop_y_e[i], crop_x_s[i]:crop_x_e[i]],
            (patch_size, patch_size), interpolation=cv2.INTER_AREA
        ) for i in range(num_patches)
    ]

    logger.info(f"Patch extraction and resizing for {num_patches} patches took {time.perf_counter() - s:.2f} seconds.")

    features_2d, features_3d = extract_features(batch_imgs, batch_depths)
    model_files = os.listdir(models_dir)
    
    # Pre-calculate feature concatenations to avoid doing it inside the loop if possible
    feats_combined = np.concatenate((features_2d, features_3d), axis=1)
    for model_name in model_files:
        # Load Model
        model = SVMModel(model_dir=models_dir)
        model.load_model(model_name) # Assuming this is fast enough or unavoidable

        dimension = model.label
        binary_mode = model.is_binary()

        inference_dir = os.path.join(results_dir, os.path.splitext(model_name)[0])
        os.makedirs(inference_dir, exist_ok=True)

        # Select Features
        if dimension == '3':
            X = features_3d
        elif dimension == '2':
            X = features_2d
        else:
            X = feats_combined

        y_pred = model.predict(X)

        # We need to find which superpixel labels (patch_labels) have enough votes.
        uxo_mask = np.zeros_like(labels, dtype=np.float32)
        unique_preds = np.unique(y_pred)
        for pred_class in unique_preds:
            # Determine if this class is a UXO class we care about
            is_uxo_class = False
            mask_val = 0

            if binary_mode:
                if pred_class == 'uxo':
                    is_uxo_class = True
                    mask_val = 1
            else:
                # Multi-class check
                if pred_class.isdigit():
                    is_uxo_class = True
                    mask_val = int(pred_class) + 1

            if is_uxo_class:
                # Filter patch indices that predicted this class
                relevant_indices = (y_pred == pred_class)

                # Get the superpixel IDs corresponding to these positive predictions
                positive_superpixels = patch_labels[relevant_indices]
                if positive_superpixels.size > 0:
                    # Count occurrences of each superpixel ID
                    # sp_ids: unique superpixel IDs
                    # counts: how many times each ID appeared in the positive list
                    sp_ids, counts = np.unique(positive_superpixels, return_counts=True)

                    # Filter IDs that meet the threshold
                    valid_sp_ids = sp_ids[counts >= threshold]

                    # Update Mask (Vectorized using np.isin)
                    # Set mask pixels to mask_val where label is in valid_sp_ids
                    if valid_sp_ids.size > 0:
                        mask_update_locs = np.isin(labels, valid_sp_ids)
                        uxo_mask[mask_update_locs] = mask_val

        # Save Results
        cv2.imwrite(f"{inference_dir}/{img_label}_mask.png", uxo_mask.astype(np.uint8))

        # Highlight
        uxo_mask[uxo_mask == 0] = np.nan # Use nan for apply_mask to ignore background
        if not binary_mode:
            inference_img = apply_mask(img, uxo_mask, max_val=max_uxo_code, mode='highlight')
        else:
            inference_img = apply_mask(img, uxo_mask, mode='highlight')

        cv2.imwrite(f"{inference_dir}/{img_label}.jpg", inference_img)
