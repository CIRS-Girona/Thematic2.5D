import numpy as np
import cv2
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple
from tqdm import tqdm


def process_image_data(
    image: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    indices: Tuple[np.ndarray, np.ndarray], # Passed as Arrays (y_coords, x_coords)
    dataset_dir: str,
    prefix: str,
    uxo_threshold: float,
    invalid_threshold: float,
    window_size: int,
    patch_size: int,
    angles: Tuple[int],
    is_uxo_batch: bool
) -> None:
    """
    Processes image, depth, and mask data for a list of specified indices (patch centers).
    Extracts patches, resizes them, normalizes depth, and saves them as images
    categorized by whether they contain a UXO based on the mask and thresholds.

    Args:
        image (np.ndarray): The input image (BGR format).
        depth (np.ndarray): The input depth map (single channel, double precision).
        mask (np.ndarray): The segmentation mask (float32), where non-positive values
                           are considered background/invalid, and positive values
                           represent different UXO instances or background.
        indices (List[Tuple[int, int]]): A list of (y, x) coordinates representing the
                                         centers of the patches to extract.
        dataset_dir (str): The base directory where the processed dataset will be saved.
                           Subdirectories for 2D/3D and classes/background will be created here.
        prefix (str): A prefix to add to the saved image filenames.
        uxo_threshold (float): The minimum proportion of non-zero pixels within a patch's mask
                               required for the patch to be considered a UXO patch.
        invalid_threshold (float): The maximum proportion of 'None' (invalid) pixels allowed
                                   within a patch's mask. Patches exceeding this are skipped.
        window_size (int): The size of the square window to extract around each index.
        patch_size (int): The target size (width and height) to which extracted patches
                          will be resized.
        angles (Tuple[int]): A tuple of angles (in degrees) for rotating UXO patches
                                  to create augmented samples.
    """
    h_img_map, w_img_map = mask.shape
    radius = window_size // 2
    y_coords, x_coords = indices
    
    # Pre-calculate directory paths to avoid doing it in the loop
    if is_uxo_batch:
        # For UXO, we must check the specific class per patch, so dirs are resolved in loop
        pass 
    else:
        # Background dirs are constant
        bg_2d_dir = os.path.join(dataset_dir, "2D", "background")
        bg_3d_dir = os.path.join(dataset_dir, "3D", "background")
        os.makedirs(bg_2d_dir, exist_ok=True)
        os.makedirs(bg_3d_dir, exist_ok=True)

    for _, (c_y, c_x) in enumerate(zip(y_coords, x_coords)):
        # Y-bounds
        if c_y - radius < 0: y_s, y_e = 0, 2 * radius
        elif c_y + radius >= h_img_map: y_s, y_e = h_img_map - 1 - 2 * radius, h_img_map - 1
        else: y_s, y_e = c_y - radius, c_y + radius

        # X-bounds
        if c_x - radius < 0: x_s, x_e = 0, 2 * radius
        elif c_x + radius >= w_img_map: x_s, x_e = w_img_map - 1 - 2 * radius, w_img_map - 1
        else: x_s, x_e = c_x - radius, c_x + radius
        
        # Slicing (NumPy views, very fast)
        m_patch = mask[y_s:y_e, x_s:x_e]
        
        # Fast Invalid Check (using NaN for invalid pixels)
        # Check percentage of NaNs
        if np.isnan(m_patch).mean() > invalid_threshold:
            continue

        # Extract Data
        t_patch = image[y_s:y_e, x_s:x_e]
        d_patch = depth[y_s:y_e, x_s:x_e]

        # Resize
        t_resized = cv2.resize(t_patch, (patch_size, patch_size), interpolation=cv2.INTER_AREA)
        d_resized = cv2.resize(d_patch, (patch_size, patch_size), interpolation=cv2.INTER_AREA)

        # Classification and Saving
        if is_uxo_batch:
            # Check if it actually meets the UXO threshold density
            # We ignore NaNs in this check by comparing > 0 (NaN comparison always False)
            valid_mask_pixels = m_patch[~np.isnan(m_patch)]
            
            if valid_mask_pixels.size > 0 and (valid_mask_pixels > 0).mean() >= uxo_threshold:
                # Get class ID (take the max or last unique to determine class)
                uxo_number = int(np.max(valid_mask_pixels)) 
                
                uxo_2d_dir = os.path.join(dataset_dir, "2D", str(uxo_number))
                uxo_3d_dir = os.path.join(dataset_dir, "3D", str(uxo_number))
                os.makedirs(uxo_2d_dir, exist_ok=True)
                os.makedirs(uxo_3d_dir, exist_ok=True)

                h_img, w_img = d_resized.shape
                center = (w_img // 2, h_img // 2)
                
                for angle in angles:
                    if angle == 0:
                        t_rot, d_rot = t_resized, d_resized
                    else:
                        M = cv2.getRotationMatrix2D(center, angle, 1.0)
                        t_rot = cv2.warpAffine(t_resized, M, (w_img, h_img))
                        d_rot = cv2.warpAffine(d_resized, M, (w_img, h_img))

                    # Use an f-string with a unique ID per original coord to avoid collisions
                    # Using c_y and c_x ensures uniqueness better than loop index 'i' across threads if needed
                    fname = f"{prefix}-{c_y}_{c_x}_{angle}.png"
                    cv2.imwrite(os.path.join(uxo_2d_dir, fname), t_rot)
                    cv2.imwrite(os.path.join(uxo_3d_dir, fname), d_rot)

        else:
            # Background Case
            # Double check it is empty (no UXO pixels)
            if np.nanmax(m_patch) <= 0:
                fname = f"{prefix}-{c_y}_{c_x}.png"
                cv2.imwrite(os.path.join(bg_2d_dir, fname), t_resized)
                cv2.imwrite(os.path.join(bg_3d_dir, fname), d_resized)


def create_dataset(
    images_path: str,
    depths_path: str,
    masks_path: str,
    dataset_dir: str,
    prefix: str = '',
    bg_per_img: int = 20_000,
    thread_count: int = 16,
    uxo_sample_rate: float = 0.01,
    uxo_threshold: float = 0.4,
    invalid_threshold: float = 0.01,
    window_size: int = 400,
    patch_size: int = 128,
    angles: Tuple[int] = (0, 90, 180, 270)
) -> None:
    """
    Creates a dataset of image and depth patches extracted from larger images,
    categorized into UXO and background classes based on provided masks.
    Uses multithreading to process images in parallel.

    Args:
        images_path (str): Path to the directory containing input images.
        depths_path (str): Path to the directory containing input depth maps.
        masks_path (str): Path to the directory containing segmentation masks.
        dataset_dir (str): The base directory where the created dataset will be saved.
        prefix (str): A prefix to add to the filenames within the dataset directory.
        bg_per_img (int): The number of background patches to sample per image.
        thread_count (int): The maximum number of worker threads to use for processing.
        uxo_sample_rate (float): The proportion of UXO pixels to sample as patch centers
                                 from the total number of UXO pixels in an image.
        uxo_threshold (float): The minimum proportion of non-zero pixels within a patch's mask
                               required for the patch to be considered a UXO patch.
        invalid_threshold (float): The maximum proportion of 'None' (invalid) pixels allowed
                                   within a patch's mask. Patches exceeding this are skipped.
        window_size (int): The size of the square window to extract around each patch center.
        patch_size (int): The target size (width and height) to which extracted patches
                          will be resized.
        angles (Tuple[int]): A tuple of angles (in degrees) for rotating UXO patches
                             to create augmented samples.
    """
    def _create_dataset(label: str) -> None:
        # Load Data
        img_path = os.path.join(images_path, f"{label}.jpg")
        if not os.path.exists(img_path): img_path = os.path.join(images_path, f"{label}.png")
        if not os.path.exists(img_path): return

        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        depth = cv2.imread(os.path.join(depths_path, f"{label}.png"), cv2.IMREAD_UNCHANGED)
        mask_raw = cv2.imread(os.path.join(masks_path, f"{label}.png"), cv2.IMREAD_UNCHANGED)

        if image is None or depth is None or mask_raw is None: return

        h, w = depth.shape

        # Convert to float for NaN support.
        mask = mask_raw.astype(np.float32)

        # Identify Invalid Pixels (Invalid Code or 0-Depth) and UXO Pixels
        is_invalid = (mask_raw[:, :, 0] == 0) | (depth == 0)
        is_uxo = (mask_raw[:, :, 0] == 2) & (~is_invalid)
        is_bg = (mask_raw[:, :, 0] == 1) & (~is_invalid)

        # We need specific locations, so finding them all is necessary.
        uxo_ys, uxo_xs = np.where(is_uxo)
        total_uxos = len(uxo_ys)
        if total_uxos > 0:
            sample_size = int(total_uxos * uxo_sample_rate)
            # Use random choice on indices, simpler than zipping and sampling
            if sample_size > 0:
                idx = np.random.choice(total_uxos, sample_size, replace=False)
                process_image_data(
                    image, depth, mask[:, :, 1], (uxo_ys[idx], uxo_xs[idx]),
                    dataset_dir, f"{label}-{prefix}", uxo_threshold, invalid_threshold,
                    window_size, patch_size, angles, is_uxo_batch=True
                )

        # Generate 20% more than needed to account for invalid hits
        attempt_count = int(bg_per_img * 1.2) 
        rand_y = np.random.randint(0, h, attempt_count)
        rand_x = np.random.randint(0, w, attempt_count)

        # Filter the coordinates
        valid_bg_indices = is_bg[rand_y, rand_x]
        bg_y = rand_y[valid_bg_indices]
        bg_x = rand_x[valid_bg_indices]

        # Trim to exact number required
        if len(bg_y) > bg_per_img:
            bg_y = bg_y[:bg_per_img]
            bg_x = bg_x[:bg_per_img]

        if len(bg_y) > 0:
            process_image_data(
                image, depth, mask[:, :, 1], (bg_y, bg_x),
                dataset_dir, f"{label}-{prefix}", uxo_threshold, invalid_threshold,
                window_size, patch_size, angles, is_uxo_batch=False
            )

    # Main Execution
    mask_files = os.listdir(masks_path)
    labels = ['.'.join(f.split('.')[:-1]) for f in mask_files if f.endswith(('.png',))]

    with ThreadPoolExecutor(max_workers=thread_count) as exe:
        list(tqdm(exe.map(_create_dataset, labels), total=len(labels)))
