import numpy as np
import cv2
import os
import gc
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, List, Optional, Callable


ADJUST_COOR: Callable[[int, int, Tuple[int, int]], Tuple[int, int]] = lambda c, r, rnge: (0, 2 * r) if c - r < 0 else (rnge[1] - 1 - 2 * r, rnge[1] - 1) if c + r >= rnge[1] else (c - r, c + r)
"""
Adjusts a coordinate and radius to fit within a given range, ensuring the resulting
window stays within bounds.

Args:
    c (int): The center coordinate.
    r (int): The radius around the center.
    rnge (Tuple[int, int]): A tuple representing the valid range (start, end).

Returns:
    Tuple[int, int]: A tuple containing the adjusted start and end coordinates of the window.
"""


def process_data(
    image: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    indices: List[Tuple[int, int]],
    dataset_dir: str,
    prefix: str,
    uxo_threshold: float,
    invalid_threshold: float,
    window_size: int,
    patch_size: int,
    angles: Tuple[int]
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
    print("Started thread")

    h, w = mask.shape
    for i, (c_y, c_x) in enumerate(indices):
        t, m, d = None, None, None
        gc.collect()

        radius = window_size // 2

        x_s, x_e = ADJUST_COOR(c_x, radius, (0, w))
        y_s, y_e = ADJUST_COOR(c_y, radius, (0, h))

        t = image[y_s:y_e, x_s:x_e, :]
        m = mask[y_s:y_e, x_s:x_e]
        d = depth[y_s:y_e, x_s:x_e]

        if np.sum(m == None) / m.size > invalid_threshold:
            continue # Skip patch if too many invalid pixels

        if t is not None and d is not None:
            t_resized = cv2.resize(t, (patch_size, patch_size), interpolation=cv2.INTER_AREA)
            d_resized = cv2.resize(d, (patch_size, patch_size), interpolation=cv2.INTER_AREA).astype(np.float32)

            # Normalize depth map to 0-255 range
            d_resized = 255 * (d_resized - np.min(d_resized)) / (np.max(d_resized) - np.min(d_resized))
            d_resized = d_resized.astype(np.uint8)

            if np.sum(m > 0) / m.size >= uxo_threshold:
                uxo_number = int(np.unique(m[m > 0])[-1])

                uxo_2d_dir = os.path.join(dataset_dir, "2D", str(uxo_number))
                uxo_3d_dir = os.path.join(dataset_dir, "3D", str(uxo_number))
                os.makedirs(uxo_2d_dir, exist_ok=True)
                os.makedirs(uxo_3d_dir, exist_ok=True)

                # Apply rotations and save augmented UXO patches
                h_img, w_img = d_resized.shape # Use shape of normalized depth for rotation center
                for angle in angles:
                    M = cv2.getRotationMatrix2D((w_img // 2, h_img // 2), angle, 1.0)  # Center, rotation angle, scale
                    t_rot = cv2.warpAffine(t_resized, M, (w_img, h_img))
                    d_rot = cv2.warpAffine(d_resized, M, (w_img, h_img))

                    cv2.imwrite(os.path.join(uxo_2d_dir, f"{prefix}-{i}_{angle}.png"), t_rot)
                    cv2.imwrite(os.path.join(uxo_3d_dir, f"{prefix}-{i}_{angle}.png"), d_rot)

                    del t_rot, d_rot
                    gc.collect()
            elif np.all(m == 0):
                # Save as background patch
                bg_2d_dir = os.path.join(dataset_dir, "2D", "background")
                bg_3d_dir = os.path.join(dataset_dir, "3D", "background")
                os.makedirs(bg_2d_dir, exist_ok=True)
                os.makedirs(bg_3d_dir, exist_ok=True)

                cv2.imwrite(os.path.join(bg_2d_dir, f"{prefix}-{i}.png"), t_resized)
                cv2.imwrite(os.path.join(bg_3d_dir, f"{prefix}-{i}.png"), d_resized)

            del t_resized, d_resized
            gc.collect()

    print(f"Finished thread for prefix: {prefix}")


def create_dataset(
    images_path: str,
    depths_path: str,
    masks_path: str,
    dataset_dir: str,
    uxo_start_code: int,
    invalid_code: int,
    prefix: str = '',
    bg_per_img: int = 20_000,
    thread_count: int = 64,
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
        uxo_start_code (int): The minimum mask value that indicates a UXO instance.
                               Mask values less than this (but not invalid_code) are
                               considered background (set to 0).
        invalid_code (int): The mask value that indicates an invalid pixel. These
                            pixels are ignored during processing.
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
    print(f"Started processing dataset {prefix}")

    mask_files = os.listdir(masks_path)
    labels = ['.'.join(f.split('.')[:-1]) for f in mask_files if os.path.isfile(os.path.join(masks_path, f))]

    with ThreadPoolExecutor(max_workers=thread_count) as exe:
        for label in labels:
            image_path_jpg = os.path.join(images_path, f"{label}.jpg")
            image_path_png = os.path.join(images_path, f"{label}.png")

            image: Optional[np.ndarray] = None
            if os.path.exists(image_path_jpg) and os.path.isfile(image_path_jpg):
                image = cv2.imread(image_path_jpg, cv2.IMREAD_UNCHANGED)
            elif os.path.exists(image_path_png) and os.path.isfile(image_path_png):
                image = cv2.imread(image_path_png, cv2.IMREAD_UNCHANGED)
            else:
                print(f"Warning: Image file not found for label {label} in {images_path}. Skipping.")
                continue

            depth = cv2.imread(os.path.join(depths_path, f"{label}.png"), cv2.IMREAD_UNCHANGED)
            mask = cv2.imread(os.path.join(masks_path, f"{label}.png"), cv2.IMREAD_UNCHANGED)

            if image is None or depth is None or mask is None:
                print(f"Warning: Couldn't read image file, depth map or mask file for label {label}. Skipping.")
                continue

            mask = mask.astype(np.float32)
            mask[mask == invalid_code] = None
            mask[depth == 0] = None
            mask[mask < uxo_start_code] = 0

            uxo_indices = np.where(mask > 0)
            uxo_indices_list = list(zip(uxo_indices[0], uxo_indices[1])) # Convert to list of (y, x) tuples

            uxo_indices_sampled = random.sample(uxo_indices_list, int(len(uxo_indices_list) * uxo_sample_rate))

            # Background indices are where mask is 0 and not NaN
            bg_indices = np.where(mask == 0)
            bg_indices_list = list(zip(bg_indices[0], bg_indices[1])) # Convert to list of (y, x) tuples

            bg_indices_sampled = random.sample(bg_indices_list, min(bg_per_img, len(bg_indices_list)))            

            # Submit the processing task for this image to the thread pool
            exe.submit(
                process_data,
                image,
                depth,
                mask,
                uxo_indices_sampled + bg_indices_sampled,
                dataset_dir,
                f"{label}-{prefix}", # Use label and prefix for unique filename prefix
                uxo_threshold,
                invalid_threshold,
                window_size,
                patch_size,
                angles
            )

            # Delete image, mask, and depth after submitting to free up memory
            del uxo_indices, uxo_indices_list, uxo_indices_sampled, bg_indices, bg_indices_list, bg_indices_sampled
            del image, mask, depth
            gc.collect()
