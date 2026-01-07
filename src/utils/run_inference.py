import numpy as np
import cv2
import os
from time import perf_counter

from .feature_extraction import extract_features
from .image_processing import process_images, superpixel_segmentation, apply_mask
from .dataset_creator import ADJUST_COOR
from ..classification import SVMModel


def run_inference(
        image_path: str,
        depth_path: str,
        models_dir: str,
        results_dir: str,
        uxo_start_code: int,
        max_uxo_code: int,
        region_size: int = 400,
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
        uxo_start_code: The starting integer code representing UXO classes in multi-class models.
        max_uxo_code: The maximum integer code representing UXO classes in multi-class models.
        region_size: Parameter for superpixel segmentation (ruler).
        window_size: Size of the window around centroids for patch extraction.
        patch_size: Desired size of the extracted and resized image/depth patches.
        subdivide_axis: Number of subdivisions along each axis within the window for patch extraction.
        threshold: Minimum number of patch predictions required to consider a superpixel region as positive.
    """
    img = cv2.imread(image_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    img_label = '.'.join(image_path.split('/')[-1].split('.')[:-1])

    # TODO: Adjust num_components and compactness as needed using config parameters
    s = perf_counter()
    labels, centroids = superpixel_segmentation(img)
    print(f"Superpixel segmentation completed in {perf_counter() - s:.2f} seconds.")

    # Loop over each centroid
    s = perf_counter()

    imgs = []
    depths = []
    patches = []
    for c_x, c_y in centroids:
        window_radius = window_size // 2

        # Calculate patch boundaries
        x_start, x_end = ADJUST_COOR(c_x, window_radius, (0, img.shape[1]))
        y_start, y_end = ADJUST_COOR(c_y, window_radius, (0, img.shape[0]))

        # Create coordinate arrays for subdivisions
        x_coords = np.linspace(x_start, x_end, subdivide_axis + 1, endpoint=True, dtype=int)
        y_coords = np.linspace(y_start, y_end, subdivide_axis + 1, endpoint=True, dtype=int)

        # Calculate steps between subdivision boundaries
        for x_step in x_coords:
            for y_step in y_coords:
                x_sub_start, x_sub_end = ADJUST_COOR(x_step, window_radius, (0, img.shape[1]))
                y_sub_start, y_sub_end = ADJUST_COOR(y_step, window_radius, (0, img.shape[0]))

                # Extract and resize image patch
                img_patch = cv2.resize(img[x_sub_start:x_sub_end, y_sub_start:y_sub_end, :], (patch_size, patch_size), interpolation=cv2.INTER_AREA)
                imgs.append(img_patch)

                # Extract, resize, and normalize depth patch
                depth_patch = cv2.resize(depth[x_sub_start:x_sub_end, y_sub_start:y_sub_end], (patch_size, patch_size), interpolation=cv2.INTER_AREA)
                depth_patch = depth_patch.astype(np.double)
                depth_patch -= np.min(depth_patch)
                depth_patch /= max(np.max(depth_patch), 1)
                depth_patch = np.nan_to_num(255 * depth_patch).astype(np.uint8).astype(np.double)
                depths.append(depth_patch)

                # Store corresponding label
                patches.append(labels[c_x, c_y])

    patches = np.array(patches)
    print(f"Patch extraction completed in {perf_counter() - s:.2f} seconds.")

    s = perf_counter()
    features_2d, features_3d = extract_features(imgs, depths)
    print(f"Feature extraction completed in {perf_counter() - s:.2f} seconds.")

    s = perf_counter()
    models = os.listdir(models_dir)
    for model_name in models:
        model = SVMModel(model_dir=models_dir)
        model.load_model(model_name)

        dimension = model.label
        binary_mode = len(model.model.classes_) == 2

        inference_dir = f"{results_dir}/{'.'.join(model_name.split('.')[:-1])}"
        os.makedirs(inference_dir, exist_ok=True)

        if dimension == '3':
            features = features_3d
        elif dimension == '2':
            features = features_2d
        else:
            features = np.concatenate((features_2d, features_3d), axis=1)

        y_pred = model.predict(features)

        uxo_mask = np.zeros_like(labels, dtype=np.float32)
        for y in np.unique(y_pred):
            regions = patches[y_pred == y]

            if (binary_mode and y == 'uxo') or (not binary_mode and y.isdigit() and int(y) >= uxo_start_code):
                for region in np.unique(regions):
                    if regions[regions == region].size < threshold:
                        uxo_mask[labels == region] = 0
                    else:
                        uxo_mask[labels == region] = int(y) if not binary_mode else 1

        cv2.imwrite(f"{inference_dir}/{img_label}_mask.png", uxo_mask.astype(np.uint8))

        uxo_mask[uxo_mask == 0] = None

        if not binary_mode:
            inference = apply_mask(img, uxo_mask, min_val=uxo_start_code, max_val=max_uxo_code, mode='highlight')
        else:
            inference = apply_mask(img, uxo_mask, mode='highlight')

        cv2.imwrite(f"{inference_dir}/{img_label}.jpg", inference)
    print(f"Inference with model {model_name} completed in {perf_counter() - s:.2f} seconds.")