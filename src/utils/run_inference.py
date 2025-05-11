import numpy as np
import cv2, os

from utils import ADJUST_COOR, extract_features, process_images, superpixel_segmentation, apply_mask
from classification import SVMModel


def run_inference(image_path, depth_path, models_dir, results_dir, uxo_start_code, max_uxo_code, region_size=400, window_size=400, patch_size=128, subdivide_axis=3, threshold=3):
    img = cv2.imread(image_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    img_label = '.'.join(image_path.split('/')[-1].split('.')[:-1])

    print("Applying Superpixel Segmentation")
    labels, centroids = superpixel_segmentation(img, ruler=1, region_size=region_size)

    # Loop over each centroid
    imgs = []
    depths = []
    patches = []
    for c_y, c_x in centroids:
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

    print("Processing patches")

    patches = np.array(patches)
    gray_images, hsv_images = process_images(imgs)
    features_2d, features_3d = extract_features(gray_images, hsv_images, depths)

    models = os.listdir(models_dir)
    for model_name in models:
        model = SVMModel(model_dir=models_dir)
        model.load_model(model_name)
        
        dimension = model.label
        binary_mode = len(model.model.classes_) == 2

        if binary_mode:
            print(f"Running binary classification ({dimension}D) on:\n{image_path}\n")
        else:
            print(f"Running multi-class classification ({dimension}D) on:\n{image_path}\n")

        inference_dir = f"{results_dir}/{'.'.join(model_name.split('.')[:-1])}"
        if not os.path.exists(inference_dir) or not os.path.isdir(inference_dir):
            os.makedirs(inference_dir)

        if dimension == '3':
            features = features_3d
        elif dimension == '2':
            features = features_2d
        else:
            features = np.concatenate((features_2d, features_3d), axis=1)

        print("Running inference")

        y_pred = model.evaluate(features)

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

        cv2.imwrite(f"{inference_dir}/{img_label}.png", inference)