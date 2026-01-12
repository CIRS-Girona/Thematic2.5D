import numpy as np
import os, cv2, csv, logging
from typing import List, Tuple
from time import perf_counter

from ..features import extract_features
from ..utils import Sensor, get_window_bounds
from ..classification import SVMModel
from . import calculate_ground_resolution, calculate_slant, calculate_UCIQE, calculate_UIQM

logger = logging.getLogger(__name__)

FIELDS: Tuple[str] = [
    "label",
    "image",
    "camera",
    "res. width",
    "res. height",
    "visibility",
    "centroid u",
    "centroid v",
    "med. depth (mm)",
    "avg. depth (mm)",
    "area (pixels)",
    "ground res. (mm / pixel)",
    "camera slant (degrees)",
    "UIQM",
    "UCIQUE",
    "2D binary",
    "3D binary",
    "25D binary",
    "2D",
    "3D",
    "25D",
]


def calculate_and_save_metrics(
    sensor: Sensor,
    models_dir: str,
    imgs_path: List[str],
    masks_path: List[str],
    depths_path: List[str],
    camera_type: str,
    visibility: float,
    output_file: str,
    window_size: int = 400,
    patch_size: int = 128,
):
    model_files = os.listdir(models_dir)

    s = perf_counter()
    logger.info("Starting metric calculations...")

    data = []
    for img_path, mask_path, depth_path in zip(imgs_path, masks_path, depths_path):
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        slant = calculate_slant(sensor, depth)
        uiqm = calculate_UIQM(img)
        ucique = calculate_UCIQE(img)

        uxos = {tuple(c) for c in mask[mask[:, :, 0] == 2]}  # Get unique UXO instances
        for uxo in uxos:
            indices = np.logical_and(np.all(mask == uxo, axis=2), depth > 0)
            v, u = np.where(indices)

            y_s, y_e = get_window_bounds(np.array((np.median(v),)), window_size // 2, depth.shape[0])
            x_s, x_e = get_window_bounds(np.array((np.median(u),)), window_size // 2, depth.shape[1])

            patch_img = cv2.resize(
                img[y_s[0]:y_e[0], x_s[0]:x_e[0]],
                (patch_size, patch_size), interpolation=cv2.INTER_NEAREST
            )
            patch_depth = cv2.resize(
                depth[y_s[0]:y_e[0], x_s[0]:x_e[0]],
                (patch_size, patch_size), interpolation=cv2.INTER_NEAREST
            )

            features_2d, features_3d = extract_features([patch_img], [patch_depth])
            feats_combined = np.concatenate((features_2d, features_3d), axis=1)

            model_results = {}
            for model_name in model_files:
                # Load Model
                model = SVMModel(model_dir=models_dir)
                model.load_model(model_name)

                dimension = model.label
                binary_mode = model.is_binary()

                # Select Features
                if dimension == '3':
                    X = features_3d
                elif dimension == '2':
                    X = features_2d
                else:
                    X = feats_combined

                y_pred = model.predict(X)

                if binary_mode:
                    model_results[f"{dimension}D binary"] = int(y_pred[0] == 'uxo')
                else:
                    model_results[f"{dimension}D"] = int(y_pred[0] == str(uxo[1]))

            data.append({
                "label": f"{uxo[1]}_{uxo[2]}",
                "image": ''.join(img_path.split('/')[-1].split('.')[:-1]),
                "camera": camera_type,
                "res. width": depth.shape[1],
                "res. height": depth.shape[0],
                "visibility": visibility,
                "centroid u": np.median(u),
                "centroid v": np.median(v),
                "med. depth (mm)": np.median(depth[v, u]),
                "avg. depth (mm)": np.mean(depth[v, u]),
                "area (pixels)": u.size,
                "ground res. (mm / pixel)": calculate_ground_resolution(sensor, u, v, depth[v, u]),
                "camera slant (degrees)": slant,
                "UIQM": uiqm,
                "UCIQUE": ucique
            } | model_results)

    with open(output_file, 'w') as f:
        writer = csv.DictWriter(f, FIELDS)
        writer.writeheader()
        writer.writerows(data)

    logger.info(f"Metric calculations completed in {perf_counter() - s:.2f} seconds. Results saved to {output_file}.")
