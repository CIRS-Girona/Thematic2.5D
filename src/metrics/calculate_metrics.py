import numpy as np
import cv2, csv, logging
from typing import List, Tuple
from time import perf_counter

from ..utils import Sensor
from .metrics import calculate_ground_resolution, calculate_slant, calculate_UCIQE, calculate_UIQM

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
]


def calculate_and_save_metrics(
    sensor: Sensor,
    imgs_path: List[str],
    masks_path: List[str],
    depths_path: List[str],
    output_path: str,
    camera_type: str,
    visibility: float
):
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

        colors = {tuple(c) for c in mask[np.any(mask != (2, 0, 0), axis=2)]}  # Get unique colors
        for color in colors:
            indices = np.logical_and(np.all(mask == color, axis=2), depth > 0)
            v, u = np.where(indices)

            data.append({
                "label": color[2],
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
            })

    if output_path.split('.')[-1].lower() != 'csv':
        output_path += '.csv'

    with open(output_path, 'w') as f:
        writer = csv.DictWriter(f, FIELDS)
        writer.writeheader()
        writer.writerows(data)

    logger.info(f"Metric calculations completed in {perf_counter() - s:.2f} seconds. Results saved to {output_path}.")
