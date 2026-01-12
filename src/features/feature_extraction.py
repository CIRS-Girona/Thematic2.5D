import numpy as np
import time, datetime, logging
from threading import Thread
import cv2
import ctypes
from typing import List, Tuple, Optional

from ..utils import load_cpp_library

logger = logging.getLogger(__name__)

# Load the C++ library
LIB = load_cpp_library("libfastfeatures.so")

# --- Ctypes Definitions ---
POINTER_DOUBLE = ctypes.POINTER(ctypes.c_double)
NP_FLOAT64_C = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
NP_UINT8_C = np.ctypeslib.ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS')

# Define argtypes (Same as before, ensuring safety)
LIB.extract_color_features_c.argtypes = [NP_UINT8_C, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, POINTER_DOUBLE]
LIB.extract_lbp_features_c.argtypes = [NP_UINT8_C, ctypes.c_int, ctypes.c_int, ctypes.c_int, POINTER_DOUBLE]
LIB.extract_glcm_features_c.argtypes = [NP_UINT8_C, ctypes.c_int, ctypes.c_int, POINTER_DOUBLE]
LIB.extract_hog_features_c.argtypes = [NP_UINT8_C, ctypes.c_int, ctypes.c_int, ctypes.c_int, POINTER_DOUBLE]
LIB.extract_hog_features_depth_c.argtypes = [NP_FLOAT64_C, ctypes.c_int, ctypes.c_int, ctypes.c_int, POINTER_DOUBLE]
LIB.extract_principal_plane_features_c.argtypes = [NP_FLOAT64_C, ctypes.c_int, ctypes.c_int, ctypes.c_double, POINTER_DOUBLE]
LIB.extract_curvatures_c.argtypes = [NP_FLOAT64_C, ctypes.c_int, ctypes.c_int, ctypes.c_double, POINTER_DOUBLE]


# Wrapper Helper to reduce code duplication
def _call_cpp(func, data, out_size, *args):
    """Generic helper for calling feature extraction functions."""
    out = np.zeros(out_size, dtype=np.double)
    func(data, data.shape[0], data.shape[1], *args, out.ctypes.data_as(POINTER_DOUBLE))
    return out

# =================================================================
#  OPTIMIZED PYTHON WRAPPERS
# =================================================================
def extract_color_features(image: np.ndarray, bins: int = 8, range_val: Tuple[int, int] = (0, 256)) -> np.ndarray:
    image_c = np.ascontiguousarray(image, dtype=np.uint8)
    return _call_cpp(LIB.extract_color_features_c, image_c, bins, bins, range_val[0], range_val[1])

def extract_lbp_features(gray: np.ndarray, n_points: int = 24) -> np.ndarray:
    gray_c = np.ascontiguousarray(gray, dtype=np.uint8)
    return _call_cpp(LIB.extract_lbp_features_c, gray_c, n_points + 2, n_points)

def extract_glcm_features(image: np.ndarray) -> np.ndarray:
    img_c = np.ascontiguousarray(image, dtype=np.uint8)
    return _call_cpp(LIB.extract_glcm_features_c, img_c, 24)

def extract_hog_features(image: np.ndarray, n_bins: int = 9) -> np.ndarray:
    img_c = np.ascontiguousarray(image, dtype=np.uint8)
    return _call_cpp(LIB.extract_hog_features_c, img_c, n_bins + 4, n_bins)

def extract_symmetry_features(depth_patch: np.ndarray, n_bins: int = 9) -> np.ndarray:
    depth_c = np.ascontiguousarray(depth_patch, dtype=np.float64)
    return _call_cpp(LIB.extract_hog_features_depth_c, depth_c, n_bins + 4, n_bins)

def extract_principal_plane_features(depth_patch: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    depth_c = np.ascontiguousarray(depth_patch, dtype=np.float64)
    return _call_cpp(LIB.extract_principal_plane_features_c, depth_c, 16, eps)

def extract_curvatures_and_surface_normals(depth_patch: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    depth_c = np.ascontiguousarray(depth_patch, dtype=np.float64)
    return _call_cpp(LIB.extract_curvatures_c, depth_c, 16, eps)

def contrast_stretch(image: np.ndarray, dtype=np.uint8) -> np.ndarray:
    """Optimized using OpenCV minMaxLoc or Normalize if full range."""
    # Using numpy quantile is fine, but for speed on small patches, 
    # cv2.normalize with MINMAX is much faster if percentiles aren't strict.
    # Sticking to percentiles as requested, but optimized array ops.
    image_f = image.astype(np.float32)
    min_v = np.percentile(image_f, 1.5)
    max_v = np.percentile(image_f, 98.5)
    
    # Avoid division by zero
    div = max_v - min_v
    if div < 1e-6: div = 1.0
    
    # In-place/efficient scaling
    image_f -= min_v
    image_f *= (255.0 / div)
    np.clip(image_f, 0, 255, out=image_f)
    return image_f.astype(dtype)


def extract_features(images: List[np.ndarray], depths: Optional[List[np.ndarray]], clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> Tuple[np.ndarray, np.ndarray]:
    data_logger = lambda name, t, f: (
        logger.info(f"Started {name}"), f(), logger.info(f"Finished {name} in {time.perf_counter() - t:.3f}s")
    )

    s = time.perf_counter()

    # Precompute HSV and Gray
    # Using list comprehensions is fine, but ensure underlying data is contiguous to avoid copy later
    images_hsv = [contrast_stretch(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)) for img in images]

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    images_gray = [clahe.apply(img[:, :, 2]) for img in images_hsv] # V channel
    images_h_ch = [np.ascontiguousarray(img[:, :, 0]) for img in images_hsv] # H channel

    logger.info(f"Preprocessing: {time.perf_counter() - s:.3f}s")

    features: dict[str, List[np.ndarray]] = {}
    ts: List[Thread] = []

    # Helper for task parallelism
    def run_task(key, func, data):
        features[key] = [func(d) for d in data]

    tasks = [
        ('gray', extract_color_features, images_gray),
        ('hsv', extract_color_features, images_h_ch),
        ('lbp', extract_lbp_features, images_gray),
        ('glcm', extract_glcm_features, images_gray),
        ('hog', extract_hog_features, images_gray)
    ]

    if depths is not None:
        tasks.extend([
            ('principal plane', extract_principal_plane_features, depths),
            ('curvatures', extract_curvatures_and_surface_normals, depths),
            ('symmetry', extract_symmetry_features, depths)
        ])

    for key, func, data in tasks:
        t = Thread(target=data_logger, args=(f"{key} feats", time.perf_counter(), lambda k=key, f=func, d=data: run_task(k, f, d)))
        ts.append(t)
        t.start()

    for t in ts:
        t.join()

    # Concatenate features
    # Ensure order matches
    f2d_list = [features['gray'], features['hsv'], features['lbp'], features['glcm'], features['hog']]
    features_2d = np.concatenate(f2d_list, axis=1)

    features_3d = np.array([])
    if depths is not None:
        f3d_list = [features['principal plane'], features['curvatures'], features['symmetry']]
        features_3d = np.concatenate(f3d_list, axis=1)

    return np.nan_to_num(features_2d), np.nan_to_num(features_3d)