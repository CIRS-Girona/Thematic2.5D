import numpy as np
import time, datetime, logging
from threading import Thread
import cv2
import ctypes
from typing import List, Tuple

from ..utils import load_cpp_library

logger = logging.getLogger(__name__)

# Load the C++ library
LIB = load_cpp_library("libfastfeatures.so")

# --- Ctypes Definitions ---
POINTER_DOUBLE = ctypes.POINTER(ctypes.c_double)
NP_FLOAT64_C = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
NP_UINT8_C = np.ctypeslib.ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS')

# extract_color_features_c
LIB.extract_color_features_c.restype = None
LIB.extract_color_features_c.argtypes = [
    NP_UINT8_C,      # img
    ctypes.c_int,    # rows
    ctypes.c_int,    # cols
    ctypes.c_int,    # bins
    ctypes.c_int,    # range_min
    ctypes.c_int,    # range_max
    POINTER_DOUBLE   # out_features
]

# extract_lbp_features_c
LIB.extract_lbp_features_c.restype = None
LIB.extract_lbp_features_c.argtypes = [
    NP_UINT8_C,      # gray
    ctypes.c_int,    # rows
    ctypes.c_int,    # cols
    ctypes.c_int,    # n_points
    POINTER_DOUBLE   # out_features
]

# extract_glcm_features_c
LIB.extract_glcm_features_c.restype = None
LIB.extract_glcm_features_c.argtypes = [
    NP_UINT8_C,      # img
    ctypes.c_int,    # rows
    ctypes.c_int,    # cols
    POINTER_DOUBLE   # out_features
]

# extract_hog_features_c
LIB.extract_hog_features_c.restype = None
LIB.extract_hog_features_c.argtypes = [
    NP_UINT8_C,      # img
    ctypes.c_int,    # rows
    ctypes.c_int,    # cols
    ctypes.c_int,    # n_bins
    POINTER_DOUBLE   # out_features
]

# extract_hog_features_depth_c
LIB.extract_hog_features_depth_c.restype = None
LIB.extract_hog_features_depth_c.argtypes = [
    NP_FLOAT64_C,    # depth map (double)
    ctypes.c_int,    # rows
    ctypes.c_int,    # cols
    ctypes.c_int,    # n_bins
    POINTER_DOUBLE   # out_features
]

# extract_principal_plane_features_c
LIB.extract_principal_plane_features_c.restype = None
LIB.extract_principal_plane_features_c.argtypes = [
    NP_FLOAT64_C,    # depth
    ctypes.c_int,    # rows
    ctypes.c_int,    # cols
    ctypes.c_double, # eps
    POINTER_DOUBLE   # out_features
]

# extract_curvatures_c
LIB.extract_curvatures_c.restype = None
LIB.extract_curvatures_c.argtypes = [
    NP_FLOAT64_C,    # depth
    ctypes.c_int,    # rows
    ctypes.c_int,    # cols
    ctypes.c_double, # eps
    POINTER_DOUBLE   # out_features
]

# =================================================================
#  PYTHON WRAPPER FUNCTIONS
# =================================================================
def extract_color_features(image: np.ndarray, bins: int = 8, range_val: Tuple[int, int] = (0, 256)) -> np.ndarray:
    """Extracts color histogram using C++ backend."""
    img_c = np.ascontiguousarray(image, dtype=np.uint8)
    rows, cols = img_c.shape[:2]

    out_features = np.zeros(bins, dtype=np.double)
    out_ptr = out_features.ctypes.data_as(POINTER_DOUBLE)
    LIB.extract_color_features_c(
        img_c, rows, cols, bins, 
        range_val[0], range_val[1], out_ptr
    )

    return out_features


def extract_lbp_features(gray: np.ndarray, n_points: int = 24) -> np.ndarray:
    """Extracts LBP histogram using C++ backend."""
    gray_c = np.ascontiguousarray(gray, dtype=np.uint8)
    rows, cols = gray_c.shape

    n_bins = n_points + 2
    out_features = np.zeros(n_bins, dtype=np.double)
    out_ptr = out_features.ctypes.data_as(POINTER_DOUBLE)
    LIB.extract_lbp_features_c(gray_c, rows, cols, n_points, out_ptr)

    # Matching Python signature (re-call color features? No, just return histogram)
    return out_features


def extract_glcm_features(image: np.ndarray) -> np.ndarray:
    """Extracts GLCM features using C++ backend."""
    img_c = np.ascontiguousarray(image, dtype=np.uint8)
    rows, cols = img_c.shape

    # 4 angles * 6 features = 24 floats
    out_features = np.zeros(24, dtype=np.double)
    out_ptr = out_features.ctypes.data_as(POINTER_DOUBLE)
    LIB.extract_glcm_features_c(img_c, rows, cols, out_ptr)

    # Reshape to match original Python: (4, 6)
    return out_features


def extract_hog_features(image: np.ndarray, n_bins: int = 9) -> np.ndarray:
    """
    Extracts HOG features using C++ backend.
    Returns 9 Histogram bins + 4 Magnitude stats (Mean, Std, Skew, Kurt).
    Total features: 13
    """
    img_c = np.ascontiguousarray(image, dtype=np.uint8)
    rows, cols = img_c.shape

    # Allocate output: bins + 4 stats
    out_features = np.zeros(n_bins + 4, dtype=np.double)
    out_ptr = out_features.ctypes.data_as(POINTER_DOUBLE)
    
    LIB.extract_hog_features_c(img_c, rows, cols, n_bins, out_ptr)

    return out_features


def extract_symmetry_features(depth_patch: np.ndarray, n_bins: int = 9) -> np.ndarray:
    """
    Extracts HOG-based symmetry features directly from floating point depth maps.
    Does NOT cast to uint8 to preserve depth gradients.
    """
    # Ensure contiguous float64 array
    depth_c = np.ascontiguousarray(depth_patch, dtype=np.float64)
    rows, cols = depth_c.shape

    # Allocate output: bins + 4 stats
    out_features = np.zeros(n_bins + 4, dtype=np.double)
    out_ptr = out_features.ctypes.data_as(POINTER_DOUBLE)
    
    # Call the NEW C function
    LIB.extract_hog_features_depth_c(depth_c, rows, cols, n_bins, out_ptr)

    return out_features


def extract_principal_plane_features(depth_patch: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Extracts 3D plane features using C++ backend."""
    depth_c = np.ascontiguousarray(depth_patch, dtype=np.float64)
    rows, cols = depth_c.shape

    # 3 (z stats) + 9 (coeffs) + 4 (geom) = 16 features
    out_features = np.zeros(16, dtype=np.double)
    out_ptr = out_features.ctypes.data_as(POINTER_DOUBLE)
    LIB.extract_principal_plane_features_c(depth_c, rows, cols, eps, out_ptr)

    return out_features


def extract_curvatures_and_surface_normals(depth_patch: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Extracts 3D curvature features using C++ backend."""
    depth_c = np.ascontiguousarray(depth_patch, dtype=np.float64)
    rows, cols = depth_c.shape

    # 8 properties * 2 stats (mean, std) = 16 features
    out_features = np.zeros(16, dtype=np.double)
    out_ptr = out_features.ctypes.data_as(POINTER_DOUBLE)
    LIB.extract_curvatures_c(depth_c, rows, cols, eps, out_ptr)

    return out_features


def contrast_stretch(image: np.ndarray, dtype = np.uint8) -> np.ndarray:
    """
    Applies contrast stretching to the input image using percentile values.

    Stretches the intensity range of each channel based on the 1.5th and 98.5th percentiles
    to improve visibility.

    Args:
        image (np.ndarray): The input image (NumPy array). Expected to be BGR or grayscale.

    Returns:
        np.ndarray: The contrast-stretched image (NumPy array), with pixel values scaled to 0-255.
    """
    image_float = image.astype(np.double)

    # Apply contrast stretching formula
    # Calculate min/max values based on percentiles for each channel
    min_vals = np.percentile(image_float, 1.5, axis=(0, 1))
    max_vals = np.percentile(image_float, 98.5, axis=(0, 1))

    # Get maximum value for dtype
    max_val = np.iinfo(dtype).max

    # Apply contrast stretching
    stretched_image = (image_float - min_vals) / (max_vals - min_vals + 1)
    return np.clip(max_val * stretched_image, 0, max_val).astype(dtype)


def extract_features(images: List[np.ndarray], depths: List[np.ndarray], clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts 2D and 3D features from lists of grayscale, HSV, and optional depth patches.

    Uses multi-threading to parallelize the feature extraction process for different
    feature types (Color, LBP, GLCM, Gabor, Principal Plane, Curvatures, Symmetry).

    Args:
        images_gray: A list of grayscale image patches as NumPy arrays.
        images_hsv: A list of HSV image patches as NumPy arrays.
        depth: An optional list of depth patches as NumPy arrays. Defaults to None.

    Returns:
        A tuple containing two NumPy arrays:
        - The first array contains the concatenated 2D features.
        - The second array contains the concatenated 3D features (empty if depth is None).
    """
    data_logger = lambda name, t, f: (
        logger.info(f"Started processing {name}: {datetime.datetime.now().isoformat()}"),
        f(),
        logger.info(f"Finished processing {name}: {round(time.perf_counter() - t, 2)} seconds")
    )

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    s = time.perf_counter()

    images_hsv = [contrast_stretch(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)) for img in images]
    images_gray = [clahe.apply(img[:, :, 2]) for img in images_hsv]

    logger.info(f"Preprocessing completed in {time.perf_counter() - s:.2f} seconds.")

    features: dict[str, List[np.ndarray]] = {}

    ts: List[Thread] = []

    # Add color features
    ts.append(Thread(target=data_logger,
        args=(
            "Gray Color Features",
            time.perf_counter(),
            lambda: features.update({'gray': [extract_color_features(img) for img in images_gray]})
        )
    ))

    # Add color HSV features
    ts.append(Thread(target=data_logger,
        args=(
            "HSV Color Features",
            time.perf_counter(),
            lambda: features.update({'hsv': [extract_color_features(img[:, :, 0]) for img in images_hsv]})
        )
    ))

    ts.append(Thread(target=data_logger,
        args=(
            "LBP Features",
            time.perf_counter(),
            lambda: features.update({'lbp': [extract_lbp_features(img) for img in images_gray]})
        )
    ))

    ts.append(Thread(target=data_logger,
        args=(
            "GLCM Features",
            time.perf_counter(),
            lambda: features.update({'glcm': [extract_glcm_features(img) for img in images_gray]})
        )
    ))

    ts.append(Thread(target=data_logger,
        args=(
            "HOG Features",
            time.perf_counter(),
            lambda: features.update({'hog': [extract_hog_features(img) for img in images_gray]})
        )
    ))

    if depths is not None:
        ts.append(Thread(target=data_logger,
            args=(
                "Principal Plane Features",
                time.perf_counter(),
                lambda: features.update({'principal plane': [extract_principal_plane_features(d) for d in depths]})
            )
        ))

        ts.append(Thread(target=data_logger,
            args=(
                "Curvature and Surface Normal Features",
                time.perf_counter(),
                lambda: features.update({'curvatures': [extract_curvatures_and_surface_normals(d) for d in depths]})
            )
        ))

        ts.append(Thread(target=data_logger,
            args=(
                "Symmetry Features",
                time.perf_counter(),
                lambda: features.update({'symmetry': [extract_symmetry_features(d) for d in depths]})
            )
        ))

    for t in ts:
        t.start()

    for t in ts:
        t.join()

    features_2d = np.concatenate((features['gray'], features['hsv'], features['lbp'], features['glcm'], features['hog']), axis=1)
    features_3d = np.concatenate((features['principal plane'], features['curvatures'], features['symmetry']), axis=1)

    return np.nan_to_num(features_2d), np.nan_to_num(features_3d)