import numpy as np
import cv2
from skimage.filters import sobel
from typing import List
import ctypes

from ..utils import load_cpp_library

# Load the C++ library
LIB = load_cpp_library("libfastmetrics.so")

# --- Define ctypes argument types for type safety ---

# Define a pointer to a double
POINTER_DOUBLE = ctypes.POINTER(ctypes.c_double)

# Define NumPy-compatible pointer types
# We enforce C-contiguous arrays for safe C++ processing
NP_FLOAT64_C = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
NP_UINT8_C = np.ctypeslib.ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS')

# --- Set up function signatures for calculate_ground_resolution_c ---
LIB.calculate_ground_resolution_c.restype = ctypes.c_double
LIB.calculate_ground_resolution_c.argtypes = [
    NP_FLOAT64_C,      # double* u
    NP_FLOAT64_C,      # double* v
    NP_FLOAT64_C,      # double* z
    ctypes.c_int,      # int n
    ctypes.c_double,   # double fx
    ctypes.c_double,   # double fy
    ctypes.c_double,   # double cx
    ctypes.c_double    # double cy
]

# --- Set up function signatures for calculate_slant_c ---
LIB.calculate_slant_c.restype = None
LIB.calculate_slant_c.argtypes = [
    NP_FLOAT64_C,      # double* depth_data
    ctypes.c_int,      # int rows
    ctypes.c_int,      # int cols
    ctypes.c_double,   # double fx
    ctypes.c_double,   # double fy
    ctypes.c_double,   # double cx
    ctypes.c_double,   # double cy
    POINTER_DOUBLE     # double* out_slant_angle
]

# --- Set up function signatures for calculate_uciqe_c ---
LIB.calculate_uciqe_c.restype = None
LIB.calculate_uciqe_c.argtypes = [
    NP_UINT8_C,        # unsigned char* lab_data
    ctypes.c_int,      # int rows
    ctypes.c_int,      # int cols
    POINTER_DOUBLE     # double* out_uciqe
]

# --- Set up function signatures for calculate_channel_eme_c ---
LIB.calculate_channel_eme_c.restype = None
LIB.calculate_channel_eme_c.argtypes = [
    NP_UINT8_C,        # unsigned char* ch_data
    ctypes.c_int,      # int rows
    ctypes.c_int,      # int cols
    ctypes.c_int,      # int blocksize
    ctypes.c_bool,     # bool is_logamee
    ctypes.c_double,   # double gamma
    ctypes.c_double,   # double k
    POINTER_DOUBLE     # double* out_eme
]

# --- Set up function signatures for calculate_uicm_c ---
LIB.calculate_uicm_c.restype = None
LIB.calculate_uicm_c.argtypes = [
    NP_FLOAT64_C,      # double* rgl_trimmed
    NP_FLOAT64_C,      # double* ybl_trimmed
    ctypes.c_int,      # int n
    POINTER_DOUBLE     # double* out_uicm
]


# =================================================================
#  PYTHON FUNCTIONS (Wrapper API)
# =================================================================
def calculate_ground_resolution(sensor: Sensor, u: np.ndarray, v: np.ndarray, z: np.ndarray) -> float:
    """
    Calculates the ground resolution (mm/px) using the fast C++ backend.
    """
    # Ensure inputs are C-contiguous float64 arrays
    u_c = np.ascontiguousarray(u, dtype=np.float64)
    v_c = np.ascontiguousarray(v, dtype=np.float64)
    z_c = np.ascontiguousarray(z, dtype=np.float64)

    n = len(u_c)
    if n == 0:
        return 0.0

    # Call the C++ function and return the double result directly
    return LIB.calculate_ground_resolution_c(
        u_c, v_c, z_c, n,
        sensor.fx, sensor.fy, sensor.cx, sensor.cy
    )


def calculate_slant(sensor: Sensor, depth: np.ndarray) -> float:
    """
    Calculates the slant angle using the fast C++/Eigen backend.
    """
    # Ensure data is C-contiguous and in the correct format (float64)
    depth_c = np.ascontiguousarray(depth, dtype=np.float64)
    rows, cols = depth_c.shape

    # Create a C-style double to store the result
    result = ctypes.c_double(0.0)

    # Call the C++ function
    LIB.calculate_slant_c(
        depth_c, rows, cols,
        sensor.fx, sensor.fy, sensor.cx, sensor.cy,
        ctypes.byref(result)
    )

    return result.value


def calculate_UCIQE(img: np.ndarray) -> float:
    """
    Calculates the UCIQE metric using the fast C++ backend.
    The BGR2LAB conversion is still done in Python.
    """
    # Pre-processing (still in Python)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Ensure data is C-contiguous and in the correct format (uint8)
    lab_c = np.ascontiguousarray(lab, dtype=np.uint8)
    rows, cols, _ = lab_c.shape

    # Create a C-style double to store the result
    result = ctypes.c_double(0.0)

    # Call the C++ function
    LIB.calculate_uciqe_c(lab_c, rows, cols, ctypes.byref(result))

    return result.value


def calculate_eme_logamee(img: np.ndarray, blocksize=8, gamma=1026, k=1026) -> List[float]:
    """
    Calculates EME/LogAMEE using the fast C++ backend for the block-wise calculation.
    Sobel and Grayscale conversions are still done in Python.
    """
    emes = []
    result = ctypes.c_double(0.0) # We can re-use this result variable
    
    for c in range(4):
        # Pre-processing
        if c == 3:
            ch = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            is_logamee = True
        else:
            # Apply sobel filter to the channel
            sobel_ch = sobel(img[:, :, c])
            # Normalize and scale sobel output similar to original
            ch = np.round(img[:, :, c] * sobel_ch).astype(np.uint8)
            is_logamee = False

        # Ensure data is C-contiguous and in the correct format
        ch_c = np.ascontiguousarray(ch, dtype=np.uint8)
        rows, cols = ch_c.shape

        # Call the C++ helper function
        LIB.calculate_channel_eme_c(
            ch_c, rows, cols, 
            blocksize, is_logamee, 
            gamma, k,
            ctypes.byref(result)
        )
        emes.append(result.value)

    return emes


def calculate_UIQM(img: np.ndarray) -> float:
    """
    Calculates the UIQM metric, using C++ helpers for UICM and EME.
    """
    # Pre-processing (Python)
    rgl = np.sort(img[:, :, 2].astype(np.float64) - img[:, :, 1].astype(np.float64), axis=None)
    ybl = np.sort((img[:, :, 2].astype(np.float64) + img[:, :, 1].astype(np.float64)) / 2 - img[:, :, 0].astype(np.float64), axis=None)

    T = int(0.1 * len(rgl))
    
    rgl_trimmed = np.ascontiguousarray(rgl[T:-T], dtype=np.float64)
    ybl_trimmed = np.ascontiguousarray(ybl[T:-T], dtype=np.float64)
    n_trimmed = len(rgl_trimmed)

    uicm_result = ctypes.c_double(0.0)
    
    # Call C++ helper for UICM
    if n_trimmed > 0:
        LIB.calculate_uicm_c(rgl_trimmed, ybl_trimmed, n_trimmed, ctypes.byref(uicm_result))
    
    uicm = uicm_result.value

    Beme, Geme, Reme, uiconm = calculate_eme_logamee(img)
    uism = 0.299 * Reme + 0.587 * Geme + 0.114 * Beme

    return 0.0282 * uicm + 0.2953 * uism + 3.5753 * uiconm
