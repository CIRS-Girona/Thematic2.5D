import numpy as np
import cv2
import ctypes
from typing import List

from ..utils import load_cpp_library, Sensor

# Load the C++ library
LIB = load_cpp_library("libfastmetrics.so")

# --- Types ---
POINTER_DOUBLE = ctypes.POINTER(ctypes.c_double)
NP_FLOAT64_C = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
NP_UINT8_C = np.ctypeslib.ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS')

# --- Signatures ---
LIB.calculate_ground_resolution_c.restype = ctypes.c_double
LIB.calculate_ground_resolution_c.argtypes = [
    NP_FLOAT64_C, NP_FLOAT64_C, NP_FLOAT64_C, ctypes.c_int,
    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double
]

LIB.calculate_slant_c.restype = None
LIB.calculate_slant_c.argtypes = [
    NP_FLOAT64_C, ctypes.c_int, ctypes.c_int,
    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
    POINTER_DOUBLE
]

LIB.calculate_uciqe_c.restype = None
LIB.calculate_uciqe_c.argtypes = [
    NP_UINT8_C, ctypes.c_int, ctypes.c_int, POINTER_DOUBLE
]

LIB.calculate_channel_eme_c.restype = None
LIB.calculate_channel_eme_c.argtypes = [
    NP_UINT8_C, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_bool, ctypes.c_double, ctypes.c_double, POINTER_DOUBLE
]

LIB.calculate_uicm_c.restype = None
LIB.calculate_uicm_c.argtypes = [
    NP_FLOAT64_C, NP_FLOAT64_C, ctypes.c_int, POINTER_DOUBLE
]

# =================================================================
#  PYTHON FUNCTIONS
# =================================================================
def calculate_ground_resolution(sensor: Sensor, u: np.ndarray, v: np.ndarray, z: np.ndarray) -> float:
    # Ensure C-contiguous
    u_c = np.ascontiguousarray(u, dtype=np.float64)
    v_c = np.ascontiguousarray(v, dtype=np.float64)
    z_c = np.ascontiguousarray(z, dtype=np.float64)
    
    return LIB.calculate_ground_resolution_c(
        u_c, v_c, z_c, len(u_c),
        sensor.fx, sensor.fy, sensor.cx, sensor.cy
    )


def calculate_slant(sensor: Sensor, depth: np.ndarray) -> float:
    depth_c = np.ascontiguousarray(depth, dtype=np.float64)
    result = ctypes.c_double(0.0)

    LIB.calculate_slant_c(
        depth_c, depth_c.shape[0], depth_c.shape[1],
        sensor.fx, sensor.fy, sensor.cx, sensor.cy,
        ctypes.byref(result)
    )
    return result.value


def calculate_UCIQE(img: np.ndarray) -> float:
    # Color conversion in OpenCV is highly optimized
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_c = np.ascontiguousarray(lab, dtype=np.uint8)

    result = ctypes.c_double(0.0)
    LIB.calculate_uciqe_c(lab_c, lab_c.shape[0], lab_c.shape[1], ctypes.byref(result))
    return result.value


def calculate_eme_logamee(img: np.ndarray, blocksize=8, gamma=1026, k=1026) -> List[float]:
    emes = []
    result = ctypes.c_double(0.0)

    # Process B, G, R channels
    for c in range(3):
        channel = img[:, :, c]

        dx = cv2.Sobel(channel, cv2.CV_16S, 1, 0)
        dy = cv2.Sobel(channel, cv2.CV_16S, 0, 1)

        # Standard approach: abs(dx) + abs(dy)
        sobel_abs = cv2.addWeighted(cv2.convertScaleAbs(dx), 0.5, cv2.convertScaleAbs(dy), 0.5, 0)
        processed = (channel.astype(np.float32) * (sobel_abs.astype(np.float32) / 255.0)).astype(np.uint8)

        ch_c = np.ascontiguousarray(processed, dtype=np.uint8)
        LIB.calculate_channel_eme_c(
            ch_c, ch_c.shape[0], ch_c.shape[1], 
            blocksize, False, gamma, k, ctypes.byref(result)
        )
        emes.append(result.value)

    # Process Gray channel (LogAMEE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_c = np.ascontiguousarray(gray, dtype=np.uint8)
    LIB.calculate_channel_eme_c(
        gray_c, gray_c.shape[0], gray_c.shape[1], 
        blocksize, True, gamma, k, ctypes.byref(result)
    )
    emes.append(result.value)

    return emes

def calculate_UIQM(img: np.ndarray) -> float:
    # Convert once to float for math
    img_f = img.astype(np.float64)
    
    # Calculate RGL and YBL
    # R - G
    rgl_data = img_f[:,:,2] - img_f[:,:,1]
    # (R + G)/2 - B
    ybl_data = (img_f[:,:,2] + img_f[:,:,1]) / 2.0 - img_f[:,:,0]
    
    # Flatten and Sort
    # Sorting is the bottleneck here (N log N).
    # Since we only trim 10%, we could use np.partition (O(N)) if we only needed mean,
    # but UICM uses std dev which requires the actual values. 
    # Sorting is likely unavoidable without a bucket approach.
    rgl = np.sort(rgl_data, axis=None)
    ybl = np.sort(ybl_data, axis=None)

    T = int(0.1 * len(rgl))
    
    # Slicing creates copies usually, ensure contiguous
    rgl_trimmed = np.ascontiguousarray(rgl[T:-T])
    ybl_trimmed = np.ascontiguousarray(ybl[T:-T])
    
    uicm_result = ctypes.c_double(0.0)
    if len(rgl_trimmed) > 0:
        LIB.calculate_uicm_c(rgl_trimmed, ybl_trimmed, len(rgl_trimmed), ctypes.byref(uicm_result))
    
    uicm = uicm_result.value

    Beme, Geme, Reme, uiconm = calculate_eme_logamee(img)
    uism = 0.299 * Reme + 0.587 * Geme + 0.114 * Beme

    return 0.0282 * uicm + 0.2953 * uism + 3.5753 * uiconm