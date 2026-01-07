from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from scipy.stats import skew, kurtosis
from typing import List, Literal, Tuple
from threading import Thread
import cv2
import numpy as np
import time
import datetime


# https://stackoverflow.com/questions/33781502/how-to-get-the-real-and-imaginary-parts-of-a-gabor-kernel-matrix-in-opencv
KERNELS: List[np.ndarray] = [
    cv2.getGaborKernel(ksize=(5, 5), sigma=sigma, theta=np.pi * theta / 4, lambd=1 / frequency, gamma=1, psi=0) +  # Real part (the cosine)
    1j * cv2.getGaborKernel(ksize=(5, 5), sigma=sigma, theta=np.pi * theta / 4, lambd=1 / frequency, gamma=1, psi=np.pi / 2)  # Complex part (the sine)
    for theta in range(4)
    for sigma in (1, 3)
    for frequency in (0.05, 0.25)
]


def contrast_enhancement(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance image contrast.

    Enhances the value (V) channel of the input BGR image in HSV color space.

    Args:
        image (np.ndarray): The input BGR image (NumPy array).
        clip_limit (float): Threshold for contrast limiting.
        tile_grid_size (Tuple[int, int]): Size of the grid for histogram equalization.

    Returns:
        np.ndarray: The contrast-enhanced BGR image (NumPy array).
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    # The value channel is the intensity of the image (gray scale)
    hsv_image[:, :, 2] = clahe.apply(hsv_image[:, :, 2])
    enhanced_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return enhanced_image


def contrast_stretch(image: np.ndarray) -> np.ndarray:
    """
    Applies contrast stretching to the input image using percentile values.

    Stretches the intensity range of each channel based on the 1.5th and 98.5th percentiles
    to improve visibility.

    Args:
        image (np.ndarray): The input image (NumPy array). Expected to be BGR or grayscale.

    Returns:
        np.ndarray: The contrast-stretched image (NumPy array), with pixel values scaled to 0-255.
    """
    # Convert to float to avoid overflow during calculations
    image_float = image.astype(np.float32)

    # Apply contrast stretching formula
    # Calculate min/max values based on percentiles for each channel
    min_vals = np.percentile(image_float, 1.5, axis=(0, 1))
    max_vals = np.percentile(image_float, 98.5, axis=(0, 1))

    # If a channel has a uniform color (min == max), avoid division by 0
    zero_range_indices = max_vals - min_vals == 0
    max_vals[zero_range_indices] = min_vals[zero_range_indices] + 1 # Add a small epsilon

    # Stretch image and clip values to [0, 255]
    stretched_image = (image_float - min_vals) / (max_vals - min_vals)
    return np.clip(255 * stretched_image, 0, 255).astype(np.uint8)


def process_images(images: List[np.ndarray], enhance_contrast: bool = True, stretch_contrast: bool = True, clip_limit: float = 40.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies a sequence of image processing steps (contrast enhancement and stretching)
    to a list of images and returns the results in grayscale and HSV format.

    Args:
        images (List[np.ndarray]): A list of input BGR images (NumPy arrays).
        enhance_contrast (bool): Whether to apply contrast enhancement (CLAHE).
        stretch_contrast (bool): Whether to apply contrast stretching.
        clip_limit (float): Clip limit for CLAHE if enhance_contrast is True.
        tile_grid_size (Tuple[int, int]): Tile grid size for CLAHE if enhance_contrast is True.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing two NumPy arrays:
            - A stacked array of grayscale images.
            - A stacked array of HSV images.
        Both arrays have shape (n_images, height, width) or (n_images, height, width, 3).
    """
    images_gray: List[np.ndarray] = []
    images_hsv: List[np.ndarray] = []

    for image in images:
        processed_image = image.copy()

        if enhance_contrast:
            processed_image = contrast_enhancement(processed_image, clip_limit=clip_limit, tile_grid_size=tile_grid_size)

        if stretch_contrast:
            processed_image = contrast_stretch(processed_image)

        images_gray.append(cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY))
        images_hsv.append(cv2.cvtColor(processed_image, cv2.COLOR_BGR2HSV))

    return np.array(images_gray), np.array(images_hsv)


def extract_color_features(image: np.ndarray, bins: int = 8, range: Tuple[int, int] = (0, 256)) -> np.ndarray:
    """
    Extracts color histogram features from an image.

    Args:
        image: The input image as a NumPy array.
        bins: The number of bins for the histogram. Defaults to 8.
        range: The range of pixel values to consider. Defaults to (0, 256).

    Returns:
        A NumPy array representing the normalized color histogram.
    """
    color_features, _ = np.histogram(image.flatten(), bins=bins, range=range, density=True)
    return color_features


def extract_lbp_features(image: np.ndarray, n_points: int = 24, radius: float = 3.0) -> np.ndarray:
    """
    Extracts Local Binary Pattern (LBP) features from a grayscale image.

    Args:
        image: The input grayscale image as a NumPy array.
        n_points: The number of circular neighboring points to consider. Defaults to 24.
        radius: The radius of the circle. Defaults to 3.0.

    Returns:
        A NumPy array representing the LBP histogram features.
    """
    # https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_local_binary_pattern.html
    lbp_image = local_binary_pattern(image, n_points, radius, method='uniform')

    n_bins = n_points + 2  # Uniform lbp has max(lbp_image) = n_points + 1, and n_bins = max(lbp_image) + 1
    return extract_color_features(lbp_image, bins=n_bins, range=(0, n_bins))


def extract_glcm_features(image: np.ndarray, distances: List[int] = [1], angles: List[float] = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], properties: List[Literal['contrast', 'dissimilarity', 'energy', 'homogeneity', 'correlation', 'ASM']] = ['dissimilarity', 'energy', 'homogeneity']) -> np.ndarray:
    """
    Extracts Gray Level Co-occurrence Matrix (GLCM) features from a grayscale image.

    Args:
        image: The input grayscale image as a NumPy array.
        distances: List of pixel pair distances. Defaults to [1].
        angles: List of pixel pair angles (in radians). Defaults to [0, pi/4, pi/2, 3*pi/4].
        properties: List of GLCM properties to compute. Defaults to ['dissimilarity', 'energy', 'homogeneity'].

    Returns:
        A NumPy array representing the concatenated GLCM properties.
    """
    # https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.graycoprops
    # https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_glcm.html#sphx-glr-auto-examples-features-detection-plot-glcm-py
    glcm = graycomatrix(image, distances=distances, angles=angles, symmetric=True, normed=True)
    return np.concatenate([graycoprops(glcm, prop).flatten() for prop in properties])


def extract_gabor_features(image: np.ndarray) -> List[float]:
    """
    Extracts Gabor filter features from a grayscale image.

    Applies a set of pre-defined Gabor kernels to the image and computes
    statistical features (local energy, mean amplitude, phase mean/std/skew/kurtosis)
    from the filter responses.

    Args:
        image: The input grayscale image as a NumPy array.

    Returns:
        A list of float values representing the concatenated Gabor features.
    """
    feats = []
    for kernel in KERNELS:
        response_real = cv2.filter2D(src=image, ddepth=-1, kernel=np.real(kernel))
        response_imag = cv2.filter2D(src=image, ddepth=-1, kernel=np.imag(kernel))

        # Source: https://stackoverflow.com/questions/20608458/gabor-feature-extraction
        response_squared = response_real ** 2 + response_imag ** 2
        local_energy = np.sum(response_squared)
        mean_amplitude = np.mean(np.sqrt(response_squared))
        phase_amplitude = np.atan2(response_imag, response_real).flatten()

        feats.extend((
            local_energy,             mean_amplitude,
            np.mean(phase_amplitude), np.std(phase_amplitude),
            skew(phase_amplitude),    kurtosis(phase_amplitude)
        ))

    return feats


# 3D Features
def extract_principal_plane_features(depth_patch: np.ndarray, eps: float = 1e-6) -> List[float]:
    """
    Extracts features related to the principal plane fitting and rugosity of a depth patch.

    Fits a 3rd order polynomial to the depth data, calculates distances to the plane,
    and computes rugosity (surface area / planar area).

    Args:
        depth_patch: The input depth patch as a NumPy array.
        eps: A small value epsilon to avoid divisions by zero.

    Returns:
        A list of float values representing the statistical properties of depth,
        polynomial coefficients, angle with vertical, mean/std distance to plane, and rugosity.
    """
    triangle_area = lambda p_1, p_2, p_3: np.linalg.norm(np.cross(p_2 - p_1, p_3 - p_1), axis=2) / 2

    # Create a grid of x and y coordinates
    h, w = depth_patch.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    # Create a 3D array of points (x, y, depth)
    points = np.dstack((x, y, depth_patch))

    # Prepare arrays for the vertices of the triangles
    p1 = points[:-1, :-1]  # Top-left
    p2 = points[1:, :-1]   # Bottom-left
    p3 = points[:-1, 1:]   # Top-right
    p4 = points[1:, 1:]    # Bottom-right

    # Calculate the area of the two triangles for all pixels at once
    area1 = triangle_area(p1, p2, p3)
    area2 = triangle_area(p2, p4, p3)

    # Sum the areas
    As = np.sum(area1 + area2)
    Ap = (w - 1) * (h - 1)

    rugosity = As / Ap

    # Flatten the arrays
    x = x.flatten()
    y = y.flatten()
    z = depth_patch.flatten()  # Flatten the depth values

    # Create the design matrix for the polynomial terms
    A = np.vstack([
        np.ones_like(z),  # p1
        x,                # p2 * x
        y,                # p3 * y
        x**2,             # p4 * x^2
        x * y,            # p5 * x * y
        y**2,             # p6 * y^2
        x**2 * y,         # p7 * x^2 * y
        x * y**2,         # p8 * x * y^2
        y**3              # p9 * y^3
    ]).T

    # Perform least squares fitting
    # coeffs are p1, p2, p3, p4, p5, p6, p7, p8, and p9
    coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)

    # Calculate the fitted z values using the polynomial equation
    z_fitted = (
        coeffs[0] +
        coeffs[1] * x +
        coeffs[2] * y +
        coeffs[3] * x**2 +
        coeffs[4] * x * y +
        coeffs[5] * y**2 +
        coeffs[6] * x**2 * y +
        coeffs[7] * x * y**2 +
        coeffs[8] * y**3
    )

    # Calculate the distances from each point to the fitted plane
    distances = np.abs(z - z_fitted)
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)

    n_z = -1
    n_x = coeffs[1]  # Coefficient for x
    n_y = coeffs[2]  # Coefficient for y

    # Calculate angle with respect to the vertical using the magnitude of the linear normal vector
    normal_magnitude = np.sqrt(n_x ** 2 + n_y ** 2 + n_z ** 2)
    theta = np.degrees(np.arccos(n_z / (normal_magnitude + eps)))

    return [np.std(z), skew(z), kurtosis(z)] + coeffs.tolist() + [theta, mean_distance, std_distance, rugosity]


def extract_curvatures_and_surface_normals(depth_patch: np.ndarray, eps: float = 1e-6) -> List[float]:
    """
    Extracts mean and standard deviation of principal curvatures, shape index,
    curvedness, and surface normal angles from a depth patch.

    Calculates first and second derivatives to compute Gaussian (G) and Mean (M)
    curvatures, then derives principal curvatures (k1, k2), Shape Index (S),
    and Curvedness (C). Also computes surface normal angles (alpha, beta).

    Args:
        depth_patch: The input depth patch as a NumPy array.
        eps: A small value epsilon to avoid divisions by zero.

    Returns:
        A list of float values representing the mean and std of the calculated features.
    """
    dx, dy = np.gradient(depth_patch)
    dxdx, dxdy = np.gradient(dx)
    dydx, dydy = np.gradient(dy)

    G = (dxdx * dydy - dxdy * dydx) / (1 + dx ** 2 + dy ** 2) ** 2
    M = (dydy + dxdx) / (2 * (1 + dx ** 2 + dy ** 2) ** (1.5))

    discriminant = np.sqrt(np.maximum(M ** 2 - G, 0))
    k1 = M + discriminant
    k2 = M - discriminant

    S = (2 / np.pi) * np.arctan2(k2 + k1, k2 - k1)
    C = np.sqrt((k1**2 + k2**2) / 2)

    # Calculate the surface normals
    nx = -dx
    ny = -dy
    nz = np.ones_like(depth_patch)  # Assuming z = depth_patch

    # Normalize the surface normals
    norm = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    nx /= norm + eps
    ny /= norm + eps
    nz /= norm + eps

    alpha = np.arctan2(ny, nx)  # Angle in the xy-plane
    beta = np.arctan2(nz, np.sqrt(nx**2 + ny**2))  # Angle from the z-axis

    return [
        np.mean(G),     np.std(G),
        np.mean(M),     np.std(M),
        np.mean(k1),    np.std(k1),
        np.mean(k2),    np.std(k2),
        np.mean(S),     np.std(S),
        np.mean(C),     np.std(C),
        np.mean(alpha), np.std(alpha),
        np.mean(beta),  np.std(beta)
    ]


def extract_features(images: List[np.ndarray], depths: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
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
    logger = lambda name, t, f: (
        print(f"Started processing {name}: {datetime.datetime.now().isoformat()}"),
        f(),
        print(f"Finished processing {name}: {round(time.perf_counter() - t, 2)} seconds")
    )

    images_gray, images_hsv = process_images(images)

    features: dict[str, List[np.ndarray]] = {}

    ts: List[Thread] = []

    # Add color features
    ts.append(Thread(target=logger,
        args=(
            "Gray Color Features",
            time.perf_counter(),
            lambda: features.update({'gray': [extract_color_features(img) for img in images_gray]})
        )
    ))

    # Add color HSV features
    ts.append(Thread(target=logger,
        args=(
            "HSV Color Features",
            time.perf_counter(),
            lambda: features.update({'hsv': [extract_color_features(img[:, :, 0]) for img in images_hsv]})
        )
    ))

    ts.append(Thread(target=logger,
        args=(
            "LBP Features",
            time.perf_counter(),
            lambda: features.update({'lbp': [extract_lbp_features(img) for img in images_gray]})
        )
    ))

    ts.append(Thread(target=logger,
        args=(
            "GLCM Features",
            time.perf_counter(),
            lambda: features.update({'glcm': [extract_glcm_features(img) for img in images_gray]})
        )
    ))

    ts.append(Thread(target=logger,
        args=(
            "Gabor Features",
            time.perf_counter(),
            lambda: features.update({'gabor': [extract_gabor_features(img) for img in images_gray]})
        )
    ))

    if depths is not None:
        ts.append(Thread(target=logger,
            args=(
                "Principal Plane Features",
                time.perf_counter(),
                lambda: features.update({'principal plane': [extract_principal_plane_features(d) for d in depths]})
            )
        ))

        ts.append(Thread(target=logger,
            args=(
                "Curvature and Surface Normal Features",
                time.perf_counter(),
                lambda: features.update({'curvatures': [extract_curvatures_and_surface_normals(d) for d in depths]})
            )
        ))

        ts.append(Thread(target=logger,
            args=(
                "Symmetry Features",
                time.perf_counter(),
                lambda: features.update({'symmetry': [extract_gabor_features(d) for d in depths]})
            )
        ))

    for t in ts:
        t.start()

    for t in ts:
        t.join()

    features_2d = np.concatenate((features['gray'], features['hsv'], features['lbp'], features['glcm'], features['gabor']), axis=1)
    features_3d = np.concatenate((features['principal plane'], features['curvatures'], features['symmetry']), axis=1)

    return np.nan_to_num(features_2d), np.nan_to_num(features_3d)