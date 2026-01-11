from typing import Literal, Tuple, List
import os, cv2, msgpack, time, random, logging
import numpy as np

from ..features import extract_features

logger = logging.getLogger(__name__)


def load_data(images_dir: str, depth_dir: str, subset: int = 0) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Loads image and optional depth data from specified directories, organized by class.

    Args:
        images_dir (str): The path to the directory containing image subdirectories for each class.
        depth_dir (str): The path to the directory containing depth map subdirectories for each class.
        subset (int): If greater than 0, loads a random subset of images up to this number,
                      maintaining class distribution based on original counts.

    Returns:
        Tuple[np.ndarray, np.ndarray, List[str]]: A tuple containing:
            - images (np.ndarray): A NumPy array of loaded images.
            - depths (np.ndarray): A NumPy array of loaded depth maps, or None if depth_dir is None or empty.
                                           Depth maps are loaded as np.double and NaN values are converted to 0.
            - labels (List[str]): A list of class labels corresponding to each loaded image.
    """
    classes = [d for d in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, d))]
    f_names: List[List[str]] = []

    # Calculate weights based on the number of images per class for subsetting
    weights = np.zeros((len(classes),))
    for i, c in enumerate(classes):
        image_list = os.listdir(os.path.join(images_dir, c))
        f_names.append(image_list)
        weights[i] = len(image_list)

    # Normalize weights to sum to 1
    total_images = np.sum(weights)
    if total_images > 0:
        weights /= total_images
    else:
        weights = np.zeros_like(weights)

    labels: List[str] = []
    images_list: List[np.ndarray] = []
    depths_list: List[np.ndarray] = []

    for i, (class_name, image_names) in enumerate(zip(classes, f_names)):
        class_image_dir = os.path.join(images_dir, class_name)
        class_depth_dir = os.path.join(depth_dir, class_name)

        if not os.path.isdir(class_image_dir):
            continue

        # Select a random subset of image names if subset is specified
        if subset > 0 and len(image_names) > 0:
            num_samples = int(weights[i] * subset)
            image_names_subset = random.sample(image_names, min(num_samples, len(image_names)))
        else:
            image_names_subset = image_names

        for image_name in image_names_subset:
            img = cv2.imread(os.path.join(class_image_dir, image_name))
            depth = cv2.imread(os.path.join(class_depth_dir, image_name), cv2.IMREAD_UNCHANGED).astype(np.float32)

            if img is not None and depth is not None: # Ensure image was loaded successfully
                images_list.append(img)
                depths_list.append(depth)
                labels.append(class_name)

    return np.array(images_list), np.array(depths_list), labels


def save_features(images_dir: str, depth_dir: str, features_dir: str, subset: int = 0) -> None:
    """
    Loads image and depth data, processes images, extracts features, and saves features and labels
    to msgpack files.

    Args:
        images_dir (str): The path to the directory containing image subdirectories.
        depth_dir (str): The path to the directory containing depth map subdirectories.
                                   If None, only 2D features are extracted/saved.
        features_dir (str): The path to the directory where extracted features and labels will be saved.
                            Creates the directory if it doesn't exist.
        subset (int): If greater than 0, loads and processes a random subset of images.
    """
    get_time = lambda t: round(time.perf_counter() - t, 2)

    if not os.path.exists(features_dir) or not os.path.isdir(features_dir):
        os.makedirs(features_dir)

    logger.info("Loading data...")

    t_start = time.perf_counter()
    images, depths, labels = load_data(images_dir, depth_dir, subset)

    with open(f"{features_dir}/labels.msgpack", 'wb') as f:
        f.write(msgpack.packb(labels))

    logger.info(f"Loaded data and saved labels: {get_time(t_start)}s")

    logger.info("Extracting features...")
    t_start = time.perf_counter()
    features_2d, features_3d = extract_features(images, depths)

    logger.info(f"Extracted features: {get_time(t_start)}s")

    if features_3d is not None and features_3d.size > 0:
        features = features_3d
        with open(f"{features_dir}/features_3D.msgpack", 'wb') as f:
            f.write(msgpack.packb(features.tolist()))

        logger.info(f"Features 3D Shape: {features.shape}")

    if features_2d is not None and features_2d.size > 0:
        features = features_2d
        with open(f"{features_dir}/features_2D.msgpack", 'wb') as f:
            f.write(msgpack.packb(features.tolist()))

        logger.info(f"Features 2D Shape: {features.shape}")


def load_features(features_dir: str, dimension: Literal['2', '25', '3'] = '2') -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads features and labels from msgpack files saved by `save_features`.

    Args:
        features_dir (str): The path to the directory where features and labels are saved.
        dimension (Literal['2', '25', '3']): The dimension of features to load.
                                            '2': Loads 2D features.
                                            '25': Loads 2.5D features (2D + 3D).
                                            '3': Loads 3D features.

    Returns:
        Tuple[np.ndarray, List[str]]: A tuple containing:
            - features (np.ndarray): A NumPy array of loaded features.
            - labels (List[str]): A list of loaded class labels.
    """

    features = None
    if dimension == '2' or dimension == '25':
        logger.info("Loading 2D Features")
        with open(f"{features_dir}/features_2D.msgpack", 'rb') as f:
            features_2d = np.array(msgpack.unpackb(f.read()))

        features = features_2d

    if dimension == '3' or dimension == '25':
        logger.info("Loading 3D Features")
        with open(f"{features_dir}/features_3D.msgpack", 'rb') as f:
            features_3d = np.array(msgpack.unpackb(f.read()))

        features = features_3d

    if dimension == '25':
        features = np.concatenate((features_2d, features_3d), axis=1)

    if features is None:
        raise ValueError(f"Invalid dimension specified: {dimension}. Must be '2', '25', or '3'.")

    with open(f"{features_dir}/labels.msgpack", 'rb') as f:
        labels = np.array(msgpack.unpackb(f.read()), dtype=str)

    return features, labels
