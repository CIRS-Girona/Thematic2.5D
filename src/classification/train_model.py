from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt
from typing import Literal
import time, os, logging

from ..utils import load_features, save_features
from . import SVMModel

logger = logging.getLogger(__name__)


def train_model(dataset_dir: str, features_dir: str, models_dir: str, results_dir: str, uxo_start_code: int, binary_mode: bool = False, test_size: float = 0.1, n_components: int = 100, dimension: Literal['2', '25', '3'] = '25', subset_size: int = 0) -> None:
    """
    Trains an SVM model using extracted features and evaluates its performance.

    Optionally extracts features from images and depth maps, loads features,
    splits data into training and testing sets, trains an SVM model, evaluates
    it, and saves the model, classification report, and confusion matrix plot.

    Args:
        dataset_dir: Directory containing image and depth data (used if features are not saved).
        features_dir: Directory to save/load the extracted features.
        models_dir: Directory to save the trained SVM model.
        results_dir: Directory to save the classification report and confusion matrix plot.
        uxo_start_code: The starting integer code representing UXO classes in multi-class mode.
        binary_mode: If True, converts multi-class labels to binary ('uxo' vs 'background'). Defaults to False.
        test_size: The proportion of the dataset to include in the test split. Defaults to 0.1.
        n_components: The number of components for PCA dimensionality reduction within the SVM model. Defaults to 100.
        dimension: Specifies which features to use ('2' for 2D, '3' for 3D, '25' for combined). Defaults to '25'.
        use_saved_features: If True, loads features from features_dir; otherwise, extracts and saves them. Defaults to True.
        subset_size: Number of samples to use for feature extraction (if use_saved_features is False). Defaults to 0 (use all).
    """
    if not os.path.exists(f"{features_dir}/features.npz"):
        save_features(f"{dataset_dir}/2D/", f"{dataset_dir}/3D/", features_dir, subset=subset_size)

    logger.info(f"Starting training for {dimension}D SVM model (binary_mode={binary_mode})...")

    # Load features and encode labels
    X_data, y_data = load_features(features_dir, dimension=dimension)

    if binary_mode:
        for y in np.unique(y_data):
            if y.isdigit() and int(y) >= uxo_start_code:
                y_data[y == y_data] = 'uxo'
            else:
                y_data[y == y_data] = 'background'

    s = time.perf_counter()
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, stratify=y_data, test_size=test_size)

    model = SVMModel(label=dimension, model_dir=models_dir, n_components=n_components)
    model.train(X_train, y_train)
    model.save_model()
    logger.info(f"Training completed in {time.perf_counter() - s:.2f} seconds.")

    y_pred = model.predict(X_test)

    if binary_mode:
        datafile_name = f"{model.name}{dimension}_binary"
    else:
        datafile_name = f"{model.name}{dimension}"

    report = classification_report(y_test, y_pred, zero_division=0)
    logger.info("\n" + report)

    os.makedirs(results_dir, exist_ok=True)
    with open(f"{results_dir}/{datafile_name}.txt", 'w') as f:
        print(report, file=f) # Write to file

    cm = confusion_matrix(y_test, y_pred, normalize='true')
    cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.model.classes_)

    cmd.plot()
    plt.tight_layout()
    plt.savefig(f"{results_dir}/{datafile_name}.jpg")
    plt.close('all')