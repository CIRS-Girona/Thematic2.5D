import numpy as np
import os, pickle
from typing import Self, Any

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem


class _ClassificationModel:
    """
    Wrapper class to integrate a classification model into a scikit-learn pipeline.
    This class provides `fit` and `transform` methods compatible with Pipeline,
    where `transform` is used for prediction.
    """
    def __init__(self, model: Any):
        """
        Initializes the wrapper with a classification model.

        Args:
            model: The classification model instance to wrap (e.g., SVC, LogisticRegression).
        """
        self.model = model

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """
        Fits the underlying classification model to the training data.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training labels.

        Returns:
            Self: The instance itself.
        """
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Makes predictions using the fitted model.

        Args:
            X (np.ndarray): Input features for prediction.

        Returns:
            np.ndarray: Predicted labels for the input features.
        """
        return self.model.predict(X)


class ClassificationModel:
    """
    A class for creating, training, evaluating, and managing classification models
    within a scikit-learn pipeline, including preprocessing steps like scaling,
    kernel mapping, and PCA.
    """
    def __init__(self, model: Any, label: str = '', model_dir: str = './models/', model_name: str = '', standardize: bool = True, pca: bool = True, kernel_mapping: bool = True, n_components: int = 100):
        """
        Initializes the ClassificationModel with a base model and pipeline options.

        Args:
            model: The base classification model instance.
            label (str): An optional label for the model file name.
            model_dir (str): Directory to save/load models. Creates the directory if it doesn't exist.
            model_name (str): Base name for the model file.
            standardize (bool): Whether to include StandardScaler in the pipeline.
            pca (bool): Whether to include PCA in the pipeline.
            kernel_mapping (bool): Whether to include Nystroem kernel mapping in the pipeline.
            n_components (int): Number of components for Nystroem and PCA if used.
        """
        self.label = label
        self.name = model_name

        self.dir = model_dir
        if not os.path.exists(self.dir) or not os.path.isdir(self.dir):
            os.makedirs(self.dir)

        pipeline = []

        if standardize:
            pipeline.append(('scaling', StandardScaler()))

        if kernel_mapping:
            pipeline.append(('kernel', Nystroem(n_jobs=-1, n_components=n_components)))

        if pca:
            pipeline.append(('pca', PCA()))

        pipeline.append(('training', _ClassificationModel(model)))

        self.model = Pipeline(pipeline, verbose=True)


    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """
        Trains the entire pipeline, including preprocessing steps and the classification model.

        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training labels.

        Returns:
            np.ndarray: The transformed training data after all pipeline steps except the final estimator.
                        This is the output of the second-to-last step's transform method.
        """
        return self.model.fit_transform(X_train, y_train)


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Makes predictions on new data using the trained pipeline.

        Args:
            X (np.ndarray): Input features for prediction.

        Returns:
            np.ndarray: Predicted labels for the input features.
        """
        return self.model.transform(X)


    def save_model(self) -> None:
        """
        Saves the trained pipeline model object to a file.

        The filename is generated based on the model name, label,
        and whether it's a binary classification model.
        """
        if hasattr(self.model.steps[-1][1], 'classes_'):
            is_binary = len(self.model.steps[-1][1].classes_) == 2
        else:
            is_binary = False

        binary_suffix = '_binary' if is_binary else ''
        filename = f"{self.name}{self.label}{binary_suffix}.pkl"

        model_path = os.path.join(self.dir, filename)
        with open(model_path, 'wb') as f:
            pickle.dump({
                'name': self.name,
                'dir': self.dir,
                'label': self.label,
                'model': self.model,
            }, f)


    def load_model(self, filename: str) -> None:
        """
        Loads a pipeline model from a file.

        Args:
            filename (str): The name of the file to load from the model directory.

        Raises:
            FileNotFoundError: If the specified model file does not exist.
        """
        model_path = os.path.join(self.dir, filename)
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                info = pickle.load(f)

                self.name = info['name']
                self.dir = info['dir']
                self.label = info['label']
                self.model = info['model']
        else:
            raise FileNotFoundError(f"No model file found at {model_path}")
