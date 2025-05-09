import numpy as np
import os, pickle, datetime

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem


class _ClassificationModel:
    def __init__(self, model):
        self.model = model

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class ClassificationModel:
    def __init__(self, model, label: str = '', model_dir: str = './models/', model_name: str = '', standardize: bool = True, pca: bool = True, kernel_mapping: bool = True, n_components: int = 100):
        self.label = label
        self.name = model_name

        self.dir = os.path.join(model_dir, self.name)
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


    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train the SVM model on the provided data.

        Args:
            X_train (np.array): Training features.
            y_train (np.array): Training labels.
        """
        return self.model.fit_transform(X_train, y_train)


    def evaluate(self, X: np.ndarray):
        """
        Evaluate the SVM model on the given data.

        Args:
            X (np.array): Input features.

        Returns:
            np.array: Prediction for y vector
        """
        return self.model.transform(X)


    def save_model(self) -> None:
        """
        Save the trained model to a file.

        Args:
            filename: Path to save the model file.
        """
        binary = '_binary' if len(self.model.classes_) == 2 else ''
        filename = f"{self.name}{self.label}{binary}.pkl"

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
        Load a model from a file.

        Args:
            filename: Path to load the model file.
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
