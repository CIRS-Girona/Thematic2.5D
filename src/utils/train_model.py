from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt
from typing import Literal
import datetime, os

from utils import load_features, save_features
from classification import SVMModel


def train_model(dataset_dir, features_dir, models_dir, results_dir, uxo_start_code, binary_mode=False, test_size=0.1, n_components: int = 100, dimension: Literal['2', '25', '3'] = '25', use_saved_features=True, subset_size=0):
    if not use_saved_features:
        save_features(f"{dataset_dir}/2D/", f"{dataset_dir}/3D/", features_dir, subset=subset_size)

    # Load features and encode labels
    X_data, y_data = load_features(features_dir, dimension=dimension)

    if binary_mode:
        for y in np.unique(y_data):
            if y.isdigit() and int(y) >= uxo_start_code:
                y_data[y == y_data] = 'uxo'
            else:
                y_data[y == y_data] = 'background'

    print(f"Training start time: {datetime.datetime.now().isoformat()}")

    # Transform and split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, stratify=y_data, test_size=test_size)

    # Train model on full dataset and save it
    model = SVMModel(label=dimension, model_dir=models_dir, n_components=n_components)
    model.train(X_train, y_train)
    model.save_model()

    # Evaluate on the test set
    y_pred = model.evaluate(X_test)

    # Save the classification report to a file
    if not os.path.exists(results_dir) or not os.path.isdir(results_dir):
        os.mkdir(results_dir)

    if binary_mode:
        datafile_name = f"{model.name}_{dimension}D_binary"
    else:
        datafile_name = f"{model.name}_{dimension}D"

    with open(f"{results_dir}/{datafile_name}.txt", 'w') as f:
        print(classification_report(y_test, y_pred, zero_division=0))
        print(classification_report(y_test, y_pred, zero_division=0), file=f)

    cm = confusion_matrix(y_test, y_pred, normalize='true')
    cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.model.classes_)

    cmd.plot()
    plt.tight_layout()
    plt.savefig(f"{results_dir}/{datafile_name}.png")
    plt.close('all')