from sklearn.svm import LinearSVC
from typing import Literal, Optional, Union

from .ClassificationModel import ClassificationModel


class SVMModel(ClassificationModel):
    """
    A specific implementation of ClassificationModel for Linear Support Vector Classification (LinearSVC).
    Initializes the pipeline with LinearSVC and standard preprocessing steps.
    """
    def __init__(self, C: float = 1.0, class_weight: Optional[Union[dict, Literal['balanced']]] = 'balanced', model_dir: str = './models/', label: str = '', standardize: bool = True, pca: bool = True, kernel_mapping: bool = True, n_components: int = 100):
        """
        Initializes the SVMModel with a LinearSVC classifier and pipeline options.

        Args:
            C (float): Regularization parameter. The strength of the regularization is
                       inversely proportional to C. Must be strictly positive.
            class_weight (Optional[Union[dict, Literal['balanced']]]): Weights associated with classes.
                                  If 'balanced', class weights will be inversely proportional
                                  to the frequency of samples in the training data.
            model_dir (str): Directory to save/load models.
            label (str): An optional label for the model file name.
            standardize (bool): Whether to include StandardScaler in the pipeline.
            pca (bool): Whether to include PCA in the pipeline.
            kernel_mapping (bool): Whether to include Nystroem kernel approximation in the pipeline.
            n_components (int): Number of components for Nystroem and PCA if used.
        """
        model = LinearSVC(C=C, class_weight=class_weight)

        super().__init__(
            model,
            model_dir=model_dir,
            model_name='SVM',
            label=label,
            standardize=standardize,
            pca=pca,
            kernel_mapping=kernel_mapping,
            n_components=n_components
        )