---
title: 'uxo-baseline: A Python Package for Supervised Classification of Underwater Military Munitions Using Multi-Modal Data'

tags:
  - Python
  - oceanography
  - machine learning
  - underwater munitions
  - computer vision
  - structure-from-motion

authors:
  - name: [Name]
    orcid: [ORCID]
    affiliation: 1
    corresponding: true
  - name: [Co-Author Name]
    orcid: [Co-Author ORCID]
    affiliation: 2

affiliations:
  - name: [Institution], [Country]
    index: 1
    ror: [Institution ROR]
  - name: [Co-Author Institution], [Country]
    index: 2

date: 09 May 2025

bibliography: paper.bib

---

# Summary

The presence of underwater military munitions (UXO) in coastal and marine environments poses significant environmental and safety risks. Accurate detection and classification of UXO are critical for remediation efforts, requiring robust methods to process multi-modal data, such as optical imagery and 3D reconstructions. UXO detection leverages computer vision, machine learning, and 3D modeling to identify munitions against complex seabed backgrounds. Existing approaches, such as those using 2D image features, often suffer from high false positive rates, necessitating the integration of geometric (2.5D/3D) data to improve accuracy.

`uxo-baseline` is an open-source Python package designed for the supervised classification of UXOs using multi-modal data, including 2D optical imagery and 2.5D digital elevation models (DEMs) derived from 3D reconstructions of the scene. Building on the methodology of Gleason et al. (2015) [@Gleason:2015], the package implements a modular pipeline for dataset creation, feature extraction, classification, and evaluation. Key features include:

- **Dataset Creation:** Automated patch extraction and labeling from underwater imagery and DEMs.
- **Feature Extraction:** Extraction of 2D (color, texture) and 2.5D (elevation, curvature, rugosity) features, with extensibility for 3D data.
- **Classification:** Support for binary (UXO vs. background) and multi-class (UXO type) classification using Support Vector Machines (SVM) and other algorithms.
- **Evaluation:** Comprehensive metrics (overall accuracy, users’ accuracy, producers’ accuracy) and visualizations (mosaics, confusion matrices).
- **Modularity:** A flexible framework allowing integration of new features, models, and data modalities.

`uxo-baseline` is designed for researchers, oceanographers, and environmental engineers working on UXO detection, as well as educators teaching computer vision or marine science. It integrates with popular scientific Python libraries (e.g., NumPy, OpenCV, scikit-learn) and supports workflows for both research and practical applications, such as seabed surveys and environmental monitoring.

# Statement of Need

UXO detection is a pressing challenge in marine environments, where legacy munitions from military activities contaminate coastlines and pose risks to ecosystems and human safety. Traditional detection methods, such as acoustic sonar or traditional 2D optical imagery, are often limited by resolution or seabed complexity. Optical imagery, combined with SfM-derived DEMs, offers high-resolution data for precise UXO identification, as demonstrated by Gleason et al. (2015) [@Gleason:2015]. However, existing software tools for UXO detection are either proprietary, domain-specific, or lack the flexibility to handle multi-modal data (2D, 2.5D, 3D) in a unified framework.

`uxo-baseline` addresses this gap by providing a free, open-source, and modular Python package that implements state-of-the-art UXO classification techniques. Unlike general-purpose computer vision libraries (e.g., OpenCV [@opencv_library]), `uxo-baseline` is tailored for underwater environments, incorporating domain-specific preprocessing (e.g., Contrast Limited Adaptive Histogram Equalization (CLAHE) for underwater images) and feature extraction (e.g., rugosity, curvature from DEMs).

The package’s modularity enables researchers to experiment with new features, classifiers, or data sources (e.g., sonar, stereo vision), while its accessibility supports educational use in courses on machine learning, oceanography, or environmental science. By providing a full pipeline from data ingestion to evaluation, `uxo-baseline` lowers barriers to entry for UXO detection research, fostering innovation in environmental monitoring and remediation.

# Background

The detection of UXO in underwater environments involves processing optical imagery to identify munitions against varied seabed backgrounds (e.g., coral reefs, seagrass). Gleason et al. (2015) demonstrated that 2D image features (color, texture) achieve moderate accuracy (>80% for binary classification) but suffer from high false positives due to background complexity [@Gleason:2015]. Incorporating 2.5D features (e.g., elevation, curvature) from SfM-derived DEMs significantly improves accuracy (89-95%) and reduces false positives, as these features capture munitions’ geometric properties.

The methodology involves:
1. **Data Acquisition:** Capturing overlapping underwater images to generate DEMs via SfM.
2. **Feature Extraction:** Computing 2D features (e.g., color histograms, Local Binary Patterns (LBP)) and 2.5D features (e.g., elevation statistics, rugosity).
3. **Classification:** Training SVM models for binary or multi-class tasks.
4. **Evaluation:** Using error matrices to compute overall accuracy, users’ accuracy (false positives), and producers’ accuracy (false negatives).

This approach, while effective, requires accessible software to automate and extend the pipeline. `uxo-baseline` builds on this foundation, implementing a modular framework that replicates and enhances Gleason et al.’s methodology while supporting additional data modalities and classifiers.

# Package Overview

`uxo-baseline` is structured as a modular pipeline with four core components, implemented as Python scripts:

1. **dataset_creator.py**: Generates labeled patches from raw images and DEMs.
2. **feature_extraction.py**: Extracts 2D and 2.5D features, with extensibility for 3D data.
3. **ClassificationModel.py**: Trains and evaluates SVM-based classifiers.
4. **main.py**: Orchestrates the pipeline, supporting configuration-driven workflows.

Flexibility and ease-of-use are emphasized throughout `uxo-baseline` codebase:

- **Configuration Files:** YAML files specify feature sets, model parameters, and evaluation metrics, enabling rapid experimentation.
- **Plugin System:** Users can add custom feature extractors or classifiers by extending base classes in `feature_extraction.py` and `ClassificationModel.py`.
- **Data Modalities:** The pipeline supports integration of new data types (e.g., sonar, hyperspectral imagery) by modifying input handlers in `dataset_creator.py`.
- **Scalability:** Parallel processing optimizes feature extraction and training for large datasets.

The package depends on standard scientific Python libraries, including NumPy [@numpy_library], OpenCV [@opencv_library], scikit-learn [@scikit-learn_library], scikit-image [@scikit-image_library], and SciPy [@scipy_library], ensuring compatibility and ease of installation.

## Dataset Creation

The `dataset_creator.py` script processes underwater images and DEMs to create labeled patches for training and testing. Key features include:

- **Patch Extraction:** Divides images and DEMs into overlapping patches (e.g., 128x128 pixels) to capture local features.
- **Class Balancing:** Oversamples UXO patches to mitigate background dominance, ensuring robust training datasets.

The script accepts raw images (e.g., JPEG, PNG) and depth maps stored in PNG format as 16-bit unsigned integers.

## Feature Extraction

The `feature_extraction.py` script computes 2D and 2.5D features, replicating and extending the feature set from Gleason et al. (2015) [@Gleason:2015]. Features are categorized as:

- **2D Features:**
  - **Color Histograms:** HSV color distributions to capture seabed and UXO appearance.
  - **Local Binary Patterns (LBP):** Texture descriptors for robustness to illumination changes [@lbp_algorithm].
  - **Gray Level Co-occurrence Matrix (GLCM):** Texture metrics (e.g., contrast, dissimilarity) for seabed characterization [@glcm_algorithm].
  - **Gabor Filters:** Edge and texture detection across multiple scales and orientations [@gabor_algorithm].

- **2.5D Features:**
  - **Elevation Statistics:** Mean, standard deviation, skewness of DEM heights.
  - **Polynomial Coefficients:** Surface approximations for shape modeling.
  - **Geometric Features:** Slope, curvature, surface normals, rugosity, and symmetry.
  - **Surface Analysis:** Rugosity and symmetry metrics to distinguish UXO from natural seabed features.

Features are standardized and optionally reduced using Principal Component Analysis (PCA) [@pca_algorithm], as suggested by Gleason et al.’s high-dimensional feature set.

## Classification

The `ClassificationModel.py` script implements SVM classifiers for binary (UXO vs. background) and multi-class (UXO type) tasks, following Gleason et al. (2015) [@Gleason:2015]. Key features include:

- **Model Architecture:** Uses scikit-learn’s SVM with radial basis function (RBF) kernels, standardized features, and optional kernel approximation (e.g., Nystroem) for scalability [@scikit-learn_library].
- **Training Modes:** Supports binary and multi-class classification, with configurable hyperparameters (e.g., C, gamma) via configuration files.
- **Alternative Classifiers:** Extensible to include Random Forests or Gradient Boosting, enabling experimentation.
- **Cross-Validation:** Implements k-fold cross-validation to ensure robust performance across seabed types.

The script outputs trained models in pickle format for reuse and supports inference on new data.

## Evaluation

The `main.py` script orchestrates the pipeline and evaluates performance using metrics from Gleason et al. (2015) [@Gleason:2015]:

- **Error Matrix:** Computes overall accuracy (OA), users’ accuracy (false positives), and producers’ accuracy (false negatives).
- **Classification Reports:** Per-class precision, recall, and F1-scores using scikit-learn [@scikit-learn_library].
- **Confusion Matrices:** Visualizes classification errors for binary and multi-class tasks.

Evaluation results are saved in TXT format, with visualizations exported as PNG files for publication or analysis.

# Mathematics

The classification pipeline in `uxo-baseline` relies on feature extraction and SVM classification. For a patch \( \mathbf{x} \in \mathbb{R}^{m \times n} \) (image) and corresponding DEM \( \mathbf{d} \in \mathbb{R}^{m \times n} \), the feature vector \( \mathbf{f} \) combines 2D and 2.5D features:

\[
\mathbf{f} = [\mathbf{f}_{2D}, \mathbf{f}_{2.5D}]
\]

where:
- \( \mathbf{f}_{2D} \): Color histograms, LBP, GLCM, and Gabor filter responses.
- \( \mathbf{f}_{2.5D} \): Elevation statistics, polynomial coefficients, slope, curvature, rugosity, etc.

The SVM classifier solves the optimization problem:

\[
\min_{\mathbf{w}, b, \xi} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^N \xi_i
\]

subject to:

\[
y_i (\mathbf{w}^T \phi(\mathbf{f}_i) + b) \geq 1 - \xi_i, \quad \xi_i \geq 0, \quad i = 1, \ldots, N
\]

where \( \mathbf{w} \) is the weight vector, \( b \) is the bias, \( \xi_i \) are slack variables, \( C \) is the regularization parameter, \( \phi \) is the kernel mapping (RBF), and \( y_i \in \{-1, 1\} \) (binary) or \( y_i \in \{1, \ldots, K\} \) (multi-class) are labels.

For multi-class classification, `uxo-baseline` uses a one-vs-rest strategy, training \( K \) binary classifiers for \( K \) UXO types.

# Applications

`uxo-baseline` can be applied in:
- **Research:** Seabed surveys for UXO detection in coastal remediation projects.
- **Education:** Graduate courses on computer vision and marine science, providing interactive workflows for analyzing underwater imagery.
- **Industry:** Environmental monitoring for UXO risk assessment in offshore infrastructure projects.

The package’s extensibility supports future applications, such as integrating sonar data or applying deep learning models (e.g., CNNs) for enhanced detection.

# Acknowledgements

We thank [Advisor/Collaborator Names] for guidance on underwater imaging and SfM techniques, and [Institution/Organization] for providing access to underwater datasets. This work was supported by [Funding Agency, Grant Number].

# References
