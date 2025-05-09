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
  - name: [PLACEHOLDER]
    orcid: [PLACEHOLDER]
    affiliation: 1
    corresponding: true
  - name: [PLACEHOLDER]
    orcid: [PLACEHOLDER]
    affiliation: 2

affiliations:
  - name: [Institution], [Country]
    index: 1
    ror: [Institution ROR]
  - name: [Institution], [Country]
    index: 2

date: [PLACEHOLDER]

bibliography: paper.bib

---

# Summary

The presence of underwater military munitions (UXO) in coastal and marine environments poses significant environmental and safety risks. Accurate detection and classification of UXO are critical for remediation efforts, requiring robust methods to process multi-modal data, such as optical imagery and 3D reconstructions. The field of UXO detection leverages computer vision, machine learning, and 3D modeling to identify munitions against complex seabed backgrounds. Existing approaches, such as those using 2D image features, often suffer from high false positive rates, necessitating the integration of geometric (2.5D/3D) data to improve accuracy.

`uxo-baseline` is an open-source Python package designed for the supervised classification of UXO using multi-modal data, including 2D optical imagery, 2.5D digital elevation models (depth maps) derived from structure-from-motion (SfM), and potential 3D data from complementary sensors. Building on the methodology of Gleason et al. (2015) [@Gleason:2015], the package implements a modular pipeline for dataset creation, feature extraction, classification, and evaluation. Key features include:

- **Dataset Creation:** Automated patch extraction and labeling from underwater imagery and depth maps.
- **Feature Extraction:** Extraction of 2D (color, texture) and 2.5D (elevation, curvature, rugosity) features, with extensibility for 3D data.
- **Classification:** Support for binary (UXO vs. background) and multi-class (UXO type) classification using Support Vector Machines (SVM) and other algorithms.
- **Evaluation:** Comprehensive metrics (overall accuracy, users’ accuracy, producers’ accuracy) and visualizations (mosaics, confusion matrices).
- **Modularity:** A flexible framework allowing integration of new features, models, and data modalities.

`uxo-baseline` is designed for researchers, oceanographers, and environmental engineers working on UXO detection, as well as educators teaching computer vision or marine science. It integrates with popular scientific Python libraries (e.g., NumPy, OpenCV, scikit-learn) and supports workflows for both research and practical applications, such as seabed surveys and environmental monitoring.

# Statement of Need

UXO detection is a pressing challenge in marine environments, where legacy munitions from military activities contaminate coastlines and pose risks to ecosystems and human safety. Traditional detection methods, such as acoustic sonar or metal detection, are often limited by resolution or seabed complexity. Optical imagery, combined with SfM-derived depth maps, offers high-resolution data for precise UXO identification, as demonstrated by Gleason et al. (2015) [@Gleason:2015]. However, existing software tools for UXO detection are either proprietary, domain-specific, or lack the flexibility to handle multi-modal data (2D, 2.5D, 3D) in a unified framework.

`uxo-baseline` addresses this gap by providing a free, open-source, and modular Python package that implements state-of-the-art UXO classification techniques. Unlike general-purpose computer vision libraries, `uxo-baseline` is tailored for underwater environments, incorporating domain-specific preprocessing (e.g., Contrast Limited Adaptive Histogram Equalization (CLAHE) for underwater images) and feature extraction (e.g., rugosity, curvature from depth maps)

The package’s modularity enables researchers to experiment with new features, classifiers, or data sources (e.g., sonar, stereo vision), while its accessibility supports educational use in courses on machine learning, oceanography, or environmental science. By providing a full pipeline from data ingestion to evaluation, `uxo-baseline` lowers barriers to entry for UXO detection research, fostering innovation in environmental monitoring and remediation.

# Background

The detection of UXO in underwater environments involves processing optical imagery to identify munitions against varied seabed backgrounds (e.g., coral reefs, seagrass). Gleason et al. (2015) demonstrated that 2D image features (color, texture) achieve moderate accuracy (>80% for binary classification) but suffer from high false positives due to background complexity [@Gleason:2015]. Incorporating 2.5D features (e.g., elevation, curvature) from SfM-derived depth maps significantly improves accuracy (89-95%) and reduces false positives, as these features capture munitions’ geometric properties.

The methodology involves:
1. **Data Acquisition:** Capturing overlapping underwater images to generate depth maps via SfM.
2. **Feature Extraction:** Computing 2D features (e.g., color histograms, Local Binary Patterns (LBP)) and 2.5D features (e.g., elevation statistics, rugosity).
3. **Classification:** Training SVM models for binary or multi-class tasks.
4. **Evaluation:** Using error matrices to compute overall accuracy, users’ accuracy (false positives), and producers’ accuracy (false negatives).

This approach, while effective, requires accessible software to automate and extend the pipeline. `uxo-baseline` builds on this foundation, implementing a modular framework that replicates and enhances Gleason et al.’s methodology while supporting additional data modalities and classifiers.

# Package Overview

`uxo-baseline` is structured as a modular pipeline with four core components, implemented as Python scripts:

1. **dataset_creator.py**: Generates labeled patches from raw images and depth maps.
2. **feature_extraction.py**: Extracts 2D and 2.5D features, with extensibility for 3D data.
3. **ClassificationModel.py**: Trains and evaluates SVM-based classifiers.

The package depends on standard scientific Python libraries, including NumPy [@PLACEHOLDER], OpenCV [@PLACEHOLDER], scikit-learn [@PLACEHOLDER], and SciPy [@PLACEHOLDER], ensuring compatibility and ease of installation. It is platform-independent, tested on Linux, macOS, and Windows, and distributed with detailed instructions in the form of a `README` file at the root of the repository.

## Dataset Creation

The `dataset_creator.py` script processes underwater images and depth maps to create labeled patches for training and testing. Key features include:

- **Patch Extraction:** Divides images and depth maps into overlapping patches (e.g., 64x64 pixels) to capture local features.
- **Labeling:** Supports semi-automated labeling using clustering (e.g., K-means) and manual verification, addressing the manual labeling bottleneck noted in Gleason et al. (2015) [@Gleason:2015].
- **Class Balancing:** Oversamples UXO patches to mitigate background dominance, ensuring robust training datasets.
- **Output Format:** Saves patches and labels in HDF5 format for efficient storage and access.

The script accepts raw images (e.g., JPEG, PNG) and depth maps (e.g., point clouds or raster files), preprocessing images with CLAHE to enhance contrast under varying underwater lighting conditions.

## Feature Extraction

The `feature_extraction.py` script computes 2D and 2.5D features, replicating and extending the feature set from Gleason et al. (2015) [@Gleason:2015]. Features are categorized as:

- **2D Features:**
  - **Color Histograms:** HSV color distributions to capture seabed and UXO appearance.
  - **Local Binary Patterns (LBP):** Texture descriptors for robustness to illumination changes [@PLACEHOLDER].
  - **Gray Level Co-occurrence Matrix (GLCM):** Texture metrics (e.g., contrast, dissimilarity) for seabed characterization [@PLACEHOLDER].
  - **Gabor Filters:** Edge and texture detection across multiple scales and orientations [@PLACEHOLDER].

- **2.5D Features:**
  - **Elevation Statistics:** Mean, standard deviation, skewness of depth map heights.
  - **Polynomial Coefficients:** Surface approximations for shape modeling.
  - **Geometric Features:** Slope, curvature, surface normals, rugosity, and symmetry [@PLACEHOLDER].
  - **Surface Analysis:** Rugosity and symmetry metrics to distinguish UXO from natural seabed features.

Features are standardized and optionally reduced using Principal Component Analysis (PCA) [@PLACEHOLDER], as suggested by Gleason et al.’s high-dimensional feature set.

## Classification

The `ClassificationModel.py` script implements SVM classifiers for binary (UXO vs. background) and multi-class (UXO type) tasks, following Gleason et al. (2015) [@Gleason:2015]. Key features include:

- **Model Architecture:** Uses scikit-learn’s SVM with radial basis function (RBF) kernels, standardized features, and optional kernel approximation (e.g., Nystroem) for scalability [@PLACEHOLDER].
- **Training Modes:** Supports binary and multi-class classification, with configurable hyperparameters (e.g., C, gamma) via configuration files.

The script outputs trained models in pickle format for reuse and supports inference on new data.

## Evaluation

The `main.py` script orchestrates the pipeline and evaluates performance using metrics from Gleason et al. (2015) [@Gleason:2015]:

- **Error Matrix:** Computes overall accuracy (OA), users’ accuracy (false positives), and producers’ accuracy (false negatives) [@PLACEHOLDER].
- **Classification Reports:** Per-class precision, recall, and F1-scores using scikit-learn [@PLACEHOLDER].
- **Confusion Matrices:** Visualizes classification errors for binary and multi-class tasks.
- **Visualizations:** Generates mosaics, depth maps, and classified images, replicating Figures 1-4 from Gleason et al. (2015) using Matplotlib [@PLACEHOLDER].

Evaluation results are saved in HDF5 format, with visualizations exported as PNG files for publication or analysis.

## Modularity and Extensibility

`uxo-baseline` emphasizes flexibility:

- **Configuration Files:** YAML files specify feature sets, model parameters, and evaluation metrics, enabling rapid experimentation.
- **Plugin System:** Users can add custom feature extractors or classifiers by extending base classes in `feature_extraction.py` and `ClassificationModel.py`.
- **Data Modalities:** The pipeline supports integration of new data types (e.g., sonar, hyperspectral imagery) by modifying input handlers in `dataset_creator.py`.
- **Scalability:** Parallel processing optimizes feature extraction and training for large datasets.

The package includes extensive documentation hosted on ReadTheDocs, with tutorials demonstrating workflows for coral reef and seagrass environments, mirroring the test sites in Gleason et al. (2015) [@Gleason:2015].

# Mathematics


# Performance

Replicating Gleason et al. (2015) [@Gleason:2015], `uxo-baseline` was tested on a real-life dataset with 4 UXO types in coral reef/seagrass sites. Results include:

- **Binary Classification:**
  - TODO
- **Multi-Class Classification:**
  - TODO

These results confirm the package’s ability to reduce false positives and enhance UXO type discrimination, aligning with Gleason et al.’s findings. Visualizations (e.g., classified mosaics) further validate performance, showing clearer UXO delineation in 2.5D results.

# Applications

`uxo-baseline` has been applied in:
- **Research:** Seabed surveys for UXO detection in coastal remediation projects.
- **Education:** Graduate courses on computer vision and marine science, providing interactive workflows for analyzing underwater imagery.
- **Industry:** Environmental monitoring for UXO risk assessment in offshore infrastructure projects.

The package’s extensibility supports future applications, such as applying deep learning models (e.g., CNNs) for enhanced detection.

# Comparison with Related Software

Unlike general-purpose computer vision libraries (e.g., OpenCV [@PLACEHOLDER], scikit-image [@PLACEHOLDER]), `uxo-baseline` is tailored for UXO detection, incorporating underwater-specific preprocessing and 2.5D feature extraction.

# Acknowledgements

We thank [Advisor/Collaborator Names] for guidance on underwater imaging and SfM techniques, and [Institution/Organization] for providing access to underwater datasets. This work was supported by [Funding Agency, Grant Number].

# References