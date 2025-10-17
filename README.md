# Thematic2.5D: A Toolkit for Evaluating 2D and 3D Feature Effects in Supervised Classification

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://docs.python.org/3/whatsnew/3.12.html)
[![Unit Tests](https://github.com/CIRS-Girona/Thematic2.5D/actions/workflows/python-app.yml/badge.svg)](https://github.com/CIRS-Girona/Thematic2.5D/actions/workflows/python-app.yml)

This project implements a modular toolkit for supervised classification and thematic mapping, inspired by the methodology presented in the paper "Improved supervised classification of underwater military munitions using height features derived from optical imagery" by Gleason et al. (2015). The package processes and analyzes multi-modal data, including optical imagery (2D), geometric data (3D), and a combined 2.5D representation, to evaluate the effectiveness of different feature modalities in classifying objects in complex environments.

## Purpose

The primary objectives of this project are to:

* Replicate the findings of Gleason et al. (2015) using Python-based tools.
* Provide a flexible framework for supervised classification and thematic mapping using multi-modal data.
* Compare the performance of SVM models trained on 2D-derived features (color, texture), 3D-derived features (curvature, rugosity), and combined optical and depth features (2.5D) to assess their relative contributions to classification accuracy.
* Establish a modular framework for building datasets, training classification models, and conducting inference tasks.

## Key Features

* **Dataset Generation:** Processes original image, depth, and mask data to create training patches.
* **Multi-Modal Data Handling:** Supports optical imagery and depth information for model training and evaluation.
* **SVM Classification:** Implements SVM models for classifying objects based on extracted features.
* **Trainable Models:** Provides functionality to train classification models on generated datasets.
* **Inference Pipeline:** Enables the application of trained models to new imagery for object detection and thematic mapping.

## Getting Started

This section outlines the steps required to use this project. Ensure you have the necessary Python environment and dependencies installed.

### 0. Installation and Testing

This step provides instruction on how to install the project and test the models on the given samples.

**Setting Up the Project:**

It is recommended that a virtual environment is used when running the pipeline. The following is one approach to setup the project using Python's `venv` environment:

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

**Testing the Pipeline:**

A testing script along with data samples organized in the required format are provided in the `tests/` directory. Before running the `test.py` script, please ensure that all the paths found in the config file are pointing correctly to the provided sample dataset and that the object codes are unchanged. The default config file is already setup to be run using the `test.py` script from the get-go.

Please make sure that the current working directory is the root directory of the repository before running the `test.py` script. The script can be run using the following command:

```bash
python tests/test.py
```

### 1. Dataset Creation

This step involves processing the original image/depth/mask data to generate training patches.

**Input Data Format:**

The project expects original data to be organized as images within a directory structure specified by `input_dir` in the `config.yaml` file. The `input_dir` directory is expected to host one or more datasets organized as folders. Within each dataset folder under `input_dir`, the following subdirectories are expected:

* `images`: Contains original 2D imagery.
* `depths`: Contains corresponding depth maps, formatted as 1-channel, 16-bit PNGs.
* `masks`: Contains corresponding masks indicating the location of target objects, formatted as 1-channel, 8-bit PNGs.

***Example:***

```
input_dir/
└── dataset_1/
    ├── images/
    │   ├── img_01.jpg
    │   ├── img_02.jpg
    │   └── ...
    ├── depths/
    │   ├── img_01.png
    │   ├── img_02.png
    │   └── ...
    └── masks/
        ├── img_01.png
        ├── img_02.png
        └── ...
└── dataset_2/
    ├── images/
    │   └── ...
    ├── depths/
    │   └── ...
    └── masks/
        └── ...
└── ...
```

**File Naming Convention:**

Data files within each subdirectory should have the same exact name - excluding the extension - to allow for data matching across different modalities.

**Output:**

Processed image patches (both optical and depth representations) will be saved in the `dataset_dir` under subfolders labeled 'background' and the different class names present in the mask.

### 2. Model Training

This step focuses on extracting features from the generated dataset patches and training an SVM classification model.

**Input:**

The training process utilizes the dataset created in the previous step, located in the `dataset_dir`. Optionally, if `use_saved_features` is set to `true` in the `config.yaml`, pre-extracted features from the `features_dir` will be loaded.

**Output:**

* Extracted features (for 2D, 3D, or 2.5D data) will be saved in the `features_dir`.
* The trained SVM model will be saved as a file in the `models_dir`.
* A classification report summarizing the model's performance will be saved in the `results_dir` along with the confusion matrix.

### 3. Inference

This step involves using a trained model to detect objects in a new, unseen image.

**Input:**

The inference process requires the following inputs, specified in the `config.yaml` file under the `run_inference` section:

* `image_path`: The path to the images folder.
* `depth_path`: The path to the corresponding depth maps folder.

**Output:**

A folder will be created for each trained model inside of `results_dir` where the inference image along with the generated mask will be saved.

### 4. Evaluate Results

This step calculates and saves the mean Intersection over Union (mIoU) for each inferred image, providing a quantitative measure of each model's performance.

**Input:**

The evaluation process requires the following inputs, specified in the `config.yaml` file under the `evaluate_results` section:

* `mask_path`: The path to the directory containing the ground truth masks.

**Output:**

The evaluation results will be saved in a file named `meanIoU.txt` located within the directory of each model under `results_dir`. This file will contain the mIoU score for each individual image as well as the average mIoU across all images.

## Results

The experimental results, consistent with the findings of Gleason et al. (2015), highlight the varying performance of models trained on different data modalities for classifying unexploded ordnances (UXOs):

* **2D Model:** Models trained solely on optical imagery exhibited a tendency to misclassify non-UXO objects, such as scales placed in the scene for measurement or rusted chains lying on the sea floor, as UXOs.
* **3D Model:** Models trained exclusively on depth information demonstrated a high false positive rate, often identifying structures with similar shapes to UXOs as potential targets. These models also struggled with accurately classifying actual UXOs in some instances.
* **2.5D Model:** The model trained on the combined 2.5D data (integrating both optical and geometric information) achieved the best overall performance. It significantly reduced the number of false positives compared to the 2D and 3D models. However, the 2.5D model did not perfectly capture and classify all UXOs present in the test data.

The following table provides visual examples of the model outputs for different data modalities on representative image patches:

<p align="center">
  <strong>Comparison of Detection Results Across Data Modalities</strong>
</p>
<table style="width:100%; text-align: center;">
  <tr>
    <th style="text-align: center;">Only 2D</th>
    <th style="text-align: center;">Only 3D</th>
    <th style="text-align: center;">2.5D</th>
  </tr>
  <tr>
    <td><img src="assets/2D_plot1_r02_c05.jpg" alt="2D Result 1"></td>
    <td><img src="assets/3D_plot1_r02_c05.jpg" alt="3D Result 1"></td>
    <td><img src="assets/25D_plot1_r02_c05.jpg" alt="2.5D Result 1"></td>
  </tr>
 <tr>
    <td><img src="assets/2D_plot1_r03_c03.jpg" alt="2D Result 2"></td>
    <td><img src="assets/3D_plot1_r03_c03.jpg" alt="3D Result 2"></td>
    <td><img src="assets/25D_plot1_r03_c03.jpg" alt="2.5D Result 2"></td>
  </tr>
  <tr>
    <td><img src="assets/2D_plot3_r03_c05.jpg" alt="2D Result 3"></td>
    <td><img src="assets/3D_plot3_r03_c05.jpg" alt="3D Result 3"></td>
    <td><img src="assets/25D_plot3_r03_c05.jpg" alt="2.5D Result 3"></td>
  </tr>
  <tr>
    <td><img src="assets/2D_plot3_r04_c04.jpg" alt="2D Result 4"></td>
    <td><img src="assets/3D_plot3_r04_c04.jpg" alt="3D Result 4"></td>
    <td><img src="assets/25D_plot3_r04_c04.jpg" alt="2.5D Result 4"></td>
  </tr>
</table>

## How to Contribute

We welcome contributions to this project! If you'd like to help improve Thematic2.5D, please follow these steps:

1.  **Open a Ticket/Issue** in the repository describing the changes or fixes you plan to make. This helps coordinate efforts and track the development process.
2.  **Fork** the repository and create a new branch for your feature or fix.
3.  **Make your changes** and ensure the code adheres to the existing style and conventions.
4.  **Write and run unit tests** to cover your changes.
5.  **Open a Pull Request (PR)** to the main repository's `main` branch.

In your pull request, please provide a clear summary of your changes and any relevant context. This will help us review and merge your contribution quickly.

## References

* Shihavuddin, A. S. M., Gracias, N., Garcia, R., Gleason, A. C. R., & Gintert, B. (2013). Image-Based Coral Reef Classification and Thematic Mapping. *Remote Sensing*, *5*(4), 1809-1841. [https://doi.org/10.3390/rs5041809](https://doi.org/10.3390/rs5041809)
* Shihavuddin, Asm, Gracias, Nuno, Garcia, Rafael, Campos, Ricard, Gleason, Arthur, & Gintert, Brooke. (2014). Automated Detection of Underwater Military Munitions Using Fusion of 2D and 2.5D Features From Optical Imagery. *Marine Technology Society Journal*, *48*(4), 7-17. [https://doi.org/10.4031/MTSJ.48.4.7](https://doi.org/10.4031/MTSJ.48.4.7)
* A. C. R. Gleason, A. Shihavuddin, N. Gracias, G. Schultz and B. E. Gintert, "Improved supervised classification of underwater military munitions using height features derived from optical imagery," OCEANS 2015 - MTS/IEEE Washington, Washington, DC, USA, 2015, pp. 1-5, doi: [10.23919/OCEANS.2015.7404580](https://doi.org/10.23919/OCEANS.2015.7404580).
