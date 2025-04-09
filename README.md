# Baseline Model for Underwater Military Munitions (UWMM) Detection

This project implements a baseline model for the detection of Underwater Military Munitions (UWMM), replicating the methodology presented in the paper "Improved supervised classification of underwater military munitions using height features derived from optical imagery" by Gleason et al. (2015). The implementation utilizes Python frameworks to process and analyze different data modalities, including 2D imagery, 3D depth data, and a combined 2.5D representation, to evaluate their effectiveness in identifying Unexploded Ordnance (UXO) in underwater environments.

## Purpose

The primary objectives of this project are to:

* Replicate the findings of Gleason et al. (2015) using Python-based tools.
* Compare the performance of SVM models trained on 2D image data, 3D depth data, and combined 2.5D data for UWMM detection.
* Establish a modular framework for building datasets, training classification models, and conducting inference for UWMM detection tasks.

## Key Features

* **Dataset Generation:** Processes tiled raw image, depth, and mask data to create training patches labeled as 'uxo' or 'background'.
* **Multi-Modal Data Handling:** Supports the use of 2D imagery, 3D depth data, and combined 2.5D data for model training and evaluation.
* **SVM Classification:** Implements SVM models for classifying potential UWMM based on extracted features.
* **Trainable Models:** Provides functionality to train classification models on the generated dataset.
* **Inference Pipeline:** Enables the application of trained models to new underwater imagery for UXO detection.
* **Configuration-Driven Workflow:** Utilizes a `config.yaml` file to manage project settings and workflow execution.

## Getting Started

This section outlines the steps required to use this project. Ensure you have the necessary Python environment and dependencies installed.

### 1. Dataset Creation

This step involves processing raw tiled data to generate training patches.

**Input Data Format:**

The project expects raw data to be organized as tiles within a directory structure specified by `tiles_dir` in the `config.yaml` file. Within each dataset folder under `tiles_dir`, the following subdirectories are expected:

* `images`: Contains tiled 2D imagery.
* `depths`: Contains corresponding tiled depth maps.
* `masks`: Contains corresponding masks indicating the location of potential UXOs.

**File Naming Convention:**

Image files within each subdirectory should have consistent naming patterns that allow for matching corresponding tiles across different modalities. The filename is expected to end with the row and column index of the tile within the larger mosaic (e.g., `plot1_18_240424_t2_ortho_r00_c01.png`).

**Configuration:**

1.  Navigate to the project directory.
2.  Open the `config.yaml` file.
3.  Under the `create_dataset` section, set the `enabled` flag to `true`.
4.  Configure other parameters within the `create_dataset` section, such as the `tiles_dir` and `dataset_dir`, according to your data organization.

**Output:**

Processed image patches (both 2D and 3D representations) will be saved in the `dataset_dir` under subfolders labeled 'uxo' and 'background'.

### 2. Model Training

This step focuses on extracting features from the generated dataset patches and training an SVM classification model.

**Input:**

The training process utilizes the dataset created in the previous step, located in the `dataset_dir`. Optionally, if `use_saved_features` is set to `true` in the `config.yaml`, pre-extracted features from the `features_dir` will be loaded.

**Configuration:**

1.  Open the `config.yaml` file.
2.  Under the `train_model` section, set the `enabled` flag to `true`.
3.  Configure parameters within the `train_model` section, paying particular attention to the `dimension` parameter, which should be set to `'2'`, `'3'`, or `'25'` to specify the data modality for training.

**Output:**

* Extracted features (for 2D, 3D, or 2.5D data) will be saved in the `features_dir`.
* The trained SVM model will be saved as a file in the `models_dir`.
* A classification report summarizing the model's performance will be saved in the `results_dir`.

### 3. Inference

This step involves using a trained model to detect potential UXOs in a new, unseen image.

**Input:**

The inference process requires the following inputs, specified in the `config.yaml` file under the `run_inference` section:

* `image_path`: The path to the new 2D image.
* `depth_path`: The path to the corresponding depth map.
* `model_name`: The filename of the trained model located in the `models_dir`.

**Configuration:**

1.  Open the `config.yaml` file.
2.  Under the `run_inference` section, set the `enabled` flag to `true`.
3.  Specify the `image_path`, `depth_path`, `model_name`, and the desired `output_file` name. Configure any other relevant inference parameters as needed.

**Output:**

An output image with detected UXOs highlighted will be saved in the `results_dir` with the filename specified by the `output_file` parameter.

## Results

The experimental results, consistent with the findings of Gleason et al. (2015), highlight the varying performance of models trained on different data modalities:

* **2D Model:** Models trained solely on 2D imagery exhibited a tendency to misclassify non-UXO objects, such as scales placed in the scene for measurement or rusted chains lying on the sea floor, as UXOs.
* **3D Model:** Models trained exclusively on 3D depth information demonstrated a high false positive rate, often identifying structures with similar shapes to UXOs as potential targets. These models also struggled with accurately classifying actual UXOs in some instances.
* **2.5D Model:** The model trained on the combined 2.5D data (integrating both 2D and 3D information) achieved the best overall performance. It significantly reduced the number of false positives compared to the 2D and 3D models. However, the 2.5D model did not perfectly capture and classify all UXOs present in the test data.

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
    <td><img src="assets/2D_plot1_r02_c05.png" alt="2D Result 1"></td>
    <td><img src="assets/3D_plot1_r02_c05.png" alt="3D Result 1"></td>
    <td><img src="assets/25D_plot1_r02_c05.png" alt="2.5D Result 1"></td>
  </tr>
 <tr>
    <td><img src="assets/2D_plot1_r03_c03.png" alt="2D Result 2"></td>
    <td><img src="assets/3D_plot1_r03_c03.png" alt="3D Result 2"></td>
    <td><img src="assets/25D_plot1_r03_c03.png" alt="2.5D Result 2"></td>
  </tr>
  <tr>
    <td><img src="assets/2D_plot3_r03_c05.png" alt="2D Result 3"></td>
    <td><img src="assets/3D_plot3_r03_c05.png" alt="3D Result 3"></td>
    <td><img src="assets/25D_plot3_r03_c05.png" alt="2.5D Result 3"></td>
  </tr>
  <tr>
    <td><img src="assets/2D_plot3_r04_c04.png" alt="2D Result 4"></td>
    <td><img src="assets/3D_plot3_r04_c04.png" alt="3D Result 4"></td>
    <td><img src="assets/25D_plot3_r04_c04.png" alt="2.5D Result 4"></td>
  </tr>
</table>

## References

* A. C. R. Gleason, A. Shihavuddin, N. Gracias, G. Schultz and B. E. Gintert, "Improved supervised classification of underwater military munitions using height features derived from optical imagery," OCEANS 2015 - MTS/IEEE Washington, Washington, DC, USA, 2015, pp. 1-5, doi: [10.23919/OCEANS.2015.7404580](https://doi.org/10.23919/OCEANS.2015.7404580).
* Shihavuddin, A. S. M., Gracias, N., Garcia, R., Gleason, A. C. R., & Gintert, B. (2013). Image-Based Coral Reef Classification and Thematic Mapping. *Remote Sensing*, *5*(4), 1809-1841. [https://doi.org/10.3390/rs5041809](https://doi.org/10.3390/rs5041809)
* Shihavuddin, Asm, Gracias, Nuno, Garcia, Rafael, Campos, Ricard, Gleason, Arthur, & Gintert, Brooke. (2014). Automated Detection of Underwater Military Munitions Using Fusion of 2D and 2.5D Features From Optical Imagery. *Marine Technology Society Journal*, *48*(4), 7-17. [https://doi.org/10.4031/MTSJ.48.4.7](https://doi.org/10.4031/MTSJ.48.4.7)
