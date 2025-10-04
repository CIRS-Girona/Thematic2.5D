import os, yaml, subprocess

# Define constants for model dimensions and sample names
MODELS = ('2', '3', '25')
SAMPLES = ('18mm', '24mm', 'GPS')

# Define directory paths relative to the TEST_DIR
TEST_DIR = "tests"
SAMPLE_DIR = f"{TEST_DIR}/samples"
DATASET_DIR = f"{TEST_DIR}/data/dataset"
FEATURES_DIR = f"{TEST_DIR}/data/features"
MODELS_DIR = f"{TEST_DIR}/data/models"
RESULTS_DIR = f"{TEST_DIR}/data/results"


def run_pipeline(conf: dict) -> None:
    """
    Writes the provided configuration dictionary to 'config.yaml' and
    executes the main pipeline script 'main.py'.

    Args:
        conf (dict): The configuration dictionary for the pipeline.
    """
    # Write the updated configuration to config.yaml
    with open("config.yaml", 'w') as f:
        yaml.safe_dump(conf, f)

    # Execute the main pipeline script
    # The subprocess call assumes 'main.py' is in the current directory
    subprocess.run(["python", "main.py"])


def data_check(data_path: str, check_dir: bool = False, check_empty: bool = False) -> None:
    """
    Performs integrity checks on a given file or directory path.

    It asserts that the path exists, checks if it's the expected file/directory
    type, and optionally checks if a directory is non-empty.

    Args:
        data_path (str): The path to the file or directory to check.
        check_dir (bool): If True, asserts the path is a directory.
                          If False, asserts the path is a file. Defaults to False.
        check_empty (bool): If True and check_dir is True, asserts the directory
                            contains at least one item. Defaults to False.
    """
    # Check if data exists
    assert os.path.exists(data_path)

    # Check that it is the correct file type (file or directory)
    if check_dir:
        assert os.path.isdir(data_path)
    else:
        assert os.path.isfile(data_path)

    # If it is a directory, check if it is empty (if check_empty is True)
    if check_empty and check_dir:
        assert len(os.listdir(data_path)) > 0


if __name__ == "__main__":
    # --- Start of Pipeline Test Execution ---

    # 1. Initial Run: Create Dataset, Extract Features, and Train a Base Model (25D)

    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Enable steps for dataset creation and model training
    config['create_dataset']['enabled'] = True
    config['train_model']['enabled'] = True
    # Disable inference and evaluation for this initial run
    config['run_inference']['enabled'] = False
    config['evaluate_results']['enabled'] = False

    # Set parameters for the initial training run
    config['train_model']['dimension'] = '25'
    # Features will be calculated and saved during this run
    config['train_model']['use_saved_features'] = False

    print("--- Running initial pipeline: Create Dataset, Extract Features, Train 25D SVM ---")
    run_pipeline(config)

    # Verify that the expected data directories and files were created and are not empty
    data_check(f"{DATASET_DIR}/2D", check_dir=True, check_empty=True)
    data_check(f"{DATASET_DIR}/3D", check_dir=True, check_empty=True)
    data_check(FEATURES_DIR, check_dir=True, check_empty=True)  # Check for saved features
    data_check(MODELS_DIR, check_dir=True, check_empty=True)    # Check for saved model
    data_check(RESULTS_DIR, check_dir=True)                     # Results dir might be empty/have logs

    # 2. Train Different SVM Models using the Pre-Calculated Features from the previous run

    # Disable dataset creation
    config['create_dataset']['enabled'] = False
    # Use the features saved from the previous run
    config['train_model']['use_saved_features'] = True

    # Iterate through all defined model dimensions (2D, 3D, 25D)
    for model in MODELS:
        # Iterate through both binary and multi-class modes
        for binary in (True, False):
            # Update configuration for the current training run
            config['train_model']['dimension'] = model
            config['train_model']['binary_mode'] = binary

            run_pipeline(config)

            # Determine the expected output file name based on binary mode
            file_name = f"SVM{model}_binary" if binary else f"SVM{model}"

            # Check for the resulting classification plot and metrics/config file
            data_check(f"{RESULTS_DIR}/{file_name}.jpg")
            data_check(f"{RESULTS_DIR}/{file_name}.txt")

    # 3. Run Inference and Evaluation on Different Samples

    # Disable model training
    config['train_model']['enabled'] = False
    # Enable inference and evaluation for the loop
    config['run_inference']['enabled'] = True
    config['evaluate_results']['enabled'] = True

    # Iterate through all defined sample sets
    for sample in SAMPLES:
        # Configure the input paths for the current sample
        config['run_inference']['image_path'] = f"{SAMPLE_DIR}/{sample}/images"
        config['run_inference']['depth_path'] = f"{SAMPLE_DIR}/{sample}/depths"

        # Configure the ground truth path for evaluation
        config['evaluate_results']['mask_path'] = f"{SAMPLE_DIR}/{sample}/masks"

        run_pipeline(config)

    # 4. Final Integrity Check for Evaluation Results

    # Verify that performance metrics and per-sample results were generated
    for model in MODELS:
        for binary in (True, False):
            file_name = f"SVM{model}_binary" if binary else f"SVM{model}"

            # Check for the Mean Intersection over Union (mIoU) summary file
            data_check(f"{RESULTS_DIR}/{file_name}-mIoU.txt")
            # Check for the directory containing results/visualizations for each sample
            data_check(f"{RESULTS_DIR}/{file_name}", check_dir=True, check_empty=True)

    print("--- All pipeline tests and data checks completed successfully. ---")