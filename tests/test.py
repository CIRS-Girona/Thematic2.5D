import os, yaml, subprocess

# Define constants for model dimensions and sample names
MODELS = ('2', '3', '25')
SAMPLES = ('18mm', '24mm', 'GPS')

# Define directory paths relative to the TEST_DIR
TEST_DIR = "tests"
INPUT_DIR = f"{TEST_DIR}/samples"
OUTPUT_DIR = f"{TEST_DIR}/output"

DATASET_DIR = f"{OUTPUT_DIR}/dataset"
FEATURES_DIR = f"{OUTPUT_DIR}/features"
MODELS_DIR = f"{OUTPUT_DIR}/models"
RESULTS_DIR = f"{OUTPUT_DIR}/results"
LOGGING_DIR = f"{OUTPUT_DIR}/logs"


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
    with open("config.yaml", 'r') as f:
        original_config = f.read()
        f.seek(0)
        config = yaml.safe_load(f)

    try:
        config['directories']['input_dir'] = INPUT_DIR
        config['directories']['output_dir'] = OUTPUT_DIR
        config['directories']['mask_suffix'] = "_binary"
        config['directories']['depth_suffix'] = ""

        config['create_dataset']['enabled'] = True
        config['train_models']['enabled'] = True
        config['run_inference']['enabled'] = True
        config['evaluate_results']['compute_metrics'] = True
        config['evaluate_results']['compute_miou'] = True

        print("--- Running pipeline ---")
        run_pipeline(config)

        # Verify that the expected data directories and files were created and are not empty
        data_check(f"{DATASET_DIR}/2D", check_dir=True, check_empty=True)
        data_check(f"{DATASET_DIR}/3D", check_dir=True, check_empty=True)
        data_check(FEATURES_DIR, check_dir=True, check_empty=True)  # Check for saved features
        data_check(MODELS_DIR, check_dir=True, check_empty=True)    # Check for saved model
        data_check(RESULTS_DIR, check_dir=True)                     # Results dir might be empty/have logs

        # Verify that performance metrics and per-sample results were generated
        for dtset in os.listdir(RESULTS_DIR):
            data_check(f"{RESULTS_DIR}/{dtset}", check_dir=True, check_empty=True)
            data_check(f"{RESULTS_DIR}/{dtset}/metrics.csv")

            for model in MODELS:
                for binary in (True, False):
                    file_name = f"SVM{model}_binary" if binary else f"SVM{model}"

                    data_check(f"{RESULTS_DIR}/{dtset}/{file_name}-cm.txt")
                    data_check(f"{RESULTS_DIR}/{dtset}/{file_name}-mIoU.txt")
                    data_check(f"{RESULTS_DIR}/{dtset}/{file_name}-stats.txt")
                    data_check(f"{RESULTS_DIR}/{dtset}/{file_name}", check_dir=True, check_empty=True)

        print("--- All pipeline tests and data checks completed successfully. ---")

    except Exception as e:
        print(f"Error occured: {e}")

    finally:
        # Restore the original configuration file
        with open("config.yaml", 'w') as f:
            f.write(original_config)
