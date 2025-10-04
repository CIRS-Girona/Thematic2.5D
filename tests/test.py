import os, yaml, subprocess

MODELS = ('2', '3', '25')
SAMPLES = ('18mm', '24mm', 'GPS')

TEST_DIR = "tests"
SAMPLE_DIR = f"{TEST_DIR}/samples"
DATASET_DIR = f"{TEST_DIR}/data/dataset"
FEATURES_DIR = f"{TEST_DIR}/data/features"
MODELS_DIR = f"{TEST_DIR}/data/models"
RESULTS_DIR = f"{TEST_DIR}/data/results"


def run_pipeline(conf: dict):
    with open("config.yaml", 'w') as f:
        yaml.safe_dump(conf, f)

    subprocess.run(["python", "main.py"])


def data_check(data_path, check_dir = False, check_empty=False):
    assert os.path.exists(data_path)  # Check if data exists

    if check_dir:  # Check that it is the correct file type
        assert os.path.isdir(data_path)
    else:
        assert os.path.isfile(data_path)

    if check_empty and check_dir:  # If it is a dir, check if it is empty
        assert len(os.listdir(data_path)) > 0


if __name__ == "__main__":
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Create dataset from samples
    config['create_dataset']['enabled'] = True
    config['train_model']['enabled'] = True
    config['run_inference']['enabled'] = False
    config['evaluate_results']['enabled'] = False

    config['train_model']['dimension'] = '25'
    config['train_model']['use_saved_features'] = False

    run_pipeline(config)

    data_check(f"{DATASET_DIR}/2D", check_dir=True, check_empty=True)
    data_check(f"{DATASET_DIR}/3D", check_dir=True, check_empty=True)
    data_check(FEATURES_DIR, check_dir=True, check_empty=True)
    data_check(MODELS_DIR, check_dir=True, check_empty=True)
    data_check(RESULTS_DIR, check_dir=True)

    # Train the different models on the dataset created
    config['create_dataset']['enabled'] = False
    config['train_model']['use_saved_features'] = True
    for model in MODELS:
        for binary in (True, False):
            config['train_model']['dimension'] = model
            config['train_model']['binary_mode'] = binary

            run_pipeline(config)

            file_name = f"SVM{model}_binary" if binary else f"SVM{model}"

            data_check(f"{RESULTS_DIR}/{file_name}.jpg")
            data_check(f"{RESULTS_DIR}/{file_name}.txt")

    # Run the trained models on the samples
    config['train_model']['enabled'] = False
    for sample in SAMPLES:
        config['run_inference']['enabled'] = True
        config['run_inference']['image_path'] = f"{SAMPLE_DIR}/{sample}/images"
        config['run_inference']['depth_path'] = f"{SAMPLE_DIR}/{sample}/depths"

        config['evaluate_results']['enabled'] = True
        config['evaluate_results']['mask_path'] = f"{SAMPLE_DIR}/{sample}/masks"

        run_pipeline(config)

    # Check data integrity
    for model in MODELS:
        for binary in (True, False):
            file_name = f"SVM{model}_binary" if binary else f"SVM{model}"

            data_check(f"{RESULTS_DIR}/{file_name}-mIoU.txt")
            data_check(f"{RESULTS_DIR}/{file_name}", check_dir=True, check_empty=True)
