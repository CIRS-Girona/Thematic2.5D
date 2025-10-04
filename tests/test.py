import os, yaml, subprocess

MODELS = ('2', '3', '25')
SAMPLES = ('18mm', '24mm', 'GPS')

TEST_DIR = "tests"
SAMPLE_DIR = f"{TEST_DIR}/samples"
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

    config['train_model']['use_saved_features'] = False
    config['create_dataset']['enabled'] = True

    run_pipeline(config)

    config['train_model']['use_saved_features'] = True
    config['create_dataset']['enabled'] = False

    assert os.path.exists(RESULTS_DIR) and os.path.isdir(RESULTS_DIR)

    for sample in SAMPLES:
        for model in MODELS:
            for binary in (True, False):
                config['train_model']['binary_mode'] = binary
                config['train_model']['dimension'] = model

                config['run_inference']['image_path'] = f"{SAMPLE_DIR}/{sample}/images"
                config['run_inference']['depth_path'] = f"{SAMPLE_DIR}/{sample}/depths"

                config['evaluate_results']['mask_path'] = f"{SAMPLE_DIR}/{sample}/masks"

                run_pipeline(config)

                data_check(
                    f"{RESULTS_DIR}/SVM{model}",
                    check_dir=True,
                    check_empty=True
                )

                data_check(f"{RESULTS_DIR}/SVM{model}-mIoU.txt")
                data_check(f"{RESULTS_DIR}/SVM{model}.jpg")
                data_check(f"{RESULTS_DIR}/SVM{model}.txt")
