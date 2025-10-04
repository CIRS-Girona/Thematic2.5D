import os, yaml, cv2, gc
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from src.utils import create_dataset, train_model, run_inference, meanIoU


if __name__ == "__main__":
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Set directories from config
    input_dir = config['directories']['input_dir']
    dataset_dir = config['directories']['dataset_dir']
    results_dir = config['directories']['results_dir']
    models_dir = config['directories']['models_dir']
    features_dir = config['directories']['features_dir']

    if not os.path.exists(input_dir) or not os.path.isdir(input_dir):
        print("The images folder doesn't exist. Please create the images folder as explained in the README file.")
        exit()

    # Create directories if they don't exist
    for dir_path in [dataset_dir, results_dir, models_dir, features_dir, f"{dataset_dir}/2D/", f"{dataset_dir}/2D/background", f"{dataset_dir}/3D/", f"{dataset_dir}/3D/background"]:
        if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
            os.makedirs(dir_path)

    # Process according to config modes
    if config['create_dataset']['enabled']:
        print("Creating dataset...")

        for dtset in os.listdir(input_dir):
            if not os.path.isdir(f"{input_dir}/{dtset}"):
                continue

            create_dataset(
                f"{input_dir}/{dtset}/images/",
                f"{input_dir}/{dtset}/depths/",
                f"{input_dir}/{dtset}/masks/",
                dataset_dir=dataset_dir,
                uxo_start_code=config['uxo_start_code'],
                invalid_code=config['invalid_code'],
                prefix=dtset,
                bg_per_img=config['create_dataset']['bg_per_img'],
                thread_count=config['create_dataset']['thread_count'],
                uxo_sample_rate=config['create_dataset']['uxo_sample_rate'],
                uxo_threshold=config['create_dataset']['uxo_threshold'],
                invalid_threshold=config['create_dataset']['invalid_threshold'],
                window_size=config['create_dataset']['window_size'],
                patch_size=config['create_dataset']['patch_size'],
                angles=config['create_dataset']['angles']
            )

    if config['train_model']['enabled']:
        print("Training model...")
        train_model(
            dataset_dir=dataset_dir,
            features_dir=features_dir,
            models_dir=models_dir,
            results_dir=results_dir,
            uxo_start_code=config['uxo_start_code'],
            binary_mode=config['train_model']['binary_mode'],
            test_size=config['train_model']['test_size'],
            n_components=config['train_model']['n_components'],
            dimension=config['train_model']['dimension'],
            use_saved_features=config['train_model']['use_saved_features'],
            subset_size=config['train_model']['subset_size']
        )

    if config['run_inference']['enabled']:
        print("Running inference...")

        args = []
        for img in os.listdir(config['run_inference']['depth_path']):
            label = '.'.join(img.split('.')[:-1])

            image_path = f"{config['run_inference']['image_path']}/{label}"
            if os.path.exists(f"{image_path}.jpg") and os.path.isfile(f"{image_path}.jpg"):
                image_path = f"{image_path}.jpg"
            elif os.path.exists(f"{image_path}.png") and os.path.isfile(f"{image_path}.png"):
                image_path = f"{image_path}.png"
            else:
                raise FileNotFoundError(f"Corresponding image {label} couldn't be found in jpg or png format.")

            args.append((
                image_path,
                f"{config['run_inference']['depth_path']}/{label}.png",
                models_dir,
                results_dir,
                config['uxo_start_code'],
                config['max_uxo_code'],
                config['run_inference']['region_size'],
                config['run_inference']['window_size'],
                config['run_inference']['patch_size'],
                config['run_inference']['subdivide_axis'],
                config['run_inference']['threshold'],
            ))

        with ThreadPoolExecutor(max_workers=config['run_inference']['thread_count']) as exe:
            list(tqdm(
                exe.map(lambda a: run_inference(*a), args),
                total=len(args)
            ))

    if config['evaluate_results']['enabled']:
        print("Evaluating results...")

        for dir in os.listdir(results_dir):
            curr_dir = f"{results_dir}/{dir}"

            if not os.path.isdir(curr_dir):
                continue

            print(f"Evaluating results for {dir}")

            miou_scores = {}
            for mask in tqdm(os.listdir(config['evaluate_results']['mask_path'])):
                label = "".join(mask.split('.')[:-1])

                if not os.path.exists(f"{curr_dir}/{label}_mask.png"):
                    continue

                mask_gt = cv2.imread(f"{config['evaluate_results']['mask_path']}/{mask}", cv2.IMREAD_UNCHANGED)
                mask_result = cv2.imread(f"{curr_dir}/{label}_mask.png", cv2.IMREAD_UNCHANGED)

                mask_gt[mask_gt < config['uxo_start_code']] = 0
                if "binary" in dir:
                    mask_gt[mask_gt > 0] = 1

                miou_scores[label] = meanIoU(mask_gt, mask_result)

                del mask_gt, mask_result
                gc.collect()

            with open(f"{results_dir}/{dir}-mIoU.txt", 'w') as f:
                f.write(f"Average mIoU score: {np.mean(tuple(miou_scores.values()))}\n\n")
                
                for label, miou_score in miou_scores.items():
                    f.write(f"{label}: {miou_score}\n")
