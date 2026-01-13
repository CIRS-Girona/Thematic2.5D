import os, yaml, time, logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from src.metrics import calculate_metrics
from src.utils import create_dataset, camera_parser
from src.classification import train_model
from src.inference import run_inference, compute_miou

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Set directories from config
    input_dir = config['directories']['input_dir']
    inference_dir = config['directories']['inference_dir']
    dataset_dir = config['directories']['dataset_dir']
    results_dir = config['directories']['results_dir']
    models_dir = config['directories']['models_dir']
    features_dir = config['directories']['features_dir']
    logging_dir = config['directories']['logging_dir']

    thread_count = config['thread_count']

    if not os.path.exists(input_dir) or not os.path.isdir(input_dir):
        print("The images folder doesn't exist. Please create the images folder as explained in the README file.")
        exit()

    # Create directories if they don't exist
    dir_paths = [
        dataset_dir,
        results_dir,
        models_dir,
        features_dir,
        logging_dir,
        f"{dataset_dir}/2D/",
        f"{dataset_dir}/2D/background",
        f"{dataset_dir}/3D/",
        f"{dataset_dir}/3D/background"
    ]

    for dir_path in dir_paths:
        if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
            os.makedirs(dir_path)

    logging.basicConfig(
        format='%(asctime)s - %(name)s - [%(levelname)s]: %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        filename=f"{config['directories']['logging_dir']}/{time.time()}.log",
        level=logging.INFO
    )

    # Process according to config modes
    if config['create_dataset']['enabled']:
        print("Creating dataset...")

        args = []
        for dtset in os.listdir(input_dir):
            if not os.path.isdir(f"{input_dir}/{dtset}"):
                continue

            image_path = f"{input_dir}/{dtset}/images/"
            depth_path = f"{input_dir}/{dtset}/depths/"
            mask_path = f"{input_dir}/{dtset}/masks/"

            for img in os.listdir(depth_path):
                label = '.'.join(img.split('.')[:-1])

                if not os.path.exists(f"{image_path}/{label}.jpg") or not os.path.isfile(f"{image_path}/{label}.jpg"):
                    raise FileNotFoundError(f"Corresponding image {label}.jpg couldn't be found in {image_path}.")
                
                args.append((
                    f"{image_path}/{label}.jpg",
                    f"{depth_path}/{label}.png",
                    f"{mask_path}/{label}.png",
                    dataset_dir,
                    dtset,
                    config['create_dataset']['bg_per_img'],
                    config['create_dataset']['uxo_sample_rate'],
                    config['create_dataset']['uxo_threshold'],
                    config['create_dataset']['invalid_threshold'],
                    config['window_size'],
                    config['patch_size'],
                    config['create_dataset']['angles']
                ))

        with ThreadPoolExecutor(max_workers=thread_count) as exe:
            list(tqdm(
                exe.map(lambda a: create_dataset(*a), args),
                total=len(args)
            ))

    if config['train_models']['enabled']:
        print("Training models...")

        args = [(binary, dimension) for binary in (True, False) for dimension in ('25', '3', '2')]
        for binary, dimension in tqdm(args):
                train_model(
                    dataset_dir=dataset_dir,
                    features_dir=features_dir,
                    models_dir=models_dir,
                    results_dir=results_dir,
                    binary_mode=binary,
                    test_size=config['train_models']['test_size'],
                    n_components=config['train_models']['n_components'],
                    dimension=dimension,
                    subset_size=config['train_models']['subset_size']
                )

    if config['run_inference']['enabled']:
        print("Running inference...")

        args = []
        for dtset in os.listdir(inference_dir):
            if not os.path.isdir(f"{inference_dir}/{dtset}"):
                continue

            os.makedirs(f"{results_dir}/{dtset}", exist_ok=True)

            image_path = f"{inference_dir}/{dtset}/images"
            depth_path = f"{inference_dir}/{dtset}/depths"

            for img in os.listdir(depth_path):
                label = '.'.join(img.split('.')[:-1])

                if not os.path.exists(f"{image_path}/{label}.jpg") or not os.path.isfile(f"{image_path}/{label}.jpg"):
                    raise FileNotFoundError(f"Corresponding image {label}.jpg couldn't be found in {image_path}.")

                args.append((
                    f"{image_path}/{label}.jpg",
                    f"{depth_path}/{label}.png",
                    models_dir,
                    f"{results_dir}/{dtset}",
                    config['max_uxo_code'],
                    config['run_inference']['num_components'],
                    config['run_inference']['compactness'],
                    config['window_size'],
                    config['patch_size'],
                    config['run_inference']['subdivide_axis'],
                    config['run_inference']['threshold'],
                ))

        with ThreadPoolExecutor(max_workers=thread_count) as exe:
            list(tqdm(
                exe.map(lambda a: run_inference(*a), args),
                total=len(args)
            ))

    if config['evaluate_results']['compute_metrics']:
        print("Computing classification metrics...")

        args = []
        for dtset in os.listdir(input_dir):
            camera_file = f"{input_dir}/{dtset}/cams.xml"
            info_file = f"{input_dir}/{dtset}/info.yaml"
            metric_file = f"{input_dir}/{dtset}/metrics.csv"

            img_path = f"{input_dir}/{dtset}/images/"
            mask_path = f"{input_dir}/{dtset}/masks/"
            depth_path = f"{input_dir}/{dtset}/depths/"

            labels = ['.'.join(label.split('.')[:-1]) for label in os.listdir(img_path)]

            sensor = camera_parser(camera_file)[0]

            with open(info_file, 'r') as f:
                info = yaml.safe_load(f)

                camera_type = info["camera_type"]
                visibility = info["visibility"]

            args.append((
                sensor,
                models_dir,
                [f"{img_path}/{label}.jpg" for label in labels],
                [f"{mask_path}/{label}.png" for label in labels],
                [f"{depth_path}/{label}.png" for label in labels],
                camera_type,
                visibility,
                metric_file,
                config['window_size'],
                config['patch_size'],
            ))

        with ThreadPoolExecutor(max_workers=thread_count) as exe:
            list(tqdm(
                exe.map(lambda a: calculate_metrics(*a), args),
                total=len(args)
            ))

    if config['evaluate_results']['compute_miou']:
        print("Evaluating mIoU results...")

        miou_scores = {}  # { dtset: {dir: { label: miou_score } } }

        args = []
        for dtset in os.listdir(results_dir):
            if not os.path.isdir(f"{results_dir}/{dtset}"):
                    continue

            for curr_dir in os.listdir(f"{results_dir}/{dtset}"):
                if not os.path.isdir(f"{results_dir}/{dtset}/{curr_dir}"):
                    continue
                elif not os.path.isdir(f"{inference_dir}/{dtset}/masks"):
                    raise FileNotFoundError(f"Ground truth masks directory not found: {inference_dir}/{dtset}/masks")

                if dtset not in miou_scores:
                    miou_scores[dtset] = {}

                if curr_dir not in miou_scores[dtset]:
                    miou_scores[dtset][curr_dir] = {}

                for mask in os.listdir(f"{inference_dir}/{dtset}/masks"):
                    label = "".join(mask.split('.')[:-1])

                    if not os.path.exists(f"{results_dir}/{dtset}/{curr_dir}/{label}_mask.png"):
                        continue

                    is_binary = "binary" in curr_dir
                    args.append((
                        dtset,
                        curr_dir,
                        label,
                        (
                            f"{inference_dir}/{dtset}/masks/{mask}",
                            f"{results_dir}/{dtset}/{curr_dir}/{label}_mask.png",
                            is_binary
                        )
                    ))

        with ThreadPoolExecutor(max_workers=thread_count) as exe:
            list(tqdm(
                exe.map(
                    lambda a: miou_scores[a[0]][a[1]].update({a[2]: compute_miou(*a[3])}),
                    args
                ),
                total=len(args)
            ))

        for dtset in miou_scores.keys():
            for curr_dir in miou_scores[dtset].keys():
                with open(f"{results_dir}/{dtset}/{curr_dir}-mIoU.txt", 'w') as f:
                    data = miou_scores[dtset][curr_dir]
                    f.write(f"Average mIoU score: {np.mean(tuple(data.values()))}\n\n")

                    for label, miou_score in data.items():
                        f.write(f"{label}: {miou_score}\n")
