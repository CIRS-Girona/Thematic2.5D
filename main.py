import os, yaml, time, logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from src.metrics import calculate_metrics
from src.utils import create_dataset, camera_parser, performance_stats
from src.classification import train_model
from src.inference import run_inference, compute_miou

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Set directories from config
    input_dir = config['directories']['input_dir']
    output_dir = config['directories']['output_dir']

    dataset_dir = f"{output_dir}/dataset"
    results_dir = f"{output_dir}/results"
    models_dir = f"{output_dir}/models"
    features_dir = f"{output_dir}/features"
    logging_dir = f"{output_dir}/logs"

    mask_suffix = config['directories']['mask_suffix']
    depth_suffix = config['directories']['depth_suffix']

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
        filename=f"{logging_dir}/{time.time()}.log",
        level=logging.INFO
    )

    dtsets = []
    for day in os.listdir(input_dir):
            if not os.path.isdir(f"{input_dir}/{day}"):
                continue

            for plot in os.listdir(f"{input_dir}/{day}"):
                if not os.path.isdir(f"{input_dir}/{day}/{plot}"):
                    continue

                for camera in os.listdir(f"{input_dir}/{day}/{plot}"):
                    if not os.path.isdir(f"{input_dir}/{day}/{plot}/{camera}"):
                        continue

                    dtsets.append((day, plot, camera))

    # Process according to config modes
    if config['create_dataset']['enabled']:
        print("Creating dataset...")

        args = []
        for day, plot, camera in dtsets:
            dtset = f"{day}_{plot}_{camera}"

            image_path = f"{input_dir}/{day}/{plot}/{camera}/images/"
            depth_path = f"{input_dir}/{day}/{plot}/{camera}/depthmaps/"
            mask_path = f"{input_dir}/{day}/{plot}/{camera}/masks/"

            for img in os.listdir(depth_path):
                label = '.'.join(img.split('.')[:-1])

                if not os.path.exists(f"{image_path}/{label}.jpg") or not os.path.isfile(f"{image_path}/{label}.jpg"):
                    raise FileNotFoundError(f"Corresponding image {label}.jpg couldn't be found in {image_path}.")
                
                args.append((
                    f"{image_path}/{label}.jpg",
                    f"{depth_path}/{label}{depth_suffix}.png",
                    f"{mask_path}/{label}{mask_suffix}.png",
                    dataset_dir,
                    dtset,
                    config['create_dataset']['bg_ratio'],
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
                    binary_mode=binary,
                    test_size=config['train_models']['test_size'],
                    n_components=config['train_models']['n_components'],
                    dimension=dimension,
                    subset_size=config['train_models']['subset_size']
                )

    if config['run_inference']['enabled']:
        print("Running inference...")

        args, data = [], {}
        for day, plot, camera in dtsets:
            dtset = f"{day}_{plot}_{camera}"

            image_path = f"{input_dir}/{day}/{plot}/{camera}/images/"
            depth_path = f"{input_dir}/{day}/{plot}/{camera}/depthmaps/"
            mask_path = f"{input_dir}/{day}/{plot}/{camera}/masks/"

            os.makedirs(f"{results_dir}/{dtset}", exist_ok=True)

            data[dtset] = []
            for img in os.listdir(depth_path):
                label = '.'.join(img.split('.')[:-1])

                if not os.path.exists(f"{image_path}/{label}.jpg") or not os.path.isfile(f"{image_path}/{label}.jpg"):
                    raise FileNotFoundError(f"Corresponding image {label}.jpg couldn't be found in {image_path}.")

                args.append((
                    dtset,
                    f"{image_path}/{label}.jpg",
                    f"{depth_path}/{label}{depth_suffix}.png",
                    f"{mask_path}/{label}{mask_suffix}.png",
                    models_dir,
                    f"{results_dir}/{dtset}",
                    config['max_uxo_code'],
                    config['run_inference']['num_components'],
                    config['run_inference']['compactness'],
                    config['window_size'],
                    config['patch_size'],
                    config['run_inference']['subdivide_axis'],
                    config['run_inference']['threshold'],
                    config['create_dataset']['uxo_threshold']
                ))

        with ThreadPoolExecutor(max_workers=thread_count) as exe:
            list(tqdm(
                exe.map(lambda a: data[a[0]].append(run_inference(*(a[1:]))), args),
                total=len(args)
            ))

        for dtset in data.keys():
            for model in data[dtset][0].keys():  # Assuming all runs_inference return the same model keys
                y_true = np.concatenate([d[model][0] for d in data[dtset] if d[model][0] is not None])
                y_pred = np.concatenate([d[model][1] for d in data[dtset] if d[model][1] is not None])
                print(np.unique(y_true))
                print(np.unique(y_pred))

                performance_stats(y_true, y_pred, f"{results_dir}/{dtset}", model)

    if config['evaluate_results']['compute_metrics']:
        print("Computing classification metrics...")

        args = []
        for day, plot, camera in dtsets:
            dtset = f"{day}_{plot}_{camera}"

            image_path = f"{input_dir}/{day}/{plot}/{camera}/images/"
            depth_path = f"{input_dir}/{day}/{plot}/{camera}/depthmaps/"
            mask_path = f"{input_dir}/{day}/{plot}/{camera}/masks/"

            camera_file = f"{input_dir}/{day}/{plot}/{camera}/cams.xml"
            metric_file = f"{results_dir}/{dtset}/metrics.csv"

            labels = ['.'.join(label.split('.')[:-1]) for label in os.listdir(image_path)]

            sensor = camera_parser(camera_file)[0]

            args.append((
                sensor,
                models_dir,
                [f"{image_path}/{label}.jpg" for label in labels],
                [f"{mask_path}/{label}{mask_suffix}.png" for label in labels],
                [f"{depth_path}/{label}{depth_suffix}.png" for label in labels],
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
        for day, plot, camera in dtsets:
            dtset = f"{day}_{plot}_{camera}"
            if not os.path.isdir(f"{results_dir}/{dtset}"):
                    continue

            mask_path = f"{input_dir}/{day}/{plot}/{camera}/masks/"
            for curr_dir in os.listdir(f"{results_dir}/{dtset}"):
                if not os.path.isdir(f"{results_dir}/{dtset}/{curr_dir}"):
                    continue
                elif not os.path.isdir(mask_path):
                    raise FileNotFoundError(f"Ground truth masks directory not found: {mask_path}")

                if dtset not in miou_scores:
                    miou_scores[dtset] = {}

                if curr_dir not in miou_scores[dtset]:
                    miou_scores[dtset][curr_dir] = {}

                for mask in os.listdir(mask_path):
                    label = "".join(mask.replace(mask_suffix, "").split('.')[:-1])

                    if not os.path.exists(f"{results_dir}/{dtset}/{curr_dir}/{label}_mask.png"):
                        continue

                    is_binary = "binary" in curr_dir
                    args.append((
                        dtset,
                        curr_dir,
                        label,
                        (
                            f"{mask_path}/{label}{mask_suffix}.png",
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
