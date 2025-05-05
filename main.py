from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt
from typing import Literal
from concurrent.futures import ThreadPoolExecutor
import datetime, cv2, os, yaml

from src.utils import ADJUST_COOR, create_dataset, load_features, save_features, extract_features, process_images, superpixel_segmentation, apply_mask
from src.classification import SVMModel


def train_model(dataset_dir, features_dir, models_dir, results_dir, uxo_start_code, binary_mode=False, test_size=0.1, n_components: int = 100, dimension: Literal['2', '25', '3'] = '25', use_saved_features=True, subset_size=0):
    if not use_saved_features:
        save_features(f"{dataset_dir}/2D/", f"{dataset_dir}/3D/", features_dir, subset=subset_size)

    # Load features and encode labels
    X_data, y_data = load_features(features_dir, dimension=dimension)

    if binary_mode:
        for y in np.unique(y_data):
            if y.isdigit() and int(y) >= uxo_start_code:
                y_data[y == y_data] = 1
            else:
                y_data[y == y_data] = 0

    print(f"Training start time: {datetime.datetime.now().isoformat()}")

    # Transform and split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, stratify=y_data, test_size=test_size)

    # Train model on full dataset and save it
    model = SVMModel(model_dir=models_dir, n_components=n_components)
    model.train(X_train, y_train)
    model.save_model()

    # Evaluate on the test set
    y_pred = model.evaluate(X_test)

    # Save the classification report to a file
    if not os.path.exists(results_dir) or not os.path.isdir(results_dir):
        os.mkdir(results_dir)

    with open(f"{results_dir}/{model.model_name}_{dimension}D.txt", 'w') as f:
        print(classification_report(y_test, y_pred, zero_division=0))
        print(classification_report(y_test, y_pred, zero_division=0), file=f)

    cm = confusion_matrix(y_test, y_pred, normalize='true')
    cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.model.classes_)
    cmd.plot()
    plt.tight_layout()
    plt.savefig(f"{results_dir}/{model.model_name}_{dimension}D.png")
    plt.close('all')


def run_inference(image_path, depth_path, models_dir, results_dir, model_name, binary_mode, uxo_start_code, region_size=400, window_size=400, patch_size=128, subdivide_axis=3, threshold=3, dimension: Literal['2', '25', '3']='25'):
    print(f"Running inference ({dimension}D) on:\n{image_path}\n")

    if not os.path.exists(results_dir) or not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    img = cv2.imread(image_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    img_label = '.'.join(image_path.split('/')[-1].split('.')[:-1])

    print("Applying Superpixel Segmentation")
    labels, centroids = superpixel_segmentation(img, ruler=1, region_size=region_size)

    # Loop over each centroid
    imgs = []
    depths = []
    patches = []
    for c_y, c_x in centroids:
        window_radius = window_size // 2

        # Calculate patch boundaries
        x_start, x_end = ADJUST_COOR(c_x, window_radius, (0, img.shape[1]))
        y_start, y_end = ADJUST_COOR(c_y, window_radius, (0, img.shape[0]))

        # Create coordinate arrays for subdivisions
        x_coords = np.linspace(x_start, x_end, subdivide_axis + 1, endpoint=True, dtype=int)
        y_coords = np.linspace(y_start, y_end, subdivide_axis + 1, endpoint=True, dtype=int)

        # Calculate steps between subdivision boundaries
        for x_step in x_coords:
            for y_step in y_coords:
                x_sub_start, x_sub_end = ADJUST_COOR(x_step, window_radius, (0, img.shape[1]))
                y_sub_start, y_sub_end = ADJUST_COOR(y_step, window_radius, (0, img.shape[0]))

                # Extract and resize image patch
                img_patch = cv2.resize(img[x_sub_start:x_sub_end, y_sub_start:y_sub_end, :], (patch_size, patch_size), interpolation=cv2.INTER_AREA)
                imgs.append(img_patch)

                # Extract, resize, and normalize depth patch
                depth_patch = cv2.resize(depth[x_sub_start:x_sub_end, y_sub_start:y_sub_end], (patch_size, patch_size), interpolation=cv2.INTER_AREA)
                depth_patch = depth_patch.astype(np.double)
                depth_patch -= np.min(depth_patch)
                depth_patch /= max(np.max(depth_patch), 1)
                depth_patch = np.nan_to_num(255 * depth_patch).astype(np.uint8).astype(np.double)
                depths.append(depth_patch)

                # Store corresponding label
                patches.append(labels[c_x, c_y])

    print("Processing patches")

    patches = np.array(patches)
    gray_images, hsv_images = process_images(imgs)
    features_2d, features_3d = extract_features(gray_images, hsv_images, None if dimension == '2' else depths)

    if dimension == '3':
        features = features_3d
    elif dimension == '2':
        features = features_2d
    else:
        features = np.concatenate([features_2d, features_3d], axis=1)

    print("Loading model and running inference")

    # SVM Model
    model = SVMModel(model_dir=models_dir)
    model.load_model(model_name)

    y_pred = model.evaluate(features)

    uxo_mask = np.zeros_like(labels)
    for y in np.unique(y_pred):
        regions = patches[y_pred == y]

        if (binary_mode and int(y) == 1) or (not binary_mode and y.isdigit() and int(y) >= uxo_start_code):
            for region in np.unique(regions):
                if regions[regions == region].size < threshold:
                    uxo_mask[labels == region] = 0
                else:
                    uxo_mask[labels == region] = int(y)

    cv2.imwrite(f"{results_dir}/{img_label}_mask.png", uxo_mask.astype(np.uint8))

    uxo_mask[uxo_mask == 0] = -1
    inference = apply_mask(img, uxo_mask, mode='highlight')
    cv2.imwrite(f"{results_dir}/{img_label}.png", inference)


if __name__ == "__main__":
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Set directories from config
    source_dir = config['directories']['source_dir']
    input_dir = f"{source_dir}/{config['directories']['input_dir']}"
    dataset_dir = f"{source_dir}/{config['directories']['dataset_dir']}"
    results_dir = f"{source_dir}/{config['directories']['results_dir']}"
    models_dir = f"{source_dir}/{config['directories']['models_dir']}"
    features_dir = f"{source_dir}/{config['directories']['features_dir']}"

    if not os.path.exists(input_dir) or not os.path.isdir(input_dir):
        print("The tiles folder doesn't exist. Please create the tiles folder as explained in the README file.")
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
                f"{input_dir}/{dtset}/images",
                f"{input_dir}/{dtset}/depths",
                f"{input_dir}/{dtset}/masks",
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

        with ThreadPoolExecutor(max_workers=config['run_inference']['thread_count']) as exe:
            
            for img in os.listdir(config['run_inference']['depth_path']):
                label = '.'.join(img.split('.')[:-1])

                exe.submit(
                    run_inference,
                    f"{config['run_inference']['image_path']}/{label}.jpg",
                    f"{config['run_inference']['depth_path']}/{label}.png",
                    models_dir,
                    f"{results_dir}/{config['run_inference']['output_dir']}",
                    config['run_inference']['model_name'],
                    config['run_inference']['binary_mode'],
                    config['uxo_start_code'],
                    config['run_inference']['region_size'],
                    config['run_inference']['window_size'],
                    config['run_inference']['patch_size'],
                    config['run_inference']['subdivide_axis'],
                    config['run_inference']['threshold'],
                    config['run_inference']['dimension']
                )