import numpy as np
import cv2, os, gc, random
from concurrent.futures import ThreadPoolExecutor


ADJUST_COOR = lambda c, r, rnge: (0, 2*r) if c - r < 0 else (rnge[1] - 1 - 2*r, rnge[1] - 1) if c + r >= rnge[1] else (c - r, c + r)


def process_data(image, depth, mask, indices, bg_max, dataset_dir, prefix, uxo_threshold, invalid_threshold, window_size, patch_size, angles):
    print("Started thread")

    bg_count = 0
    w, h = mask.shape
    t, m, d = None, None, None
    for i, (c_y, c_x) in enumerate(indices):
        del t, m, d
        gc.collect()

        radius = window_size // 2

        x_s, x_e = ADJUST_COOR(c_x, radius, (0, w - 1))
        y_s, y_e = ADJUST_COOR(c_y, radius, (0, h - 1))

        t = image[y_s:y_e, x_s:x_e, :]
        m = mask[y_s:y_e, x_s:x_e]
        d = depth[y_s:y_e, x_s:x_e]

        # If the patch has any invalid area, skip
        if np.sum(m == -1)/m.size > invalid_threshold:
            continue

        t = cv2.resize(t, (patch_size, patch_size), interpolation=cv2.INTER_AREA)
        d = cv2.resize(d, (patch_size, patch_size), interpolation=cv2.INTER_AREA)

        d = np.astype(d, np.double)
        d -= np.min(d)
        d /= np.max(d)
        d = np.astype(255 * d, np.uint8)

        if np.sum(m > 0)/m.size >= uxo_threshold:
            uxo_number = np.sort(np.unique(m))[-1]

            if not os.path.exists(f"{dataset_dir}/2D/{uxo_number}/") or not os.path.isdir(f"{dataset_dir}/2D/{uxo_number}/"):
                os.mkdir(f"{dataset_dir}/2D/{uxo_number}/")
                os.mkdir(f"{dataset_dir}/3D/{uxo_number}/")

            h_img, w_img = d.shape
            for angle in angles:
                M = cv2.getRotationMatrix2D((w_img//2, h_img//2), angle, 1)  # Center, rotation angle, scale
                t_rot = cv2.warpAffine(t, M, (w_img, h_img))
                d_rot = cv2.warpAffine(d, M, (w_img, h_img))

                cv2.imwrite(f"{dataset_dir}/2D/{uxo_number}/{prefix}-{i}_{angle}.png", t_rot)
                cv2.imwrite(f"{dataset_dir}/3D/{uxo_number}/{prefix}-{i}_{angle}.png", d_rot)

            del t_rot, d_rot
            gc.collect()
        elif np.all(m == 0) and bg_count < bg_max:
            cv2.imwrite(f"{dataset_dir}/2D/background/{prefix}-{i}.png", t)
            cv2.imwrite(f"{dataset_dir}/3D/background/{prefix}-{i}.png", d)
            bg_count += 1


def create_dataset(images_path, depths_path, masks_path, dataset_dir, prefix='', bg_per_img=20_000, thread_count=64, uxo_sample_rate=0.01, uxo_threshold=0.4, invalid_threshold=0.01, window_size=400, patch_size=128, angles=(0, 90, 180, 270)):
    print(f"Started processing dataset {prefix}")

    labels = [l.split(".")[0] for l in os.listdir(masks_path)]

    with ThreadPoolExecutor(max_workers=thread_count) as exe:
        for label in labels:
            image = cv2.imread(f"{images_path}/{label}.jpg", cv2.IMREAD_UNCHANGED)
            depth = cv2.imread(f"{depths_path}/{label}.png", cv2.IMREAD_UNCHANGED)

            if image is None or depth is None:
                del image, depth
                gc.collect()
                continue

            mask = np.astype(cv2.imread(f"{masks_path}/{label}.png", cv2.IMREAD_UNCHANGED), np.int32)

            mask[mask < 3] = 0  # Set Non-UXO pixels to 0

            # Set invalid pixels to -1
            mask[mask == 99] = -1
            mask[depth == 0] = -1

            if np.all(np.unique(mask) == -1):
                del image, mask, depth
                gc.collect()
                print(f"Finished processing image {prefix}")
                continue

            uxo_indices = np.where(mask > 0)
            uxo_indices = list(zip(uxo_indices[0], uxo_indices[1]))
            uxo_indices = random.sample(uxo_indices, int(len(uxo_indices) * uxo_sample_rate))

            # Oversample so that there are enough valid patches
            bg_indices = np.where(mask == 0)
            bg_indices = list(zip(bg_indices[0], bg_indices[1]))
            bg_indices = random.sample(bg_indices, 2 * bg_per_img) if len(bg_indices) > 2 * bg_per_img else bg_indices

            indices = uxo_indices + bg_indices

            del uxo_indices, bg_indices
            gc.collect()

            exe.submit(
                process_data,
                image,
                depth,
                mask,
                indices,
                bg_per_img,
                dataset_dir,
                f"{label}-{prefix}",
                uxo_threshold,
                invalid_threshold,
                window_size,
                patch_size,
                angles
            )

    print(f"Finished processing image {prefix}")

    del image, mask, depth, indices
    gc.collect()