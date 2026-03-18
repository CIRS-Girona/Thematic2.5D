import cv2

from ..utils import meanIoU


def compute_miou(
        mask_gt_path: str,
        mask_path: str,
        is_binary: bool = False
):
    mask_gt = cv2.imread(mask_gt_path, cv2.IMREAD_UNCHANGED)
    mask_result = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

    if is_binary:
        mask_gt[mask_gt > 0] = 1

    return meanIoU(mask_gt, mask_result)