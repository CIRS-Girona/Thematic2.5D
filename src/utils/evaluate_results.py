import numpy as np

def meanIoU(mask_gt: np.ndarray, mask: np.ndarray) -> float:
    """
    Calculate the mean Intersection over Union (IoU) score between two segmentation masks.

    Args:
        mask_gt (np.ndarray): Ground truth segmentation mask, where each element represents a class label.
        mask (np.ndarray): Predicted segmentation mask, with the same shape and class labels as `mask_gt`.

    Returns:
        float: The mean IoU score across all classes. Returns a value between 0.0 and 1.0, where 1.0 indicates perfect overlap and 0.0 indicates no overlap.
    """
    # Get unique classes in both masks
    classes = np.unique(np.concatenate([mask_gt, mask]))

    iou_scores = 0.0
    for cls in classes:
        # Create binary masks for the current class
        mask1_cls = mask_gt == cls
        mask2_cls = mask == cls

        # Calculate intersection and union
        intersection = np.sum(np.logical_and(mask1_cls, mask2_cls))
        union = np.sum(np.logical_or(mask1_cls, mask2_cls))

        # Calculate IoU for the class (handle case where union is 0)
        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = intersection / union

        iou_scores += iou

    return iou_scores / len(classes)