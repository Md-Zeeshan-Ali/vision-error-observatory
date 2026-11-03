import cv2
import numpy as np
from scipy import ndimage


def extract_failure_regions(
    error_map: np.ndarray,
    valid_mask: np.ndarray,
    percentile: float = 85.0,
    min_area: int = 25,
    failure_mask: np.ndarray | None = None,
    opening_radius: int = 0,
) -> list[dict]:
    """Connected failure regions.

    ``failure_mask`` gives the failing pixels directly (e.g. pixels failing delta_1); without it the
    image's errors above ``percentile`` are used. A morphological opening of ``opening_radius`` px
    removes isolated speckle before labelling.
    """
    if error_map.shape != valid_mask.shape:
        raise ValueError("error_map and valid_mask must have identical shapes")
    if failure_mask is None:
        valid_errors = error_map[valid_mask]
        if valid_errors.size == 0:
            return []
        failure_mask = error_map > float(np.percentile(valid_errors, percentile))
    binary = failure_mask & valid_mask
    if opening_radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * opening_radius + 1, 2 * opening_radius + 1))
        binary = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool) & valid_mask
    labels, count = ndimage.label(binary)
    regions = []
    for region_id, box in enumerate(ndimage.find_objects(labels), start=1):
        if box is None:
            continue
        local = labels[box] == region_id  # compare inside the region's box only
        area = int(local.sum())
        if area < min_area:
            continue
        mask = np.zeros(labels.shape, dtype=bool)
        mask[box] = local
        y_slice, x_slice = box
        regions.append({
            "region_id": region_id,
            "bbox": [x_slice.start, y_slice.start, x_slice.stop - x_slice.start, y_slice.stop - y_slice.start],
            "area": area,
            "mean_error": float(error_map[mask].mean()),
            "mask": mask,
        })
    return regions
