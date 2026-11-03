"""Objective ("silver") labels for failure regions, derived from NYU's human annotations.

They are independent of VEO's image features (or, for boundaries, only partly overlapping) and
cover every region, so they complement small human studies. Per region we record the fraction
of pixels supporting each category and a binary label at a fixed share.

  reflection       on reflective classes (mirror, window, television, monitor, whiteboard) or on
                   classes the annotators marked as reflections
  thin_object      on thin parts of annotated objects: pixels removed by an opening of the object's
                   mask with a ``thin_width``-px disc (chair legs, poles, rods), excluding structure
  boundary_mixing  near a boundary between two annotated segments with a depth jump of more than
                   ``boundary_log_jump`` in log depth (an occluding object boundary)
  textureless      on walls or ceilings

There is no annotation for lighting, so ``poor_lighting`` has no silver label.
"""

import re
from dataclasses import dataclass

import cv2
import numpy as np

REFLECTIVE_NAMES = {"mirror", "window", "television", "monitor", "whiteboard"}
REFLECTION_PATTERN = re.compile(r"reflection")
STRUCTURE_NAMES = {"wall", "floor", "ceiling", "floor mat"}
FLAT_NAMES = {"wall", "ceiling"}
SILVER_CATEGORIES = ("reflection", "thin_object", "boundary_mixing", "textureless")


@dataclass(frozen=True)
class SilverParams:
    thin_width: int = 9
    boundary_log_jump: float = 0.1
    boundary_radius: int = 3
    label_share: float = 0.3  # a region gets the silver label when this share of its pixels supports it


def class_ids(names: list[str]) -> dict[str, set[int]]:
    """Label ids (1-based, as in NYU ``labels``) for each class group."""
    groups: dict[str, set[int]] = {"reflective": set(), "structure": set(), "flat": set()}
    for index, name in enumerate(names, start=1):
        if name in REFLECTIVE_NAMES or REFLECTION_PATTERN.search(name):
            groups["reflective"].add(index)
        if name in STRUCTURE_NAMES:
            groups["structure"].add(index)
        if name in FLAT_NAMES:
            groups["flat"].add(index)
    return groups


def _disk(radius: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def silver_maps(labels: np.ndarray, instances: np.ndarray | None, depth: np.ndarray, groups: dict[str, set[int]], params: SilverParams | None = None) -> dict[str, np.ndarray]:
    """Per-pixel support for each silver category."""
    params = params or SilverParams()
    labels = labels.astype(np.int64)
    instances = np.zeros_like(labels) if instances is None else instances.astype(np.int64)
    segment = labels * 1000 + instances  # one id per annotated object (0 = unlabelled)
    annotated = labels > 0

    thin = np.zeros(labels.shape, dtype=bool)
    kernel = _disk(params.thin_width // 2)
    structure = np.isin(labels, list(groups["structure"]))
    for segment_id in np.unique(segment[annotated & ~structure]):
        mask = (segment == segment_id).astype(np.uint8)
        thin |= (mask > 0) & (cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) == 0)

    measured = np.isfinite(depth) & (depth > 0)
    log_depth = np.log(np.where(measured, depth, 1.0)).astype(np.float32)
    local_range = cv2.dilate(log_depth, np.ones((3, 3), np.uint8)) - cv2.erode(log_depth, np.ones((3, 3), np.uint8))
    segment_edge = (cv2.dilate(segment.astype(np.float32), np.ones((3, 3), np.uint8)) != cv2.erode(segment.astype(np.float32), np.ones((3, 3), np.uint8)))
    occluding = segment_edge & (local_range > params.boundary_log_jump) & measured & annotated
    boundary = cv2.dilate(occluding.astype(np.uint8), _disk(params.boundary_radius)) > 0

    return {
        "reflection": np.isin(labels, list(groups["reflective"])),
        "thin_object": cv2.dilate(thin.astype(np.uint8), _disk(2)) > 0,
        "boundary_mixing": boundary,
        "textureless": np.isin(labels, list(groups["flat"])),
        "annotated": annotated,
    }


def region_silver(maps: dict[str, np.ndarray], mask: np.ndarray, params: SilverParams | None = None) -> dict:
    params = params or SilverParams()
    shares = {category: float(maps[category][mask].mean()) for category in SILVER_CATEGORIES}
    return {
        "shares": shares,
        "labels": [category for category, share in shares.items() if share >= params.label_share],
        "annotated_share": float(maps["annotated"][mask].mean()),
    }
