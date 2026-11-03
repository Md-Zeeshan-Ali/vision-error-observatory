"""Per-region evidence features.

Feature maps are computed once per image and then averaged inside each failure region.
Geometry cues use log depth, so they do not depend on the scene's absolute scale.

  grad_energy          mean Sobel gradient of the lightly smoothed grey image (grey levels per pixel)
  corner_strength      mean Shi-Tomasi minimum eigenvalue (structure in two directions)
  depth_edge_fraction  share of region pixels within ``edge_radius`` px of a depth edge, where a depth
                       edge has |grad log depth| > ``edge_log_threshold`` per pixel
  depth_step           90th percentile of |grad log depth| on the ring around the region
  thin_fraction        share of region pixels on, or next to, a thin structure nearer than its
                       surroundings (white top-hat of disparity relative to disparity)
  thin_image_fraction  share of region pixels on, or next to, a thin bright or dark line in the image
                       (top-hat / black-hat of grey level); used when geometry comes from the
                       prediction, which often smooths thin objects away
  highlight_ratio      share of pixels that are bright, unsaturated and brighter than their surroundings
  clipped_fraction     share of pixels at the ends of the grey range (under- or over-exposed)
  brightness, contrast mean and standard deviation of grey level
  aspect_ratio         bounding-box aspect ratio (informational)
  mean_depth_m         mean geometry depth in the region
  coarse_grad          gradient energy after heavier smoothing (large-scale structure)
  value_p95            95th percentile of HSV value
  saturation_mean      mean HSV saturation
  log_area             log of the region's pixel count

Features that need the error target are added by ``error_context_features``:

  signed_error         mean (prediction - truth) / truth; positive means the surface is predicted too far
  sensor_hole_fraction share of the region's neighbourhood with no depth measurement
"""

from dataclasses import dataclass, fields

import cv2
import numpy as np


@dataclass(frozen=True)
class FeatureParams:
    edge_log_threshold: float = 0.05
    edge_radius: int = 3
    thin_kernel: int = 9
    thin_relative_threshold: float = 0.1
    highlight_value: int = 200
    highlight_saturation: int = 50
    highlight_local_delta: int = 25
    thin_image_delta: int = 25

    @classmethod
    def from_config(cls, config) -> "FeatureParams":
        return cls(**{field.name: getattr(config, field.name) for field in fields(cls)})


@dataclass
class FeatureMaps:
    gray: np.ndarray
    grad: np.ndarray
    corner: np.ndarray
    depth: np.ndarray
    log_depth_grad: np.ndarray
    near_depth_edge: np.ndarray
    near_thin: np.ndarray
    near_thin_image: np.ndarray
    highlight: np.ndarray
    clipped: np.ndarray
    coarse_grad: np.ndarray
    value: np.ndarray
    saturation: np.ndarray


def _disk(radius: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def compute_feature_maps(rgb: np.ndarray, depth: np.ndarray, params: FeatureParams | None = None) -> FeatureMaps:
    params = params or FeatureParams()
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    smoothed = cv2.GaussianBlur(gray, (0, 0), 1.0)
    # A 3x3 Sobel responds with 8x the per-pixel slope
    grad = np.hypot(cv2.Sobel(smoothed, cv2.CV_32F, 1, 0), cv2.Sobel(smoothed, cv2.CV_32F, 0, 1)) / 8.0
    corner = cv2.cornerMinEigenVal(smoothed, blockSize=5, ksize=3)
    coarse = cv2.GaussianBlur(gray, (0, 0), 4.0)
    coarse_grad = np.hypot(cv2.Sobel(coarse, cv2.CV_32F, 1, 0), cv2.Sobel(coarse, cv2.CV_32F, 0, 1)) / 8.0

    depth = depth.astype(np.float32)
    measured = np.isfinite(depth) & (depth > 0)
    log_depth = np.log(np.where(measured, depth, 1.0)).astype(np.float32)
    log_grad = np.hypot(cv2.Sobel(log_depth, cv2.CV_32F, 1, 0), cv2.Sobel(log_depth, cv2.CV_32F, 0, 1)) / 8.0
    # Ignore gradients that touch unmeasured pixels
    log_grad[cv2.erode(measured.astype(np.uint8), np.ones((3, 3), np.uint8)) == 0] = 0.0
    edges = (log_grad > params.edge_log_threshold).astype(np.uint8)
    near_edge = cv2.dilate(edges, _disk(params.edge_radius)) > 0

    disparity = np.where(measured, 1.0 / np.where(measured, depth, 1.0), 0.0).astype(np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params.thin_kernel, params.thin_kernel))
    tophat = cv2.morphologyEx(disparity, cv2.MORPH_TOPHAT, kernel)
    thin = (tophat / np.maximum(disparity, 1e-6) > params.thin_relative_threshold) & measured
    near_thin = cv2.dilate(thin.astype(np.uint8), _disk(2)) > 0
    line_strength = np.maximum(
        cv2.morphologyEx(smoothed, cv2.MORPH_TOPHAT, kernel), cv2.morphologyEx(smoothed, cv2.MORPH_BLACKHAT, kernel)
    )
    near_thin_image = cv2.dilate((line_strength > params.thin_image_delta).astype(np.uint8), _disk(2)) > 0

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    value, saturation = hsv[:, :, 2], hsv[:, :, 1]
    local = value.astype(np.int16) - cv2.medianBlur(value, 21).astype(np.int16)
    highlight = (value >= params.highlight_value) & (saturation <= params.highlight_saturation) & (local >= params.highlight_local_delta)
    clipped = (gray >= 250) | (gray <= 5)
    return FeatureMaps(gray, grad, corner, depth, log_grad, near_edge, near_thin, near_thin_image, highlight, clipped, coarse_grad, value, saturation)


def region_features(maps: FeatureMaps, mask: np.ndarray) -> dict[str, float]:
    if not mask.any():
        raise ValueError("Region mask is empty")
    ys, xs = np.where(mask)
    width, height = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
    # The boundary ring reaches 2 px past the region, so a 3 px margin around its box is exact
    crop = (slice(max(ys.min() - 3, 0), ys.max() + 4), slice(max(xs.min() - 3, 0), xs.max() + 4))
    local = mask[crop].astype(np.uint8)
    ring = (cv2.dilate(local, np.ones((5, 5), np.uint8)) > 0) & ~(cv2.erode(local, np.ones((5, 5), np.uint8)) > 0)
    ring_gradients = maps.log_depth_grad[crop][ring]
    depth = maps.depth[mask]
    depth = depth[np.isfinite(depth) & (depth > 0)]
    gray = maps.gray[mask]
    return {
        "grad_energy": float(maps.grad[mask].mean()),
        "corner_strength": float(maps.corner[mask].mean()),
        "depth_edge_fraction": float(maps.near_depth_edge[mask].mean()),
        "depth_step": float(np.percentile(ring_gradients, 90)) if ring_gradients.size else 0.0,
        "thin_fraction": float(maps.near_thin[mask].mean()),
        "thin_image_fraction": float(maps.near_thin_image[mask].mean()),
        "highlight_ratio": float(maps.highlight[mask].mean()),
        "clipped_fraction": float(maps.clipped[mask].mean()),
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "aspect_ratio": float(max(width, height) / max(min(width, height), 1)),
        "mean_depth_m": float(depth.mean()) if depth.size else 0.0,
        "coarse_grad": float(maps.coarse_grad[mask].mean()),
        "value_p95": float(np.percentile(maps.value[mask], 95)),
        "saturation_mean": float(maps.saturation[mask].mean()),
        "log_area": float(np.log(int(mask.sum()))),
    }


def error_context_features(mask: np.ndarray, aligned: np.ndarray, target: np.ndarray, radius: int = 7) -> dict[str, float]:
    """Direction of the region's error, and how much of its surroundings the depth sensor could not measure."""
    measured = np.isfinite(target) & (target > 0)
    pixels = mask & measured & np.isfinite(aligned)
    ys, xs = np.where(mask)
    crop = (slice(max(ys.min() - radius, 0), ys.max() + radius + 1), slice(max(xs.min() - radius, 0), xs.max() + radius + 1))
    neighbourhood = cv2.dilate(mask[crop].astype(np.uint8), _disk(radius)) > 0
    return {
        "signed_error": float(np.mean((aligned[pixels] - target[pixels]) / target[pixels])) if pixels.any() else 0.0,
        "sensor_hole_fraction": float(1.0 - measured[crop][neighbourhood].mean()),
    }


def extract_features(rgb: np.ndarray, depth: np.ndarray, region: dict, params: FeatureParams | None = None) -> dict[str, float]:
    """Features for one region; compute the maps once with compute_feature_maps when scoring many."""
    return region_features(compute_feature_maps(rgb, depth, params), region["mask"])
