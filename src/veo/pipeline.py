from dataclasses import asdict

import numpy as np

from .config import ExperimentConfig
from .features import FeatureParams, compute_feature_maps, error_context_features, region_features
from .label_model import load_label_model
from .metrics import align_scale_shift, compute_error_map, compute_metrics, compute_ratio_map
from .regions import extract_failure_regions
from .scoring import classify


def align_prediction(prediction: np.ndarray, target: np.ndarray, valid_mask: np.ndarray, config: ExperimentConfig) -> tuple[np.ndarray, float, float]:
    """Map a raw prediction to metric depth as configured; identity when alignment is disabled."""
    if not config.align_scale_shift:
        return prediction.astype(np.float32), 1.0, 0.0
    return align_scale_shift(prediction.astype(np.float32), target, valid_mask, config.prediction_space, config.max_prediction_m)


def failure_mask(aligned: np.ndarray, target: np.ndarray, valid_mask: np.ndarray, error_map: np.ndarray, config: ExperimentConfig) -> np.ndarray:
    if config.region_mode == "delta":
        return compute_ratio_map(aligned, target, valid_mask) > config.delta_threshold
    if config.region_mode == "percentile":
        return error_map > float(np.percentile(error_map[valid_mask], config.error_percentile))
    raise ValueError(f"Unknown region_mode: {config.region_mode}")


def compute_regions(target: np.ndarray, prediction: np.ndarray, valid_mask: np.ndarray, config: ExperimentConfig) -> dict:
    """Alignment, error maps and failure regions (with masks) for one sample."""
    valid_mask = valid_mask & np.isfinite(target) & (target > 0.0) & np.isfinite(prediction)
    if not valid_mask.any():
        raise ValueError("Cannot analyze a sample with no finite, positive target/prediction pixels")
    aligned, scale, shift = align_prediction(prediction, target, valid_mask, config)
    error_map = compute_error_map(aligned, target, valid_mask)
    failing = failure_mask(aligned, target, valid_mask, error_map, config) & valid_mask
    regions = extract_failure_regions(error_map, valid_mask, config.error_percentile, config.min_region_area, failing, config.opening_radius)
    return {"valid_mask": valid_mask, "aligned": aligned, "scale": scale, "shift": shift, "error_map": error_map, "failing": failing, "regions": regions}


def analyze_sample(
    rgb: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    valid_mask: np.ndarray,
    config: ExperimentConfig,
    geometry_depth: np.ndarray | None = None,
) -> dict:
    """Metrics, failure regions and per-region hypotheses for one sample.

    ``target`` is the depth errors are measured against (it may have holes). ``geometry_depth`` is a
    dense depth used for geometry cues; it defaults to ``target``, or to the aligned prediction
    when ``config.geometry_source == "prediction"``.
    """
    if rgb.shape[:2] != target.shape or target.shape != prediction.shape:
        raise ValueError("RGB, target, and prediction dimensions do not match")
    computed = compute_regions(target, prediction, valid_mask, config)
    valid_mask, failing, regions = computed["valid_mask"], computed["failing"], computed["regions"]
    aligned_prediction, scale, shift = computed["aligned"], computed["scale"], computed["shift"]
    metrics = compute_metrics(aligned_prediction, target, valid_mask)
    if config.geometry_source == "prediction":
        geometry = aligned_prediction
    elif config.geometry_source == "ground_truth":
        geometry = target if geometry_depth is None else geometry_depth
    else:
        raise ValueError(f"Unknown geometry_source: {config.geometry_source}")
    maps = compute_feature_maps(rgb, geometry, FeatureParams.from_config(config)) if regions else None
    settings = asdict(config)
    label_model = load_label_model(config.label_model)
    result_regions = []
    for region in regions:
        features = {**region_features(maps, region["mask"]), **error_context_features(region["mask"], aligned_prediction, target)}
        result_regions.append({
            "region_id": region["region_id"],
            "bbox": region["bbox"],
            "area": region["area"],
            "mean_error": region["mean_error"],
            "features": features,
            **(label_model.classify(features, settings) if label_model else classify(features, config.include_reflection, settings)),
        })
    return {
        "metrics": metrics,
        "alignment": {
            "scale": scale,
            "shift": shift,
            "enabled": config.align_scale_shift,
            "prediction_space": config.prediction_space,
        },
        "failure_rule": {"mode": config.region_mode, "delta_threshold": config.delta_threshold, "error_percentile": config.error_percentile},
        "valid_pixels": int(valid_mask.sum()),
        "failure_pixel_fraction": float(failing.sum() / valid_mask.sum()),
        "num_regions": len(result_regions),
        "regions": result_regions,
        "config": settings,
    }
