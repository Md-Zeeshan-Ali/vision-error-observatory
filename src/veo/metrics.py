import numpy as np
from scipy.stats import spearmanr


def _evaluation_mask(prediction: np.ndarray, target: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    if prediction.shape != target.shape or valid_mask.shape != target.shape:
        raise ValueError("Prediction, target, and valid_mask must have identical shapes")
    return valid_mask & np.isfinite(prediction) & np.isfinite(target) & (target > 0.0)


def _to_depth(aligned: np.ndarray, prediction_space: str, max_depth: float | None) -> np.ndarray:
    """Aligned values as depth, capped at ``max_depth``.

    An affine fit in disparity can go to zero or below for some pixels; without a cap those pixels
    become arbitrarily deep and dominate mean errors.
    """
    if prediction_space == "inverse_depth":
        floor = 1.0 / max_depth if max_depth else 1e-6
        return 1.0 / np.clip(aligned, floor, None)
    return np.clip(aligned, 1e-3, max_depth) if max_depth else aligned


def align_scale_shift(
    prediction: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray,
    prediction_space: str = "depth",
    max_depth: float | None = None,
) -> tuple[np.ndarray, float, float]:
    if prediction_space not in {"depth", "inverse_depth"}:
        raise ValueError(f"Unsupported prediction space: {prediction_space}")
    evaluation_mask = _evaluation_mask(prediction, target, valid_mask)
    if int(evaluation_mask.sum()) < 2:
        return prediction.astype(np.float32), 1.0, 0.0
    p = prediction[evaluation_mask].astype(np.float64)
    t = target[evaluation_mask].astype(np.float64)
    if prediction_space == "inverse_depth":
        t = 1.0 / t
    design = np.stack([p, np.ones_like(p)], axis=1)
    scale, shift = np.linalg.lstsq(design, t, rcond=None)[0]
    aligned = _to_depth(prediction * scale + shift, prediction_space, max_depth)
    return aligned.astype(np.float32), float(scale), float(shift)


def align_scale_only(
    prediction: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray,
    prediction_space: str = "inverse_depth",
    max_depth: float | None = None,
) -> tuple[np.ndarray, float]:
    if prediction_space not in {"depth", "inverse_depth"}:
        raise ValueError(f"Unsupported prediction space: {prediction_space}")
    evaluation_mask = _evaluation_mask(prediction, target, valid_mask)
    if int(evaluation_mask.sum()) < 2:
        return prediction.astype(np.float32), 1.0
    p = prediction[evaluation_mask].astype(np.float64)
    t = target[evaluation_mask].astype(np.float64)
    if prediction_space == "inverse_depth":
        t = 1.0 / t
    denominator = float(np.dot(p, p))
    if denominator <= 1e-12:
        return prediction.astype(np.float32), 1.0
    scale = float(np.dot(p, t) / denominator)
    aligned = _to_depth(prediction * scale, prediction_space, max_depth)
    return aligned.astype(np.float32), scale


def compute_metrics(prediction: np.ndarray, target: np.ndarray, valid_mask: np.ndarray) -> dict[str, float]:
    evaluation_mask = _evaluation_mask(prediction, target, valid_mask)
    if not evaluation_mask.any():
        raise ValueError("Cannot compute metrics with an empty valid mask")
    pred = prediction[evaluation_mask].astype(np.float64)
    truth = target[evaluation_mask].astype(np.float64)
    safe_pred = np.clip(pred, 1e-6, None)
    abs_rel = np.mean(np.abs(pred - truth) / truth)
    sq_rel = np.mean((pred - truth) ** 2 / truth)
    mae = np.mean(np.abs(pred - truth))
    rmse = np.sqrt(np.mean((pred - truth) ** 2))
    ratio = np.maximum(safe_pred / truth, truth / safe_pred)
    spearman = spearmanr(pred, truth).statistic if np.std(pred) and np.std(truth) else 0.0
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "abs_rel": float(abs_rel),
        "sq_rel": float(sq_rel),
        "delta1": float(np.mean(ratio < 1.25)),
        "delta2": float(np.mean(ratio < 1.25**2)),
        "delta3": float(np.mean(ratio < 1.25**3)),
        "spearman": float(spearman) if np.isfinite(spearman) else 0.0,
    }


def compute_error_map(prediction: np.ndarray, target: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    error_map = np.zeros_like(target, dtype=np.float32)
    evaluation_mask = _evaluation_mask(prediction, target, valid_mask)
    error_map[evaluation_mask] = np.abs(prediction[evaluation_mask] - target[evaluation_mask]) / target[evaluation_mask]
    return error_map


def compute_ratio_map(prediction: np.ndarray, target: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Per-pixel max(pred/gt, gt/pred), the quantity thresholded by the delta accuracy metrics; 1 where invalid."""
    ratio = np.ones_like(target, dtype=np.float32)
    evaluation_mask = _evaluation_mask(prediction, target, valid_mask)
    pred = np.clip(prediction[evaluation_mask], 1e-6, None)
    truth = target[evaluation_mask]
    ratio[evaluation_mask] = np.maximum(pred / truth, truth / pred)
    return ratio


def tolerant_failure(prediction: np.ndarray, target: np.ndarray, failing: np.ndarray, radius: int, threshold: float = 1.25) -> np.ndarray:
    """Failing pixels that still fail when the prediction may be off by up to ``radius`` px.

    A pixel is kept only if no predicted value within the (2r+1)x(2r+1) window matches its measured
    depth within ``threshold``. Failures that vanish under a 1-3 px tolerance are what slight
    misregistration between image and sensor would produce at depth edges.
    """
    if radius < 0:
        raise ValueError(f"Radius must be non-negative, got {radius}")
    height, width = target.shape
    padded = np.pad(prediction.astype(np.float32), radius, constant_values=np.nan)
    truth = np.where(failing, target, np.nan).astype(np.float32)
    matched = np.zeros(target.shape, dtype=bool)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            shifted = np.clip(padded[dy:dy + height, dx:dx + width], 1e-6, None)
            with np.errstate(invalid="ignore"):
                matched |= np.maximum(shifted / truth, truth / shifted) <= threshold
    return failing & ~matched
