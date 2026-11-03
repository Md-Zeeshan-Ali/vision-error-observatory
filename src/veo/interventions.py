"""Model-in-the-loop tests: edit the input, re-run the depth model, and measure the change in region error.

Each test compares regions VEO assigns to a hypothesis with matched control regions (other labels,
similar area, depth and starting error) that receive the same edit. If the hypothesis names a real
cause, the edit should reduce error more in the labelled regions than in their controls.

Statistics are on paired differences (labelled minus control change in error): mean with a
scene-level bootstrap interval, a sign-flip permutation p-value, Cohen's d_z, and Holm correction
across tests.
"""

from collections.abc import Callable

import cv2
import numpy as np

from .metrics import align_scale_shift

Predict = Callable[[np.ndarray], np.ndarray]


def feathered(mask: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """Soft version of a mask so edits have no hard seams (which would add edges of their own)."""
    return np.clip(cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma) * 1.5, 0.0, 1.0)


def add_texture(rgb: np.ndarray, mask: np.ndarray, amplitude: float = 25.0, seed: int = 0) -> np.ndarray:
    """Add band-limited, zero-mean grey texture inside the region (colour and mean brightness kept)."""
    noise = cv2.GaussianBlur(np.random.default_rng(seed).normal(0, 1, mask.shape).astype(np.float32), (0, 0), 1.2)
    noise *= amplitude / max(float(noise.std()), 1e-6)
    weight = feathered(mask)
    # Zero mean over the edited area, so the edit changes texture but not brightness
    noise -= float((noise * weight).sum() / max(weight.sum(), 1e-6))
    edited = rgb.astype(np.float32) + (noise * weight)[..., None]
    return np.clip(edited, 0, 255).astype(np.uint8)


def brighten(rgb: np.ndarray, mask: np.ndarray, gamma: float = 0.5) -> np.ndarray:
    """Gamma lift inside the region: dark pixels gain the most, highlights barely change."""
    lifted = 255.0 * (rgb.astype(np.float32) / 255.0) ** gamma
    weight = feathered(mask)[..., None]
    return np.clip(rgb * (1 - weight) + lifted * weight, 0, 255).astype(np.uint8)


def predict_at_scale(predict: Predict, rgb: np.ndarray, scale: float) -> np.ndarray:
    """Run the model on an upscaled image and bring the prediction back to the original grid."""
    height, width = rgb.shape[:2]
    upscaled = cv2.resize(rgb, (int(round(width * scale)), int(round(height * scale))), interpolation=cv2.INTER_CUBIC)
    return cv2.resize(predict(upscaled), (width, height), interpolation=cv2.INTER_AREA)


def region_error(prediction: np.ndarray, target: np.ndarray, valid: np.ndarray, mask: np.ndarray, prediction_space: str,
                 max_depth: float | None = None) -> float:
    """Mean absolute relative error in the region after aligning the whole prediction to the target."""
    aligned, _, _ = align_scale_shift(prediction, target, valid, prediction_space, max_depth)
    pixels = mask & valid
    return float(np.mean(np.abs(aligned[pixels] - target[pixels]) / target[pixels]))


def match_controls(targets: list[dict], candidates: list[dict], keys=("log_area", "mean_error", "log_depth"), seed: int = 0) -> list[tuple[dict, dict]]:
    """Greedy nearest-neighbour matching on standardised keys, without replacement, never the same region."""
    if not targets or not candidates:
        return []
    rng = np.random.default_rng(seed)
    pool = np.asarray([[c[k] for k in keys] for c in candidates], dtype=float)
    mean, std = pool.mean(axis=0), pool.std(axis=0) + 1e-9
    pool = (pool - mean) / std
    free = np.ones(len(candidates), dtype=bool)
    pairs = []
    for index in rng.permutation(len(targets)):
        target = targets[index]
        point = (np.asarray([target[k] for k in keys], dtype=float) - mean) / std
        distance = np.where(free, np.linalg.norm(pool - point, axis=1), np.inf)
        for same in (i for i, c in enumerate(candidates) if (c["sample_id"], c["region_id"]) == (target["sample_id"], target["region_id"])):
            distance[same] = np.inf
        best = int(np.argmin(distance))
        if not np.isfinite(distance[best]):
            break
        free[best] = False
        pairs.append((target, candidates[best]))
    return pairs


def paired_statistics(differences: np.ndarray, scenes: list[str], replicates: int = 2000, permutations: int = 10000, seed: int = 0) -> dict:
    """Mean paired difference with a scene bootstrap interval, sign-flip permutation p-value and Cohen's d_z."""
    differences = np.asarray(differences, dtype=float)
    if differences.size < 2:
        return {"n": int(differences.size), "mean": float(differences.mean()) if differences.size else None}
    rng = np.random.default_rng(seed)
    scene_array = np.asarray(scenes)
    unique = np.unique(scene_array)
    by_scene = [np.flatnonzero(scene_array == scene) for scene in unique]
    draws = [differences[np.concatenate([by_scene[i] for i in rng.integers(0, len(unique), len(unique))])].mean() for _ in range(replicates)]
    observed = abs(differences.mean())
    signs = rng.choice([-1.0, 1.0], size=(permutations, differences.size))
    p_value = (np.sum(np.abs((signs * differences).mean(axis=1)) >= observed) + 1) / (permutations + 1)
    spread = differences.std(ddof=1)
    return {
        "n": int(differences.size),
        "scenes": int(len(unique)),
        "mean": float(differences.mean()),
        "lower": float(np.percentile(draws, 2.5)),
        "upper": float(np.percentile(draws, 97.5)),
        "p_value": float(p_value),
        "cohens_dz": float(differences.mean() / spread) if spread > 0 else None,
    }


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values (monotone, capped at 1)."""
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted, running = {}, 0.0
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * value))
        adjusted[name] = running
    return adjusted
