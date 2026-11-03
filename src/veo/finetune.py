"""Fine-tuning guided by VEO's diagnosis (docs/FINETUNE_PROTOCOL.md): loss weights, per-image outcomes and contrasts.

Ground truth is used to find the training regions, which is legitimate at training time. The
fine-tuned model needs none at inference.
"""

from collections import defaultdict

import numpy as np

from .interventions import paired_statistics
from .silver import SILVER_CATEGORIES

LOSS_SHARE = 0.2  # targeted pixels carry this share of the loss across the training set


def target_weight(prevalence: float, share: float = LOSS_SHARE) -> float:
    """Weight that gives pixels with this prevalence the given share of the total loss weight."""
    if not 0.0 < prevalence < 1.0:
        raise ValueError(f"Prevalence must be between 0 and 1, got {prevalence}")
    return share * (1.0 - prevalence) / (prevalence * (1.0 - share))


def derangement(count: int, seed: int) -> np.ndarray:
    """A random permutation with no fixed points: every image gets another image's mask."""
    if count < 2:
        raise ValueError("A derangement needs at least two items")
    order = np.random.default_rng(seed).permutation(count)
    result = np.empty(count, dtype=np.int64)
    result[order] = np.roll(order, 1)
    return result


def scale_shift_loss(prediction, disparity, valid, weights=None):
    """Mean over images of a weighted L1 error in disparity after a per-image affine fit.

    The fit is unweighted, as in evaluation, so weights change what is learned and not how it is
    scored. Each image is divided by its mean disparity so near and far scenes count alike, and
    averaged over valid pixels so up-weighting adds to the loss rather than renormalising it.
    """
    losses = []
    for index in range(len(prediction)):
        mask = valid[index]
        x, y = prediction[index][mask], disparity[index][mask]
        x_mean, y_mean = x.mean(), y.mean()
        scale = ((x - x_mean) * (y - y_mean)).sum() / (x - x_mean).pow(2).sum().clamp_min(1e-8)
        error = (scale * (x - x_mean) + y_mean - y).abs() / y_mean
        losses.append((error * weights[index][mask]).mean() if weights is not None else error.mean())
    return sum(losses) / len(losses)


def image_outcomes(aligned: np.ndarray, target: np.ndarray, valid: np.ndarray, failing: np.ndarray, metrics: dict, maps: dict[str, np.ndarray]) -> dict:
    """Overall metrics and error on annotated pixels of each category; None where an image has none."""
    relative = np.abs(aligned - target) / np.where(target > 0, target, 1.0)
    outcomes = {"abs_rel": metrics["abs_rel"], "delta1": metrics["delta1"], "failing_pixels": float(failing.sum() / valid.sum())}
    for category in SILVER_CATEGORIES:
        mask = maps[category] & valid
        outcomes[f"abs_rel_{category}"] = float(relative[mask].mean()) if mask.any() else None
        outcomes[f"failing_{category}"] = float(failing[mask].mean()) if mask.any() else None
    return outcomes


def _unit_differences(first: dict[str, dict], second: dict[str, dict], outcome: str, scene_of: dict[str, str]) -> dict[str, float]:
    by_scene: dict[str, list[float]] = defaultdict(list)
    for sample_id in first.keys() & second.keys():
        a, b = first[sample_id][outcome], second[sample_id][outcome]
        if a is not None and b is not None:
            by_scene[scene_of.get(sample_id, sample_id)].append(a - b)
    return {scene: float(np.mean(values)) for scene, values in by_scene.items()}


def _across_seeds(runs: dict[int, dict[str, dict]]) -> dict[str, dict]:
    """Each image's outcomes averaged over the seeds of one condition."""
    collected: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for records in runs.values():
        for sample_id, outcomes in records.items():
            for name, value in outcomes.items():
                if value is not None:
                    collected[sample_id][name].append(value)
    names = {name for records in runs.values() for outcomes in records.values() for name in outcomes}
    return {sample_id: {name: float(np.mean(values[name])) if values.get(name) else None for name in names} for sample_id, values in collected.items()}


def contrast(first: dict[int, dict[str, dict]], second: dict[int, dict[str, dict]], outcome: str, scene_of: dict[str, str], seed: int = 42) -> dict:
    """first minus second, with scenes as the unit, and the same difference for each paired seed.

    Scene statistics average over training seeds, so they don't reflect how much training varies.
    ``seeds_agree`` is the check on that: every seed pair must show the same direction.
    """
    differences = _unit_differences(_across_seeds(first), _across_seeds(second), outcome, scene_of)
    scenes = sorted(differences)
    per_seed = {}
    for training_seed in sorted(first.keys() & second.keys()):
        values = _unit_differences(first[training_seed], second[training_seed], outcome, scene_of)
        per_seed[str(training_seed)] = float(np.mean(list(values.values())))
    signs = {np.sign(value) for value in per_seed.values()}
    return {
        **paired_statistics(np.asarray([differences[scene] for scene in scenes]), scenes, seed=seed),
        "per_seed": per_seed,
        "seeds_agree": len(signs) == 1 and 0.0 not in signs,
    }
