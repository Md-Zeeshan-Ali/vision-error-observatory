"""Rule-based failure hypotheses with an abstention gate.

Each score is a linear ramp on one feature, clipped to [0, 1]. The scales are hand-set defaults
that are calibrated on training data in a later phase; they were not tuned on test data.
Depth occlusions are covered by ``boundary_mixing``: from a single image the two are not separable.
"""

import numpy as np

CATEGORIES = ("textureless", "boundary_mixing", "thin_object", "reflection", "poor_lighting")

DEFAULTS = {
    "texture_grad_scale": 8.0,
    "boundary_fraction_scale": 0.5,
    "thin_fraction_scale": 0.3,
    "reflection_scale": 5.0,
    "dark_level": 50.0,
    "min_score": 0.4,
    "ambiguity_margin": 0.10,
    "label_threshold": 0.5,
    "geometry_source": "ground_truth",
}


def clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def thin_evidence(features: dict[str, float], geometry_source: str) -> float:
    """Thin structures from depth; with predicted geometry also from image lines, since a model that
    misses a thin object usually removes it from its own depth map as well."""
    if geometry_source == "prediction":
        return max(features["thin_fraction"], features.get("thin_image_fraction", 0.0))
    return features["thin_fraction"]


def score_categories(features: dict[str, float], include_reflection: bool = True, thresholds: dict | None = None) -> dict[str, float]:
    settings = {**DEFAULTS, **(thresholds or {})}
    dark = clip01((settings["dark_level"] - features["brightness"]) / settings["dark_level"])
    scores = {
        "textureless": clip01(1.0 - features["grad_energy"] / settings["texture_grad_scale"]),
        "boundary_mixing": clip01(features["depth_edge_fraction"] / settings["boundary_fraction_scale"]),
        "thin_object": clip01(thin_evidence(features, settings["geometry_source"]) / settings["thin_fraction_scale"]),
        "poor_lighting": max(dark, clip01(2.0 * features["clipped_fraction"])),
    }
    if include_reflection:
        scores["reflection"] = clip01(features["highlight_ratio"] * settings["reflection_scale"])
    return scores


def classify(features: dict[str, float], include_reflection: bool = True, thresholds: dict | None = None) -> dict:
    """Top hypothesis, or ``ambiguous`` when the top score is weak or a runner-up is close.

    ``labels`` lists every category whose score reaches ``label_threshold`` (regions can have several causes).
    """
    settings = {**DEFAULTS, **(thresholds or {})}
    scores = score_categories(features, include_reflection, settings)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_cause, top_score = ranked[0]
    margin = top_score - (ranked[1][1] if len(ranked) > 1 else 0.0)
    ambiguous = top_score < settings["min_score"] or margin < settings["ambiguity_margin"]
    confidence = "Low" if ambiguous else "High" if top_score > 0.6 else "Medium"
    return {
        "scores": scores,
        "top_cause": "ambiguous" if ambiguous else top_cause,
        "labels": [name for name, score in ranked if score >= settings["label_threshold"]],
        "confidence": confidence,
        "score_margin": float(margin),
    }
