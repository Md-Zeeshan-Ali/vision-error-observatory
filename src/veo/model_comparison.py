"""Paired comparison of depth models over one split (docs/MODEL_COMPARISON_PROTOCOL.md)."""

from collections import defaultdict

import numpy as np

from .interventions import holm, paired_statistics

LABELS = ("textureless", "boundary_mixing", "thin_object", "reflection", "poor_lighting", "ambiguous")


def image_outcomes(record: dict) -> dict[str, float]:
    valid = record["valid_pixels"]
    outcomes = {
        "abs_rel": record["metrics"]["abs_rel"],
        "delta1": record["metrics"]["delta1"],
        "failing_pixels": record["failure_pixel_fraction"],
    }
    for label in LABELS:
        outcomes[f"pixels_{label}"] = sum(r["area"] for r in record["regions"] if r["top_cause"] == label) / valid
    return outcomes


def compare(first: list[dict], second: list[dict], scene_of: dict[str, str], seed: int = 42) -> dict:
    """first minus second, on images both models have; each scene counts once."""
    outcomes_a = {record["sample_id"]: image_outcomes(record) for record in first}
    outcomes_b = {record["sample_id"]: image_outcomes(record) for record in second}
    shared = sorted(outcomes_a.keys() & outcomes_b.keys())
    if not shared:
        raise ValueError("The two models share no images")
    by_scene: dict[str, list[str]] = defaultdict(list)
    for sample_id in shared:
        by_scene[scene_of.get(sample_id, sample_id)].append(sample_id)
    scenes = sorted(by_scene)
    result = {}
    for name in outcomes_a[shared[0]]:
        differences = np.asarray([np.mean([outcomes_a[i][name] - outcomes_b[i][name] for i in by_scene[scene]]) for scene in scenes])
        result[name] = {
            **paired_statistics(differences, scenes, seed=seed),
            "first_mean": float(np.mean([outcomes_a[i][name] for i in shared])),
            "second_mean": float(np.mean([outcomes_b[i][name] for i in shared])),
        }
    adjusted = holm({name: row["p_value"] for name, row in result.items() if "p_value" in row})
    for name, value in adjusted.items():
        row = result[name]
        row["p_holm"] = value
        row["reliable"] = bool(value < 0.05 and (row["upper"] < 0 or row["lower"] > 0))
    return {"images": len(shared), "scenes": len(scenes), "outcomes": result}
