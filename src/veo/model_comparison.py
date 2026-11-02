"""Paired comparison of depth models over one split (docs/MODEL_COMPARISON_PROTOCOL.md)."""

from collections import defaultdict

import numpy as np

from .interventions import holm, paired_statistics

LABELS = ("textureless", "boundary_mixing", "thin_object", "reflection", "poor_lighting", "ambiguous")
FINGERPRINT_CONTRASTS = ("family_dav2_small_vs_dpt_hybrid", "backbone_hybrid_vs_large")
VALIDATED_AUROC = 0.7


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


def image_composition(record: dict) -> dict[str, float] | None:
    """Each label's share of the image's region pixels; None for an image with no regions."""
    total = sum(region["area"] for region in record["regions"])
    if not total:
        return None
    return {label: sum(r["area"] for r in record["regions"] if r["top_cause"] == label) / total for label in LABELS}


def _paired(outcomes_a: dict[str, dict], outcomes_b: dict[str, dict], scene_of: dict[str, str], seed: int) -> dict:
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


def compare(first: list[dict], second: list[dict], scene_of: dict[str, str], seed: int = 42) -> dict:
    """first minus second, on images both models have; each scene counts once."""
    return _paired(
        {record["sample_id"]: image_outcomes(record) for record in first},
        {record["sample_id"]: image_outcomes(record) for record in second},
        scene_of, seed,
    )


def compare_composition(first: list[dict], second: list[dict], scene_of: dict[str, str], seed: int = 42) -> dict:
    """The same test on the mix of labels, on images where both models have regions."""
    shares_a, shares_b = ({record["sample_id"]: image_composition(record) for record in records} for records in (first, second))
    return _paired(
        {sample_id: shares for sample_id, shares in shares_a.items() if shares is not None},
        {sample_id: shares for sample_id, shares in shares_b.items() if shares is not None},
        scene_of, seed,
    )


def fingerprint_verdict(by_dataset: dict[str, dict], label_auroc: dict[str, dict[str, float | None]]) -> dict:
    """Apply the pre-registered rule: a label passes only if reliable with one sign on every dataset."""
    report = {}
    for contrast in FINGERPRINT_CONTRASTS:
        labels = {}
        for label in LABELS:
            rows = {dataset: comparison["contrasts"][contrast]["composition"]["outcomes"][label] for dataset, comparison in by_dataset.items()}
            signs = {np.sign(row["mean"]) for row in rows.values()}
            passes = len(signs) == 1 and all(row.get("reliable", False) for row in rows.values())
            aurocs = label_auroc.get(label, {})
            labels[label] = {
                "passes": passes,
                "failure_type_difference": passes and bool(aurocs) and all(value is not None and value >= VALIDATED_AUROC for value in aurocs.values()),
                "datasets": {dataset: {key: row.get(key) for key in ("mean", "lower", "upper", "p_holm", "reliable")} for dataset, row in rows.items()},
            }
        report[contrast] = {"verdict": "supported" if any(entry["passes"] for entry in labels.values()) else "cut", "labels": labels}
    return report
