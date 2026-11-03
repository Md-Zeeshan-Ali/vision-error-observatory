import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from .data import load_depth, load_rgb, make_valid_mask
from .metrics import align_scale_shift, compute_error_map
from .scoring import classify

ABSTAIN_LABEL = "ambiguous"

# Value each signal is reset to when it is ablated; each entry changes exactly one feature
NEUTRAL_VALUES = {
    "grad_energy": 8.0,  # the default texture scale: textureless score 0
    "depth_edge_fraction": 0.0,
    "thin_fraction": 0.0,
    "highlight_ratio": 0.0,
    "brightness": 128.0,
    "clipped_fraction": 0.0,
}


def cohens_kappa(first: list[str], second: list[str]) -> float:
    if len(first) != len(second) or not first:
        raise ValueError("Both label lists must be non-empty and have equal length")
    total = len(first)
    categories = set(first) | set(second)
    observed = sum(left == right for left, right in zip(first, second, strict=True)) / total
    first_counts = Counter(first)
    second_counts = Counter(second)
    expected = sum((first_counts[label] / total) * (second_counts[label] / total) for label in categories)
    if expected == 1.0:
        return 1.0
    return float((observed - expected) / (1.0 - expected))


def labels_from_confusion(confusion: dict[str, dict[str, int]]) -> tuple[list[str], list[str]]:
    """Expand a {reference: {predicted: count}} matrix back into paired label lists."""
    reference, predicted = [], []
    for ref_label, row in confusion.items():
        for pred_label, count in row.items():
            reference.extend([ref_label] * count)
            predicted.extend([pred_label] * count)
    return reference, predicted


def per_class_report(reference: list[str], predicted: list[str]) -> dict[str, dict[str, float]]:
    classes = sorted(set(reference) | set(predicted))
    report = {}
    for label in classes:
        true_positive = sum(r == label and p == label for r, p in zip(reference, predicted, strict=True))
        predicted_count = sum(p == label for p in predicted)
        support = sum(r == label for r in reference)
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        report[label] = {
            "correct": true_positive,
            "predicted": predicted_count,
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return report


def agreement_report(reference: list[str], predicted: list[str], abstain_label: str = ABSTAIN_LABEL) -> dict:
    """Agreement of predicted labels with a reference rater, over all items and over non-abstained items.

    Selective metrics are reported together with coverage and with the accuracy of always
    predicting the reference's most frequent label on the same subset.
    """
    if len(reference) != len(predicted) or not reference:
        raise ValueError("reference and predicted must be non-empty and have equal length")
    total = len(reference)
    per_class = per_class_report(reference, predicted)
    committed = [(r, p) for r, p in zip(reference, predicted, strict=True) if p != abstain_label]
    report = {
        "n": total,
        "accuracy": sum(r == p for r, p in zip(reference, predicted, strict=True)) / total,
        "kappa": cohens_kappa(reference, predicted),
        "macro_f1": float(np.mean([row["f1"] for row in per_class.values()])),
        "per_class": per_class,
        "abstained": total - len(committed),
        "coverage": len(committed) / total,
        "selective": None,
    }
    if committed:
        ref_committed = [r for r, _ in committed]
        pred_committed = [p for _, p in committed]
        majority_label, majority_count = Counter(ref_committed).most_common(1)[0]
        report["selective"] = {
            "n": len(committed),
            "accuracy": sum(r == p for r, p in committed) / len(committed),
            "kappa": cohens_kappa(ref_committed, pred_committed),
            "majority_label": majority_label,
            "majority_baseline_accuracy": majority_count / len(committed),
        }
    return report


def model_neutral_values(label_model) -> dict[str, float]:
    """Each feature of a fitted label model at its training mean, plus the rule-based lighting signals."""
    neutral = {name: NEUTRAL_VALUES[name] for name in ("brightness", "clipped_fraction")}
    for model in label_model.categories.values():
        neutral.update(zip(model["features"], model["mean"], strict=True))
    return neutral


def ablation_change(records: list[dict], dropped_signal: str, include_reflection: bool = True, label_model=None, settings: dict | None = None) -> dict[str, int]:
    neutral_values = model_neutral_values(label_model) if label_model else NEUTRAL_VALUES
    if dropped_signal not in neutral_values:
        raise ValueError(f"Unsupported ablation signal: {dropped_signal}")
    changed = 0
    total = 0
    for record in records:
        for region in record["regions"]:
            features = dict(region["features"])
            if dropped_signal not in features:
                continue
            features[dropped_signal] = neutral_values[dropped_signal]
            new_label = label_model.classify(features, settings)["top_cause"] if label_model else classify(features, include_reflection)["top_cause"]
            changed += new_label != region["top_cause"]
            total += 1
    return {"changed_regions": changed, "total_regions": total}


def export_rating_set(
    results_path: Path,
    rgb_dir: Path,
    depth_dir: Path,
    prediction_dir: Path,
    output_dir: Path,
    count: int = 200,
    seed: int = 42,
    prediction_space: str | None = "inverse_depth",
    max_depth_m: float = 10.0,
) -> int:
    """Export blinded rating panels.

    Raw predictions are aligned to metric depth (``prediction_space`` of the model output, or
    None if already metric) so the prediction and error panels match what VEO analyzed.
    """
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    regions = [(record["sample_id"], region) for record in payload["records"] for region in record["regions"]]
    if not regions:
        raise ValueError("Results file contains no regions")
    regions.sort(key=lambda item: item[1]["mean_error"])
    indices = np.linspace(0, len(regions) - 1, min(count, len(regions)), dtype=int)
    selected = [regions[index] for index in indices]
    rng = np.random.default_rng(seed)
    rng.shuffle(selected)
    panel_dir = output_dir / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for rating_index, (sample_id, region) in enumerate(selected, start=1):
        rgb = load_rgb(rgb_dir / f"{sample_id}.png")
        depth = load_depth(depth_dir / f"{sample_id}.npy")
        valid = make_valid_mask(depth, max_depth_m)
        prediction = np.asarray(np.load(prediction_dir / f"{sample_id}.npy"), dtype=np.float32)
        if prediction_space is not None:
            prediction, _, _ = align_scale_shift(prediction, depth, valid, prediction_space, max_depth_m)
        error = compute_error_map(prediction, depth, valid)
        x, y, width, height = region["bbox"]
        slices = (slice(y, y + height), slice(x, x + width))
        figure = Figure(figsize=(12, 3))  # no pyplot, so no GUI backend is needed
        axes = figure.subplots(1, 4)
        axes[0].imshow(rgb[slices])
        axes[0].set_title("RGB")
        axes[1].imshow(depth[slices], cmap="magma")
        axes[1].set_title("Ground truth")
        axes[2].imshow(prediction[slices], cmap="magma")
        axes[2].set_title("Prediction (aligned)")
        axes[3].imshow(error[slices], cmap="inferno")
        axes[3].set_title("Relative error")
        for axis in axes:
            axis.axis("off")
        panel_path = panel_dir / f"rating_{rating_index:04d}.png"
        figure.tight_layout()
        figure.savefig(panel_path, dpi=120)
        rows.append({"rating_id": f"rating_{rating_index:04d}", "panel_path": str(panel_path), "sample_id": sample_id, "human_label": "", "human_confidence": "", "notes": ""})
    with (output_dir / "rating_template.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "rater_instructions.md").write_text(
        "# VEO Blind Region Rating\n\nAssign one primary label to each panel without inspecting VEO predictions or scores.\n\n"
        "Allowed labels: `textureless`, `boundary_mixing` (including occlusion edges), `thin_object`, `poor_lighting`, `reflection`, `ambiguous`.\n\n"
        "Use `ambiguous` when evidence is insufficient or multiple labels are equally plausible. Record confidence as `high`, `medium`, or `low`. Do not infer the label from the filename.\n",
        encoding="utf-8",
    )
    return len(rows)
