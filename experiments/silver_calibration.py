"""Objective-label evaluation and calibration of VEO's failure hypotheses (Phase 3).

For each split, region masks are recomputed from the cached predictions (results files store only
bounding boxes), checked against the stored regions, and given silver labels from NYU's semantic
and instance annotations. Models are fitted on the train split and scored on the test split.
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from veo.calibration import build_rows, evaluate, fit, public_models
from veo.label_model import load_label_model
from veo.reports import write_result
from veo.silver import SILVER_CATEGORIES


def top_label_precision(rows: list[dict]) -> dict:
    """How often VEO's top label is confirmed by the silver labels, for categories that have them."""
    report = {}
    for category in SILVER_CATEGORIES:
        chosen = [row for row in rows if row["top_cause"] == category]
        confirmed = sum(category in row["silver"]["labels"] for row in chosen)
        report[category] = {"predicted": len(chosen), "confirmed": confirmed, "precision": confirmed / len(chosen) if chosen else None}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-results", type=Path, required=True)
    parser.add_argument("--test-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    train_rows, settings = build_rows(args.train_results, args.output.with_name(args.output.stem + "_train.checkpoint.jsonl"))
    test_rows, _ = build_rows(args.test_results, args.output.with_name(args.output.stem + "_test.checkpoint.jsonl"))
    models = fit(train_rows)
    payload = {
        "train": {"regions": len(train_rows), "silver_counts": dict(Counter(label for row in train_rows for label in row["silver"]["labels"]))},
        "test": {
            "regions": len(test_rows),
            "silver_counts": dict(Counter(label for row in test_rows for label in row["silver"]["labels"])),
            "mean_annotated_share": float(np.mean([row["silver"]["annotated_share"] for row in test_rows])) if test_rows else None,
            "top_label_precision": top_label_precision(test_rows),
            "evaluation": evaluate(models, test_rows, settings, label_model=load_label_model(settings.get("label_model"))),
        },
        "models": public_models(models),
    }
    write_result(args.output, payload, inputs={"train_results": args.train_results, "test_results": args.test_results})
    for category, entry in payload["test"]["evaluation"]["categories"].items():
        parts = [f"{name} AUROC {entry[name]['auroc']:.3f}" for name in ("hand_set", "label_model", "calibrated", "learned") if name in entry and entry[name]["auroc"] is not None]
        print(f"{category:<16} positives {entry['positives']:>5}  " + "  ".join(parts))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
