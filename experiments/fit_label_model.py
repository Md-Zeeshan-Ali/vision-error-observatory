"""Fit the v3 label model on the train split only (docs/LABEL_MODEL_V3.md).

Forward selection of at most 3 physically motivated features per category, using scene-grouped
5-fold cross-validated AUROC. Writes the model file and a report of out-of-fold performance.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from veo.calibration import (
    FEATURES,
    _best_f1_threshold,
    _label_metrics,
    build_rows,
    feature_matrix,
    silver_matrix,
)
from veo.reports import write_result
from veo.scoring import classify
from veo.silver import SILVER_CATEGORIES

CANDIDATES = {
    "textureless": ("grad_energy", "coarse_grad", "corner_strength", "contrast", "log_area"),
    "boundary_mixing": ("depth_edge_fraction", "depth_step", "log_area"),
    "thin_object": ("thin_fraction", "thin_image_fraction", "depth_edge_fraction", "aspect_ratio", "log_area"),
    "reflection": ("highlight_ratio", "sensor_hole_fraction", "signed_error", "value_p95", "saturation_mean", "clipped_fraction"),
}
MAX_FEATURES = 3
MIN_GAIN = 0.005


def out_of_fold(x: np.ndarray, labels: np.ndarray, groups: np.ndarray, make_model, folds: int = 5) -> np.ndarray:
    predictions = np.zeros(len(labels))
    for train, held_out in GroupKFold(n_splits=folds).split(x, labels, groups):
        model = make_model().fit(x[train], labels[train])
        predictions[held_out] = model.predict_proba(x[held_out])[:, 1]
    return predictions


def standardise(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-9)


def forward_select(rows: list[dict], labels: np.ndarray, groups: np.ndarray, candidates: tuple[str, ...]) -> tuple[list[str], list[dict]]:
    chosen, best, trace = [], 0.5, []
    while len(chosen) < MAX_FEATURES:
        trials = {}
        for name in candidates:
            if name not in chosen:
                x = standardise(feature_matrix(rows, chosen + [name]))
                trials[name] = roc_auc_score(labels, out_of_fold(x, labels, groups, lambda: LogisticRegression(max_iter=1000)))
        pick = max(trials, key=trials.get)
        if trials[pick] < best + MIN_GAIN:
            break
        chosen.append(pick)
        best = trials[pick]
        trace.append({"added": pick, "cv_auroc": best})
    return chosen, trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-results", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, default=Path("configs/label_model_v3.json"))
    parser.add_argument("--report", type=Path, default=Path("results/label_model_v3/fit_report.json"))
    args = parser.parse_args()

    rows, settings = build_rows(args.train_results, args.report.with_name("train_rows.checkpoint.jsonl"))
    silver = silver_matrix(rows)
    groups = np.asarray([row["scene"] for row in rows])
    all_features = feature_matrix(rows, FEATURES)
    rule_scores = [classify(row["features"], True, {**settings, "label_model": None})["scores"] for row in rows]
    categories, report = {}, {}
    out_of_fold_ratio = {}
    for column, category in enumerate(SILVER_CATEGORIES):
        labels = silver[:, column]
        chosen, trace = forward_select(rows, labels, groups, CANDIDATES[category])
        x = feature_matrix(rows, chosen)
        mean, std = x.mean(axis=0), x.std(axis=0) + 1e-9
        scaled = (x - mean) / std
        oof = out_of_fold(scaled, labels, groups, lambda: LogisticRegression(max_iter=1000))
        threshold = _best_f1_threshold(labels, oof)
        final = LogisticRegression(max_iter=1000).fit(scaled, labels)
        categories[category] = {
            "features": chosen, "mean": mean.tolist(), "std": std.tolist(), "coef": final.coef_[0].tolist(),
            "intercept": float(final.intercept_[0]), "threshold": threshold, "train_prevalence": float(labels.mean()),
        }
        boosted = out_of_fold(all_features, labels, groups, lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=42))
        rules = np.asarray([scores[category] for scores in rule_scores])
        report[category] = {
            "selection": trace,
            "cv_auroc_v2_rules": float(roc_auc_score(labels, rules)),
            "cv_auroc_v3": float(roc_auc_score(labels, oof)),
            "cv_auroc_learned": float(roc_auc_score(labels, boosted)),
            "cv_label_metrics_v2_rules": _label_metrics(labels, rules >= 0.5),
            "cv_label_metrics_v3": _label_metrics(labels, oof >= threshold),
        }
        out_of_fold_ratio[category] = oof / max(threshold, 1e-9)
        print(f"{category:<16} features {chosen}  CV AUROC rules {report[category]['cv_auroc_v2_rules']:.3f}  v3 {report[category]['cv_auroc_v3']:.3f}  learned {report[category]['cv_auroc_learned']:.3f}", flush=True)

    # Top-label behaviour with the v3 rule, out of fold (poor_lighting has no silver label and is left out here)
    ratio_matrix = np.column_stack([out_of_fold_ratio[category] for category in SILVER_CATEGORIES])
    top = ratio_matrix.argmax(axis=1)
    committed = ratio_matrix.max(axis=1) >= 1.0
    top_labels = {}
    for column, category in enumerate(SILVER_CATEGORIES):
        chosen_rows = committed & (top == column)
        top_labels[category] = {"predicted": int(chosen_rows.sum()), "confirmed": int(silver[chosen_rows, column].sum()),
                                "precision": float(silver[chosen_rows, column].mean()) if chosen_rows.any() else None,
                                "base_rate": float(silver[:, column].mean())}
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.model_output.write_text(json.dumps({"version": 3, "fitted_on": str(args.train_results), "categories": categories}, indent=2) + "\n", encoding="utf-8")
    write_result(args.report, {"train_regions": len(rows), "train_scenes": len(set(groups.tolist())), "categories": report,
                               "cv_top_label": top_labels, "cv_coverage": float(committed.mean()), "model": categories})
    print(json.dumps({"coverage": float(committed.mean()), **{k: (v["predicted"], v["confirmed"]) for k, v in top_labels.items()}}))
    print(f"Saved {args.model_output} and {args.report}")


if __name__ == "__main__":
    main()
