"""Analysis of a multi-rater human study, following docs/HUMAN_STUDY_PROTOCOL.md.

Inputs are the study manifest (written by ``veo.study.build_study``) and the rating files exported by
the labeling tool. Primary consensus is Dawid-Skene; majority vote is reported as a check.
"""

import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

from .agreement import (
    dawid_skene,
    fleiss_kappa,
    krippendorff_alpha_nominal,
    majority_vote,
    multilabel_consensus,
)
from .scoring import classify
from .silver import SILVER_CATEGORIES
from .validation import ABSTAIN_LABEL, agreement_report, cohens_kappa

PRACTICE_PASS = 0.4  # raters below this share of practice agreement are excluded from the primary analysis


def load_ratings(paths: list[Path], study_id: str, rater_ids: list[str] | None = None) -> list[dict]:
    """Rating files as exported. ``rater_ids`` assigns the study's ID to each file, in order, when raters typed
    their ID differently; the typed ID is kept as ``typed_rater_id`` and the files themselves are never changed."""
    if rater_ids is not None and len(rater_ids) != len(paths):
        raise ValueError("Give one rater ID per ratings file")
    raters = []
    for index, path in enumerate(paths):
        rating = json.loads(path.read_text(encoding="utf-8"))
        if rating.get("study_id") != study_id:
            raise ValueError(f"{path} belongs to study {rating.get('study_id')}, not {study_id}")
        if rater_ids is not None:
            rating = {**rating, "typed_rater_id": rating.get("rater_id"), "rater_id": rater_ids[index]}
        raters.append(rating)
    if len({rater["rater_id"] for rater in raters}) != len(raters):
        raise ValueError("Duplicate rater ids")
    return raters


def practice_scores(raters: list[dict], practice: list[dict]) -> dict[str, float | None]:
    reference = {item["id"]: item["reference"] for item in practice}
    scores = {}
    for rater in raters:
        answered = [item_id for item_id in rater.get("practice", {}) if item_id in reference]
        scores[rater["rater_id"]] = (
            sum(rater["practice"][item_id]["primary"] == reference[item_id] for item_id in answered) / len(answered) if answered else None
        )
    return scores


def _cluster_bootstrap(values_by_item: dict[str, tuple], scene_of: dict[str, str], statistic, replicates: int, seed: int) -> dict:
    items = sorted(values_by_item)
    point = statistic([values_by_item[item] for item in items])
    scenes = sorted({scene_of[item] for item in items})
    by_scene = {scene: [item for item in items if scene_of[item] == scene] for scene in scenes}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(replicates):
        sample = [values_by_item[item] for scene in rng.choice(scenes, size=len(scenes)) for item in by_scene[scene]]
        try:
            draws.append(statistic(sample))
        except (ValueError, ZeroDivisionError):
            continue
    return {"estimate": point, "lower": float(np.percentile(draws, 2.5)), "upper": float(np.percentile(draws, 97.5))}


def _accuracy(pairs: list[tuple]) -> float:
    return sum(reference == predicted for reference, predicted, *_ in pairs) / len(pairs)


def _kappa(pairs: list[tuple]) -> float:
    return cohens_kappa([pair[0] for pair in pairs], [pair[1] for pair in pairs])


def _weighted_accuracy(pairs: list[tuple]) -> float:
    weights = np.asarray([pair[2] for pair in pairs])
    return float(np.sum(weights * [pair[0] == pair[1] for pair in pairs]) / weights.sum())


def calibrated_label(features: dict, models: dict, settings: dict | None = None, threshold: float = 0.5) -> str:
    """Top label from the Phase 3 logistic models (poor_lighting keeps its hand-set score)."""
    scores = {"poor_lighting": classify(features, True, settings)["scores"]["poor_lighting"]}
    for category, model in models.items():
        if model is None:
            continue
        x = (np.asarray([features.get(name, 0.0) for name in model["features"]]) - np.asarray(model["mean"])) / np.asarray(model["std"])
        scores[category] = float(1.0 / (1.0 + np.exp(-(x @ np.asarray(model["coef"]) + model["intercept"]))))
    top, value = max(scores.items(), key=lambda item: item[1])
    return top if value >= threshold else ABSTAIN_LABEL


def analyze(manifest: dict, raters: list[dict], calibration: dict | None = None, replicates: int = 1000, seed: int = 42, apply_exclusions: bool = True) -> dict:
    items = {item["id"]: item for item in manifest["items"]}
    practice = practice_scores(raters, manifest.get("practice", []))
    included = [r for r in raters if not apply_exclusions or practice[r["rater_id"]] is None or practice[r["rater_id"]] >= PRACTICE_PASS]
    if len(included) < 2:
        raise ValueError("At least two raters must pass the practice round")

    primary = {item_id: {r["rater_id"]: r["ratings"][item_id]["primary"] for r in included if item_id in r["ratings"]} for item_id in items}
    primary = {item_id: by_rater for item_id, by_rater in primary.items() if by_rater}
    label_sets = {item_id: {r["rater_id"]: r["ratings"][item_id]["labels"] for r in included if item_id in r["ratings"]} for item_id in primary}
    complete = {item_id: by_rater for item_id, by_rater in primary.items() if len(by_rater) == len(included)}

    pairwise = []
    for first, second in combinations([r["rater_id"] for r in included], 2):
        shared = [item_id for item_id, by_rater in primary.items() if first in by_rater and second in by_rater]
        if shared:
            pairwise.append(cohens_kappa([primary[i][first] for i in shared], [primary[i][second] for i in shared]))

    ds = dawid_skene(primary)
    majority = majority_vote(primary)
    consensus = ds["labels"]
    scene_of = {item_id: items[item_id]["scene"] for item_id in primary}

    pairs = {item_id: (consensus[item_id], items[item_id]["top_cause"], 1.0 / items[item_id]["inclusion_probability"]) for item_id in primary}
    reference = [pairs[i][0] for i in sorted(pairs)]
    predicted = [pairs[i][1] for i in sorted(pairs)]
    report = agreement_report(reference, predicted)
    committed = {i: pair for i, pair in pairs.items() if pair[1] != ABSTAIN_LABEL}
    bootstrap = {
        "accuracy": _cluster_bootstrap(pairs, scene_of, _accuracy, replicates, seed),
        "kappa": _cluster_bootstrap(pairs, scene_of, _kappa, replicates, seed),
        "weighted_accuracy": _cluster_bootstrap(pairs, scene_of, _weighted_accuracy, replicates, seed),
    }
    if committed:
        bootstrap["selective_accuracy"] = _cluster_bootstrap(committed, scene_of, _accuracy, replicates, seed)
        majority_label = Counter(pair[0] for pair in committed.values()).most_common(1)[0][0]
        bootstrap["selective_majority_baseline"] = _cluster_bootstrap(
            {i: (pair[0], majority_label) for i, pair in committed.items()}, scene_of, _accuracy, replicates, seed
        )

    human_sets = multilabel_consensus(label_sets)
    multilabel = {}
    for category in SILVER_CATEGORIES + ("poor_lighting",):
        human = {i for i, labels in human_sets.items() if category in labels}
        veo = {i for i in primary if category in items[i]["labels"]}
        silver = {i for i in primary if category in items[i].get("silver_labels", [])}
        entry = {"human": len(human), "veo": len(veo), "veo_and_human": len(veo & human)}
        entry["veo_precision"] = len(veo & human) / len(veo) if veo else None
        entry["veo_recall"] = len(veo & human) / len(human) if human else None
        if category in SILVER_CATEGORIES:
            entry.update({"silver": len(silver), "silver_and_human": len(silver & human),
                          "silver_precision": len(silver & human) / len(silver) if silver else None,
                          "silver_recall": len(silver & human) / len(human) if human else None})
        multilabel[category] = entry

    result = {
        "raters": {r["rater_id"]: {"practice_agreement": practice[r["rater_id"]], "included": r in included, "rated": len(r["ratings"]),
                                   **({"typed_rater_id": r["typed_rater_id"]} if "typed_rater_id" in r else {})} for r in raters},
        "items_rated": len(primary),
        "inter_rater": {
            "fleiss_kappa": fleiss_kappa(complete) if len(complete) >= 2 and len(included) >= 2 else None,
            "fleiss_items": len(complete),
            "krippendorff_alpha": krippendorff_alpha_nominal(primary),
            "mean_pairwise_cohen_kappa": float(np.mean(pairwise)) if pairwise else None,
            "dawid_skene_rater_accuracy": ds["rater_accuracy"],
        },
        "consensus": {
            "method": "dawid_skene",
            "label_counts": dict(Counter(consensus.values())),
            "majority_vote_agreement": sum(majority[i] == consensus[i] for i in primary) / len(primary),
            "majority_vote_unresolved": sum(majority[i] == "unresolved" for i in primary),
        },
        "veo_vs_consensus": {**report, "bootstrap": bootstrap},
        "multilabel": multilabel,
    }
    rule_labels = {i: classify(items[i]["features"], True)["top_cause"] for i in primary if "grad_energy" in items[i]["features"]}
    if rule_labels:
        rule_pairs = {i: (consensus[i], label, pairs[i][2]) for i, label in rule_labels.items()}
        result["v2_rules_vs_consensus"] = {
            **agreement_report([rule_pairs[i][0] for i in sorted(rule_pairs)], [rule_pairs[i][1] for i in sorted(rule_pairs)]),
            "bootstrap": {"accuracy": _cluster_bootstrap(rule_pairs, scene_of, _accuracy, replicates, seed)},
        }
    if calibration and calibration.get("models"):
        calibrated = {i: calibrated_label(items[i]["features"], calibration["models"]) for i in primary}
        calibrated_pairs = {i: (consensus[i], calibrated[i], pairs[i][2]) for i in primary}
        result["calibrated_vs_consensus"] = {
            **agreement_report([calibrated_pairs[i][0] for i in sorted(calibrated_pairs)], [calibrated_pairs[i][1] for i in sorted(calibrated_pairs)]),
            "bootstrap": {"accuracy": _cluster_bootstrap(calibrated_pairs, scene_of, _accuracy, replicates, seed)},
        }
    per_class = {}
    for category in sorted({pair[1] for pair in pairs.values()} - {ABSTAIN_LABEL}):
        chosen = {i: pair for i, pair in pairs.items() if pair[1] == category}
        if len(chosen) < 2:
            continue
        per_class[category] = {
            "precision": _cluster_bootstrap(chosen, scene_of, _accuracy, replicates, seed),
            # Chance level: how often the consensus picks this class at all
            "base_rate": _cluster_bootstrap({i: (pair[0], category) for i, pair in pairs.items()}, scene_of, _accuracy, replicates, seed),
        }
    result["veo_vs_consensus"]["per_class_bootstrap"] = per_class
    selective = bootstrap.get("selective_accuracy")
    baseline = bootstrap.get("selective_majority_baseline")
    above_chance = sorted(c for c, row in per_class.items() if row["precision"]["lower"] > row["base_rate"]["upper"])
    result["preregistered_checks"] = {
        "selective_accuracy_above_majority_baseline": bool(selective and baseline and selective["lower"] > baseline["upper"]),
        "classes_with_precision_above_base_rate": above_chance,
        "at_least_three_classes_above_base_rate": len(above_chance) >= 3,
        "inter_rater_alpha_at_least_0_4": result["inter_rater"]["krippendorff_alpha"] >= 0.4,
    }
    if apply_exclusions and len(included) < len(raters):
        # Pre-registered sensitivity analysis: the same analysis with the excluded raters kept in
        everyone = analyze(manifest, raters, calibration, replicates, seed, apply_exclusions=False)
        result["sensitivity_all_raters"] = {
            "inter_rater": everyone["inter_rater"],
            "veo_vs_consensus_accuracy": everyone["veo_vs_consensus"]["bootstrap"]["accuracy"],
            "veo_vs_consensus_kappa": everyone["veo_vs_consensus"]["bootstrap"]["kappa"],
            "preregistered_checks": everyone["preregistered_checks"],
        }
    return result
