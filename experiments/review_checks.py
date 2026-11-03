"""Checks asked for in review of the paper draft (exploratory: decided after every result in the paper).

1. Single-cue baselines: AUROC of each selected cue alone, and of region area, beside the label model.
2. Top-label precision and recall against silver labels.
3. Base rates: how often silver-positive pixels fail compared with other annotated pixels.
4. Edge tolerance: which failing pixels survive when the prediction may be off by 1-3 px.
5. Boundary failure as an absolute share of valid pixels per model, not only as a share of region pixels.
6. Hypersim specular dose-response with the starting error held fixed.

    python experiments/review_checks.py
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from external_ibims import label_rows
from scipy.stats import rankdata, spearmanr

from veo.calibration import bootstrap_auroc
from veo.config import config_from_dict
from veo.data import RGBDepthPair
from veo.interventions import paired_statistics
from veo.label_model import load_label_model
from veo.metrics import tolerant_failure
from veo.pipeline import compute_regions
from veo.reports import write_result
from veo.runner import load_sample
from veo.silver import SILVER_CATEGORIES, class_ids, silver_maps

NYU = Path("outputs/nyu_test")
NYU_ROWS = Path("results/nyu_test/depth-anything-v2-small-hf/silver_calibration_test.checkpoint.jsonl")
LABEL_MODEL = Path("configs/label_model_v3.json")
MODEL_SLUGS = ("depth-anything-v2-large-hf", "depthpro-hf", "depth-anything-v2-base-hf", "zoedepth-nyu-kitti",
               "depth-anything-v2-small-hf", "marigold-depth-v1-1", "dpt-hybrid-midas", "dpt-large")
SAVED_PREDICTIONS = ("depth-anything-v2-small-hf", "dpt-hybrid-midas")  # the NYU models whose predictions are on disk
RADII = (1, 2, 3)


def nyu_rows() -> list[dict]:
    rows = []
    for line in NYU_ROWS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.extend(json.loads(line)["value"])
    return rows


def cue_baselines(rows: list[dict], categories: tuple[str, ...]) -> dict:
    """AUROC of the label model and of each of its cues alone, oriented by the fitted sign, and of log area."""
    model = load_label_model(LABEL_MODEL)
    groups = np.asarray([row["scene"] for row in rows])
    report = {}
    for category in categories:
        labels = np.asarray([category in row["silver"]["labels"] for row in rows])
        spec = model.categories[category]
        entry = {"positives": int(labels.sum()), "regions": len(rows),
                 "label_model": bootstrap_auroc(labels, np.asarray([model.probabilities(row["features"])[category] for row in rows]), groups)}
        for name, coefficient in zip(spec["features"], spec["coef"], strict=True):
            entry[name] = bootstrap_auroc(labels, np.sign(coefficient) * np.asarray([row["features"][name] for row in rows]), groups)
        entry["log_area_alone"] = bootstrap_auroc(labels, np.asarray([row["features"]["log_area"] for row in rows]), groups)
        report[category] = entry
    return report


def top_label_scores(rows: list[dict], categories: tuple[str, ...]) -> dict:
    report = {}
    for category in categories:
        silver = np.asarray([category in row["silver"]["labels"] for row in rows])
        top = np.asarray([row["top_cause"] == category for row in rows])
        hits = int((silver & top).sum())
        report[category] = {"top_label": int(top.sum()), "silver": int(silver.sum()), "both": hits,
                            "precision": hits / top.sum() if top.sum() else None, "recall": hits / silver.sum() if silver.sum() else None}
    return report


def pixel_checks(slug: str) -> dict:
    """Base rates by silver category, and how much failure survives a 1-3 px tolerance, on NYU test."""
    results_path = NYU / slug / "model_results.json"
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    config = config_from_dict(payload["config"])
    split = config.dataset.depth_dir.parent
    groups = class_ids(json.loads((split / "label_names.json").read_text(encoding="utf-8")))
    scene_of = {row["sample_id"]: row["scene"] for row in json.loads((split / "metadata.json").read_text(encoding="utf-8"))}
    # Per scene: [failing, total] for positive and for other annotated pixels of each category
    counts = {category: defaultdict(lambda: np.zeros(4)) for category in SILVER_CATEGORIES}
    kept = defaultdict(lambda: np.zeros(1 + len(RADII)))  # failing pixels, then those still failing at each radius
    for index, record in enumerate(payload["records"], start=1):
        sample_id = record["sample_id"]
        pair = RGBDepthPair(sample_id, split / "rgb" / f"{sample_id}.png", split / "depth" / f"{sample_id}.npy")
        _, target, dense, valid = load_sample(pair, config)
        computed = compute_regions(target, np.load(NYU / slug / "predictions" / f"{sample_id}.npy"), valid, config.experiment)
        valid, failing = computed["valid_mask"], computed["failing"]
        instances = split / "instances" / f"{sample_id}.npy"
        maps = silver_maps(np.load(split / "labels" / f"{sample_id}.npy"), np.load(instances) if instances.exists() else None, dense, groups)
        annotated = maps["annotated"] & valid
        scene = scene_of[sample_id]
        for category in SILVER_CATEGORIES:
            positive = maps[category] & valid
            negative = annotated & ~maps[category]
            counts[category][scene] += [(failing & positive).sum(), positive.sum(), (failing & negative).sum(), negative.sum()]
        tolerant = [tolerant_failure(computed["aligned"], target, failing, radius) for radius in RADII]
        masks = {region["region_id"]: region["mask"] for region in computed["regions"]}
        for region in record["regions"]:
            mask = masks[region["region_id"]] & failing
            kept[region["top_cause"]] += [mask.sum(), *[(survivor & mask).sum() for survivor in tolerant]]
        band = maps["boundary_mixing"] & failing
        kept["silver_boundary_band"] += [band.sum(), *[(survivor & band).sum() for survivor in tolerant]]
        kept["outside_boundary_band"] += [(failing & ~maps["boundary_mixing"]).sum(), *[(survivor & ~maps["boundary_mixing"]).sum() for survivor in tolerant]]
        if index % 100 == 0:
            print(f"  {slug}: {index}/{len(payload['records'])} images", flush=True)
    base_rates = {}
    for category, by_scene in counts.items():
        scenes = sorted(scene for scene, row in by_scene.items() if row[1] > 0 and row[3] > 0)
        totals = np.sum([by_scene[scene] for scene in scenes], axis=0)
        differences = np.asarray([by_scene[scene][0] / by_scene[scene][1] - by_scene[scene][2] / by_scene[scene][3] for scene in scenes])
        base_rates[category] = {"fail_rate_positive": totals[0] / totals[1], "fail_rate_other_annotated": totals[2] / totals[3],
                                "rate_ratio": (totals[0] / totals[1]) / (totals[2] / totals[3]),
                                "scene_difference": paired_statistics(differences, scenes)}
    tolerance = {name: {"failing_pixels": int(row[0]), **{f"kept_at_{radius}px": float(row[1 + i] / row[0]) if row[0] else None for i, radius in enumerate(RADII)}}
                 for name, row in kept.items()}
    return {"base_rates": base_rates, "edge_tolerance": tolerance}


def boundary_area() -> dict:
    """Boundary-labelled failure as a share of all valid pixels, beside each model's failing share."""
    report = {}
    for dataset in ("nyu_test", "diode", "ibims"):
        rows = {}
        for slug in MODEL_SLUGS:
            summary = json.loads(Path(f"results/{dataset}/{slug}/summary.json").read_text(encoding="utf-8"))
            rows[slug] = {"failing_share": summary["mean_failure_pixel_fraction"], "boundary_of_valid": summary["top_cause_pixel_fraction"]["boundary_mixing"],
                          "boundary_of_regions": summary["top_cause_pixel_fraction"]["boundary_mixing"] / summary["region_pixel_fraction"]}
        failing = [row["failing_share"] for row in rows.values()]
        report[dataset] = {"models": rows,
                           "spearman_failing_vs_boundary_of_valid": float(spearmanr(failing, [row["boundary_of_valid"] for row in rows.values()]).statistic),
                           "spearman_failing_vs_boundary_of_regions": float(spearmanr(failing, [row["boundary_of_regions"] for row in rows.values()]).statistic)}
    return report


def specular_partial(replicates: int = 2000, seed: int = 0) -> dict:
    """Spearman of specular light with the change in error, with starting error partialled out, scenes resampled."""
    rows = json.loads(Path("results/hypersim/hypersim_reflection.json").read_text(encoding="utf-8"))["rows"]

    def partial(index: np.ndarray) -> float:
        ranks = [rankdata(np.asarray([rows[i][key] for i in index])) for key in ("specular", "change", "before")]
        design = np.column_stack([np.ones(len(index)), ranks[2]])
        residuals = [values - design @ np.linalg.lstsq(design, values, rcond=None)[0] for values in ranks[:2]]
        return float(np.corrcoef(*residuals)[0, 1])

    scenes = np.asarray([row["scene"] for row in rows])
    by_scene = {scene: np.flatnonzero(scenes == scene) for scene in np.unique(scenes)}
    rng = np.random.default_rng(seed)
    values = [partial(np.concatenate([by_scene[scene] for scene in rng.choice(list(by_scene), size=len(by_scene))])) for _ in range(replicates)]
    everything = np.arange(len(rows))
    return {"regions": len(rows), "scenes": len(by_scene), "plain_spearman": float(spearmanr([row["specular"] for row in rows], [row["change"] for row in rows]).statistic),
            "partial_spearman": partial(everything), "lower": float(np.percentile(values, 2.5)), "upper": float(np.percentile(values, 97.5))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ibims-split", type=Path, default=Path("data/ibims/core"))
    parser.add_argument("--output", type=Path, default=Path("results/review_checks.json"))
    args = parser.parse_args()

    rows = nyu_rows()
    ibims, _ = label_rows(Path("outputs/ibims/depth-anything-v2-small-hf/model_results.json"), args.ibims_split)
    ibims_edges = [row for row in ibims if row["has_edges"]]
    report = {
        "note": "Exploratory: chosen after every result in the paper, in response to review.",
        "cue_baselines": {"nyu_test": cue_baselines(rows, SILVER_CATEGORIES),
                          "ibims": {**cue_baselines(ibims_edges, ("boundary_mixing",)), **cue_baselines(ibims, ("reflection",))}},
        "top_label": {"nyu_test": top_label_scores(rows, SILVER_CATEGORIES),
                      "ibims": {**top_label_scores(ibims_edges, ("boundary_mixing",)), **top_label_scores(ibims, ("reflection",))}},
        "boundary_area": boundary_area(),
        "specular_partial": specular_partial(),
        "pixels": {slug: pixel_checks(slug) for slug in SAVED_PREDICTIONS},
    }
    write_result(args.output, report)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
