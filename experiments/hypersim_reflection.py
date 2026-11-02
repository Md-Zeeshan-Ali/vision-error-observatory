"""Phase 5 reflection test on Hypersim: remove the non-diffuse light and re-run the depth model.

For each frame the model sees the original render and the same render without its residual
(non-diffuse) layer, tone-mapped at the same exposure. VEO analyses the original. Reported:

  paired      regions VEO labels reflection vs matched controls (reflection score 0): change in error
  dose        Spearman correlation between a region's specular light and its change in error
  detection   AUROC of VEO's reflection score for regions with strong specular light, and for regions
              on annotated reflective surfaces (window, mirror, television, whiteboard)
  global      change in whole-image AbsRel
"""

import argparse
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from veo.calibration import evaluate
from veo.checkpoint import JsonlLog, resumed
from veo.config import load_config
from veo.data import make_valid_mask
from veo.hypersim import (
    REFLECTIVE_NYU40,
    SILVER_GROUPS,
    ensure_metadata,
    fetch_frames,
    load_frame,
    select_frames,
)
from veo.interventions import match_controls, paired_statistics, region_error
from veo.label_model import load_label_model
from veo.metrics import align_scale_shift, compute_metrics
from veo.pipeline import analyze_sample, compute_regions
from veo.provenance import config_hash
from veo.reports import write_result
from veo.silver import region_silver, silver_maps

SPECULAR_STRONG = 10.0  # mean grey levels of specular light in a region


def _scene_bootstrap(values: np.ndarray, scenes: list[str], statistic, replicates: int = 2000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    scene_array = np.asarray(scenes)
    unique = np.unique(scene_array)
    groups = [np.flatnonzero(scene_array == scene) for scene in unique]
    draws = []
    for _ in range(replicates):
        picked = np.concatenate([groups[i] for i in rng.integers(0, len(unique), len(unique))])
        try:
            draws.append(statistic(values[picked]))
        except ValueError:
            continue
    return {"estimate": float(statistic(values)), "lower": float(np.nanpercentile(draws, 2.5)), "upper": float(np.nanpercentile(draws, 97.5))}


def _auroc(rows: list[dict], positive) -> dict | None:
    labels = np.asarray([positive(row) for row in rows])
    if not 0 < labels.sum() < len(labels):
        return None
    scores = np.asarray([row["reflection_score"] for row in rows])
    return {"positives": int(labels.sum()), **_scene_bootstrap(
        np.column_stack([labels, scores]), [row["scene"] for row in rows],
        lambda pairs: roc_auc_score(pairs[:, 0], pairs[:, 1]) if 0 < pairs[:, 0].sum() < len(pairs) else float("nan"),
    )}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-id", default="depth-anything/Depth-Anything-V2-Small-hf")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--config", type=Path, default=Path("configs/nyu.yaml"), help="VEO rules (experiment section)")
    parser.add_argument("--cache", type=Path, default=Path("data/hypersim"))
    parser.add_argument("--per-scene", type=int, default=3)
    parser.add_argument("--max-depth", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from veo.inference import TransformersDepthPredictor

    split_csv, camera_csv = ensure_metadata(args.cache)
    frames = select_frames(split_csv, camera_csv, "test", args.per_scene, args.seed)
    print(f"{len(frames)} frames from {len({f[0] for f in frames})} standard-camera test scenes", flush=True)
    fetch_frames(frames, args.cache)
    experiment = replace(load_config(args.config).experiment, max_prediction_m=args.max_depth)
    predictor = TransformersDepthPredictor(args.model_id, args.device)
    space = experiment.prediction_space if experiment.align_scale_shift else "depth"
    reflective_ids = list(REFLECTIVE_NYU40)

    rows, images = [], []
    log = JsonlLog(args.output.with_suffix(".checkpoint.jsonl"), config_hash({
        "model": args.model_id, "experiment": asdict(experiment), "max_depth": args.max_depth, "frames": frames,
    }))
    finished = log.load()
    for entry in finished.values():
        images.extend(entry["images"])
        rows.extend(entry["rows"])
    names = {f"{scene}/{camera}/{frame}": (scene, camera, frame) for scene, camera, frame in frames}
    for name in resumed(list(names), set(finished)):
        scene, camera, frame = names[name]
        sample = load_frame(args.cache, scene, camera, frame)
        depth = sample["depth"]
        valid = make_valid_mask(depth, args.max_depth)
        if valid.mean() < 0.5:
            log.append(name, {"images": [], "rows": []})
            continue
        image_start, row_start = len(images), len(rows)
        original, diffuse = predictor.predict(sample["rgb"]), predictor.predict(sample["rgb_diffuse"])
        before_metrics = compute_metrics(align_scale_shift(original, depth, valid, space, args.max_depth)[0], depth, valid)
        after_metrics = compute_metrics(align_scale_shift(diffuse, depth, valid, space, args.max_depth)[0], depth, valid)
        images.append({"scene": scene, "frame": f"{camera}/{frame}", "abs_rel_before": before_metrics["abs_rel"],
                       "abs_rel_after": after_metrics["abs_rel"], "specular_mean": float(sample["specular"][valid].mean())})
        analysis = analyze_sample(sample["rgb"], depth, original, valid, experiment)
        masks = {region["region_id"]: region["mask"] for region in compute_regions(depth, original, valid, experiment)["regions"]}
        silver = silver_maps(np.maximum(sample["semantic"], 0), np.maximum(sample["instance"], 0), depth, SILVER_GROUPS)
        for region in analysis["regions"]:
            mask = masks[region["region_id"]]
            before = region_error(original, depth, valid, mask, space, args.max_depth)
            after = region_error(diffuse, depth, valid, mask, space, args.max_depth)
            rows.append({
                "sample_id": f"{scene}/{camera}/{frame}", "region_id": region["region_id"], "scene": scene,
                "top_cause": region["top_cause"], "labels": region["labels"], "reflection_score": region["scores"].get("reflection", 0.0),
                "specular": float(sample["specular"][mask].mean()),
                "reflective_share": float(np.isin(sample["semantic"][mask], reflective_ids).mean()),
                "log_area": float(np.log(region["area"])), "mean_error": region["mean_error"],
                "log_depth": float(np.log(max(region["features"]["mean_depth_m"], 0.1))),
                "before": before, "after": after, "change": after - before,
                "features": region["features"], "silver": region_silver(silver, mask),
            })
        log.append(name, {"images": images[image_start:], "rows": rows[row_start:]})
        print(f"  {scene} {camera} {frame}: {analysis['num_regions']} regions, AbsRel {before_metrics['abs_rel']:.4f} -> {after_metrics['abs_rel']:.4f}", flush=True)

    targets = [row for row in rows if row["top_cause"] == "reflection"]
    controls = [row for row in rows if "reflection" not in row["labels"]]
    pairs = match_controls(targets, controls, seed=args.seed)
    differences = np.asarray([t["change"] - c["change"] for t, c in pairs])
    paired = paired_statistics(differences, [t["scene"] for t, _ in pairs], seed=args.seed) if pairs else {"n": 0}
    changes = np.asarray([[row["specular"], row["change"]] for row in rows])
    dose = _scene_bootstrap(changes, [row["scene"] for row in rows], lambda v: spearmanr(v[:, 0], v[:, 1]).statistic) if len(rows) > 2 else None
    image_changes = np.asarray([image["abs_rel_after"] - image["abs_rel_before"] for image in images])
    payload = {
        "model_id": args.model_id,
        "frames": len(images),
        "scenes": len({image["scene"] for image in images}),
        "regions": len(rows),
        "regions_labelled_reflection": len(targets),
        "paired_reflection_vs_control": {"pairs": len(pairs), "difference": paired,
                                         "mean_target_change": float(np.mean([t["change"] for t, _ in pairs])) if pairs else None,
                                         "mean_control_change": float(np.mean([c["change"] for _, c in pairs])) if pairs else None},
        "dose_response_spearman": dose,
        "detection_auroc": {
            "strong_specular": _auroc(rows, lambda row: row["specular"] >= SPECULAR_STRONG),
            "annotated_reflective_surface": _auroc(rows, lambda row: row["reflective_share"] >= 0.3),
        },
        "global_abs_rel_change": paired_statistics(image_changes, [image["scene"] for image in images], seed=args.seed) if len(images) > 1 else None,
        # Clean check: the model was fitted on NYU and is not refitted here
        "external_label_check": evaluate({}, rows, asdict(experiment), label_model=load_label_model(experiment.label_model)),
        "images": images,
        "rows": [{key: value for key, value in row.items() if key not in ("features", "silver")} for row in rows],
    }
    write_result(args.output, payload, experiment)
    print(f"Paired (reflection - control): {paired}")
    print(f"Dose-response Spearman (specular vs change): {dose}")
    print(f"Detection: {payload['detection_auroc']}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
