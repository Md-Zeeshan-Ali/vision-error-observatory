"""Phase 5: model-in-the-loop tests on the NYU test split.

  texture     add fine texture inside textureless regions vs matched controls
  exposure    gamma-lift poor_lighting regions vs matched controls
  resolution  run the model at 1.5x input size; boundary_mixing and thin_object regions vs matched controls

The depth model is re-run on every edited image and the whole prediction is re-aligned, as the
pipeline does. A negative paired difference (labelled minus control change in region error) means
the edit helped the labelled regions more, which supports VEO's hypothesis.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from veo.checkpoint import JsonlLog, resumed
from veo.config import config_from_dict
from veo.data import RGBDepthPair, load_rgb
from veo.interventions import (
    add_texture,
    brighten,
    holm,
    match_controls,
    paired_statistics,
    predict_at_scale,
    region_error,
)
from veo.pipeline import compute_regions
from veo.provenance import config_hash
from veo.reports import write_result
from veo.runner import load_sample

FIXTURE_MODEL = "synthetic-disparity"
TESTS = {
    # name: (selects labelled regions, selects control candidates, edit)
    "texture": (lambda r: r["top_cause"] == "textureless", lambda r: "textureless" not in r["labels"], "texture"),
    "exposure": (lambda r: r["top_cause"] == "poor_lighting", lambda r: "poor_lighting" not in r["labels"], "brighten"),
    "resolution_boundary": (
        lambda r: r["top_cause"] == "boundary_mixing",
        lambda r: not {"boundary_mixing", "thin_object"} & set(r["labels"]),
        "resolution",
    ),
    "resolution_thin": (
        lambda r: "thin_object" in r["labels"],
        lambda r: not {"boundary_mixing", "thin_object"} & set(r["labels"]),
        "resolution",
    ),
}


def region_table(payload: dict, scenes: dict[str, str]) -> list[dict]:
    rows = []
    for record in payload["records"]:
        for region in record["regions"]:
            rows.append({
                "sample_id": record["sample_id"], "region_id": region["region_id"], "scene": scenes.get(record["sample_id"], record["sample_id"]),
                "top_cause": region["top_cause"], "labels": region["labels"], "scores": region["scores"],
                "log_area": float(np.log(region["area"])), "mean_error": region["mean_error"],
                "log_depth": float(np.log(max(region["features"]["mean_depth_m"], 0.1))),
            })
    return rows


def make_predictor(model_id: str, device: str):
    if model_id == FIXTURE_MODEL:
        from veo.fixtures import SyntheticDisparityPredictor

        synthetic = SyntheticDisparityPredictor(0)
        # The synthetic model reads the true depth, resized to whatever input size it is given
        return lambda rgb, dense: synthetic.predict(rgb, cv2.resize(dense, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST))
    from veo.inference import TransformersDepthPredictor

    model = TransformersDepthPredictor(model_id, device)
    return lambda rgb, dense: model.predict(rgb)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-path", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--per-test", type=int, default=120, help="labelled regions per test (randomly sampled when more exist)")
    parser.add_argument("--scale", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.results_path.read_text(encoding="utf-8"))
    config = config_from_dict(payload["config"])
    split_dir = config.dataset.depth_dir.parent
    metadata = split_dir / "metadata.json"
    scenes = {row["sample_id"]: row["scene"] for row in json.loads(metadata.read_text(encoding="utf-8"))} if metadata.exists() else {}
    table = region_table(payload, scenes)
    rng = np.random.default_rng(args.seed)

    pairs_by_test = {}
    for name, (is_target, is_control, _) in TESTS.items():
        targets = [row for row in table if is_target(row)]
        if len(targets) > args.per_test:
            targets = [targets[i] for i in rng.choice(len(targets), args.per_test, replace=False)]
        candidates = [row for row in table if is_control(row) and not is_target(row)]
        pairs_by_test[name] = match_controls(targets, candidates, seed=args.seed)

    # Work out every (image -> regions to edit) once, then visit each image a single time
    jobs: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for name, pairs in pairs_by_test.items():
        for target, control in pairs:
            for row in (target, control):
                jobs[row["sample_id"]].add((TESTS[name][2], row["region_id"]))
    predict = make_predictor(args.model_id, args.device)
    space = config.experiment.prediction_space if config.experiment.align_scale_shift else "depth"
    changes: dict[tuple[str, str, int], dict] = {}
    log = JsonlLog(args.output.with_suffix(".checkpoint.jsonl"), config_hash({
        "results": str(args.results_path), "model": args.model_id, "scale": args.scale, "seed": args.seed,
        "per_test": args.per_test, "config": payload["config"], "jobs": {k: sorted(v) for k, v in jobs.items()},
    }))
    finished = log.load()
    for sample_id, entries in finished.items():
        for entry in entries:
            changes[(entry["edit"], sample_id, entry["region_id"])] = entry["result"]
    for sample_id in resumed(sorted(jobs), set(finished)):
        wanted = jobs[sample_id]
        pair = RGBDepthPair(sample_id, split_dir / "rgb" / f"{sample_id}.png", split_dir / "depth" / f"{sample_id}.npy")
        rgb = load_rgb(pair.rgb_path)
        _, target, dense, valid = load_sample(pair, config)
        original = np.load(args.results_path.parent / "predictions" / f"{sample_id}.npy")
        masks = {region["region_id"]: region["mask"] for region in compute_regions(target, original, valid, config.experiment)["regions"]}
        upscaled = None
        if any(edit == "resolution" for edit, _ in wanted):
            upscaled = predict_at_scale(lambda image, dense=dense: predict(image, dense), rgb, args.scale)
        for edit, region_id in sorted(wanted):
            mask = masks[region_id]
            if edit == "texture":
                edited = predict(add_texture(rgb, mask, seed=region_id), dense)
            elif edit == "brighten":
                edited = predict(brighten(rgb, mask), dense)
            else:
                edited = upscaled
            before = region_error(original, target, valid, mask, space, config.experiment.max_prediction_m)
            after = region_error(edited, target, valid, mask, space, config.experiment.max_prediction_m)
            changes[(edit, sample_id, region_id)] = {"before": before, "after": after, "change": after - before}
        log.append(sample_id, [{"edit": edit, "region_id": region_id, "result": changes[(edit, sample_id, region_id)]} for edit, region_id in sorted(wanted)])

    tests = {}
    for name, pairs in pairs_by_test.items():
        edit = TESTS[name][2]
        rows = []
        for target, control in pairs:
            t, c = changes[(edit, target["sample_id"], target["region_id"])], changes[(edit, control["sample_id"], control["region_id"])]
            rows.append({
                "target": [target["sample_id"], target["region_id"]], "control": [control["sample_id"], control["region_id"]],
                "scene": target["scene"], "target_change": t["change"], "control_change": c["change"],
                "target_before": t["before"], "control_before": c["before"], "difference": t["change"] - c["change"],
            })
        statistics = paired_statistics(np.asarray([row["difference"] for row in rows]), [row["scene"] for row in rows], seed=args.seed) if rows else {"n": 0}
        tests[name] = {
            "edit": edit,
            "pairs": len(rows),
            "mean_target_change": float(np.mean([row["target_change"] for row in rows])) if rows else None,
            "mean_control_change": float(np.mean([row["control_change"] for row in rows])) if rows else None,
            "mean_target_before": float(np.mean([row["target_before"] for row in rows])) if rows else None,
            "mean_control_before": float(np.mean([row["control_before"] for row in rows])) if rows else None,
            "difference": statistics,
            "rows": rows,
        }
    adjusted = holm({name: test["difference"]["p_value"] for name, test in tests.items() if "p_value" in test["difference"]})
    for name, value in adjusted.items():
        tests[name]["difference"]["p_holm"] = value
        stats = tests[name]["difference"]
        tests[name]["supports_hypothesis"] = bool(stats["upper"] < 0 and value < 0.05)

    write_result(args.output, {"model_id": args.model_id, "scale": args.scale, "tests": tests}, config, inputs={"veo_results": args.results_path})
    for name, test in tests.items():
        stats = test["difference"]
        if stats.get("n", 0) >= 2:
            print(f"{name:<20} pairs {test['pairs']:>4}  diff {stats['mean']:+.4f} [{stats['lower']:+.4f}, {stats['upper']:+.4f}]  p_holm {stats.get('p_holm', float('nan')):.4f}")
        else:
            print(f"{name:<20} pairs {test['pairs']:>4}  (too few)")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
