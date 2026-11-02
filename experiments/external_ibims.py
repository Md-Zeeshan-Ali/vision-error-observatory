"""Phase 6 external check on iBims-1: laser-scanned rooms with annotated depth edges and planes.

Runs every model, then scores frozen label model v3 on Depth Anything V2 Small's regions against the
silver labels in docs/MODEL_COMPARISON_PROTOCOL.md. Nothing is refitted.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from veo.calibration import bootstrap_auroc, evaluate
from veo.config import DatasetConfig, OutputConfig, config_from_dict, load_config
from veo.data import RGBDepthPair
from veo.ibims import ensure_core, silver_maps
from veo.label_model import load_label_model
from veo.pipeline import compute_regions
from veo.reports import write_result
from veo.runner import load_sample, run_model
from veo.silver import region_silver
from veo.summary import write_summary_markdown

PRIMARY = "depth-anything/Depth-Anything-V2-Small-hf"
MODELS = (
    PRIMARY,
    "depth-anything/Depth-Anything-V2-Base-hf",
    "depth-anything/Depth-Anything-V2-Large-hf",
    "Intel/dpt-hybrid-midas",
    "Intel/dpt-large",
    "Intel/zoedepth-nyu-kitti",
    "apple/DepthPro-hf",
    "prs-eth/marigold-depth-v1-1",
)


def slug(model_id: str) -> str:
    return model_id.split("/")[-1].lower()


def label_rows(results_path: Path, split: Path) -> tuple[list[dict], dict]:
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    config = config_from_dict(payload["config"])
    metadata = {row["sample_id"]: row for row in json.loads((split / "metadata.json").read_text(encoding="utf-8"))}
    rows = []
    for record in payload["records"]:
        sample_id = record["sample_id"]
        pair = RGBDepthPair(sample_id, split / "rgb" / f"{sample_id}.png", split / "depth" / f"{sample_id}.npy")
        _, target, _, valid = load_sample(pair, config)
        computed = compute_regions(target, np.load(results_path.parent / "predictions" / f"{sample_id}.npy"), valid, config.experiment)
        masks = {region["region_id"]: region["mask"] for region in computed["regions"]}
        if masks.keys() != {region["region_id"] for region in record["regions"]}:
            raise RuntimeError(f"{sample_id}: recomputed regions differ from the stored results")
        with np.load(split / "masks" / f"{sample_id}.npz") as stored:
            maps = silver_maps(dict(stored))
        for region in record["regions"]:
            mask = masks[region["region_id"]]
            rows.append({
                "sample_id": sample_id, "scene": metadata[sample_id]["scene"], "has_edges": metadata[sample_id]["has_edges"],
                "region_id": region["region_id"], "features": region["features"], "top_cause": region["top_cause"],
                "silver": region_silver(maps, mask), "planar_share": float(maps["planar"][mask].mean()),
            })
    return rows, payload["config"]["experiment"]


def label_check(rows: list[dict], experiment: dict) -> dict:
    label_model = load_label_model(experiment["label_model"])
    with_edges = [row for row in rows if row["has_edges"]]
    everything = evaluate({}, rows, experiment, label_model=label_model)["categories"]
    groups = np.asarray([row["scene"] for row in rows])
    textureless = np.asarray([label_model.probabilities(row["features"])["textureless"] for row in rows])
    reflective = np.asarray(["reflection" in row["silver"]["labels"] for row in rows])
    holes = np.asarray([row["features"]["sensor_hole_fraction"] for row in rows])
    return {
        "regions": len(rows),
        "images": len({row["sample_id"] for row in rows}),
        "categories": {
            "boundary_mixing": {**evaluate({}, with_edges, experiment, label_model=label_model)["categories"]["boundary_mixing"], "regions": len(with_edges)},
            "textureless": everything["textureless"],
            "reflection": everything["reflection"],
        },
        "textureless_any_plane": bootstrap_auroc(np.asarray([row["planar_share"] >= 0.3 for row in rows]), textureless, groups),
        "reflection_sensor_holes_only": bootstrap_auroc(reflective, holes, groups),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/nyu.yaml"), help="VEO rules (experiment section)")
    parser.add_argument("--data-dir", type=Path, default=Path("data/ibims"))
    parser.add_argument("--outputs", type=Path, default=Path("outputs/ibims"))
    parser.add_argument("--results", type=Path, default=Path("results/ibims"))
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    split = ensure_core(args.data_dir)
    base = load_config(args.config)
    dataset = DatasetConfig(
        name="ibims1", rgb_dir=split / "rgb", depth_dir=split / "depth", manifest=split / "manifest.txt",
        depth_scale=1.0, max_depth_m=10.0, split="core",
    )
    for model_id in args.models:
        config = replace(base, dataset=dataset, outputs=OutputConfig(directory=args.outputs / slug(model_id)))
        print(model_id, flush=True)
        run_model(config, model_id, args.device, save_panels=False, save_predictions=model_id == PRIMARY)
        target = args.results / slug(model_id)
        target.mkdir(parents=True, exist_ok=True)
        (target / "summary.json").write_bytes((config.outputs.directory / "summary.json").read_bytes())
    if PRIMARY in args.models:
        results_path = args.outputs / slug(PRIMARY) / "model_results.json"
        rows, experiment = label_rows(results_path, split)
        report = label_check(rows, experiment)
        write_result(args.results / "label_check.json", report, inputs={"results": results_path})
        for category, entry in report["categories"].items():
            print(f"{category}: positives {entry['positives']}, label model AUROC {entry['label_model']['auroc']}")
    print(f"Done: {write_summary_markdown(args.results)}")


if __name__ == "__main__":
    main()
