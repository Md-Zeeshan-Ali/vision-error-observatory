"""Export the web app's scene data from the current results (official NYU test split, label model v3).

Demo scenes are chosen automatically from the test split, one per label, so the page never shows images
the label model was fitted on. Activation maps come from Depth Anything V2 Small, the model being evaluated.
Writes web_app/data/app_data_v2.js; publish it as a new release asset (see scripts/README.md).
"""

import argparse
import base64
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from veo.config import config_from_dict
from veo.data import RGBDepthPair, load_rgb
from veo.label_model import load_label_model
from veo.pipeline import compute_regions
from veo.runner import load_sample

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
FOCUS = {
    "textureless": "Textureless surface",
    "boundary_mixing": "Depth boundary",
    "reflection": "Reflective or transparent surface",
    "thin_object": "Thin object",
}
LAYERS = {
    3: "Layer 3 (early): low-level gradients, edge tokens and local contrast.",
    6: "Layer 6 (middle): texture grouping and local surface orientation.",
    9: "Layer 9 (late): object parts and spatial relations.",
    12: "Layer 12 (final): scene-level layout.",
}
TAXONOMY = [
    {"id": "textureless", "label": "Textureless Surface", "metric_name": "Gradient energy, corner strength, contrast",
     "formula": "Logistic model fitted on the NYU train split",
     "desc": "Plain surfaces with little visual detail. Weakly validated: close to chance against wall and ceiling annotations.",
     "remedy": "No remedy is established. Adding texture did not help labelled regions more than controls in VEO's tests."},
    {"id": "boundary_mixing", "label": "Depth Boundary", "metric_name": "Depth-edge share, depth step",
     "formula": "Logistic model fitted on the NYU train split",
     "desc": "Errors along an edge between nearer and farther surfaces, including occlusion edges. The best-validated label: it holds up on NYU test and on synthetic Hypersim scenes.",
     "remedy": "A larger input size reduced error in these regions slightly more than in matched controls."},
    {"id": "thin_object", "label": "Thin Object", "metric_name": "Thin depth structure, thin image lines, region area",
     "formula": "Logistic model fitted on the NYU train split",
     "desc": "Narrow objects such as legs, poles and cables that are missed or thickened.",
     "remedy": "No remedy is established."},
    {"id": "reflection", "label": "Reflective Surface", "metric_name": "Nearby sensor holes, signed error, saturation",
     "formula": "Logistic model fitted on the NYU train split",
     "desc": "Mirrors, windows and screens. The label relies on nearby depth-sensor failures, so it works for Kinect-style sensors and not in general. Some of these errors may be errors in the ground truth rather than the model.",
     "remedy": "No remedy is established. Removing specular light in synthetic scenes did reduce error in proportion to its strength."},
    {"id": "poor_lighting", "label": "Poor Lighting", "metric_name": "Brightness, clipped pixels",
     "formula": "Hand-set rule (no annotation exists to fit against)",
     "desc": "Regions too dark or overexposed to show detail. Not validated against any annotation.",
     "remedy": "No remedy is established."},
    {"id": "ambiguous", "label": "Ambiguous", "metric_name": "Abstention",
     "formula": "No category reaches its threshold",
     "desc": "The evidence does not support any one hypothesis, so VEO does not name a cause.",
     "remedy": "Inspect the region directly."},
]


def jpeg(image_bgr: np.ndarray, quality: int = 86) -> str:
    success, buffer = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        raise ValueError("Failed to encode image")
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("ascii")


def millimetres(depth: np.ndarray) -> str:
    return base64.b64encode(np.clip(np.round(depth * 1000.0), 0, 65535).astype(np.uint16).tobytes()).decode("ascii")


def colour(values: np.ndarray, low: float, high: float, colormap: int) -> np.ndarray:
    return cv2.applyColorMap((np.clip((values - low) / max(high - low, 1e-6), 0, 1) * 255).astype(np.uint8), colormap)


def choose_scenes(records: list[dict]) -> list[tuple[dict, str, dict]]:
    """For each label, the image whose largest region with that top label is biggest; every image used once."""
    chosen, used = [], set()
    for focus in FOCUS:
        candidates = [(region["area"], record["sample_id"], region["region_id"]) for record in records for region in record["regions"]
                      if region["top_cause"] == focus and record["sample_id"] not in used]
        if not candidates:
            continue
        _, sample_id, region_id = max(candidates)
        record = next(item for item in records if item["sample_id"] == sample_id)
        used.add(sample_id)
        chosen.append((record, focus, next(region for region in record["regions"] if region["region_id"] == region_id)))
    return chosen


def activations(model, processor, rgb: np.ndarray, device: str) -> dict:
    height, width = rgb.shape[:2]
    inputs = processor(images=rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        hidden = model(**inputs, output_hidden_states=True).hidden_states
    rows, columns = inputs["pixel_values"].shape[-2] // 14, inputs["pixel_values"].shape[-1] // 14
    maps = {}
    for layer, description in LAYERS.items():
        tokens = hidden[layer][0, 1:, :]
        magnitude = torch.norm(tokens, dim=-1).reshape(rows, columns).cpu().numpy()
        scaled = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-6)
        image = cv2.resize(cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS), (width, height), interpolation=cv2.INTER_CUBIC)
        maps[f"layer_{layer}"] = {"image": jpeg(image, 85), "description": description, "tokens_shape": [rows, columns, int(tokens.shape[-1])],
                                  "min_act": float(magnitude.min()), "max_act": float(magnitude.max())}
    return maps


def display_scores(region: dict, label_model) -> tuple[dict, float]:
    """Each score as a share of twice its label threshold: 0.50 or more means the label applies."""
    thresholds = {name: model["threshold"] for name, model in label_model.categories.items()}
    thresholds["poor_lighting"] = 0.5
    scores = {name: min(1.0, value / thresholds[name] / 2.0) for name, value in region["scores"].items()}
    ranked = sorted(scores.values(), reverse=True)
    return scores, float(ranked[0] - ranked[1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-path", type=Path, default=Path("outputs/nyu_test/depth-anything-v2-small-hf/model_results.json"))
    parser.add_argument("--output", type=Path, default=Path("web_app/data/app_data_v2.js"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    payload = json.loads(args.results_path.read_text(encoding="utf-8"))
    config = config_from_dict(payload["config"])
    label_model = load_label_model(config.experiment.label_model)
    split_dir = config.dataset.depth_dir.parent
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID).to(args.device).eval()

    scenes = []
    for record, focus, region in choose_scenes(payload["records"]):
        sample_id = record["sample_id"]
        pair = RGBDepthPair(sample_id, split_dir / "rgb" / f"{sample_id}.png", split_dir / "depth" / f"{sample_id}.npy")
        rgb = load_rgb(pair.rgb_path)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        _, target, dense, valid = load_sample(pair, config)
        computed = compute_regions(target, np.load(args.results_path.parent / "predictions" / f"{sample_id}.npy"), valid, config.experiment)
        aligned, error = computed["aligned"], computed["error_map"]
        height, width = target.shape
        low, high = float(np.percentile(dense, 1)), float(np.percentile(dense, 99))
        rows = np.linspace(0, height - 1, 60, dtype=int)
        columns = np.linspace(0, width - 1, 80, dtype=int)
        regions = []
        for item in record["regions"]:
            scores, margin = display_scores(item, label_model)
            regions.append({key: item[key] for key in ("region_id", "bbox", "area", "mean_error", "features", "top_cause", "labels", "confidence")}
                           | {"scores": scores, "score_margin": margin})
        x, y, w, h = region["bbox"]
        scenes.append({
            "sample_id": sample_id,
            "title": f"{FOCUS[focus]}: test image {sample_id}",
            "category_focus": FOCUS[focus],
            "default_region": {"x": x + w // 2, "y": y + h // 2, "desc": f"Largest region labelled {focus} in the test split"},
            "width": width,
            "height": height,
            "depth_range": [round(float(dense.min()), 3), round(float(dense.max()), 3)],
            "pred_range": [round(float(aligned[valid].min()), 3), round(float(aligned[valid].max()), 3)],
            "metrics": {**{key: float(record["metrics"][key]) for key in ("rmse", "mae", "abs_rel", "sq_rel", "delta1", "delta2", "delta3", "spearman")},
                        "failure_pixel_fraction": record["failure_pixel_fraction"], "total_regions": len(regions)},
            "alignment": {"scale": record["alignment"]["scale"], "shift": record["alignment"]["shift"], "space": "inverse_depth",
                          "formula": "1 / (scale * P_raw + shift), capped at 10 m"},
            "images": {
                "rgb": jpeg(bgr, 88),
                "depth_gt": jpeg(colour(dense, low, high, cv2.COLORMAP_TURBO), 85),
                "depth_pred": jpeg(colour(aligned, low, high, cv2.COLORMAP_TURBO), 85),
                "error_map": jpeg(colour(error, 0.0, 0.35, cv2.COLORMAP_INFERNO), 85),
            },
            "raw_depth": {"gt_b64_uint16": millimetres(dense), "pred_b64_uint16": millimetres(aligned), "unit": "millimeter", "scale_factor": 1000.0},
            "mesh_3d": {
                "grid_w": 80, "grid_h": 60,
                "depth_gt": dense[np.ix_(rows, columns)].round(3).tolist(),
                "depth_pred": aligned[np.ix_(rows, columns)].round(3).tolist(),
                "rgb_colors": (rgb[np.ix_(rows, columns)] / 255.0).round(2).tolist(),
            },
            "feature_maps": activations(model, processor, rgb, args.device),
            "regions": regions,
        })
        print(f"{sample_id}: {focus}, {len(regions)} regions", flush=True)

    data = {
        "version": "3.0",
        "dataset": {"name": "NYU Depth V2", "sensor": "Microsoft Kinect RGB-D structured-light sensor", "resolution": "480x640", "units": "meters",
                    "split": "Official test split (654 images); errors on sensor-measured pixels only"},
        "model": {"name": "Depth Anything V2 (ViT-Small)", "encoder": "DINOv2 ViT-Small (12 transformer blocks, patch size 14)",
                  "decoder": "DPT with multi-scale fusion", "checkpoint": MODEL_ID, "max_depth": 10.0, "patch_size": 14},
        "scenes": scenes,
        "taxonomy": TAXONOMY,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(f"window.VEO_DATA = {json.dumps(data)};", encoding="utf-8")
    print(f"Wrote {args.output} ({args.output.stat().st_size / 1e6:.1f} MB, {len(scenes)} scenes)")


if __name__ == "__main__":
    main()
