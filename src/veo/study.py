"""Build a blinded multi-rater study package from analysed results.

The package is a folder raters open offline: ``index.html`` (the labeling tool with the item list
embedded), ``images/`` and ``RATER_GUIDELINE.md``. The study team keeps ``manifest.json``, which links
item ids to regions, VEO's outputs, silver labels and sampling weights; raters never see it.

Sampling (protocol in docs/HUMAN_STUDY_PROTOCOL.md): up to ``per_stratum`` regions for each VEO top
label, plus ``random_count`` regions drawn uniformly from the rest, at most ``max_per_image`` regions
per image. Each item records its inclusion probability so estimates can be reweighted to the test set.
"""

import json
import shutil
import uuid
from collections.abc import Iterator
from importlib import resources
from pathlib import Path

import cv2
import numpy as np

from .config import config_from_dict
from .data import RGBDepthPair, load_rgb
from .pipeline import compute_regions
from .runner import load_sample
from .silver import class_ids, region_silver, silver_maps

CATEGORIES = [
    {"value": "textureless", "label": "Textureless surface"},
    {"value": "boundary_mixing", "label": "Depth boundary / occlusion edge"},
    {"value": "thin_object", "label": "Thin object"},
    {"value": "reflection", "label": "Reflective or transparent surface"},
    {"value": "poor_lighting", "label": "Too dark or overexposed"},
    {"value": "ambiguous", "label": "Other / cannot tell"},
]
STRATA = [category["value"] for category in CATEGORIES]
TOOL_VERSION = "1"


def load_results(results_path: Path) -> tuple[dict, list[dict]]:
    """Results payload plus one entry per region (with its scene, from the split's metadata)."""
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    metadata = config_from_dict(payload["config"]).dataset.depth_dir.parent / "metadata.json"
    scenes = {row["sample_id"]: row["scene"] for row in json.loads(metadata.read_text(encoding="utf-8"))} if metadata.exists() else {}
    regions = [
        {"sample_id": record["sample_id"], "scene": scenes.get(record["sample_id"], record["sample_id"]), "region_id": region["region_id"],
         "top_cause": region["top_cause"], "labels": region.get("labels", []), "bbox": region["bbox"], "area": region["area"],
         "features": region["features"]}
        for record in payload["records"] for region in record["regions"]
    ]
    return payload, regions


def stratified_sample(regions: list[dict], per_stratum: int, random_count: int, max_per_image: int, seed: int) -> list[dict]:
    """Stratified by VEO top label plus a uniform random top-up; returns copies with inclusion probabilities."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(regions))
    per_image: dict[str, int] = {}
    chosen: list[dict] = []
    taken: set[int] = set()

    def take(index: int, stratum: str) -> bool:
        region = regions[index]
        if index in taken or per_image.get(region["sample_id"], 0) >= max_per_image:
            return False
        per_image[region["sample_id"]] = per_image.get(region["sample_id"], 0) + 1
        taken.add(index)
        chosen.append({**region, "stratum": stratum})
        return True

    population = {stratum: sum(region["top_cause"] == stratum for region in regions) for stratum in STRATA}
    for stratum in STRATA:
        picked = 0
        for index in order:
            if picked >= per_stratum:
                break
            if regions[index]["top_cause"] == stratum and take(int(index), stratum):
                picked += 1
    remaining = len(regions) - len(taken)
    random_taken = 0
    for index in order:
        if random_taken >= random_count:
            break
        if take(int(index), "random"):
            random_taken += 1
    for item in chosen:
        if item["stratum"] == "random":
            item["inclusion_probability"] = random_taken / remaining if remaining else 1.0
        else:
            stratum_size = population[item["stratum"]]
            picked = sum(other["stratum"] == item["stratum"] for other in chosen)
            item["inclusion_probability"] = picked / stratum_size if stratum_size else 1.0
    return chosen


def iter_samples(payload: dict, results_path: Path) -> Iterator[tuple[dict, dict]]:
    """For each record: the record plus RGB, depths, region masks and the split folder."""
    config = config_from_dict(payload["config"])
    split_dir = config.dataset.depth_dir.parent
    for record in payload["records"]:
        sample_id = record["sample_id"]
        pair = RGBDepthPair(sample_id, split_dir / "rgb" / f"{sample_id}.png", split_dir / "depth" / f"{sample_id}.npy")
        _, target, dense, valid = load_sample(pair, config)
        prediction = np.load(results_path.parent / "predictions" / f"{sample_id}.npy")
        computed = compute_regions(target, prediction, valid, config.experiment)
        yield record, {"rgb": load_rgb(pair.rgb_path), "target": target, "dense": dense, "split_dir": split_dir, **computed}


def _colourise(values: np.ndarray, low: float, high: float, colormap: int) -> np.ndarray:
    scaled = np.clip((values - low) / max(high - low, 1e-6), 0, 1)
    return cv2.applyColorMap((scaled * 255).astype(np.uint8), colormap)


def render_item(sample: dict, mask: np.ndarray, context_path: Path, detail_path: Path, panel_height: int = 260) -> None:
    """Whole image with the region outlined, and four aligned close-ups (RGB, true depth, prediction, error)."""
    bgr = cv2.cvtColor(sample["rgb"], cv2.COLOR_RGB2BGR)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    context = bgr.copy()
    cv2.drawContours(context, contours, -1, (0, 255, 255), 2)
    cv2.imwrite(str(context_path), context, [cv2.IMWRITE_JPEG_QUALITY, 92])

    ys, xs = np.where(mask)
    height, width = mask.shape
    pad = max(12, int(0.5 * max(np.ptp(ys), np.ptp(xs))))
    crop = (slice(max(ys.min() - pad, 0), min(ys.max() + pad + 1, height)), slice(max(xs.min() - pad, 0), min(xs.max() + pad + 1, width)))
    true_disparity = 1.0 / np.clip(sample["dense"][crop], 1e-3, None)
    predicted_disparity = 1.0 / np.clip(sample["aligned"][crop], 1e-3, None)
    # Disparity colouring so near is bright. Both panels share one range taken from both depths,
    # so a prediction far off the true depth shows as a colour difference rather than a blank panel.
    both = np.concatenate([true_disparity.ravel(), predicted_disparity.ravel()])
    both = both[np.isfinite(both) & (both < 1e3)]
    low, high = (np.percentile(both, 2), np.percentile(both, 98)) if both.size else (0.1, 1.0)
    true_panel = _colourise(true_disparity, low, high, cv2.COLORMAP_MAGMA)
    predicted_panel = _colourise(predicted_disparity, low, high, cv2.COLORMAP_MAGMA)
    error_panel = _colourise(sample["error_map"][crop], 0.0, 0.5, cv2.COLORMAP_INFERNO)
    error_panel[~sample["valid_mask"][crop]] = (90, 90, 90)  # no sensor measurement
    local_contours, _ = cv2.findContours(mask[crop].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    panels = [bgr[crop].copy(), true_panel, predicted_panel, error_panel]
    for panel in panels:
        cv2.drawContours(panel, local_contours, -1, (0, 255, 255), 1)
    scale = panel_height / panels[0].shape[0]
    resized = [cv2.resize(panel, (max(1, int(panel.shape[1] * scale)), panel_height), interpolation=cv2.INTER_NEAREST) for panel in panels]
    gap = np.full((panel_height, 6, 3), 255, dtype=np.uint8)
    strip = resized[0]
    for panel in resized[1:]:
        strip = np.hstack([strip, gap, panel])
    cv2.imwrite(str(detail_path), strip, [cv2.IMWRITE_JPEG_QUALITY, 92])


def practice_candidates(train_results: Path, count: int, seed: int) -> list[dict]:
    """Train-split regions whose silver label is a single category with a clear majority of pixels."""
    payload, _ = load_results(train_results)
    rng = np.random.default_rng(seed)
    candidates = []
    groups = None
    for record, sample in iter_samples(payload, train_results):
        if groups is None:
            groups = class_ids(json.loads((sample["split_dir"] / "label_names.json").read_text(encoding="utf-8")))
        instances_path = sample["split_dir"] / "instances" / f"{record['sample_id']}.npy"
        maps = silver_maps(
            np.load(sample["split_dir"] / "labels" / f"{record['sample_id']}.npy"),
            np.load(instances_path) if instances_path.exists() else None, sample["dense"], groups,
        )
        for region in sample["regions"]:
            silver = region_silver(maps, region["mask"])
            if len(silver["labels"]) == 1 and silver["shares"][silver["labels"][0]] >= 0.6:
                candidates.append({"sample_id": record["sample_id"], "region_id": region["region_id"], "reference": silver["labels"][0]})
    by_label: dict[str, list[dict]] = {}
    for index in rng.permutation(len(candidates)):
        by_label.setdefault(candidates[index]["reference"], []).append(candidates[index])
    chosen: list[dict] = []
    while len(chosen) < count and any(by_label.values()):  # round-robin across labels
        for label in sorted(by_label):
            if by_label[label] and len(chosen) < count:
                chosen.append(by_label[label].pop())
    return chosen


def _render_set(results_path: Path, wanted: dict[tuple[str, int], dict], images_dir: Path) -> None:
    """Render each wanted region and attach its silver labels when the split has annotations."""
    payload, _ = load_results(results_path)
    needed = {sample_id for sample_id, _ in wanted}
    payload = {**payload, "records": [record for record in payload["records"] if record["sample_id"] in needed]}
    groups = None
    for record, sample in iter_samples(payload, results_path):
        masks = {region["region_id"]: region["mask"] for region in sample["regions"]}
        labels_path = sample["split_dir"] / "labels" / f"{record['sample_id']}.npy"
        maps = None
        if labels_path.exists():
            groups = groups or class_ids(json.loads((sample["split_dir"] / "label_names.json").read_text(encoding="utf-8")))
            instances_path = sample["split_dir"] / "instances" / f"{record['sample_id']}.npy"
            maps = silver_maps(np.load(labels_path), np.load(instances_path) if instances_path.exists() else None, sample["dense"], groups)
        for (sample_id, region_id), item in wanted.items():
            if sample_id != record["sample_id"]:
                continue
            context, detail = images_dir / f"{item['id']}_context.jpg", images_dir / f"{item['id']}_detail.jpg"
            if not (context.exists() and detail.exists()):
                render_item(sample, masks[region_id], context, detail)
            if maps is not None:
                silver = region_silver(maps, masks[region_id])
                item["silver_labels"], item["silver_shares"] = silver["labels"], silver["shares"]


def build_study(test_results: Path, output_dir: Path, train_results: Path | None = None, per_stratum: int = 40,
                random_count: int = 60, practice_count: int = 20, max_per_image: int = 3, seed: int = 42) -> dict:
    # Item ids are seeded, so a rebuild reuses images already rendered by an interrupted run
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    def new_id() -> str:
        return uuid.UUID(int=int(rng.integers(0, 2**63)) << 64 | int(rng.integers(0, 2**63))).hex[:12]

    _, regions = load_results(test_results)
    sampled = stratified_sample(regions, per_stratum, random_count, max_per_image, seed)
    items = {(item["sample_id"], item["region_id"]): {**item, "id": new_id()} for item in sampled}
    _render_set(test_results, items, images_dir)
    practice = {}
    if train_results is not None and practice_count:
        practice = {(c["sample_id"], c["region_id"]): {**c, "id": new_id()} for c in practice_candidates(train_results, practice_count, seed)}
        _render_set(train_results, practice, images_dir)

    study_id = new_id()
    tool_data = {
        "study_id": study_id,
        "tool_version": TOOL_VERSION,
        "categories": CATEGORIES,
        "items": [{"id": item["id"], "context": f"images/{item['id']}_context.jpg", "detail": f"images/{item['id']}_detail.jpg"} for item in items.values()],
        "practice": [
            {"id": item["id"], "context": f"images/{item['id']}_context.jpg", "detail": f"images/{item['id']}_detail.jpg", "reference": item["reference"]}
            for item in practice.values()
        ],
    }
    template = resources.files("veo").joinpath("templates/study_tool.html").read_text(encoding="utf-8")
    # Embedded as JSON inside a script tag; escape "</" so item data can never close the tag
    (output_dir / "index.html").write_text(template.replace("/*STUDY_DATA*/", json.dumps(tool_data).replace("</", "<\\/")), encoding="utf-8")
    guideline = Path(__file__).resolve().parents[2] / "docs" / "RATER_GUIDELINE.md"
    if guideline.exists():
        shutil.copyfile(guideline, output_dir / "RATER_GUIDELINE.md")
    manifest = {
        "study_id": study_id,
        "test_results": str(test_results),
        "train_results": str(train_results) if train_results else None,
        "seed": seed,
        "design": {"per_stratum": per_stratum, "random_count": random_count, "max_per_image": max_per_image, "practice_count": practice_count},
        "items": list(items.values()),
        "practice": list(practice.values()),
    }
    # Kept outside the folder sent to raters so VEO's outputs cannot be seen
    referenced = {Path(entry[key]).name for entry in tool_data["items"] + tool_data["practice"] for key in ("context", "detail")}
    for image in images_dir.iterdir():
        if image.name not in referenced:
            image.unlink()  # left over from a study built with other settings
    manifest_path = output_dir.parent / f"{output_dir.name}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"output_dir": output_dir, "manifest": manifest_path, "items": len(items), "practice": len(practice)}
