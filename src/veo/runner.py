import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .checkpoint import RecordStore, atomic_save_npy, atomic_write_text, load_npy
from .config import Config
from .data import (
    RGBDepthPair,
    discover_manifest,
    discover_pairs,
    load_depth,
    load_rgb,
    make_valid_mask,
    validate_pairs,
)
from .inference import (
    OfficialMetricDepthPredictor,
    PrecomputedPredictor,
    TransformersDepthPredictor,
)
from .label_model import model_fingerprint
from .metrics import align_scale_only, align_scale_shift, compute_error_map, compute_metrics
from .pipeline import align_prediction, analyze_sample
from .provenance import config_hash, provenance
from .regions import extract_failure_regions
from .reports import write_correlations, write_summary
from .visualization import save_panel

# (pair, rgb, target_shape) -> raw prediction with the target's shape
PredictFn = Callable[[RGBDepthPair, np.ndarray, tuple[int, int]], np.ndarray]


def _pairs(config: Config) -> list[RGBDepthPair]:
    pairs = discover_manifest(config.dataset.manifest) if config.dataset.manifest else discover_pairs(config.dataset.rgb_dir, config.dataset.depth_dir)
    validate_pairs(pairs, config.dataset.rgb_dir, config.dataset.depth_dir)
    return pairs


def _load_pair(pair: RGBDepthPair, config: Config) -> tuple[np.ndarray, np.ndarray]:
    rgb = load_rgb(pair.rgb_path)
    target = load_depth(pair.depth_path, config.dataset.depth_scale)
    if target.shape != rgb.shape[:2]:
        raise ValueError(f"RGB/depth shape mismatch for {pair.sample_id}: {rgb.shape[:2]} vs {target.shape}")
    return rgb, target


def load_sample(pair: RGBDepthPair, config: Config, max_depth_m: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """RGB, error target, dense geometry depth and evaluation mask for one pair.

    With ``raw_depth_dir`` set, errors are measured on sensor depth only and the dense depth
    from the manifest is kept for geometry cues.
    """
    rgb, dense = _load_pair(pair, config)
    target = dense
    if config.dataset.raw_depth_dir is not None:
        target = load_depth(config.dataset.raw_depth_dir / f"{pair.sample_id}.npy", config.dataset.depth_scale)
        if target.shape != dense.shape:
            raise ValueError(f"Raw/dense depth shape mismatch for {pair.sample_id}")
    max_depth = config.dataset.max_depth_m if max_depth_m is None else max_depth_m
    return rgb, target, dense, make_valid_mask(target, max_depth, config.dataset.eval_crop)


def run_pairs(
    config: Config,
    pairs: list[RGBDepthPair],
    predict: PredictFn,
    results_name: str,
    metadata: dict,
    save_predictions: bool = False,
    max_depth_m: float | None = None,
    save_panels: bool = True,
    resume: bool = True,
) -> list[dict]:
    """Analyze every pair and write results JSON, summary, region CSV, correlations and panels.

    Each image's record is checkpointed as soon as it is done, and saved predictions are reused, so an
    interrupted run continues where it stopped. Checkpoints made with other settings are ignored.
    """
    output_dir = config.outputs.directory
    prediction_dir = output_dir / "predictions"
    if save_predictions:
        prediction_dir.mkdir(parents=True, exist_ok=True)
    key = config_hash({"config": asdict(config), "metadata": metadata, "max_depth_m": max_depth_m, "label_model": model_fingerprint(config.experiment.label_model)})
    store = RecordStore(output_dir / "checkpoints" / Path(results_name).stem, key)
    records = []
    reused = 0
    for index, pair in enumerate(pairs, start=1):
        record = store.get(pair.sample_id) if resume else None
        if record is None:
            rgb, target, dense, valid_mask = load_sample(pair, config, max_depth_m)
            cached = load_npy(prediction_dir / f"{pair.sample_id}.npy") if save_predictions and resume else None
            if cached is not None and cached.shape == target.shape:
                prediction = cached.astype(np.float32)
            else:
                prediction = predict(pair, rgb, target.shape)
                if save_predictions:
                    atomic_save_npy(prediction_dir / f"{pair.sample_id}.npy", prediction)
            result = analyze_sample(rgb, target, prediction, valid_mask, config.experiment, geometry_depth=dense)
            record = {"sample_id": pair.sample_id, **result}
            if save_panels:
                # Panels show the same aligned prediction the analysis used, not the raw model output
                aligned, _, _ = align_prediction(prediction, target, valid_mask & np.isfinite(prediction), config.experiment)
                save_panel(rgb, target, aligned, compute_error_map(aligned, target, valid_mask), output_dir / "panels" / f"{pair.sample_id}.png", pair.sample_id)
            store.put(pair.sample_id, record)
        else:
            reused += 1
        records.append(record)
        if index % 100 == 0 or index == len(pairs):
            print(f"      {index}/{len(pairs)} images" + (f" ({reused} from checkpoints)" if reused else ""), flush=True)
    payload = {**metadata, "config": asdict(config), "provenance": provenance(config), "records": records}
    atomic_write_text(output_dir / results_name, json.dumps(payload, default=str, indent=2))
    write_summary(records, output_dir)
    write_correlations(records, output_dir / "correlations.csv")
    return records


def run_precomputed(config: Config, prediction_dir: Path) -> list[dict]:
    predictor = PrecomputedPredictor(prediction_dir)
    return run_pairs(
        config,
        _pairs(config),
        lambda pair, rgb, shape: predictor.predict_for_sample(pair.sample_id, shape),
        "precomputed_results.json",
        {"prediction_dir": str(prediction_dir)},
    )


def run_model(config: Config, model_id: str, device: str = "auto", limit: int | None = None, save_panels: bool = True, save_predictions: bool = True) -> list[dict]:
    pairs = _pairs(config)
    if limit is not None:
        pairs = pairs[:limit]
    loaded = {}

    def predict(pair, rgb, shape):
        # Loaded on first use, so a resumed run with every prediction cached never loads the model
        if "model" not in loaded:
            loaded["model"] = TransformersDepthPredictor(model_id, device)
        return loaded["model"].predict(rgb)

    return run_pairs(
        config,
        pairs,
        predict,
        "model_results.json",
        {"model_id": model_id},
        save_predictions=save_predictions,
        save_panels=save_panels,
    )


def run_official_metric(config: Config, official_repo: Path, checkpoint: Path, encoder: str, max_depth: float, device: str = "auto") -> list[dict]:
    predictor = OfficialMetricDepthPredictor(official_repo, checkpoint, encoder, max_depth, device)
    return run_pairs(
        config,
        _pairs(config),
        lambda pair, rgb, shape: predictor.predict(rgb),
        "official_metric_results.json",
        {"model": f"metric_hypersim_{encoder}"},
        save_predictions=True,
        max_depth_m=max_depth,
    )


def _scale_sign(scale: float | None) -> str:
    if scale is None:
        return "not_fitted"
    if scale > 0:
        return "positive"
    if scale < 0:
        return "negative"
    return "zero"


def _evaluate_alignment_protocol(
    prediction: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray,
    protocol: str,
    max_depth: float | None = None,
) -> dict:
    evaluated, scale, shift, space = _aligned_prediction(prediction, target, valid_mask, protocol, max_depth)
    return {
        "protocol": protocol,
        "metrics": compute_metrics(evaluated, target, valid_mask),
        "alignment": {
            "scale": scale,
            "shift": shift,
            "scale_sign": _scale_sign(scale),
            "prediction_space": space,
        },
    }


def _aligned_prediction(
    prediction: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray,
    protocol: str,
    max_depth: float | None = None,
) -> tuple[np.ndarray, float | None, float | None, str]:
    if protocol == "none":
        evaluated = prediction.astype(np.float32)
        scale, shift, space = None, None, "raw_inverse_depth"
    elif protocol == "direct_depth_scale_shift":
        evaluated, scale, shift = align_scale_shift(prediction, target, valid_mask, "depth", max_depth)
        space = "depth"
    elif protocol == "inverse_depth_scale_shift":
        evaluated, scale, shift = align_scale_shift(prediction, target, valid_mask, "inverse_depth", max_depth)
        space = "inverse_depth"
    elif protocol == "inverse_depth_scale_only":
        evaluated, scale = align_scale_only(prediction, target, valid_mask, "inverse_depth", max_depth)
        shift, space = 0.0, "inverse_depth"
    else:
        raise ValueError(f"Unknown alignment protocol: {protocol}")
    return evaluated, scale, shift, space


def _scene_lookup(config: Config) -> dict[str, str]:
    metadata_path = config.dataset.depth_dir.parent / "metadata.json"
    if not metadata_path.exists():
        return {}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {record["sample_id"]: record["scene"] for record in metadata}


def _bootstrap_mean_ci(records: list[dict], metric: str, scene_lookup: dict[str, str], seed: int, replicates: int = 1000) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for record in records:
        grouped.setdefault(scene_lookup.get(record["sample_id"], record["sample_id"]), []).append(record[metric])
    scenes = sorted(grouped)
    if not scenes:
        return {"lower": float("nan"), "upper": float("nan")}
    scene_means = np.asarray([np.mean(grouped[scene]) for scene in scenes], dtype=float)
    generator = np.random.default_rng(seed)
    samples = generator.integers(0, len(scene_means), size=(replicates, len(scene_means)))
    bootstrap_means = scene_means[samples].mean(axis=1)
    return {"lower": float(np.percentile(bootstrap_means, 2.5)), "upper": float(np.percentile(bootstrap_means, 97.5))}


def _region_sensitivity(
    prediction: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray,
    protocol: str,
    percentiles: tuple[float, ...],
    min_area: int,
    max_depth: float | None = None,
) -> dict[str, dict[str, float]]:
    evaluated, _, _, _ = _aligned_prediction(prediction, target, valid_mask, protocol, max_depth)
    error_map = compute_error_map(evaluated, target, valid_mask)
    result = {}
    for percentile in percentiles:
        regions = extract_failure_regions(error_map, valid_mask, percentile, min_area)
        region_pixels = sum(region["area"] for region in regions)
        result[str(int(percentile))] = {
            "num_regions": float(len(regions)),
            "region_pixels": float(region_pixels),
            "mean_region_area": float(region_pixels / len(regions)) if regions else 0.0,
        }
    return result


def run_alignment_ablation(
    config: Config,
    model_id: str,
    device: str = "auto",
    limit: int | None = None,
    start: int = 0,
    predictor=None,
) -> dict:
    """Compare alignment protocols per sample. Cached predictions are reused; the model is
    only loaded (or ``predictor`` used) for samples without one."""
    pairs = _pairs(config)
    if start < 0:
        raise ValueError("start must be non-negative")
    pairs = pairs[start:]
    if limit is not None:
        pairs = pairs[:limit]
    protocols = (
        "none",
        "direct_depth_scale_shift",
        "inverse_depth_scale_shift",
        "inverse_depth_scale_only",
    )
    if config.experiment.primary_protocol not in protocols:
        raise ValueError(f"Unsupported primary protocol: {config.experiment.primary_protocol}")
    sensitivity_percentiles = (75.0, 85.0, 90.0, 95.0)
    scene_lookup = _scene_lookup(config)
    records = []
    prediction_dir = config.outputs.directory / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    store = RecordStore(config.outputs.directory / "checkpoints" / "alignment_ablation", config_hash({"config": asdict(config), "protocols": protocols}))

    def append_record(pair, rgb, target, valid_mask, prediction):
        protocol_results = {
            protocol: _evaluate_alignment_protocol(prediction, target, valid_mask, protocol, config.experiment.max_prediction_m)
            for protocol in protocols
        }
        record = {
            "sample_id": pair.sample_id,
            "scene": scene_lookup.get(pair.sample_id, pair.sample_id),
            "protocols": protocol_results,
            "region_sensitivity": {
                "direct_depth_scale_shift": _region_sensitivity(
                    prediction, target, valid_mask, "direct_depth_scale_shift", sensitivity_percentiles, config.experiment.min_region_area,
                    config.experiment.max_prediction_m,
                ),
                "inverse_depth_scale_shift": _region_sensitivity(
                    prediction, target, valid_mask, "inverse_depth_scale_shift", sensitivity_percentiles, config.experiment.min_region_area,
                    config.experiment.max_prediction_m,
                ),
            },
        }
        store.put(pair.sample_id, record)
        records.append(record)

    pending = []
    for pair in pairs:
        stored = store.get(pair.sample_id)
        if stored is not None:
            records.append(stored)
            continue
        rgb, target, _, valid_mask = load_sample(pair, config)
        cached = load_npy(prediction_dir / f"{pair.sample_id}.npy")
        if cached is not None:
            prediction = cached.astype(np.float32)
            if prediction.shape != target.shape:
                raise ValueError(f"Cached prediction for {pair.sample_id} has shape {prediction.shape}; expected {target.shape}")
            append_record(pair, rgb, target, valid_mask, prediction)
        else:
            pending.append((pair, rgb, target, valid_mask))

    if pending and predictor is None:
        predictor = TransformersDepthPredictor(model_id, device)
    for batch_start in range(0, len(pending), 4):
        batch = pending[batch_start:batch_start + 4]
        rgb_batch = [item[1] for item in batch]
        if hasattr(predictor, "predict_batch"):
            predictions = predictor.predict_batch(rgb_batch, batch_size=4)
        else:
            predictions = [predictor.predict(rgb) for rgb in rgb_batch]
        for (pair, rgb, target, valid_mask), prediction in zip(batch, predictions, strict=True):
            atomic_save_npy(prediction_dir / f"{pair.sample_id}.npy", prediction)
            append_record(pair, rgb, target, valid_mask, prediction)
    metric_names = ("mae", "rmse", "abs_rel", "sq_rel", "delta1", "delta2", "delta3", "spearman")
    summary = {}
    for protocol in protocols:
        protocol_records = [record["protocols"][protocol] for record in records]
        fitted_scales = [record["alignment"]["scale"] for record in protocol_records if record["alignment"]["scale"] is not None]
        summary[protocol] = {
            "num_samples": len(protocol_records),
            "mean_metrics": {
                name: float(np.mean([record["metrics"][name] for record in protocol_records]))
                for name in metric_names
            } if protocol_records else {},
            "scale_sign_counts": {
                sign: sum(record["alignment"]["scale_sign"] == sign for record in protocol_records)
                for sign in ("positive", "negative", "zero", "not_fitted")
            },
            "mean_scale": float(np.mean(fitted_scales)) if fitted_scales else None,
            "std_scale": float(np.std(fitted_scales)) if fitted_scales else None,
            "bootstrap_ci95": {
                name: _bootstrap_mean_ci(
                    [{"sample_id": item["sample_id"], name: item["protocols"][protocol]["metrics"][name]} for item in records],
                    name,
                    scene_lookup,
                    config.experiment.random_seed,
                )
                for name in metric_names
            },
        }
    paired_differences = []
    for record in records:
        sample_differences = {"sample_id": record["sample_id"], "scene": record["scene"], "metrics": {}}
        for left_index, left in enumerate(protocols):
            for right in protocols[left_index + 1:]:
                key = f"{left}_minus_{right}"
                sample_differences["metrics"][key] = {
                    name: record["protocols"][left]["metrics"][name] - record["protocols"][right]["metrics"][name]
                    for name in metric_names
                }
        paired_differences.append(sample_differences)
    region_comparison = {}
    for percentile in sensitivity_percentiles:
        key = str(int(percentile))
        direct = [record["region_sensitivity"]["direct_depth_scale_shift"][key] for record in records]
        inverse = [record["region_sensitivity"]["inverse_depth_scale_shift"][key] for record in records]
        region_comparison[key] = {
            "mean_direct_regions": float(np.mean([item["num_regions"] for item in direct])) if direct else 0.0,
            "mean_inverse_regions": float(np.mean([item["num_regions"] for item in inverse])) if inverse else 0.0,
            "mean_direct_region_area": float(np.mean([item["mean_region_area"] for item in direct])) if direct else 0.0,
            "mean_inverse_region_area": float(np.mean([item["mean_region_area"] for item in inverse])) if inverse else 0.0,
        }
    output_dir = config.outputs.directory
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_id": model_id,
        "config": asdict(config),
        "provenance": provenance(config),
        "primary_protocol": config.experiment.primary_protocol,
        "protocols": protocols,
        "records": records,
        "summary": summary,
        "paired_differences": paired_differences,
        "region_comparison": region_comparison,
        "causal_claims": False,
    }
    atomic_write_text(output_dir / "alignment_ablation.json", json.dumps(payload, default=str, indent=2))
    return payload
