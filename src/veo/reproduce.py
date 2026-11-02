"""One command from raw data to a results folder: splits, predictions, analysis, experiments, summary.

``fixture=True`` runs every stage on a small synthetic NYU-format dataset with a synthetic predictor,
so the pipeline can be checked end to end without downloads or a GPU.
"""

import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

from .checkpoint import mark_done, stage_done
from .config import Config, DatasetConfig, OutputConfig, load_config
from .data import discover_manifest, load_depth
from .fixtures import SyntheticDisparityPredictor, write_fixture_mat, write_fixture_splits
from .inference import PrecomputedPredictor
from .label_model import model_fingerprint
from .nyu_dataset import create_scene_splits, extract_split, official_splits
from .provenance import config_hash
from .reports import write_result
from .runner import run_alignment_ablation, run_model, run_pairs
from .summary import write_summary_markdown
from .validation import cohens_kappa

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = REPO_ROOT / "experiments"
FIXTURE_MODEL = "synthetic-disparity"
# Sensitivity of the failure definition (protocol v2); the primary rule is delta 1.25
FAILURE_RULES = {
    "delta_1.10": {"region_mode": "delta", "delta_threshold": 1.10},
    "delta_1.5625": {"region_mode": "delta", "delta_threshold": 1.5625},
    "percentile_85": {"region_mode": "percentile", "error_percentile": 85.0},
    # Speckle cleanup: does a 1 px opening erase thin failure bands (edges, thin objects)?
    "delta_1.25_opening_1": {"region_mode": "delta", "delta_threshold": 1.25, "opening_radius": 1},
}


def _slug(model_id: str) -> str:
    return model_id.split("/")[-1].lower()


def split_records(mat_path: Path, seed: int) -> tuple[str, dict[str, list[dict]]]:
    """The official train/test split when splits.mat sits next to the dataset, else a seeded scene split."""
    splits_path = mat_path.with_name("splits.mat")
    if splits_path.exists():
        return "official", official_splits(mat_path, splits_path)
    return f"scene_seed_{seed}", create_scene_splits(mat_path, seed)


def prepare_split(mat_path: Path, split_root: Path, seed: int, which: str = "test") -> tuple[Path, str]:
    """Extract one split (``test`` or ``train``) with its manifest and scene metadata; returns its folder and split name."""
    split_name, splits = split_records(mat_path, seed)
    records = splits[which] if which in splits else splits["development"]  # seeded splits call train "development"
    split_dir = split_root / which
    extract_split(mat_path, records, split_dir)
    (split_dir / "metadata.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    lines = [f"{split_dir / 'rgb' / record['sample_id']}.png {split_dir / 'depth' / record['sample_id']}.npy" for record in records]
    (split_dir / "manifest.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return split_dir, split_name


def _split_config(base: Config, split_dir: Path, output_dir: Path, split_name: str, eval_crop: str | None, which: str = "test") -> Config:
    raw_dir = split_dir / "raw_depth"
    dataset = DatasetConfig(
        name=base.dataset.name,
        rgb_dir=split_dir / "rgb",
        depth_dir=split_dir / "depth",
        manifest=split_dir / "manifest.txt",
        depth_scale=base.dataset.depth_scale,
        max_depth_m=base.dataset.max_depth_m,
        split=f"{split_name}_{which}",
        # Errors on sensor depth when the config asks for it and the dataset provides it
        raw_depth_dir=raw_dir if base.dataset.raw_depth_dir is not None and raw_dir.exists() else None,
        eval_crop=eval_crop,
    )
    return replace(base, dataset=dataset, outputs=OutputConfig(directory=output_dir))


def _predict(config: Config, model_id: str, fixture: bool, seed: int, device: str, save_panels: bool = True) -> Path:
    """Predict and analyze every pair of the configured split; returns the results file."""
    if fixture:
        predictor = SyntheticDisparityPredictor(seed)
        run_pairs(
            config,
            discover_manifest(config.dataset.manifest),
            lambda pair, rgb, shape: predictor.predict(rgb, load_depth(pair.depth_path, config.dataset.depth_scale)),
            "model_results.json",
            {"model_id": model_id},
            save_predictions=True,
            save_panels=save_panels,
        )
    else:
        run_model(config, model_id, device, save_panels=save_panels)
    return config.outputs.directory / "model_results.json"


def _run_experiment(script: str, *args: str | Path) -> None:
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(REPO_ROOT / "src"), os.environ.get("PYTHONPATH")]))}
    subprocess.run([sys.executable, str(EXPERIMENTS / script), *map(str, args)], check=True, env=env)


def geometry_source_agreement(config: Config, pairs: list) -> dict:
    """Labels from prediction-only geometry (no ground truth) versus the ground-truth audit labels.

    Failure regions depend only on the error map, so both runs produce the same regions and can be
    compared one to one.
    """
    predictor = PrecomputedPredictor(config.outputs.directory / "predictions")
    deploy_config = replace(
        config,
        experiment=replace(config.experiment, geometry_source="prediction"),
        outputs=OutputConfig(directory=config.outputs.directory / "geometry_prediction"),
    )
    run_pairs(
        deploy_config, pairs, lambda pair, rgb, shape: predictor.predict_for_sample(pair.sample_id, shape),
        "model_results.json", {"geometry_source": "prediction"}, save_panels=False,
    )

    def top_labels(directory: Path) -> dict[tuple[str, int], tuple[str, frozenset]]:
        records = json.loads((directory / "model_results.json").read_text(encoding="utf-8"))["records"]
        return {(r["sample_id"], g["region_id"]): (g["top_cause"], frozenset(g["labels"])) for r in records for g in r["regions"]}

    audit, deploy = top_labels(config.outputs.directory), top_labels(deploy_config.outputs.directory)
    keys = sorted(audit.keys() & deploy.keys())
    if not keys:
        return {"regions": 0}
    audit_top, deploy_top = [audit[key][0] for key in keys], [deploy[key][0] for key in keys]
    categories = sorted({label for key in keys for label in audit[key][1] | deploy[key][1]})
    per_label = {}
    for label in categories:
        in_audit = [label in audit[key][1] for key in keys]
        in_deploy = [label in deploy[key][1] for key in keys]
        per_label[label] = {
            "audit": sum(in_audit),
            "prediction_only": sum(in_deploy),
            "both": sum(a and d for a, d in zip(in_audit, in_deploy, strict=True)),
        }
    return {
        "regions": len(keys),
        "top_cause_agreement": sum(a == d for a, d in zip(audit_top, deploy_top, strict=True)) / len(keys),
        "top_cause_kappa": cohens_kappa(audit_top, deploy_top),
        "per_label": per_label,
    }


def failure_rule_sensitivity(config: Config, pairs: list, primary_summary: dict) -> dict:
    """Re-run the analysis on cached predictions under each alternative failure rule."""
    predictor = PrecomputedPredictor(config.outputs.directory / "predictions")
    keys = ("num_regions", "mean_failure_pixel_fraction", "region_pixel_fraction", "cause_counts", "label_pixel_fraction")
    rules = {f"delta_{config.experiment.delta_threshold:g} (primary)": {key: primary_summary.get(key) for key in keys}}
    for name, overrides in FAILURE_RULES.items():
        rule_config = replace(
            config,
            experiment=replace(config.experiment, **overrides),
            outputs=OutputConfig(directory=config.outputs.directory / "failure_rules" / name),
        )
        run_pairs(
            rule_config, pairs, lambda pair, rgb, shape: predictor.predict_for_sample(pair.sample_id, shape),
            "model_results.json", {"failure_rule": overrides}, save_panels=False,
        )
        summary = json.loads((rule_config.outputs.directory / "summary.json").read_text(encoding="utf-8"))
        rules[name] = {key: summary.get(key) for key in keys}
    return {"rules": rules}


def _slim_alignment(payload: dict) -> dict:
    """Keep the summary and per-scene scales; drop per-sample metric dumps."""
    scales = [
        {
            "sample_id": record["sample_id"],
            "direct_depth_scale": record["protocols"]["direct_depth_scale_shift"]["alignment"]["scale"],
            "inverse_depth_scale": record["protocols"]["inverse_depth_scale_shift"]["alignment"]["scale"],
        }
        for record in payload["records"]
    ]
    return {"model_id": payload["model_id"], "summary": payload["summary"], "region_comparison": payload["region_comparison"], "per_scene_scales": scales}


def _experiment_stage(stages: Path, key: str, output: Path, script: str, *args: str | Path) -> None:
    """Run an experiment script unless it already finished with these settings and its output exists."""
    if output.exists() and stage_done(stages, output.stem + "_" + output.parent.name, key):
        print(f"      (resumed) {script} already done")
        return
    _run_experiment(script, *args, "--output", output)
    mark_done(stages, output.stem + "_" + output.parent.name, key)


def reproduce(
    workdir: Path,
    config_path: Path,
    fixture: bool = False,
    mat_path: Path | None = None,
    model_ids: tuple[str, ...] = ("depth-anything/Depth-Anything-V2-Small-hf",),
    rater1: Path | None = None,
    rater2: Path | None = None,
    device: str = "auto",
    seed: int = 42,
    fresh: bool = False,
    extra_models: tuple[str, ...] = (),
) -> Path:
    """Run every stage; an interrupted run resumes from its checkpoints when called again.

    Per-image work (predictions, analysis records, alignment comparisons, interventions) is
    checkpointed as it finishes, and finished stages are skipped. ``fresh`` discards the results and
    analysis checkpoints (extracted splits and model predictions are kept, since they do not depend
    on VEO's settings). Checkpoints are keyed by settings, not by code version: after changing VEO's
    code, use ``fresh`` to recompute.
    """
    run_name = "fixture" if fixture else "nyu_test"
    outputs = workdir / "outputs" / run_name
    results = workdir / "results" / run_name
    stages = outputs / "stages"
    if fresh:
        shutil.rmtree(results, ignore_errors=True)
        shutil.rmtree(stages, ignore_errors=True)
        for checkpoints in outputs.glob("**/checkpoints"):
            shutil.rmtree(checkpoints, ignore_errors=True)
        for log in outputs.glob("**/*.checkpoint.jsonl"):
            log.unlink()
    results.mkdir(parents=True, exist_ok=True)

    if fixture:
        mat_path = outputs / "nyu_fixture.mat"
        if not mat_path.exists():
            write_fixture_mat(mat_path, scenes=12)
            write_fixture_splits(mat_path.with_name("splits.mat"), scenes=12)
        model_ids = (FIXTURE_MODEL,)
    elif mat_path is None or not mat_path.exists():
        raise FileNotFoundError(f"NYU labelled .mat not found at {mat_path}. Download it with: veo download-nyu")
    base = load_config(config_path)
    key = config_hash({"config": asdict(base), "primary_model": model_ids[0], "seed": seed, "fixture": fixture,
                       "label_model": model_fingerprint(base.experiment.label_model),
                       "data": [mat_path.name, mat_path.stat().st_size]})

    print(f"[1/5] Extracting the test split from {mat_path}", flush=True)
    split_dir, split_name = prepare_split(mat_path, outputs / "splits", seed)

    eval_crop = None if fixture else base.dataset.eval_crop  # the eigen crop is defined for 480x640 frames only
    primary_results = None
    for model_id in model_ids:
        slug = _slug(model_id)
        config = _split_config(base, split_dir, outputs / slug, split_name, eval_crop)
        print(f"[2/5] Predicting and analyzing with {model_id}", flush=True)
        _predict(config, model_id, fixture, seed, device)
        alignment = run_alignment_ablation(config, model_id, device)  # reuses the cached predictions
        model_results = results / slug
        model_results.mkdir(exist_ok=True)
        shutil.copyfile(config.outputs.directory / "summary.json", model_results / "summary.json")
        write_result(model_results / "alignment_ablation.json", _slim_alignment(alignment), config)
        if primary_results is None:
            print("      failure-rule sensitivity on cached predictions", flush=True)
            primary_summary = json.loads((model_results / "summary.json").read_text(encoding="utf-8"))
            sensitivity = failure_rule_sensitivity(config, discover_manifest(config.dataset.manifest), primary_summary)
            write_result(model_results / "failure_rules.json", sensitivity, config)
            print("      ground-truth-free geometry on cached predictions", flush=True)
            agreement = geometry_source_agreement(config, discover_manifest(config.dataset.manifest))
            write_result(model_results / "geometry_agreement.json", agreement, config)
            primary_results = config.outputs.directory / "model_results.json"
            primary_model, primary_slug, primary_config = model_id, slug, config

    for model_id in extra_models:
        # Analysis only: enough for the model comparison, and it keeps predictions off the disk
        slug = _slug(model_id)
        config = _split_config(base, split_dir, outputs / slug, split_name, eval_crop)
        print(f"[2/5] Predicting and analyzing with {model_id} (no panels or saved predictions)", flush=True)
        run_model(config, model_id, device, save_panels=False, save_predictions=False)
        (results / slug).mkdir(exist_ok=True)
        shutil.copyfile(config.outputs.directory / "summary.json", results / slug / "summary.json")
    if len(model_ids) + len(extra_models) > 1:
        comparison_key = config_hash({"key": key, "models": sorted((*model_ids, *extra_models))})
        _experiment_stage(stages, comparison_key, results / "model_comparison.json", "compare_models.py", "--root", outputs)

    print("[3/5] Running experiments on the primary model", flush=True)
    primary_predictions = primary_results.parent / "predictions"
    _experiment_stage(stages, key, results / "signal_ablation.json", "signal_ablation.py", "--results-path", primary_results)
    _experiment_stage(
        stages, key, results / "scoring_sanity_check.json", "scoring_sanity_check.py", "--results-path", primary_results,
        "--rgb-dir", split_dir / "rgb", "--depth-dir", split_dir / "depth",
    )
    _experiment_stage(
        stages, key, results / "oracle_boundary_filter.json", "oracle_boundary_filter.py", "--results-path", primary_results,
        "--prediction-dir", primary_predictions, "--depth-dir", primary_config.dataset.raw_depth_dir or primary_config.dataset.depth_dir,
        *(["--crop", eval_crop] if eval_crop else []),
    )
    print("      model-in-the-loop interventions (texture, exposure, resolution)", flush=True)
    _experiment_stage(
        stages, key, results / primary_slug / "interventions.json", "causal_interventions.py",
        "--results-path", primary_results, "--model-id", primary_model, "--device", device,
    )
    if rater1:
        args = ["--rater1", rater1, "--results-path", primary_results, *(["--rater2", rater2] if rater2 else [])]
        _experiment_stage(stages, key, results / "human_agreement.json", "human_agreement.py", *args)

    if (split_dir / "labels").exists():
        print(f"[4/5] Objective-label calibration: fit on the train split with {primary_model}, score on test", flush=True)
        train_dir, _ = prepare_split(mat_path, outputs / "splits", seed, "train")
        train_config = _split_config(base, train_dir, outputs / primary_slug / "train", split_name, eval_crop, "train")
        train_results = _predict(train_config, primary_model, fixture, seed, device, save_panels=False)
        _experiment_stage(
            stages, key, results / primary_slug / "silver_calibration.json", "silver_calibration.py",
            "--train-results", train_results, "--test-results", primary_results,
        )
        _experiment_stage(
            stages, key, results / primary_slug / "patch_cnn.json", "patch_cnn_baseline.py",
            "--train-results", train_results, "--test-results", primary_results,
            "--checkpoints", results / primary_slug / "silver_calibration", "--device", device, *(["--epochs", "2"] if fixture else []),
        )

    print("[5/5] Writing summary", flush=True)
    summary = write_summary_markdown(results)
    print(f"Done: {summary}", flush=True)
    return summary
