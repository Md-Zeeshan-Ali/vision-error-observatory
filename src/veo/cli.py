import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from .config import Config, DatasetConfig, ExperimentConfig, OutputConfig, load_config
from .data import discover_manifest, discover_pairs, validate_pairs
from .download import NYU_LABELED_URL, NYU_SPLITS_URL, download_file
from .nyu_dataset import create_scene_splits, extract_labeled_mat, extract_split
from .pipeline import analyze_sample
from .reports import write_manifest
from .reproduce import reproduce
from .runner import run_alignment_ablation, run_model, run_official_metric, run_precomputed
from .validation import export_rating_set


def download_nyu(destination: Path, url: str = NYU_LABELED_URL, splits_url: str = NYU_SPLITS_URL) -> None:
    """Download the NYU Depth V2 labelled set (about 3 GB) and its official train/test split; print SHA-256s."""
    from .provenance import file_sha256

    for source, target in ((url, destination), (splits_url, destination.with_name("splits.mat"))):
        if target.exists():
            print(f"Already present: {target}")
        else:
            print(f"Downloading {source} -> {target}", flush=True)
            download_file(source, target)
        print(f"sha256 {file_sha256(target)}  {target.name}")


def run_with_retries(work, retries: int, wait_seconds: float) -> None:
    """Run ``work(attempt)``; after an error, wait and run it again (checkpoints make this a resume).

    Covers transient failures such as a dropped connection. Ctrl+C stops immediately; after a
    shutdown, running the same command again resumes from the checkpoints.
    """
    import time

    for attempt in range(1, retries + 2):
        try:
            work(attempt)
            return
        except KeyboardInterrupt:
            raise
        except Exception as error:  # noqa: BLE001 - any failure is retried; checkpoints keep finished work
            if attempt > retries:
                raise
            print(f"Run failed ({type(error).__name__}: {error}). Resuming in {wait_seconds:.0f} s "
                  f"(attempt {attempt + 1} of {retries + 1}).", flush=True)
            time.sleep(wait_seconds)


def run_smoke() -> None:
    height, width = 48, 64
    rng = np.random.default_rng(42)
    rgb = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    target = np.full((height, width), 2.0, dtype=np.float32)
    target[:, width // 2:] = 4.0
    prediction = target + rng.normal(0, 0.15, size=target.shape).astype(np.float32)
    prediction[18:30, 28:40] += 1.5
    valid_mask = np.ones_like(target, dtype=bool)
    result = analyze_sample(rgb, target, prediction, valid_mask, ExperimentConfig(min_region_area=5))
    print(f"Smoke test passed: AbsRel={result['metrics']['abs_rel']:.4f}, regions={result['num_regions']}")


def configured(args, config: Config, max_depth: float | None = None) -> Config:
    dataset = DatasetConfig(
        name=config.dataset.name,
        rgb_dir=args.rgb_dir or config.dataset.rgb_dir,
        depth_dir=args.depth_dir or config.dataset.depth_dir,
        manifest=args.manifest if args.manifest else config.dataset.manifest,
        depth_scale=config.dataset.depth_scale,
        max_depth_m=max_depth if max_depth is not None else config.dataset.max_depth_m,
        split=config.dataset.split,
    )
    return Config(dataset=dataset, experiment=config.experiment, outputs=OutputConfig(directory=args.output_dir or config.outputs.directory))


def main() -> None:
    parser = argparse.ArgumentParser(prog="veo")
    parser.add_argument("command", choices=["smoke", "manifest", "prepare-nyu-splits", "extract-nyu", "evaluate-splits", "compare-alignment", "prepare-rating", "run-precomputed", "run-model", "run-official-metric", "reproduce", "download-nyu", "prepare-study"])
    parser.add_argument("--config", type=Path, default=Path("configs/nyu.yaml"))
    parser.add_argument("--rgb-dir", type=Path)
    parser.add_argument("--depth-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--prediction-dir", type=Path, default=Path("data/nyu/predictions"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-id", default="depth-anything/Depth-Anything-V2-Small-hf")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--official-repo", type=Path, default=Path("official/Depth-Anything-V2"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/depth_anything_v2_metric_hypersim_vits.pth"))
    parser.add_argument("--encoder", choices=["vits", "vitb", "vitl"], default="vits")
    parser.add_argument("--max-depth", type=float, default=20.0)
    parser.add_argument("--mat-path", type=Path, default=Path("data/nyu/nyu_depth_v2_labeled.mat"))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-root", type=Path, default=Path("data/nyu/splits"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--results-path", type=Path)
    parser.add_argument("--fixture", action="store_true", help="reproduce: run on a synthetic dataset (no downloads)")
    parser.add_argument("--workdir", type=Path, default=Path("."), help="reproduce: root for outputs/ and results/")
    parser.add_argument("--models", nargs="+", help="reproduce: Hugging Face model ids (default: --model-id)")
    parser.add_argument("--train-results", type=Path, help="prepare-study: train model_results.json for practice items")
    parser.add_argument("--per-stratum", type=int, default=40)
    parser.add_argument("--random-count", type=int, default=60)
    parser.add_argument("--practice-count", type=int, default=20)
    parser.add_argument("--extra-models", nargs="+", help="reproduce: more models, analysed for the comparison only")
    parser.add_argument("--fresh", action="store_true", help="reproduce: discard results and analysis checkpoints first")
    parser.add_argument("--retries", type=int, default=3, help="reproduce: resume automatically this many times after an error")
    parser.add_argument("--retry-wait", type=float, default=60.0, help="reproduce: seconds to wait before resuming")
    parser.add_argument("--rater1", type=Path)
    parser.add_argument("--rater2", type=Path)
    args = parser.parse_args()
    if args.command == "smoke":
        run_smoke()
    elif args.command == "reproduce":
        run_with_retries(
            lambda attempt: reproduce(
                args.workdir,
                args.config,
                fixture=args.fixture,
                mat_path=args.mat_path,
                model_ids=tuple(args.models or [args.model_id]),
                rater1=args.rater1,
                rater2=args.rater2,
                device=args.device,
                seed=args.seed,
                fresh=args.fresh and attempt == 1,  # retries resume rather than start over
                extra_models=tuple(args.extra_models or ()),
            ),
            args.retries,
            args.retry_wait,
        )
    elif args.command == "download-nyu":
        download_nyu(args.mat_path)
    elif args.command == "prepare-study":
        from .study import build_study

        if not args.results_path:
            raise ValueError("--results-path (test model_results.json) is required for prepare-study")
        built = build_study(
            args.results_path, args.output_dir or Path("study"), train_results=args.train_results,
            per_stratum=args.per_stratum, random_count=args.random_count, practice_count=args.practice_count, seed=args.seed,
        )
        print(f"Study with {built['items']} items and {built['practice']} practice items: {built['output_dir']}")
        print(f"Keep the manifest away from raters: {built['manifest']}")
    elif args.command == "manifest":
        config = load_config(args.config)
        config = configured(args, config)
        pairs = discover_manifest(config.dataset.manifest) if config.dataset.manifest else discover_pairs(config.dataset.rgb_dir, config.dataset.depth_dir)
        validate_pairs(pairs, config.dataset.rgb_dir, config.dataset.depth_dir)
        manifest_path = config.outputs.directory / f"{config.dataset.split}_manifest.json"
        write_manifest(pairs, manifest_path, config.dataset.split)
        print(f"Manifest created for {len(pairs)} samples: {manifest_path}")
    elif args.command == "extract-nyu":
        extracted = extract_labeled_mat(args.mat_path, args.rgb_dir or Path("data/nyu/rgb"), args.depth_dir or Path("data/nyu/depth"), args.start, args.count)
        print(f"Extracted {extracted} official NYU RGB/depth pairs")
    elif args.command == "prepare-nyu-splits":
        split_root = args.output_dir or Path("data/nyu/splits")
        splits = create_scene_splits(args.mat_path, args.seed)
        scene_sets = {}
        for split, records in splits.items():
            scene_sets[split] = sorted({record["scene"] for record in records})
            split_dir = split_root / split
            extract_split(args.mat_path, records, split_dir)
            (split_dir / "metadata.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
            (split_dir / "manifest.txt").write_text("\n".join(f"{split_dir / 'rgb' / record['sample_id']}.png {split_dir / 'depth' / record['sample_id']}.npy" for record in records) + "\n", encoding="utf-8")
        (split_root / "split_summary.json").write_text(json.dumps({"seed": args.seed, "scene_sets": scene_sets}, indent=2), encoding="utf-8")
        print({split: len(records) for split, records in splits.items()})
    elif args.command == "run-precomputed":
        config = load_config(args.config)
        config = configured(args, config)
        records = run_precomputed(config, args.prediction_dir)
        print(f"Processed {len(records)} NYU samples. Results: {config.outputs.directory / 'precomputed_results.json'}")
    elif args.command == "run-model":
        config = load_config(args.config)
        config = configured(args, config)
        records = run_model(config, args.model_id, args.device, args.limit)
        print(f"Processed {len(records)} NYU samples. Results: {config.outputs.directory / 'model_results.json'}")
    elif args.command == "evaluate-splits":
        base_config = load_config(args.config)
        split_root = args.split_root
        for split in ("development", "validation", "test"):
            split_config = Config(
                dataset=DatasetConfig(
                    name=base_config.dataset.name,
                    rgb_dir=split_root / split / "rgb",
                    depth_dir=split_root / split / "depth",
                    manifest=split_root / split / "manifest.txt",
                    depth_scale=base_config.dataset.depth_scale,
                    max_depth_m=base_config.dataset.max_depth_m,
                    split=split,
                ),
                experiment=base_config.experiment,
                outputs=OutputConfig(directory=(args.output_dir or Path("outputs/nyu_evaluation")) / split),
            )
            records = run_model(split_config, args.model_id, args.device, args.limit)
            print(f"{split}: processed {len(records)} samples")
    elif args.command == "compare-alignment":
        base_config = load_config(args.config)
        split_root = args.split_root
        for split in ("development", "validation", "test"):
            split_config = Config(
                dataset=DatasetConfig(
                    name=base_config.dataset.name,
                    rgb_dir=split_root / split / "rgb",
                    depth_dir=split_root / split / "depth",
                    manifest=split_root / split / "manifest.txt",
                    depth_scale=base_config.dataset.depth_scale,
                    max_depth_m=base_config.dataset.max_depth_m,
                    split=split,
                ),
                experiment=base_config.experiment,
                outputs=OutputConfig(directory=(args.output_dir or Path("outputs/alignment_ablation")) / split),
            )
            payload = run_alignment_ablation(split_config, args.model_id, args.device, args.limit, args.start)
            print(f"{split}: processed {len(payload['records'])} samples")
    elif args.command == "prepare-rating":
        if not args.results_path:
            raise ValueError("--results-path is required for prepare-rating")
        count = export_rating_set(args.results_path, args.rgb_dir, args.depth_dir, args.prediction_dir, args.output_dir or Path("outputs/rating_set"), args.limit or 200, args.seed)
        print(f"Exported {count} blinded rating regions")
    elif args.command == "run-official-metric":
        config = load_config(args.config)
        config = configured(args, config, args.max_depth)
        config = replace(config, experiment=replace(config.experiment, model_name=f"metric_hypersim_{args.encoder}", align_scale_shift=False))
        records = run_official_metric(config, args.official_repo, args.checkpoint, args.encoder, args.max_depth, args.device)
        print(f"Processed {len(records)} samples. Results: {config.outputs.directory / 'official_metric_results.json'}")


if __name__ == "__main__":
    main()
