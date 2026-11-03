"""Phase 7: does up-weighting what VEO diagnosed improve a fine-tuned model? (docs/FINETUNE_PROTOCOL.md)

Fine-tunes the decoder of Depth Anything V2 Small on NYU train under five conditions and three seeds,
then scores every model on the official test split. Runs resume from their last finished epoch.

    python experiments/guided_finetune.py --pilot          # train split only: compares learning rates
    python experiments/guided_finetune.py                  # all runs, then the analysis
    python experiments/guided_finetune.py --sweep          # amendment: weaker weights and a hard-example baseline
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from veo.checkpoint import JsonlLog, atomic_write_text, resumed
from veo.config import Config, config_from_dict
from veo.data import RGBDepthPair
from veo.finetune import (
    LOSS_SHARE,
    contrast,
    derangement,
    image_outcomes,
    scale_shift_loss,
    target_weight,
)
from veo.interventions import holm
from veo.metrics import compute_metrics
from veo.pipeline import compute_regions
from veo.provenance import config_hash
from veo.reports import write_result
from veo.runner import load_sample
from veo.silver import class_ids, silver_maps

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
ROOT = Path("outputs/nyu_test/depth-anything-v2-small-hf")
TARGETS = ("boundary_mixing", "textureless")
ANY_FAILURE = "any_failure"  # every failure region, whatever its label: the diagnosis-free baseline
SEEDS = (0, 1, 2)
BATCH_SIZE = 4
LEARNING_RATE = 3e-5
EPOCHS = 4


@dataclass(frozen=True)
class Condition:
    name: str
    target: str | None = None  # which regions are up-weighted
    shuffled: bool = False  # True: masks come from other images, so they match in size but not place
    share: float = LOSS_SHARE  # share of the training loss the up-weighted pixels carry


CONDITIONS = (
    Condition("uniform"),
    Condition("veo_boundary", "boundary_mixing"),
    Condition("random_boundary", "boundary_mixing", True),
    Condition("veo_textureless", "textureless"),
    Condition("random_textureless", "textureless", True),
)

# Amendment of 2026-11-03 (docs/FINETUNE_PROTOCOL.md): weaker weights, and up-weighting every failure region
SWEEP_SHARES = (0.05, 0.10)
SWEEP_CONDITIONS = (
    *(Condition(f"{kind}_{short}_share{round(share * 100):02d}", target, kind == "random", share)
      for share in SWEEP_SHARES for short, target in (("boundary", "boundary_mixing"), ("textureless", "textureless")) for kind in ("veo", "random")),
    Condition("failure_weighted", ANY_FAILURE),
)


def load_split(results_path: Path) -> tuple[Config, list[str], dict[str, str]]:
    config = config_from_dict(json.loads(results_path.read_text(encoding="utf-8"))["config"])
    split = config.dataset.depth_dir.parent
    metadata = json.loads((split / "metadata.json").read_text(encoding="utf-8"))
    return config, sorted(row["sample_id"] for row in metadata), {row["sample_id"]: row["scene"] for row in metadata}


def pair_of(config: Config, sample_id: str) -> RGBDepthPair:
    split = config.dataset.depth_dir.parent
    return RGBDepthPair(sample_id, split / "rgb" / f"{sample_id}.png", split / "depth" / f"{sample_id}.npy")


def build_masks(config: Config, ids: list[str], folder: Path) -> dict[str, float]:
    """Save each training image's targeted regions from the pretrained model; returns each target's prevalence."""
    folder.mkdir(parents=True, exist_ok=True)
    payload = json.loads((ROOT / "train" / "model_results.json").read_text(encoding="utf-8"))
    records = {record["sample_id"]: record for record in payload["records"]}
    keys = (*TARGETS, ANY_FAILURE)
    pixels = dict.fromkeys(keys, 0)
    valid_pixels = 0
    for sample_id in ids:
        path = folder / f"{sample_id}.npz"
        _, target, _, valid = load_sample(pair_of(config, sample_id), config)
        complete = False
        if path.exists():
            with np.load(path) as stored:
                complete = all(key in stored.files for key in keys)
        if not complete:
            computed = compute_regions(target, np.load(ROOT / "train" / "predictions" / f"{sample_id}.npy"), valid, config.experiment)
            masks = {region["region_id"]: region["mask"] for region in computed["regions"]}
            combined = {label: np.zeros(target.shape, dtype=bool) for label in keys}
            for region in records[sample_id]["regions"]:
                combined[ANY_FAILURE] |= masks[region["region_id"]]
                if region["top_cause"] in combined:
                    combined[region["top_cause"]] |= masks[region["region_id"]]
            temporary = folder / f"{sample_id}.tmp.npz"
            np.savez_compressed(temporary, **combined)
            temporary.replace(path)
        with np.load(path) as stored:
            for label in keys:
                pixels[label] += int(stored[label].sum())
        valid_pixels += int(valid.sum())
    return {label: pixels[label] / valid_pixels for label in keys}


def load_model(device: str):
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID).to(device)
    model.backbone.requires_grad_(False)
    return processor, model


def predict(model, processor, rgb: np.ndarray, device: str) -> np.ndarray:
    import torch

    model.eval()
    inputs = processor(images=rgb, return_tensors="pt").to(device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        output = model(**inputs)
    result = processor.post_process_depth_estimation(output, target_sizes=[rgb.shape[:2]])[0]["predicted_depth"]
    return result.float().cpu().numpy().astype(np.float32)


def train_run(condition: Condition, seed: int, config: Config, ids: list[str], mask_dir: Path, prevalence: dict[str, float], folder: Path,
              learning_rate: float = LEARNING_RATE, epochs: int = EPOCHS, device: str = "cuda", on_epoch=None) -> Path:
    """Train one condition and seed, saving after each epoch; returns the final decoder weights."""
    import torch
    from torch.nn.functional import interpolate

    final = folder / "decoder.pt"
    if final.exists():
        return final
    folder.mkdir(parents=True, exist_ok=True)
    state_path = folder / "training_state.pt"
    settings = config_hash({"condition": condition.name, "seed": seed, "lr": learning_rate, "epochs": epochs, "ids": ids, "batch": BATCH_SIZE})
    torch.manual_seed(seed)
    processor, model = load_model(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.01)
    steps_per_epoch = int(np.ceil(len(ids) / BATCH_SIZE))
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * steps_per_epoch)
    scaler = torch.amp.GradScaler("cuda")
    first_epoch = 0
    if state_path.exists():
        state = torch.load(state_path, map_location=device, weights_only=False)
        if state["settings"] == settings:
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            schedule.load_state_dict(state["schedule"])
            scaler.load_state_dict(state["scaler"])
            first_epoch = state["epoch"] + 1
    source = derangement(len(ids), seed) if condition.shuffled else np.arange(len(ids))
    weight = target_weight(prevalence[condition.target], condition.share) if condition.target else None
    for epoch in range(first_epoch, epochs):
        # The order and flips depend only on seed and epoch, so paired conditions see the same batches
        rng = np.random.default_rng([seed, epoch])
        order = rng.permutation(len(ids))
        flips = rng.random(len(ids)) < 0.5
        model.train()
        model.backbone.eval()
        total = 0.0
        for start in range(0, len(ids), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            images, targets, valid_masks, weights = [], [], [], []
            for index in batch:
                rgb, target, _, valid = load_sample(pair_of(config, ids[index]), config)
                pixel_weights = np.ones(target.shape, dtype=np.float32)
                if condition.target:
                    with np.load(mask_dir / f"{ids[source[index]]}.npz") as stored:
                        pixel_weights[stored[condition.target]] = weight
                if flips[index]:
                    rgb, target, valid, pixel_weights = rgb[:, ::-1].copy(), target[:, ::-1].copy(), valid[:, ::-1].copy(), pixel_weights[:, ::-1].copy()
                images.append(rgb)
                targets.append(1.0 / np.where(valid, target, 1.0))
                valid_masks.append(valid)
                weights.append(pixel_weights)
            inputs = processor(images=images, return_tensors="pt").to(device)
            disparity = torch.as_tensor(np.stack(targets), device=device)
            mask = torch.as_tensor(np.stack(valid_masks), device=device)
            pixel_weight = torch.as_tensor(np.stack(weights), device=device) if condition.target else None
            with torch.autocast("cuda", dtype=torch.float16):
                output = model(**inputs).predicted_depth
            output = interpolate(output.float().unsqueeze(1), size=disparity.shape[-2:], mode="bicubic", align_corners=True).squeeze(1)
            loss = scale_shift_loss(output, disparity, mask, pixel_weight)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            schedule.step()
            total += float(loss.detach()) * len(batch)
        print(f"  {folder.name} epoch {epoch + 1}/{epochs}: loss {total / len(ids):.4f}", flush=True)
        temporary = state_path.with_suffix(".tmp")
        torch.save({"settings": settings, "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "schedule": schedule.state_dict(), "scaler": scaler.state_dict()}, temporary)
        temporary.replace(state_path)
        if on_epoch:
            on_epoch(epoch, model, processor)
    decoder = {name: value for name, value in model.state_dict().items() if not name.startswith("backbone.")}
    temporary = final.with_suffix(".tmp")
    torch.save(decoder, temporary)
    temporary.replace(final)
    state_path.unlink(missing_ok=True)
    return final


def score(model, processor, config: Config, ids: list[str], groups: dict, sample_id_log: JsonlLog | None = None, device: str = "cuda") -> dict[str, dict]:
    """Per-image outcomes with the same alignment and failure rule as every other result."""
    finished = sample_id_log.load() if sample_id_log else {}
    results = dict(finished)
    split = config.dataset.depth_dir.parent
    for sample_id in resumed(ids, set(finished)):
        pair = pair_of(config, sample_id)
        rgb, target, dense, valid = load_sample(pair, config)
        computed = compute_regions(target, predict(model, processor, rgb, device), valid, config.experiment)
        instances_path = split / "instances" / f"{sample_id}.npy"
        maps = silver_maps(np.load(split / "labels" / f"{sample_id}.npy"), np.load(instances_path) if instances_path.exists() else None, dense, groups)
        metrics = compute_metrics(computed["aligned"], target, computed["valid_mask"])
        results[sample_id] = image_outcomes(computed["aligned"], target, computed["valid_mask"], computed["failing"], metrics, maps)
        if sample_id_log:
            sample_id_log.append(sample_id, results[sample_id])
    return results


def groups_of(config: Config) -> dict:
    return class_ids(json.loads((config.dataset.depth_dir.parent / "label_names.json").read_text(encoding="utf-8")))


def pilot(args) -> None:
    """Uniform condition on 90% of train scenes, scored on the other 10%. No test image is read."""
    config, ids, scene_of = load_split(ROOT / "train" / "model_results.json")
    scenes = sorted(set(scene_of.values()))
    held = set(np.random.default_rng(7).permutation(scenes)[: len(scenes) // 10].tolist())
    fitting = [sample_id for sample_id in ids if scene_of[sample_id] not in held]
    holdout = [sample_id for sample_id in ids if scene_of[sample_id] in held]
    groups = groups_of(config)
    mask_dir = args.outputs / "masks"
    prevalence = build_masks(config, ids, mask_dir)
    print(f"prevalence {prevalence}; weights { {label: round(target_weight(value), 1) for label, value in prevalence.items()} }", flush=True)
    processor, model = load_model(args.device)

    def report(name: str, scoring_model, scoring_processor) -> dict:
        outcomes = score(scoring_model, scoring_processor, config, holdout, groups, device=args.device)
        summary = {key: float(np.mean([row[key] for row in outcomes.values() if row[key] is not None])) for key in ("abs_rel", "delta1", "abs_rel_boundary_mixing", "abs_rel_textureless")}
        print(f"  {name}: " + "  ".join(f"{key} {value:.4f}" for key, value in summary.items()), flush=True)
        return summary

    print(f"{len(fitting)} fitting images, {len(holdout)} held-out images in {len(held)} scenes", flush=True)
    report_rows = {"pretrained": report("pretrained", model, processor)}
    del model
    for learning_rate in args.pilot_rates:
        rows = {}

        def on_epoch(epoch, trained, trained_processor, name=learning_rate, store=rows):
            store[f"epoch_{epoch + 1}"] = report(f"lr {name} epoch {epoch + 1}", trained, trained_processor)

        train_run(CONDITIONS[0], 0, config, fitting, mask_dir, prevalence, args.outputs / "pilot" / f"lr_{learning_rate}", learning_rate, args.pilot_epochs, args.device, on_epoch)
        report_rows[f"lr_{learning_rate}"] = rows
    atomic_write_text(args.outputs / "pilot" / "pilot.json", json.dumps(report_rows, indent=2))


def analyse(runs: dict[str, dict[int, dict[str, dict]]], baseline: dict[str, dict], scene_of: dict[str, str]) -> dict:
    """The registered contrasts: each is first minus second on one outcome."""
    outcome = {"boundary_mixing": "abs_rel_boundary_mixing", "textureless": "abs_rel_textureless"}
    pretrained = dict.fromkeys(SEEDS, baseline)
    primary = {
        "boundary": contrast(runs["veo_boundary"], runs["random_boundary"], outcome["boundary_mixing"], scene_of),
        "textureless": contrast(runs["veo_textureless"], runs["random_textureless"], outcome["textureless"], scene_of),
    }
    for name, value in holm({name: row["p_value"] for name, row in primary.items()}).items():
        row = primary[name]
        row["p_holm"] = value
        row["supported"] = bool(value < 0.05 and row["upper"] < 0 and row["seeds_agree"])
    secondary = {
        "boundary_vs_uniform": contrast(runs["veo_boundary"], runs["uniform"], outcome["boundary_mixing"], scene_of),
        "textureless_vs_uniform": contrast(runs["veo_textureless"], runs["uniform"], outcome["textureless"], scene_of),
        "boundary_overall_vs_random": contrast(runs["veo_boundary"], runs["random_boundary"], "abs_rel", scene_of),
        "textureless_overall_vs_random": contrast(runs["veo_textureless"], runs["random_textureless"], "abs_rel", scene_of),
        "specificity_on_boundary_pixels": contrast(runs["veo_boundary"], runs["veo_textureless"], outcome["boundary_mixing"], scene_of),
        "specificity_on_wall_pixels": contrast(runs["veo_textureless"], runs["veo_boundary"], outcome["textureless"], scene_of),
    }
    for name, value in holm({name: row["p_value"] for name, row in secondary.items()}).items():
        secondary[name]["p_holm"] = value
    context = {"uniform_vs_pretrained": contrast(runs["uniform"], pretrained, "abs_rel", scene_of)}
    means = {
        name: {key: float(np.mean([np.mean([row[key] for row in records.values() if row[key] is not None]) for records in seeds.values()]))
               for key in ("abs_rel", "delta1", "failing_pixels", *outcome.values())}
        for name, seeds in {**runs, "pretrained": {0: baseline}}.items()
    }
    return {"primary": primary, "secondary": secondary, "context": context, "condition_means": means}


def analyse_sweep(runs: dict[str, dict[int, dict[str, dict]]], scene_of: dict[str, str]) -> dict:
    """The amendment's contrasts, each first minus second, with the claim rule of the primary test."""
    outcome = {"boundary": "abs_rel_boundary_mixing", "textureless": "abs_rel_textureless"}

    def judged(family: dict) -> dict:
        for name, value in holm({name: row["p_value"] for name, row in family.items()}).items():
            row = family[name]
            row["p_holm"] = value
            row["supported"] = bool(value < 0.05 and row["upper"] < 0 and row["seeds_agree"])
        return family

    sweep = judged({
        f"{short}_share{round(share * 100):02d}": contrast(runs[f"veo_{short}_share{round(share * 100):02d}"], runs[f"random_{short}_share{round(share * 100):02d}"], outcome[short], scene_of)
        for share in SWEEP_SHARES for short in outcome
    })
    hard_example = judged({
        "boundary_vs_failure_weighted": contrast(runs["veo_boundary"], runs["failure_weighted"], outcome["boundary"], scene_of),
        "textureless_vs_failure_weighted": contrast(runs["veo_textureless"], runs["failure_weighted"], outcome["textureless"], scene_of),
    })
    descriptive = {
        f"failure_weighted_vs_uniform_{name}": contrast(runs["failure_weighted"], runs["uniform"], key, scene_of)
        for name, key in (("boundary", outcome["boundary"]), ("textureless", outcome["textureless"]), ("overall", "abs_rel"))
    }
    means = {
        name: {key: float(np.mean([np.mean([row[key] for row in records.values() if row[key] is not None]) for records in seeds.values()]))
               for key in ("abs_rel", "delta1", "failing_pixels", *outcome.values())}
        for name, seeds in runs.items()
    }
    return {"sweep": sweep, "hard_example": hard_example, "descriptive": descriptive, "condition_means": means}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outputs", type=Path, default=Path("outputs/finetune"))
    parser.add_argument("--output", type=Path, default=Path("results/finetune/guided_finetune.json"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--pilot-rates", type=float, nargs="+", default=[1e-5, 3e-5])
    parser.add_argument("--pilot-epochs", type=int, default=6)
    parser.add_argument("--sweep", action="store_true", help="run the amendment's conditions and write --sweep-output")
    parser.add_argument("--sweep-output", type=Path, default=Path("results/finetune/weight_sweep.json"))
    args = parser.parse_args()
    if args.pilot:
        pilot(args)
        return

    import torch

    train_config, train_ids, _ = load_split(ROOT / "train" / "model_results.json")
    test_config, test_ids, test_scenes = load_split(ROOT / "model_results.json")
    groups = groups_of(test_config)
    mask_dir = args.outputs / "masks"
    prevalence = build_masks(train_config, train_ids, mask_dir)
    print(f"prevalence {prevalence}", flush=True)

    def scored(name: str, weights: Path | None) -> dict[str, dict]:
        log = JsonlLog(args.outputs / "scores" / f"{name}.jsonl", config_hash({"name": name, "ids": test_ids}))
        if len(log.load()) == len(test_ids):
            return log.load()
        processor, model = load_model(args.device)
        if weights is not None:
            model.load_state_dict(torch.load(weights, map_location=args.device), strict=False)
        result = score(model, processor, test_config, test_ids, groups, log, args.device)
        del model
        torch.cuda.empty_cache()
        return result

    (args.outputs / "scores").mkdir(parents=True, exist_ok=True)
    baseline = scored("pretrained", None)
    conditions = (*CONDITIONS, *SWEEP_CONDITIONS) if args.sweep else CONDITIONS
    runs: dict[str, dict[int, dict[str, dict]]] = {condition.name: {} for condition in conditions}
    for seed in SEEDS:  # one full set of conditions per seed, so an early stop leaves complete pairs
        for condition in conditions:
            name = f"{condition.name}_seed{seed}"
            print(name, flush=True)
            weights = train_run(condition, seed, train_config, train_ids, mask_dir, prevalence, args.outputs / "runs" / name, device=args.device)
            torch.cuda.empty_cache()
            runs[condition.name][seed] = scored(name, weights)
    if args.sweep:
        sweep_report = {
            "model": MODEL_ID, "seeds": list(SEEDS), "learning_rate": LEARNING_RATE, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "prevalence": prevalence,
            "weights": {condition.name: target_weight(prevalence[condition.target], condition.share) for condition in conditions if condition.target},
            "conditions": [condition.name for condition in conditions], **analyse_sweep(runs, test_scenes),
        }
        write_result(args.sweep_output, sweep_report)
        for family in ("sweep", "hard_example"):
            for name, row in sweep_report[family].items():
                print(f"{name}: {row['mean']:+.5f} [{row['lower']:+.5f}, {row['upper']:+.5f}] p_holm {row['p_holm']:.4f} seeds {row['per_seed']} supported {row['supported']}")
        print(f"Saved {args.sweep_output}")
        return
    report = {
        "model": MODEL_ID, "seeds": list(SEEDS), "learning_rate": LEARNING_RATE, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        "prevalence": prevalence, "weights": {label: target_weight(value) for label, value in prevalence.items()},
        "conditions": [condition.name for condition in CONDITIONS], **analyse(runs, baseline, test_scenes),
    }
    write_result(args.output, report)
    for name, row in report["primary"].items():
        print(f"{name}: {row['mean']:+.5f} [{row['lower']:+.5f}, {row['upper']:+.5f}] p_holm {row['p_holm']:.4f} seeds {row['per_seed']} supported {row['supported']}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
