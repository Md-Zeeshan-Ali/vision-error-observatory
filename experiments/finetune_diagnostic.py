"""Exploratory check after guided_finetune.py: did up-weighting act on the training pixels it targeted?

For seed 0, scores each model on the 795 training images, separately on the up-weighted regions and on
the rest. This was added after seeing the results, to tell a working mechanism from a bug, and isn't
one of the registered tests in docs/FINETUNE_PROTOCOL.md.
"""

import json
from pathlib import Path

import numpy as np
import torch
from guided_finetune import MODEL_ID, ROOT, TARGETS, load_model, load_split, pair_of, predict

from veo.metrics import align_scale_shift
from veo.pipeline import compute_regions
from veo.reports import write_result
from veo.runner import load_sample

OUTPUTS = Path("outputs/finetune")
MODELS = ("pretrained", "uniform_seed0", "veo_boundary_seed0", "random_boundary_seed0", "veo_textureless_seed0", "random_textureless_seed0")


def main() -> None:
    config, ids, _ = load_split(ROOT / "train" / "model_results.json")
    report = {}
    for name in MODELS:
        processor, model = load_model("cuda")
        if name != "pretrained":
            model.load_state_dict(torch.load(OUTPUTS / "runs" / name / "decoder.pt", map_location="cuda"), strict=False)
        sums = {label: {"error": 0.0, "pixels": 0, "failing": 0} for label in (*TARGETS, "other")}
        for sample_id in ids:
            rgb, target, _, valid = load_sample(pair_of(config, sample_id), config)
            prediction = predict(model, processor, rgb, "cuda")
            aligned, _, _ = align_scale_shift(prediction, target, valid, config.experiment.prediction_space, config.experiment.max_prediction_m)
            computed = compute_regions(target, prediction, valid, config.experiment)
            relative = np.abs(aligned - target) / np.where(target > 0, target, 1.0)
            with np.load(OUTPUTS / "masks" / f"{sample_id}.npz") as stored:
                masks = {label: stored[label] & computed["valid_mask"] for label in TARGETS}
            masks["other"] = computed["valid_mask"] & ~masks["boundary_mixing"] & ~masks["textureless"]
            for label, mask in masks.items():
                sums[label]["error"] += float(relative[mask].sum())
                sums[label]["pixels"] += int(mask.sum())
                sums[label]["failing"] += int(computed["failing"][mask].sum())
        report[name] = {label: {"abs_rel": row["error"] / row["pixels"], "failing_share": row["failing"] / row["pixels"], "pixels": row["pixels"]} for label, row in sums.items()}
        print(name, {label: round(row["abs_rel"], 4) for label, row in report[name].items()}, flush=True)
        del model
        torch.cuda.empty_cache()
    write_result(Path("results/finetune/training_pixel_check.json"), {"model": MODEL_ID, "models": report, "note": json.dumps("exploratory; added after seeing the results")})


if __name__ == "__main__":
    main()
