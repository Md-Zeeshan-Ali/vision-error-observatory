"""Oracle boundary filtering: bilateral smoothing inside boundary_mixing regions.

The target pixels are chosen using the ground-truth error map, so this is an upper-bound
illustration that cannot run at inference time. There is no control condition yet.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from veo.data import make_valid_mask
from veo.metrics import align_scale_shift, compute_error_map, compute_metrics
from veo.reports import write_result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-path", type=Path, default=Path("outputs/nyu_final/test/model_results.json"))
    parser.add_argument("--prediction-dir", type=Path, default=Path("outputs/alignment_ablation_full/test/predictions"))
    parser.add_argument("--depth-dir", type=Path, default=Path("data/nyu/splits/test/depth"), help="error target: use sensor depth under protocol v2")
    parser.add_argument("--crop", default=None, help="evaluation crop, e.g. eigen")
    parser.add_argument("--max-depth", type=float, default=10.0)
    parser.add_argument("--percentile", type=float, default=85.0)
    parser.add_argument("--min-pixels", type=int, default=25)
    parser.add_argument("--output", type=Path, default=Path("results/nyu_test/oracle_boundary_filter.json"))
    args = parser.parse_args()

    records = json.loads(args.results_path.read_text(encoding="utf-8"))["records"]
    filter_params = {"d": 5, "sigmaColor": 0.10, "sigmaSpace": 3}
    samples = []
    for record in records:
        # v2 regions list every supported label; v1 result files only have top_cause
        boxes = [region["bbox"] for region in record["regions"] if "boundary_mixing" in region.get("labels", [region["top_cause"]])]
        if not boxes:
            continue
        sample_id = record["sample_id"]
        target = np.load(args.depth_dir / f"{sample_id}.npy")
        valid = make_valid_mask(target, args.max_depth, args.crop)
        aligned, _, _ = align_scale_shift(np.load(args.prediction_dir / f"{sample_id}.npy"), target, valid, "inverse_depth", args.max_depth)
        error = compute_error_map(aligned, target, valid)
        threshold = float(np.percentile(error[valid], args.percentile))
        mask = np.zeros_like(valid)
        for x, y, w, h in boxes:
            mask[y:y + h, x:x + w] |= error[y:y + h, x:x + w] > threshold  # oracle: uses ground-truth error
        mask &= valid
        if mask.sum() < args.min_pixels:
            continue
        refined = aligned.copy()
        refined[mask] = cv2.bilateralFilter(aligned.astype(np.float32), **filter_params)[mask]
        diff_before, diff_after = aligned[mask] - target[mask], refined[mask] - target[mask]
        samples.append({
            "sample_id": sample_id,
            "pixels": int(mask.sum()),
            "region_rmse_before": float(np.sqrt(np.mean(diff_before ** 2))),
            "region_rmse_after": float(np.sqrt(np.mean(diff_after ** 2))),
            "region_absrel_before": float(np.mean(np.abs(diff_before) / target[mask])),
            "region_absrel_after": float(np.mean(np.abs(diff_after) / target[mask])),
            "global_rmse_before": compute_metrics(aligned, target, valid)["rmse"],
            "global_rmse_after": compute_metrics(refined, target, valid)["rmse"],
        })

    def mean(key: str) -> float:
        return float(np.mean([sample[key] for sample in samples]))

    summary = {key: mean(key) for key in samples[0] if key not in ("sample_id", "pixels")} if samples else {}
    payload = {
        "oracle": True,
        "target": str(args.depth_dir),
        "selection": f"pixels in boxes of regions labelled boundary_mixing with ground-truth relative error above the image's {args.percentile:g}th percentile",
        "filter": {"method": "cv2.bilateralFilter", **filter_params},
        "evaluated_scenes": len(samples),
        "mean": summary,
        "samples": samples,
    }
    write_result(args.output, payload, inputs={"veo_results": args.results_path})
    for key, value in summary.items():
        print(f"{key:<22} {value:.4f}")
    print(f"Saved {args.output} ({len(samples)} scenes)")


if __name__ == "__main__":
    main()
