"""Phase 6: paired model comparisons on one split, using each model's results file.

    python experiments/compare_models.py --root outputs/nyu_test --output results/nyu_test/model_comparison.json
"""

import argparse
import json
from pathlib import Path

from veo.model_comparison import compare, compare_composition
from veo.reports import write_result

CONTRASTS = {
    "size_small_vs_base": ("depth-anything-v2-small-hf", "depth-anything-v2-base-hf"),
    "size_base_vs_large": ("depth-anything-v2-base-hf", "depth-anything-v2-large-hf"),
    "backbone_hybrid_vs_large": ("dpt-hybrid-midas", "dpt-large"),
    "family_dav2_small_vs_dpt_hybrid": ("depth-anything-v2-small-hf", "dpt-hybrid-midas"),
    "family_dav2_small_vs_marigold": ("depth-anything-v2-small-hf", "marigold-depth-v1-1"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="folder with <model>/model_results.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scan-units", action="store_true", help="use DIODE's scans, the second part of each image id, instead of its 3 scenes")
    args = parser.parse_args()

    records: dict[str, list[dict]] = {}
    scene_of: dict[str, str] = {}
    for results in sorted(args.root.glob("*/model_results.json")):
        payload = json.loads(results.read_text(encoding="utf-8"))
        records[results.parent.name] = payload["records"]
        metadata = Path(payload["config"]["dataset"]["depth_dir"]).parent / "metadata.json"
        if metadata.exists():
            scene_of.update({row["sample_id"]: row["sample_id"].split("_")[1] if args.scan_units else row["scene"] for row in json.loads(metadata.read_text(encoding="utf-8"))})
    contrasts, skipped = {}, []
    for name, (first, second) in CONTRASTS.items():
        if first in records and second in records:
            contrasts[name] = {
                "first": first, "second": second, **compare(records[first], records[second], scene_of),
                "composition": compare_composition(records[first], records[second], scene_of),
            }
        else:
            skipped.append(name)
    write_result(args.output, {"models": sorted(records), "contrasts": contrasts, "skipped": skipped})
    for name, contrast in contrasts.items():
        reliable = [outcome for outcome, row in contrast["outcomes"].items() if row.get("reliable")]
        print(f"{name}: {contrast['scenes']} scenes; reliable differences: {reliable or 'none'}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
