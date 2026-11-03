"""Phase 6 external check on DIODE indoor validation frames (another sensor, frames VEO has never seen).

DIODE has no semantic labels, so this reports model metrics, failure extent and label distributions,
not whether the labels are correct.
"""

import argparse
from dataclasses import replace
from pathlib import Path

from veo.config import DatasetConfig, OutputConfig, load_config
from veo.diode import ensure_indoor
from veo.runner import run_model
from veo.summary import write_summary_markdown

MODELS = (
    "depth-anything/Depth-Anything-V2-Small-hf",
    "depth-anything/Depth-Anything-V2-Base-hf",
    "depth-anything/Depth-Anything-V2-Large-hf",
    "Intel/dpt-hybrid-midas",
    "Intel/dpt-large",
    "Intel/zoedepth-nyu-kitti",
    "apple/DepthPro-hf",
    "prs-eth/marigold-depth-v1-1",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/nyu.yaml"), help="VEO rules (experiment section)")
    parser.add_argument("--data-dir", type=Path, default=Path("data/diode"))
    parser.add_argument("--outputs", type=Path, default=Path("outputs/diode"))
    parser.add_argument("--results", type=Path, default=Path("results/diode"))
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    split = ensure_indoor(args.data_dir)
    base = load_config(args.config)
    dataset = DatasetConfig(
        name="diode_indoor", rgb_dir=split / "rgb", depth_dir=split / "depth", manifest=split / "manifest.txt",
        depth_scale=1.0, max_depth_m=10.0, split="val_indoors",
    )
    for model_id in args.models:
        slug = model_id.split("/")[-1].lower()
        config = replace(base, dataset=dataset, outputs=OutputConfig(directory=args.outputs / slug))
        print(f"{model_id}", flush=True)
        run_model(config, model_id, args.device, save_panels=False, save_predictions=False)
        target = args.results / slug
        target.mkdir(parents=True, exist_ok=True)
        (target / "summary.json").write_bytes((config.outputs.directory / "summary.json").read_bytes())
    print(f"Done: {write_summary_markdown(args.results)}")


if __name__ == "__main__":
    main()
