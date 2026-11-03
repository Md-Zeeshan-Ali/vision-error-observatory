from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DatasetConfig:
    name: str = "hypersim"
    rgb_dir: Path = Path("data/hypersim/rgb")
    depth_dir: Path = Path("data/hypersim/depth")
    manifest: Path | None = None  # None: pair files by stem across rgb_dir and depth_dir
    depth_scale: float = 1000.0
    max_depth_m: float = 10.0
    split: str = "official_test"
    # Sensor depth with holes (0 = not measured). When set, errors are measured on it only and
    # depth_dir (dense, in-painted) is used solely for geometry cues such as depth edges.
    raw_depth_dir: Path | None = None
    eval_crop: str | None = None  # "eigen": standard NYU evaluation crop for 480x640 frames


@dataclass(frozen=True)
class ExperimentConfig:
    model_name: str = "precomputed"
    align_scale_shift: bool = False
    prediction_space: str = "depth"
    primary_protocol: str = "inverse_depth_scale_shift"
    # Aligned predictions are capped at this depth (metres), as in standard NYU evaluation; None = no cap
    max_prediction_m: float | None = None
    # Failure regions: "delta" marks pixels with max(pred/gt, gt/pred) > delta_threshold;
    # "percentile" marks the image's top (100 - error_percentile)% relative errors (v1 behaviour)
    region_mode: str = "delta"
    delta_threshold: float = 1.25
    error_percentile: float = 85.0
    min_region_area: int = 25
    opening_radius: int = 0  # >0 removes speckle but also erases thin failure bands (edges, thin objects)
    # Depth used for geometry features: "ground_truth" (audit) or "prediction" (no ground truth needed)
    geometry_source: str = "ground_truth"
    include_reflection: bool = True
    label_model: str | None = None  # fitted label model (docs/LABEL_MODEL_V3.md); None = the hand-set v2 rules
    texture_grad_scale: float = 8.0
    edge_log_threshold: float = 0.05
    edge_radius: int = 3
    boundary_fraction_scale: float = 0.5
    thin_kernel: int = 9
    thin_relative_threshold: float = 0.1
    thin_fraction_scale: float = 0.3
    thin_image_delta: int = 25
    highlight_value: int = 200
    highlight_saturation: int = 50
    highlight_local_delta: int = 25
    reflection_scale: float = 5.0
    dark_level: float = 50.0
    min_score: float = 0.4
    ambiguity_margin: float = 0.10
    label_threshold: float = 0.5
    random_seed: int = 42


@dataclass(frozen=True)
class OutputConfig:
    directory: Path = Path("outputs")


@dataclass(frozen=True)
class Config:
    dataset: DatasetConfig = DatasetConfig()
    experiment: ExperimentConfig = ExperimentConfig()
    outputs: OutputConfig = OutputConfig()


def _build(cls, values: dict, section: str):
    """Instantiate a config dataclass, converting types and rejecting unknown keys."""
    known = {field.name: field for field in fields(cls)}
    unknown = sorted(set(values) - set(known))
    if unknown:
        raise ValueError(f"Unknown {section} config keys: {unknown}")
    converted = {}
    for name, value in values.items():
        default = getattr(cls, name)
        if value is None:
            converted[name] = None
        elif isinstance(default, Path) or name.endswith(("_dir", "manifest", "directory")):
            converted[name] = Path(value)
        elif isinstance(default, bool):
            converted[name] = bool(value)
        elif isinstance(default, int):
            converted[name] = int(value)
        elif isinstance(default, float):
            converted[name] = float(value)
        else:
            converted[name] = value
    return cls(**converted)


def config_from_dict(values: dict) -> Config:
    """Rebuild a Config from its dict form, e.g. the ``config`` block stored in a results file."""
    return Config(
        dataset=_build(DatasetConfig, values.get("dataset", {}), "dataset"),
        experiment=_build(ExperimentConfig, values.get("experiment", {}), "experiment"),
        outputs=_build(OutputConfig, values.get("outputs", {}), "outputs"),
    )


def load_config(path: Path) -> Config:
    values = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Config(
        dataset=_build(DatasetConfig, values.get("dataset", {}), "dataset"),
        experiment=_build(ExperimentConfig, values.get("experiment", {}), "experiment"),
        outputs=_build(OutputConfig, values.get("outputs", {}), "outputs"),
    )
