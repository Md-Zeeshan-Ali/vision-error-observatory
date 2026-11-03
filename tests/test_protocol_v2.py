"""Protocol v2: absolute failure rule, evaluation crop, sensor-depth target, official split, strict config."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from veo.config import Config, DatasetConfig, ExperimentConfig, OutputConfig, load_config
from veo.data import RGBDepthPair, make_valid_mask
from veo.fixtures import (
    SyntheticDisparityPredictor,
    _scene,
    write_fixture_mat,
    write_fixture_splits,
)
from veo.metrics import compute_ratio_map
from veo.nyu_dataset import extract_split, official_splits
from veo.pipeline import analyze_sample
from veo.runner import load_sample


def test_ratio_map_matches_delta_definition():
    target = np.array([[1.0, 2.0], [4.0, 0.0]], dtype=np.float32)
    prediction = np.array([[1.25, 1.0], [5.0, 3.0]], dtype=np.float32)
    ratio = compute_ratio_map(prediction, target, np.ones_like(target, dtype=bool))
    assert np.allclose(ratio, [[1.25, 2.0], [1.25, 1.0]])  # invalid (zero) target -> 1


def test_delta_regions_are_absolute_but_percentile_regions_are_not():
    rng = np.random.default_rng(3)
    rgb, depth = _scene(rng, 96, 128)
    near_perfect = depth * (1.0 + rng.normal(0, 0.01, depth.shape)).astype(np.float32)
    valid = np.ones_like(depth, dtype=bool)
    delta = analyze_sample(rgb, depth, near_perfect, valid, ExperimentConfig(region_mode="delta"))
    percentile = analyze_sample(rgb, depth, near_perfect, valid, ExperimentConfig(region_mode="percentile"))
    assert delta["num_regions"] == 0 and delta["failure_pixel_fraction"] == 0.0
    assert percentile["failure_pixel_fraction"] > 0.1  # always flags ~15% of pixels, however good the prediction


def test_aligned_depth_is_capped_when_the_disparity_fit_goes_negative():
    from veo.metrics import align_scale_shift

    target = np.linspace(1.0, 9.0, 400, dtype=np.float32).reshape(20, 20)
    prediction = 1.0 / target
    prediction[0, :5] = -2.0  # pixels the fit maps below zero disparity
    valid = np.ones_like(target, dtype=bool)
    uncapped, _, _ = align_scale_shift(prediction, target, valid, "inverse_depth")
    capped, _, _ = align_scale_shift(prediction, target, valid, "inverse_depth", max_depth=10.0)
    assert uncapped.max() > 10.0
    assert capped.max() <= 10.0 + 1e-4
    assert np.allclose(capped[5:], uncapped[5:])  # ordinary pixels are unchanged


def test_eigen_crop():
    depth = np.ones((480, 640), dtype=np.float32)
    mask = make_valid_mask(depth, crop="eigen")
    assert mask.sum() == (471 - 45) * (601 - 41)
    assert not mask[:45].any() and not mask[:, 601:].any()
    with pytest.raises(ValueError, match="480x640"):
        make_valid_mask(np.ones((48, 64)), crop="eigen")
    with pytest.raises(ValueError, match="Unknown"):
        make_valid_mask(depth, crop="square")


def test_unknown_config_keys_are_rejected(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("experiment:\n  delta_treshold: 1.1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="delta_treshold"):
        load_config(path)


def test_repository_configs_load():
    for name in ("nyu.yaml", "hypersim.yaml"):
        load_config(Path(__file__).resolve().parents[1] / "configs" / name)


@pytest.fixture(scope="module")
def fixture_split(tmp_path_factory):
    root = tmp_path_factory.mktemp("nyu")
    mat = write_fixture_mat(root / "nyu.mat", scenes=6)
    splits = official_splits(mat, write_fixture_splits(root / "splits.mat", scenes=6, test_scenes=2))
    extract_split(mat, splits["test"], root / "test")
    return root, splits


def test_official_split_is_scene_disjoint_and_one_based(fixture_split):
    _, splits = fixture_split
    assert [record["index"] for record in splits["test"]] == [8, 9, 10, 11]
    assert len(splits["train"]) == 8
    assert not {r["scene"] for r in splits["train"]} & {r["scene"] for r in splits["test"]}


def test_official_split_rejects_shared_scenes(tmp_path):
    from scipy.io import savemat

    mat = write_fixture_mat(tmp_path / "nyu.mat", scenes=3)
    savemat(tmp_path / "splits.mat", {"trainNdxs": np.array([[1.0], [3.0]]), "testNdxs": np.array([[2.0]])})
    with pytest.raises(ValueError, match="shares scenes"):
        official_splits(mat, tmp_path / "splits.mat")


def test_extract_split_writes_sensor_depth_and_labels(fixture_split):
    root, _ = fixture_split
    split = root / "test"
    for folder in ("rgb", "depth", "raw_depth", "labels"):
        assert len(list((split / folder).iterdir())) == 4
    assert json.loads((split / "label_names.json").read_text(encoding="utf-8")) == ["wall", "object"]
    raw = np.load(split / "raw_depth" / "scene_0009.npy")
    assert (raw == 0).any() and (raw > 0).any()


def test_load_sample_measures_errors_on_sensor_depth_only(fixture_split):
    root, _ = fixture_split
    split = root / "test"
    config = Config(
        dataset=DatasetConfig(rgb_dir=split / "rgb", depth_dir=split / "depth", raw_depth_dir=split / "raw_depth"),
        outputs=OutputConfig(directory=root / "out"),
    )
    pair = RGBDepthPair("scene_0009", split / "rgb" / "scene_0009.png", split / "depth" / "scene_0009.npy")
    rgb, target, dense, valid = load_sample(pair, config)
    assert np.array_equal(target, np.load(split / "raw_depth" / "scene_0009.npy"))
    assert (dense > 0).all()  # in-painted depth stays dense for geometry cues
    assert not valid[target == 0].any()
    dense_config = replace(config, dataset=replace(config.dataset, raw_depth_dir=None))
    assert load_sample(pair, dense_config)[3].sum() > valid.sum()


def test_geometry_from_prediction_needs_no_ground_truth_geometry():
    rgb, depth = _scene(np.random.default_rng(5), 96, 128)
    prediction = SyntheticDisparityPredictor(5).predict(rgb, depth)
    config = ExperimentConfig(align_scale_shift=True, prediction_space="inverse_depth", geometry_source="prediction")
    result = analyze_sample(rgb, depth, prediction, np.ones_like(depth, dtype=bool), config)
    assert result["num_regions"] > 0
    with pytest.raises(ValueError, match="geometry_source"):
        analyze_sample(rgb, depth, prediction, np.ones_like(depth, dtype=bool), replace(config, geometry_source="oracle"))
