import numpy as np
import pytest

from veo.interventions import (
    add_texture,
    brighten,
    holm,
    match_controls,
    paired_statistics,
    predict_at_scale,
    region_error,
)


def _mask():
    mask = np.zeros((60, 80), dtype=bool)
    mask[20:40, 30:50] = True
    return mask


def test_texture_is_local_and_keeps_brightness():
    rgb = np.full((60, 80, 3), 128, dtype=np.uint8)
    edited = add_texture(rgb, _mask(), amplitude=25, seed=1)
    inside = edited[_mask()].astype(float)
    assert inside.std() > 10 and abs(inside.mean() - 128) < 3
    assert np.array_equal(edited[:5], rgb[:5])  # far from the region: untouched


def test_brighten_lifts_dark_pixels_inside_only():
    rgb = np.full((60, 80, 3), 30, dtype=np.uint8)
    edited = brighten(rgb, _mask())
    assert edited[30, 40, 0] > 80 and edited[0, 0, 0] == 30


def test_predict_at_scale_returns_original_size():
    seen = {}

    def predict(image):
        seen["shape"] = image.shape
        return np.ones(image.shape[:2], dtype=np.float32)

    assert predict_at_scale(predict, np.zeros((60, 80, 3), dtype=np.uint8), 1.5).shape == (60, 80)
    assert seen["shape"][:2] == (90, 120)


def test_region_error_is_zero_for_a_perfect_disparity():
    target = np.linspace(1.0, 4.0, 60 * 80, dtype=np.float32).reshape(60, 80)
    valid = np.ones_like(target, dtype=bool)
    assert region_error(2.0 / target + 0.1, target, valid, _mask(), "inverse_depth") == pytest.approx(0.0, abs=1e-5)


def test_matching_never_pairs_a_region_with_itself():
    rows = [{"sample_id": "a", "region_id": i, "log_area": float(i), "mean_error": 0.3, "log_depth": 1.0} for i in range(6)]
    pairs = match_controls(rows[:3], rows, seed=0)
    assert len(pairs) == 3
    assert all((t["sample_id"], t["region_id"]) != (c["sample_id"], c["region_id"]) for t, c in pairs)
    assert len({id(c) for _, c in pairs}) == 3  # without replacement


def test_paired_statistics_detects_a_real_effect_and_not_noise():
    rng = np.random.default_rng(0)
    scenes = [f"s{i % 20}" for i in range(200)]
    effect = paired_statistics(rng.normal(-0.05, 0.05, 200), scenes, permutations=2000)
    assert effect["upper"] < 0 and effect["p_value"] < 0.01 and effect["cohens_dz"] < -0.5
    null = paired_statistics(rng.normal(0.0, 0.05, 200), scenes, permutations=2000)
    assert null["lower"] < 0 < null["upper"] and null["p_value"] > 0.05


def test_holm_adjustment():
    adjusted = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})
