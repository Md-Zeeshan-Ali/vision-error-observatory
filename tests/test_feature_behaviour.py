"""What each feature should respond to, on synthetic regions.

The first five cases below were v1 defects (marked xfail until Phase 2 rebuilt the features).
"""

import numpy as np
import pytest

from veo.features import extract_features
from veo.scoring import score_categories


def _scores(rgb: np.ndarray, depth: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    ys, xs = np.where(mask)
    x0, y0 = int(xs.min()), int(ys.min())
    region = {"bbox": [x0, y0, int(xs.max()) - x0 + 1, int(ys.max()) - y0 + 1], "mask": mask}
    return score_categories(extract_features(rgb, depth, region))


def _canvas(value: float = 128.0, size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    return np.full((size, size, 3), value, dtype=np.uint8), np.full((size, size), 3.0, dtype=np.float32)


def _box(size: int = 64, top: int = 12, left: int = 12, height: int = 40, width: int = 40) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mask[top:top + height, left:left + width] = True
    return mask


def test_flat_region_is_textureless():
    rgb, depth = _canvas()
    assert _scores(rgb, depth, _box())["textureless"] > 0.9


def test_checkerboard_is_not_textureless():
    rgb, depth = _canvas()
    rows, cols = np.mgrid[0:64, 0:64]
    rgb[((rows // 4 + cols // 4) % 2).astype(bool)] = 230
    rgb[~((rows // 4 + cols // 4) % 2).astype(bool)] = 30
    assert _scores(rgb, depth, _box())["textureless"] < 0.1


def test_flat_white_wall_is_not_reflection():
    rgb, depth = _canvas(235.0)
    assert _scores(rgb, depth, _box())["reflection"] < 0.5


def test_diagonal_pole_is_thin():
    rgb, depth = _canvas()
    mask = np.zeros((64, 64), dtype=bool)
    for i in range(10, 54):
        mask[i, i - 1:i + 2] = True
    depth[mask] = 1.5  # a pole in front of a wall 3 m away
    assert _scores(rgb, depth, mask)["thin_object"] > 0.5


def test_long_strip_of_wall_is_not_thin():
    rgb, depth = _canvas()
    assert _scores(rgb, depth, _box(top=30, height=4, left=4, width=56))["thin_object"] < 0.5


def test_smooth_gradient_is_textureless():
    rgb, depth = _canvas()
    rgb[:] = np.linspace(40, 220, 64, dtype=np.uint8)[None, :, None]
    assert _scores(rgb, depth, _box())["textureless"] > 0.5


def test_large_relative_depth_step_is_boundary():
    rgb, depth = _canvas()
    depth[:, 32:] = 1.5
    depth[:, :32] = 1.0
    assert _scores(rgb, depth, _box(left=20, width=24))["boundary_mixing"] > 0.5


def test_same_step_is_found_at_any_scale():
    # Log-depth edges: a 1 m -> 1.5 m step and a 4 m -> 6 m step are the same relative jump
    rgb, near = _canvas()
    near[:, 32:], near[:, :32] = 1.5, 1.0
    assert _scores(rgb, near * 4.0, _box(left=20, width=24))["boundary_mixing"] == pytest.approx(
        _scores(rgb, near, _box(left=20, width=24))["boundary_mixing"]
    )


def test_gentle_slope_is_not_a_boundary():
    rgb, depth = _canvas()
    depth[:] = np.linspace(2.0, 3.0, 64, dtype=np.float32)[None, :]
    assert _scores(rgb, depth, _box())["boundary_mixing"] == 0.0


def test_local_highlight_is_reflection():
    rgb, depth = _canvas(120.0)
    rgb[24:40, 24:40] = 250  # specular spot much brighter than its surroundings
    assert _scores(rgb, depth, _box(top=22, left=22, height=20, width=20))["reflection"] > 0.5


def _thin_line_in_image_only():
    rgb, depth = _canvas(150.0)
    mask = np.zeros((64, 64), dtype=bool)
    mask[8:56, 30:33] = True
    rgb[:, 31] = 40  # a dark 1 px pole that the depth map does not contain
    return rgb, depth, mask


def test_image_line_counts_as_thin_only_without_ground_truth_geometry():
    rgb, depth, mask = _thin_line_in_image_only()
    region = {"bbox": [30, 8, 3, 48], "mask": mask}
    features = extract_features(rgb, depth, region)
    assert features["thin_image_fraction"] > 0.9 and features["thin_fraction"] == 0.0
    assert score_categories(features)["thin_object"] == 0.0
    assert score_categories(features, thresholds={"geometry_source": "prediction"})["thin_object"] > 0.5


def test_dark_region_is_poor_lighting():
    rgb, depth = _canvas(12.0)
    assert _scores(rgb, depth, _box())["poor_lighting"] > 0.5
    rgb, depth = _canvas(128.0)
    assert _scores(rgb, depth, _box())["poor_lighting"] == 0.0
