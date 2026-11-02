import numpy as np
import pytest

from veo.finetune import contrast, derangement, image_outcomes, scale_shift_loss, target_weight
from veo.silver import SILVER_CATEGORIES


def test_target_weight_gives_the_requested_share_of_the_loss():
    weight = target_weight(0.01, 0.2)
    assert 0.01 * weight / (0.99 + 0.01 * weight) == pytest.approx(0.2)
    with pytest.raises(ValueError):
        target_weight(0.0)


def test_derangement_moves_every_mask():
    order = derangement(50, 3)
    assert sorted(order.tolist()) == list(range(50)) and not np.any(order == np.arange(50))
    assert np.array_equal(order, derangement(50, 3)) and not np.array_equal(order, derangement(50, 4))


def _batch():
    torch = pytest.importorskip("torch")  # the model extra is not installed in CI
    disparity = torch.linspace(0.1, 1.0, 64).reshape(1, 8, 8)
    return disparity, torch.ones_like(disparity, dtype=torch.bool)


def test_loss_ignores_scale_and_shift():
    disparity, valid = _batch()
    assert float(scale_shift_loss(3.0 * disparity + 2.0, disparity, valid)) == pytest.approx(0.0, abs=1e-5)
    assert float(scale_shift_loss(disparity.flip(1) ** 2, disparity, valid)) > 0.01


def test_loss_uses_only_valid_pixels():
    disparity, valid = _batch()
    prediction = disparity.clone()
    prediction[0, 0, 0] = 50.0
    valid = valid.clone()
    valid[0, 0, 0] = False
    assert float(scale_shift_loss(prediction, disparity, valid)) == pytest.approx(0.0, abs=1e-5)


def test_weights_change_the_loss_but_not_the_fit():
    disparity, valid = _batch()
    import torch

    prediction = disparity.clone()
    prediction[0, 0, :4] += 0.5
    weights = torch.ones_like(disparity)
    plain = float(scale_shift_loss(prediction, disparity, valid, weights))
    assert plain == pytest.approx(float(scale_shift_loss(prediction, disparity, valid)))
    weights[0, 0, :4] = 10.0
    assert float(scale_shift_loss(prediction, disparity, valid, weights)) > plain


def test_gradients_reach_the_prediction():
    disparity, valid = _batch()
    prediction = (disparity ** 2).clone().requires_grad_(True)
    scale_shift_loss(prediction, disparity, valid).backward()
    assert prediction.grad is not None and float(prediction.grad.abs().sum()) > 0


def test_outcomes_measure_error_on_annotated_pixels_only():
    target = np.full((4, 4), 2.0)
    aligned = target.copy()
    aligned[0, :] = 3.0
    valid = np.ones((4, 4), dtype=bool)
    maps = {category: np.zeros((4, 4), dtype=bool) for category in SILVER_CATEGORIES}
    maps["boundary_mixing"][0, :] = True
    maps["textureless"][1, :] = True
    failing = aligned / target > 1.25
    outcomes = image_outcomes(aligned, target, valid, failing, {"abs_rel": 0.125, "delta1": 0.75}, maps)
    assert outcomes["abs_rel_boundary_mixing"] == pytest.approx(0.5) and outcomes["abs_rel_textureless"] == pytest.approx(0.0)
    assert outcomes["failing_boundary_mixing"] == 1.0 and outcomes["failing_pixels"] == pytest.approx(0.25)
    assert outcomes["abs_rel_reflection"] is None


def _runs(level, seeds=(0, 1, 2), spread=0.0):
    rng = np.random.default_rng(int(level * 1000))
    return {seed: {f"img{i}": {"x": level + seed * spread + rng.normal(0, 0.001), "y": None} for i in range(30)} for seed in seeds}


def test_contrast_reports_scenes_and_seed_agreement():
    scene_of = {f"img{i}": f"scene{i // 3}" for i in range(30)}
    result = contrast(_runs(0.10), _runs(0.20), "x", scene_of)
    assert result["n"] == 10 and result["upper"] < 0 and result["seeds_agree"] and result["mean"] == pytest.approx(-0.1, abs=0.005)
    assert set(result["per_seed"]) == {"0", "1", "2"}


def test_contrast_flags_seeds_that_disagree():
    first = {0: {f"img{i}": {"x": 0.1} for i in range(6)}, 1: {f"img{i}": {"x": 0.3} for i in range(6)}}
    second = {0: {f"img{i}": {"x": 0.2} for i in range(6)}, 1: {f"img{i}": {"x": 0.2} for i in range(6)}}
    assert not contrast(first, second, "x", {})["seeds_agree"]


def test_contrast_skips_images_without_the_outcome():
    first = {0: {"a": {"x": None}, "b": {"x": 0.1}, "c": {"x": 0.2}}}
    second = {0: {"a": {"x": 0.5}, "b": {"x": 0.2}, "c": {"x": 0.4}}}
    result = contrast(first, second, "x", {})
    assert result["n"] == 2 and result["mean"] == pytest.approx(-0.15)
