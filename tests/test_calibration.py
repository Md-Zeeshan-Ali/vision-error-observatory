import json

import numpy as np
import pytest

from veo.calibration import FEATURES, evaluate, fit, public_models


def _rows(count: int, seed: int) -> list[dict]:
    """Synthetic regions where silver labels depend on the features VEO uses, plus noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(count):
        features = {name: float(rng.uniform(0, 1)) for name in FEATURES}
        features["grad_energy"] = float(rng.uniform(0, 16))
        features["brightness"] = float(rng.uniform(0, 255))
        labels = []
        if features["grad_energy"] + rng.normal(0, 2) < 5:
            labels.append("textureless")
        if features["depth_edge_fraction"] + rng.normal(0, 0.1) > 0.6:
            labels.append("boundary_mixing")
        if features["thin_fraction"] > 0.8:
            labels.append("thin_object")
        rows.append({"scene": f"scene_{index // 4}", "features": features, "silver": {"labels": labels}})
    return rows


@pytest.fixture(scope="module")
def models():
    return fit(_rows(600, seed=1))


def test_calibrated_model_ranks_by_the_informative_feature(models):
    report = evaluate(models, _rows(400, seed=2), replicates=100)
    textureless = report["categories"]["textureless"]
    assert textureless["calibrated"]["auroc"] > 0.85
    assert textureless["calibrated"]["lower"] <= textureless["calibrated"]["auroc"] <= textureless["calibrated"]["upper"]
    assert textureless["learned"]["auroc"] > 0.8
    assert models["textureless"]["coef"][0] < 0  # more gradient energy -> less likely textureless
    assert 0.0 <= models["textureless"]["threshold"] <= 1.0


def test_category_without_positives_is_skipped(models):
    assert models["reflection"] is None
    report = evaluate(models, _rows(100, seed=3), replicates=100)
    assert "calibrated" not in report["categories"]["reflection"]
    assert report["categories"]["reflection"]["hand_set"]["auroc"] is None


def test_hand_set_rule_is_reported_with_label_metrics(models):
    entry = evaluate(models, _rows(200, seed=4), replicates=100)["categories"]["boundary_mixing"]["hand_set"]
    assert set(entry) >= {"auroc", "lower", "upper", "precision", "recall", "f1", "assigned"}


def test_public_models_are_json_serialisable(models):
    text = json.dumps(public_models(models))
    assert "_logistic" not in text and "coef" in text
