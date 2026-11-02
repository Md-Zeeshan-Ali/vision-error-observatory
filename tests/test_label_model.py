import json
from pathlib import Path

import numpy as np
import pytest

from veo.features import error_context_features
from veo.label_model import LabelModel, load_label_model, model_fingerprint
from veo.validation import ablation_change, model_neutral_values

REPO = Path(__file__).resolve().parents[1]


def _category(feature: str, threshold: float, coef: float = 2.0) -> dict:
    return {"features": [feature], "mean": [0.5], "std": [0.25], "coef": [coef], "intercept": -1.0, "threshold": threshold}


@pytest.fixture
def model():
    return LabelModel({"boundary_mixing": _category("depth_edge_fraction", 0.5), "thin_object": _category("thin_fraction", 0.1)})


def _features(**values):
    return {"grad_energy": 8.0, "highlight_ratio": 0.0, "brightness": 128.0, "clipped_fraction": 0.0, "depth_edge_fraction": 0.0, "thin_fraction": 0.0, **values}


def test_rare_label_wins_through_its_own_threshold(model):
    # Boundary has the higher probability, but relative to its threshold thin is the stronger claim
    result = model.classify(_features(depth_edge_fraction=0.7, thin_fraction=0.4))
    assert result["scores"]["boundary_mixing"] > result["scores"]["thin_object"]
    assert result["top_cause"] == "thin_object"


def test_no_ratio_reaching_one_abstains(model):
    result = model.classify(_features())
    assert result["top_cause"] == "ambiguous" and result["labels"] == [] and result["confidence"] == "Low"


def test_poor_lighting_keeps_its_rule(model):
    result = model.classify(_features(brightness=5.0))
    assert result["top_cause"] == "poor_lighting" and result["scores"]["poor_lighting"] == pytest.approx(0.9)


def test_all_labels_at_or_above_threshold_are_listed(model):
    result = model.classify(_features(depth_edge_fraction=1.0, thin_fraction=0.5))
    assert set(result["labels"]) == {"boundary_mixing", "thin_object"}


def test_committed_model_file_loads_and_uses_only_allowed_features():
    path = "configs/label_model_v3.json"
    loaded = load_label_model(path)
    assert set(loaded.categories) == {"reflection", "thin_object", "boundary_mixing", "textureless"}
    assert model_fingerprint(path)
    stored = json.loads((REPO / path).read_text(encoding="utf-8"))
    assert all(len(category["features"]) <= 3 for category in stored["categories"].values())
    assert load_label_model(None) is None


def test_ablation_under_a_label_model_uses_training_means():
    strict = LabelModel({"thin_object": _category("thin_fraction", 0.5)})
    assert model_neutral_values(strict)["thin_fraction"] == 0.5
    region = {"features": _features(thin_fraction=0.9), "top_cause": "thin_object"}
    assert ablation_change([{"regions": [region]}], "thin_fraction", label_model=strict)["changed_regions"] == 1  # now abstains


def test_error_context_features():
    target = np.full((40, 40), 2.0, dtype=np.float32)
    aligned = target * 1.5
    mask = np.zeros((40, 40), dtype=bool)
    mask[15:25, 15:25] = True
    assert error_context_features(mask, aligned, target) == pytest.approx({"signed_error": 0.5, "sensor_hole_fraction": 0.0})
    target[:, :20] = 0.0  # half of the neighbourhood was not measured
    context = error_context_features(mask, aligned, target)
    assert 0.3 < context["sensor_hole_fraction"] < 0.7 and context["signed_error"] == pytest.approx(0.5)
