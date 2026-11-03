"""Fitted failure-label model (docs/LABEL_MODEL_V3.md).

Each category is a small logistic regression on physically motivated features with its own label
threshold. Ratios of probability to threshold make categories comparable, so a rare, specific label can
win over a common one. ``poor_lighting`` has no objective label, so it keeps its hand-set rule.
"""

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from .scoring import score_categories

REPO_ROOT = Path(__file__).resolve().parents[2]
POOR_LIGHTING_THRESHOLD = 0.5


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.exists() else REPO_ROOT / candidate


def model_fingerprint(path: str | None) -> str | None:
    """Content hash, so checkpoints made with another fit of the model are not reused."""
    return hashlib.sha256(_resolve(path).read_bytes()).hexdigest()[:16] if path else None


class LabelModel:
    def __init__(self, categories: dict):
        self.categories = categories

    def probabilities(self, features: dict[str, float]) -> dict[str, float]:
        scores = {}
        for category, model in self.categories.items():
            x = np.asarray([features.get(name, 0.0) for name in model["features"]], dtype=float)
            standardised = (x - np.asarray(model["mean"])) / np.asarray(model["std"])
            scores[category] = float(1.0 / (1.0 + np.exp(-(standardised @ np.asarray(model["coef"]) + model["intercept"]))))
        return scores

    def classify(self, features: dict[str, float], settings: dict | None = None) -> dict:
        scores = self.probabilities(features)
        rules = {**(settings or {}), "label_model": None}
        scores["poor_lighting"] = score_categories(features, True, rules)["poor_lighting"]
        thresholds = {**{category: model["threshold"] for category, model in self.categories.items()}, "poor_lighting": POOR_LIGHTING_THRESHOLD}
        ratios = sorted(((scores[name] / thresholds[name], name) for name in scores), reverse=True)
        top_ratio, top_cause = ratios[0]
        committed = top_ratio >= 1.0
        margin = top_ratio - ratios[1][0]
        return {
            "scores": scores,
            "top_cause": top_cause if committed else "ambiguous",
            "labels": [name for ratio, name in ratios if ratio >= 1.0],
            "confidence": "Low" if not committed else "High" if top_ratio >= 1.5 else "Medium",
            "score_margin": float(margin),
        }


@lru_cache(maxsize=4)
def load_label_model(path: str | None) -> LabelModel | None:
    if not path:
        return None
    return LabelModel(json.loads(_resolve(path).read_text(encoding="utf-8"))["categories"])
