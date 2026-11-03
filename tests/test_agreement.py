import numpy as np
import pytest

from veo.agreement import (
    dawid_skene,
    fleiss_kappa,
    fleiss_kappa_table,
    krippendorff_alpha_nominal,
    majority_vote,
    multilabel_consensus,
)


def test_fleiss_kappa_matches_the_published_example():
    # Fleiss (1971) style example used on Wikipedia: 10 items, 14 raters, 5 categories -> 0.210
    table = np.array([
        [0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0], [2, 2, 8, 1, 1],
        [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2], [6, 5, 2, 1, 0], [0, 2, 2, 3, 7],
    ])
    assert fleiss_kappa_table(table) == pytest.approx(0.210, abs=5e-4)


def test_fleiss_kappa_from_ratings_and_perfect_agreement():
    perfect = {f"i{n}": {"a": label, "b": label, "c": label} for n, label in enumerate("xyxzy")}
    assert fleiss_kappa(perfect) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        fleiss_kappa({"i1": {"a": "x", "b": "x"}, "i2": {"a": "x"}})


def test_krippendorff_alpha_hand_computed_case():
    # Two raters, units (a,a),(a,b),(b,b),(b,b): alpha = 1 - 7 * 2 / (2 * 3 * 5) = 0.5333
    ratings = {"u1": {"r1": "a", "r2": "a"}, "u2": {"r1": "a", "r2": "b"}, "u3": {"r1": "b", "r2": "b"}, "u4": {"r1": "b", "r2": "b"}}
    assert krippendorff_alpha_nominal(ratings) == pytest.approx(1 - 14 / 30)


def test_krippendorff_alpha_ignores_single_ratings():
    ratings = {"u1": {"r1": "a", "r2": "a"}, "u2": {"r1": "b", "r2": "b"}, "u3": {"r1": "a"}}
    assert krippendorff_alpha_nominal(ratings) == pytest.approx(1.0)


def test_majority_vote_marks_ties():
    votes = majority_vote({"i1": {"a": "x", "b": "x", "c": "y"}, "i2": {"a": "x", "b": "y"}})
    assert votes == {"i1": "x", "i2": "unresolved"}


def test_dawid_skene_downweights_an_unreliable_rater():
    rng = np.random.default_rng(0)
    truth = rng.choice(["p", "q", "r"], size=300)
    def noisy(label: str, accuracy: float) -> str:
        return label if rng.random() < accuracy else str(rng.choice([c for c in "pqr" if c != label]))

    ratings = {
        f"i{n}": {"good1": noisy(label, 0.9), "good2": noisy(label, 0.85), "bad1": noisy(label, 0.4), "bad2": noisy(label, 0.4)}
        for n, label in enumerate(truth)
    }
    result = dawid_skene(ratings)
    accuracy = np.mean([result["labels"][f"i{n}"] == label for n, label in enumerate(truth)])
    majority = majority_vote(ratings)
    majority_accuracy = np.mean([majority[f"i{n}"] == label for n, label in enumerate(truth)])
    assert accuracy > 0.85 and accuracy > majority_accuracy + 0.1
    for rater, true_accuracy in {"good1": 0.9, "good2": 0.85, "bad1": 0.4, "bad2": 0.4}.items():
        assert result["rater_accuracy"][rater] == pytest.approx(true_accuracy, abs=0.1)


def test_multilabel_consensus():
    consensus = multilabel_consensus({"i1": {"a": ["x", "y"], "b": ["x"], "c": ["y", "z"]}})
    assert consensus == {"i1": {"x", "y"}}
