"""Agreement statistics for multi-rater studies.

Ratings are given as ``{item_id: {rater_id: label}}``; raters may skip items.
"""

from collections import Counter

import numpy as np


def _categories(ratings: dict[str, dict[str, str]]) -> list[str]:
    return sorted({label for by_rater in ratings.values() for label in by_rater.values()})


def fleiss_kappa(ratings: dict[str, dict[str, str]]) -> float:
    """Fleiss' kappa; every item must have the same number of ratings."""
    counts = {len(by_rater) for by_rater in ratings.values()}
    if len(counts) != 1 or counts.pop() < 2:
        raise ValueError("Fleiss' kappa needs the same number (>= 2) of ratings per item")
    categories = _categories(ratings)
    table = np.asarray([[Counter(by_rater.values())[c] for c in categories] for by_rater in ratings.values()], dtype=float)
    return fleiss_kappa_table(table)


def fleiss_kappa_table(table: np.ndarray) -> float:
    """Fleiss' kappa from an items x categories table of rating counts."""
    raters = table.sum(axis=1)[0]
    p_category = table.sum(axis=0) / table.sum()
    p_item = ((table * (table - 1)).sum(axis=1)) / (raters * (raters - 1))
    observed, expected = p_item.mean(), (p_category**2).sum()
    return float((observed - expected) / (1 - expected)) if expected < 1 else 1.0


def krippendorff_alpha_nominal(ratings: dict[str, dict[str, str]]) -> float:
    """Krippendorff's alpha for nominal data; items with fewer than two ratings are ignored."""
    categories = _categories(ratings)
    index = {label: i for i, label in enumerate(categories)}
    coincidence = np.zeros((len(categories), len(categories)))
    for by_rater in ratings.values():
        values = [index[label] for label in by_rater.values()]
        if len(values) < 2:
            continue
        counts = np.bincount(values, minlength=len(categories)).astype(float)
        pairs = np.outer(counts, counts) - np.diag(counts)  # ordered pairs of different raters
        coincidence += pairs / (len(values) - 1)
    totals = coincidence.sum(axis=1)
    n = totals.sum()
    disagreement_observed = coincidence.sum() - np.trace(coincidence)
    disagreement_expected = (np.outer(totals, totals).sum() - (totals**2).sum()) / (n - 1)
    return float(1 - disagreement_observed / disagreement_expected) if disagreement_expected > 0 else 1.0


def majority_vote(ratings: dict[str, dict[str, str]], tie: str = "unresolved") -> dict[str, str]:
    consensus = {}
    for item, by_rater in ratings.items():
        ranked = Counter(by_rater.values()).most_common()
        consensus[item] = ranked[0][0] if len(ranked) == 1 or ranked[0][1] > ranked[1][1] else tie
    return consensus


def dawid_skene(ratings: dict[str, dict[str, str]], iterations: int = 100, tolerance: float = 1e-6) -> dict:
    """Dawid-Skene EM: per-rater confusion matrices and a posterior over true labels per item.

    Initialised from vote shares; smoothing keeps unseen confusions from collapsing to zero.
    """
    items = sorted(ratings)
    raters = sorted({rater for by_rater in ratings.values() for rater in by_rater})
    categories = _categories(ratings)
    k = len(categories)
    label_index = {label: i for i, label in enumerate(categories)}
    observed = np.zeros((len(items), len(raters), k))
    for i, item in enumerate(items):
        for rater, label in ratings[item].items():
            observed[i, raters.index(rater), label_index[label]] = 1.0
    posterior = observed.sum(axis=1) + 1e-3
    posterior /= posterior.sum(axis=1, keepdims=True)
    for _ in range(iterations):
        prior = posterior.mean(axis=0)
        confusion = np.einsum("it,irl->rtl", posterior, observed) + 0.01
        confusion /= confusion.sum(axis=2, keepdims=True)
        log_likelihood = np.log(prior)[None, :] + np.einsum("irl,rtl->it", observed, np.log(confusion))
        updated = np.exp(log_likelihood - log_likelihood.max(axis=1, keepdims=True))
        updated /= updated.sum(axis=1, keepdims=True)
        converged = np.abs(updated - posterior).max() < tolerance
        posterior = updated
        if converged:
            break
    return {
        "labels": {item: categories[int(np.argmax(posterior[i]))] for i, item in enumerate(items)},
        "confidence": {item: float(posterior[i].max()) for i, item in enumerate(items)},
        "rater_accuracy": {
            rater: float(np.sum(prior * np.diag(confusion[r]))) for r, rater in enumerate(raters)
        },
        "categories": categories,
    }


def multilabel_consensus(label_sets: dict[str, dict[str, list[str]]], share: float = 0.5) -> dict[str, set[str]]:
    """Labels chosen by at least ``share`` of an item's raters."""
    consensus = {}
    for item, by_rater in label_sets.items():
        counts = Counter(label for labels in by_rater.values() for label in set(labels))
        consensus[item] = {label for label, count in counts.items() if count >= share * len(by_rater)}
    return consensus
