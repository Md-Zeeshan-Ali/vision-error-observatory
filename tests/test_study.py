import json
import re
from pathlib import Path

import numpy as np
import pytest

from veo.human_study import analyze, load_ratings
from veo.reproduce import reproduce
from veo.study import CATEGORIES, build_study, stratified_sample

REPO = Path(__file__).resolve().parents[1]


def test_stratified_sample_caps_strata_and_images():
    regions = [{"sample_id": f"s{i % 10}", "region_id": i, "top_cause": cause, "labels": [], "bbox": [0, 0, 1, 1], "area": 30}
               for i, cause in enumerate(["textureless"] * 50 + ["reflection"] * 3 + ["ambiguous"] * 20)]
    chosen = stratified_sample(regions, per_stratum=5, random_count=4, max_per_image=8, seed=0)
    strata = [item["stratum"] for item in chosen]
    assert strata.count("textureless") == 5 and strata.count("reflection") == 3 and strata.count("random") == 4
    assert max(sum(item["sample_id"] == s for item in chosen) for s in {r["sample_id"] for r in regions}) <= 8
    reflection = next(item for item in chosen if item["stratum"] == "reflection")
    assert reflection["inclusion_probability"] == 1.0  # all 3 reflection regions were taken
    assert len({(item["sample_id"], item["region_id"]) for item in chosen}) == len(chosen)


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("study")
    reproduce(workdir, REPO / "configs" / "nyu.yaml", fixture=True)
    outputs = workdir / "outputs" / "fixture" / "synthetic-disparity"
    built = build_study(outputs / "model_results.json", workdir / "study", train_results=outputs / "train" / "model_results.json",
                        per_stratum=6, random_count=6, practice_count=4, max_per_image=4, seed=1)
    return built, json.loads(built["manifest"].read_text(encoding="utf-8"))


def test_package_contents_and_blinding(study):
    built, manifest = study
    folder = built["output_dir"]
    html = (folder / "index.html").read_text(encoding="utf-8")
    data = json.loads(re.search(r'<script id="study-data" type="application/json">(.*?)</script>', html, re.S).group(1))
    assert len(data["items"]) == len(manifest["items"]) == built["items"] > 0
    for item in data["items"]:
        assert (folder / item["context"]).exists() and (folder / item["detail"]).exists()
        assert set(item) == {"id", "context", "detail"}
    for leaked in ("top_cause", "scene_0", "inclusion_probability", "silver"):
        assert leaked not in html
    assert not (folder / "manifest.json").exists() and built["manifest"].parent != folder
    assert all("silver_labels" in item for item in manifest["items"])
    assert [c["value"] for c in data["categories"]][-1] == "ambiguous"


def _simulated_rating(manifest: dict, rater: str, accuracy: float, seed: int) -> dict:
    """What the tool exports, from a rater who agrees with a hidden truth with some probability."""
    rng = np.random.default_rng(seed)
    values = [c["value"] for c in CATEGORIES]

    def answer(truth: str) -> dict:
        primary = truth if rng.random() < accuracy else str(rng.choice([v for v in values if v != truth]))
        return {"primary": primary, "labels": [] if primary == "ambiguous" else [primary], "confidence": "medium", "notes": ""}

    truth = {item["id"]: (item["silver_labels"] or ["ambiguous"])[0] for item in manifest["items"]}
    return {
        "study_id": manifest["study_id"], "rater_id": rater,
        "practice": {item["id"]: answer(item["reference"]) for item in manifest["practice"]},
        "ratings": {item_id: answer(label) for item_id, label in truth.items()},
    }


def test_analysis_of_simulated_raters(study, tmp_path):
    _, manifest = study
    paths = []
    for index, (rater, accuracy) in enumerate([("r1", 0.9), ("r2", 0.85), ("r3", 0.8), ("noise", 0.0)]):
        path = tmp_path / f"ratings_{rater}.json"
        path.write_text(json.dumps(_simulated_rating(manifest, rater, accuracy, index)), encoding="utf-8")
        paths.append(path)
    raters = load_ratings(paths, manifest["study_id"])
    result = analyze(manifest, raters, replicates=50)
    assert result["raters"]["noise"]["included"] is False  # fails the practice round
    assert result["sensitivity_all_raters"]["inter_rater"]["krippendorff_alpha"] < result["inter_rater"]["krippendorff_alpha"]
    assert result["items_rated"] == len(manifest["items"])
    assert result["inter_rater"]["krippendorff_alpha"] > 0.5
    assert result["inter_rater"]["fleiss_items"] == len(manifest["items"])
    bootstrap = result["veo_vs_consensus"]["bootstrap"]
    assert bootstrap["accuracy"]["lower"] <= bootstrap["accuracy"]["estimate"] <= bootstrap["accuracy"]["upper"]
    assert set(result["preregistered_checks"]) >= {"selective_accuracy_above_majority_baseline", "inter_rater_alpha_at_least_0_4"}
    json.dumps(result)

    from veo.summary import _human_study

    text = " ".join(_human_study(json.loads(json.dumps(result))))
    assert "Krippendorff" in text and "Pre-registered check" in text


def test_rater_ids_are_assigned_without_changing_the_files(study, tmp_path):
    _, manifest = study
    path = tmp_path / "ratings_typed.json"
    original = json.dumps(_simulated_rating(manifest, "r1", accuracy=0.8, seed=5))
    path.write_text(original, encoding="utf-8")
    [rater] = load_ratings([path], manifest["study_id"], ["R4"])
    assert rater["rater_id"] == "R4" and rater["typed_rater_id"] == "r1"
    assert path.read_text(encoding="utf-8") == original
    with pytest.raises(ValueError):
        load_ratings([path], manifest["study_id"], ["R4", "R5"])


def test_ratings_from_another_study_are_rejected(study, tmp_path):
    path = tmp_path / "ratings_x.json"
    path.write_text(json.dumps({"study_id": "other", "rater_id": "x", "ratings": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="belongs to study"):
        load_ratings([path], study[1]["study_id"])
