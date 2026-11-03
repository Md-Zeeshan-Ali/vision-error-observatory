import numpy as np
import pytest
from PIL import Image

from veo.data import discover_pairs, load_depth, validate_pairs
from veo.features import extract_features
from veo.inference import PrecomputedPredictor
from veo.metrics import align_scale_only, align_scale_shift, compute_metrics, tolerant_failure
from veo.nyu_dataset import create_scene_splits, extract_labeled_mat
from veo.regions import extract_failure_regions
from veo.reports import write_summary
from veo.scoring import classify
from veo.validation import cohens_kappa


def test_alignment_recovers_affine_depth():
    prediction = np.arange(16, dtype=np.float32).reshape(4, 4)
    target = 2.0 * prediction + 1.0
    aligned, scale, shift = align_scale_shift(prediction, target, np.ones((4, 4), dtype=bool))
    assert np.isclose(scale, 2.0)
    assert np.isclose(shift, 1.0)
    assert np.allclose(aligned, target)


def test_alignment_handles_inverse_depth_predictions():
    target = np.array([[2.0, 4.0], [5.0, 10.0]], dtype=np.float32)
    inverse_prediction = 3.0 / target + 0.2
    aligned, scale, shift = align_scale_shift(
        inverse_prediction, target, np.ones_like(target, dtype=bool), "inverse_depth"
    )
    assert np.isfinite((scale, shift)).all()
    assert np.allclose(aligned, target, atol=1e-5)


def test_scale_only_alignment_handles_inverse_depth_predictions():
    target = np.array([[2.0, 4.0], [5.0, 10.0]], dtype=np.float32)
    inverse_prediction = 3.0 / target
    aligned, scale = align_scale_only(inverse_prediction, target, np.ones_like(target, dtype=bool))
    assert np.isclose(scale, 1.0 / 3.0)
    assert np.allclose(aligned, target, atol=1e-5)


def test_metrics_use_only_valid_pixels():
    prediction = np.array([[1.0, 3.0], [100.0, 1.0]], dtype=np.float32)
    target = np.ones((2, 2), dtype=np.float32)
    valid = np.array([[True, True], [False, True]])
    metrics = compute_metrics(prediction, target, valid)
    assert np.isclose(metrics["abs_rel"], 2.0 / 3.0)
    assert {"mae", "rmse", "abs_rel", "sq_rel", "delta1", "delta2", "delta3"} <= metrics.keys()


def test_metrics_ignore_nonfinite_predictions():
    prediction = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)
    target = np.ones((2, 2), dtype=np.float32)
    metrics = compute_metrics(prediction, target, np.ones((2, 2), dtype=bool))
    assert np.isfinite(list(metrics.values())).all()


def test_tolerant_failure_forgives_only_small_shifts():
    # A depth edge predicted 2 px too far right: the band between fails strictly, but not with a 2 px tolerance
    target = np.where(np.arange(12)[None, :] < 6, 1.0, 4.0).repeat(5, axis=0).astype(np.float32)
    prediction = np.where(np.arange(12)[None, :] < 8, 1.0, 4.0).repeat(5, axis=0).astype(np.float32)
    failing = np.maximum(prediction / target, target / prediction) > 1.25
    assert failing.sum() == 10
    assert tolerant_failure(prediction, target, failing, 1).sum() == 5
    assert not tolerant_failure(prediction, target, failing, 2).any()
    # A wrong value over a whole area is not forgiven
    assert tolerant_failure(2.0 * target, target, np.ones_like(failing), 3).all()
    with pytest.raises(ValueError):
        tolerant_failure(prediction, target, failing, -1)


def test_regions_respect_area_and_mask():
    error = np.zeros((8, 8), dtype=np.float32)
    error[2:5, 2:5] = 1.0
    valid = np.ones_like(error, dtype=bool)
    regions = extract_failure_regions(error, valid, percentile=80, min_area=3)
    assert len(regions) == 1
    assert regions[0]["area"] == 9


def _features(**overrides):
    features = {
        "grad_energy": 8.0, "corner_strength": 0.0, "depth_edge_fraction": 0.0, "depth_step": 0.0,
        "thin_fraction": 0.0, "highlight_ratio": 0.0, "clipped_fraction": 0.0, "brightness": 128.0,
        "contrast": 20.0, "aspect_ratio": 1.0, "mean_depth_m": 3.0,
    }
    return {**features, **overrides}


def test_classification_abstains_on_competing_evidence():
    result = classify(_features(grad_energy=0.0, depth_edge_fraction=0.5))
    assert result["top_cause"] == "ambiguous"
    assert result["confidence"] == "Low"
    assert set(result["labels"]) == {"textureless", "boundary_mixing"}
    assert "score_margin" in result


def test_classification_abstains_without_evidence():
    result = classify(_features())
    assert result["top_cause"] == "ambiguous" and result["labels"] == []


def test_classification_accepts_clear_evidence():
    result = classify(_features(depth_edge_fraction=0.45))
    assert result["top_cause"] == "boundary_mixing"
    assert result["confidence"] == "High"


def test_reflection_can_be_disabled():
    assert "reflection" not in classify(_features(highlight_ratio=1.0), include_reflection=False)["scores"]


def test_nyu_pair_discovery_and_precomputed_prediction(tmp_path):
    rgb_dir = tmp_path / "rgb"
    depth_dir = tmp_path / "depth"
    prediction_dir = tmp_path / "predictions"
    rgb_dir.mkdir()
    depth_dir.mkdir()
    prediction_dir.mkdir()
    Image.fromarray(np.zeros((3, 4, 3), dtype=np.uint8)).save(rgb_dir / "scene_0001.png")
    Image.fromarray(np.full((3, 4), 2000, dtype=np.uint16)).save(depth_dir / "scene_0001.png")
    np.save(prediction_dir / "scene_0001.npy", np.ones((3, 4), dtype=np.float32))
    pairs = discover_pairs(rgb_dir, depth_dir)
    validate_pairs(pairs)
    assert load_depth(pairs[0].depth_path).mean() == 2.0
    assert PrecomputedPredictor(prediction_dir).predict_for_sample("scene_0001", (3, 4)).shape == (3, 4)


def test_summary_exports_region_csv(tmp_path):
    records = [{"sample_id": "scene_0001", "metrics": {"abs_rel": 0.1, "rmse": 0.2, "delta1": 0.9}, "num_regions": 1, "regions": [{"region_id": 1, "mean_error": 0.4, "area": 8, "top_cause": "occlusion", "confidence": "High"}]}]
    write_summary(records, tmp_path)
    assert (tmp_path / "summary.json").exists()
    assert "occlusion" in (tmp_path / "regions.csv").read_text(encoding="utf-8")


def test_extract_official_nyu_mat_layout(tmp_path):
    mat_path = tmp_path / "nyu.mat"
    import h5py

    with h5py.File(mat_path, "w") as handle:
        handle.create_dataset("images", data=np.zeros((1, 3, 4, 3), dtype=np.uint8))
        handle.create_dataset("depths", data=np.full((1, 4, 3), 2.0, dtype=np.float32))
    extracted = extract_labeled_mat(mat_path, tmp_path / "rgb", tmp_path / "depth")
    assert extracted == 1
    assert Image.open(tmp_path / "rgb" / "scene_0001.png").size == (4, 3)
    assert np.allclose(np.load(tmp_path / "depth" / "scene_0001.npy"), 2.0)


def test_scene_splits_are_disjoint(tmp_path):
    mat_path = tmp_path / "nyu.mat"
    import h5py

    with h5py.File(mat_path, "w") as handle:
        refs = handle.create_group("#refs#")
        scene_refs = []
        for index, name in enumerate(["a", "a", "b", "b", "c", "c"]):
            key = f"{index:04d}"
            refs.create_dataset(key, data=np.frombuffer(name.encode("utf-16le"), dtype=np.uint16))
            scene_refs.append(refs[key].ref)
        handle.create_dataset("scenes", data=np.asarray(scene_refs, dtype=h5py.ref_dtype).reshape(1, -1))
    splits = create_scene_splits(mat_path, seed=42, train_fraction=0.5, validation_fraction=0.25)
    scene_sets = [{record["scene"] for record in records} for records in splits.values()]
    assert not (scene_sets[0] & scene_sets[1] or scene_sets[0] & scene_sets[2] or scene_sets[1] & scene_sets[2])


def test_features_handle_one_pixel_wide_region():
    rgb = np.zeros((3, 1, 3), dtype=np.uint8)
    depth = np.ones((3, 1), dtype=np.float32)
    region = {"bbox": [0, 0, 1, 3], "mask": np.ones((3, 1), dtype=bool)}
    features = extract_features(rgb, depth, region)
    assert features["depth_edge_fraction"] == 0.0 and features["aspect_ratio"] == 3.0


def test_features_ignore_unmeasured_depth():
    rgb = np.full((20, 20, 3), 128, dtype=np.uint8)
    depth = np.full((20, 20), 2.0, dtype=np.float32)
    depth[:, 10:] = 0.0  # sensor holes must not look like depth edges
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 2:8] = True
    assert extract_features(rgb, depth, {"bbox": [2, 5, 6, 10], "mask": mask})["depth_edge_fraction"] == 0.0


def test_cohens_kappa_perfect_agreement():
    assert cohens_kappa(["a", "b", "a"], ["a", "b", "a"]) == 1.0
