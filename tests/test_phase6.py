import io
import json
import sys
import tarfile
import types

import numpy as np
import pytest

from veo.config import Config
from veo.diode import prepare_indoor
from veo.hypersim import SILVER_GROUPS
from veo.ibims import prepare_core
from veo.ibims import silver_maps as ibims_silver_maps
from veo.inference import (
    AFFINE_DEPTH_MODELS,
    MarigoldDepthPredictor,
    TransformersDepthPredictor,
    load_predictor,
)
from veo.metrics import align_scale_shift
from veo.model_comparison import (
    LABELS,
    compare,
    compare_composition,
    fingerprint_verdict,
    image_composition,
    image_outcomes,
)
from veo.runner import config_for_model
from veo.silver import region_silver, silver_maps
from veo.summary import _ibims_labels, _model_comparison


def _record(sample_id, abs_rel, failing):
    return {
        "sample_id": sample_id, "valid_pixels": 1000, "failure_pixel_fraction": failing,
        "metrics": {"abs_rel": abs_rel, "delta1": 1 - abs_rel},
        "regions": [{"area": 100, "top_cause": "textureless"}, {"area": 50, "top_cause": "ambiguous"}],
    }


def test_image_outcomes_share_pixels_per_label():
    outcomes = image_outcomes(_record("a", 0.1, 0.15))
    assert outcomes["pixels_textureless"] == pytest.approx(0.1) and outcomes["pixels_ambiguous"] == pytest.approx(0.05)


def test_comparison_uses_scenes_as_the_unit():
    rng = np.random.default_rng(0)
    ids = [f"img{i}" for i in range(120)]
    scene_of = {sample_id: f"scene{i // 3}" for i, sample_id in enumerate(ids)}  # 40 scenes
    first = [_record(i, 0.10 + rng.normal(0, 0.005), 0.1) for i in ids]
    second = [_record(i, 0.13 + rng.normal(0, 0.005), 0.1) for i in ids]
    result = compare(first, second, scene_of)
    assert result["scenes"] == 40 and result["images"] == 120
    abs_rel = result["outcomes"]["abs_rel"]
    assert abs_rel["reliable"] and abs_rel["upper"] < 0 and abs_rel["n"] == 40
    assert not result["outcomes"]["failing_pixels"].get("reliable", False)
    text = "\n".join(_model_comparison({"contrasts": {"x": {"first": "a", "second": "b", **result}}}))
    assert "abs_rel" in text and "yes" in text


def test_comparison_needs_shared_images():
    with pytest.raises(ValueError, match="share no images"):
        compare([_record("a", 0.1, 0.1)], [_record("b", 0.1, 0.1)], {})


def _npy(array) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array)
    return buffer.getvalue()


def test_diode_preparation_keeps_indoor_frames_and_applies_the_mask(tmp_path):
    from PIL import Image

    png = io.BytesIO()
    Image.new("RGB", (4, 3)).save(png, format="PNG")
    depth = np.full((3, 4, 1), 2.0, dtype=np.float32)
    mask = np.ones((3, 4), dtype=np.uint8)
    mask[0, 0] = 0
    archive = tmp_path / "val.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for folder, stem in (("val/indoors/scene_00019/scan_00183", "a"), ("val/outdoor/scene_00022/scan_00001", "b")):
            for suffix, payload in ((".png", png.getvalue()), ("_depth.npy", _npy(depth)), ("_depth_mask.npy", _npy(mask))):
                info = tarfile.TarInfo(f"{folder}/{stem}{suffix}")
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
    output = prepare_indoor(archive, tmp_path / "prepared")
    assert [record["sample_id"] for record in json.loads((output / "metadata.json").read_text(encoding="utf-8"))] == ["a"]
    assert json.loads((output / "metadata.json").read_text(encoding="utf-8"))[0]["scene"] == "scene_00019"
    prepared = np.load(output / "depth" / "a.npy")
    assert prepared.shape == (3, 4) and prepared[0, 0] == 0.0 and prepared[1, 1] == 2.0
    assert (output / "rgb" / "a.png").exists() and not (output / "rgb" / "b.png").exists()
    assert prepare_indoor(tmp_path / "gone.tar.gz", output) == output  # already prepared: the archive is not needed


class _Tensor:
    def __init__(self, array):
        self._array = array

    def squeeze(self):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._array


def test_metric_models_are_inverted_to_disparity(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)))
    pipeline = lambda *args, **kwargs: (lambda image: {"predicted_depth": _Tensor(np.full((4, 4), 2.0, dtype=np.float32))})  # noqa: E731
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(pipeline=pipeline))
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    assert TransformersDepthPredictor("Intel/zoedepth-nyu-kitti").predict(rgb)[0, 0] == pytest.approx(0.5)
    assert TransformersDepthPredictor("Intel/dpt-large").predict(rgb)[0, 0] == pytest.approx(2.0)


def test_silver_labels_work_with_hypersim_class_ids():
    semantic = np.full((60, 60), 1, dtype=np.int32)  # wall
    semantic[10:30, 10:30] = 19  # mirror
    depth = np.full((60, 60), 3.0, dtype=np.float32)
    maps = silver_maps(semantic, np.zeros_like(semantic), depth, SILVER_GROUPS)
    on_mirror = np.zeros((60, 60), dtype=bool)
    on_mirror[15:25, 15:25] = True
    assert region_silver(maps, on_mirror)["labels"] == ["reflection"]
    on_wall = np.zeros((60, 60), dtype=bool)
    on_wall[40:55, 40:55] = True
    assert region_silver(maps, on_wall)["labels"] == ["textureless"]


def test_composition_is_the_mix_of_labels_and_skips_images_without_regions():
    assert image_composition(_record("a", 0.1, 0.15))["textureless"] == pytest.approx(100 / 150)
    assert image_composition({**_record("a", 0.1, 0.0), "regions": []}) is None


def test_composition_ignores_how_much_each_model_fails():
    ids = [f"img{i}" for i in range(12)]
    small = [_record(i, 0.1, 0.1) for i in ids]
    # Ten times the area with the same mix of labels, so there is nothing to find
    large = [{**_record(i, 0.1, 0.1), "regions": [{"area": 1000, "top_cause": "textureless"}, {"area": 500, "top_cause": "ambiguous"}]} for i in ids]
    outcomes = compare_composition(small, large, {})["outcomes"]
    assert outcomes["textureless"]["mean"] == pytest.approx(0.0) and not outcomes["textureless"]["reliable"]


def _comparison(label_means):
    outcomes = {
        label: {"mean": label_means.get(label, (0.0, False))[0], "lower": -1, "upper": 1, "p_holm": 0.01, "reliable": label_means.get(label, (0.0, False))[1]}
        for label in LABELS
    }
    return {"contrasts": {name: {"composition": {"outcomes": outcomes}} for name in ("family_dav2_small_vs_dpt_hybrid", "backbone_hybrid_vs_large")}}


def test_fingerprint_needs_one_reliable_sign_on_every_dataset():
    good = _comparison({"textureless": (0.1, True)})
    assert fingerprint_verdict({"a": good, "b": good}, {})["backbone_hybrid_vs_large"]["verdict"] == "supported"
    flipped = _comparison({"textureless": (-0.1, True)})
    assert fingerprint_verdict({"a": good, "b": flipped}, {})["backbone_hybrid_vs_large"]["verdict"] == "cut"
    unreliable = _comparison({"textureless": (0.1, False)})
    assert fingerprint_verdict({"a": good, "b": unreliable}, {})["backbone_hybrid_vs_large"]["verdict"] == "cut"


def test_a_failure_type_claim_also_needs_a_validated_label():
    good = _comparison({"boundary_mixing": (0.1, True), "textureless": (0.1, True)})
    labels = fingerprint_verdict({"a": good}, {"boundary_mixing": {"h": 0.8, "i": 0.75}, "textureless": {"h": 0.5, "i": 0.8}})["backbone_hybrid_vs_large"]["labels"]
    assert labels["boundary_mixing"]["failure_type_difference"]
    assert labels["textureless"]["passes"] and not labels["textureless"]["failure_type_difference"]
    assert labels["ambiguous"]["failure_type_difference"] is False


def _ibims_mat(path, name, edges_on=True):
    from scipy.io import savemat

    shape = (30, 40)
    invalid = np.ones(shape, dtype=np.uint8)
    invalid[0, 0] = 0
    transparent = np.ones(shape, dtype=np.uint8)
    transparent[10:20, 10:20] = 0
    edges = np.zeros(shape, dtype=np.uint8)
    if edges_on:
        edges[5, :] = 1
    wall = np.zeros(shape, dtype=np.uint8)
    wall[20:, :] = 2  # plane numbers, not just 0 and 1
    savemat(path, {"data": {
        "image_name": name, "rgb": np.zeros((*shape, 3), dtype=np.uint8), "depth": np.full(shape, 3.0), "mask_invalid": invalid, "mask_transp": transparent,
        "edges": edges, "mask_wall": wall, "mask_floor": np.zeros(shape, dtype=np.uint8), "mask_table": np.zeros(shape, dtype=np.uint8),
    }})


def test_ibims_preparation_and_silver_labels(tmp_path):
    import zipfile

    archive = tmp_path / "core.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for name, edges_on in (("office_01", True), ("office_02", False)):
            _ibims_mat(tmp_path / f"{name}.mat", name, edges_on)
            bundle.write(tmp_path / f"{name}.mat", f"ibims1_core_mat/{name}.mat")
    output = prepare_core(archive, tmp_path / "prepared")
    records = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert [(r["sample_id"], r["scene"], r["has_edges"]) for r in records] == [("office_01", "office", True), ("office_02", "office", False)]
    depth = np.load(output / "depth" / "office_01.npy")
    assert depth[0, 0] == 0.0 and depth[1, 1] == 3.0  # invalid pixels become unmeasured
    assert depth[12, 12] == 3.0  # transparent pixels stay measured
    with np.load(output / "masks" / "office_01.npz") as stored:
        maps = ibims_silver_maps(dict(stored))
    region = np.zeros((30, 40), dtype=bool)
    region[12:18, 12:18] = True
    assert region_silver(maps, region)["labels"] == ["reflection"]
    region = np.zeros((30, 40), dtype=bool)
    region[24:28, 5:15] = True
    assert region_silver(maps, region)["labels"] == ["textureless"]
    region = np.zeros((30, 40), dtype=bool)
    region[3:8, 25:35] = True
    assert region_silver(maps, region)["labels"] == ["boundary_mixing"]
    assert prepare_core(tmp_path / "gone.zip", output) == output  # already prepared: the archive is not needed


def test_ibims_summary_section():
    entry = {"positives": 3, "label_model": {"auroc": 0.8, "lower": 0.7, "upper": 0.9}, "hand_set": {"auroc": 0.6, "lower": 0.5, "upper": 0.7}}
    text = "\n".join(_ibims_labels({
        "regions": 9, "images": 2, "categories": {"boundary_mixing": entry},
        "textureless_any_plane": entry["hand_set"], "reflection_sensor_holes_only": entry["hand_set"],
    }))
    assert "`boundary_mixing`" in text and "9 regions from 2 images" in text


class _MarigoldPipe:
    def __init__(self):
        self.calls = []

    def set_progress_bar_config(self, **kwargs):
        pass

    def to(self, device):
        return self

    def __call__(self, image, **kwargs):
        self.calls.append(kwargs)
        return types.SimpleNamespace(prediction=np.full((1, 4, 4, 1), 0.25))


def test_marigold_uses_the_registered_settings_and_is_not_inverted(monkeypatch):
    pipe = _MarigoldPipe()
    torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False), float16="half", float32="full",
        Generator=lambda device: types.SimpleNamespace(manual_seed=lambda seed: f"seed{seed}"),
    )
    loaded = {}

    def from_pretrained(model_id, **kwargs):
        loaded.update(kwargs)
        return pipe

    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "diffusers", types.SimpleNamespace(MarigoldDepthPipeline=types.SimpleNamespace(from_pretrained=from_pretrained)))
    predictor = load_predictor("prs-eth/marigold-depth-v1-1", "cpu")
    assert isinstance(predictor, MarigoldDepthPredictor)
    prediction = predictor.predict(np.zeros((4, 4, 3), dtype=np.uint8))
    assert prediction.shape == (4, 4) and prediction[0, 0] == pytest.approx(0.25)
    assert loaded == {"variant": None, "torch_dtype": "full"}
    call = pipe.calls[0]
    assert (call["num_inference_steps"], call["ensemble_size"], call["processing_resolution"], call["generator"]) == (4, 1, 768, "seed42")


def test_affine_depth_models_are_aligned_in_depth_space():
    config = Config()
    marigold = config_for_model(config, next(iter(AFFINE_DEPTH_MODELS)))
    assert marigold.experiment.prediction_space == "depth" and marigold.experiment.primary_protocol == "direct_depth_scale_shift"
    assert config_for_model(config, "Intel/dpt-large") is config
    # An affine depth map is recovered exactly in depth space, which disparity alignment wouldn't do
    target = np.linspace(1.0, 8.0, 400, dtype=np.float32).reshape(20, 20)
    aligned, _, _ = align_scale_shift((target - 1.0) / 7.0, target, np.ones_like(target, dtype=bool), "depth", 10.0)
    assert np.allclose(aligned, target, atol=1e-3)

