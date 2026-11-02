import io
import json
import sys
import tarfile
import types

import numpy as np
import pytest

from veo.diode import prepare_indoor
from veo.hypersim import SILVER_GROUPS
from veo.inference import TransformersDepthPredictor
from veo.model_comparison import compare, image_outcomes
from veo.silver import region_silver, silver_maps
from veo.summary import _model_comparison


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
