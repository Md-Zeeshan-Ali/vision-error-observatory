"""Interrupted runs must resume from their checkpoints and give the same results."""

import json
from pathlib import Path

import pytest

from veo.checkpoint import (
    JsonlLog,
    RecordStore,
    atomic_write_text,
    load_npy,
    mark_done,
    run_stage,
    stage_done,
)
from veo.cli import run_with_retries
from veo.config import Config, DatasetConfig, ExperimentConfig, OutputConfig
from veo.data import load_depth
from veo.fixtures import SyntheticDisparityPredictor, write_fixture_mat, write_fixture_splits
from veo.nyu_dataset import extract_split, official_splits
from veo.reproduce import reproduce
from veo.runner import _pairs, run_pairs

REPO = Path(__file__).resolve().parents[1]


def test_atomic_write_leaves_no_temporary_file(tmp_path):
    atomic_write_text(tmp_path / "a.json", "{}")
    assert (tmp_path / "a.json").read_text() == "{}"
    assert [p.name for p in tmp_path.iterdir()] == ["a.json"]


def test_record_store_ignores_other_settings(tmp_path):
    RecordStore(tmp_path, "settings-a").put("img1", {"x": 1})
    assert RecordStore(tmp_path, "settings-a").get("img1") == {"x": 1}
    assert RecordStore(tmp_path, "settings-b").get("img1") is None


def test_jsonl_log_ignores_a_line_cut_off_by_a_crash(tmp_path):
    log = JsonlLog(tmp_path / "run.checkpoint.jsonl", "k")
    log.append("a", 1)
    log.append("b", 2)
    with (tmp_path / "run.checkpoint.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"key": "k", "item": "c", "val')  # power cut mid-write
    assert JsonlLog(tmp_path / "run.checkpoint.jsonl", "k").load() == {"a": 1, "b": 2}


def test_truncated_array_is_treated_as_missing(tmp_path):
    (tmp_path / "p.npy").write_bytes(b"\x93NUMPY truncated")
    assert load_npy(tmp_path / "p.npy") is None


def test_stage_runs_once_per_settings(tmp_path):
    calls = []
    run_stage(tmp_path, "s", "k1", lambda: calls.append(1), log=lambda message: None)
    run_stage(tmp_path, "s", "k1", lambda: calls.append(2), log=lambda message: None)
    run_stage(tmp_path, "s", "k2", lambda: calls.append(3), log=lambda message: None)
    assert calls == [1, 3]
    mark_done(tmp_path, "t", "k")
    assert stage_done(tmp_path, "t", "k") and not stage_done(tmp_path, "t", "other")


@pytest.fixture
def split_config(tmp_path):
    mat = write_fixture_mat(tmp_path / "nyu.mat", scenes=6)
    records = official_splits(mat, write_fixture_splits(tmp_path / "splits.mat", scenes=6, test_scenes=3))["test"]
    split = tmp_path / "test"
    extract_split(mat, records, split)
    (split / "manifest.txt").write_text(
        "".join(f"{split / 'rgb' / r['sample_id']}.png {split / 'depth' / r['sample_id']}.npy\n" for r in records), encoding="utf-8"
    )
    return Config(
        dataset=DatasetConfig(rgb_dir=split / "rgb", depth_dir=split / "depth", manifest=split / "manifest.txt", raw_depth_dir=split / "raw_depth"),
        experiment=ExperimentConfig(align_scale_shift=True, prediction_space="inverse_depth", max_prediction_m=10.0),
        outputs=OutputConfig(directory=tmp_path / "out"),
    )


def test_interrupted_analysis_resumes_without_recomputing(split_config):
    pairs = _pairs(split_config)
    calls = []

    def predict(pair, rgb, shape, fail_after=None):
        if fail_after is not None and len(calls) >= fail_after:
            raise ConnectionError("simulated dropped connection")
        calls.append(pair.sample_id)
        # Deterministic per image, so a fresh run and a resumed run see identical predictions
        return SyntheticDisparityPredictor(sum(map(ord, pair.sample_id))).predict(rgb, load_depth(pair.depth_path))

    with pytest.raises(ConnectionError):
        run_pairs(split_config, pairs, lambda p, r, s: predict(p, r, s, fail_after=3), "model_results.json", {}, save_predictions=True)
    assert len(calls) == 3
    first_three = list(calls)
    resumed = run_pairs(split_config, pairs, predict, "model_results.json", {}, save_predictions=True)
    assert len(resumed) == len(pairs)
    assert set(calls[3:]).isdisjoint(first_three)  # finished images were not predicted again
    fresh = run_pairs(split_config, pairs, predict, "model_results.json", {}, save_predictions=True, resume=False)
    assert json.dumps(fresh, default=str) == json.dumps(resumed, default=str)


def test_extraction_resumes_and_skips_finished_frames(tmp_path):
    mat = write_fixture_mat(tmp_path / "nyu.mat", scenes=3)
    records = official_splits(mat, write_fixture_splits(tmp_path / "splits.mat", scenes=3, test_scenes=1))["train"]
    extract_split(mat, records, tmp_path / "train")
    first = tmp_path / "train" / "depth" / f"{records[0]['sample_id']}.npy"
    before = first.stat().st_mtime_ns
    (tmp_path / "train" / "rgb" / f"{records[-1]['sample_id']}.png").unlink()  # as if cut off mid-split
    extract_split(mat, records, tmp_path / "train")
    assert first.stat().st_mtime_ns == before
    assert (tmp_path / "train" / "rgb" / f"{records[-1]['sample_id']}.png").exists()


def test_second_reproduce_resumes_every_stage(tmp_path, capsys):
    config = REPO / "configs" / "nyu.yaml"
    first = reproduce(tmp_path, config, fixture=True).read_text(encoding="utf-8")
    capsys.readouterr()
    second = reproduce(tmp_path, config, fixture=True).read_text(encoding="utf-8")
    output = capsys.readouterr().out
    assert second == first
    assert output.count("already done") >= 5 and "from checkpoints" in output
    reproduce(tmp_path, config, fixture=True, fresh=True)
    assert "already done" not in capsys.readouterr().out


def test_retries_resume_after_transient_errors():
    attempts = []

    def flaky(attempt):
        attempts.append(attempt)
        if attempt < 3:
            raise ConnectionError("wifi dropped")

    run_with_retries(flaky, retries=3, wait_seconds=0)
    assert attempts == [1, 2, 3]
    with pytest.raises(ConnectionError):
        run_with_retries(lambda attempt: (_ for _ in ()).throw(ConnectionError("down")), retries=1, wait_seconds=0)
