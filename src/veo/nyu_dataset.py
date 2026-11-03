import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from .checkpoint import atomic_save_npy

# Optional per-frame layers of the labelled .mat, saved next to rgb/ and depth/ when present.
# rawDepths: sensor depth projected to the RGB frame (0 = not measured); labels: NYU class ids.
OPTIONAL_LAYERS = {"rawDepths": ("raw_depth", np.float32), "labels": ("labels", np.uint16), "instances": ("instances", np.uint8)}


def extract_labeled_mat(mat_path: Path, rgb_dir: Path, depth_dir: Path, start: int = 0, count: int | None = None) -> int:
    """Extract aligned RGB/depth pairs from the official NYU v2 labeled MAT file.

    The v7.3 MAT file stores MATLAB arrays in reversed HDF5 order. The official
    ``depths`` variable is the dense, in-painted depth in metres.
    """
    with h5py.File(mat_path, "r") as handle:
        total = int(handle["images"].shape[0])
        stop = total if count is None else min(total, start + count)
        if start < 0 or start >= total or start >= stop:
            raise ValueError(f"Invalid extraction range: start={start}, count={count}, total={total}")
        rgb_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)
        for index in range(start, stop):
            sample_id = f"scene_{index + 1:04d}"
            image = np.asarray(handle["images"][index], dtype=np.uint8).transpose(2, 1, 0)
            depth = np.asarray(handle["depths"][index], dtype=np.float32).transpose(1, 0)
            Image.fromarray(image, mode="RGB").save(rgb_dir / f"{sample_id}.png")
            np.save(depth_dir / f"{sample_id}.npy", depth)
    return stop - start


def _cell_strings(handle: h5py.File, key: str) -> list[str]:
    """Decode a MATLAB cell array of strings (stored as uint16 code units behind object references)."""
    references = np.asarray(handle[key][:]).reshape(-1)
    return [handle[reference][()].tobytes().decode("utf-16le").rstrip("\x00") for reference in references]


def _scene_names(handle: h5py.File) -> list[str]:
    return _cell_strings(handle, "scenes")


def create_scene_splits(mat_path: Path, seed: int = 42, train_fraction: float = 0.70, validation_fraction: float = 0.15) -> dict[str, list[dict]]:
    with h5py.File(mat_path, "r") as handle:
        scenes = _scene_names(handle)
    unique_scenes = sorted(set(scenes))
    generator = np.random.default_rng(seed)
    shuffled = unique_scenes.copy()
    generator.shuffle(shuffled)
    train_end = int(len(shuffled) * train_fraction)
    validation_end = train_end + int(len(shuffled) * validation_fraction)
    partitions = {
        "development": set(shuffled[:train_end]),
        "validation": set(shuffled[train_end:validation_end]),
        "test": set(shuffled[validation_end:]),
    }
    return {
        split: [{"index": index, "sample_id": f"scene_{index + 1:04d}", "scene": scene} for index, scene in enumerate(scenes) if scene in selected]
        for split, selected in partitions.items()
    }


def official_splits(mat_path: Path, splits_path: Path) -> dict[str, list[dict]]:
    """The official NYU Depth V2 train (795) / test (654) split from ``splits.mat`` (1-based indices)."""
    from scipy.io import loadmat

    split_indices = loadmat(splits_path)
    with h5py.File(mat_path, "r") as handle:
        scenes = _scene_names(handle)
    result = {}
    for split, key in (("train", "trainNdxs"), ("test", "testNdxs")):
        indices = sorted(int(value) - 1 for value in np.asarray(split_indices[key]).reshape(-1))
        result[split] = [{"index": index, "sample_id": f"scene_{index + 1:04d}", "scene": scenes[index]} for index in indices]
    shared = {record["scene"] for record in result["train"]} & {record["scene"] for record in result["test"]}
    if shared:
        raise ValueError(f"Official split shares scenes between train and test: {sorted(shared)[:5]}")
    return result


def extract_split(mat_path: Path, split_records: list[dict], output_dir: Path) -> int:
    """Write rgb/, depth/ (dense) and, when the file has them, raw_depth/, labels/ and instances/.

    Frames whose files all exist are skipped, so an interrupted extraction continues where it stopped.
    Each file is written to a temporary name first, so a cut-off write is never mistaken for a finished one.
    """
    rgb_dir = output_dir / "rgb"
    depth_dir = output_dir / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(mat_path, "r") as handle:
        layers = {key: spec for key, spec in OPTIONAL_LAYERS.items() if key in handle}
        for directory, _ in layers.values():
            (output_dir / directory).mkdir(exist_ok=True)
        if "names" in handle:
            (output_dir / "label_names.json").write_text(json.dumps(_cell_strings(handle, "names")), encoding="utf-8")
        for record in split_records:
            index, sample_id = record["index"], record["sample_id"]
            targets = [rgb_dir / f"{sample_id}.png", depth_dir / f"{sample_id}.npy"]
            targets += [output_dir / directory / f"{sample_id}.npy" for directory, _ in layers.values()]
            if all(path.exists() for path in targets):
                continue
            image = np.asarray(handle["images"][index], dtype=np.uint8).transpose(2, 1, 0)
            temporary = rgb_dir / f".{sample_id}.tmp.png"
            Image.fromarray(image, mode="RGB").save(temporary)
            temporary.replace(rgb_dir / f"{sample_id}.png")
            atomic_save_npy(depth_dir / f"{sample_id}.npy", np.asarray(handle["depths"][index], dtype=np.float32).transpose(1, 0))
            for key, (directory, dtype) in layers.items():
                atomic_save_npy(output_dir / directory / f"{sample_id}.npy", np.asarray(handle[key][index], dtype=dtype).transpose(1, 0))
    return len(split_records)
