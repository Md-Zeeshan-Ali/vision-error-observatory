"""Hypersim frames for the reflection test: the final render with and without its non-diffuse light.

Hypersim stores each rendered frame as HDR layers: ``color = diffuse_reflectance * diffuse_illumination +
residual``, where the residual holds non-diffuse (mostly specular) light. Removing it gives the same
scene without reflections, a counterfactual that real photos cannot provide.

Files are read from the public per-scene zips with HTTP range requests, so only the needed members
are downloaded (about 6 MB per frame instead of gigabytes per scene).
"""

import csv
import io
import zipfile
from pathlib import Path

import h5py
import numpy as np

from .data import load_depth
from .download import HttpRangeFile, download_file

SCENE_URL = "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/{scene}.zip"
SPLIT_CSV_URL = "https://raw.githubusercontent.com/apple/ml-hypersim/main/evermotion_dataset/analysis/metadata_images_split_scene_v1.csv"
CAMERA_CSV_URL = "https://raw.githubusercontent.com/apple/ml-hypersim/main/contrib/mikeroberts3000/metadata_camera_parameters.csv"
FINAL = ("color", "residual")
GEOMETRY = ("depth_meters", "semantic", "semantic_instance")
# NYU40 ids used by Hypersim's semantic labels
REFLECTIVE_NYU40 = {9: "window", 19: "mirror", 25: "television", 30: "whiteboard"}
# The same class groups as veo.silver, in NYU40 ids: wall, floor, ceiling, floor mat; wall, ceiling
SILVER_GROUPS = {"reflective": set(REFLECTIVE_NYU40), "structure": {1, 2, 22, 20}, "flat": {1, 22}}


def standard_camera_scenes(camera_csv: Path) -> set[str]:
    """Scenes rendered with Hypersim's default 60-degree, unshifted 1024x768 camera (focal 886.81 px)."""
    scenes = set()
    with camera_csv.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (abs(float(row["M_proj_00"]) - 1.7320508) < 1e-4 and abs(float(row["M_proj_02"])) < 1e-6
                    and abs(float(row["M_proj_12"])) < 1e-6 and float(row["settings_output_img_width"]) == 1024.0):
                scenes.add(row["scene_name"])
    return scenes


def select_frames(split_csv: Path, camera_csv: Path, split: str = "test", per_scene: int = 3, seed: int = 42) -> list[tuple[str, str, int]]:
    """Seeded sample of public-release frames from standard-camera scenes of one official split."""
    standard = standard_camera_scenes(camera_csv)
    by_scene: dict[str, list[tuple[str, int]]] = {}
    with split_csv.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["included_in_public_release"] == "True" and row["split_partition_name"] == split and row["scene_name"] in standard:
                by_scene.setdefault(row["scene_name"], []).append((row["camera_name"], int(row["frame_id"])))
    rng = np.random.default_rng(seed)
    frames = []
    for scene in sorted(by_scene):
        options = by_scene[scene]
        for index in sorted(rng.choice(len(options), min(per_scene, len(options)), replace=False)):
            frames.append((scene, *options[index]))
    return frames


def frame_path(cache: Path, scene: str, camera: str, frame: int, kind: str) -> Path:
    return cache / scene / camera / f"frame.{frame:04d}.{kind}.hdf5"


def fetch_frames(frames: list[tuple[str, str, int]], cache: Path) -> None:
    """Download the needed layers of each frame (skipping files already cached)."""
    by_scene: dict[str, list[tuple[str, int]]] = {}
    for scene, camera, frame in frames:
        by_scene.setdefault(scene, []).append((camera, frame))
    for scene, wanted in sorted(by_scene.items()):
        missing = [(camera, frame, kind) for camera, frame in wanted for kind in FINAL + GEOMETRY
                   if not frame_path(cache, scene, camera, frame, kind).exists()]
        if not missing:
            continue
        archive = zipfile.ZipFile(io.BufferedReader(HttpRangeFile(SCENE_URL.format(scene=scene)), buffer_size=1 << 20))
        for camera, frame, kind in missing:
            folder = "final_hdf5" if kind in FINAL else "geometry_hdf5"
            member = f"{scene}/images/scene_{camera}_{folder}/frame.{frame:04d}.{kind}.hdf5"
            target = frame_path(cache, scene, camera, frame, kind)
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".part")
            partial.write_bytes(archive.read(member))
            partial.replace(target)
        print(f"  fetched {scene} ({len(missing)} files)", flush=True)


def ensure_metadata(cache: Path) -> tuple[Path, Path]:
    return (download_file(SPLIT_CSV_URL, cache / "metadata_images_split_scene_v1.csv"),
            download_file(CAMERA_CSV_URL, cache / "metadata_camera_parameters.csv"))


def _read(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        return np.asarray(handle["dataset"], dtype=np.float32)


def tonemap(color: np.ndarray, scale: float | None = None) -> tuple[np.ndarray, float]:
    """Hypersim's tone mapping: scale so the 90th-percentile brightness maps to 0.8 after gamma 1/2.2.

    Pass the ``scale`` from the original render to tone-map an edited render at the same exposure.
    """
    finite = np.nan_to_num(color, nan=0.0, posinf=0.0, neginf=0.0)
    if scale is None:
        brightness = 0.3 * finite[..., 0] + 0.59 * finite[..., 1] + 0.11 * finite[..., 2]
        current = float(np.percentile(brightness, 90))
        scale = 0.8 ** 2.2 / current if current > 1e-4 else 0.0
    mapped = np.power(np.maximum(scale * finite, 0.0), 1.0 / 2.2)
    return (np.clip(mapped, 0.0, 1.0) * 255.0).round().astype(np.uint8), scale


def load_frame(cache: Path, scene: str, camera: str, frame: int) -> dict:
    """RGB with and without non-diffuse light (same exposure), planar depth, NYU40 semantics, specular map."""
    color = _read(frame_path(cache, scene, camera, frame, "color"))
    residual = _read(frame_path(cache, scene, camera, frame, "residual"))
    rgb, scale = tonemap(color)
    rgb_diffuse, _ = tonemap(color - residual, scale)
    depth = load_depth(frame_path(cache, scene, camera, frame, "depth_meters"))
    with h5py.File(frame_path(cache, scene, camera, frame, "semantic"), "r") as handle:
        semantic = np.asarray(handle["dataset"], dtype=np.int32)
    with h5py.File(frame_path(cache, scene, camera, frame, "semantic_instance"), "r") as handle:
        instance = np.asarray(handle["dataset"], dtype=np.int32)
    specular = np.abs(rgb.astype(np.float32) - rgb_diffuse.astype(np.float32)).mean(axis=2)
    return {"rgb": rgb, "rgb_diffuse": rgb_diffuse, "depth": depth, "semantic": semantic, "instance": instance, "specular": specular}
