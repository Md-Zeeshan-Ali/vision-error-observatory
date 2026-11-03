"""iBims-1 core set (100 laser-scanned indoor images), prepared in the layout the pipeline reads.

The archive holds one ``<room>_<number>.mat`` per image with RGB, depth, and masks for invalid and
transparent pixels, annotated depth edges and selected wall, floor and table planes. The room type
is the only scene information, so it is used as the scene. Masks are saved per image for the silver
labels defined in docs/MODEL_COMPARISON_PROTOCOL.md.
"""

import hashlib
import io
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np

from .checkpoint import atomic_save_npy, atomic_write_text
from .download import download_file
from .silver import SilverParams, _disk

IBIMS_URL = "https://dataserv.ub.tum.de/s/m1455541/download?path=%2F&files=ibims1_core_mat.zip"
IBIMS_SHA256 = "459a9ae5b6969d00fd18f4c87e9927fa8bdea55319bdb3901cb68f99b8fdb36e"


def prepare_core(archive: Path, output_dir: Path) -> Path:
    """Write rgb/, depth/, masks/, metadata.json and manifest.txt from the archive; returns the output folder."""
    if (output_dir / "manifest.txt").exists():
        return output_dir
    from PIL import Image
    from scipy.io import loadmat

    rgb_dir, depth_dir, mask_dir = output_dir / "rgb", output_dir / "depth", output_dir / "masks"
    for folder in (rgb_dir, depth_dir, mask_dir):
        folder.mkdir(parents=True, exist_ok=True)
    records = []
    with zipfile.ZipFile(archive) as bundle:
        for name in sorted(item for item in bundle.namelist() if item.endswith(".mat")):
            # Contents are read as bytes and written to names we choose; archive paths are never used for writing
            data = loadmat(io.BytesIO(bundle.read(name)), squeeze_me=True, struct_as_record=False)["data"]
            sample_id = Path(name).stem
            edges = np.asarray(data.edges) > 0
            depth = np.nan_to_num(np.asarray(data.depth, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            depth[np.asarray(data.mask_invalid) == 0] = 0.0
            Image.fromarray(np.asarray(data.rgb, dtype=np.uint8)).save(rgb_dir / f"{sample_id}.png")
            atomic_save_npy(depth_dir / f"{sample_id}.npy", depth)
            np.savez_compressed(
                mask_dir / f"{sample_id}.npz", edges=edges, transparent=np.asarray(data.mask_transp) == 0,
                wall=np.asarray(data.mask_wall) > 0, floor=np.asarray(data.mask_floor) > 0, table=np.asarray(data.mask_table) > 0,
            )
            records.append({"sample_id": sample_id, "scene": sample_id.rsplit("_", 1)[0], "has_edges": bool(edges.any())})
    atomic_write_text(output_dir / "metadata.json", json.dumps(records, indent=2))
    lines = [f"{rgb_dir / (record['sample_id'] + '.png')} {depth_dir / (record['sample_id'] + '.npy')}" for record in records]
    atomic_write_text(output_dir / "manifest.txt", "\n".join(lines) + "\n")
    return output_dir


def ensure_core(data_dir: Path) -> Path:
    """Download (resumable), verify and prepare the core set."""
    prepared = data_dir / "core"
    if (prepared / "manifest.txt").exists():
        return prepared
    archive = data_dir / "ibims1_core_mat.zip"
    if not archive.exists():
        download_file(IBIMS_URL, archive)
    actual = hashlib.sha256(archive.read_bytes()).hexdigest()
    if actual != IBIMS_SHA256:
        raise RuntimeError(f"Checksum mismatch for {archive}: expected {IBIMS_SHA256}, got {actual}")
    return prepare_core(archive, prepared)


def silver_maps(masks, params: SilverParams | None = None) -> dict[str, np.ndarray]:
    """Per-pixel support in the form ``veo.silver.region_silver`` expects, plus ``planar`` for the sensitivity check."""
    params = params or SilverParams()
    edges = masks["edges"].astype(np.uint8)
    return {
        "reflection": masks["transparent"],
        "thin_object": np.zeros(edges.shape, dtype=bool),  # not annotated, and not reported
        "boundary_mixing": cv2.dilate(edges, _disk(params.boundary_radius)) > 0,
        "textureless": masks["wall"],
        "planar": masks["wall"] | masks["floor"] | masks["table"],
        "annotated": np.ones(edges.shape, dtype=bool),
    }
