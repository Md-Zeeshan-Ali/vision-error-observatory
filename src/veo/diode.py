"""DIODE indoor validation frames (laser-scanned depth), prepared in the layout the pipeline reads.

The archive holds indoor and outdoor frames as ``val/indoors/scene_x/scan_y/<id>.png``, ``<id>_depth.npy``
and ``<id>_depth_mask.npy``. Only indoor frames are kept. Pixels without a laser return become 0,
which the pipeline treats as unmeasured. Finished frames are skipped, so preparation can resume.
"""

import io
import json
import tarfile
from pathlib import Path

import numpy as np

from .checkpoint import atomic_save_npy, atomic_write_text
from .download import download_file

DIODE_VAL_URL = "http://diode-dataset.s3.amazonaws.com/val.tar.gz"


def prepare_indoor(archive: Path, output_dir: Path) -> Path:
    """Write rgb/, depth/, metadata.json and manifest.txt from the archive; returns the output folder."""
    if (output_dir / "manifest.txt").exists():
        return output_dir
    rgb_dir, depth_dir = output_dir / "rgb", output_dir / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    pending: dict[str, dict[str, bytes]] = {}
    records = []
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            parts = Path(member.name).parts
            if not member.isfile() or "indoors" not in parts:
                continue
            name = parts[-1]
            kind = "mask" if name.endswith("_depth_mask.npy") else "depth" if name.endswith("_depth.npy") else "rgb" if name.endswith(".png") else None
            if kind is None:
                continue
            sample_id = name.removesuffix("_depth_mask.npy").removesuffix("_depth.npy").removesuffix(".png")
            # Contents are read as bytes and written to names we choose; archive paths are never used for writing
            files = pending.setdefault(sample_id, {})
            files[kind] = tar.extractfile(member).read()
            if len(files) < 3:
                continue
            del pending[sample_id]
            scene = next(part for part in parts if part.startswith("scene_"))
            records.append({"sample_id": sample_id, "scene": scene})
            rgb_path = rgb_dir / f"{sample_id}.png"
            if not rgb_path.exists():
                temporary = rgb_dir / f".{sample_id}.tmp"
                temporary.write_bytes(files["rgb"])
                temporary.replace(rgb_path)
            depth_path = depth_dir / f"{sample_id}.npy"
            if not depth_path.exists():
                depth = np.squeeze(np.load(io.BytesIO(files["depth"]))).astype(np.float32)
                mask = np.load(io.BytesIO(files["mask"])) > 0
                atomic_save_npy(depth_path, np.where(mask, depth, 0.0).astype(np.float32))
    records.sort(key=lambda record: record["sample_id"])
    atomic_write_text(output_dir / "metadata.json", json.dumps(records, indent=2))
    lines = [f"{rgb_dir / (record['sample_id'] + '.png')} {depth_dir / (record['sample_id'] + '.npy')}" for record in records]
    atomic_write_text(output_dir / "manifest.txt", "\n".join(lines) + "\n")
    return output_dir


def ensure_indoor(data_dir: Path, keep_archive: bool = False) -> Path:
    """Download (resumable) and prepare the indoor frames, then free the archive's disk space."""
    prepared = data_dir / "val_indoors"
    if (prepared / "manifest.txt").exists():
        return prepared
    archive = download_file(DIODE_VAL_URL, data_dir / "val.tar.gz")
    prepare_indoor(archive, prepared)
    if not keep_archive:
        archive.unlink()
    return prepared
