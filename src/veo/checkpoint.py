"""Checkpointing so long runs resume after a crash, a lost connection or a shutdown.

Everything is written atomically (to a temporary file, then renamed), so an interruption can never
leave a half-written file that would later be mistaken for finished work. Checkpoints carry a key
(a hash of the settings that produced them); a checkpoint with a different key is ignored.
"""

import json
import os
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.npy")
    np.save(temporary, array)
    temporary.replace(path)


def load_npy(path: Path) -> np.ndarray | None:
    """A cached array, or None when missing or unreadable (e.g. written by a pre-checkpoint version and cut short)."""
    if not path.exists():
        return None
    try:
        return np.load(path)
    except (ValueError, OSError, EOFError):
        return None


class RecordStore:
    """One JSON file per finished item; items saved under another key are treated as missing."""

    def __init__(self, directory: Path, key: str):
        self.directory, self.key = directory, key
        directory.mkdir(parents=True, exist_ok=True)

    def _path(self, item: str) -> Path:
        return self.directory / f"{item}.json"

    def get(self, item: str) -> dict | None:
        path = self._path(item)
        if not path.exists():
            return None
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        return stored["value"] if stored.get("key") == self.key else None

    def put(self, item: str, value) -> None:
        atomic_write_text(self._path(item), json.dumps({"key": self.key, "value": value}, default=str))


class JsonlLog:
    """Append-only log of finished items; a line cut off by a crash is ignored on reload."""

    def __init__(self, path: Path, key: str):
        self.path, self.key = path, key
        path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, dict]:
        done = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("key") == self.key:
                    done[entry["item"]] = entry["value"]
        return done

    def append(self, item: str, value) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"key": self.key, "item": item, "value": value}, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def stage_done(directory: Path, name: str, key: str) -> bool:
    marker = directory / f"{name}.done"
    if not marker.exists():
        return False
    try:
        return json.loads(marker.read_text(encoding="utf-8")).get("key") == key
    except (json.JSONDecodeError, OSError):
        return False


def mark_done(directory: Path, name: str, key: str) -> None:
    atomic_write_text(directory / f"{name}.done", json.dumps({"key": key}))


def run_stage(directory: Path, name: str, key: str, work: Callable[[], None], log: Callable[[str], None] = print) -> None:
    """Run ``work`` unless this stage already finished with the same key."""
    if stage_done(directory, name, key):
        log(f"      (resumed) {name} already done")
        return
    work()
    mark_done(directory, name, key)


def resumed(items: list[str], done: set[str]) -> Iterator[str]:
    """Items still to do, with a one-line note when a previous run had finished some."""
    remaining = [item for item in items if item not in done]
    if len(remaining) < len(items):
        print(f"      resuming: {len(items) - len(remaining)} of {len(items)} already done", flush=True)
    yield from remaining
