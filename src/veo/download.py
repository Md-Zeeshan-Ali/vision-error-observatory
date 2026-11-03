"""Resumable HTTP downloads for the public datasets (standard library only)."""

import io
import time
import urllib.error
import urllib.request
from pathlib import Path

NYU_LABELED_URL = "https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
NYU_SPLITS_URL = "https://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat"
USER_AGENT = "veo-dataset-downloader/1.0"


def _remote_size(url: str) -> int | None:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        length = response.headers.get("Content-Length")
    return int(length) if length else None


def download_file(url: str, destination: Path, retries: int = 20, chunk_size: int = 1 << 20) -> Path:
    """Download ``url`` to ``destination``, resuming a ``.part`` file across dropped connections."""
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    total = _remote_size(url)
    for attempt in range(1, retries + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        if total is not None and offset >= total:
            break
        headers = {"User-Agent": USER_AGENT}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as response:
                if offset and response.status != 206:
                    offset = 0  # server ignored the range; start over
                with partial.open("ab" if offset else "wb") as handle:
                    last_report = time.monotonic()
                    while chunk := response.read(chunk_size):
                        handle.write(chunk)
                        if total and time.monotonic() - last_report > 30:
                            print(f"  {handle.tell() / total:.1%} of {total / 1e9:.2f} GB", flush=True)
                            last_report = time.monotonic()
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as error:
            print(f"  attempt {attempt}/{retries} interrupted at {partial.stat().st_size if partial.exists() else 0} bytes: {error}", flush=True)
            time.sleep(min(5 * attempt, 60))
            continue
        if total is None or partial.stat().st_size >= total:
            break
    size = partial.stat().st_size if partial.exists() else 0
    if total is not None and size != total:
        raise RuntimeError(f"Incomplete download of {url}: {size} of {total} bytes")
    partial.replace(destination)
    return destination


class HttpRangeFile(io.RawIOBase):
    """Read-only, seekable view of a remote file using HTTP range requests.

    Wrap it in ``io.BufferedReader`` and pass it to ``zipfile.ZipFile`` to read single members of a
    large remote archive without downloading the whole archive.
    """

    def __init__(self, url: str, retries: int = 5):
        super().__init__()
        self.url, self.retries = url, retries
        size = _remote_size(url)
        if size is None:
            raise OSError(f"Server did not report a size for {url}")
        self.size, self.position = size, 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}[whence]
        self.position = max(0, base + offset)
        return self.position

    def readinto(self, buffer) -> int:
        if self.position >= self.size or len(buffer) == 0:
            return 0
        end = min(self.position + len(buffer), self.size) - 1
        request = urllib.request.Request(self.url, headers={"User-Agent": USER_AGENT, "Range": f"bytes={self.position}-{end}"})
        for attempt in range(1, self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    data = response.read()
                break
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
                if attempt == self.retries:
                    raise
                time.sleep(2 * attempt)
        buffer[: len(data)] = data
        self.position += len(data)
        return len(data)
