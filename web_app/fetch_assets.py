"""Fetch the large site files (video, scene data) that are published as GitHub release assets.

Files already present with the expected SHA-256 are left alone, so this is safe to run on every
build. Standard library only, so it runs inside the slim Docker image.

    python web_app/fetch_assets.py            # download missing or outdated files
    python web_app/fetch_assets.py --check    # exit 1 if any file is missing or differs
"""

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

RELEASE = "https://github.com/MdZeeshanAli01/VEO_V2/releases/download/site-assets-v2"
WEB_ROOT = Path(__file__).resolve().parent
ASSETS = {
    "veo_explainer.mp4": "a4decc3920ded3e2116f8232a7872544d2c479ff1fc2c0320c8f2510da5cf58d",
    "data/app_data_v2.js": "52e41dece1719031893dd5ff6d41415e73206837855057bd2586fb34a803d4cd",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def is_current(path: Path, expected: str) -> bool:
    return path.is_file() and sha256(path) == expected


def fetch(relative: str, expected: str) -> None:
    target = WEB_ROOT / relative
    partial = target.with_name(target.name + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"{RELEASE}/{Path(relative).name}"
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, partial)
    actual = sha256(partial)
    if actual != expected:
        partial.unlink()
        raise RuntimeError(f"Checksum mismatch for {relative}: expected {expected}, got {actual}")
    partial.replace(target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="only verify; do not download")
    args = parser.parse_args()
    stale = [name for name, digest in ASSETS.items() if not is_current(WEB_ROOT / name, digest)]
    if args.check:
        for name in stale:
            print(f"missing or outdated: {name}")
        return 1 if stale else 0
    for name in stale:
        fetch(name, ASSETS[name])
    print("Site assets ready" if not stale else f"Fetched {len(stale)} file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
