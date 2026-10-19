"""Fetch unchanged official archives, with published size and MD5 validation."""

import argparse
import hashlib
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

METADATA = json.loads(Path(__file__).with_name("organamnist.json").read_text(encoding="utf-8"))


def file_md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_asset(asset: dict, data_dir: Path) -> Path:
    """Only promote a completed, checksum-verified file into the dataset cache."""
    data_dir.mkdir(parents=True, exist_ok=True)
    destination = data_dir / asset["filename"]
    if destination.exists():
        if destination.stat().st_size != asset["bytes"] or file_md5(destination) != asset["md5"]:
            raise ValueError(f"Checksum mismatch in cached file: {destination}")
        print(f"Verified cache: {destination.name}", flush=True)
        return destination
    partial = destination.with_suffix(".npz.part")
    request = urllib.request.Request(
        asset["url"], headers={"User-Agent": "ct-mae-dataset-explorer/0.1"}
    )
    digest = hashlib.md5(usedforsecurity=False)
    received = 0
    last_update = time.monotonic()
    print(f"Downloading {destination.name}: {asset['bytes'] / 1e6:.1f} MB", flush=True)
    with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as stream:
        while chunk := response.read(1024 * 1024):
            stream.write(chunk)
            digest.update(chunk)
            received += len(chunk)
            if time.monotonic() - last_update >= 5:
                print(f"  {destination.name}: {received / asset['bytes']:.0%}", flush=True)
                last_update = time.monotonic()
    if received != asset["bytes"] or digest.hexdigest() != asset["md5"]:
        raise ValueError(
            f"Checksum mismatch in download: {destination.name}; partial file retained"
        )
    partial.replace(destination)
    print(f"Verified download: {destination.name}", flush=True)
    return destination


def official_asset(size: int) -> dict:
    asset = dict(METADATA["assets"][str(size)])
    asset["url"] = f"{METADATA['record']}/files/{asset['filename']}?download=1"
    return asset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--sizes", type=int, nargs="+", choices=[28, 64, 128, 224], default=[28, 64]
    )
    args = parser.parse_args()
    sizes = sorted(set(args.sizes))
    with ThreadPoolExecutor(max_workers=min(3, len(sizes))) as pool:
        jobs = [pool.submit(fetch_asset, official_asset(size), args.data_dir) for size in sizes]
        for job in jobs:
            job.result()


if __name__ == "__main__":
    main()
