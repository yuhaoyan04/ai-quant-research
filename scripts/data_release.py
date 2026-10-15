"""Package or restore the exact research lake as chunked GitHub Release assets.

Archives are uncompressed tar streams because the Parquet payload is already
Zstandard-compressed.  Streams are split below GitHub's 2 GiB per-asset limit,
so packaging never needs a second full copy of the 11+ GiB lake on disk.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
from typing import BinaryIO


RELEASE_VERSION = "data-v1"
CHUNK_BYTES = 1_500_000_000
LARGE_TABLES = {
    "factors_raw": ["data/lake/curated/factors_raw"],
    "eligible_panel": ["data/lake/curated/eligible_factor_panel_staging"],
    "multi_horizon_panel": ["data/lake/curated/multi_horizon_factor_panel"],
    "validated_bars": ["data/lake/curated/daily_bars_validated"],
    "model_features": ["data/lake/curated/model_features"],
    "baostock_bars": ["data/lake/curated/daily_bars_baostock"],
    "labels_raw": ["data/lake/curated/labels_raw"],
    "multi_horizon_labels": ["data/lake/curated/multi_horizon_labels_staging"],
}


class ChunkWriter(io.RawIOBase):
    def __init__(self, output: Path, prefix: str, chunk_bytes: int) -> None:
        self.output = output
        self.prefix = prefix
        self.chunk_bytes = chunk_bytes
        self.output.mkdir(parents=True, exist_ok=True)
        self.index = 0
        self.current: BinaryIO | None = None
        self.current_size = 0
        self.current_hash = hashlib.sha256()
        self.assets: list[dict[str, object]] = []

    def writable(self) -> bool:
        return True

    def tell(self) -> int:
        return sum(int(item["bytes"]) for item in self.assets) + self.current_size

    def _open_next(self) -> None:
        self.index += 1
        path = self.output / f"{self.prefix}.part{self.index:02d}"
        self.current = path.open("wb")
        self.current_size = 0
        self.current_hash = hashlib.sha256()

    def _close_current(self) -> None:
        if self.current is None:
            return
        path = Path(self.current.name)
        self.current.close()
        self.assets.append({
            "name": path.name,
            "bytes": self.current_size,
            "sha256": self.current_hash.hexdigest(),
        })
        self.current = None

    def write(self, data: bytes | bytearray) -> int:
        view = memoryview(data)
        written = 0
        while view:
            if self.current is None:
                self._open_next()
            capacity = self.chunk_bytes - self.current_size
            block = view[:capacity]
            assert self.current is not None
            self.current.write(block)
            self.current_hash.update(block)
            size = len(block)
            self.current_size += size
            written += size
            view = view[size:]
            if self.current_size == self.chunk_bytes:
                self._close_current()
        return written

    def close(self) -> None:
        self._close_current()
        super().close()


class ConcatenatedReader(io.RawIOBase):
    def __init__(self, parts: list[Path]) -> None:
        self.parts = iter(parts)
        self.current: BinaryIO | None = None

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: bytearray) -> int:
        target = memoryview(buffer)
        total = 0
        while total < len(target):
            if self.current is None:
                try:
                    self.current = next(self.parts).open("rb")
                except StopIteration:
                    break
            count = self.current.readinto(target[total:])
            if count:
                total += count
            else:
                self.current.close()
                self.current = None
        return total

    def close(self) -> None:
        if self.current is not None:
            self.current.close()
        super().close()


def group_paths(repo: Path, group: str) -> list[Path]:
    if group in LARGE_TABLES:
        return [repo / item for item in LARGE_TABLES[group]]
    if group != "remaining":
        raise KeyError(f"unknown group: {group}")
    curated = repo / "data" / "lake" / "curated"
    excluded = {Path(item).name for paths in LARGE_TABLES.values() for item in paths}
    paths = [path for path in sorted(curated.iterdir()) if path.name not in excluded]
    paths.extend([
        repo / "data" / "lake" / "manifests",
        repo / "data" / "lake" / "monitoring",
        repo / "data" / "lake" / "checkpoints",
        repo / "data" / "logs",
    ])
    return [path for path in paths if path.exists()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(repo: Path, group: str, assets: list[dict[str, object]]) -> Path:
    path = repo / "data_release_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "release": RELEASE_VERSION,
        "format": "concatenated_uncompressed_tar_parts",
        "chunk_bytes": CHUNK_BYTES,
        "groups": {},
    }
    payload["groups"][group] = assets
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def pack(repo: Path, output: Path, group: str) -> None:
    paths = group_paths(repo, group)
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing data paths: {missing}")
    writer = ChunkWriter(output, f"ai-quant-{RELEASE_VERSION}-{group}.tar", CHUNK_BYTES)
    try:
        with tarfile.open(fileobj=writer, mode="w|") as archive:
            for path in paths:
                archive.add(path, arcname=path.relative_to(repo).as_posix(), recursive=True)
    finally:
        writer.close()
    assets = [{"group": group, "part": number, **item}
              for number, item in enumerate(writer.assets, start=1)]
    manifest = write_manifest(repo, group, assets)
    print(json.dumps({"group": group, "assets": assets,
                      "manifest": str(manifest)}, indent=2))


def verify_assets(asset_dir: Path, manifest: dict) -> None:
    for assets in manifest["groups"].values():
        for item in assets:
            path = asset_dir / item["name"]
            if not path.exists():
                raise FileNotFoundError(path)
            if path.stat().st_size != int(item["bytes"]):
                raise ValueError(f"size mismatch: {path}")
            actual = sha256_file(path)
            if actual != item["sha256"]:
                raise ValueError(f"SHA-256 mismatch: {path}")


def restore(repo: Path, asset_dir: Path, manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verify_assets(asset_dir, manifest)
    root = repo.resolve()
    for group, assets in manifest["groups"].items():
        parts = [asset_dir / item["name"] for item in sorted(assets, key=lambda x: x["part"])]
        with io.BufferedReader(ConcatenatedReader(parts), buffer_size=8 * 1024 * 1024) as stream:
            with tarfile.open(fileobj=stream, mode="r|") as archive:
                for member in archive:
                    destination = (root / member.name).resolve()
                    if root not in destination.parents and destination != root:
                        raise ValueError(f"unsafe archive member: {member.name}")
                    if member.issym() or member.islnk():
                        raise ValueError(f"links are not allowed: {member.name}")
                    archive.extract(member, path=root)
        print(f"restored {group}")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    pack_parser = subparsers.add_parser("pack")
    pack_parser.add_argument("--group", required=True,
                             choices=[*LARGE_TABLES, "remaining"])
    pack_parser.add_argument("--output", type=Path, default=Path("release_staging"))
    pack_parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    restore_parser = subparsers.add_parser("restore")
    restore_parser.add_argument("--asset-dir", type=Path, required=True)
    restore_parser.add_argument("--manifest", type=Path,
                                default=Path("data_release_manifest.json"))
    restore_parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    if args.command == "pack":
        pack(repo, args.output.resolve(), args.group)
    else:
        restore(repo, args.asset_dir.resolve(), args.manifest.resolve())


if __name__ == "__main__":
    main()
