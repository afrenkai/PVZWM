from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manifest local-only PVZWM research artifacts"
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--remote-repo", required=True)
    parser.add_argument("--roots", nargs="+", default=["artifacts", "data/raw"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    files = sorted(
        path
        for root in args.roots
        for path in (project_root / root).rglob("*")
        if path.is_file()
        and not path.name.endswith((".duckdb", ".duckdb.wal"))
    )
    entries = []
    for index, path in enumerate(files, start=1):
        relative = path.relative_to(project_root).as_posix()
        entries.append(
            {
                "path": relative,
                "remote_uri": f"hf://datasets/{args.remote_repo}/{relative}",
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "git_blob_sha1": git_blob_sha1(path),
            }
        )
        if index == 1 or index % 100 == 0 or index == len(files):
            print(f"[{index}/{len(files)}] {relative}")

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "remote_repo": args.remote_repo,
        "source_roots": args.roots,
        "file_count": len(entries),
        "file_bytes": sum(entry["bytes"] for entry in entries),
        "files": entries,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
