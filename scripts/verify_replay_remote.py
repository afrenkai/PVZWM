from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify remote replay objects against a snapshot manifest"
    )
    parser.add_argument("manifest", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    repo_id = manifest["remote_repo"]
    expected = {entry["path"]: entry for entry in manifest["databases"]}
    info = HfApi().repo_info(
        repo_id,
        repo_type="dataset",
        files_metadata=True,
    )
    remote = {sibling.rfilename: sibling for sibling in info.siblings}

    missing = sorted(set(expected) - set(remote))
    unexpected = sorted(
        path for path in remote if path.endswith(".duckdb") and path not in expected
    )
    size_mismatches: list[str] = []
    hash_mismatches: list[str] = []
    missing_hashes: list[str] = []
    for path, entry in expected.items():
        sibling = remote.get(path)
        if sibling is None:
            continue
        if sibling.size != entry["bytes"]:
            size_mismatches.append(path)
        remote_hash = sibling.lfs.sha256 if sibling.lfs else None
        if remote_hash is None:
            missing_hashes.append(path)
        elif remote_hash != entry["sha256"]:
            hash_mismatches.append(path)

    result = {
        "repo": repo_id,
        "revision": info.sha,
        "private": info.private,
        "expected_databases": len(expected),
        "remote_databases": sum(path.endswith(".duckdb") for path in remote),
        "missing": missing,
        "unexpected": unexpected,
        "size_mismatches": size_mismatches,
        "hash_mismatches": hash_mismatches,
        "missing_hashes": missing_hashes,
    }
    print(json.dumps(result, indent=2))
    if any(
        result[key]
        for key in (
            "missing",
            "unexpected",
            "size_mismatches",
            "hash_mismatches",
            "missing_hashes",
        )
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
