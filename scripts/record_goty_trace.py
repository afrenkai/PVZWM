from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from pvzwm.portable import PortableBridgeClient, PortableProcessConfig
from pvzwm.portable_trace import (
    canonical_snapshot,
    expand_tape_command,
    first_snapshot_difference,
    snapshot_digest,
)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"{path}:{line_number}: expected a JSON object")
        records.append(value)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record or verify tick-level GOTY bridge conformance traces"
    )
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--tape", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--float-tolerance", type=float, default=0.0)
    parser.add_argument("--no-xvfb", action="store_true")
    args = parser.parse_args()

    commands = [
        expanded
        for command in read_jsonl(args.tape)
        for expanded in expand_tape_command(command)
    ]
    references = read_jsonl(args.reference) if args.reference else None
    if references is not None and len(references) != len(commands):
        raise RuntimeError(
            f"Reference has {len(references)} records; tape expands to {len(commands)}"
        )

    save_directory = tempfile.TemporaryDirectory(prefix="pvzwm-goty-trace-")
    config = PortableProcessConfig(
        binary=args.binary,
        resources=args.resources,
        frame_width=0,
        frame_height=0,
        xvfb_run=None if args.no_xvfb else "xvfb-run",
        save_directory=Path(save_directory.name),
    )
    client = PortableBridgeClient(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    previous_snapshot: dict[str, object] | None = None
    try:
        with args.output.open("w") as output:
            for index, command in enumerate(commands):
                if command.get("cmd") == "step" and previous_snapshot is not None:
                    action = int(command.get("action", -1))
                    legal_actions = previous_snapshot.get("legal_actions")
                    if not isinstance(legal_actions, list) or action not in legal_actions:
                        raise RuntimeError(
                            f"Tape requests illegal action {action} at record {index}"
                        )
                snapshot = canonical_snapshot(client.request(command))
                record = {
                    "schema_version": 1,
                    "index": index,
                    "command": command,
                    "digest": snapshot_digest(snapshot),
                    "snapshot": snapshot,
                }
                if references is not None:
                    expected = references[index]
                    expected_snapshot = expected.get("snapshot")
                    difference = first_snapshot_difference(
                        expected_snapshot,
                        snapshot,
                        float_tolerance=args.float_tolerance,
                    )
                    if difference is not None:
                        raise RuntimeError(f"Trace diverged at record {index}: {difference}")
                output.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
                output.write("\n")
                previous_snapshot = snapshot
    finally:
        client.close()
        save_directory.cleanup()

    print(
        json.dumps(
            {
                "records": len(commands),
                "output": str(args.output),
                "reference_verified": references is not None,
            }
        )
    )


if __name__ == "__main__":
    main()
