#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from pvzwm.goty_reference import (
    ACCEPTANCE_RESULTS,
    build_evidence_record,
    record_acceptance_result,
    write_json_atomic,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record one original-GOTY black-box comparison result."
    )
    parser.add_argument("plan", type=Path)
    parser.add_argument("scenario")
    parser.add_argument("--result", choices=sorted(ACCEPTANCE_RESULTS), required=True)
    parser.add_argument("--reference-artifact", action="append", type=Path, default=[])
    parser.add_argument("--candidate-artifact", action="append", type=Path, default=[])
    parser.add_argument("--notes")
    args = parser.parse_args()

    plan_path = args.plan.expanduser().resolve()
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("Acceptance plan must contain a JSON object.")
    evidence_root = plan_path.parent
    reference_artifacts = [
        build_evidence_record(path, evidence_root=evidence_root)
        for path in args.reference_artifact
    ]
    candidate_artifacts = [
        build_evidence_record(path, evidence_root=evidence_root)
        for path in args.candidate_artifact
    ]
    updated = record_acceptance_result(
        payload,
        args.scenario,
        args.result,
        reference_artifacts=reference_artifacts,
        candidate_artifacts=candidate_artifacts,
        notes=args.notes,
    )
    write_json_atomic(plan_path, updated)

    scenarios = updated["scenarios"]
    counts = Counter(scenario["result"] for scenario in scenarios)
    print(
        json.dumps(
            {
                "plan": str(args.plan),
                "scenario": args.scenario,
                "result": args.result,
                "counts": dict(sorted(counts.items())),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
