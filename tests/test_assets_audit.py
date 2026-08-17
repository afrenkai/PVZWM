import json
from pathlib import Path

import numpy as np

import pvzwm.data.audit as audit_module
from pvzwm.assets import parse_mediawiki_assets
from pvzwm.data.audit import audit_data
from pvzwm.data.store import ReplayStore, Transition


def test_parse_wiki_asset_metadata() -> None:
    payload = {
        "query": {
            "pages": {
                "1": {
                    "title": "File:Peashooter1.png",
                    "imageinfo": [
                        {
                            "size": 12,
                            "width": 96,
                            "height": 96,
                            "url": "https://example.test/Peashooter1.png",
                            "descriptionurl": "https://example.test/wiki/File:Peashooter1.png",
                            "sha1": "abc",
                            "extmetadata": {
                                "LicenseShortName": {"value": "Fair use"},
                            },
                        }
                    ],
                }
            }
        }
    }
    assets = parse_mediawiki_assets(json.dumps(payload).encode())
    assert len(assets) == 1
    assert assets[0].filename == "Peashooter1.png"
    assert assets[0].license_short_name == "Fair use"


def test_data_audit_separates_synthetic_and_original(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(audit_module, "discover_macos_pvz", list)
    data = tmp_path / "data"
    data.mkdir()
    transition = Transition(
        np.zeros(2, dtype=np.float32),
        0,
        0.0,
        np.ones(2, dtype=np.float32),
        False,
    )
    with ReplayStore(data / "mixed.duckdb") as store:
        with store.episode("symbolic", 0, {"observation": "vector"}) as writer:
            writer.append(transition)
        with store.episode("live_screen", 0, {"game_id": "original_pvz"}) as writer:
            writer.append(transition)
            writer.append(transition)
        with store.episode("python_port", 0, {"game_id": "marblexu_python_port"}) as writer:
            writer.append(transition)
    status = audit_data(data)
    assert status["synthetic_transitions"] == 1
    assert status["original_game_transitions"] == 2
    assert status["python_port_transitions"] == 1
    assert status["ready_for_original_game_training"] is True
    assert status["ready_for_python_port_training"] is True


def test_data_audit_reports_reference_readiness(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(audit_module, "discover_macos_pvz", list)
    reference = tmp_path / "data" / "reference" / "local"
    reference.mkdir(parents=True)
    (reference / "goty-reference.json").write_text(
        json.dumps(
            {
                "kind": "pvz-goty-reference",
                "content_id": "a" * 64,
                "baseline_kind": "installation",
                "declared_version": "1.2.0.1073",
                "distribution": "steam-windows",
                "ready_for_black_box_comparison": True,
            }
        )
    )

    status = audit_data(tmp_path / "data")

    assert status["reference_metadata_files"] == 1
    assert status["ready_for_original_game_comparison"] is True
    assert status["reference_manifests"] == [
        {
            "path": str(reference / "goty-reference.json"),
            "content_id": "a" * 64,
            "baseline_kind": "installation",
            "declared_version": "1.2.0.1073",
            "distribution": "steam-windows",
            "ready_for_black_box_comparison": True,
        }
    ]
