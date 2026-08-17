from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "audit_goty_mode_playthroughs",
    Path("scripts/audit_goty_mode_playthroughs.py"),
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_mode_playthrough_integer_spec() -> None:
    assert _MODULE.parse_integer_spec("17,25,27..30") == (17, 25, 27, 28, 29, 30)


@pytest.mark.parametrize("value", ["", "17,17", "30..17"])
def test_mode_playthrough_integer_spec_rejects_invalid_values(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        _MODULE.parse_integer_spec(value)


def test_survival_stage_win_is_not_complete_before_the_fifth_stage() -> None:
    payload = {"won": True, "board_result": 1, "survival_stage": 0}
    assert not _MODULE.mode_terminal(2, payload)
    payload["survival_stage"] = 4
    assert _MODULE.mode_terminal(2, payload)


def test_non_survival_engine_win_is_terminal() -> None:
    assert _MODULE.mode_terminal(22, {"won": True, "board_result": 1})


def test_each_mode_uses_fresh_clients_and_save_directories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clients: list[object] = []

    class FakeClient:
        def __init__(self, config: object) -> None:
            self.config = config
            self.closed = False
            clients.append(self)

        def close(self) -> None:
            self.closed = True

    def fake_play_mode(reference: object, candidate: object, **_: object) -> dict:
        return {"reference": reference, "candidate": candidate}

    monkeypatch.setattr(_MODULE, "PortableBridgeClient", FakeClient)
    monkeypatch.setattr(_MODULE, "play_mode", fake_play_mode)
    common = {
        "binary": Path("reference"),
        "candidate_binary": Path("candidate"),
        "resources": Path("resources"),
        "seed": 7,
        "ticks_per_action": 10,
        "max_ticks": 100,
    }
    _MODULE.play_isolated_mode(game_mode=1, **common)
    _MODULE.play_isolated_mode(game_mode=4, **common)

    assert len(clients) == 4
    assert all(client.closed for client in clients)
    save_directories = [client.config.save_directory for client in clients]
    assert len(set(save_directories)) == 4
