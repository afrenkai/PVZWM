import io
import base64
from pathlib import Path

import numpy as np
import pytest

from pvzwm.portable import (
    PortableAction,
    PortableActionCodec,
    PortableActionKind,
    PortableBridgeClient,
    PortableObservation,
    PortableProcessConfig,
    parse_protocol_record,
)


def test_portable_action_codec_round_trip() -> None:
    codec = PortableActionCodec()
    actions = [
        PortableAction(PortableActionKind.WAIT),
        PortableAction(PortableActionKind.PLANT, slot=9, row=5, col=8),
        PortableAction(PortableActionKind.SHOVEL, row=4, col=3),
        PortableAction(PortableActionKind.BOARD_CLICK, row=2, col=7),
        PortableAction(PortableActionKind.COLLECT),
    ]
    assert codec.size == 650
    for action in actions:
        assert codec.decode(codec.encode(action)) == action


def test_portable_codec_rejects_bad_coordinates() -> None:
    codec = PortableActionCodec()
    with pytest.raises(ValueError):
        codec.plant(10, 0, 0)
    with pytest.raises(ValueError):
        codec.board_click(6, 0)
    with pytest.raises(ValueError):
        codec.decode(650)


def test_portable_protocol_ignores_engine_logs() -> None:
    assert parse_protocol_record("INFO: loading resources\n") is None
    assert parse_protocol_record('PVZWM {"ok":true,"ready":true}\n') == {
        "ok": True,
        "ready": True,
    }


def test_portable_client_consumes_records_already_buffered_by_text_io() -> None:
    class FakeProcess:
        def __init__(self) -> None:
            self.stdin = io.StringIO()
            self.stdout = io.StringIO(
                "INFO: startup\n"
                'PVZWM {"ok":true,"ready":true,"action_count":650}\n'
                'PVZWM {"ok":true,"pong":true}\n'
            )
            self.returncode = None

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: float) -> int:
            self.returncode = 0
            return 0

        def terminate(self) -> None:
            self.returncode = -15

        def kill(self) -> None:
            self.returncode = -9

    fake = FakeProcess()
    config = PortableProcessConfig(
        Path("bridge"), Path("resources"), startup_timeout_seconds=1.0
    )
    client = PortableBridgeClient(config, process_factory=lambda *args, **kwargs: fake)
    try:
        assert client.request({"cmd": "ping"})["pong"] is True
    finally:
        client.close()


def test_portable_process_command_never_opens_a_listener() -> None:
    config = PortableProcessConfig(Path("/opt/pvz/pvzwm-bridge"), Path("/opt/pvz/data"))
    command = config.command()
    assert command[:2] == ["xvfb-run", "-a"]
    assert all("listen" not in argument.lower() for argument in command)
    assert all("port" not in argument.lower() for argument in command)


def test_portable_observation_has_stable_vector_shape() -> None:
    observation = PortableObservation(
        {
            "area": 5,
            "stage": 1,
            "sun": 50,
            "wave": 0,
            "won": False,
            "board_result": 0,
            "plants": [{"type": 33, "row": 0, "col": 0, "health": 300}],
            "zombies": [
                {
                    "type": 0,
                    "row": 2,
                    "x": 600,
                    "body_health": 270,
                    "helm_health": 0,
                    "shield_health": 0,
                }
            ],
            "seed_packets": [{"slot": 0, "type": 1, "active": True, "refresh": 0}],
        }
    )
    vector = observation.vector()
    assert observation.level_id == "5-1"
    assert vector.shape == (250,)
    assert vector.dtype == np.float32
    assert np.isfinite(vector).all()


def test_portable_observation_decodes_chw_rgb_frame() -> None:
    pixels = bytes([255, 0, 0, 0, 255, 0])
    observation = PortableObservation(
        {
            "area": 1,
            "stage": 1,
            "frame": {
                "encoding": "rgb8-base64",
                "width": 2,
                "height": 1,
                "source_width": 800,
                "source_height": 600,
                "data": base64.b64encode(pixels).decode("ascii"),
            },
        }
    )
    frame = observation.frame()
    assert frame is not None
    assert frame.shape == (3, 1, 2)
    assert frame.dtype == np.uint8
    assert frame[:, 0, 0].tolist() == [255, 0, 0]
    assert frame[:, 0, 1].tolist() == [0, 255, 0]


def test_portable_process_config_validates_frame_dimensions() -> None:
    with pytest.raises(ValueError):
        PortableProcessConfig(Path("bridge"), Path("resources"), frame_width=160, frame_height=0)
    with pytest.raises(ValueError):
        PortableProcessConfig(Path("bridge"), Path("resources"), frame_width=801, frame_height=600)
