import base64
import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pvzwm.portable import (
    PortableAction,
    PortableActionCodec,
    PortableActionKind,
    PortableBridgeClient,
    PortableEnv,
    PortableLoadoutCurriculum,
    PortableObservation,
    PortableProcessConfig,
    PortableSequentialCurriculum,
    parse_protocol_record,
    portable_process_config_from_checkpoint,
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
        assert client.ready["action_count"] == 650
        assert client.request({"cmd": "ping"})["pong"] is True
    finally:
        client.close()


def test_portable_process_command_never_opens_a_listener() -> None:
    config = PortableProcessConfig(Path("/opt/pvz/pvzwm-bridge"), Path("/opt/pvz/data"))
    command = config.command()
    assert command[:3] == ["xvfb-run", "-a", "-s"]
    assert command[3] == "-screen 0 800x600x24 -nolisten tcp"
    assert "-listen" not in command[3]


def test_portable_debugger_wraps_the_private_pipe_process() -> None:
    config = PortableProcessConfig(
        Path("/opt/pvz/pvzwm-bridge"), Path("/opt/pvz/data"), debugger=True
    )
    command = config.command()
    assert command[:3] == ["xvfb-run", "-a", "-s"]
    assert command[3] == "-screen 0 800x600x24 -nolisten tcp"
    assert command[4:6] == ["gdb", "--batch"]
    assert "thread apply all bt" in command
    assert command[-2:] == ["/opt/pvz/pvzwm-bridge", "-resdir=/opt/pvz/data"]


def test_portable_process_command_accepts_isolated_save_directory() -> None:
    config = PortableProcessConfig(
        Path("/opt/pvz/pvzwm-bridge"),
        Path("/opt/pvz/data"),
        xvfb_run=None,
        save_directory=Path("/tmp/pvz checkpoint"),
    )
    assert config.command() == [
        "/opt/pvz/pvzwm-bridge",
        "-resdir=/opt/pvz/data",
        "-savedir=/tmp/pvz checkpoint",
    ]


def test_portable_observation_has_stable_vector_shape() -> None:
    observation = PortableObservation(
        {
            "area": 5,
            "stage": 1,
            "sun": 50,
            "wave": 0,
            "won": False,
            "board_result": 0,
            "plants": [
                {"type": 33, "row": 0, "col": 0, "health": 300, "state": 1}
            ],
            "zombies": [
                {
                    "type": 0,
                    "row": 2,
                    "x": 600,
                    "body_health": 270,
                    "helm_health": 0,
                    "shield_health": 0,
                    "phase": 3,
                }
            ],
            "seed_packets": [{"slot": 0, "type": 1, "active": True, "refresh": 0}],
        }
    )
    vector = observation.vector()
    assert observation.level_id == "5-1"
    assert vector.shape == (585,)
    assert vector.dtype == np.float32
    assert np.isfinite(vector).all()
    assert ((0.0 <= vector) & (vector <= 1.0)).all()
    assert observation.vector_version == 2

    legacy = PortableObservation(observation.payload, vector_version=1).vector()
    assert legacy.shape == (250,)
    assert legacy.dtype == np.float32


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


def test_portable_capture_frame_observes_without_stepping() -> None:
    pixels = base64.b64encode(bytes([1, 2, 3, 4, 5, 6])).decode("ascii")

    class FakeProcess:
        def __init__(self) -> None:
            self.stdin = io.StringIO()
            self.stdout = io.StringIO(
                'PVZWM {"ok":true,"ready":true,"action_count":650}\n'
                'PVZWM {"ok":true,"area":1,"stage":4,"legal_actions":[0]}\n'
                'PVZWM {"ok":true,"area":1,"stage":4,"legal_actions":[0],'
                f'"frame":{{"encoding":"rgb8-base64","width":2,"height":1,'
                f'"data":"{pixels}"}}}}\n'
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
    environment = PortableEnv(
        PortableProcessConfig(
            Path("bridge"), Path("resources"), frame_width=0, frame_height=0
        ),
        process_factory=lambda *args, **kwargs: fake,
    )
    try:
        environment.reset(seed=1)
        frame = environment.capture_frame(2, 1)
        assert frame.shape == (3, 1, 2)
        assert environment._step_count == 0
        assert '"cmd":"observe"' in fake.stdin.getvalue()
    finally:
        environment.close()


def test_portable_process_config_validates_frame_dimensions() -> None:
    with pytest.raises(ValueError):
        PortableProcessConfig(Path("bridge"), Path("resources"), frame_width=160, frame_height=0)
    with pytest.raises(ValueError):
        PortableProcessConfig(Path("bridge"), Path("resources"), frame_width=801, frame_height=600)


def test_portable_process_config_validates_seed_loadout() -> None:
    assert PortableProcessConfig(
        Path("bridge"), Path("resources"), seed_types=(1, 0, 16)
    ).seed_types == (1, 0, 16)
    with pytest.raises(ValueError, match="duplicates"):
        PortableProcessConfig(
            Path("bridge"), Path("resources"), seed_types=(1, 1)
        )
    with pytest.raises(ValueError, match=r"\[0, 48\)"):
        PortableProcessConfig(
            Path("bridge"), Path("resources"), seed_types=(48,)
        )


def test_portable_checkpoint_config_is_weights_only_safe(tmp_path: Path) -> None:
    config = PortableProcessConfig(
        Path("bridge"),
        Path("resources"),
        seed_types=(1, 0, 16),
        save_directory=Path("save-slot"),
    )
    checkpoint = tmp_path / "portable.pt"
    torch.save({"portable_config": config.checkpoint_dict()}, checkpoint)
    restored = torch.load(checkpoint, weights_only=True)
    assert restored["portable_config"]["binary"] == "bridge"
    assert restored["portable_config"]["resources"] == "resources"
    assert restored["portable_config"]["vector_version"] == 2
    assert restored["portable_config"]["reward_mode"] == "bridge"
    assert restored["portable_config"]["max_steps"] == 3_600
    assert restored["portable_config"]["disable_mowers"] is False
    assert restored["portable_config"]["seed_types"] == (1, 0, 16)
    assert restored["portable_config"]["save_directory"] == "save-slot"
    assert portable_process_config_from_checkpoint(
        {**restored["portable_config"], "seed_types": [1, 0, 16]}
    ).seed_types == (1, 0, 16)


def test_portable_observation_reports_mower_state() -> None:
    observation = PortableObservation(
        {
            "area": 3,
            "stage": 1,
            "mowers_ready": 4,
            "mowers_triggered": 2,
            "mowers": [
                {"row": 0, "state": 1},
                {"row": 3, "state": 2},
            ],
        }
    )

    assert observation.mowers_ready == 4
    assert observation.mowers_triggered == 2
    assert observation.mower_states == (1, -1, -1, 2, -1, -1)


def test_portable_observation_reports_loadout_space() -> None:
    observation = PortableObservation(
        {"seed_bank_capacity": 7, "available_seed_types": [0, 1, 16]}
    )
    assert observation.seed_bank_capacity == 7
    assert observation.available_seed_types == (0, 1, 16)


def test_portable_clipped_damage_reward_removes_spawn_penalties() -> None:
    class FakeProcess:
        def __init__(self) -> None:
            self.stdin = io.StringIO()
            self.stdout = io.StringIO(
                'PVZWM {"ok":true,"ready":true,"action_count":650}\n'
                'PVZWM {"ok":true,"area":1,"stage":4,"legal_actions":[0]}\n'
                'PVZWM {"ok":true,"area":1,"stage":4,"legal_actions":[0],'
                '"done":false,"reward":-2.5}\n'
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

    environment = PortableEnv(
        PortableProcessConfig(
            Path("bridge"),
            Path("resources"),
            frame_width=0,
            frame_height=0,
            reward_mode="clipped-damage",
        ),
        process_factory=lambda *args, **kwargs: FakeProcess(),
    )
    try:
        environment.reset(seed=1)
        _, reward, _, _, info = environment.step(0)
        assert reward == 0.0
        assert info["bridge_reward"] == -2.5
        assert info["reward_mode"] == "clipped-damage"
    finally:
        environment.close()


def test_portable_checkpoint_restore_preserves_legacy_vector_shape() -> None:
    raw = PortableProcessConfig(Path("bridge"), Path("resources")).checkpoint_dict()
    raw.pop("vector_version")
    legacy = portable_process_config_from_checkpoint(raw, state_dim=250)
    current = portable_process_config_from_checkpoint(raw, state_dim=585)
    assert legacy.vector_version == 1
    assert current.vector_version == 2


def test_portable_environment_uses_configured_level_and_truncates() -> None:
    class FakeProcess:
        def __init__(self) -> None:
            self.stdin = io.StringIO()
            self.stdout = io.StringIO(
                'PVZWM {"ok":true,"ready":true,"action_count":650}\n'
                'PVZWM {"ok":true,"area":3,"stage":1,"legal_actions":[0]}\n'
                'PVZWM {"ok":true,"area":3,"stage":1,"legal_actions":[0],'
                '"done":false,"reward":0}\n'
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
        Path("bridge"),
        Path("resources"),
        level=21,
        max_steps=1,
        frame_width=0,
        frame_height=0,
        disable_mowers=True,
        seed_types=(1, 0, 16, 3, 2, 4, 5),
    )
    environment = PortableEnv(config, process_factory=lambda *args, **kwargs: fake)
    try:
        observation, info = environment.reset(seed=9)
        assert info["level"] == "3-1"
        assert observation.vector(object()).shape == (585,)
        _, _, terminated, truncated, _ = environment.step(0)
        assert not terminated
        assert truncated
        assert '"level":21' in fake.stdin.getvalue()
        assert '"seed":9' in fake.stdin.getvalue()
        assert '"disable_mowers":1' in fake.stdin.getvalue()
        assert '"seed_types":[1,0,16,3,2,4,5]' in fake.stdin.getvalue()
    finally:
        environment.close()


def test_portable_curriculum_advances_only_after_a_win() -> None:
    class FakeProcess:
        def __init__(self) -> None:
            self.stdin = io.StringIO()
            self.stdout = io.StringIO(
                'PVZWM {"ok":true,"ready":true,"action_count":650}\n'
                'PVZWM {"ok":true,"area":1,"stage":1,"legal_actions":[0]}\n'
                'PVZWM {"ok":true,"area":1,"stage":1,"legal_actions":[0],'
                '"done":true,"won":true,"reward":100}\n'
                'PVZWM {"ok":true,"area":1,"stage":2,"legal_actions":[0]}\n'
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
        Path("bridge"), Path("resources"), frame_width=0, frame_height=0
    )
    curriculum = PortableSequentialCurriculum(
        config, final_level=3, process_factory=lambda *args, **kwargs: fake
    )
    try:
        _, info = curriculum.reset(seed=1)
        assert info["curriculum_level_number"] == 1
        _, _, terminated, _, info = curriculum.step(0)
        assert terminated
        assert info["curriculum_episode_level_number"] == 1
        assert info["curriculum_level_number"] == 2
        assert info["curriculum_completed_levels"] == (1,)
        observation, info = curriculum.reset(seed=2)
        assert observation.level_id == "1-2"
        assert info["curriculum_level_number"] == 2
        assert '"level":1' in fake.stdin.getvalue()
        assert '"level":2' in fake.stdin.getvalue()
    finally:
        curriculum.close()


def test_portable_loadout_curriculum_selects_an_authentic_episode_loadout() -> None:
    class FakeProcess:
        def __init__(self) -> None:
            self.stdin = io.StringIO()
            self.stdout = io.StringIO(
                'PVZWM {"ok":true,"ready":true,"action_count":650}\n'
                'PVZWM {"ok":true,"area":3,"stage":1,"legal_actions":[0]}\n'
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
    loadouts = ((16, 1, 0, 7, 5, 3), (6, 3, 0, 7, 5, 1))
    curriculum = PortableLoadoutCurriculum(
        PortableProcessConfig(
            Path("bridge"), Path("resources"), frame_width=0, frame_height=0
        ),
        loadouts,
        process_factory=lambda *args, **kwargs: fake,
    )
    try:
        _, info = curriculum.reset(seed=42)
        selected = info["loadout_curriculum_episode_loadout"]
        assert selected in loadouts
        assert info["loadout_curriculum_size"] == 2
        serialized = json.dumps(list(selected), separators=(",", ":"))
        assert f'"seed_types":{serialized}' in fake.stdin.getvalue()
    finally:
        curriculum.close()


def test_portable_curriculum_validates_rehearsal_probability() -> None:
    config = PortableProcessConfig(Path("bridge"), Path("resources"))
    with pytest.raises(ValueError, match="rehearsal"):
        PortableSequentialCurriculum(config, final_level=2, rehearsal_probability=1.1)


def test_portable_curriculum_restores_completed_rehearsal_prefix() -> None:
    class FakeProcess:
        def __init__(self) -> None:
            self.stdin = io.StringIO()
            self.stdout = io.StringIO(
                'PVZWM {"ok":true,"ready":true,"action_count":650}\n'
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

    config = PortableProcessConfig(Path("bridge"), Path("resources"), level=4)
    fake = FakeProcess()
    curriculum = PortableSequentialCurriculum(
        config,
        final_level=8,
        completed_levels=(1, 2, 3),
        process_factory=lambda *args, **kwargs: fake,
    )
    try:
        assert curriculum.completed_levels == [1, 2, 3]
    finally:
        curriculum.close()
    with pytest.raises(ValueError, match="precede"):
        PortableSequentialCurriculum(config, final_level=8, completed_levels=(4,))
