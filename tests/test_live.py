from pathlib import Path

import numpy as np

from pvzwm.data.store import ReplayStore
from pvzwm.domain import ActionKind, PVZAction
from pvzwm.live import (
    Box,
    LiveGameAdapter,
    Point,
    ScreenGeometry,
    parse_actions,
    record_live_actions,
)


class FakeScreen:
    def __init__(self) -> None:
        self.calls = 0

    def capture(self, box: Box) -> np.ndarray:
        self.calls += 1
        return np.full((box.height, box.width, 3), self.calls, dtype=np.uint8)


class FakePointer:
    def __init__(self) -> None:
        self.clicks: list[Point] = []

    def click(self, point: Point) -> None:
        self.clicks.append(point)


def geometry() -> ScreenGeometry:
    return ScreenGeometry(
        Box(0, 0, 180, 100),
        Box(0, 10, 180, 100),
        {
            ActionKind.PLANT_SUNFLOWER: Point(10, 5),
            ActionKind.PLANT_PEASHOOTER: Point(20, 5),
            ActionKind.PLANT_WALLNUT: Point(30, 5),
        },
        Point(170, 5),
    )


def test_live_adapter_maps_actions_and_is_dry_by_default() -> None:
    screen = FakeScreen()
    pointer = FakePointer()
    adapter = LiveGameAdapter(
        geometry(), screen, pointer, output_height=20, output_width=36, sleeper=lambda _: None
    )
    frame = adapter.observe()
    assert frame.shape == (3, 20, 36)
    action = adapter.codec.encode(PVZAction(ActionKind.PLANT_PEASHOOTER, 2, 4))
    plan = adapter.perform(action)
    assert plan[0] == Point(20, 5)
    assert plan[1] == Point(90, 55)
    assert pointer.clicks == []


def test_live_recording_with_fake_backends(tmp_path: Path) -> None:
    screen = FakeScreen()
    pointer = FakePointer()
    adapter = LiveGameAdapter(
        geometry(),
        screen,
        pointer,
        output_height=20,
        output_width=36,
        execute=True,
        sleeper=lambda _: None,
    )
    actions = parse_actions(["sunflower 0 0", "wait", "0  # categorical"], adapter.codec)
    with ReplayStore(tmp_path / "live.duckdb") as store:
        _, count = record_live_actions(adapter, store, actions)
        transitions = list(store.transitions())
    assert count == 3
    assert len(transitions) == 3
    assert len(pointer.clicks) == 2
    assert not np.array_equal(transitions[0].observation, transitions[0].next_observation)
