from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Protocol

import numpy as np
from PIL import Image, ImageGrab

from pvzwm.data.store import ReplayStore, Transition
from pvzwm.domain import ActionCodec, ActionKind, GameConfig, PVZAction


@dataclass(frozen=True, slots=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class Box:
    left: int
    top: int
    right: int
    bottom: int

    def __post_init__(self) -> None:
        if self.right <= self.left or self.bottom <= self.top:
            raise ValueError("Box must have positive width and height")

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


class ScreenBackend(Protocol):
    def capture(self, box: Box) -> np.ndarray: ...


class PointerBackend(Protocol):
    def click(self, point: Point) -> None: ...


class PillowScreenBackend:
    def capture(self, box: Box) -> np.ndarray:
        image = ImageGrab.grab(bbox=(box.left, box.top, box.right, box.bottom))
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
        return array.copy()


class QuartzPointerBackend:
    """macOS pointer backend, imported lazily so data-only workflows stay portable."""

    def click(self, point: Point) -> None:
        try:
            import Quartz
        except ImportError as error:
            raise RuntimeError(
                "Live clicks require pyobjc-framework-Quartz; install it explicitly with uv"
            ) from error
        location = (point.x, point.y)
        down = Quartz.CGEventCreateMouseEvent(
            None, Quartz.kCGEventLeftMouseDown, location, Quartz.kCGMouseButtonLeft
        )
        up = Quartz.CGEventCreateMouseEvent(
            None, Quartz.kCGEventLeftMouseUp, location, Quartz.kCGMouseButtonLeft
        )
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, down)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, up)


class NullPointerBackend:
    def click(self, point: Point) -> None:
        raise RuntimeError("Pointer click attempted while live execution is disabled")


@dataclass(frozen=True, slots=True)
class ScreenGeometry:
    capture: Box
    lawn: Box
    seed_packets: dict[ActionKind, Point]
    shovel: Point
    rows: int = 5
    cols: int = 9

    def __post_init__(self) -> None:
        required = {
            ActionKind.PLANT_SUNFLOWER,
            ActionKind.PLANT_PEASHOOTER,
            ActionKind.PLANT_WALLNUT,
        }
        if not required.issubset(self.seed_packets):
            missing = sorted(kind.name for kind in required - self.seed_packets.keys())
            raise ValueError(f"Missing seed-packet coordinates: {missing}")
        if self.rows < 1 or self.cols < 1:
            raise ValueError("rows and cols must be positive")

    def cell_center(self, row: int, col: int) -> Point:
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError("Cell is outside the calibrated lawn")
        return Point(
            self.lawn.left + (col + 0.5) * self.lawn.width / self.cols,
            self.lawn.top + (row + 0.5) * self.lawn.height / self.rows,
        )

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> ScreenGeometry:
        def box(name: str) -> Box:
            coordinates = value[name]
            if not isinstance(coordinates, list) or len(coordinates) != 4:
                raise ValueError(f"{name} must be [left, top, right, bottom]")
            return Box(*(int(item) for item in coordinates))

        def point(coordinates: object) -> Point:
            if not isinstance(coordinates, list) or len(coordinates) != 2:
                raise ValueError("Point must be [x, y]")
            return Point(float(coordinates[0]), float(coordinates[1]))

        raw_packets = value.get("seed_packets")
        if not isinstance(raw_packets, dict):
            raise ValueError("seed_packets must be an object")
        name_to_kind = {
            "sunflower": ActionKind.PLANT_SUNFLOWER,
            "peashooter": ActionKind.PLANT_PEASHOOTER,
            "wallnut": ActionKind.PLANT_WALLNUT,
        }
        packets = {
            name_to_kind[name]: point(coordinates)
            for name, coordinates in raw_packets.items()
            if name in name_to_kind
        }
        return cls(
            box("capture"),
            box("lawn"),
            packets,
            point(value["shovel"]),
            int(value.get("rows", 5)),
            int(value.get("cols", 9)),
        )

    @classmethod
    def load(cls, path: str | Path) -> ScreenGeometry:
        return cls.from_dict(json.loads(Path(path).read_text()))


def prepare_frame(frame: np.ndarray, height: int = 96, width: int = 160) -> np.ndarray:
    if frame.ndim != 3 or frame.shape[-1] not in {3, 4}:
        raise ValueError("Captured frame must be HWC RGB or RGBA")
    rgb = frame[..., :3]
    if rgb.dtype != np.uint8:
        if np.issubdtype(rgb.dtype, np.floating):
            rgb = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    image = Image.fromarray(rgb, mode="RGB").resize((width, height), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.transpose(array, (2, 0, 1)).copy()


class LiveGameAdapter:
    def __init__(
        self,
        geometry: ScreenGeometry,
        screen: ScreenBackend,
        pointer: PointerBackend,
        *,
        output_height: int = 96,
        output_width: int = 160,
        settle_seconds: float = 0.5,
        execute: bool = False,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.geometry = geometry
        self.screen = screen
        self.pointer = pointer
        self.output_height = output_height
        self.output_width = output_width
        self.settle_seconds = settle_seconds
        self.execute = execute
        self.sleeper = sleeper
        self.codec = ActionCodec(GameConfig(rows=geometry.rows, cols=geometry.cols))

    def observe(self) -> np.ndarray:
        return prepare_frame(
            self.screen.capture(self.geometry.capture),
            self.output_height,
            self.output_width,
        )

    def click_plan(self, action_id: int) -> tuple[Point, ...]:
        action = self.codec.decode(action_id)
        if action.kind == ActionKind.WAIT:
            return ()
        destination = self.geometry.cell_center(action.row, action.col)
        if action.kind == ActionKind.SHOVEL:
            return self.geometry.shovel, destination
        return self.geometry.seed_packets[action.kind], destination

    def perform(self, action_id: int) -> tuple[Point, ...]:
        plan = self.click_plan(action_id)
        if self.execute:
            for point in plan:
                self.pointer.click(point)
        self.sleeper(self.settle_seconds)
        return plan


def parse_actions(lines: Iterable[str], codec: ActionCodec) -> list[int]:
    actions: list[int] = []
    names = {
        "wait": ActionKind.WAIT,
        "sunflower": ActionKind.PLANT_SUNFLOWER,
        "peashooter": ActionKind.PLANT_PEASHOOTER,
        "wallnut": ActionKind.PLANT_WALLNUT,
        "shovel": ActionKind.SHOVEL,
    }
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        fields = line.split()
        try:
            if len(fields) == 1 and fields[0].lstrip("-").isdigit():
                action_id = int(fields[0])
                codec.decode(action_id)
            else:
                kind = names[fields[0].lower()]
                if kind == ActionKind.WAIT and len(fields) == 1:
                    action_id = 0
                elif len(fields) == 3:
                    action_id = codec.encode(PVZAction(kind, int(fields[1]), int(fields[2])))
                else:
                    raise ValueError("expected `wait`, an action id, or `kind row col`")
        except (KeyError, ValueError) as error:
            raise ValueError(f"Invalid action on line {line_number}: {raw_line.rstrip()}") from error
        actions.append(action_id)
    return actions


def record_live_actions(
    adapter: LiveGameAdapter,
    store: ReplayStore,
    actions: Iterable[int],
    *,
    seed: int = 0,
    source: str = "live_screen",
    game_id: str = "original_pvz",
) -> tuple[str, int]:
    observation = adapter.observe()
    with store.episode(
        source,
        seed,
        {
            "observation": "pixels",
            "reward": "unlabelled_zero_placeholder",
            "execute": adapter.execute,
            "game_id": game_id,
        },
    ) as writer:
        for action in actions:
            adapter.perform(action)
            next_observation = adapter.observe()
            writer.append(
                Transition(
                    observation,
                    action,
                    0.0,
                    next_observation,
                    False,
                    False,
                    None,
                )
            )
            observation = next_observation
    return writer.episode_id, writer.length
