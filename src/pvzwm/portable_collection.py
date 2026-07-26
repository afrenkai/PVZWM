from __future__ import annotations

import numpy as np

from pvzwm.portable import (
    PortableActionCodec,
    PortableObservation,
    collect_sun_first_mask,
)


def encode_portable_observation(
    observation: PortableObservation,
    mode: str,
    frame: np.ndarray | None = None,
) -> np.ndarray:
    """Encode an engine observation without silently changing data domains."""

    if mode == "vector":
        return observation.vector()
    if mode == "pixels":
        if frame is None:
            raise ValueError("Pixel collection requires a rendered frame")
        return np.asarray(frame, dtype=np.uint8)
    raise ValueError(f"Unsupported portable observation mode: {mode}")


class PortableExplorationPolicy:
    """Collect transient sun, then sample uniformly from legal control actions."""

    def act(self, legal_mask: np.ndarray, rng: np.random.Generator) -> int:
        codec = PortableActionCodec()
        mask = np.asarray(legal_mask, dtype=np.bool_)
        if mask.shape != (codec.size,):
            raise ValueError(f"Expected legal mask shape ({codec.size},), got {mask.shape}")

        legal_actions = np.flatnonzero(collect_sun_first_mask(codec, mask))
        if not legal_actions.size:
            raise ValueError("Portable engine returned no legal actions")
        return int(rng.choice(legal_actions))


def parse_adventure_levels(specification: str) -> list[int]:
    """Parse raw levels or stage IDs, including inclusive ranges."""

    def level_number(value: str) -> int:
        value = value.strip()
        if "-" not in value:
            level = int(value)
        else:
            area_text, stage_text = value.split("-", maxsplit=1)
            area, stage = int(area_text), int(stage_text)
            if not 1 <= area <= 5 or not 1 <= stage <= 10:
                raise ValueError(f"Adventure stage outside 1-1 through 5-10: {value}")
            level = (area - 1) * 10 + stage
        if not 1 <= level <= 50:
            raise ValueError(f"Adventure level outside [1, 50]: {level}")
        return level

    levels: list[int] = []
    for item in specification.split(","):
        item = item.strip()
        if not item:
            continue
        if ".." in item:
            start_text, end_text = item.split("..", maxsplit=1)
            start, end = level_number(start_text), level_number(end_text)
            if end < start:
                raise ValueError(f"Adventure range runs backwards: {item}")
            levels.extend(range(start, end + 1))
        else:
            levels.append(level_number(item))
    if not levels:
        raise ValueError("At least one Adventure level is required")
    return list(dict.fromkeys(levels))
