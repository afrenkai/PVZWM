from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import numpy as np


class RandomProjectionCellEncoder:
    """Locality-sensitive state cells without game-specific feature selection."""

    def __init__(
        self,
        state_dim: int,
        projections: int = 20,
        seed: int = 0,
        cell_width: float = 0.02,
    ) -> None:
        if state_dim < 1 or projections < 1 or cell_width <= 0.0:
            raise ValueError("state_dim, projections, and cell_width must be positive")
        self.state_dim = state_dim
        self.projections = projections
        self.seed = seed
        self.cell_width = cell_width
        generator = np.random.default_rng(seed)
        self.projection = generator.standard_normal(
            (projections, state_dim), dtype=np.float32
        ) / np.sqrt(state_dim)

    def encode(self, state: np.ndarray) -> bytes:
        flat = np.asarray(state, dtype=np.float32).reshape(-1)
        if flat.size != self.state_dim:
            raise ValueError(
                f"State has {flat.size} values; expected {self.state_dim}"
            )
        projected = self.projection @ flat
        quantized = np.rint(projected / self.cell_width).astype(np.int16)
        return quantized.tobytes()


@dataclass(slots=True)
class ArchiveCell:
    key: bytes
    seed: int
    actions: tuple[int, ...]
    episode_return: float
    selections: int = 0
    discoveries: int = 0


@dataclass(frozen=True, slots=True)
class ArchiveUpdate:
    inserted: bool
    replaced: bool
    cell: ArchiveCell


class TrajectoryArchive:
    """Go-Explore archive storing exact action prefixes back to generic state cells."""

    def __init__(self) -> None:
        self.cells: dict[bytes, ArchiveCell] = {}

    def add(
        self,
        key: bytes,
        seed: int,
        actions: Sequence[int],
        episode_return: float,
        *,
        parent_key: bytes | None = None,
    ) -> ArchiveUpdate:
        candidate_actions = tuple(int(action) for action in actions)
        existing = self.cells.get(key)
        inserted = existing is None
        replaced = False
        if existing is None:
            existing = ArchiveCell(
                key=key,
                seed=int(seed),
                actions=candidate_actions,
                episode_return=float(episode_return),
            )
            self.cells[key] = existing
            if parent_key is not None and parent_key in self.cells:
                self.cells[parent_key].discoveries += 1
        elif (
            episode_return > existing.episode_return
            or (
                episode_return == existing.episode_return
                and len(candidate_actions) < len(existing.actions)
            )
        ):
            existing.seed = int(seed)
            existing.actions = candidate_actions
            existing.episode_return = float(episode_return)
            replaced = True
        return ArchiveUpdate(inserted, replaced, existing)

    def selection_weights(
        self,
        strategy: str = "count",
        frontier_fraction: float = 0.2,
        branch_ids: Sequence[object] | None = None,
    ) -> np.ndarray:
        if not self.cells:
            raise ValueError("Cannot select from an empty archive")
        if strategy not in {"count", "frontier", "frontier-branch"}:
            raise ValueError(f"Unsupported archive selection strategy: {strategy}")
        if not 0.0 < frontier_fraction <= 1.0:
            raise ValueError("frontier_fraction must be in (0, 1]")
        cells = list(self.cells.values())
        if strategy == "frontier-branch":
            if branch_ids is None or len(branch_ids) != len(cells):
                raise ValueError(
                    "frontier-branch selection requires one branch id per archive cell"
                )
        elif branch_ids is not None:
            raise ValueError("branch ids are only valid for frontier-branch selection")
        if strategy == "count":
            weights = np.asarray(
                [
                    1.0
                    / np.sqrt(1.0 + cell.selections)
                    / np.sqrt(1.0 + cell.discoveries)
                    for cell in cells
                ],
                dtype=np.float64,
            )
        else:
            depths = np.asarray([len(cell.actions) for cell in cells], dtype=np.int64)
            frontier_depth = int(
                np.quantile(depths, 1.0 - frontier_fraction, method="lower")
            )
            on_frontier = depths >= frontier_depth
            weights = np.asarray(
                [
                    np.sqrt(1.0 + cell.discoveries)
                    / np.sqrt(1.0 + cell.selections)
                    for cell in cells
                ],
                dtype=np.float64,
            )
            # Productive cells should remain promising return points. Retain a
            # small probability of revisiting shallower regions without letting
            # a fragmented archive overwhelm its trajectory frontier.
            weights *= np.where(on_frontier, 1.0, 0.01)
            if strategy == "frontier-branch":
                assert branch_ids is not None
                branch_counts = Counter(branch_ids)
                rarity = np.asarray(
                    [
                        min(
                            10.0,
                            np.sqrt(len(cells) / branch_counts[branch_id]),
                        )
                        for branch_id in branch_ids
                    ],
                    dtype=np.float64,
                )
                weights *= rarity
        weights /= weights.sum()
        return weights

    def select(
        self,
        generator: np.random.Generator,
        strategy: str = "count",
        frontier_fraction: float = 0.2,
        branch_ids: Sequence[object] | None = None,
    ) -> ArchiveCell:
        cells = list(self.cells.values())
        weights = self.selection_weights(
            strategy, frontier_fraction, branch_ids=branch_ids
        )
        cell = cells[int(generator.choice(len(cells), p=weights))]
        cell.selections += 1
        return cell

    def payload(self, encoder: RandomProjectionCellEncoder) -> dict[str, object]:
        return {
            "schema_version": 1,
            "encoder": {
                "kind": "random_projection_quantized",
                "state_dim": encoder.state_dim,
                "projections": encoder.projections,
                "seed": encoder.seed,
                "cell_width": encoder.cell_width,
            },
            "cells": [
                {
                    "key": cell.key.hex(),
                    "seed": cell.seed,
                    "actions": list(cell.actions),
                    "episode_return": cell.episode_return,
                    "selections": cell.selections,
                    "discoveries": cell.discoveries,
                }
                for cell in self.cells.values()
            ],
        }

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, object],
        encoder: RandomProjectionCellEncoder,
    ) -> TrajectoryArchive:
        raw_encoder = payload.get("encoder")
        expected = {
            "kind": "random_projection_quantized",
            "state_dim": encoder.state_dim,
            "projections": encoder.projections,
            "seed": encoder.seed,
            "cell_width": encoder.cell_width,
        }
        if raw_encoder != expected:
            raise ValueError("Archive cell encoder does not match the requested encoder")
        raw_cells = payload.get("cells")
        if not isinstance(raw_cells, list):
            raise ValueError("Archive payload has no cell list")
        archive = cls()
        for raw_cell in raw_cells:
            if not isinstance(raw_cell, dict):
                raise ValueError("Archive cell must be an object")
            cell = ArchiveCell(
                key=bytes.fromhex(str(raw_cell["key"])),
                seed=int(raw_cell["seed"]),
                actions=tuple(int(action) for action in raw_cell["actions"]),
                episode_return=float(raw_cell["episode_return"]),
                selections=int(raw_cell.get("selections", 0)),
                discoveries=int(raw_cell.get("discoveries", 0)),
            )
            archive.cells[cell.key] = cell
        return archive
