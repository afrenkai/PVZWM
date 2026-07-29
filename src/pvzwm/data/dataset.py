from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from pvzwm.data.store import ReplayStore, Transition


def _model_tensor(value) -> torch.Tensor:
    tensor = torch.from_numpy(value)
    if tensor.dtype == torch.uint8:
        # Keep replay pixels compact and defer normalization until after the
        # batched transfer. Normalizing every overlapping sequence window on
        # the CPU starves accelerators and expands host-to-device traffic 4x.
        return tensor
    return tensor.to(torch.float32)


class TransitionDataset(Dataset[tuple[torch.Tensor, ...]]):
    """In-memory training view; DuckDB remains the durable source of truth."""

    def __init__(
        self,
        path: str | Path,
        limit: int | None = None,
        episode_ids: list[str] | None = None,
        include_legal_mask: bool = False,
    ) -> None:
        with ReplayStore(path, read_only=True) as store:
            self.items: list[Transition] = list(
                store.transitions(limit=limit, episode_ids=episode_ids)
            )
        self.include_legal_mask = include_legal_mask
        if include_legal_mask and any(item.legal_mask is None for item in self.items):
            raise ValueError("Selected replay transitions are missing legal-action masks")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        item = self.items[index]
        tensors = (
            _model_tensor(item.observation),
            torch.tensor(item.action, dtype=torch.long),
            torch.tensor(item.reward, dtype=torch.float32),
            _model_tensor(item.next_observation),
            # A collector time limit ends an episode window but is not an
            # environment terminal for learned dynamics.
            torch.tensor(item.terminated, dtype=torch.float32),
        )
        if not self.include_legal_mask:
            return tensors
        assert item.legal_mask is not None
        return (*tensors, torch.from_numpy(item.legal_mask).to(torch.bool))


class SequenceDataset(Dataset[tuple[torch.Tensor, ...]]):
    """Episode-safe fixed windows for recurrent world-model training."""

    def __init__(
        self,
        path: str | Path,
        length: int = 16,
        stride: int = 4,
        episode_ids: list[str] | None = None,
        include_legal_mask: bool = False,
    ) -> None:
        with ReplayStore(path, read_only=True) as store:
            self.items = list(
                store.transition_sequences(length, stride, episode_ids=episode_ids)
            )
        self.include_legal_mask = include_legal_mask
        if include_legal_mask and any(
            item.legal_mask is None
            for sequence in self.items
            for item in sequence
        ):
            raise ValueError("Selected replay transitions are missing legal-action masks")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        sequence = self.items[index]
        tensors = (
            torch.stack([_model_tensor(item.observation) for item in sequence]),
            torch.tensor([item.action for item in sequence], dtype=torch.long),
            torch.tensor([item.reward for item in sequence], dtype=torch.float32),
            torch.stack([_model_tensor(item.next_observation) for item in sequence]),
            torch.tensor(
                [item.terminated for item in sequence], dtype=torch.float32
            ),
        )
        if not self.include_legal_mask:
            return tensors
        return (
            *tensors,
            torch.stack(
                [torch.from_numpy(item.legal_mask).to(torch.bool) for item in sequence]
            ),
        )
