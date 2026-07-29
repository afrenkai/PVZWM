from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from pvzwm.data.dataset import SequenceDataset, TransitionDataset
from pvzwm.device import DeviceInfo, autocast_context
from pvzwm.distributed import DistributedContext


@dataclass(frozen=True, slots=True)
class TrainConfig:
    epochs: int = 10
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    gradient_clip: float = 10.0
    warmup_epochs: int = 0
    final_learning_rate_divisor: float = 1.0
    num_workers: int = 0


class _LossAdapter(nn.Module):
    """Route loss computation through forward so DDP observes the graph."""

    def __init__(self, model: nn.Module, loss_options: dict[str, object]) -> None:
        super().__init__()
        self.model = model
        self.loss_options = loss_options

    def forward(self, *tensors: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.model.loss(*tensors, **self.loss_options)


def _to_model_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    transferred = tensor.to(device)
    if transferred.dtype == torch.uint8:
        return transferred.to(torch.float32).div_(255.0)
    return transferred


def _training_loader(
    dataset: TransitionDataset | SequenceDataset,
    model: nn.Module,
    device_info: DeviceInfo,
    config: TrainConfig,
    distributed: DistributedContext,
    loss_options: dict[str, object],
) -> tuple[DataLoader, DistributedSampler | None, nn.Module]:
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=distributed.world_size,
            rank=distributed.rank,
            shuffle=True,
        )
        if distributed.enabled
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=device_info.backend == "cuda",
    )
    model.to(device_info.device)
    training_model: nn.Module = _LossAdapter(model, loss_options)
    if distributed.enabled:
        training_model = torch.nn.parallel.DistributedDataParallel(
            training_model,
            device_ids=[distributed.local_rank],
            output_device=distributed.local_rank,
        )
    return loader, sampler, training_model


def _learning_rate_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    if config.warmup_epochs < 0:
        raise ValueError("Warmup epochs cannot be negative")
    if config.warmup_epochs > config.epochs:
        raise ValueError("Warmup epochs cannot exceed training epochs")
    if config.final_learning_rate_divisor < 1.0:
        raise ValueError("Final learning-rate divisor must be at least one")
    total_steps = max(config.epochs * steps_per_epoch, 1)
    warmup_steps = config.warmup_epochs * steps_per_epoch
    decay_steps = max(total_steps - warmup_steps, 1)
    final_factor = 1.0 / config.final_learning_rate_divisor

    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            progress = step / warmup_steps
            return 0.001 + 0.999 * progress
        progress = min(max(step - warmup_steps, 0) / decay_steps, 1.0)
        return final_factor + 0.5 * (1.0 - final_factor) * (
            1.0 + math.cos(math.pi * progress)
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def train_transition_model(
    model: nn.Module,
    database: str | Path,
    device_info: DeviceInfo,
    config: TrainConfig | None = None,
    episode_ids: list[str] | None = None,
    distributed: DistributedContext | None = None,
    include_legal_mask: bool = False,
    **loss_options: object,
) -> list[dict[str, float]]:
    config = config or TrainConfig()
    distributed = distributed or DistributedContext(device_info)
    dataset = TransitionDataset(
        database, episode_ids=episode_ids, include_legal_mask=include_legal_mask
    )
    if not dataset:
        raise ValueError("Replay database contains no transitions")
    loader, sampler, training_model = _training_loader(
        dataset, model, device_info, config, distributed, loss_options
    )
    optimizer = torch.optim.AdamW(
        training_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = _learning_rate_scheduler(optimizer, config, len(loader))
    history: list[dict[str, float]] = []
    for epoch in range(config.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        totals: dict[str, float] = {}
        batches = 0
        training_model.train()
        for tensors in loader:
            tensors = tuple(
                _to_model_device(tensor, device_info.device) for tensor in tensors
            )
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device_info):
                losses = training_model(*tensors)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                training_model.parameters(), config.gradient_clip
            )
            optimizer.step()
            scheduler.step()
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach().cpu())
            batches += 1
        totals, batches = distributed.reduce_epoch(totals, batches)
        history.append(
            {"epoch": float(epoch + 1), **{name: value / batches for name, value in totals.items()}}
        )
    return history


def train_sequence_model(
    model: nn.Module,
    database: str | Path,
    device_info: DeviceInfo,
    config: TrainConfig | None = None,
    sequence_length: int = 16,
    sequence_stride: int = 4,
    episode_ids: list[str] | None = None,
    distributed: DistributedContext | None = None,
    include_legal_mask: bool = False,
    **loss_options: object,
) -> list[dict[str, float]]:
    config = config or TrainConfig()
    distributed = distributed or DistributedContext(device_info)
    dataset = SequenceDataset(
        database,
        sequence_length,
        sequence_stride,
        episode_ids=episode_ids,
        include_legal_mask=include_legal_mask,
    )
    if not dataset:
        raise ValueError("Replay database has no episodes long enough for this sequence length")
    loader, sampler, training_model = _training_loader(
        dataset, model, device_info, config, distributed, loss_options
    )
    optimizer = torch.optim.AdamW(
        training_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = _learning_rate_scheduler(optimizer, config, len(loader))
    history: list[dict[str, float]] = []
    for epoch in range(config.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        totals: dict[str, float] = {}
        batches = 0
        training_model.train()
        for tensors in loader:
            tensors = tuple(
                _to_model_device(tensor, device_info.device) for tensor in tensors
            )
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device_info):
                losses = training_model(*tensors)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                training_model.parameters(), config.gradient_clip
            )
            optimizer.step()
            scheduler.step()
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach().cpu())
            batches += 1
        totals, batches = distributed.reduce_epoch(totals, batches)
        history.append(
            {"epoch": float(epoch + 1), **{name: value / batches for name, value in totals.items()}}
        )
    return history
