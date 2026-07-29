from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

from pvzwm.device import DeviceInfo, select_device


@dataclass(slots=True)
class DistributedContext:
    device_info: DeviceInfo
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    initialized_here: bool = False

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.enabled:
            dist.barrier()

    def reduce_epoch(
        self, totals: dict[str, float], batches: int
    ) -> tuple[dict[str, float], int]:
        if not self.enabled:
            return totals, batches
        name_payload: list[object] = [sorted(totals) if self.is_main else None]
        dist.broadcast_object_list(name_payload, src=0)
        names = [str(name) for name in name_payload[0]]
        values = torch.tensor(
            [*(totals.get(name, 0.0) for name in names), float(batches)],
            dtype=torch.float64,
            device=self.device_info.device,
        )
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        return (
            {name: float(values[index].item()) for index, name in enumerate(names)},
            int(values[-1].item()),
        )

    def close(self) -> None:
        if self.initialized_here and dist.is_initialized():
            dist.destroy_process_group()


def initialize_distributed(preference: str = "auto") -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return DistributedContext(select_device(preference))
    if os.environ.get("PVZWM_ALLOW_DDP_NETWORK") != "1":
        raise RuntimeError(
            "Multi-process DDP can open network listeners; set "
            "PVZWM_ALLOW_DDP_NETWORK=1 only inside a verified network namespace "
            "or firewall policy"
        )
    if preference in {"mps", "cpu"}:
        raise RuntimeError("Multi-process training requires CUDA/NCCL")
    if not torch.cuda.is_available():
        raise RuntimeError("torchrun requested multiple ranks but CUDA is unavailable")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    initialized_here = not dist.is_initialized()
    if initialized_here:
        dist.init_process_group(
            backend="nccl", init_method="env://", device_id=device
        )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    major, _ = torch.cuda.get_device_capability(device)
    # Ampere and newer have native bfloat16, which is preferable to float16
    # for the covariance- and distance-heavy VISReg objective.
    autocast_dtype = torch.bfloat16 if major >= 8 else torch.float16
    return DistributedContext(
        DeviceInfo(device, "cuda", True, autocast_dtype),
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        initialized_here=initialized_here,
    )
