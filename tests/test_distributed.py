import torch

from pvzwm.distributed import DistributedContext, initialize_distributed


def test_single_process_distributed_context_is_a_noop(monkeypatch) -> None:
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    context = initialize_distributed("cpu")
    assert context.device_info.device == torch.device("cpu")
    assert context.enabled is False
    assert context.is_main is True
    assert context.reduce_epoch({"loss": 3.0}, 2) == ({"loss": 3.0}, 2)
    context.barrier()
    context.close()
