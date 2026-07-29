# One-node 8×A100 training

PVZWM uses one process per GPU through PyTorch DDP/NCCL. On A100, CUDA
autocast selects bfloat16; VISReg's numerically sensitive internals still cast
to float32. `--batch-size` is per rank, so 128 gives a global batch of 1024.

Place the repository and DuckDB replay on storage visible at the same path to
all ranks, then run on the GPU host:

```bash
uv sync --frozen
uv run torchrun --nnodes=1 --nproc-per-node=8 \
  --master-addr=127.0.0.1 --master-port=29437 \
  -m pvzwm.cli train-wm \
  --database data/port_pixels.duckdb --source python_port \
  --model jepa --regularizer visreg --regularization-weight 0.5 \
  --epochs 100 --batch-size 32 --num-workers 4 \
  --latent-dim 512 --hidden-dim 1024 --slices 4096 \
  --conv-pool-height 3 --conv-pool-width 5 --device cuda \
  --output artifacts/a100_port_jepa.pt
```

The rendezvous address must remain loopback-only. Do not bind it to `0.0.0.0`,
start a notebook/dashboard, add SSH port forwarding, or expose any training
listener. Change the local port if it is already occupied; no inbound firewall
opening is required.

The token-model comparison uses the same launch shape:

```bash
uv run torchrun --nnodes=1 --nproc-per-node=8 \
  --master-addr=127.0.0.1 --master-port=29438 \
  -m pvzwm.cli train-wm --database data/port_pixels.duckdb \
  --source python_port --model token --token-quantizer fsq \
  --fsq-levels 8,8,4 --sequence-length 8 --sequence-stride 8 \
  --epochs 100 --batch-size 8 --device cuda \
  --output artifacts/a100_token_fsq.pt
```

The JEPA example has a global batch of 256. Only rank zero evaluates, writes the checkpoint, records the experiment, and
prints JSON. Every rank reads the training split into memory and DDP shards its
indices, so allow RAM for eight decoded copies of the replay. The current path
is single-node; multi-node launch and streaming DuckDB shards remain future
work. Validate the remote CUDA/NCCL stack with a short `--epochs 1` run before
starting the long comparison. The six-episode pixel corpus is too small to
justify eight GPUs; collect a substantially broader multi-level corpus first.
