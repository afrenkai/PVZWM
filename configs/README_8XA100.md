# One-node 8×A100 training without listeners

PVZWM's secure remote profile runs independent single-process CUDA jobs. Each
job selects one GPU with `CUDA_VISIBLE_DEVICES`; it communicates with the game
through inherited stdin/stdout pipes and opens no training port. CUDA autocast
selects bfloat16 on A100, while numerically sensitive VISReg internals remain
float32.

Do not assume that `torchrun --master-addr=127.0.0.1` implies a loopback-only
listener. The Torch Elastic build audited on the current host bound its store
to a wildcard socket. Consequently the no-exposed-ports profile does not use
`torchrun`, DDP, notebooks, dashboards, SSH port forwarding, or inbound firewall
rules.

Place the repository and DuckDB replay on the GPU host, then assign independent
models or seeds across the eight devices:

```bash
uv sync --frozen

CUDA_VISIBLE_DEVICES=0 uv run pvzwm train-wm \
  --database data/port_pixels.duckdb --source python_port \
  --model jepa --regularizer visreg --regularization-weight 0.5 \
  --epochs 100 --batch-size 32 --num-workers 4 \
  --latent-dim 512 --hidden-dim 1024 --slices 4096 \
  --conv-pool-height 3 --conv-pool-width 5 --device cuda \
  --output artifacts/a100_port_jepa_seed0.pt

CUDA_VISIBLE_DEVICES=1 uv run pvzwm train-wm \
  --database data/port_pixels.duckdb --source python_port \
  --model token --token-quantizer fsq --fsq-levels 8,8,4 \
  --sequence-length 8 --sequence-stride 8 \
  --epochs 100 --batch-size 8 --device cuda \
  --output artifacts/a100_token_fsq_seed1.pt
```

Use the remaining devices for matched seeds, regularizers, architectures, or RL
continuations. A single job's `--batch-size` is its full batch size. Only that
process evaluates, writes its checkpoint, and records its experiment, so each
job must use a distinct output and experiment-database path.

The code still supports PyTorch DDP for controlled environments, but it is not
part of this repository's no-listener remote recipe. Re-enable it only inside a
verified network namespace or firewall policy and confirm actual bindings with
`ss -ltnp`; command-line rendezvous addresses alone are not sufficient proof.
The fail-closed runtime additionally requires
`PVZWM_ALLOW_DDP_NETWORK=1` before initializing a multi-process group.
