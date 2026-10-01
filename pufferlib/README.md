# PufferLib CUDA environment overlay

This directory contains the device-only PVZWM environment for PufferLib. The
runtime environment is CUDA C and follows PufferLib's GPU environment ABI:

- one device-resident `Env` per rollout slot;
- `puf_vec_create`, `puf_bind_stream`, `puf_reset`, `puf_step`, and `puf_close`;
- bfloat16 observations written directly into PufferLib's device buffer;
- float actions, rewards, and terminals written without host transfers; and
- automatic episode reset after a terminal transition.

Overlay the two directories onto a PufferLib checkout:

```bash
cp -R pufferlib/ocean/pvzwm /path/to/PufferLib/ocean/
cp pufferlib/config/pvzwm.ini /path/to/PufferLib/config/
cd /path/to/PufferLib
./build.sh pvzwm --cu
./puffer train
```

The action space remains one categorical head with 181 actions. Action zero is
wait. The remaining actions are four contiguous 45-cell blocks: sunflower,
peashooter, wall-nut, and shovel. PufferLib records the requested action in the
rollout next to the resulting observation, reward, and terminal, so downstream
models have an explicit action-conditioning signal: `action[t]` produces
`observation[t+1]`.

PufferLib's current GPU environment constructor does not expose its dynamic
action-mask buffer to the environment. An action that is unavailable because
of sun, cooldown, or occupancy therefore executes as wait and receives
`invalid_action_penalty`. The requested action is still retained in the
rollout, so this behavior is deterministic and learnable rather than silently
substituting another plant action.
