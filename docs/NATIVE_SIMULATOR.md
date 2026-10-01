# PufferLib CUDA environment

The active high-throughput environment target is a CUDA C implementation for
PufferLib. PvZ-Portable remains the full-game fidelity oracle. The older scalar
C simulator remains a test oracle during migration, but it is not the intended
rollout runtime.

The PufferLib overlay lives in `pufferlib/`. Its single environment source,
`ocean/pvzwm/pvzwm.cu`, owns the transition rules, PufferLib GPU lifecycle, and
CUDA kernels. Runtime and training configuration lives in
`config/pvzwm.ini`.

Copy the overlay into a PufferLib checkout and build the CUDA source explicitly:

```bash
cp -R pufferlib/ocean/pvzwm /path/to/PufferLib/ocean/
cp pufferlib/config/pvzwm.ini /path/to/PufferLib/config/
cd /path/to/PufferLib
./build.sh pvzwm --cu
./puffer train
```

## Action-conditioned contract

The action space is a single 181-class categorical head. This preserves the
existing action codec exactly:

- `0`: wait;
- `1..45`: plant sunflower by row-major cell;
- `46..90`: plant peashooter by row-major cell;
- `91..135`: plant wall-nut by row-major cell;
- `136..180`: shovel by row-major cell.

Every environment step consumes the requested action from PufferLib's device
action buffer before advancing economy, combat, and time. PufferLib stores that
action beside the resulting observation, reward, and terminal in its rollout.
That stored action is the conditioning signal for a dynamics or world model:
`action[t]` produces `observation[t+1]`.

The current PufferLib GPU constructor does not pass its dynamic action-mask
buffer into GPU environments. Illegal requested actions therefore advance as a
wait action and add the configured `invalid_action_penalty`. This keeps every
categorical action deterministic and preserves the requested action in the
rollout. If the GPU ABI later exposes the mask buffer, legality can move into
the sampler without changing action IDs or transition rules.

## Device layout

One `Env` is allocated per PufferLib rollout slot. State is fixed-capacity and
contains no pointers or device-side allocation. Eight CUDA lanes cooperate per
environment: lane zero applies the action and advances state, then all lanes
write the 192 bfloat16 observation features. Rewards and terminals are written
directly to PufferLib buffers on the bound rollout stream.

Terminal steps retain their final reward and terminal flag, append episode
statistics to the PufferLib log, and reset the environment before writing the
next observation. This matches PufferLib's CUDA environment lifecycle and
keeps CUDA graph capture valid.

## Fidelity gates

Grow mechanics behind differential checks against deterministic zero-render
observations from the authentic bridge:

1. Day lawn: sun economy, seed recharge, normal zombies, projectiles, mowers,
   flags, and terminal state.
2. Night: graves, mushrooms, sleep/wake state, and no passive sky sun.
3. Pool and fog: six rows, water lanes, lily pads, aquatic zombies, and
   visibility.
4. Roof: flower pots, slope-aware projectiles, bungees, and catapults.
5. Conveyor, bowling, Whack-a-Zombie, and boss-specific interactions.
6. All 50 Adventure stages with seeded event traces compared at every engine
   tick before the CUDA backend replaces authentic collection.

For each gate, compare requested actions, executed effects, entity
spawn/despawn events, health, currency, wave state, reward inputs, and terminal
results. Pixel equivalence is not a goal for the headless core; rendered
evaluation remains the job of PvZ-Portable.

## Performance rules

- No allocation, Python call, JSON serialization, rendering, or host transfer
  in `puf_step`.
- Fixed-capacity state and deterministic per-environment RNG.
- PufferLib-owned contiguous device buffers at the learner boundary.
- One stable action ABI so replay data never loses its conditioning meaning.
- CUDA graph-safe reset and step paths.
- Profile complete rollout plus policy inference, not an isolated transition
  kernel.
