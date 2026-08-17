# Native simulator roadmap

The concrete precedent is Elliot Arledge's
[minecraftbench](https://github.com/Infatoshi/minecraftbench.com): rewrite a
pinned game version as a PufferLib-style batched C/GPU RL environment, then
score it against a real-game oracle. The key idea is excellent. Its own results
also show why PVZWM should use differential verification and staged fidelity
gates instead of trusting a large one-shot generated rewrite.

PVZWM already embeds a full C++ implementation through PvZ-Portable. That
backend is the fidelity oracle, but its renderer, resource graph, and
one-process-per-game control path are poorly shaped for high-throughput RL.
The native simulator is a separate headless implementation whose primary
contract is batched state transition, not presentation.

The first implementation in `native/` ports the existing 181-action,
192-feature clean-room simulator to allocation-free C11. A batch owns one
contiguous state array, and callers provide contiguous observation, reward,
termination, and legality buffers. Its xorshift RNG is reproducible within the
native backend but intentionally does not pretend to match NumPy or the
authentic engine RNG. Full-game parity will require the authentic RNG/event
schedule or an injected event tape as part of the conformance ABI.

Build and benchmark it:

```bash
make -C native
uv run pytest tests/test_native.py
uv run scripts/benchmark_native_env.py --envs 4096 --steps 10000
```

On macOS the output is `native/build/libpvzwm_native.dylib`; Linux produces a
`.so`. The Python wrapper can also compile into a temporary directory for
isolated tests.

## Accelerator order

Metal is the first accelerator target because the active development machine is
Apple Silicon. CUDA is the compatible training-host target, not the source of
truth.

1. Keep one scalar C transition implementation and a versioned state layout.
2. Add a Metal compute backend using shared `MTLBuffer` storage and one
   environment per grid thread. Apple unified memory avoids a PCIe transfer,
   but command-encoding and framework copies still count.
3. Benchmark complete rollout plus policy inference, not an isolated shader.
   PyTorch MPS does not provide a general public zero-copy bridge for arbitrary
   Metal buffers, so a faster simulator can still lose end to end.
4. Add the CUDA kernel from the same conformance vectors after the Metal path
   is correct. On discrete NVIDIA systems, state and rollout buffers must stay
   device-resident.
5. Retain the CPU C backend as the sanitizer/debug target and behavioral
   reference for both GPU implementations.

The current Command Line Tools installation includes the Metal framework and
Objective-C headers but not the offline `metal`/`metallib` shader compilers.
The first backend can compile embedded MSL with `newLibraryWithSource`; install
full Xcode only when switching to offline `.metallib` builds.

## Fidelity gates

Grow the implementation by mechanics, with tick-level differential tests
against deterministic zero-render observations from the authentic bridge:

1. Day lawn: sun economy, seed recharge, normal zombies, projectiles, mowers,
   flags, and terminal state.
2. Night: graves, mushrooms, sleep/wake state, and no passive sky sun.
3. Pool and fog: six rows, water lanes, lily pads, aquatic zombies, visibility.
4. Roof: flower pots, slope-aware projectiles, bungees, and catapults.
5. Conveyor, bowling, Whack-a-Zombie, and boss-specific interaction modes.
6. All 50 Adventure stages, with seeded event traces compared at every engine
   tick before the native backend replaces authentic collection.

For each gate, compare legal actions, entity spawn/despawn events, health,
currency, wave state, reward inputs, and terminal results. Pixel equivalence is
not a goal for the headless core; rendered evaluation remains the job of
PvZ-Portable.

## Performance rules

- No allocation, Python call, JSON serialization, or rendering in `step`.
- Fixed-capacity state and deterministic per-environment RNG.
- Contiguous buffers at the learner boundary.
- CPU correctness and sanitizer tests before accelerator optimization.
- Profile before moving branch-heavy mechanics to GPU. On an M5, a vectorized
  C batch can beat Metal at small batch sizes because GPU dispatch is not free.
- ABI-version observations and actions. Replays from different semantic
  versions must never mix silently.
