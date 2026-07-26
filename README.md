# PVZWM

PVZWM is a GPL-3.0-only world-model and reinforcement-learning research platform
for Plants vs. Zombies. The authentic-campaign backend is the community
[PvZ-Portable](https://github.com/wszqkzqk/PvZ-Portable) engine, patched with a
process-local stdin/stdout RL bridge. It loads Adventure levels `1-1` through
`5-10`, steps the engine's fixed 800×600 logical board deterministically, and
reports structured state plus legal actions without opening a socket. The
Python/Pygame [PythonPlantsVsZombies](https://github.com/marblexu/PythonPlantsVsZombies)
adapter and clean-room symbolic environment remain useful fast testbeds.

Campaign research pages can be revision-pinned into DuckDB with `uv run pvzwm
sync-campaign-wiki --pretty`. The archive retains source URLs, revision IDs,
retrieval time, SHA-256, sequential predecessor/successor IDs, and the wiki's
CC-BY-SA-4.0 attribution boundary.

The older Python port remains deliberately labelled as partial: it provides
`1-1`, `1-2`, and mechanics-reduced proxies, not a full campaign. The new C++
bridge has verified deterministic resets for representative day, night, pool,
fog, roof, and boss stages, including intro-placed mowers, graves, and flower
pots. Full sequential-policy validation across all 50 stages is still pending;
special interaction modes need end-to-end action-coverage tests before being
called solved.

The architectural comparison is grounded in the current primary literature,
especially [LeWorldModel](https://arxiv.org/abs/2603.19312),
[VISReg](https://arxiv.org/abs/2606.02572),
[DreamerV3](https://github.com/danijar/dreamerv3),
[IRIS](https://github.com/eloialonso/iris),
[DIAMOND](https://github.com/eloialonso/diamond), and
[TD-MPC2](https://github.com/nicklashansen/tdmpc2). See
[the research map](docs/RESEARCH.md) for the full comparison and data policy.
See [the data boundary](docs/DATA.md) for the exact distinction between synthetic
transitions, local wiki assets, and original-game captures.
See [the baseline experiment log](docs/EXPERIMENTS.md) for matched MPS results
and representation-collapse diagnostics.

## Quick start

```bash
uv sync --dev
uv run pvzwm collect-port --episodes 10 --policy heuristic \
  --database data/port_pixels.duckdb
uv run pvzwm collect-port --episodes 10 --policy random --observation vector \
  --database data/port_vectors.duckdb
uv run pvzwm collect --episodes 10 --policy random
uv run pvzwm collect --episodes 10 --policy heuristic
uv run pvzwm train-wm --model mlp --epochs 5
uv run pvzwm train-wm --model ensemble --ensemble-size 5 --epochs 5 \
  --output artifacts/ensemble.pt
uv run pvzwm train-wm --model jepa --regularizer visreg --epochs 5 \
  --output artifacts/jepa_visreg.pt
uv run pvzwm train-wm --database data/port_pixels.duckdb --model lewm \
  --regularizer visreg --sequence-length 16 --slices 1024 \
  --output artifacts/lewm_visreg.pt
uv run pvzwm train-wm --model jepa --regularizer sigreg --epochs 5 \
  --output artifacts/jepa_sigreg.pt
uv run pvzwm train-wm --model jepa --regularizer vicreg --epochs 5 \
  --output artifacts/jepa_vicreg.pt
uv run pvzwm train-wm --model rssm --sequence-length 16 --epochs 5 \
  --output artifacts/rssm.pt
uv run pvzwm train-wm --database data/port_pixels.duckdb --model token \
  --token-quantizer vq --sequence-length 8 --epochs 10 \
  --output artifacts/token_vq.pt
uv run pvzwm train-wm --database data/port_pixels.duckdb --model token \
  --token-quantizer fsq --fsq-levels 8,8,4 --sequence-length 8 --epochs 10 \
  --output artifacts/token_fsq.pt
uv run pvzwm train-wm --database data/port_pixels.duckdb --model flow \
  --flow-latent-height 3 --flow-latent-width 5 --device mps \
  --output artifacts/representation_flow.pt
uv run pvzwm collect --observation pixels --episodes 10 \
  --database data/pixel_replay.duckdb
uv run pvzwm train-wm --database data/pixel_replay.duckdb --model jepa \
  --device mps --output artifacts/pixel_jepa_visreg.pt
uv run pvzwm train-wm --database data/port_replay.duckdb --source python_port \
  --model jepa --device mps --output artifacts/port_jepa_visreg.pt
uv run pvzwm eval-wm --database data/port_replay.duckdb \
  --checkpoint artifacts/port_jepa_visreg.pt --split validation --device mps
uv run pvzwm eval-planner --checkpoint artifacts/port_ensemble.pt \
  --environment python-port --port-level 1 --device mps
uv run pvzwm train-agent --agent dqn --steps 50000 \
  --output artifacts/dqn.pt
uv run pvzwm train-agent --agent ppo --steps 50000 \
  --output artifacts/ppo.pt
uv run pvzwm train-agent --agent dqn --environment python-port \
  --steps 50000 --output artifacts/port_dqn.pt
uv run pvzwm train-agent --agent ppo --environment python-port \
  --steps 50000 --output artifacts/port_ppo.pt
uv run pvzwm train-agent --agent bc --environment portable \
  --database data/success-curriculum-level1-3.duckdb \
  --bc-success-only --bc-informative-fraction 0.5 --bc-epochs 100 \
  --output artifacts/success_bc.pt
uv run pvzwm train-agent --agent bc --environment portable \
  --database data/frontier-level4-elite-gen0-1-200.duckdb \
  --bc-top-fraction 0.2 --bc-informative-fraction 0.5 --bc-epochs 100 \
  --output artifacts/level4_top_return_bc.pt
uv run pvzwm train-agent --agent ppo --environment portable \
  --initialize-from artifacts/success_bc.pt --portable-level 1 \
  --portable-curriculum-end 3 --portable-rehearsal-probability 0.3 \
  --steps 50000 \
  --output artifacts/success_bc_ppo.pt
uv run pvzwm train-agent --agent ppo --environment portable \
  --portable-binary vendor/PvZ-Portable/build/pvzwm-bridge \
  --portable-resources data/pvz-goty --portable-level 4 \
  --portable-reward-mode clipped-damage --steps 50000 \
  --output artifacts/level4_reward_ablation.pt
uv run pvzwm train-agent --agent dyna-dqn --environment python-port \
  --port-campaign-stage 1-3 --database data/port_vectors.duckdb \
  --world-model artifacts/port_ensemble.pt --model-ratio 0.5 \
  --model-rollout-horizon 1 --steps 50000 --output artifacts/dyna_dqn.pt
uv run pvzwm eval-agent --checkpoint artifacts/dqn.pt --episodes 20
uv run pvzwm eval --policy heuristic --episodes 20
uv run pytest
uv run pvzwm data-status --pretty
uv run pvzwm merge-replay --database data/combined.duckdb \
  data/shard_a.duckdb data/shard_b.duckdb
```

The resumable authentic-campaign runner checkpoints every training chunk and
restores every already-completed level as rehearsal support. Its frontier moves
only on ordered terminal wins:

```bash
uv run scripts/run_full_campaign.py \
  --checkpoint artifacts/frontier-level10-ppo-seq10-g1-highent-400k.pt \
  --output-directory artifacts/full-campaign \
  --portable-binary vendor/PvZ-Portable/build/pvzwm-bridge \
  --portable-resources data/pvz-goty \
  --start-level 11 --completed-through 10 --final-level 50 \
  --device cuda
```

Authentic resized pixel trajectories for environment-model training can be
collected from every frozen frontier policy without changing its action rule:

```bash
uv run scripts/collect_stage_policy_pixels.py \
  --database data/authentic-policy-pixels.duckdb \
  --levels 1-10 2-1 2-2 --episodes 2 --width 160 --height 120 \
  --device cuda
```

Consolidate the frozen specialists into one level-conditioned actor before
full-prefix RL rehearsal:

```bash
uv run scripts/distill_stage_teachers.py \
  --manifest data/stage_policy_library.json \
  --initialize-from artifacts/frontier-level13-from-reinforce-default-400k.pt \
  --levels 1-1 1-2 1-3 1-4 1-5 1-6 1-7 1-8 1-9 1-10 2-1 2-2 2-3 \
  --episodes 2 --updates 5000 --batch-size 512 \
  --output artifacts/prefix-1-1-to2-3-distilled.pt --device cuda
```

Teacher collection uses authentic on-policy states, while every optimization
batch gives each level equal mass. `scripts/eval_prefix_policy.py` then tests
that single checkpoint on every stage; routed teacher results are not counted
as student generalization.

## Authentic full-game backend

Install the SDL development dependencies listed by PvZ-Portable, then build the
game and private pipe bridge:

```bash
scripts/setup_portable_backend.sh
```

Place `main.pak` and `properties/` under one resource directory. They are not
stored in this repository. Verify representative sequential Adventure levels
through the Python wrapper with `uv run`:

```bash
uv run scripts/smoke_portable.py \
  --binary vendor/PvZ-Portable/build/pvzwm-bridge \
  --resources data/pvz-goty
```

Collect visual transitions from any sequential stage range into DuckDB:

```bash
uv run scripts/collect_portable.py \
  --binary vendor/PvZ-Portable/build/pvzwm-bridge \
  --resources data/pvz-goty \
  --database data/portable_visual.duckdb \
  --levels 1-1..5-10 --episodes-per-level 1 \
  --width 160 --height 120 --ticks-per-step 25
```

Collect compact authentic semantic dynamics without framebuffer transfer, hold
out complete stages, and train legality-aware ensemble dynamics:

```bash
uv run scripts/collect_portable.py \
  --binary vendor/PvZ-Portable/build/pvzwm-bridge \
  --resources data/pvz-goty \
  --database data/portable_vectors.duckdb \
  --levels 1-1..5-10 --episodes-per-level 1 \
  --observation vector --ticks-per-step 25
uv run pvzwm train-wm --database data/portable_vectors.duckdb \
  --source pvz_portable_vector --split-by-metadata level_id \
  --model ensemble --ensemble-size 5 --predict-legality \
  --output artifacts/portable_ensemble.pt
```

Search learned-policy episodes while retaining full logical-resolution video only for the first
win. The policy remains on the compact vector path; zero-tick engine observations
sample 800×600 frames every eight decisions, and failed candidate videos are
discarded:

```bash
uv run scripts/record_portable_agent.py \
  --checkpoint artifacts/frontier-level4-lh-g1-l1-e01-75k.pt \
  --database data/level4_video_search.duckdb --level 4 \
  --observation vector --episodes 100 --stochastic --stop-on-win \
  --video-output artifacts/level4_win.mp4 \
  --video-frame-stride 8 --width 800 --height 600
```

The game canvas is fixed at 800×600 logical pixels. PvZ-Portable initially
creates an 800×600 desktop window, but that SDL window is resizable; its options
dialog only controls fullscreen. Fullscreen uses the desktop drawable, and the
current renderer fits the 4:3 game area inside it with centered letterboxing.
Model observations bypass presentation scaling: the bridge renders directly
into a private 800×600 OpenGL framebuffer inside the engine process, then
deliberately downsamples only when a smaller training frame is requested. Human
capture crops the centered 4:3 presentation rectangle before exporting canonical
800×600 logical-resolution playback. Use an 800×600 Xvfb screen to avoid any
intermediate resampling. `xvfb-run` uses a local X display with no TCP listener,
and the RL protocol itself uses only inherited stdin/stdout pipes.

New portable training uses a 3,600-decision safety budget so multi-flag stages
can reach their terminal wave. Older checkpoints retain their historical cap;
override it explicitly with `eval-agent --portable-episode-steps 3600` or
`record_portable_agent.py --episode-steps 3600`. Evaluations report both the
effective cap and truncation rate so a time-limit cutoff cannot be mistaken for
a game loss or a shaped-return success.

- Protocol-v3 legality that exposes board clicks only in board-interaction modes,
  plus explicit boss creation when the headless reset skips the 5-10 intro.
- Authentic semantic replay collection and whole-stage held-out splits. The
  current vector-v2 corpus has 585 features and 12,552 transitions covering all
  50 stages; its content hash and exact layout are in
  `data/portable_rich_v2_manifest.json`. The 250-feature vector-v1 corpus remains
  available as a versioned comparison.

`--device auto` prefers Apple MPS, then CUDA, then CPU. Numerically sensitive
VISReg computations remain float32 on MPS. For the optional one-node 8×A100
path, see [configs/README_8XA100.md](configs/README_8XA100.md). The secure remote
profile assigns independent single-process jobs to GPUs so no Torch Elastic
rendezvous listener is opened; CUDA autocast still uses bfloat16.

## Implemented experiment surface

- A headless, deterministic-time adapter for `plants-vs-zombies==0.0.5` that
  records both upstream sprite frames and structured internal state to DuckDB.
  It uses automatic sun pickup because the shared placement action space does
  not yet contain a sun-click action.
- An ordered 50-stage Adventure manifest that keeps authentic stage IDs separate
  from six upstream demo files. The current shared-action adapter supports demos
  0–2, partially supports demo 3, and does not yet support conveyor/bowling demos.
- A compiled PvZ-Portable bridge with deterministic Adventure reset/step,
  an explicit authentic seed-chooser API, unlocked-seed and packet-capacity
  observations, authentic lawn-item setup, a fixed 650-action
  semantic vocabulary, legal-action masks, structured plants/zombies/seed
  state, and direct logical-framebuffer RGB observations over newline-delimited
  JSON on stdin/stdout. The old biome-aware chooser remains only as a replay
  compatibility default; learned runs can select the ordered episode loadout.
- Portable control treats a legally available sun pickup as a deterministic
  `COLLECT` macro-action. The transition remains in replay for world-model
  learning, but the singleton action mask gives the actor no strategic choice
  or gradient on pickup; spending, timing, placement, seed use, and zero-sun
  behavior remain learned.
- A win-gated sequential curriculum wrapper that starts at a configured
  Adventure stage and unlocks exactly the next stage after a completed win.
- The full-campaign runner accepts a frontier-specific learned loadout archive
  template (`{level}` or `{stage}`). In this mode it trains one authentic
  frontier at a time, promotes only a terminal win, then requires a new archive
  for the newly unlocked level instead of silently carrying one deck across
  different stages.
- A revision-pinned wiki campaign archive in DuckDB, with all 50 level pages and
  explicit predecessor/successor progression metadata.
- A seedable five-lane PvZ-like environment with sun economy, placement masks,
  plants, zombies, combat, lawn mowers, rewards, and terminal states.
- Structured action encoding for wait, three plant types, and shovel across all
  cells (181 categorical choices on a 5×9 lawn).
- DuckDB episode, transition, and experiment tables with NumPy-safe replay.
- Raw-row DuckDB shard merging for parallel, noncontending collection.
- Whole-episode, source-aware train/validation splits plus open-loop state/latent,
  reward, and termination metrics that never cross episode boundaries.
- Episode-safe recurrent windows and a calibrated, source-linked PvZ1 core stat
  fixture under `data/reference/`.
- Trainable residual MLP dynamics, bootstrap epistemic deep ensembles, Gaussian
  RSSM, an action-interleaved causal Transformer with VQ or FSQ pixel tokens,
  a from-scratch representation-autoencoder latent flow model, a one-step
  action-conditioned JEPA, and a separate causal history-conditioned LeWM with
  VISReg by default and SIGReg retained as an ablation.
  On the first authentic held-out-stage pixel split, matching VISReg's
  view/batch geometry and BatchNorm projector eliminates encoder collapse
  (100% active features), although the current predictor remains 57% below
  latent persistence; a LeWM-objective correction and matched VQ baseline are
  active rather than overstating that encoder result as a solved world model.
- A clean-room RGB renderer for from-scratch pixel-model development without
  redistributing copyrighted assets.
- Random and economy/defense diagnostics, trainable masked Double-DQN and PPO,
  successful- and top-return-replay behavior cloning, flat or portable-spatial
  actors, checkpointed Random Network Distillation, discrete IQL with twin
  critics and advantage-weighted actors, configurable PPO/GAE credit assignment,
  PPO initialization from distilled checkpoints, Dyna Double-DQN trained on real plus uncertainty-filtered
  learned-model transitions, checkpoint evaluation, and legality- and
  uncertainty-aware categorical CEM planning.
- CLI collection, model training, checkpointing, and baseline evaluation.
- Full-logical-resolution learned-policy recording via
  `uv run scripts/record_portable_agent.py`; a success-replay policy has
  held-out stochastic wins through 2-2. The initial 1-7 audits were invalidated
  when semantic traces showed the 1,200-decision watchdog truncating healthy
  play at waves 11–14 of 20. With a 3,600-decision full-stage budget, the
  selected RND-PPO specialist wins 4/20 on 1-7; sequential continuations win
  2/20 on 1-8, 3/20 on 1-9, and 14/20 on 1-10. The same 1-10 checkpoint wins
  8/10 on 2-1 with zero night updates. On an identical 20-seed 2-2 block, a
  400k-step default PPO continuation wins 13/20 versus 9/20 for a 20k-step
  REINFORCE continuation; default PPO is therefore selected. Replaying the
  same seeds with deterministic sun pickup gives 12/20, a one-win decrease
  that establishes the operational `collect_sun_first_v1` baseline. The next
  default PPO continuation wins 20/20 held-out 2-3 seeds under that baseline.
  The same checkpoint then wins 16/20 on 2-4 with zero 2-4 updates. Its first
  mechanic-shift test is 0/20 on 2-5 Whack-a-Zombie, where board clicks become
  legal for the first time. Flat PPO raises mean return from -96.2875 to
  -77.727 but remains 0/20; a matched high-entropy/RND continuation is also
  0/20 and falls back to -94.531. Critic-free spatial REINFORCE then produces
  the first authentic held-out 2-5 win (1/20, seed 213016), opening 2-6 in
  sequence; the same checkpoint wins 15/20 on 2-6 with zero 2-6 updates and
  then records sequential zero-shot wins through 2-10, completing the night
  world. The newly exposed frontier is 3-1, the first pool board. A local-
  convolution spatial PPO is now selected on 2-5 at 7/20 held-out wins and
  exact mean return 20.213, outperforming ordinary spatial PPO, GRPO, and
  offline IQL under the same environment-return objective.
  A direct
  one-network 2-1→5-10 curriculum is active, but the full
  campaign is not yet claimed solved. Earlier specialists remain routed by
  stage for regression safety. Exact
  checkpoints, hashes, corrected horizons, and limitations are in
  `data/sequential_frontier_manifest.json`.
- An optional `clipped-damage` portable reward mode that removes negative wave-
  spawn health deltas while preserving +100 wins and −100 losses. The mode is
  stored in checkpoints and does not replace the default raw bridge reward.
- Episode-safe open-loop residual/ensemble training via
  `--dynamics-rollout-length`; prediction and planning checkpoints are selected
  separately because lower held-out MSE did not imply stronger CEM control.

The clean-room simulator is an infrastructure testbed, not a frame-perfect
recreation. The installed Python port bundles its own sprites; upstream describes
the project as for personal learning and noncommercial use. Port-generated data
is tagged `python_port`, while actual game captures are tagged separately as
`live_screen` with `game_id=original_pvz`.

## Licensing and provenance

PVZWM is licensed under `GPL-3.0-only`; see `LICENSE` and
`LICENSES/GPL-3.0-only.txt`. The selected Python port is an external dependency.
Its PyPI record has conflicting license fields (an MIT free-text field and a
GPLv3 classifier), so replay metadata preserves the exact package version,
upstream URL, project license choice, and asset-use notice instead of silently
collapsing those distinctions.

## Live screen recording

Copy `configs/screen.example.json`, replace its absolute coordinates with the
game window's capture area, lawn bounds, seed packets, and shovel, then prepare
an action script like `configs/actions.example.txt`.

```bash
uv sync --extra live

# Capture-only dry run: never moves the pointer.
uv run pvzwm record-live --geometry configs/screen.local.json \
  --actions configs/actions.example.txt --database data/live_replay.duckdb

# Explicit opt-in after confirming calibration and macOS permissions.
uv run pvzwm record-live --geometry configs/screen.local.json \
  --actions configs/actions.example.txt --database data/live_replay.duckdb --execute

# Live recordings currently have action labels but not reward/terminal labels.
uv run pvzwm train-wm --database data/live_replay.duckdb --model jepa \
  --reward-weight 0 --done-weight 0 --device mps \
  --output artifacts/live_jepa.pt

# Train the clean-room perception bootstrap and inspect a dry-run policy loop.
uv run pvzwm train-perception --device mps --output artifacts/perception.pt
uv run pvzwm play-live --geometry configs/screen.local.json \
  --perception artifacts/perception.pt --agent artifacts/dqn.pt --steps 10
```

The macOS click backend imports Quartz only when `--execute` is present and
requires Screen Recording and Accessibility permission. Capture-only mode is the
default. Live episodes explicitly mark reward values as unlabelled placeholders.
`play-live` likewise only prints its predicted action and click coordinates until
`--execute` is explicitly supplied. Synthetic perception is a bootstrap; real
captures should be labelled and used for domain adaptation before trusting it.
