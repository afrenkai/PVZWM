# Data boundary

## Python-port trajectories

`uv run pvzwm collect-port` executes `plants-vs-zombies==0.0.5` through a
headless Pygame adapter. Episodes use source `python_port` and game id
`marblexu_python_port`. They can contain compact 160×120 sprite frames or the
shared 192-value board vector. They are higher fidelity than the clean-room
simulator, but they are not original-game captures.

The upstream names `level_0.json` through `level_5.json` are recorded as
`port-demo-0` through `port-demo-5`. Their `campaign_stage` is always null; they
must never be interpreted as Adventure `1-1` through `1-6`.

The adapter records the upstream URL, exact package version, timing settings,
automatic-sun setting, GPL-3.0-only project choice, and upstream's
personal-learning/noncommercial asset notice in every episode.

The broad remote port replay is intentionally not copied into the local tree.
Its verification hash and exact scenario/transition composition live in
`data/remote_broad_manifest.json`. It contains rendered frames from the selected
Python port, not captures from the original PopCap executable.

## Sequential campaign proxies

The port adapter can load partial `1-1`/`1-2` definitions and 14 regular-wave
proxies (`1-3`, `1-4`, `1-6`–`1-9`, `2-1`–`2-4`, and `2-6`–`2-9`). Each proxy
reads its wave count, listed wave compositions, zombie roster, location, and
reward unlock from `data/campaign.duckdb`. It records the exact wiki revision
and separately lists inferred timing, rows, unlisted-wave fill, and unsupported
mechanic substitutions. These stages are useful for learning dynamics and a
sequential curriculum, but they are not authentic reconstructions or evidence
that the whole game is RL-playable.

`uv run pvzwm data-status --pretty` is the authoritative local audit. It reports
each data source separately and never treats simulator transitions, wiki icons,
or a browser clone as original-game experience.

## Data tiers

| Tier | Source | Can train dynamics? | Current role |
|---|---|---:|---|
| Reference metadata | wiki.gg page text, CC-BY-SA-4.0, with revision/source URLs | No | Campaign implementation and calibration |
| Wiki assets | Local-only, hashed PvZ1 icons | No, not alone | Perception/domain-randomization references |
| Symbolic transitions | Clean-room `PVZEnv` | Yes, synthetic only | Algorithm and MPS development |
| Rendered transitions | Clean-room rasterizer | Yes, synthetic only | Pixel pipeline bootstrap |
| Python-port transitions | GPL Pygame port, tagged `python_port` | Yes, port domain only | Main controllable visual RL environment |
| PvZ-Portable engine transitions | GPL/LGPL community engine plus user-supplied PvZ resources, tagged `pvz_portable_*` | Yes | Authentic 50-stage mechanics, semantic RL, and visual world models |
| Original-game captures | Local PvZ window plus recorded actions | Yes | Required real target domain |

Authentic portable trajectories use action semantics
`collect_sun_first_v1`: whenever the bridge reports collectible sun, the
effective legal mask contains only `COLLECT`. This is recorded as an ordinary
transition so dynamics learning sees the pickup, while strategic control begins
again from the resulting state.

The first frozen-policy authentic pixel corpus is versioned in
`data/authentic_stage_policy_pixels_manifest.json`. It contains 9,537 complete
160×120 transitions across 1-10, 2-1, and 2-2: five winning episodes and one
terminal 2-2 loss. The loss is intentionally retained for reward/termination
and failure-dynamics coverage. It is the first corpus used for authentic causal
LeWM training; the earlier broad port corpus remains a Python-port architecture
comparison rather than authentic-game evidence.
Authentic held-out-stage world-model runs, hashes, collapse diagnostics, and
the positive-skill acceptance rule are tracked separately in
`data/authentic_world_models_manifest.json`.

### Remote DuckDB pixel storage

Pixel replay database work runs only on `ubuntu@129.213.92.137`; the local tree
contains code and manifests, not a local copy of the optimized database. The
selected remote copy is
`data/authentic-stage-policy-pixels-1-10-to2-2-compact-rg4096.duckdb`: 447,492,096
bytes with SHA-256
`5ca7efe212140239d8d78bf59ebcbd6e6c0282f2deca04f78bdd4179c5f48d3b`.
Byte-level SQL verification found no mismatched episodes, experiments, or
transitions relative to the 1,776,562,176-byte source.

The compact schema stores each observation frame once and stores an explicit
`next_observation` only for the final transition of each episode. Readers
reconstruct internal next frames from the following row. New databases use
4,096-row row groups, one transaction per complete episode, a 256 MB checkpoint
threshold, and no redundant transition ART index. For the authentic corpus,
9,531 of 9,537 duplicated next-frame BLOBs disappear. Materializing all 2,375
episode-safe sequence windows took a median 19.03 seconds versus 25.79 seconds
from the original database, while disk use fell 74.8%.

DuckDB reported both raw NumPy and application-zlib image BLOB segments as
uncompressed. The durable DuckDB layout therefore keeps per-frame zlib. A
remote 1,000-frame probe also found a promising derived training format:
DuckDB-produced raw-frame/Zstd Parquet used 9,796,808 bytes versus 47,984,640
bytes for the transactional zlib DuckDB sample, with median read-and-decode
times of 0.346 and 0.736 seconds. This is recorded as a frozen-snapshot
candidate, not a replacement for transactional collection. Exact runs and
official documentation links are in
`data/remote_duckdb_storage_manifest.json`.

### Private replay snapshot

The complete checkpointed replay corpus is stored in the private Hugging Face
dataset `afrenkai/PVZWM-replays`. The immutable snapshot contains 470 DuckDB
databases totaling 26,126,327,808 bytes. Fifteen copied WAL-backed databases
were recovered and checkpointed before upload; the source files were not
modified.

`data/replay_storage_manifest.json` records the remote revision, SHA-256 digest,
size, schema, table counts, WAL status, and `hf://` URI for every database.
`data/artifact_storage_manifest.json` separately records 1,070 trained
checkpoints, logs, evaluation records, media files, and raw reference assets
totaling 1,195,780,155 bytes.
Restore the corpus with:

```bash
uvx --from huggingface-hub hf download afrenkai/PVZWM-replays \
  --repo-type dataset --local-dir pvzwm-replays
```

Verify every remote object against the Git-tracked manifest with:

```bash
uv run --no-project --with huggingface-hub \
  python scripts/verify_replay_remote.py data/replay_storage_manifest.json
```

Verify the non-database research artifacts with:

```bash
uv run --no-project --with huggingface-hub \
  python scripts/verify_artifact_remote.py data/artifact_storage_manifest.json
```

Create a new non-destructive snapshot with
`scripts/snapshot_replay_corpus.py`. Do not back up a WAL without its matching
database file.

`data/portable-vectors-rich-v2-50.duckdb` is the current local semantic corpus.
Its manifest records the SHA-256, exact stage/seed range, 585-value vector-v2
layout, 650-action vocabulary, and protocol version. Protocol v3 removed ordinary-play
board clicks from legal masks while preserving them for Whack-a-Zombie and
Scary Potter modes. The corpus covers resets and partial trajectories for every
stage; only 1-1 currently has a learned-policy completion benchmark.

New collections default to semantic vector version 2 (585 values). It preserves
the v1 grids while adding plant state, zombie type, sub-cell position, separate
body/helm/shield health, zombie phase, campaign timing, scene, tutorial, and
entity-count features. Checkpoints retain their vector version; pre-versioned
250-input checkpoints are restored explicitly as v1.

Vector v2 remains an observation rather than a complete engine serialization.
Projectile state, coin positions, lawn mowers, grid items, and internal RNG are
latent; semantic models therefore solve a partially observed prediction task.
The pixel track observes visible effects but still cannot recover hidden RNG.

The earlier `data/portable-vectors-v3-50.duckdb` is retained as a compact
vector-v1 comparison. Its separate manifest prevents its 250-feature layout
from being mixed with current 585-feature checkpoints.

`data/portable-vectors-rich-v2-policy-augmented.duckdb` merges the 50-stage
coverage corpus with 50 stochastic-PPO episodes from 1-2. It contains 100
episodes and 34,718 transitions, including one recorded 1-2 win. The source
shard hashes, action distribution, fixed held-out stages, and limitations are
recorded in `data/portable_policy_augmented_manifest.json`.

The success-curriculum corpora preserve the trajectory lineage used for
planner distillation and iterative policy/world-model learning:

| Corpus | Episodes | Transitions | Successful episodes | SHA-256 |
|---|---:|---:|---:|---|
| `data/cem-demo-level1-8.duckdb` | 8 | 1,835 | 8 on 1-1 | `3f09636a51ac44e089c18490624b262ade46482bf2ec0169c66eb5e857c065d4` |
| `data/success-curriculum-level1-2.duckdb` | 48 | 20,963 | 8 on 1-1, 8 on 1-2 | `a0df3e41ab4657c890ccf857e985bb8c0ae3c9472f33056b206c1a589afb834e` |
| `data/success-curriculum-level1-3.duckdb` | 128 | 56,647 | 8 on 1-1, 8 on 1-2, 4 on 1-3 | `16a63e1f518012af46a13dc431dab8541d1b549b3c3998733253ceecf1c5da3e` |
| `data/success-curriculum-level1-4.duckdb` | 328 | 150,497 | 8 on 1-1, 8 on 1-2, 4 on 1-3, 4 on 1-4 | `55fb8d0f1d5b21b414d6442201be03845211c66a20b598d3f29f9fc09be8edfa` |

The level-1-4 corpus has 24 successful episodes and 14,803 transitions belonging
to successful episodes. These are selected training corpora, not evaluation sets.
Exact producers, hashes, per-level success counts, and partial-observability
limitations are in `data/success_curriculum_manifest.json`.

Level 1-4 was the first difficult frontier. `data/frontier-level4-random-100.duckdb`
contains 100 complete legality-aware exploration episodes and 40,144
transitions, with zero wins. It is merged with the 50-stage coverage and
1-1→1-3 success corpora in
`data/portable-vectors-rich-v2-success-level1-4-frontier.duckdb`: 278 episodes
and 109,343 transitions (SHA-256
`d418ee2ad41a78a17a129d1832be666f4bb34a19896be033befe6a291c70c229`).
This original frontier shard is useful failure-dynamics coverage, not a success
corpus. Subsequent stochastic learned-policy evaluation produced four authentic
1-4 wins across two independent 100-episode generations. Their combined replay
is `data/frontier-level4-elite-gen0-1-200.duckdb`: 200 episodes and 93,850
transitions (SHA-256
`5290ef883c584ecabee747d9fe87d333cb5ac935838b14086919c533da0515fa`).
A third 100-episode policy generation produced zero wins and is retained
separately rather than hidden. Exact components, seeds, policy lineage, and
limitations are recorded in `data/frontier_level1_4_manifest.json` and
`data/level4_policy_iteration_manifest.json`.

`data/frontier-level4-video-search-85000.duckdb` preserves the 43 vector
episodes used to find a full-logical-resolution visual win at seed 85042: 18,721 transitions and
one win (SHA-256
`0d2f91c105be15792b22721d14d9b768de51611664d94cee247ff21b54ada23a`).
Its paired 800×600 MP4 samples the same live episode through zero-tick engine
observations, bypassing the resizable SDL presentation; the frame requests do
not alter the action or simulation clock.

The elite-augmented world-model corpus merges 50-stage coverage, earlier
success/frontier data, and the two successful generations. It lives at
`data/portable-vectors-rich-v2-level4-elite-augmented.duckdb` with 478 episodes,
203,193 transitions, and SHA-256
`53fc89d41db710d44f348e6cfb3156d5c5119442f5f204319a8ea6fcfa0b7510`.

Discrete IQL uses `data/frontier-level4-offline-iql-243.duckdb`: 243 complete
1-4 episodes, 112,571 transitions, and five wins (SHA-256
`8817a2aaf4140a787b79af2fc80e41b1a40431e0ddff49c7524433364f93ec87`).
The selected IQL actor generated one additional win in the independent
100-episode `data/frontier-level4-iql-selected-record-92000-100.duckdb`.

The selected level-1-6 actor has a separate 50-episode replay with nine wins and
29,153 transitions at `data/frontier-level6-selected-record-98000-50.duckdb`
(SHA-256
`972ad272eb13dda386d3f9e3abab78e1ff221bf7e18a52575351522fa28feb21`).
These trajectories, the additional IQL batch, and the earlier elite corpus form
`data/portable-vectors-rich-v2-sequential-level6-augmented.duckdb`: 628 episodes
and 278,577 transitions (SHA-256
`f06638ddd070b25ec11d1d44013afed5f72083a6f6dc872468271813d01ab230`).
The exact win-gated checkpoint lineage through solved level 1-8 is in
`data/sequential_frontier_manifest.json`.

Level-1-7 policy iteration starts with
`data/frontier-level7-rollouts-combined-300.duckdb`: 300 episodes and 157,003
transitions (SHA-256
`2b17e72561ed458f8919cb2694921277166c2a229b0e35914ce9f2030c8ece85`).
All 300 have explicit `won=false`; 22 positive-return time-limit episodes had
previously been misclassified by the return-only success heuristic. Corrected
3,600-step evaluation of the selected RND-PPO checkpoint is stored in
`artifacts/frontier-level7-extended-ppo-g999-rnd005-125000-20.duckdb` with four
wins in 20 episodes. The IQL evaluation stops on its first live win at episode
13 and retains `artifacts/level7-iql-win-logical-800x600.mp4` (226 frames,
800×600 logical pixels). These are engine-process observations from the same
winning rollout, not a seed-only action replay.

Merging all capped rollouts, corrected full-horizon evaluations, and live video
searches with the prior sequential corpus produces
`data/portable-vectors-rich-v2-sequential-level7-augmented.duckdb`: 1,172
episodes and 584,934 transitions (SHA-256
`52ce74246edbe2cf30114cf114e7a5ad84f3406f1f41aaba922fcc62928c0f43`).
Level 1-7 contributes 545 episodes, 306,613 transitions, and 11 explicit wins.
The corpus is locally verified by both byte hash and a read-only DuckDB query.

Sequential 1-8 PPO evaluation adds 73 level-1-8 episodes, 39,774 transitions,
and four explicit wins. Together with the level-1-7 corpus these form
`data/portable-vectors-rich-v2-sequential-level8-augmented.duckdb`: 1,244
episodes and 624,708 transitions (SHA-256
`b3f3290ea0e73b9141e3e1acbbde6a243c2cc497dd440db3c686df08f1a36e5e`).
This corpus trains the terminal-corrected
world model: wrapper time limits end episode windows but no longer masquerade
as environment terminal states.

The selected 1-8 actor also has a live engine-process recording at
`artifacts/level8-rnd-win-logical-800x600.mp4` (51 frames, SHA-256
`73981a06511c497da3b7284e88c5490ee70e6490d878174dc660348951ad0e2b`).
Its paired 11-episode search replay is
`artifacts/frontier-level8-rnd-live-video-search-150000.duckdb` (SHA-256
`0439a4b29b88e4b885ca8ca2baec9695247836fc1d68ee4672445fbaa054a84c`).

The level-1-8 offline-control shard is
`data/frontier-level8-offline.duckdb`: 83 episodes, 46,791 transitions, and five
explicit wins (SHA-256
`00ce87cde88be855617534769001d4ebb9274ad8ebe00427bb697653ab970735`).
It supports the matched stochastic and argmax IQL evaluations without mixing
earlier-stage actions.

The four 1-9 policy evaluations, selected-policy smoke batch, and native video
search add 101 episodes and 56,981 transitions, including five wins. Merged with
the level-1-8 corpus they form
`data/portable-vectors-rich-v2-sequential-level9-augmented.duckdb`: 1,345
episodes and 681,689 transitions (SHA-256
`143220b151806e2dbbb7b63754aa1668665caccd3c9b6a399c22a06e7a429b4b`).
The selected live 1-9 win is
`artifacts/level9-rnd-win-logical-800x600.mp4` (53 frames, SHA-256
`327f76161fb4351cd8e20a5b9aa076df8a0f9028d8c04d980db17ebe8f4e602c`).
The same 101 level-1-9 episodes are available without earlier-stage data at
`data/frontier-level9-offline.duckdb` (SHA-256
`3ebd3b35224add7f819353ac643b5f3ce51b8528e9b5963179511573385ac9b0`) for
the matched IQL and return-ranked BC comparisons.

The selected 1-10 policy contributes a fresh 20-episode vector audit with 14
wins at `artifacts/frontier-level10-eval-g1-highent-165000-20.duckdb`
(SHA-256 `259e4b435e6a24941a63fc12254bb0b2bce479426c39c60130b0253f5abfc870`).
Its zero-update night transfer is stored separately at
`artifacts/zero-shot-level11-g1-highent-166000-10.duckdb`: 10 level-2-1
episodes, eight wins, and SHA-256
`3bbd8c2fbb4d36c20f0003d7ded11999401358b6c37768a7a410165ed9091e11`.
The independent native 2-1 win has 1,179 full-resolution pixel transitions in
`artifacts/level11-zero-shot-win-video.duckdb` and a 51-frame 800×600 video at
`artifacts/level11-zero-shot-win-logical-800x600.mp4` (video SHA-256
`86188812a71da9072b0462f304107be8f25a280b1850ab4f06bda706bb655f9c`).

Level 2-2 is a real adaptation boundary: the unchanged checkpoint is 0/10,
while its 20k-decision REINFORCE continuation is 9/20 on disjoint seeds. That
held-out replay is `artifacts/level12-reinforce-eval-190000-20.duckdb`
(SHA-256 `3441b98ba1999f091c87646b86cadd90731b28e568dab0410ca804adb9516e33`).
These vector shards are appropriate for structured dynamics; the new
`scripts/collect_stage_policy_pixels.py` command creates a resized authentic
pixel corpus from the frozen policy library for causal LeWM/token/flow training.

The wiki asset sync intentionally writes to git-ignored `data/raw/`. Its manifest
retains API URL, file description URL, dimensions, upstream metadata, delivered
byte hashes, and whether the CDN response matches the upstream SHA-1/size. The
assets are not redistributed by this repository.

```bash
uv run pvzwm sync-campaign-wiki --database data/campaign.duckdb --pretty
uv run pvzwm sync-assets --output data/raw/wiki_gg --download
uv run pvzwm data-status --pretty
```

`wiki_level_pages` is a local research archive rather than GPL source code.
Each row carries the page URL, revision, retrieval time, text hash, and
CC-BY-SA-4.0 marker. Images and embedded game media have separate rights and
are not included in the level-text archive.

## Recording the original game

The original game must be installed and running locally. The repository cannot
download a commercial copy or infer window coordinates. Once it is available:

1. Copy `configs/screen.example.json` to the ignored/local filename of your
   choice and calibrate capture, lawn, seed-packet, and shovel coordinates.
2. Run `uv sync --extra live` and grant macOS Screen Recording permission.
3. Use `record-live` without `--execute` to validate capture-only behavior.
4. Add `--execute` only after checking every predicted coordinate.
5. Confirm `original_game_transitions` is nonzero in `data-status`.

```bash
uv run pvzwm record-live --geometry configs/screen.local.json \
  --actions configs/actions.example.txt --database data/live_replay.duckdb

uv run pvzwm record-live --geometry configs/screen.local.json \
  --actions configs/actions.example.txt --database data/live_replay.duckdb \
  --execute

uv run pvzwm data-status --pretty
```

Live action scripts currently provide action labels but not trustworthy reward or
terminal labels. JEPA representation/dynamics training must therefore set
`--reward-weight 0 --done-weight 0` until those labels are added.

## Current machine audit

At the time this document was added, targeted searches of Spotlight,
`/Applications`, Steam, CrossOver, Whisky, and macOS containers found no PvZ
installation. There were zero original-game transitions. This is an environmental
fact, not a model limitation; the capture/control implementation is ready once a
local game window exists.
