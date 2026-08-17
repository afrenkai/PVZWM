# Project direction and engineering record

Last updated: 2026-08-09

This is the living source of truth for PVZWM's goals, priorities, architectural
decisions, progress, and maintainer preferences. Update it whenever the project
direction changes; do not rely on chat history to preserve a decision.

## North star

Build a fully playable, fidelity-tested implementation of the English Plants
vs. Zombies Game of the Year Edition 1.2.0.1073, then use the same verified
mechanics as the basis for a high-throughput C/Metal/CUDA RL environment and
benchmark for world models. The playable game is the behavioral oracle, not
the final systems target.

"Fully playable" means a human can launch the normal game from a terminal and
play the complete product with mouse and keyboard. It does not mean a
tick-limited agent loop. Automated tick limits exist only for tests, trace
recording, and future training.

The legally owned original game is the behavioral authority. PvZ-Portable is
the working, cross-platform C++ implementation and executable oracle in this
repository. A Steam or other legitimately owned copy should be used for
black-box comparison, with the exact edition, version, and asset hashes
recorded. See [GOTY_FIDELITY.md](GOTY_FIDELITY.md) for the detailed contract.

## Current priorities

In order:

1. Complete and verify a real human teacher playthrough from Adventure 1-1
   through 5-10 on Apple Silicon Metal, including every embedded set piece and
   a resumable checkpoint after every level.
2. Preserve each human segment as deterministic teacher data and prove all 50
   checkpoints reopen without mutating the captured artifacts.
3. Keep the same Metal-only human target source-compatible with x86_64 Macs,
   acknowledging that no Intel test machine is currently available.
4. Return to detailed original-GOTY parity comparison after the teacher loop
   works end to end; parity remains the north star but is not the immediate
   stopping gate.
5. Move to CUDA after the Metal teacher milestone when GPU-rich training hosts
   are available. RL training itself remains deferred.

RL infrastructure should continue to compile and pass preservation tests, but
new RL algorithms and training runs are not a current priority.

The first complete human campaign is teleoperation-style demonstration data:
one valuable trajectory under its recorded RNG seeds, not a claim of coverage
over every seed. Each segment must retain its exact initial save, native input
stream, seed, checkpoints, and terminal boundary so later dataset conversion,
behavior cloning, offline RL, and world-model evaluation share a reproducible
source artifact. Additional seeded demonstrations can be collected later using
the same format.

## Maintainer preferences

- Human play is a first-class CLI workflow: `make play` must open a normal,
  unrestricted game and run until the player quits.
- `make play` starts in a predictable high-DPI window at the largest 4:3 size
  that fits the current display while retaining an 800x600 logical canvas.
  Fullscreen is an explicit launch opt-in or an in-game choice, not a property-
  or save-dependent startup surprise. The fixed 800x600 physical window remains
  available through `PVZ_GOTY_NATIVE_RESOLUTION=0`.
- The primary development machine is an Apple Silicon M5 Mac.
- The playable macOS executable must use Metal and must not create an OpenGL
  context or link Apple's OpenGL framework. It should fail clearly if Metal is
  unavailable.
- Preserve the original 100 Hz simulation clock. Pace human presentation at
  60 Hz; an accelerated environment may run without real-time presentation.
- Treat x86_64 Macs as a best-effort compatibility target. They are a declining
  platform, so avoid architecture-specific assumptions but do not compromise
  the primary arm64 path for unsupported pre-Metal hardware.
- Development happens in Ghostty. Ordinary command output already flows
  through the user's Ghostty terminal when invoked there.
- If PVZWM gains an embedded terminal, terminal inspector, or VT parser, prefer
  `libghostty` through its C ABI. Do not link it merely to print ordinary CLI
  output: the external terminal emulator, not the process, renders stdout.
- Prefer established research-engineering terminology and simple,
  conventional interfaces.
- Profile before adding a performance dependency. Rust or Zig implementation
  language alone is not evidence that a library will make this workload
  faster; ABI crossings, ownership, copies, build complexity, and determinism
  all count.
- Keep the official bug-compatible baseline. PvZ-Portable's `DO_FIX_BUGS`
  remains `OFF` unless explicitly running a separate community-fixes
  experiment.
- Named human save slots and the explicitly enabled teacher-play checkpoint
  and 1x/2x pacing controls are the only intended user-facing behavioral
  extensions. Both controls are teacher-session-only and replayable. Ordinary
  play remains GOTY-compatible; otherwise, fix differences toward GOTY instead
  of inventing project-specific mechanics or UI.

## Architecture and language policy

The project has three distinct roles:

| Role | Implementation | Contract |
| --- | --- | --- |
| Human game and fidelity oracle | Pinned PvZ-Portable C++ | Full GOTY behavior, assets, UI, audio, saves, and human input |
| Automated oracle bridge | C++ executable with a versioned line protocol | Deterministic traces, structured state, optional frames |
| Native simulator | Allocation-free C11 with a stable C ABI | Batched mechanics suitable for Metal and later CUDA |

The full oracle is roughly 110,000 lines of first-party C/C++ when bundled
SDL-Mixer-X is excluded; gameplay alone is about 59,000 lines. A one-shot
rewrite would discard working fidelity and create a verification problem larger
than the performance problem. Keep it in C++, expose narrow C ABIs, and port
mechanics incrementally into the C simulator behind tick-level differential
tests.

CUDA `.cu` code is C++-based, so making every source file pure C is neither
necessary nor sufficient for CUDA compatibility. The stable boundary should be
C, with plain structs, explicit ownership, fixed-width types, and versioned
layouts.

Objective-C is not inherently faster than C. Use Objective-C or Objective-C++
only as a thin boundary when an Apple API requires it. If direct Metal becomes
necessary, Apple's `metal-cpp` is the preferred way for the existing C++ oracle
to call Metal without moving game logic into Objective-C. A future C simulator
can call a small C ABI implemented on top of Metal.

## Progress ledger

Completed:

- Pinned PvZ-Portable to revision
  `e2627de8446dfd573bce3046d41435911fb488aa`.
- Pulled the July 30 upstream fidelity release and integrated it with the
  Metal/bridge patch. This adds clean-room Zombatar editing and Flag Zombie
  rendering, lossless GOTY Zombatar profile records, and upstream corrections
  for projectile collisions, Spikerock timing, grave spawning, zombie audio,
  demo determinism, and related original-game behavior.
- Added `make play` and a normal human-input launch path using legally supplied
  GOTY assets.
- Made the CLI launch windowed by default even when GOTY properties or a saved
  registry value request fullscreen. Added `PVZ_GOTY_FULLSCREEN=1 make play`
  as the explicit fullscreen entry point while retaining in-game mode changes.
- Made human play high-DPI by default. The launcher asks Metal for the largest
  4:3 window that fits the current display and enables the Retina drawable while
  preserving the deterministic 800x600 logical canvas; fixed-size physical
  launch remains available with `PVZ_GOTY_NATIVE_RESOLUTION=0`.
- Added named human save slots through `make play SAVE_SLOT=<name>` and
  `make save-slots`, plus non-overwriting `make fork-save FROM=<slot>
  TO=<checkpoint>`. Each slot isolates the full ordinary PvZ save directory,
  including its native mid-level resume, while plain `make play` leaves the
  existing default save untouched. Forking permits several resumable
  checkpoints from the same human playthrough.
- Added `make play-teacher SAVE_SLOT=<name>` for opt-in human validation and
  future teacher-data collection. Each Adventure reward screen exposes one
  `SAVE CHECKPOINT` control. A successful click flushes the profile and native
  replayable input demo, inserts a level-boundary marker, and atomically copies
  the full save directory into a unique, non-overwriting teacher session.
  Ordinary `make play` does not expose or record this extension. The demo
  inspector validates marker metadata without loading the full command stream.
- Added a teacher-only in-level `SPEED: 1X`/`SPEED: 2X` toggle backed by the
  engine's existing update multiplier. It doubles wall-clock game speed while
  preserving the 100 Hz simulation timestep, persists across levels within a
  launch, and records the toggle click in the deterministic demo stream.
- Made teacher campaigns durable across process restarts. One named save slot
  now owns a stable `<slot>.campaign` artifact; every launch records a separate
  deterministic segment while all atomic level checkpoints accumulate in the
  same campaign. A non-destructive importer carries the existing legacy 1-1
  through 1-3 run forward without changing its bytes.
- Added `make verify-teacher-campaign`, the human 1-1 through 5-10 acceptance
  gate. It requires the exact ordered set of 50 checkpoint directories and
  demo markers, labels all ten embedded Adventure set pieces, checks ordinary
  next-level progression, and requires the post-Zomboss save to wrap to level
  1 with `finished_adventure=1`. It then deterministically replays every human
  segment and reopens every checkpoint through Metal while enforcing immutable
  source hashes. The code and synthetic gates are complete; the authoritative
  50-level human artifact is still being collected.
- Completed the full World 1 teacher milestone in a real Metal human
  playthrough through Adventure 1-10, including Wall-nut Bowling, the day
  conveyor finale, and levels played with the teacher-only 2x pacing control.
  All ten checkpoint manifests and demo markers are ordered and valid, and the
  final save advances to 2-1. The campaign is now valid through 3-5, bringing
  it to 25 accepted boundaries. Its teacher stream includes the intended
  coin-threshold Crazy Dave seed-slot offer and the human acceptance clicks.
  The post-2-4 save confirms one packet upgrade, seven seed slots, and the
  Suburban Almanac unlock while advancing to 2-5. The initial three complete
  saves
  agree with native-demo markers at
  updates 10,744, 25,979, and 45,043. The checkpoint-time flush contained
  52,768 replay-command bytes; normal shutdown after a long idle tail finalized
  the retained demo at update 157,704 with 63,225 bytes. The inspector now also
  decodes `users.dat` and `userN.dat`; the three native profiles contain the
  expected next Adventure levels 2, 3, and 4.
- Added deterministic native teacher-demo verification. `-play-to-end`
  and `-play-to=<update>` accelerate replay, reject unconsumed synchronized
  commands, check the requested boundary, and can write a machine-readable
  report. Presentation suppression (`-demo-blackout`) and window suppression
  (`-demo-hidden`) are independent opt-in flags. `make verify-teacher` is
  visible by default and exposes them as `BLACKOUT=1` and `HIDDEN=1`; combining
  both gives the maximum-throughput gate without entering the AeroSpace layout.
  The verifier does not replay a later idle tail, while visible accelerated
  replay uses a distinct window title. Blackout still performs the two
  bootstrap draws required by the original title-screen loader, then suppresses
  all intermediate presentation. The recorded 1-1 through 1-3 trajectory now
  verifies natively through update 45,044 in about 0.85 seconds of replay-loop
  time on the primary Apple Silicon development machine.
- Added native checkpoint-resumption verification. Each captured save is copied
  to temporary storage and reopened through the Metal engine; the selected
  profile is checked against the native parser and manifest before the
  recovered stage is initialized. The three current human checkpoints load
  Adventure levels 2, 3, and 4 and initialize 1-2, 1-3, and 1-4 respectively,
  while before-and-after hashes prove the teacher artifacts are unchanged.
- Corrected wrapped inline-format state in the Almanac/text renderer. Format
  tokens discovered by wrap lookahead no longer leak color, visibility flags,
  fonts, or line spacing into an earlier line; clipped lines still advance
  state. `make test-string-format` covers the locked Magnet-shroom weakness
  text and a wrap-boundary spacing regression with a recording font.
- Confirmed the game launches and is playable on the active Apple Silicon Mac.
- Split the macOS human renderer from the automated bridge renderer.
- Implemented the human renderer through SDL's native `metal` driver, including
  textured geometry, transforms, clipping, nearest/linear filtering, normal and
  additive blending, repeating textures, resizing, 4:3 letterboxing, and
  logical pointer mapping.
- Removed OpenGL source and framework linkage from the human macOS executable.
  The separate bridge still retains its private OpenGL framebuffer for deferred
  environment work.
- Restored GOTY's 100 Hz fixed simulation while presenting at 60 Hz. Human
  review confirmed the revised pacing looks correct.
- Restored GOTY's night-music stream resynchronization. The bundled OpenMPT
  codec now exposes its tempo-factor C ABI to SDL-Mixer-X, allowing the main
  and drum streams to slow or accelerate by the original small correction.
  Tune starts and resumes reset to native tempo. `make test-audio` validates
  slower, faster, and reset factors with an in-memory copyright-free MOD; a
  complete 2-1 run passes with exact OpenGL/Metal gameplay state. The same
  loader fix now checks the actual OpenMPT internal-buffer allocation instead
  of accidentally rechecking an earlier allocation.
- Centralized the human game-music selector and exposed its result through the
  deterministic bridge without enabling host-timed audio. The independent
  50-stage campaign contract now checks every Adventure tune, including puzzle
  and minigame set pieces, conveyor music only on 1-10 through 3-10, silence on
  Stormy Night 4-10, and the Zomboss theme on 5-10. All 150 reset/update
  checkpoints and the subsequent 50-stage legal-win run remain exact between
  the reference and Metal renderers.
- Extended that independent selector contract to all 70 secondary modes and
  210 reset/update checkpoints. It covers terrain themes, Zombiquarium's night
  theme, Zen Garden and Tree of Wisdom, all puzzle stages, the Column conveyor
  override, Dark Stormy Night silence, and Dr. Zomboss's Revenge. Every
  checkpoint remains exact between the reference and Metal renderers.
- Corrected the terminal performance statistic to report wall-clock presented
  FPS instead of dividing by draw-call CPU time.
- Added a deterministic bridge protocol and a day-lawn terminal-win action
  tape; representative Adventure stage resets cover day, night, pool, fog,
  roof, and boss environments.
- Added a hidden Metal differential bridge and `make test-renderers`. The gate
  runs the OpenGL and Metal renderers from identical seeded state at ticks 0,
  500, and 2,000 across representative day, night, pool, fog, roof, and boss
  stages. All 18 canonical gameplay snapshots match exactly. Four frames are
  byte-identical; the worst observed frame has a mean absolute channel error of
  0.034/255 across 1.81% of pixels. Same-backend control runs are byte-exact,
  and inspected residuals are cross-API blend rounding and edge coverage.
- Added `make test-campaign`, an exhaustive Adventure-state gate. It validates
  GOTY-specific background, row, seed-bank, seed-chooser, conveyor,
  board-click, grave, vase, and boss invariants at ticks 0, 500, and 2,000 for
  all 50 stages. All 150 checkpoints match exactly between the OpenGL and Metal
  bridges. It also executes ten targeted interactions with exact cross-renderer
  state: Whack-a-Zombie damage, Vasebreaker destruction, Wall-nut Bowling,
  Little Trouble, Bungee Blitz, every world-ending conveyor, and Zomboss
  placement.
- Added `make test-modes`, an exhaustive secondary-mode initialization and
  active-update gate. The private bridge can directly reset all 70
  non-Adventure native modes without changing human unlock behavior. Survival,
  Mini-games, Limbo, Zen Garden, Tree of Wisdom, Vasebreaker, and I, Zombie all
  pass their mode-specific invariants at ticks 0, 500, and 2,000; all 210
  canonical gameplay snapshots are exact between OpenGL and Metal. The bridge
  now tears down the previous board before changing modes, preventing
  cross-mode cleanup from interpreting stale objects under the next mode.
- Moved every automated GOTY smoke, trace, campaign, renderer, and mode process
  to an isolated temporary save directory. Automated verification can now run
  alongside human play without reading, overwriting, or mid-level-saving into
  the human profile.
- Added `make test-saves`, an exhaustive native mid-level save gate. All 50
  Adventure boards and all 67 secondary modes that use a board save are
  serialized at tick 500, reloaded through the actual Continue-dialog path,
  and compared exactly with an uninterrupted OpenGL oracle both immediately
  and after another 1,000 ticks. Fresh-process Metal checks additionally prove
  persisted profile selection for Adventure 4-1 and Dr. Zomboss's Revenge.
  The first gate exposed unsaved global random state; the extensible v4 format
  now stores and restores the complete Mersenne Twister state plus both engine
  seed fields. Existing v4 files remain readable because the fields are
  optional TLV additions.
- Expanded `make test-playthroughs` into a native terminal-state gate for all
  Adventure stages from 1-1 through 5-10. A deterministic verifier uses only
  legal player actions; all 50 stages reach the authentic win result across
  1,900,010 engine ticks and 190,001 decisions, with exact OpenGL/Metal
  gameplay state after every action. World 4 adds the native Vasebreaker
  break, pickup, and cursor-placement sequence plus progression-aware fog
  defense. The Vasebreaker verifier consumes recovered seed vases first,
  distributes one Peashooter and Squash per lane, and opens the closest
  remaining zombie vase first so the limited close-range defenses are not
  spent on distant threats. World 5 adds Pogo, Bungee, Catapult, and Dr.
  Zomboss responses.
  Conveyor Blover packets are retained until a Balloon appears, and one use is
  allowed to resolve before another can be consumed.
- Added `make test-mode-playthroughs` as the first terminal-state gate beyond
  Adventure. All five stages of Survival: Day, Survival: Night, Survival: Pool,
  and Survival: Fog, plus Seeing Stars, Whack-a-Zombie, Last Stand, Sunny Day, Big Time, and
  Art Challenge Sunflower, Grave Danger, and all nine finite Vasebreaker and
  I, Zombie levels reach their native win result using only advertised player
  controls. Each Survival mode executes four real seed-chooser repicks while
  retaining the lawn. The reference and Metal bridges match exactly after all
  115,419 board decisions / 1,154,190 engine ticks plus those sixteen repicks
  and Last Stand's Onslaught-button interaction.
  Grave Danger completes all 20 waves and legally triggers three lawn mowers.
  Survival: Fog uses a
  human-legal completed profile with Pool Cleaners, one purchased packet slot,
  and purchased Cattail plus Pumpkin support; its final chooser replaces a
  retained Sun-shroom packet with Cherry Bomb to clear the last Football wave. Each
  secondary mode runs in a fresh pair of native processes and private save
  directories, while all five Survival stages remain continuous in one board;
  this prevents unrelated lazy resource caches from perturbing later mode RNG.
  The Vasebreaker policy breaks every native vase, places every recovered seed
  with the cursor, and clears every revealed zombie. It models Leftpeater
  direction, Threepeater three-lane and column coverage, Spikeweed paths,
  Tall-nut interception, each lane's visible zombie-vase load, and heavy-zombie
  reveal ordering while leaving Jack-in-the-box until last. The policy fills
  the actual GOTY art outlines; the remaining 41 secondary modes are still
  outside the terminal-completion claim.
  The I, Zombie policy uses the native sun economy and zombie packets, keeps
  instant-kill sacrifices single-file, and handles Digger, Bungee, Ladder,
  Football, Garlic, Chomper, and Tall-nut interactions. Its final-level policy
  follows visible board phases rather than timing or internal-state shortcuts.
  Survival: Night now uses a completed human profile with one purchased packet
  slot and seed upgrades. It establishes two Sun-shrooms per lane, constructs
  and protects one Fume-to-Gloom core per row, performs four native repicks,
  and retains four of five lawn mowers through its fifth-stage win.
- Fixed accelerated reset after the bridge skips the visual level intro. Lawn
  mowers and purchased Roof Cleaners are now made visible at x=-21, matching
  the completed native cutscene, instead of remaining hidden at their
  construction position x=-160 where zombies could cross the loss boundary
  first. Sunny Day now legitimately consumes one mower while the reference and
  Metal states remain exact. An exploratory Survival: Roof run with ordinary
  six-packet play and purchased Roof Cleaners now reaches the fifth stage's
  final wave and triggers all five cleaners; it still loses and is therefore
  not included in the terminal-completion count.
- Moved bridge RNG initialization after old-board teardown. Effect cleanup can
  consume randomness, so seeding before `KillBoard()` made a private reset
  depend on the board it replaced. Multi-mode terminal audits additionally
  start a fresh process pair per mode, isolating lazy resource caches without
  interrupting any mode's real multi-stage continuity.
- Fixed the verifier's incomplete click synthesis: plant, shovel, and board
  clicks now emit matching mouse-down and mouse-up events. The missing release
  left `mIgnoreMouseUp` latched and made Last Stand's otherwise visible
  Onslaught button stay disabled. The verifier now advertises that native
  button as a separate UI interaction without changing the fixed 650-action
  board vocabulary.
- Replaced the ambiguous generic `bonus` campaign labels with the actual GOTY
  set pieces: Wall-nut Bowling, Whack-a-Zombie, Little Trouble, Vasebreaker,
  and Bungee Blitz.
- Added the initial allocation-free C11 simulator and Python wrapper while
  explicitly deferring its expansion until after Metal human-play work.
- Removed per-draw heap allocation from Metal geometry conversion, retained
  reusable triangle/upload scratch buffers, and moved `MemoryImage::GetBits()`
  out of the texture upload inner pixel loop.
- Set an explicit macOS 12.0 deployment target instead of inheriting macOS 26.0
  from the active SDK.
- Added `PVZ_MACOS_ARCHS`, `PVZ_MACOS_DEPLOYMENT_TARGET`, and `make setup-intel`.
- Cross-compiled every playable source file, plus the Metal differential
  bridge, to x86_64 Mach-O objects. Final linking cannot be tested on this
  arm64-only installation because its Homebrew SDL, PNG, JPEG, Ogg/Vorbis,
  mpg123, and OpenMPT libraries contain no x86_64 slices.
- Pinned a reproducible Ruff 0.16.1 correctness/import baseline in
  `pyproject.toml` (`E4`, `E7`, `E9`, `F`, and `I`, with assigned lambdas
  explicitly allowed), declared the intentional package re-exports, and
  applied the safe import-order cleanup. `uvx ruff check .` now passes over the
  complete Python repository.
- Ran Clang-Tidy's `clang-analyzer-*` checks with Apple SDK headers on the
  bridge and separately on the allocation-free C11 simulator. Both current
  paths analyze without diagnostics.
- Clang static analysis of the new save, RNG, and bridge path found and fixed
  null-allocation handling in `TodList` and an uninitialized legacy save
  header. The changed save-path sources now analyze cleanly for both arm64 and
  x86_64.
- Passed all 397 Python tests, the six-stage GOTY bridge smoke test, the
  50-level/150-checkpoint Adventure differential gate, the
  70-mode/210-checkpoint secondary-mode differential gate, and the
  117-board/238-comparison native save gate, the 50-stage terminal playthrough
  gate, the 29-mode/1,154,190-tick terminal secondary-mode gate, and the
  six-environment Metal/OpenGL renderer gate after the optimization and
  compatibility changes.
- Live-tested the default human launch in an isolated profile: it reported an
  800x600 windowed Metal renderer with acceleration and vsync, and presented
  57.3 FPS during the sampled startup.
- Human-tested the pulled clean-room Zombatar flow on the actual windowed Metal
  executable; the previously inert selector action now opens and works.
- Added atomic late-Adventure test-slot bootstrapping. It either wins the
  preceding board through legal native actions or, when explicitly requested,
  uses the native Adventure initializer; both paths flush the original profile
  serializer, decode the resulting save, reopen it through Metal, and refuse
  installation on any mismatch. The legal 5-9 path also collects the level
  award, waits 500 native reward-transition ticks, finalizes the original win
  path, persists Adventure level 50, and reopens it in a fresh Metal process.
  The isolated `before-5-10` slot validates at Adventure level 50 while
  `world1-clean` remains at level 4.
- Added a privacy-safe original-GOTY baseline workflow. It hashes the owned
  installation without retaining absolute paths or copyrighted contents,
  distinguishes assets from a runnable/versioned reference, and initializes a
  134-scenario acceptance matrix spanning 14 UI/profile flows, all 50 Adventure
  stages, and all 70 secondary modes. Evidence records retain relative paths,
  sizes, and SHA-256 hashes. The present five-file GOTY resource bundle is
  captured as assets-only content
  `2a9bfcb68d40971912803122415f45e96ae564db8711b5a409a74c193ca5d68d`
  and is not falsely counted as original-executable evidence.

Still required:

- Exercise every menu, mode, Adventure stage, save/load path, and special
  interaction against the owned original GOTY reference. The native
  self-round-trip gate does not replace this black-box comparison.
- Continue the clean-profile human World 1 playthrough from 1-4 through 1-10;
  the first three levels and their checkpoint/teacher-data path are validated.
  Use the pass to identify the first behavioral mismatch before widening the
  manual campaign run.
- Complete the human Adventure playthrough and record discrepancies without
  confusing the now-complete automated verifier frontier with native-engine
  fidelity evidence.
- Add terminal-state player-action gates for the remaining finite secondary
  content: normal and hard Survival and the remaining Mini-games. All nine
  finite Vasebreaker and all nine finite I, Zombie stages are now covered.
  Treat endless modes, Zen Garden, and Tree
  of Wisdom as milestone-based persistent-mode checks rather than inventing a
  terminal condition. The existing 70-mode gate proves initialization, 2,000
  active updates, music selection, and renderer identity—not completion.
- Add original-game visual, audio, timing, and profile/save compatibility
  fixtures that can be legally retained.
- Collect Instruments traces for startup, texture loading, steady-state
  rendering, and animation-heavy levels before undertaking renderer batching
  or direct-Metal work.
- Run and package on a real x86_64 Mac with native x86_64 dependencies.
- Promote the isolated Metal bridge to the default observation path when
  environment work resumes, retaining the OpenGL bridge only as a temporary
  differential oracle.
- Port authentic mechanics into the C core gate by gate, then implement Metal
  compute and CUDA from the same conformance vectors.

## Performance policy and next measurements

Optimize complete user-visible or rollout latency, not isolated microbenchmarks.
For the human game, collect:

- launch-to-first-frame and launch-to-main-menu time;
- CPU allocation count/bytes during resource loading and a representative
  level;
- CPU time in image conversion, texture upload, SDL command generation, audio,
  update, and draw;
- GPU frame time, command-buffer count, state changes, texture uploads, and
  drawable waits;
- frame pacing percentiles, not only average FPS;
- peak resident memory and retained CPU image backing.

Use Instruments Time Profiler, Allocations, Metal System Trace, and signposts
first on macOS. Add Tracy's C instrumentation only if cross-platform traces
would materially improve diagnosis.

The next safe renderer optimizations, contingent on profiles, are:

1. Cache redundant blend/filter/clip state changes.
2. Batch consecutive compatible SDL geometry submissions.
3. Avoid retaining CPU texture copies for images that can never require
   `RecoverBits`.
4. Move to SDL3 GPU or direct Metal only if SDL Renderer submission is a proven
   bottleneck.

The native simulator keeps stricter rules: no allocation, Python call, JSON,
rendering, or device transfer in `step`; contiguous batch buffers; deterministic
per-environment state; and end-to-end learner benchmarks.

## C-ABI dependency decisions

| Candidate | Decision | Rationale |
| --- | --- | --- |
| [libghostty](https://github.com/ghostty-org/ghostty) | Preferred for a future embedded terminal/VT feature | Cross-platform Zig/C library and C-compatible ABI. `libghostty-vt` is usable, but API signatures remain in flux. It does not accelerate writes rendered by an already-running external Ghostty terminal. |
| [SDL Renderer](https://wiki.libsdl.org/SDL2/CategoryRender) | Keep now | Already supplies the native Metal human path with low integration and fidelity risk. |
| [SDL3 GPU](https://wiki.libsdl.org/SDL3/CategoryGPU) | Evaluate after profiling | Modern C API for graphics and compute over Metal/Vulkan/D3D12, explicit command buffers, and resource uploads. Migration cost is substantial and is not automatically a speedup for this 2D game. |
| [metal-cpp](https://developer.apple.com/metal/cpp/) | Preferred direct-Metal fallback | Apple documents a direct C++ mapping with no measurable overhead. Hide it behind the renderer boundary or a C ABI; use only if SDL is proven limiting. |
| [sokol](https://github.com/floooh/sokol) | Defer | Small, portable C headers with Metal and other backends. Attractive for a new pure-C frontend, but replacing working SDL duplicates window/input/audio integration. |
| [wgpu-native](https://github.com/gfx-rs/wgpu-native) / [wgpu](https://github.com/gfx-rs/wgpu) | Do not adopt now | Capable Rust implementation with a C API and multiple backends, but adds a large runtime/build surface and another shader/toolchain layer without an identified bottleneck. |
| [mimalloc](https://github.com/microsoft/mimalloc) | Benchmark only if allocations dominate | Mature C allocator and macOS support. The native `step` path is already allocation-free; allocator replacement should follow an Allocations trace, not precede one. |
| [yyjson](https://github.com/ibireme/yyjson) | Defer to bridge work | Fast C JSON could reduce future line-protocol overhead, but JSON and the bridge are outside the human render path and RL work is deferred. A binary protocol may ultimately be better. |
| [Tracy](https://github.com/wolfpld/tracy) | Optional profiling aid | Useful frame profiler with C instrumentation. Prefer zero-dependency Apple tooling first; add Tracy when comparable macOS/Linux/CUDA timelines are needed. |
| ECS replacement such as Flecs | Do not use in the oracle | Changing entity storage/order threatens GOTY event order and determinism. The native C simulator's fixed-capacity arrays are simpler for PvZ-scale state. |

Every new dependency needs:

1. a reproducible before/after workload;
2. measured end-to-end improvement;
3. arm64 and x86_64 macOS compatibility;
4. a pinned version and license review;
5. explicit ownership, thread-safety, and determinism rules at its C ABI.

## Intel Mac compatibility

The supported source configuration is Metal-only, x86_64, macOS 12.0 or newer:

```bash
make setup-intel
```

That target writes to `vendor/PvZ-Portable/build-macos-x86_64`. On an Intel Mac,
install native x86_64 dependencies first. On Apple Silicon, a complete Intel
link requires x86_64 libraries from a separate Rosetta Homebrew installation
(normally under `/usr/local`) or a matching vcpkg `x64-osx` toolchain. The
current `/opt/homebrew` libraries are arm64-only, so this machine can validate
all source compilation but not produce or run the final Intel executable.

A universal binary may be requested with:

```bash
PVZ_MACOS_ARCHS='arm64;x86_64' \
PVZ_PORTABLE_BUILD_DIR=vendor/PvZ-Portable/build-macos-universal \
scripts/setup_portable_backend.sh
```

Every linked dependency must also be universal. Intel Macs without Metal
support are intentionally unsupported; there is no human OpenGL fallback.

## Quality gates

Before a handoff:

- build `pvz-portable` and `pvzwm-bridge` on arm64;
- run `make test-string-format`;
- verify the human binary links Metal and QuartzCore but not OpenGL;
- run `make test-renderers` on macOS;
- run `make test-campaign` on macOS;
- run `make test-modes` on macOS;
- run `uvx ruff check .` using the repository-pinned rule set;
- run Clang-Tidy/analyzer checks on changed C/C++ plus the native C core;
- treat Clang-Tidy as the current C/C++ lint gate; do not apply default
  clang-format without first adopting a project style file, because the
  vendored codebase intentionally follows a different existing layout;
- compile the playable C/C++ source set for x86_64;
- run unit tests and the GOTY bridge smoke test when WindowServer access is
  available;
- regenerate and reverse-check the pinned vendor patch;
- run `git diff --check` on source changes. Exclude the generated patch file
  itself because its payload necessarily preserves whitespace from upstream
  context, then run the check inside the vendor tree.

## Definition of done

The human game is done only when a clean profile can play the complete GOTY
product, including secondary modes and persistent systems, with documented
comparison evidence against the owned original. The environment is done only
when its versioned C state machine matches the oracle at every tick across the
representative trace suite and achieves a measured end-to-end throughput win.
Neither a successful launch nor a high tick rate alone satisfies those goals.
