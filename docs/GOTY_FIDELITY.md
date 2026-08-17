# GOTY fidelity contract

The target is the English Plants vs. Zombies Game of the Year Edition
1.2.0.1073. The playable implementation uses the legally supplied GOTY
`main.pak` and `properties/` data with the LGPL-3.0-or-later PvZ-Portable
engine. The current asset identity is:

```text
main.pak SHA-256
0cbdba7ecd6106cca3190976061922ef61011c1896205b10cbc204aaa9d88cba
```

`make play` is the supported human entry point. It invokes
`scripts/play_goty.sh`; on macOS the script selects
`build-macos/pvz-portable` (arm64 by default); on Linux it selects `build/`.
Profiles, settings, Adventure progress, purchases, and Zen Garden state use the
platform's normal PvZ-Portable application-data directory unless
`PVZ_GOTY_SAVE_DIR` is explicitly set.

This is an unrestricted real-time game process. It opens the ordinary SDL
window and accepts human mouse and keyboard input until the player quits. It
does not use the RL bridge, an agent, a fixed action tape, or a tick limit:

```bash
make play
```

The CLI entry point deliberately starts windowed at the game's 800×600 logical
resolution, overriding both GOTY's fullscreen property and a saved startup
mode. This makes terminal-driven testing predictable without preventing a
player from changing the display mode in the in-game options. Fullscreen is
available as an explicit launch choice:

```bash
PVZ_GOTY_FULLSCREEN=1 make play
```

The finite tick counts elsewhere in this repository apply only to automated
regression traces and training rollouts.

The first live teacher run has now validated the opt-in reward-screen control
through Adventure 1-3. Its checkpoints after 1-1, 1-2, and 1-3 contain complete
save directories and match markers in the native input demo at updates 10,744,
25,979, and 45,043. This is direct human-path evidence; automated tests did not
synthesize those clicks. The native profile files in those snapshots decode to
next Adventure levels 2, 3, and 4 respectively, matching every manifest.
The human process remained open after the last checkpoint, so normal shutdown
later finalized the demo at update 157,704 with 63,225 command bytes. That
post-checkpoint idle tail is retained but is not checkpoint evidence.

`make verify-teacher RUN='<teacher-run>' EXPECTED=<count>` provides the native
replay gate. It uses an isolated save directory and the Metal executable,
fast-forwards without intermediate presentation in a hidden window through one
update after the last checkpoint marker, and requires a machine-readable
target report. Post-checkpoint idle time is not part of the checkpoint claim.
Visible accelerated replay, blackout, and hidden-window behavior are separate
flags so a verifier never unexpectedly takes over an AeroSpace tile.
Blackout retains the original engine's minimal title-screen bootstrap draws so
resource loading starts normally, then suppresses subsequent Metal
presentation. The current three-checkpoint human run passes this gate exactly
at update 45,044.

`make verify-teacher-resume RUN='<teacher-run>' EXPECTED=<count>` provides a
separate native checkpoint-resumption gate. It copies every checkpoint to
temporary storage, starts the Metal engine against that copy, compares the
engine-selected profile with the native save parser and checkpoint manifest,
and initializes the recovered next stage. The current human checkpoints load
Adventure levels 2, 3, and 4 and initialize stages 1-2, 1-3, and 1-4
respectively. Source checkpoint hashes are required to remain unchanged.

## Human save slots

Named save slots are the one intentional user-facing extension to GOTY
behavior. Each slot is a separate complete save directory, so it independently
retains profiles, settings, Adventure progress, purchases, Zen Garden state,
and the game's ordinary mid-level resume data:

```bash
make play SAVE_SLOT=world1-a
make play SAVE_SLOT=world1-b
make save-slots
make fork-save FROM=world1-a TO=before-1-5
```

Plain `make play` retains the platform's native default save directory. Named
slots do not copy, move, or overwrite that save. Slot names contain 1-64
letters, digits, periods, underscores, or hyphens and must begin with a letter
or digit. `fork-save` makes a complete point-in-time copy of an existing slot
under a new name and refuses to overwrite any existing destination, allowing
multiple resumable checkpoints from one human playthrough. The game must be
closed before forking so its save files are current and internally consistent.
The default slot root is `~/Library/Application Support/PVZWM/save-slots` on macOS and
`${XDG_DATA_HOME:-~/.local/share}/pvzwm/save-slots` on Linux. It can be changed
with `PVZ_GOTY_SAVE_ROOT`; `PVZ_GOTY_SAVE_DIR` remains available as the
lower-level explicit-directory override.

## Original-game reference

A legitimately owned Steam installation of Plants vs. Zombies GOTY Edition is
the appropriate black-box reference. Keep the original executable and data
outside source control. Compare a clean profile in the original and this build
with the same visible choices, recording the level, profile state, inputs,
screenshots, audio, and outcome for every discrepancy.

Steam currently labels its legacy macOS build incompatible with macOS 10.15 and
newer. On a current Apple Silicon Mac, the original executable therefore
normally needs a Windows machine, VM, or compatibility layer for live
side-by-side comparison. That restriction does not apply to this repository's
native arm64 executable.

Record the exact version and hashes from the reference installation before
comparing. Do not silently mix results from a differently patched Steam,
EA/Origin, retail, or language build; if the owned copy differs from
1.2.0.1073, treat it as a separate baseline until its behavioral differences
are characterized.

Create the local, ignored reference manifest and acceptance matrix with:

```bash
make reference-baseline \
  REFERENCE_ROOT='/path/to/Plants vs. Zombies' \
  REFERENCE_VERSION=1.2.0.1073 \
  REFERENCE_DISTRIBUTION=steam-windows
```

The manifest hashes every regular installation file but retains neither file
contents nor the absolute installation path. It refuses symbolic links and
marks a baseline ready only when it contains `main.pak`, a recognized PopCap
game executable, and a declared version. The acceptance plan contains 134
individually reportable scenarios: 14 title/profile/menu/cross-cutting flows,
all 50 Adventure stages, and all 70 native secondary modes. Both files live
under ignored `data/reference/local/`.

Keep screenshots, audio clips, and other copyrighted evidence beside that
local plan, then record a result with content hashes:

```bash
uv run scripts/record_goty_acceptance.py \
  data/reference/local/goty-acceptance.json \
  adventure/1-1 --result match \
  --reference-artifact data/reference/local/reference-1-1.png \
  --candidate-artifact data/reference/local/candidate-1-1.png
```

Evidence must remain below the plan directory, and the JSON stores only its
relative path, byte size, and SHA-256. `pvzwm data-status` reports whether a
runnable, versioned reference manifest is available. The currently supplied
asset bundle has been recorded separately as assets-only content
`2a9bfcb68d40971912803122415f45e96ae564db8711b5a409a74c193ca5d68d`;
it is correctly not accepted as executable black-box evidence.

The default build sets PvZ-Portable's `DO_FIX_BUGS` option to `OFF`. Its
community fixes intentionally alter official behavior and therefore cannot be
part of the fidelity baseline. A user may test them separately with
`PVZ_PORTABLE_FIX_BUGS=ON make setup`, but results from that build must not be
treated as original-game conformance evidence.

## What identical means

Completion requires evidence for all of the following, not merely a compatible
RL action space:

- The title flow, profiles, menus, Almanac, shop, achievements, credits, help,
  options, Zombatar, pause/restart, and save/load behavior are playable.
- Adventure levels 1-1 through 5-10 have the GOTY terrain, seed progression,
  wave schedules, plants, zombies, projectiles, economy, Crazy Dave sequences,
  rewards, and terminal transitions.
- Mini-games, Puzzle, Survival, Zen Garden, Tree of Wisdom, and the Limbo modes
  exposed by GOTY are functional.
- Mouse/keyboard input, 800x600 logical rendering, animation timing, audio, and
  music behave equivalently.
- GOTY `user*.dat` profile compatibility and portable mid-level saves preserve
  game state.
- Seeded gameplay is deterministic within a build. Cross-backend conformance
  compares gameplay state at every engine tick.

PvZ-Portable is currently the executable cross-platform implementation
available in this workspace. It is a candidate implementation, not the final
authority. The legally supplied PopCap 1.2.0.1073 game is the behavioral
reference for a strict fidelity claim. Automated original-game comparison
additionally requires locally recorded reference inputs and observations; those
copyrighted artifacts are not stored in this repository.

Except for the named human save-slot wrapper above, custom game behavior is out
of scope. Fidelity discrepancies should be fixed toward GOTY rather than
retained as project-specific mechanics or UI.

The pinned engine includes the clean-room Zombatar editor, its terms screen,
GOTY-compatible 72-byte profile records, and custom Flag Zombie rendering.
This replaced the earlier no-op main-menu button and prevents existing GOTY
Zombatar records from being discarded when a profile is loaded and saved. The
selector-to-Zombatar flow has been human-tested successfully on the windowed
Metal executable.

Manual Zombatar acceptance remains part of the human GOTY comparison:

1. Launch an isolated slot with `make play SAVE_SLOT=zombatar-check`.
2. Create or select a profile, open Zombatar, accept its terms screen, create a
   head, and finish back to the selector.
3. Quit and relaunch the same slot; confirm the head remains editable.
4. Reach a flag wave and confirm the customized head is used by the Flag
   Zombie.
5. Compare the editor layout, category choices, colors, persistence, and
   in-game composition with the owned 1.2.0.1073 executable.

## Differential protocol

Protocol v4 of the private pipe bridge records enough state to diagnose the
first divergence instead of comparing only score or pixels:

- startup renderer and selected profile name, ID, Adventure level, coins, and
  completion state;
- board counters, wave state, sun, legal actions, terminal state, and RNG state;
- seed packet type/availability/recharge;
- plant position, health, state, action counters, sleep, and squash state;
- zombie position, velocity, health layers, phase/counters, status effects, and
  source wave;
- projectile position, velocity, motion, age, and damage flags;
- coin motion/collection state;
- grid items including graves, craters, ladders, pots, portals, and brains;
- mower state and optional rendered frames.

The accelerated bridge suppresses music and foley. PvZ uses a shared random
stream for foley variation and gameplay; allowing real-time audio instances to
expire while the engine advances faster than real time makes later gameplay
randomness depend on host timing. The human `make play` executable is unaffected
and retains full audio. Original-game audio behavior remains part of the human
fidelity contract.

The human tracker-music backend supports the persistent tempo multiplier that
GOTY uses to keep its separately rendered night main and drum streams aligned.
The resynchronizer slows a stream that is ahead, speeds one that is behind, and
returns to the module's native tempo when aligned or when a tune starts or
resumes. `make test-audio` constructs a copyright-free ProTracker module in
memory and requires the OpenMPT C ABI to apply, report, and reset both slower
and faster tempo factors. A complete 2-1 terminal run additionally exercises
the dual-stream night path with exact OpenGL/Metal gameplay state. Perceptual
comparison with the original GOTY mix remains a manual acceptance item.

The engine's game-music selector is now a single query used by both
`StartGameMusic` in the human path and the private verifier. Bridge observations
report that selected tune without starting real-time audio, preserving
deterministic accelerated runs. The campaign gate independently asserts the
full 50-stage Adventure matrix: day, night, pool, fog, and roof themes;
minigame music for the four applicable set pieces; puzzle music for 4-5;
conveyor music for 1-10, 2-10, and 3-10; deliberate silence for Stormy Night
4-10; and the final-boss theme for 5-10.

The day-lawn tape currently reaches a legal terminal win in 6,743 recorded
updates, including three zombies, four waves, sun collection, two planted
peashooters, and projectile combat. A second process reproduces every
canonical snapshot exactly.

`make test-campaign` expands reset and active-update coverage to all 50
Adventure stages:

```bash
make test-campaign
```

It validates stage identity, terrain/background, row count, day/night, pool,
fog and roof flags, seed-bank capacity, seed chooser versus conveyor behavior,
legal-action shape, graves, vases, and initial Zomboss presence. The current
run compares ticks 0, 500, and 2,000 for every stage: all 150 canonical
snapshots are exact between the OpenGL and Metal bridges. The same gate then
performs ten targeted interactions: it damages a surfaced zombie in 2-5,
breaks a vase in 4-5, and places a legal conveyor seed in 1-5, 1-10, 2-10,
3-5, 3-10, 4-10, 5-5, and 5-10. Every resulting state is exact across the two
renderers.

This proves deterministic cross-renderer behavior and the enumerated
Adventure invariants. It does not prove that all 50 stages can yet be won by a
complete human/input trace, nor does it replace black-box comparison with the
owned PopCap executable.

`make test-playthroughs` adds full terminal-state evidence for all five
Adventure worlds:

```bash
make test-playthroughs
```

The verifier uses only the bridge-visible board state and ordinary legal
player actions: seed selection, planting, collection, and the native
board-click path. It does not edit entities, health, sun, waves, timers, or
results. Levels 1-1 through 5-10 all reach `BOARDRESULT_WON`, including
Wall-nut Bowling, Whack-a-Zombie, the pool set pieces, Vasebreaker, Bungee
Blitz, the world-ending conveyors, and Dr. Zomboss. The 4-5 path performs the
native break, collect, and usable-seed cursor-placement sequence; the fog
policy legally reserves Blover, Cactus, Hypno-shroom, and Tangle Kelp
responses where required. The roof paths cover Pogo, Bungee, Catapult, and
Zomboss-specific counters. Across 1,900,010 engine ticks and 190,001
decisions, every intermediate canonical state is exact between the OpenGL and
Metal bridges.

This is stronger evidence than initialization or short-horizon survival, but
it is not a human playthrough and is not original-PopCap conformance evidence.

`make test-modes` provides the corresponding initialization and active-update
gate for all 70 non-Adventure native modes:

```bash
make test-modes
```

It covers all Survival variants; the 20 Mini-games; Limbo modes; Zen Garden;
Tree of Wisdom; all ten Vasebreaker stages; and all ten I, Zombie stages.
Each mode is checked at ticks 0, 500, and 2,000 for its identity, background,
row count, seed-bank capacity, scene, legal-action structure, entities, and
mode-specific setup such as scary pots, brains, aquarium zombies, and Zomboss.
The independent contract also checks GOTY's selected music: the five terrain
themes, minigame and conveyor overrides, the Zombiquarium night theme, Zen
Garden and Tree of Wisdom music, puzzle music for all Vasebreaker and I,
Zombie stages, silence in Dark Stormy Night, and the final-boss theme in Dr.
Zomboss's Revenge.
All 210 canonical state snapshots are currently exact between the OpenGL and
Metal bridges. The gate preserves GOTY-specific distinctions including
Zombiquarium's five logical rows and the double update rate in Zombie Nimble
Zombie Quick.

The mode reset is private verifier functionality and does not bypass unlocks
in the human game. Every automated GOTY verifier now runs in its own temporary
save directory, preventing a simultaneous human session from seeing or
overwriting verifier progress. This gate proves deterministic initialization
and the first 2,000 updates, not mode completion or equivalence to the owned
PopCap executable.

`make test-mode-playthroughs` adds terminal evidence for twenty-nine finite
secondary modes. A deterministic verifier uses only controls advertised by
the native board and reaches authentic wins across all five stages of
Survival: Day, Survival: Night, Survival: Pool, and Survival: Fog, plus Seeing Stars,
Whack-a-Zombie, Last Stand, Sunny Day, Big Time, Art Challenge Sunflower, and
Grave Danger, plus all nine finite Vasebreaker and I, Zombie levels.
Each Survival mode performs four real seed-chooser repicks while preserving
its plants, sun, and lawn state. Survival: Pool uses an ordinary purchased
Pool Cleaner profile; two cleaners trigger during the terminal trace.
Survival: Fog uses a completed human profile with Pool Cleaners, one purchased
packet slot, and the purchased Cattail upgrade. The verifier banks sun for two
Cattails, protects them with Pumpkins, and uses the final chooser to add Cherry
Bomb alongside Jalapeno; five cleaners trigger and the native fifth-stage win
is reached after 126,460 ticks. The two art
challenges in that set are completed by filling their actual GOTY outlines,
not by forcing the board result. Last Stand performs 41 legal placements,
then activates the real `START ONSLAUGHT` button through its ordinary
hover/down/up input path. Grave Danger completes its 20-wave night board after
78 legal placements and three real mower triggers. The Vasebreaker verifier
legally opens every vase and places every recovered seed through the native
cursor. It accounts for Leftpeater direction, Threepeater three-lane and
column coverage, Spikeweed paths, Tall-nut interception, per-lane vase threat,
and heavy-zombie reveal order; Jack-in-the-box remains last.
The I, Zombie verifier spends only harvested sun and deploys through the native
zombie seed packets. It handles lane economy, single-file instant-kill
sacrifices, Garlic and Tall-nut traversal, Chomper clusters, rear Digger
attacks, and the final puzzle's Bungee/Ladder/Football/Digger solution phases.
Survival: Night uses the same human-legal completed-profile assumptions: one
purchased packet slot and Crazy Dave seed upgrades. Two Sun-shrooms per lane
fund protected Fume-to-Gloom cores, and four real repicks retain that lawn
through the native fifth-stage win.
Across 1,154,190 engine ticks, 115,419 board decisions, and seventeen UI
interactions, the reference and Metal bridges remain exactly equal. This
terminal gate does not yet cover the other 41 modes.
Each mode receives a fresh reference/Metal process pair and two private save
directories; all five stages and four chooser repicks of a Survival mode remain
continuous inside that process. This keeps independent modes reproducible even
when their lazy resource caches consume shared random state differently.

The accelerated reset also settles every lawn mower at the same visible x=-21
position produced by the completed native intro cutscene. Previously the
bridge created the mower objects but left them hidden at x=-160, allowing a
zombie to reach the board edge without triggering a mower. Sunny Day now uses
one mower in the terminal gate. Survival: Roof has separate exploratory
evidence through the fifth stage's final wave with five triggered purchased
Roof Cleaners, but it still loses and is not claimed as a completed mode.

`make test-saves` exercises the native mid-level save lifecycle:

```bash
make test-saves
```

It covers all 50 Adventure boards and the 67 secondary modes for which
`Board::NeedSaveGame` creates a mid-level board file. Ice Level, Zen Garden,
and Tree of Wisdom deliberately use their persistent profile state instead and
are excluded from this board-file gate. At tick 500, the Metal bridge writes
the actual CRC-protected v4 save, validates its header, size, and payload CRC,
reloads it through `TryLoadGame` and the native Continue dialog, and must match
an uninterrupted OpenGL process exactly. Another 1,000 ticks must remain
exact. The current 117 boards pass all 234 immediate/continued comparisons.

Two additional checks terminate the saving Metal process, start a fresh one
against the same temporary directory, require the same profile to be selected,
and repeat the reload and continuation comparisons for Adventure 4-1 and Dr.
Zomboss's Revenge. Those four comparisons bring the gate total to 238.

The initial fresh-process Zomboss check found that v4 saves preserved board
entities but not the process-global random generator. A resumed board therefore
diverged from its uninterrupted trajectory. The format now carries the full
Mersenne Twister state, the app-level seed, and the framework seed as optional
TLV fields; old v4 files that predate those fields remain readable. This proves
native self-round-trip and deterministic continuation, not compatibility with
an owned PopCap mid-level file. Original 1.2.0.1073 black-box save comparison
remains a separate acceptance requirement.

The wrapped inline-text formatter has a dedicated engine regression:

```bash
make test-string-format
```

It checks the GOTY Almanac sequence
`{KEYWORD}Weakness:{STAT} fume-shroom{METAL} and
magnet-shroom{KEYWORD}` across wrap widths while Magnet-shroom is locked. It
also checks that a format token found by lookahead after a wrap boundary cannot
retroactively change the preceding line's spacing. The formatter now maintains
separate scan-ahead and line-start state, including when clipped lines are not
drawn.

## Apple graphics

The primary playable macOS binary is native arm64 and Metal-only. The same
renderer source cross-compiles for x86_64, with a macOS 12.0 deployment target;
a final Intel link and runtime test require native x86_64 dependencies and real
Intel hardware. Its C++ render implementation uses SDL geometry and texture
APIs while explicitly selecting SDL's native `metal` driver. It creates no
OpenGL context, does not compile the OpenGL renderer into the playable target,
and does not link Apple's OpenGL framework. It deliberately fails startup
rather than silently selecting a different renderer.

The Metal implementation covers the game's textured triangles, transformed
sprites, clipping, nearest/linear sampling, normal and additive blending,
repeating pool-effect textures, resizable 4:3 letterboxing, and logical pointer
mapping. It retains the original 100 Hz fixed gameplay clock while presenting
human-visible frames at 60 Hz. A successful launch reports:

```text
Window mode: windowed (800x600 logical)
Renderer: metal (Metal, accelerated=yes, vsync=yes, max_texture=16384x16384)
```

`pvzwm-bridge` is a separate executable and currently retains the existing
private OpenGL framebuffer for environment observations. A hidden
`pvzwm-metal-bridge` renders the same deterministic bridge state through Metal
for differential testing. Neither bridge implementation is linked into or used
by `make play`.

Run the renderer conformance gate with:

```bash
make test-renderers
```

The gate compares ticks 0, 500, and 2,000 in representative day, night, pool,
fog, roof, and boss stages. Canonical gameplay state must be byte-for-byte
identical. The current 18 snapshots all satisfy that requirement. Four rendered
frames are byte-identical; the maximum observed mean absolute channel error is
0.034/255 and the maximum changed-pixel fraction is 1.81%. Same-renderer
control runs are byte-exact. Inspection localized the remaining differences to
cross-API blend rounding and triangle edge coverage, so the automated limits
are conservatively set to 0.05/255 mean absolute error and 2.5% changed pixels.
These tolerances are a regression envelope, not evidence that the complete
product has finished visual certification against the owned GOTY executable.

The next conformance artifact is a schema-versioned JSONL action tape:

```json
{"cmd":"reset","level":1,"seed":7,"seed_types":[0],"frame_width":0,"frame_height":0}
{"cmd":"step","action":0,"ticks":1,"frame_width":0,"frame_height":0}
```

`reset` accepts either an Adventure `level` or a secondary `game_mode`, but
never both. The latter is a private conformance control used to initialize all
70 Survival, Mini-Game, Puzzle, Limbo, and persistent modes directly; it does
not change human unlock or menu behavior.

Each output snapshot is canonicalized by entity type and stable spatial fields,
then hashed. A native C/Metal backend is accepted for a mechanic only after it
matches the GOTY-backed oracle trace through terminal state across the
representative seed set.

Record the first day-lawn conformance tape on macOS with:

```bash
uv run scripts/record_goty_trace.py \
  --binary vendor/PvZ-Portable/build-macos/pvzwm-bridge \
  --resources data/pvz-goty \
  --tape tests/fixtures/goty_day_1_1_tape.jsonl \
  --output /tmp/goty-day-1-1-reference.jsonl --no-xvfb
```

Repeat the command with
`--reference /tmp/goty-day-1-1-reference.jsonl` to require exact replay.
