# Human GOTY playtest protocol

The campaign acceptance gate is a clean-profile 1-1 through 5-10 playthrough
on the native Metal executable. It includes all ten Adventure set pieces:
Wall-nut Bowling, every world-ending conveyor, Whack a Zombie, Big Trouble
Little Zombie, Vasebreaker, Stormy Night, Bungee Blitz, and Dr. Zomboss.
Teacher mode makes every completed level both a resumable save boundary and a
replayable human-input trajectory. This is teleoperation-style demonstration
data for the eventual RL/world-model benchmark; a single campaign intentionally
represents the RNG seeds it encountered rather than exhaustive seed coverage.

## Run

From Ghostty in the repository root:

```bash
make play-teacher SAVE_SLOT=world1-clean
```

The launcher prints a stable teacher-campaign directory and a unique demo
segment. The game remains an ordinary, unrestricted, windowed human session.
On macOS it uses a high-DPI Metal window at the largest 4:3 size that fits the
current display while preserving the game's deterministic 800x600 logical
coordinate system. Set `PVZ_GOTY_NATIVE_RESOLUTION=0` to request the legacy
800x600 physical window.
You may quit normally at any checkpoint and run the same command later. The
named save resumes the next level, while the new demo segment and subsequent
checkpoints are added to the same campaign artifact.

During an active level, the teacher-only `SPEED: 1X` button below `MENU`
toggles between 1x and 2x wall-clock speed. It leaves the 100 Hz simulation
timestep unchanged, persists across levels until the process exits, and is
captured as ordinary replayable input in the demonstration stream. Ordinary
`make play` sessions do not expose the control. The selected speed remains in
effect across transitions, but the button is hidden during seed selection and
level-intro UI so it does not cover the native controls.

For every level from 1-1 through 5-10:

1. Play normally with mouse and keyboard.
2. On the level reward screen, dismiss the Achievements overlay first if one
   appears.
3. Click `SAVE CHECKPOINT` exactly once.
4. Wait for the label to change to `CHECKPOINT SAVED`.
5. Continue to the next level, or quit and resume later with the same command.

After defeating Dr. Zomboss, save the `after-5-10` checkpoint before proceeding
to the credits. Then run the complete gate using the campaign directory printed
at launch:

```bash
make verify-teacher-campaign \
  RUN="$HOME/Library/Application Support/PVZWM/teacher-runs/world1-clean.campaign" \
  BLACKOUT=1 HIDDEN=1
```

A complete pass requires exactly 50 ordered checkpoints and demo markers. It
rejects a missing, duplicate, mislabeled, or out-of-order stage; verifies that
1-1 through 5-9 save the next numeric level; and requires the 5-10 save to wrap
to level 1 with `finished_adventure=1`. It then replays every recorded segment
through the native game and reopens all 50 checkpoints through Metal. Source
hashes before and after every resume must match.

The automated gate creates an isolated temporary save, initializes the native
Metal renderer, and fast-forwards the input demo through one update after the
last checkpoint marker. It succeeds only when every demo-synchronized operation
through that boundary is consumed. This deliberately excludes any long idle
tail recorded between the final checkpoint and process exit. Runs whose last
checkpoint is also the demo endpoint are replayed to the exact header update.
The replay is visible by default under the distinct
`PvZ Portable - Demo Replay` window title so AeroSpace can match it separately.
For an incomplete run or a focused segment, the lower-level gates remain
available:

```bash
make verify-teacher \
  RUN='/path/to/legacy-single-run-or-segment' \
  EXPECTED=3 BLACKOUT=1 HIDDEN=1
make verify-teacher-resume \
  RUN='/path/to/campaign-or-legacy-run' EXPECTED=3
```

`BLACKOUT=1` suppresses intermediate Metal presentation and `HIDDEN=1` keeps
the window out of the tiling layout; either can be enabled independently.
Direct native invocations expose the same controls as `-demo-blackout` and
`-demo-hidden`. The loader's required bootstrap frames are still rendered
before blackout takes effect.

The resume gate separately copies each captured save to temporary storage and
opens it with the native Metal engine. It requires the engine-selected profile
name, ID, and Adventure level to agree with the decoded save and manifest, then
initializes the recovered next stage through the bridge. Source hashes before
and after every acceptance run must match, so validation cannot alter the
teacher artifact.

## Continuing the existing 1-1 through 1-3 run

Legacy teacher runs used one directory per process. Importing one copies it
into the durable campaign format without altering the source artifact:

```bash
make import-teacher-run \
  RUN="$HOME/Library/Application Support/PVZWM/teacher-runs/world1-clean.pMBNYe" \
  SAVE_SLOT=world1-clean
make play-teacher SAVE_SLOT=world1-clean
```

The existing named save is already at 1-4, so the first new segment continues
there. Subsequent launches use the same `world1-clean.campaign` automatically.

## What is captured

- `segments/segment-N/teacher.dmo`: independently replayable native mouse,
  keyboard, focus, timing, and deterministic demo streams for each launch.
- `segments/segment-N/start-save/`: the immutable save state at segment reset;
  together with the demo header's RNG seed, this makes resumed demonstrations
  independently reproducible.
- `checkpoints/after-W-N/`: an atomic copy of the complete named save directory
  after each win, including the native profile and registry state.
- `pvzwm-checkpoint.json`: the completed and next level, demo marker, demo
  update, and engine version for each snapshot.
- `checkpoints.jsonl`: a compact campaign-level checkpoint index.

The checkpoint click is outside the gameplay policy. Its demo marker is the
trajectory boundary; a future converter can exclude that UI click and replay
the preceding native input through the bridge to derive observation/action
training records.

## Reporting a mismatch

Stop at the first material difference from the owned English GOTY reference and
record:

- level and approximate point in the level;
- expected GOTY behavior and observed behavior;
- whether it changes mechanics, timing, input, audio, visuals, or saves;
- the printed teacher-campaign directory;
- a screenshot or short capture when the difference is visual or temporal.

Do not delete or edit the teacher campaign. Its last successful checkpoint lets the
same boundary be copied into a new named slot for focused reproduction:

```bash
make fork-checkpoint \
  CHECKPOINT="$RUN/checkpoints/after-1-4" \
  TO=repro-starting-1-5
make play SAVE_SLOT=repro-starting-1-5
```

Forking never overwrites an existing slot and leaves the captured checkpoint
unchanged.
