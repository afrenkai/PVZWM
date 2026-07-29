# Research map

This project is an experimental comparison, not a claim that one architecture is
universally best. All primary experiments train from scratch on PvZ data. Public
weights may be used only in a separately labelled transfer baseline.

## Why PvZ is an interesting world-model task

The original game combines a large masked discrete action space (plant, row,
column), delayed effects, stochastic waves, resource constraints, partially
observed cooldowns, and long horizons. It also exposes three useful observation
levels:

1. symbolic simulator state for debugging dynamics and planning;
2. object/grid state extracted from game frames;
3. raw pixels plus action and reward logs.

Every model is evaluated at matched data budgets on one-step prediction,
open-loop rollout, calibration, planning return, sample efficiency, throughput,
and out-of-distribution waves/layouts.

## Model tracks

| Family | Local implementation target | Primary reference |
|---|---|---|
| Deterministic / ensemble dynamics | MLP residual and probabilistic ensemble over structured state | PETS-style uncertainty baseline |
| Persistence-residual pixel dynamics | action-conditioned convolutional delta initialized to copy the current frame exactly | persistence is the explicit held-out baseline; changed-pixel loss follows the sparse-motion lesson from video prediction rather than a plant-specific target |
| Recurrent state-space | discrete or Gaussian RSSM with actor-critic imagination | [DreamerV3 paper and code](https://github.com/danijar/dreamerv3) |
| Action-conditioned JEPA | one-step baseline plus a separate causal history-conditioned LeWM | [LeWorldModel paper](https://arxiv.org/abs/2603.19312), [official code](https://github.com/lucas-maes/le-wm) |
| Gaussian JEPA regularization | VISReg is the default; SIGReg remains a matched ablation | [LeJEPA](https://github.com/galilai-group/lejepa), [VISReg](https://arxiv.org/abs/2606.02572) |
| Token autoregression | VQ or finite-scalar tokenizer plus causal dynamics Transformer | [IRIS official code](https://github.com/eloialonso/iris), [FSQ paper/code](https://github.com/google-research/google-research/tree/master/fsq) |
| Pixel diffusion | action-conditioned denoising world model | [DIAMOND official code](https://github.com/eloialonso/diamond) |
| Representation-autoencoder diffusion | semantic visual encoder, learned decoder, latent diffusion forcing | [MIRA paper](https://arxiv.org/abs/2607.05352), [official code](https://github.com/mira-wm/mira) |
| Object-centric JEPA | plant/zombie slots and relational prediction | [C-JEPA official code](https://github.com/galilai-group/cjepa) |
| Task-oriented latent model | joint consistency, reward, value, and policy objectives | [TD-MPC2 official code](https://github.com/nicklashansen/tdmpc2) |
| Policy/value dynamics | MuZero representation, dynamics, prediction, and MCTS | [MuZero paper](https://arxiv.org/abs/1911.08265) |

The newer [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel)
interface informs the separation between collection, training, planning, and
evaluation. We keep the local implementation small and PyTorch-native so it can
run on Apple MPS.

For long-horizon policy exploration, the grouped-RL track also includes
[PSN-GRPO](https://arxiv.org/abs/2602.02555): one fixed Gaussian parameter
perturbation generates each complete rollout, while the clean policy is updated
with a truncated noisy-to-clean importance ratio. This tests coherent
trajectory-level exploration without encoding PvZ plant costs, action families,
or stage tactics.

## VISReg integration

VISReg is a representation regularizer, not a world model. Given embeddings
`z`, its regularization has three parts:

- center: penalize the squared feature mean;
- scale: match per-feature standard deviation to one;
- shape: normalize scale with a stopped gradient, project onto random unit
  directions, and match sorted projections to standard-normal quantiles using
  sliced 2-Wasserstein distance.

The JEPA objective combines action-conditioned next-embedding prediction with
this regularizer. VISReg, the original Epps–Pulley SIGReg statistic, VICReg,
and no anti-collapse regularization are executable under one matched protocol.
A stop-gradient/EMA target encoder remains a pending architectural comparison.

The released VISReg code treats embeddings as `[views, batch, features]`,
computes center/scale/shape independently across the batch for every view, and
places a BatchNorm MLP projection head after the visual backbone. LeWM maps
the `T+1` unique trajectory states to those views. The released LeWM design also
uses separate BatchNorm MLPs on encoded and predicted latents and an additive
`prediction + λ * regularizer` objective; both that form and the VISReg image-
pretraining convex mixture are executable. Warmup, cosine decay, gradient
clipping, and weight decay are exposed as training controls. The raw-encoder/
flattened-time implementation is retained only in failed ablation checkpoints
because it collapsed even after matching λ, epochs, batch size, and learning
rate.

The executable `lewm` model is distinct from the earlier one-step `jepa`
baseline. It uses a causal action-conditioned Transformer over trajectory
history and autoregressive latent rollout. The original LeWM recipe used
SIGReg, but the newer VISReg paper identifies vanishing SIGReg gradients near
collapse and replaces the coupled Gaussian test with explicit center, scale,
and sliced-Wasserstein shape terms. PVZWM therefore defaults this causal model
to VISReg and keeps SIGReg only for a controlled comparison. Reward and terminal
heads are detached probes: they learn environment signals without altering the
JEPA representation gradient.

## Game-world-model implementation notes

The game-focused models suggest three distinct pixel tracks rather than one
generic video predictor:

1. IRIS uses discrete visual tokens and causal autoregression; the local VQ/FSQ
   token model covers this family.
2. DIAMOND trains the RL agent inside a compact diffusion environment model and
   alternates real collection, world-model fitting, and imagination. Its
   uncertainty and small-detail consistency findings are especially relevant
   after the first offline Dyna policy exploited a vector ensemble.
3. [MIRA](https://mira-wm.com/blog-post/) and
   [Open-Oasis](https://github.com/etched-ai/open-oasis) use latent diffusion
   forcing. MIRA's official implementation pairs a representation autoencoder
   with an action-conditioned diffusion Transformer and reports that semantic
   representation quality mattered more than reconstruction alone.

The compact PvZ MIRA-style target will use the existing learned visual encoder
(with a separately trained decoder), per-timestep diffusion noise, causal action
conditioning, and autoregressive latent rollout. Unlike MIRA's four synchronized
players, PvZ has one controller; plant/zombie state is therefore used as a
privileged evaluation probe, not fed to the pixel model. Evaluation will include
long-rollout drift, inverse-dynamics action recoverability, object/board-state
probe error, reward/terminal calibration, and real-environment policy return.
MIRA's 5B/600M scale and 10,000-hour corpus are reference points, not sensible
defaults for this 8×A100 experiment.

The first compact implementation is now executable as `--model flow`. It trains
the representation encoder and decoder from scratch—no released MIRA model or
gated DINO weights—and uses VISReg plus action-conditioned latent flow matching.
This preserves the architectural comparison requested here without importing
external weights or pretending a 5B-parameter reproduction fits the current
data budget.

Because every completed pixel model was substantially worse than copying the
last frame, `--model delta` tests the smallest corrective hypothesis before a
larger video architecture. Its final residual convolution is zero-initialized,
making the untrained prediction exactly persistence. It then learns only the
action-conditioned image delta, with one loss over all pixels and a second over
pixels that actually changed between consecutive frames. The change mask is
computed directly from frame difference and contains no plant, row, or level
labels. Reward, terminal, and legal-action heads remain environment probes. It
uses the same whole-level 2-9/2-10 holdout and may enter planning only if its
open-loop pixel skill relative to persistence is positive.

The first residual run is the closest pixel model so far but still fails that
gate: horizon-1 MSE is 0.004659 versus persistence at 0.004605, and mean
horizon-1--8 skill is -0.242. Its optimized objective was numerically dominated
by changed-pixel, reward, and class-balanced legality losses; the full-frame
pixel term contributed only a small fraction. A matched pixel-priority run now
reduces the changed-pixel coefficient from 1.0 to 0.1 and each auxiliary probe
from its original weight to 0.01. This tests loss interference while retaining
the same model, replay, split, and exact-persistence initialization.

## Control tracks

| Category | Algorithms |
|---|---|
| Model-free | masked DQN / Double-DQN, distributional DQN, PPO, recurrent PPO |
| Planning | random shooting, CEM over categorical actions, beam search, MCTS |
| Model-based RL | Dyna Double-DQN (implemented), Dreamer actor-critic, TD-MPC-style value-guided MPC, MuZero |
| Offline | behavior cloning, CQL, IQL, decision transformer |
| Non-learning controls | random, scripted economy/defense heuristic, oracle simulator MPC |

“Try all” means run representatives of each materially different family under a
shared protocol, not silently sweep incomparable external leaderboard numbers.
The experiment registry will record model, agent, seed, data budget, wall time,
device, and git revision in DuckDB.

## Water-level exploration ladder

Stage 3-1 is a useful generalization boundary: the prior policy reaches it only
after learning day and night play, but the environment introduces a new
prerequisite interaction and remains sparse-reward.  We do not encode water,
Lily Pads, Peashooters, plant prices, or desired placements in the reward.  The
following ladder is ordered by how directly each method addresses observed
failure, and every rung retains authentic terminal game reward as the policy
selection criterion.

| Priority | Method applied here | Primary paper / implementation evidence |
|---|---|---|
| 1 | Factor action choice into family/seed slot and coordinate; preserve rare semantic branches across a bounded action trace in Go-Explore | [Revisiting Action Factorization (2026)](https://arxiv.org/abs/2606.26574) finds branching critics compute-efficient and autoregressive actions strongest overall; [Go-Explore](https://arxiv.org/abs/2004.12919) and its [official repository](https://github.com/uber-research/go-explore) motivate remember-return-explore |
| 1b | Preserve the teacher's demonstrated valid-action structure as policy initialization rather than relearning the combinatorial action manifold during 3-1 | [SPIN (2026)](https://arxiv.org/abs/2601.04441) reports faster and stronger offline RL from a separately learned action-structure model |
| 1c | Co-evolve the authentic ordered seed-packet loadout and the within-level controller, retaining diverse generic action strategies through local competition | [LOKI (NeurIPS 2025)](https://openreview.net/forum?id=B6DhWv3DZo) makes shared-controller/design co-optimization substantially more sample-efficient; [Sample-Efficient Quality-Diversity by Cooperative Coevolution (ICLR 2024)](https://openreview.net/forum?id=JDud6zbpFv) and [PPGA](https://arxiv.org/abs/2305.13795) motivate alternating discrete MAP-Elites search with policy-gradient improvement |
| 2 | Replace brittle exact-prefix return with a goal-conditioned policy trained on achieved archive states through hindsight relabeling | Go-Explore's stochastic robustification; [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495) learns from failed sparse-reward episodes without reward engineering |
| 2b | Train for cross-trajectory stitching, not only imitation of complete traces: cluster generic achieved states and relabel a goal with a later state from a nearby trajectory | [Closing the Gap between TD and Supervised Learning (2024)](https://arxiv.org/abs/2401.11237) and its [official code](https://github.com/RajGhugare19/stitching-is-combinatorial-generalisation) |
| 2c | Expand only through archive states that the current policy can reliably reach, promoting a state to the next curriculum layer after repeated restore/achievement checks | [Layered State Discovery (2023)](https://arxiv.org/abs/2302.03789) formalizes incremental exploration through layers of controllable states |
| 2d | Learn the reliable boundary explicitly from partial successes and failed returns, then refine graph-edge costs from observed low-level failures | [Strict Subgoal Execution / Frontier Experience Replay (ICLR 2026)](https://arxiv.org/abs/2506.21039) and its [official PyTorch code](https://github.com/Jaebak1996/SSE) separate admissible from unreachable subgoals |
| 2e | Keep options interruptible when their predicted value or execution confidence falls; a long build option must not lock the policy out of reacting to a changing board | [On the Suboptimality of SMDPs in HRL (ICLR 2026 submission)](https://openreview.net/forum?id=wH2KwMOojc) identifies fixed option commitment as a structural source of suboptimality; [S3 (ALA 2026)](https://openreview.net/forum?id=lW1hRZPuUk) constrains coarse-dynamics uncertainty in risky irreversible tasks |
| 2f | Learn asymmetric multistep temporal distance between achieved states so the controller can stitch short reliable interactions into a long plan without assuming that reachability is reversible | [Scaling Goal-conditioned RL with Multistep Quasimetric Distances (ICLR 2026)](https://openreview.net/forum?id=UElh7vzgKX) combines multistep value backup with a quasimetric goal distance |
| 2g | Use a self-supervised contrastive goal representation as an exploration control, and compare its learned frontier with graph effective resistance rather than adding another raw novelty bonus | [Demystifying Emergent Exploration in Goal-Conditioned RL (ICLR 2026)](https://openreview.net/forum?id=mwgYORsqtv) analyzes how low-rank goal representations change from exploration to exploitation; [Graph-Theoretic Intrinsic Reward (ICLR 2026)](https://openreview.net/forum?id=W8bKDPf1Ko) directs exploration through effective resistance |
| 3 | Learn controllable interaction novelty from representation change and episodic visitation, with several intrinsic/extrinsic mixtures sharing one recurrent policy | [RIDE](https://arxiv.org/abs/2002.12292) and its [official repository](https://github.com/facebookresearch/impact-driven-exploration); [Never Give Up](https://arxiv.org/abs/2002.06038); [Agent57](https://arxiv.org/abs/2003.13350) |
| 3b | Adapt the intrinsic coefficient from an extrinsic-performance constraint instead of fixing it for the whole run | [Constrained Intrinsic Motivation (2024)](https://arxiv.org/abs/2407.09247) and its [official code](https://github.com/x-zheng16/CIM) target the bias introduced when intrinsic reward distracts from the task |
| 4 | Turn archive cells into learned, achievable intermediate goals chosen for progress toward terminal value rather than hand-labelled mechanics | [DISCOVER (2025)](https://arxiv.org/abs/2505.19850) and its [official repository](https://github.com/LeanderDiazBone/discover); [Director](https://arxiv.org/abs/2206.04114) and its [official repository](https://github.com/danijar/director) |
| 4b | Replace exact full-state goals with learned feature-subset options, and rank reachable frontier goals by progress/cost-to-go | [Going Beyond State-Reaching (2026)](https://openreview.net/forum?id=nhwBVHYIlG) targets option explosion by learning abstract subgoals; [Search-Inspired Exploration (ICLR 2026 submission)](https://openreview.net/forum?id=Rqs5MbvQ3m) balances frontier novelty, reachability, cost-to-come, and cost-to-go |
| 4c | Build a temporal-adjacency graph in learned state space; train an exploration policy to reach graph frontiers and self-imitate successful connections | [Active exploration and hierarchical Self-Imitation (AAAI 2026)](https://ojs.aaai.org/index.php/AAAI/article/view/40076) |
| 4d | Represent subgoals as changes to a small set of learned entities/factors instead of copying the whole board vector | [Hierarchical Entity-centric RL with Factored Subgoal Diffusion (ICLR 2026)](https://openreview.net/forum?id=TimC6hxVHj) and its [official PyTorch code](https://github.com/DanHrmti/HECRL) improve long-horizon image-based sparse-reward control; [Action-Controllable Factorization (2025)](https://arxiv.org/abs/2510.02484) learns independently controllable factors from pixels |
| 4e | Preserve multiple generic subgoal hypotheses across the day/night-to-pool shift, then let authentic return select among them | [Generalizing Subgoals from Single Instances (ICLR 2026 submission)](https://openreview.net/forum?id=jAoh2pj9Id) demonstrates hypothesis-preserving transfer on Montezuma's Revenge and MiniGrid |
| 5 | Once a world model beats persistence, plan toward epistemic novelty and train an achiever over imagined latent goals | [Optimistic World Models (2026)](https://arxiv.org/abs/2602.10044); [Plan2Explore](https://arxiv.org/abs/2005.05960); [LEXA](https://arxiv.org/abs/2110.09514); [DreamerV3](https://arxiv.org/abs/2301.04104) and its [reference implementation](https://github.com/danijar/dreamerv3) |
| 5b | If a world model passes the persistence gate, explicitly train transitions between replay subgoals rather than assuming it can connect trajectories | [MUN (2024)](https://arxiv.org/abs/2411.02446) targets arbitrary key-state navigation and cross-trajectory model generalization |

The immediate archive ablation uses a 64-decision semantic trace.  The trace is
only a set of recent engine action branches such as wait, collect, or an opaque
seed-slot index.  It assigns no bonus to a particular branch.  Its purpose is
to prevent a rare valid interaction from becoming statistically ordinary on
the next WAIT.  This is motivated by the current 3-1 archive: 47 cells descend
from a trajectory containing seed slot 2, but only 28 of 525 selections revisit
those paths, while the last-action selector remains at depth 407 with no
positive-return cell.

The initial bridge also hid an episode-level confound: a compatibility helper
chose a fixed biome-aware deck before every reset. Level 3-1 actually exposes
six ordered seed slots and 17 unlocked seed types (enum values 0 through 16).
The protocol now accepts an exact legal unique loadout and reports both the
authentic capacity and unlocked set. `ordered-loadout-map-elites` mutates packet
identity and slot order, bins candidates only by generic action-trajectory
descriptors, and uses mean unmodified engine return as quality. It does not
require any named plant. Its output is a standard PPO checkpoint with the
selected loadout, enabling an alternating `loadout QD -> controller ES ->
loadout QD` cooperative-coevolution loop. Fog remains locked until a fresh
terminal 3-1 win.

The sequential campaign runner also treats this as a frontier-level decision.
With `--loadout-archive-template`, it resolves a distinct MAP-Elites archive for
the current stage, trains only that stage, and refuses to continue after a win
until the next stage's archive exists. This prevents a loadout learned for 3-1
from becoming an accidental fixed deck for 3-2 or later biomes.

The demo frontier is now frozen at 2-10. A fresh stochastic router invocation
starting at seed 72,002 verified every policy hash, attempted levels strictly in
the order 1-1 through 2-10, stopped each stage at its first authentic terminal
win, and exposed only the immediate successor. It completed all 20 stages in
119 attempts; first-win counts ranged from 1 to 62, with the rare 1-4 specialist
accounting for the maximum. Two failed attempts reached their decision cap, but
none of the 20 winning episodes was truncated. The proof uses 11 frozen
stage-specialist checkpoints. It is continuous at the campaign-router level,
not a claim that one monolithic actor or one persistent engine process solves
the prefix; the bridge is reset between attempts and stages.

The first frozen-controller loadout pass is diagnostic, not a promotion: it
produced seven behavior elites and 0/10 held-out wins. Its highest held-out
return deck omitted Lily Pad, demonstrating that short-horizon combat return
can favor land-only behavior before any terminal success exists. We therefore
retain the full diverse archive—including three Lily-containing elites—and
train one packet-identity-conditioned VDN-PPO controller across those decks.
Only after controller co-adaptation do we rerank/mutate the loadouts again.

That co-adaptation test is now negative but informative. The VDN-PPO controller
collapsed to nearly pure wait: both archive-seeded reranks mapped every deck to
one behavior cell and scored exactly -100 on held-out episodes. To separate
controller collapse from seed choice, matched stochastic outcome-ES runs then
preserved the pre-collapse teacher and evolved policy weights independently for
all seven archived decks. No run won. The top non-Lily and Lily candidates were
extended with larger populations and 20 held-out episodes; they reached mean
returns -99.6111 and -99.7125 respectively, still 0/20. Loadout choice clearly
changes behavior—the Lily candidate uses more slots and makes many more
zero-sun placements—but neither controller family converts that diversity into
terminal pool completion. The authentic loadout interface remains required;
these specific co-adaptation methods are rejected.

The first trace-aware checkpoint confirms that the selector changes behavior:
after 150 additional selections, seed-slot-2 descendant cells increase from 47
to 232 and their deepest prefix from 379 to 391.  The global frontier remains
407 with no positive return.  This separates two problems: interaction
discovery is now demonstrated, while robust return and long-horizon extension
remain unsolved.  Hindsight goals therefore precede further intrinsic-reward
or reward-shaping experiments.

The first robustification step is an archive-start PPO curriculum. It replays
exact engine-valid prefixes to a selected cell, excludes those replayed actions
and rewards from the policy update, and gives the learned policy sole control
thereafter with the unmodified configured environment reward. A fixed fraction
of resets still begins at the original 3-1 start. This implements the return
phase of Go-Explore and the directed-intermediate-goal principle of DISCOVER
without identifying Lily Pads, water rows, plant costs, or desired board
statistics in code. Acceptance remains a fresh full-level terminal win.

The original, not-yet-verified archive-start spatial PPO produced no training
wins and then 0/40 fresh wins across argmax and stochastic evaluation. Its best
argmax episodes increased local bridge return (mean -99.641, maximum -98.861)
without completing 3-1. This is exactly why neither replay depth nor modest
combat return is used as a promotion signal; only terminal completion advances
the campaign.

Because collectible timing makes some nominal archive prefixes brittle, the
curriculum is now filtered by repeated authentic replay before training. Of 383
candidate prefixes at depths 280--337, 225 distinct cells survived two replay
passes. The verifier permits only symmetric WAIT/COLLECT timing repair, records
every executed action, and recomputes the resulting cell key; it does not repair
placements or introduce game-specific state features. This is the operational
version of Layered State Discovery's controllability gate: unreliable cells are
not allowed to define the next training layer.

The two 2026 hierarchical-exploration papers also explain why the earlier raw
random-projection archive plateaued: thousands of nearly identical full states
became separate goals, while depth alone preferred survival over useful
interaction. If verified archive + RIDE still fails, the next structural step
is therefore not a finer state hash. It is a goal-conditioned option learner
whose goals are generic learned feature subsets, with selection restricted to
reliably achievable frontier goals and ranked by learned progress toward
terminal value. Hindsight relabeling supplies training targets from failed
rollouts; no option is named after a plant, lane type, or board tactic.

SSE sharpens the next implementation boundary: a replayable engine prefix is
not yet a learned subgoal. We will train a goal-conditioned low-level policy on
verified and failed prefix returns, estimate the probability and decision cost
of reaching each generic latent cell, and expose only the empirical reliable
frontier to the high-level selector. ASI contributes the complementary
expansion rule: prefer temporally adjacent frontier cells and self-imitate the
connections that actually worked. This directly targets the observed
discover-but-cannot-return failure without rewarding a Lily Pad or any other
specific plant.

The official SSE implementation makes that boundary concrete. It stores
successful and failed high-level transitions in separate replay partitions,
records a partial-success edge when an episode stops short of its requested
subgoal, removes graph edges around repeatedly failed regions, and locally
refines the graph resolution there. The PvZ adaptation will retain those
mechanisms but replace Euclidean maze coordinates with learned generic board
features. An edge is admissible only when a learned low-level policy—not exact
engine replay—reaches it repeatedly on fresh seeds.

The first held-out goal-IQL worker met that admissibility test on verified
prefixes, but its fresh-start hierarchy plateaued at mean archive depth 25.6.
After adding 40 complete clean-reset failures as hindsight data, the next
source-stratified worker reaches 15/17 held-out fresh-source goals in argmax
mode and 16/17 stochastically, while retaining 37/40 verified-prefix goals. No
mower fires in any of these gates. Failure-depth-bounded full episodes then
move the mean reached depth to 36.0 in both modes, but remain 0/20 terminal
wins. These results validate the low-level worker and reject the current
depth-100 edge as too large; the resulting failures become another hindsight
layer rather than grounds for a hand-authored pool tactic.

The resulting water strategy is a staged learned curriculum:

1. Run VDN-PPO over the existing autoregressive action actor so kind, opaque
   seed slot, row, and column receive separate credit. Conditional factors are
   masked and range weights are renormalized; no plant or lane is named.
2. Relabel every failed continuation with generic states it actually achieved,
   learn short goal-conditioned return policies, and estimate each edge's
   success probability and decision cost from held-out attempts.
3. Select the reliable frontier using learning progress, uncertainty, temporal
   adjacency, cost-to-come, and learned cost-to-go. Keep a clean-start fraction
   so prefix competence cannot masquerade as full-level play.
4. Interrupt an option when its reachability confidence falls or a primitive
   action has higher learned value. Successful connections enter
   self-imitation replay; repeated failures narrow or remove the edge.
5. Promote only a fresh authentic terminal 3-1 win, then run a mower-disabled
   robustness block before exposing 3-2. This is the only step that changes the
   sequential campaign frontier.

The learned edge model is directional. Building a board state and recovering
from it are not interchangeable, so the selector estimates an asymmetric
multistep temporal distance rather than Euclidean distance in the 585-value
vector. A small quasimetric/contrastive ablation will determine whether the
same representation can both stitch archive fragments and create useful
exploration pressure. Graph effective resistance is retained as a frontier
ranking control, not added to authentic reward, because it can prioritize
bottlenecks without assuming a named pool mechanic. Candidate methods advance
only if held-out goal reachability improves; terminal game completion remains
the policy criterion.

The official HECRL code supplies a useful selection rule for step 3: generate a
set of candidate subgoals, reject candidates whose learned current-to-subgoal
value lies outside the worker's competence radius, select the survivor with the
best subgoal-to-task value, and bypass the subgoal when direct task value is
already higher. Its offline buffer mixes next-state, later-same-trajectory, and
random-trajectory goals, while the high-level actor is advantage-weighted toward
future states that improve goal value. We will learn the competence cutoff from
held-out edge-success quantiles and replan on confidence/value changes rather
than copy HECRL's task-specific fixed threshold or 25-step commitment.

The first executable low-level worker now implements this replay interface. A
remote-only DuckDB corpus replays each twice-verified archive prefix up to four
additional times and retains only complete vector/action/legal-mask sequences.
This stricter gate accepted 60 of 225 prefixes (18,253 transitions) and rejected
165 that still produced a non-housekeeping illegal action; the rejected traces
do not enter training. Goal-IQL uses shared spatial encoders for current and
goal states, an autoregressive 650-action actor, and a 20/60/20 relabel mixture
of next-state, later-same-trajectory, and random-trajectory goals. Its learned
value is directional, because build and recovery paths need not be symmetric.
This worker establishes short-horizon reachability; a high-level frontier
selector is promoted only after held-out goal-achievement tests.

The first all-data fit separates immediate from random-trajectory goals in its
training diagnostic, but that is not a generalization result. The promotion fit
therefore uses an episode-disjoint 48/12 split. Its engine gate restores each
unseen prefix to a sampled start, asks for a nontrivial future archive cell at
1, 4, 16, or 64 decisions, and gives the learned worker up to twice the recorded
horizon. A task enters the denominator only if replaying the held-out recorded
continuation still reaches that cell under the current engine. Argmax and
stochastic policies are reported separately, along with restore failures and
mower use.

The held-out argmax gate passes this low-level criterion: 45 of 48 sampled
tasks remained reachable by their recorded continuation, and the learned worker
reached 42/45 (93.3%). It was 10/10, 12/12, 11/11, and 9/12 at offsets 1, 4,
16, and 64 respectively, with no mower use. This promotes the worker for a
generic high-level selector experiment; it does not promote Adventure progress,
which still requires a fresh terminal 3-1 win.

The stochastic gate independently reaches 40/46 replay-eligible goals (87.0%),
with offset success of 12/12, 11/11, 9/11, and 8/12. The agreement rules out a
pure argmax artifact and clears the worker for hierarchical use; it still says
nothing by itself about terminal pool completion.

The first high-level executor is now a held-out-calibrated generic frontier
selector. It fits a goal-value competence threshold by balanced accuracy on the
two engine reachability reports, considers only deeper random-projection archive
cells inside that learned region, and blocks/replans a goal after execution
failure. Depth is archive action depth, not a plant or board statistic. The
fresh-start evaluation reports maximum robustly reached depth and terminal game
outcome separately; only the latter can advance the campaign.

The first ten fresh starts reached mean archive depth 25.6 and maximum 72 with
zero wins or mower use. Several seeds robustly chained depths 24, 44, 64, and
72, but after an option failed the optimistic value model proposed unrelated
deeper goals. This is useful negative evidence about held-out-prefix calibration
under a truly fresh-state distribution. The matched revision turns the first
failed depth into a ceiling: it may try another goal at or below that layer but
cannot skip forward until an option actually succeeds.

The first depth-bounded report is retained but its mean return is invalid: one
seed exhausted eligible goals before the game terminated, leaving an incomplete
zero-return trajectory in the aggregate. The corrected evaluator continues
housekeeping-only wait/collect actions after hierarchy exhaustion until the
engine returns win/loss or the configured cap, and records those fallback steps
explicitly. No return from an incomplete hierarchy episode is used for policy
selection.

The corrected argmax gate is 0/10 terminal wins, mean authentic return -99.963,
mean reached depth 25.6, and maximum depth 72; one episode required the explicit
completion fallback and none used a mower. This confirms real fresh-start early
layer chaining but rejects the current hierarchy as a pool solution. A matched
stochastic gate tests whether sampling expands that fresh-state competence.

It does not: stochastic execution is 0/10, mean return -99.98, mean reached
depth 18.4, and maximum 72, again without mowers. The held-out restored-prefix
worker is useful but its competence does not yet transfer deeply enough from a
clean reset. The evaluator can now append every full vector/action/legal-mask
trajectory to a remote-only DuckDB. A new 40-episode collection supplies the
actual fresh-start distribution for another hindsight-IQL layer rather than
tuning the selector against these evaluation seeds.

That collection contains 40 complete episodes and 16,140 transitions; combined
with the verified-prefix corpus it yields 100 episodes and 34,393 transitions.
No episode won, but one clean-reset trajectory robustly chained archive goals to
depth 128. The next worker warm-starts every actor, critic, target critic, and
value parameter from the held-out-passing checkpoint, then uses a source-
stratified 80/20 episode split so fresh and restored behavior are represented in
both training and validation.

Options are not assumed to help automatically. [When Do Skills Help RL?
(ICML 2024)](https://arxiv.org/abs/2406.07897) predicts that skills mainly aid
exploration when successful behavior is compressible and warns that weak fixed
macro-actions can make learning worse. We will therefore learn termination and
reachability from replay, compare against the flat verified-archive controls,
and retain a hierarchical policy only on fresh terminal success.

The implemented autoregressive actor factors the normalized joint probability
as `kind -> opaque seed slot -> row -> conditional column`, then maps it back to
the unchanged 650-action vocabulary before legality masking. It contains no
seed identities, costs, water-row labels, or placement targets. Controlled
clean-start and archive-start PPO runs use a branch-balanced autoregressive BC
teacher fitted to exactly the same trace replay as the spatial-convolution
teacher. Plain teacher-initialized PPO and PSN-GRPO both completed 400,000
environment steps with zero training wins, so coherent parameter noise alone
does not bridge the sparse pool frontier. A branching critic remains a distinct
future action-credit ablation. The first RIDE rung is implemented as learned inverse-dynamics
features, feature-space transition magnitude, and generic episodic visitation
decay from a fixed random projection. The clean-start RIDE control also ended
with zero wins across 1,087 training episodes; its mean intrinsic reward was
0.660 and repeatedly reached the 5.0 clip while recent extrinsic return fell to
-100.0. This is evidence that fixed novelty can distract from survival, not
evidence against controllable-change representations themselves. Clean-start RIDE is the control; the two
current priority runs combine the verified archive with RIDE under spatial and
autoregressive policies. If fixed-weight RIDE discovers more interaction but
hurts survival, the next ablation is CIM-style adaptive weighting, not a
plant-specific bonus. Intrinsic return is never evidence of level completion.
A genuine terminal 3-1 win on a fresh seed is required to unlock 3-2, followed
by a mower-disabled robustness block.

The completed fresh controls reinforce that decision. Clean autoregressive PPO
is 0/20 argmax (mean return -99.869), clean fixed-RIDE spatial PPO is 0/20
argmax (mean -100.0), and verified-archive fixed-RIDE spatial PPO is 0/40 over
argmax and stochastic modes (both means -100.0). None used a mower. Fixed RIDE
therefore fails both discovery-to-control transfer and fresh survival here; it
is not carried into VDN-PPO by default.

Clean-start VDN-PPO also failed: 0 training wins in 1,223 episodes followed by
0/20 argmax and 0/20 stochastic fresh evaluations, both at mean return -100
and with no mower use. Branching credit assignment alone therefore does not
solve pool exploration; VDN remains useful inside the loadout-conditioned and
goal-conditioned curricula where the missing long-horizon structure is present.
Verified-archive VDN-PPO is likewise 0/40 fresh: argmax mean return -100 and
stochastic mean -99.9865, with no mower use. The matched clean and archive
results reject VDN-PPO as a standalone pool solution while preserving it as a
credit-assignment component in the co-adaptation experiment.

Two matched verified-archive-only PPO runs now isolate that interaction: one
uses the spatial policy and one the autoregressive action factorization. Their
paired verified-archive + RIDE runs differ only by the intrinsic module. This
four-way comparison determines whether reliable return states, action
factorization, or controllable-change novelty contributes; it avoids attributing
an improvement to a bundle of simultaneous interventions.

The CIM-style controller is now executable. It maintains a smoothed extrinsic
episode-return estimate and a monotone learned reference, updates a nonnegative
dual variable when current extrinsic performance falls below that reference,
and applies `base_weight / (1 + dual)` to RIDE. The return gap is normalized by
the learned reference magnitude, so there is no hand-picked PvZ return target.
Its complete state is checkpointed for continuation. A verified-archive spatial
PPO run now compares this adaptive coefficient with the otherwise matched fixed
0.05 RIDE run.

The action-factorization paper's full derivation changes the branching-critic
choice. The next credit-assignment experiment should be VDN-PPO, not PPO-MIX:
learn a global state value plus per-factor advantage streams, center each stream
under the behavior policy, fit their sum to the value-only TD residual, and
weight each factor's GAE by the range of its learned advantages. The paper shows
that the weights sum back to ordinary GAE while moving variance toward factors
with state-dependent agency. Its practitioner guidance explicitly warns against
PPO-MIX/QPLEX as action spaces grow; PvZ already has 650 joint actions. For the
conditional PvZ schema, inactive slot/row/column factors must be masked and the
remaining weights renormalized rather than pretending every action uses all
four branches.

That VDN-PPO path is now implemented as
`portable-vdn-autoregressive`. The behavior policy's legal joint distribution
is marginalized once per rollout into the four factor distributions, critic
advantages are centered under those frozen probabilities, and the critic fits
their active sum to the value-only TD target. Range importance anneals from
uniform to full weighting over training. The per-factor weighted GAEs are
constructed so their sum is ordinary normalized GAE, including after inactive
factor masking.

World-model optimism remains gated.  The 2026 optimistic loss is not applied to
a predictor that is worse than persistence, because that would amplify model
bias rather than provide meaningful exploration.  FSQ, RSSM, and representation
flow must first pass the held-out dynamics criterion already recorded in the
authentic world-model manifest.

[MIRA (2026)](https://arxiv.org/abs/2607.05352) and its
[official code](https://github.com/mira-wm/mira) add three useful world-model
controls without changing that gate: a representation autoencoder rather than
raw-pixel capacity, action recoverability plus privileged-state probes for
mechanics instead of visual quality alone, and causal latent flow matching. The
released default uses 0.1 action dropout and explicitly conditions denoising on
clean past latents to avoid the diffusion-forcing train/inference mismatch; we
now implement both controls rather than inferring a different recipe from the
demo prose. Action dropout zeroes the flow action condition without changing
the embedding or checkpoint shape; outcome heads retain the real action. For
PvZ, authentic vectors remain evaluation probes rather
than pixel-WM inputs, and a model must pass both persistence and action-following
tests before any imagined transition reaches RL.
The evaluator now adds a direct one-step counterfactual action-following probe:
for each transition with another legal action, it compares target-pixel error
under the recorded action against the model's prediction under that alternative.
It reports both the error margin and the fraction preferring the recorded action,
so an action-agnostic flow cannot pass on reconstruction quality alone.

[WorldCompass (2026)](https://arxiv.org/abs/2602.09022) suggests a later
post-training step once that base model is credible: branch several next-clip
predictions from a shared history, score action following and visual validity
separately, and progress the target horizon. It is not a license to optimize a
currently inaccurate model; the shared-prefix group rollout is useful only
after positive held-out skill.

All three four-episode baseline runs failed that gate. To separate model-family
failure from data scarcity, FSQ and representation flow/VisReg are also trained
on the expanded split: 18 episodes spanning 1-10 through 2-8, with the six 2-9
and 2-10 episodes held out by level. They remain barred from planning unless
their held-out pixel skill relative to persistence becomes positive.

The implemented Dyna path follows Sutton's integration of real experience and
learned-model planning ([1990 paper](https://www.derongliu.org/adp/adp-cdrom/refs/sutton19900216.pdf)).
It starts short imagined branches from replay states, following the model-usage
finding in [MBPO](https://github.com/jannerm/mbpo), and subtracts ensemble
uncertainty from imagined rewards while rejecting high-uncertainty branches,
following the conservative principle in [MOPO](https://arxiv.org/abs/2005.13239).
The world model predicts environment dynamics only. A masked Double-DQN learns
the control policy; scripted policies are data/debug baselines, never an
acceptance criterion.

Plant selection is a learned episode-level decision, not a fixed campaign
constant. The vector observation exposes the ordered seed-packet identities,
availability, and recharge state, so one controller can condition its in-level
actions on different authentic decks. Each frontier maintains its own ordered
loadout archive; old decks are re-evaluated with common random seeds before
mutation, and the archive is searched again for every newly trained PPO, DQN,
or Dyna-DQN controller. A 3-1 promotion therefore evaluates the coupled
loadout/controller pair on fresh seeds. It does not reject a controller using
only the game's compatibility deck or carry a successful daytime deck into the
pool by assumption. Value-based controllers expose an explicit legal-action
epsilon-greedy loadout gate, kept separate from actor-policy categorical
sampling, so exploration variants do not silently change algorithm semantics.

The pool vector-model gate now has two passing ensembles. The one-step model
achieves 0.374 held-out state skill over persistence. The stronger eight-step,
changed-state-weighted ensemble reaches 0.488 skill, 0.920 legality recall, and
0.999 legality precision. Both are limited to short uncertainty-gated Dyna
branches; the latter is the primary pool Dyna model. These metrics only license
imagined training data, not Adventure progress.

The matched controller result is negative. Real-only DQN, one-step-ensemble
Dyna, three-step multistep-ensemble Dyna, and a conservative horizon-one/25%
model-ratio Dyna all completed 100,000 updates. Every controller received its
own ordered-loadout MAP-Elites search and 20 fresh held-out episodes. Their win
rates are all zero. The strongest return is the three-step multistep Dyna at
-98.754; an explicit epsilon-0.05 rerun is also 0/20 at -98.800. Useful vector
dynamics therefore improves behavior but does not by itself bridge terminal
pool exploration.

The next abstraction uses the passed pool JEPA rather than a hand-authored water
feature. All 42,008 states from 120 authentic episodes are encoded and grouped
by deterministic spherical k-means. Each of 256 cosine clusters stores a real
medoid state, whose own replay timestep is its depth; cluster depth is never
inflated by a deep outlier. The existing goal actor can therefore consume a
real vector goal while option completion uses learned cluster equivalence.
Across a common ten-seed screen of all seven learned decks, no hierarchy wins.
Two Lily-Pad decks nevertheless reach mean medoid depth 45.8 and maximum 98,
whereas the environment-return winner reaches no cluster goal. This diversity
is retained rather than collapsed into a manually chosen pool statistic.

Cluster-equivalent hindsight IQL is now implemented. Next/future/random goals
are represented by their real cluster medoid, and reward zero is assigned when
the next state belongs to the same learned JEPA cluster, even when the raw
585-dimensional vectors differ. The warm-start corpus adds 40 complete JEPA
hierarchy trajectories to the prior replay for 160 episodes and 56,261
transitions. This changes goal equivalence only; it adds no plant identity,
water-row rule, or shaped reward.

The first cluster-equivalent worker reaches 48/48 held-out restored-prefix
tasks in both argmax and stochastic modes, including 12/12 at offset 64, with no
mower use. Its fresh full-archive screen is still 0/35, but the return-leading
deck now reaches mean medoid depth 136.6 and maximum 345 instead of zero before
cluster-equivalent training. The 20-seed gates remain 0/20 in both modes
(argmax mean depth 101.3, stochastic 84.5). A second layer therefore adds these
40 clean-start failure trajectories, producing 200 episodes and 71,598
transitions. Its competence threshold is calibrated jointly on verified
prefixes and clean-start sources; the all-success first-layer threshold is not
reused. The resulting threshold is -3.1675 from 71 eligible tasks with 66
successes. Restored-prefix reachability remains strong at 44/47 without mower
use, but the common fresh five-seed screen regresses to 0/35 and return -100 for
every inherited deck. Only `[4, 7, 12, 13, 3, 0]` reaches an accepted cluster,
at depth 91. Further replay stacking is therefore stopped.

The next outer loop searches the full authentic unlocked seed-packet space for
the selected goal worker. This matters because plant selection is an action at
the beginning of every level: the within-level worker and ordered loadout must
be evaluated as one policy. The seven inherited decks remain useful seeds, but
they are not treated as the complete action set. Selection fitness remains the
unmodified terminal environment return; learned latent progress is diagnostic
and diversity-preserving, not a plant-specific reward.

That outer action is now implemented as worker-specific ordered-loadout
MAP-Elites. Each run evaluates 24 decks for 12 generations over all 17 unlocked
types and the authentic six-packet capacity, with eight parallel game workers.
Half of proposals are full-set restarts, so mutation is not trapped around the
seven inherited decks. Diversity uses generic executed-action frequencies;
fitness orders terminal win rate first and raw environment return second.

One-seed evolutionary winners were optimistic. V1 reached a training-cell
return of -98.55 but its selected deck was 0/20 at -99.780; V2 reached -99.73
but was 0/20 at -99.973. The final selector therefore reevaluates every retained
behavior elite on five common fresh seeds before a disjoint 20-seed gate. V2
remains 0/20 at -99.960. V1 is recalibrated with three external clean-start
failures explicitly excluded from training, moving its competence threshold
from -48.23 to -45.76. Its learned deck `[11, 15, 12, 6, 16, 0]` improves to
-99.465 over the final 20 seeds but still wins 0/20. Its zero-sun placement
fraction is 0.427 despite that statistic never entering fitness. This supports
the learned-affordance hypothesis while rejecting loadout choice alone as a
solution to the remaining 3-1 control failure.

## Data and asset policy

Gameplay facts and an initial taxonomy can be checked against the community
[wiki.gg PvZ plant index](https://plantsvszombies.wiki.gg/wiki/Plants_%28PvZ%29).
Wiki text or images are not automatically copied into this repository. Raw game
assets remain the property of their respective owners; users must provide a
legally obtained game installation or their own captures. Derived metadata keeps
source URL, retrieval time, license/terms notes, and content hash.

The symbolic simulator is deliberately clean-room and approximate. It exists to
validate learning and planning infrastructure before introducing screen capture,
computer vision error, and real-game timing.

## Apple Silicon rules

- Select `mps` before CUDA/CPU when available.
- Keep numerically sensitive distribution matching and losses in float32.
- Avoid CUDA-only fused kernels and libraries.
- Benchmark eager and `torch.compile` separately; do not assume compilation is
  faster on every MPS shape.
- Log CPU fallbacks because silent fallback can invalidate throughput results.
