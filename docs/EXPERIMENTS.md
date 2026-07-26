# Baseline experiment log

These are research baselines, not final claims. Early Python-port experiments
ran locally on MPS; explicitly labelled authentic experiments ran on private
A100s. All splits are whole-episode or whole-stage splits.

## Authentic rich-vector-v2 world models

The vector-v2 corpus contains 12,552 transitions from all 50 Adventure stages.
Its 585 features retain plant and zombie type, health, internal phase/state,
sub-cell position, seed packets, and campaign timing. Every comparison uses the
same 40 training stages, 10 wholly unseen validation stages, seed 19, 650-action
vocabulary, and eight-step open-loop evaluation. A persistence baseline copies
the initial state through all eight predicted transitions.

The initial five-epoch comparison exposed an important failure hidden by raw
low MSE:

| Model | Model MSE | Persistence MSE | Skill vs persistence | Reward MAE |
|---|---:|---:|---:|---:|
| Residual MLP | 0.01123 | 0.001430 | −685.5% | 0.01583 |
| 5-member ensemble | **0.002954** | 0.001430 | −106.5% | **0.008085** |
| Gaussian RSSM | 0.01385 | 0.001430 | −868.3% | 0.008528 |
| VISReg JEPA | latent 0.009811 | latent 0.001139 | −761.3% | 0.008438 |

JEPA latent MSE is not numerically comparable with state MSE. Its cosine
distance is 0.03901, effective rank 14.73, and active-feature fraction 100%.
Despite those healthy representation diagnostics, it does not yet predict
unseen-stage latent evolution better than copying the initial embedding.

Residual-state initialization was then isolated from sparse-change weighting.
The state head now initializes to zero, so an untrained residual model exactly
matches persistence instead of adding random drift.

| Ensemble ablation, 20 epochs | State MSE | Skill vs persistence | Reward MAE |
|---|---:|---:|---:|
| Zero-init, ordinary state loss | **0.001410** | **+1.41%** | **0.007688** |
| State weight 10, changed-feature weight 5 | 0.002612 | −82.65% | 0.01217 |
| State weight 20, changed-feature weight 20 | 0.006293 | −340.0% | 0.01070 |

The ordinary zero-initialized objective is retained. The positive skill is
real but marginal: it beats persistence through horizon six and trails it at
horizons seven and eight.

Direct open-loop training was then tested with episode-safe windows and no
teacher forcing after the first state. It averages state, reward, done, and
legality losses across the imagined trajectory:

| Ensemble rollout objective, 20 epochs | State MSE | Skill vs persistence | Reward MAE |
|---|---:|---:|---:|
| Horizon 2 | 0.001945 | −36.00% | 0.007174 |
| Horizon 4 | 0.002606 | −82.22% | 0.006652 |
| Horizon 8 | **0.001833** | **−28.14%** | **0.006410** |
| Horizon 4, policy-augmented | 0.004517 | −215.80% | 0.008691 |

Rollout-only training does not improve held-out state prediction, so the
one-step ensemble remains the selected predictor. Surprisingly, the horizon-8
model is much better for planning: categorical CEM wins 9/9 authentic 1-1
episodes across seeds 10000–10008, with mean 219.3 decisions and 94.33% wait
actions. This is a control result rather than an MSE result, so both checkpoints
are retained for their separate roles.

Adding 50 stochastic-PPO 1-2 episodes expands training from 12,552 to 34,718
transitions and includes one recorded 1-2 win. Retraining the selected ensemble
does not improve the primary metrics: state skill falls from +1.41% to +1.20%
and reward MAE rises from 0.00769 to 0.01185. Legality BCE improves from 0.0437
to 0.0320. Mostly losing policy data improves action-mask coverage but is not a
substitute for deliberately oversampling successful/reward-bearing experience.

## Authentic sequential curriculum control

`PortableSequentialCurriculum` enforces the Adventure order. A frontier stage
unlocks only after a win; optional rehearsal may revisit already-unlocked
stages but can never unlock a later stage out of order. Checkpoints record every
completed episode's level, return, and win flag.

The first 30k-step masked Double-DQN run used no rehearsal. It reached 1-3,
which proves it won 1-1 and 1-2 in order, but its 74 completed episodes were
distributed 1 on level 1, 3 on level 2, and 70 on level 3. Its final greedy
policy catastrophically forgot the earlier behavior:

| Greedy DQN evaluation | 1-1 | 1-2 | 1-3 |
|---|---:|---:|---:|
| Win rate, 10 seeds | 0% | 0% | 0% |
| Mean return | 0.000 | −99.720 | −99.768 |

Adding 30% rehearsal and extending DQN to 50k steps produces 31/32 training
wins on 1-1, 3/47 on 1-2, and 0/49 on 1-3. Greedy evaluation still scores 0/10
on all three stages. Evaluation at its final training exploration rate
(ε=0.05) scores 10/10 on 1-1 but 0/10 on 1-2 and 1-3. Rehearsal preserves the
easy-stage behavior only while exploration remains active.

A matched 50k-step PPO run used 30% rehearsal over already-unlocked levels.
During training it recorded 24/24 wins on 1-1, 2/33 on 1-2, and 1/77 on 1-3.
The final stochastic policy retains some of that cross-level behavior, whereas
its masked argmax mode collapses:

| PPO evaluation, 10 seeds | 1-1 | 1-2 | 1-3 |
|---|---:|---:|---:|
| Masked argmax win rate | 0% | 0% | 0% |
| Stochastic win rate | **100%** | **10%** | 0% |
| Stochastic mean return | 101.350 | −78.960 | −99.655 |

An additional 20-seed stochastic 1-2 sweep wins 2/20 with mean return −79.157.
Pooled across disjoint evaluation seeds, PPO wins 3/30 (10%) on 1-2.

Offline Dyna remains unsuccessful on the rich corpus. The conservative run
accepted 625,146 imagined transitions and rejected 14,854 by uncertainty; the
loose run accepted 1,280,000 and rejected none. Both score 0/5 on each of 1-1,
1-2, and 1-3. The replay audit explains the collapse pressure: 11,385 of
12,552 actions are wait and 11,694 rewards are zero. A stratified replay
ablation therefore samples half of each real batch from non-wait or rewarded
transitions. Its first evaluation scores 0/5 on 1-1, 1/5 on 1-2, and 0/5 on
1-3. An additional 20-seed 1-2 sweep wins 1/20 with mean return −89.628;
pooled across disjoint seeds, stratified Dyna wins 2/25 (8%) on 1-2.

Categorical CEM with the selected one-step predictor fails 1-1: it chooses wait
for 100% of 1,200 steps and returns zero. The rollout-h8 planner reverses that
result on 1-1, winning 9/9, but scores 0/8 on 1-2 with mean return −99.438.
On 1-2 it selects 15–22 distinct actions per episode and waits 87.75% overall,
so the failure is no longer the trivial all-wait collapse. These controls are
retained as model-exploitation and data-coverage evidence, not discarded
hyperparameters.

The initial categorical update also contained a scale bug: adding a pseudocount
of 0.5 to every one of 650 actions gave the prior 325 virtual samples against
only 32 real elites. The corrected update assigns a fixed total prior mass over
currently legal actions. A matched 1-2 sweep over horizons 8, 16, 24, and 32,
larger sample counts, and uncertainty penalties still wins 0/7. Longer horizons
increase predicted returns without improving real returns, which is direct
evidence of model exploitation rather than insufficient CEM search alone.

Eight successful 1-1 CEM trajectories were then recorded and distilled with
legality-masked behavior cloning. Balancing the replay sampler to draw 50%
non-wait decisions produces 10/10 held-out wins on 1-1 in both argmax and
stochastic modes. PPO initialized from that policy generates successful 1-2
experience; retaining successful trajectories and repeating the loop produces
the following current policy benchmark:

| Success-replay BC, held-out seeds | 1-1 | 1-2 | 1-3 |
|---|---:|---:|---:|
| Stochastic | 10/10 | 4/10 | 3/50 |
| Masked argmax | 10/10 | 0/10 | 2/10 |

The 1-3 stochastic result pools five disjoint 10-seed batches and has mean
return −87.066. This is the first single checkpoint with held-out stochastic
wins on all three sequential levels, but 6% on 1-3 is not mastery. A continued
PPO checkpoint reaches 20/40 (50%) on held-out 1-2 yet 0/10 on 1-3, showing why
single-level continuation is not a sufficient curriculum.

Successful replay also improves the learned dynamics. Adding the 1-1/1-2
success corpus to the 50-stage coverage replay changes the matched ensemble
results as follows:

| Success-augmented ensemble | State MSE | Skill vs persistence | Reward MAE |
|---|---:|---:|---:|
| One-step objective | **0.001395** | **+2.45%** | 0.007604 |
| Horizon-8 rollout objective | 0.001589 | −11.10% | **0.005563** |

The rollout model still trails persistence as a predictor, but improves over
the earlier −28.14% rollout checkpoint and its CEM planner wins 1/4 on 1-2
(versus 0/8 before), 2/2 on 1-1, and 0/2 on 1-3. This is a positive iteration
of the world-model loop, not evidence that planning has solved the curriculum.

### Level 1-4 frontier

PPO initialized from the 1-1→1-3 success-replay policy and trained with 30%
rehearsal reached 1-4 in order. Training recorded 23/23 wins on 1-1, 4/26 on
1-2, 1/22 on 1-3, and 0/124 on 1-4. Its initial stochastic evaluation retained
10/10, 2/10, 1/10, and 0/10 wins respectively. This unlocked 1-4 but did not
initially solve it.

Frontier-only controls confirm that more of the same optimizer is insufficient:

| Level 1-4 training/control | Episodes | Wins |
|---|---:|---:|
| PPO continuation, learning rate 3e-4 | 202 | 0 |
| PPO continuation, learning rate 1e-4 | 191 | 0 |
| Double-DQN | 256 | 0 |
| PPO, clipped-damage reward | 195 | 0 |
| Double-DQN, clipped-damage reward | 261 | 0 |
| Legality-aware random exploration | 100 | 0 |

`clipped-damage` is an explicit ablation that clips negative health deltas from
new wave spawns but preserves +100 wins and −100 losses. It adds no handcrafted
strategy and is stored in the checkpoint; it does not replace the default raw
bridge reward.

Four 75k-step Random Network Distillation sweeps also score zero training wins:
clipped-damage with intrinsic weights 0.01, 0.05, and 0.1, and raw bridge reward
with weight 0.05. The RND predictor, target, optimizer, and running novelty scale
are checkpointed. Novelty remains measurable, but the selected RND checkpoint
forgets 1-2 and 1-3 in a matched held-out sweep, so novelty alone is not the
missing curriculum mechanism.

An experiment-config audit found that the CLI exposed discount, GAE, PPO clip,
value, and entropy flags without forwarding them into `PPOTrainConfig`. The four
historical checkpoint filenames containing `lh`, `g1`, `l1`, or `e05` therefore
do **not** establish long-horizon or entropy ablations: their immutable DuckDB
rows all record effective discount 0.99, GAE lambda 0.95, and entropy 0.01. The
selected run did use the requested 2,048-step rollout batch, recorded no win
during 75k training steps, and scored 0/10 on its first held-out 1-4 batch, yet
a wider 100-episode stochastic recording produced two authentic wins. Those
wins remain valid but are attributed to rollout-2048 PPO, not undiscounted
credit assignment. Forwarding is fixed as of 2026-07-15 and the true matched
long-horizon sweep must be rerun.

A portable-specific spatial actor encodes the 6×9 lawn and seed packets with
shared cell/slot representations and factorized plant logits. Its selected
success-replay clone scores 10/10, 4/10, 1/10, and 0/10 on levels 1-1 through
1-4. Three fixed-level 100k PPO variants (standard, Monte Carlo, and Monte
Carlo plus RND) all remain at zero wins. Spatial structure improves transfer on
earlier levels but is not sufficient for the frontier.

The first successful policy-iteration loop ranks complete episodes by realized
return, clones the top 20%, and balances half of each update toward non-wait
actions. Its evidence is deliberately reported by generation:

| Learned-policy generation | 1-4 episodes | Wins | Interpretation |
|---|---:|---:|---|
| Rollout-2048 PPO | 100 | 2 | First authentic learned-policy 1-4 wins |
| Top-return BC child | 100 | 2 | Independent producer also retains rare wins |
| Next top-return BC child | 100 | 0 | Regression; iteration is not monotonic |

The first two generations contain 93,850 transitions and four exact winning
trajectories. Uniform success-only cloning over the combined 1-1→1-4 corpus is
not robust: the selected clone scores 10/10, 0/10, 1/10, and 0/10. Therefore
1-4 is now *occasionally solved* at 2% in two independent 100-episode producer
batches, but it is not mastered and 1-5 is not yet a reliable sequential
frontier.

A separate full-logical-resolution visual sweep confirms this is not merely a terminal-label
artifact. The rollout-2048 policy wins seed 85042 after 1,132 decisions (return
108.159), stopping a bounded sweep after 43 episodes. During that live episode,
zero-tick `observe` requests sampled 143 frames from the engine's fixed 800×600
logical framebuffer; they do not advance game time or substitute recorded
actions. The resizable SDL window may have a different presentation resolution;
the retained H.264 video bypasses that scaling and is
`artifacts/level4-learned-win-native-800x600.mp4`. Replaying an old action list
from only the episode seed did not reproduce the original rollout, which is
expected because vector-v2 deliberately omits engine RNG and coin positions;
the visual proof therefore records a new win live rather than claiming
bit-exact offline reconstruction.

The 100 random failures add 40,144 complete 1-4 transitions. Merging them with
all-stage coverage and successful 1-1→1-3 replay yields 278 episodes and
109,343 transitions. Eight-A100 training on that corpus substantially improves
the one-step ensemble:

| Frontier ensemble | State MSE | Persistence MSE | Skill | Legality recall / precision |
|---|---:|---:|---:|---:|
| One-step, 20 epochs | **0.000952** | 0.001083 | **+12.11%** | 93.5% / 87.0% |
| Rollout-h8, 10 epochs | 0.001211 | 0.001083 | −11.81% | 91.7% / **99.4%** |

Despite the predictive gain, one-step CEM is 0/4 on 1-4 and rollout-h8 CEM is
also 0/4. The rollout planner is somewhat more active, choosing 13–22 distinct
actions and waiting 86.7–88.9%, but it still cannot produce a success. The next
iteration needs directed exploration or a stronger long-horizon value signal,
not another unqualified claim that lower MSE solves control.

Adding the four successful 1-4 trajectories and their surrounding rollouts
expands the world-model replay to 478 episodes and 203,193 transitions. A new
eight-A100 one-step ensemble obtains state MSE 0.000995 against persistence
0.001131, or +12.06% skill, with reward MAE 0.00825 and legality recall/precision
92.45%/90.61%. Its CEM controller is still 0/8 on 1-4. Successful experience
has entered the learned environment model, but current model-based planning has
not converted it into a win.

### Offline IQL and sequential progress through 1-8

Discrete IQL retains all 243 level-1-4 episodes rather than discarding failures.
It oversamples the five successful episodes, learns twin discrete critics and an
expectile value model, then weights legal behavior cloning by learned advantage.
Four matched 50-epoch variants produce this held-out result:

| IQL actor | 1-4 wins / 50 | Earlier-level retention |
|---|---:|---|
| Flat, expectile .7, success fraction .5 | 1 | not selected |
| Flat, expectile .9, success fraction .5 | 1 | not selected |
| **Flat, expectile .7, success fraction .75** | **2** | 20/20 on 1-1, 10/20 on 1-2, 6/20 on 1-3 |
| Spatial, expectile .7, success fraction .5 | 0 | not selected |

An independent 100-episode recording of the selected actor wins only 1/100, so
the apparent 4% rate in the first batch is optimistic noise. IQL's demonstrated
gain is cross-level retention, not robust 1-4 mastery.

Using that actor as the win-gated initializer advances the authentic campaign:

| Exposed frontier | Selected effective `(discount, GAE)` | Training frontier wins | Fresh held-out frontier wins |
|---|---|---:|---:|
| 1-5 | (0.99, 0.95) | 79/284 | 13/30 |
| 1-6 | (0.999, 0.995) | 10/159 | 3/30 plus 9/50 independent |
| 1-7 | (0.999, 0.995) + RND | 0 under the invalid 1,200-step cap | **4/20** with the corrected 3,600-step cap |
| 1-8 | (0.999, 0.995) + RND | reached after an authentic 1-7 win | **2/20**, plus 0/5 independent smoke |
| 1-9 | (0.999, 0.995) + RND | reached after an authentic 1-8 win | **3/20**, plus 0/5 independent smoke |
| 1-10 | (1.0, 1.0), entropy .05 + RND .01 | reached after an authentic 1-9 win | **14/20** |
| 2-1 | unchanged 1-10 checkpoint | zero night-training updates | **8/10 zero-shot** |
| 2-2 | default PPO, 400k decisions | 14/20 recent training episodes | **13/20** |
| 2-3 | default PPO, 400k decisions | 16/20 recent training episodes | **20/20** with deterministic sun pickup |

Every frontier was exposed only after a real win on its predecessor. The first
1-7 report was wrong for a concrete infrastructure reason: all evaluations used
the 1,200-decision cap inherited from shorter stages. Twenty-two nominally
positive failures reached that cap while still alive at waves 11–14 of the
20-wave level. They were truncations, not wins or losses. The replay selector
now honors explicit `metadata.won` before shaped return, the full-game default
is 3,600 decisions, and evaluators report truncation rate and their effective
cap.

The selected 1-10 continuation is the first checkpoint in this sequence to
cross a world boundary without an update: it wins 8/10 fresh 2-1 night seeds.
Zero-shot 2-2 is 0/10, making the newly unlocked Sun-shroom stage the first
night acquisition frontier. A critic-free 20k-decision REINFORCE continuation
wins 9/20 disjoint 2-2 seeds (mean return 2.70935, no truncations), while the
early 20k GRPO control records no recent training wins. A 400k-step default PPO
continuation wins 13/20 on the exact same seed block (mean return 50.53155,
mean length 1710.95, no truncations) and also 13/20 on an independent block, so
it replaces REINFORCE as the selected 2-2 specialist. RND-PPO is 9/20 and the
high-entropy PPO branch is 4/20 on their independent blocks. On the same 20
selection seeds, deterministic sun pickup changes the selected PPO result from
13/20 to 12/20 (mean return 40.16025), so pickup automation is retained as an
explicit control simplification rather than claimed as a performance gain. A native 800×600
recording independently verifies the 2-1 terminal win at seed 166000.

Continuing from the earlier REINFORCE branch unlocks 2-3 in order. On the same
20 held-out seeds and under `collect_sun_first_v1`, default PPO wins 20/20
(mean return 109.3739) while the high-entropy plus RND branch wins 14/20.
The default checkpoint is frozen as the 2-3 specialist. Without any 2-4
updates, that same actor wins 16/20 fresh 2-4 seeds (mean return 86.03825,
mean length 1947.2, no truncations), so 2-4 is frozen as a genuine zero-shot
transfer result. A matched 400k-step default continuation is also 16/20 but
has lower exact mean environment return (84.4531); high-entropy/RND is 10/20
with mean return 16.33705. The zero-update actor therefore remains selected on
the unmodified environment-return objective. The sequentially unlocked 2-5
Whack-a-Zombie audit is 0/20
(mean return -96.2875, mean length 114.95): the actor had never trained the
board-click action family because it was illegal on earlier stages. PPO
adaptation uses authentic zombie-damage and terminal rewards, with no hand-
selected hammer targets. A 400k-step flat PPO continuation improves mean return
to -77.727 and more than doubles average nonterminal click reward, but remains
0/20 on disjoint held-out seeds. This is partial targeting skill, not a solved
stage. Argmax collapses to only three distinct board cells and scores -97.9075,
so stochastic sampling is necessary but not sufficient. A spatial PPO actor
with a shared per-cell board-click head is now the primary learned-control
branch. The matched high-entropy/RND continuation is also rejected: it wins
0/20 held-out seeds, returns -94.531 on average, and improves neither the flat
PPO return nor click accuracy (13.97% of board clicks receive positive reward,
versus 20.86% for flat PPO). A spatial REINFORCE run supplies a critic-free
comparison on the same mechanic. Vector/click alignment confirms that the
observation carries a strong learnable signal: 66.63% of clicks on a cell whose
vector encodes a zombie receive positive reward, versus 14.78% otherwise.
However, exact-cell occupancy explains only 37.44% of positive clicks. A
backward-compatible `spatial-conv` PPO branch therefore learns local 3x3 board
context before scoring cells. That neighborhood contains a zombie for 99.29%
of positive clicks; clicks with such context are positive 35.04% of the time,
versus 0.36% without it. The branch changes the inductive bias but adds no
scripted target selection or reward shaping.

The critic-free spatial REINFORCE branch is the first to clear the mechanic: it
wins 1/20 disjoint held-out episodes, with a win at seed 213016 (return 148.57,
657 decisions). Across all 20 it returns -59.0995 on average with mean length
496.9 and no truncations. This is a genuine learned 2-5 win and unlocks 2-6;
the checkpoint is a valid candidate but remains subject to exact-mean-return
selection against completed spatial PPO and the active convolutional PPO/GRPO
branches. A deterministic replay of the winning stochastic RNG trajectory is
being rendered at the engine's native logical 800x600 resolution.

Spatial PPO is the stronger completed 2-5 candidate: 3/20 held-out wins, exact
mean return -24.061, mean length 597.6, and no truncations. Its board-click
positive fraction is 25.92% and nonterminal click reward is 0.08083, both above
spatial REINFORCE (23.12%, 0.06729). It is the current selected checkpoint on
the unmodified mean-return objective, pending the already-running local-
convolution PPO and GRPO comparisons. The REINFORCE checkpoint nevertheless
generalizes zero-shot beyond the mechanic: it records authentic wins on 2-6,
2-7, 2-8, and 2-9 in strict order, so 2-10 is now the exposed frontier.
The complete 2-6 block is 15/20 wins with exact mean return 57.6033, mean
length 1021.4, and no truncations, despite receiving zero 2-6 updates.
The first attempt to reconstruct the earlier CUDA-sampled REINFORCE win from
its seed and RNG burn-in diverged and lost, so it is explicitly not accepted as
a win video. A fresh stop-on-win search now renders the current-best spatial
PPO online and will retain a native 800x600 clip only from a newly observed
terminal win.

The fresh video search succeeds after six episodes at seed 220005 (return
155.34, 650 decisions). The retained artifact is a genuine 164-frame 800x600
render of that winning episode; five losing candidate clips were discarded.
By contrast, an extended 200-episode search with the earlier flat PPO never
reproduces a win (mean return -76.98585), strengthening the conclusion that
shared spatial structure—not merely more stochastic trials—closed 2-5.

The complete intervening 2-9 audit is 12/20 wins with exact mean return 44.224,
mean length 1855.6, and no truncations. The same zero-update branch then wins
2-10 on its first audited seed (219000,
return 173.606), completing the night world and unlocking 3-1. The active 3-1
audit confirms the next real distribution shift: pool geometry and water lanes.
It is 0/20 with mean return -99.96075 and mean length 369.45. PPO adaptation
from the transfer-capable REINFORCE checkpoint is active; no 3-2 data is exposed
until an authentic 3-1 win occurs. Matched continuations compare PPO initialized
from the transfer checkpoint, PPO initialized from the current-best 2-5 PPO,
and critic-free REINFORCE; none receives a scripted pool or Lily Pad policy.
Group-relative policy optimization and a matched PPO+RND continuation add
exploration controls on the same locked 3-1 frontier.
The 200k-step critic-free REINFORCE continuation is the first completed pool
branch and is rejected: it collapses to 0/20 wins with a -100.0 recent mean,
then scores 0/20 and exactly -100.0 on its disjoint block (mean length 332.15).
An outcome-only antithetic evolution branch now warm-starts from the original
transfer policy and optimizes only authentic 3-1 episode return. It is rejected
at 0/10 and exactly -100.0 on its built-in disjoint holdout. Its perturbations
move behavioral descriptors, including zero-sun placement frequency, but do
not produce survival skill.
The matched plain PPO continuation also has 0/20 recent wins and mean return
-99.8835, then is rejected at 0/20 held out, mean return -99.9185, and mean
length 548.8.
PPO initialized from the stronger ordinary spatial-PPO checkpoint and the
group-relative branch also finish with 0/20 recent wins. Both are rejected on
disjoint blocks: PPO-init PPO scores 0/20, mean -99.746 and length 504.6; GRPO
scores 0/20, mean -99.993 and length 388.15.
A new initializer control separates pool adaptation from hammer specialization:
the flat plant-placement PPO that solved 2-3 and transferred 16/20 to 2-4 now
runs 400k-step 3-1 adaptation after also failing zero-shot (0/20, mean -99.977,
length 366.85). It finishes with 0/20 recent wins and is rejected on its disjoint
block at 0/20, mean -99.9785, and mean length 399.9. It receives no scripted
Lily Pad or water-lane decisions.
A non-myopic PPO+RND control uses discount 1.0, GAE lambda 1.0, and higher
entropy so delayed combat return can credit earlier pad-and-plant decisions.
The authentic reward and legal-action interface remain unchanged.
A critic-free plant-policy branch likewise uses undiscounted full-episode
returns and higher entropy, testing whether the value baseline itself obscures
the long pad-to-combat credit path. It also fails: 0/20 held out, mean -99.9845,
and mean length 353.0.
Two PSN-GRPO branches warm-start from that same plant-capable flat policy and
compare fixed per-rollout parameter-noise scales 0.004 and 0.01. The noisy
policy remains fixed for an entire episode, and truncated importance sampling
corrects its mismatch with the clean policy being optimized. This is a generic
long-horizon exploration test: it receives no plant-cost, packet, row, or pool
heuristic.
A matched pair uses the selected spatial-convolution 2-5 policy at the same
noise scales, separating coherent exploration from the initializer's spatial
inductive bias. This is motivated by held-out traces showing that failed flat
PPO does occasionally place Peashooters, including in water rows; the remaining
gap is not explained by one missing action family alone.
Uniform-legal 3-1 exploration completes an unscripted 50-episode replay shard
with 0/50 wins, mean return -100.0, and mean length 348.92. It receives no
action-family or placement preference beyond deterministic sun collection and
never reaches a legal 100-sun Peashooter action: random legal sampling spends
on cheaper packets first. The shard remains useful for offline and
learned-dynamics experiments, but uniform action exploration is rejected as a
pool solution.
The matched short-horizon PPO+RND branch finishes at 0/20 recent wins and mean
-99.994 despite nonzero intrinsic reward (mean 0.0923, max 1.444), then is
rejected at 0/20 held out, mean -99.9501 and length 327.25.
The undiscounted long-horizon PPO+RND continuation also fails to convert
novelty into survival: after 400k decisions and 1,085 episodes it has 0/20
recent wins, mean return -99.991, mean intrinsic reward 0.1259, and maximum
intrinsic reward 1.3304. It is rejected before held-out selection because it
never produced an authentic training win. The plant-capable flat PSN-GRPO
branches at parameter-noise scales 0.004 and 0.01 likewise finish 400k decisions
with 0/20 recent wins and means -100.0 and -99.978. Their matched
spatial-convolution branches remain active.

The next generic hard-exploration comparison follows Go-Explore's explicit
remember/return/explore decomposition. It stores exact engine action prefixes,
returns to them by deterministic replay, encodes the full normalized vector
with quantized random projections, and then mixes the selected learned policy
with uniform legal exploration. The archive uses no packet identity, plant
cost, row, water, Lily Pad, or other game-specific descriptor. A remote smoke
test expands five cells in 24 decisions with zero restore failures. Matched
5,000-iteration searches now compare policy-heavy epsilon 0.25 and
exploration-heavy epsilon 0.75; only archive-expanding or winning trajectories
are retained in their remote DuckDB shards. The first brief preflight shards
are retained but excluded after exposing a boundary-label issue: fixed-length
nonterminal branches now mark their last stored transition as wrapper-truncated.
The corrected searches use `v2` artifact names and checkpoint their archives
every 25 iterations.
The complete 2-10 block is 12/20 wins with mean return 70.6609, mean length
1312.7, and no truncations.

The spatial-convolution GRPO branch does not match PPO: it is 0/20 on held-out
2-5 seeds with mean return -71.189 and mean length 480.65. It is rejected by
both win count and the exact mean-return objective; local context alone does
not make the same-seed group-relative update competitive here.

The matched spatial-convolution PPO completes 400k steps with a much stronger
recent training block (10/20 wins, mean return 51.5945), then wins 7/20 on its
predeclared disjoint block with exact mean return 20.213, mean length 618.85,
and no truncations. It replaces ordinary spatial PPO (3/20, mean -24.061) under
the unchanged exact-mean-return objective. This closes the active 2-5 model
selection comparison.

A fresh current-best visualization search observes a terminal win on its second
episode, seed 230001 (return 151.57, 622 decisions). The retained artifact is a
157-frame native-logical 800x600 MP4; its replay contains the preceding loss
and the winning episode, so the clip is not a reconstructed trajectory.

Offline spatial-convolution IQL also finishes fitting the mixed authentic 2-5
replay (306 episodes, 122,011 transitions, five wins), but it is rejected on
the disjoint stochastic block: 0/20 wins, mean return -92.667, and mean length
167.5. This is substantially worse than online spatial PPO and confirms that
the sparse five-win replay does not support reliable advantage-weighted
behavior recovery at this setting.

The completed zero-shot 2-8 block is 13/20 wins (mean return 38.75495, mean
length 1080.05, no truncations), so its gate is also robust rather than a
single lucky episode.
The preceding 2-7 block is likewise complete at 9/20 wins (mean return
8.23795, mean length 1566.3, no truncations).

Matched authentic-pixel FSQ-token and RSSM baselines are now running on the
same whole-level 2-2 holdout as VQ, LeWM, and representation flow. They train
from scratch and retain the same eight-step evaluation horizon.

The first authentic-pixel LeWM run used the repository's historical generic
JEPA settings (VISReg λ=.05, 10 epochs, 3e-4) and collapsed on the entirely
held-out 2-2 stage: active representation fraction 0, mean feature standard
deviation 0.000101, and latent skill versus persistence −1774.25. It is retained
as a failed ablation. A hyperparameter-only correction using the released
VISReg recipe's λ=.9, 100 epochs, and 9e-4 learning rate also collapses: active
fraction 0, mean feature standard deviation 0.0000235, and latent skill versus
persistence −75260.97. Its checkpoint is
`artifacts/authentic-lewm-visreg-paper-recipe-holdout-level12-100ep.pt`
(SHA-256 `2dc295c070c94b263860ab6c657a5fd562dd87de830067f25135d771eac03276`).
This falsifies the idea that the mixture weight alone was the issue.

The third ablation aligns the released implementation's geometry and optimizer,
not only its scalar settings: VISReg receives `[views, batch, features]` with
each source/target time as a view, LeWM uses a BatchNorm projection MLP before
the predictive latent, and training uses five warmup epochs, cosine decay to
`lr/1000`, gradient clipping at 1, and weight decay 0.05. Old checkpoints retain
the projection-free architecture when loaded. On the exact same split this
eliminates representation collapse: active fraction is 1.0, mean feature
standard deviation is 0.495, and effective rank is 7.72. The predictor still
does not pass the dynamics bar: held-out latent MSE is 0.03760 versus 0.02395
for persistence, or −57.0% skill. It is retained as the first successful
authentic-pixel encoder rather than called a usable world model.

The active follow-up matches the released LeWM objective and architecture more
closely: unique `T+1` trajectory states, separate encoder and predictor
BatchNorm projectors, additive `prediction + 0.09 * VISReg`, three-frame
history, and a six-layer 192-D causal predictor. It keeps all features active
but collapses 95.9% of variance into one direction (effective rank 1.27) and is
584% worse than persistence, so it is also rejected. The next test freezes the
successful noncollapsed encoder and initializes an action-conditioned residual
predictor exactly at persistence; this separates representation drift from
predictor capacity. The encoder remains identical and fully active, but the
trained residual reaches latent MSE 0.04408 against the same 0.02395 persistence
baseline (-84.1% skill). This is worse than the jointly trained official-
geometry predictor, so freezing solved representation drift but not conditional
dynamics. It is rejected for planning. A VQ causal-token model trains on the
identical held-out-stage split as an orthogonal IRIS-family baseline.

The VQ causal-token baseline also fails the held-out dynamics bar. Only 4 of 64
codes are used (6.25%, perplexity 3.44), and its eight-step mean pixel MSE is
0.05572 versus 0.004353 for copying the last frame (-1180% skill). Its reward
MAE is 0.01130 and terminal Brier score is near zero, but those easy auxiliary
signals do not rescue visual dynamics. It is retained as a collapsed-codebook
ablation and is not used for imagination.
Exact hashes and acceptance states are in
`data/authentic_world_models_manifest.json`.

Corrected fresh evaluation produces real terminal wins: the pre-continuation
γ=.999 actor wins 1/30, the high-entropy 300k continuation wins 2/20, and the
selected RND-PPO continuation wins **4/20**. Return-ranked BC is sensitive to
selection pressure: the top-5% clone wins 1/20 while the top-20% clone is 0/20.
Discrete IQL trained on all 300 explicitly non-winning episodes nevertheless
wins a live seed after 13 attempts; the same episode is retained as a 226-frame
800×600 video. Continued PPO still forgets earlier gates, so stage routing
remains required rather than claiming one campaign-master policy.

Three 400k-step sequential continuations then expose 1-8 only after a real 1-7
win. Default PPO and RND-PPO each win 2/20 on fresh 1-8 seeds with no
time-limit truncations; the high-entropy variant is 0/20. The RND actor is
retained because its mean held-out return is slightly better (-77.56 versus
-77.96); a disjoint five-episode smoke batch is 0/5, for a pooled 2/25 result.
A live full-framebuffer sweep reaches another authentic 1-8 win after 11
attempts and retains 51 sampled 800×600 frames from that same rollout.

Level-1-8-only IQL trains on 83 episodes, 46,791 transitions, and five explicit
wins. Its stochastic policy is 0/20, but masked argmax is 1/10; behavior fit
alone therefore understates the importance of evaluation mode. Four sequential
1-9 continuations give a clearer online result: selected RND-PPO is 3/20,
high-entropy PPO is 1/20, and default PPO plus the IQL-initialized continuation
are both 0/20. None truncate. A live search records another 800×600 terminal win
on episode 16, seed 159015. Repeating the offline branch on the 101-episode
level-1-9-only shard does not help: IQL argmax and top-10% BC are each 0/10.

Feeding the new 1-4 and nine 1-6 successes back into the world model expands
training to 628 episodes and 278,577 transitions. The matched eight-A100 ensemble
improves eight-step state skill from +12.06% to **+13.87%** against persistence;
legality recall/precision are 92.34%/94.06% and reward MAE is 0.00853. CEM is
still 0/4 on both 1-6 and 1-7, selecting 43–45 distinct actions and waiting about
88%. Better prediction has again not yielded long-horizon strategic control.

The larger level-1-7 corpus also trains a five-replica ensemble to state MSE
0.000836 versus persistence 0.001077, or +22.38% skill, with reward MAE
0.00387 and legality recall/precision 91.47%/96.74%. This checkpoint is retained
as a noncanonical boundary-done ablation: it began before the target pipeline
was corrected and therefore treats collector time limits as done. The canonical
level-1-8 run uses terminal-only targets and single-process A100 execution so
it opens no distributed rendezvous socket.

On the corrected 1,244-episode level-1-8 corpus, the three-member ensemble is
the only tested vector model that beats its eight-step persistence baseline,
but only narrowly: state MSE 0.001119 versus 0.001142, or +1.97% skill. Residual
MLP is -2.90%, VISReg-JEPA latent skill is -13.63%, and RSSM is -972%. The
boundary-done ablation's +22.38% is therefore not presented as canonical; both
label semantics and the training split/model size changed, and the honest
terminal-corrected comparison is substantially harder.

Machine-readable control results are in
`data/authentic_rich_v2_control_eval.json`. Seventy-seven CUDA training/ablation rows
are consolidated in `data/authentic-rich-v2-experiments.duckdb` (SHA-256
`1d747d7771ccb1d26b2da3163ae66af612e7943c6333b4285bb1aa897e4c4c5e`).

## Authentic protocol-v3 semantic world models and control

The corrected PvZ-Portable protocol-v3 replay contains 50 episodes and 12,460
transitions, one trajectory for every Adventure stage from 1-1 through 5-10.
Observations are legacy vector-v1 250-value structured engine states and masks use the authentic
650-action vocabulary. Validation holds out all transitions from 10 entire
stages, rather than leaking a second episode of the same stage across folds.

| Model, 5 epochs | 8-step held-out result | Other diagnostic |
|---|---:|---:|
| Residual MLP | state MSE 0.01053 | reward MAE 0.03206 |
| 5-member ensemble | **state MSE 0.003525** | reward MAE **0.01137**; epistemic variance 0.000465 |
| Gaussian RSSM | state MSE 0.01426 | reward MAE 0.01797 |
| VISReg JEPA | latent cosine distance 0.03846 | effective rank 11.92; 100% active features |

Legality is sparse: only 0.44% of actions are truly legal on average over the
open-loop windows. The ensemble has 91.4% legality recall and 100% precision at
threshold 0.5; raw 99.7% accuracy is dominated by illegal negatives and should
not be used for selection.

Authentic 1-1 control with 20k real environment steps:

| Agent / evaluation | Held-out result |
|---|---:|
| Masked Double-DQN, greedy | **20/20 wins**, mean return 101.350 |
| PPO, masked argmax | 0/20 wins |
| PPO, stochastic sampling | 16/20 wins, mean return 81.282 |
| Conservative ensemble Dyna, greedy | 0/10 wins |
| Looser ensemble Dyna, greedy | **10/10 wins**, mean return 101.350 |

The Dyna comparison changes both model ratio and conservatism: the failed run
uses 25% imagined data, uncertainty penalty 5, and threshold 0.05; the winning
run uses 50%, penalty 1, and threshold 0.2. It demonstrates learned-model
control but is not an isolated hyperparameter attribution. Neither result yet
proves sequential transfer beyond 1-1.

The 5-10 sweep originally found a deterministic native crash at tick 1,000.
GDB located it in `Challenge::UpdateConveyorBelt`: skipping the intro had also
skipped Dr. Zomboss creation, so `GetBossZombie()` returned null. The bridge now
spawns Zomboss explicitly after removing preview zombies; corrected 5-10 traces
run beyond the former crash and are present in the protocol-v3 corpus.

## Authentic 50-stage visual pipeline smoke test

The PvZ-Portable bridge produced one eight-transition `3×120×160` RGB episode
for every Adventure stage from 1-1 through 5-10. The resulting private
DuckDB coverage shard contains 50 episodes and 400 transitions, all with
650-entry legal-action masks and nonconstant frames. These deliberately short
segments prove reset/step/render/replay coverage; they do not cover full waves
or demonstrate wins.

Three one-epoch CUDA jobs used 40 stage episodes for training and 10 held-out
stage episodes for evaluation at horizon four:

| Model | Primary held-out result | Diagnostic |
|---|---:|---:|
| VISReg JEPA, 64-D | latent MSE 0.0443 | effective rank 4.69; 100% active features |
| FSQ token WM, 3×5 grid | pixel MSE 0.1009 | utilization 4.69%; perplexity 2.02 |
| Representation flow, 3×5 grid | pixel MSE 0.1085 | legal accuracy 61.4%; precision 14.6% |

This run also found and fixed an action-vocabulary integration bug: the model
factory had assumed the clean-room environment's 181 actions instead of
inferring the authentic replay's 650-action mask. The checkpoints remain on
the private A100 host; experiment rows are mirrored in
`data/authentic_experiments.duckdb`. Longer, strategically diverse episodes
are required before comparing model quality.

## Vector dynamics, 3 epochs, horizon 16

| Model | Open-loop state MSE | Reward MAE | Done Brier | Note |
|---|---:|---:|---:|---|
| Residual MLP | 0.1039 | 0.0368 | 0.0046 | Error grows from 0.0027 at h1 to 0.3411 at h16 |
| 5-member bootstrap ensemble | **0.0217** | 0.0234 | **0.0028** | Mean epistemic state variance 0.00093 |
| Gaussian RSSM | 0.0385 | 0.0313 | 0.0897 | State rollout is stable; termination calibration is poor |
| VISReg JEPA | latent 0.0307 | 0.0148 | 0.0042 | Latent MSE is not comparable to state MSE |

## Vector JEPA regularizers, 3 epochs, horizon 16

| Regularizer | Cosine distance ↓ | Reward MAE ↓ | Done Brier ↓ | Effective rank ↑ | Top eigenvalue ↓ |
|---|---:|---:|---:|---:|---:|
| None | **0.0337** | 0.0239 | 0.0044 | **16.43** | 0.202 |
| VICReg | 0.0338 | 0.0212 | **0.0041** | **16.43** | 0.200 |
| VISReg | 0.0504 | **0.0148** | 0.0042 | 15.80 | **0.180** |
| SIGReg | 0.8489 | 0.0278 | 0.0810 | 11.23 | 0.296 |

The structured vector encoder does not collapse even without regularization at
this small data budget. VISReg gives the best reward head and most even leading
spectrum; VICReg gives the best normalized latent rollout among explicit
regularizers. Faithful Epps–Pulley SIGReg has a much larger, sample-scaled
initial statistic and converges slowly here: after 30 epochs its cosine distance
improves to 0.3093 and done Brier to 0.0050, but it still trails the alternatives.
This is evidence about this architecture/data regime, not a reason to alter the
published SIGReg statistic.

## Pixel JEPA, 10 epochs, horizon 8

| Encoder / regularizer | Cosine distance ↓ | Reward MAE ↓ | Effective rank ↑ | Active features ↑ | Top eigenvalue ↓ |
|---|---:|---:|---:|---:|---:|
| 1×1 global pool / strong VISReg | 0.1387 | 0.0105 | 1.37 | 0.69 | 0.93 |
| 3×5 spatial pool / strong VISReg | 0.1100 | 0.0286 | 3.46 | 1.00 | 0.50 |

The spatial encoder substantially improves collapse diagnostics and normalized
latent direction prediction, while its reward head is currently worse. Raw
latent MSE is scale-sensitive (0.0116 global versus 0.2089 spatial) and should
not be used alone to select a representation. More episodes and matched seeds
are required before selecting the architecture.

Checkpoints are `artifacts/port_mlp.pt`, `artifacts/port_rssm.pt`,
`artifacts/port_ensemble.pt`,
`artifacts/port_jepa_vector.pt`,
`artifacts/port_jepa_pixels_visreg_strong.pt`, and
`artifacts/port_jepa_pixels_visreg_spatial.pt`. Experiment metadata is recorded
in `data/experiments.duckdb`.

## Learned-model control smoke test

On Python-port demo 1 at seed 21, legality-aware categorical CEM using the
five-member ensemble completed the level with return 18.896. The heuristic also
won with return 24.326. CEM still selected wait for roughly 96% of decisions, so
this proves the end-to-end learned planning path works but not competitive
control. The planner replaces predicted-illegal future actions with wait and can
penalize ensemble uncertainty; the earlier first-action-only mask produced
out-of-distribution plans.

The offline Dyna Double-DQN pipeline has also been exercised end to end on a
deliberately tiny clean-room fixture: 24 real vector transitions trained a
two-member ensemble, and eight Q updates consumed 16 imagined transitions.
The resulting policy had 0% wins, as expected at that budget. This is a
checkpoint/CLI integration test, not a performance result; campaign claims
require a substantially broader port corpus and evaluation on held-out seeds.

### Stage 1-1 learned-control comparison, A100

The partial Python-port 1-1 adapter produced 13,080 vector transitions from 32
random-policy episodes. Random won all 32, so this is a pipeline/curriculum
check, not evidence of difficult control. A five-member ensemble trained on 24
episodes and evaluated on eight held-out episodes reached 8-step state MSE
0.000742, reward MAE 0.0211, done Brier 0.0000202, and mean epistemic variance
0.000109.

| Dyna setting | Imagined accepted | Rejected by uncertainty | Held-out port result |
|---|---:|---:|---:|
| 50% model, penalty 1, max uncertainty 0.05 | 1,471,257 | 1,088,743 | 0/20 wins, return −2.4 |
| 25% model, penalty 5, max uncertainty 0.02 | 381,678 | 258,322 | 5/5 wins, return 20.434 |

This is a concrete model-exploitation result: the looser learned policy failed
despite low average held-out dynamics error, while conservative model use
recovered a winning policy. The second result is preliminary because 1-1 is
overly forgiving and has only five evaluation seeds; later stages and random/
model-free matched controls are required.

## Representation-flow world model smoke test

A one-epoch MPS run validates the from-scratch MIRA-family implementation on
five training and one held-out Python-port pixel episodes. With a deliberately
small 2×3 latent grid and one Transformer layer, its 2-step pixel MSE was
0.07898. This is an integration result only; the next matched A100 comparison
must use the same replay, latent grid, parameter budget, seeds, and horizon as
VQ/FSQ and DIAMOND-style diffusion.

The subsequent 8×A100 run used the same broad 22-episode replay and 10-epoch
budget as the existing token experiments, with a 3×5 latent grid. It used a
four-layer, 128-wide flow Transformer and therefore is not parameter-matched to
the two-layer token models.

| Model | Pixel MSE ↓ | Reward MAE ↓ | Done Brier ↓ |
|---|---:|---:|---:|
| VQ causal tokens | 0.0412 | 0.0190 | **0.000428** |
| FSQ causal tokens | 0.0426 | 0.0302 | **0.000245** |
| Representation flow | **0.0281** | **0.0122** | 0.00139 |

Representation flow is the strongest visual/reward result so far on this
corpus, but weaker on termination calibration. A parameter-matched rerun,
stochastic multi-sample evaluation, action-recoverability probe, and learned
policy return are required before preferring it over discrete tokens.

## Token world model, 8×A100, 10 epochs

Both runs use the same five training episodes, one held-out episode, 3×5 token
grid, 64-token vocabulary, two-layer causal Transformer, and eight-step
open-loop evaluation. The historical runs used eight A100-SXM4 80 GB GPUs with
bfloat16 DDP. A later socket audit found that this Torch Elastic build could
bind its rendezvous store to a wildcard address despite a loopback command-line
argument. Current remote runs therefore use independent single-process GPU jobs
with no listener.

| Quantizer | Pixel MSE ↓ | Reward MAE ↓ | Done Brier ↓ | Held-out utilization ↑ | Perplexity ↑ |
|---|---:|---:|---:|---:|---:|
| learned VQ | **0.0485** | **0.00734** | 0.0163 | **0.0938** (6/64) | **3.59** |
| FSQ `[8, 8]` | 0.0527 | 0.0111 | **0.0149** | 0.0781 (5/64) | 1.34 |

FSQ removes a learned codebook and its commitment loss, but it does not force a
tiny, low-diversity encoder dataset to occupy every implicit code. On this data,
VQ has better visual rollout and broader token use. The comparison checkpoints
are `artifacts/a100_token_10ep.pt` and
`artifacts/a100_token_fsq_10ep.pt`; neither is a final large-data result.

### Broader five-scenario replay

A second matched comparison uses 22 episodes and 9,872 pixel transitions from
port demos 0–2 plus partial campaign stages 1-1 and 1-2. The exact remote-only
replay path, SHA-256, tensor shape, scenario counts, and source boundary are in
`data/remote_broad_manifest.json`.

| Quantizer | Pixel MSE ↓ | Reward MAE ↓ | Done Brier ↓ | Held-out utilization ↑ | Perplexity ↑ |
|---|---:|---:|---:|---:|---:|
| learned VQ | **0.0412** | **0.0190** | 0.000428 | **0.1719** (11/64) | **7.42** |
| FSQ `[8, 8]` | 0.0426 | 0.0302 | **0.000245** | 0.1094 (7/64) | 4.40 |

The broader corpus improves VQ pixel rollout and nearly doubles its effective
token use. Reward error is not directly comparable to the one-scenario run
because this split mixes five reward/timing regimes. Checkpoints are
`artifacts/a100_token_vq_broad_10ep.pt` and
`artifacts/a100_token_fsq_broad_10ep.pt`.
