# DanZero+ DMC + FableDan Selector: Preliminary Results

Date: 2026-11-04 (UTC+8)

## Summary

The feature-indexing bug in the initial adapter was corrected: DanZero+
card vectors are rank-major, not suit-major. The old low DMC scores below are
**invalid and superseded**. With the corrected adapter, frozen DMC Top-1 won
74.0% against the local rule agent over 5,000 rounds (seed 20261109), with
average reward +1.282. This is strong enough to proceed to a Top-2 FableDan
selector training experiment, but does not yet validate that learned selector.

## Runs

All runs use the local single-round FableDan engine, seat-swapped teams, and
the DanZero+ checkpoint `Danzero_plus/actor_torch/q_network.ckpt`.

| Setup | Rounds | Seed | Win rate / reward | Status |
| --- | ---: | ---: | ---: | --- |
| Rule vs rule | 1,000 | 20261104 | 50.1% / +0.053 | Complete sanity check |
| DMC Top-1 vs rule, old adapter | 5,000 | 20261107 | 30.3% / -1.141 | Invalid: feature index bug |
| DMC Top-1 vs rule, old adapter | 1,000 | 20261104 | 31.0% / -1.135 | Invalid: feature index bug |
| Random choice within DMC Top-2, old adapter | 1,000 | 20261105 | 20.9% / -1.688 | Invalid: feature index bug |
| DMC Top-1 vs rule, corrected adapter | 1,000 | 20261108 | 71.3% / +1.166 | Complete |
| DMC Top-1 vs rule, corrected adapter | 5,000 | 20261109 | 74.0% / +1.282 | Complete |
| FableDan Transformer over DMC Top-2 vs rule (`final.pt`) | 1,000 | 20261110 | 84.4% / +1.891 | Complete held-out evaluation |
| FableDan Top-2 selector vs DanZero+ DMC Top-1 | 1,000 | 20261111 | 65.0% / +0.862 | Complete head-to-head, seat-swapped |
| Existing smoke Transformer checkpoint vs rule | 300 observed | 20261106 | 11.3% / -2.113 | Interrupted; not a valid final evaluation |

The corrected adapter maps FableDan card IDs into DanZero+'s rank-major
encoding and preserves the finished-teammate sentinel. The initial
suit-major mapping made prior standalone and random-Top-2 results unusable.

## Important caveats

- FableDan's one-round engine does not provide DanZero+'s season/team-rank
  context. The adapter approximates those fields, so this is not a faithful
  reproduction of DanZero+'s original runtime input; validate any learned
  selector against the same adapter and report this limitation.
- The Transformer checkpoint came from a tiny smoke run (about 68 collected
  samples), and its 300-round evaluation was stopped early. It cannot establish
  whether DMC+NTP learning works or fails; it is superseded by the full run
  below.
- The Top-2 selector trained for 12 hours on two RTX 3090s, reaching cycle
  1,890 and 31,018,859 collected decisions. Its final checkpoint achieved
  84.4% over a separate 1,000-round evaluation (seed 20261110). This is
  promising evidence that the FableDan selector can improve on the frozen
  DMC Top-1 baseline, but the result is not a direct PPO comparison and has
  sampling uncertainty; the match-context approximation still applies.
- In a direct 1,000-round head-to-head (seed 20261111), the FableDan Top-2
  selector beat DanZero+ DMC Top-1 65.0% of rounds, with +0.862 average
  reward, using seat swaps. No PPO checkpoint was available at that stage;
  the newly trained PPO pilot comparison is reported below.
- `uv run python tests/test_all.py` passes, including filter-adapter,
  frozen-weight, candidate-index, and game-engine checks. Passing software
  tests alone do not validate strategic quality; the corrected multi-thousand
  round baseline is the first such quality check.

## Reproduction

```bash
uv run python -m fabledan.evaluate --a dmc \
  --filter-ckpt Danzero_plus/actor_torch/q_network.ckpt \
  --b rule --games 5000 --seed 20261109 --log-every 500

uv run python -m fabledan.evaluate --a rule --b rule \
  --games 1000 --seed 20261104 --log-every 250
```

## Reproducing the selector evaluation

```bash
uv run python -m fabledan.evaluate --a ckpts/dmc-top2-selector/final.pt \
  --filter-ckpt Danzero_plus/actor_torch/q_network.ckpt --candidate-k 2 \
  --b rule --games 1000 --seed 20261110 --log-every 200
```

This tests the PPO replacement hypothesis against the local Rule baseline,
not parity with PPO. The next useful checks are an independent-seed 1,000-round
evaluation and a controlled PPO comparison if a runnable PPO baseline is
available. Keep the feature-context approximation caveat in conclusions.

## DanZero+ PPO throughput trials

Initial short trials on 2026-11-05 did not demonstrate that moving each seat's
small DMC network to a separate GPU improved throughput. Those runs used
different warm-start weights, inconsistent timing windows and no cumulative
decision counters; their previously quoted update rates are not a controlled
speed comparison. GPU placement was reverted, the artificial 0.1-second sleep
per optimizer step was removed, and one intra-op thread per actor remains the
default to avoid CPU oversubscription.

The subsequent parallel-table runner retains the official game executable and
PPO objective, isolates each table's network ports, fixes first-decision weight
loading and trajectory-buffer resets, synchronizes replay insertion and sampling,
and records cumulative received decisions and actual optimizer steps. The first
parallel trial had a receive/sample race and is marked invalidated; it is not
used for quality conclusions. See `Danzero_plus/LOCAL_TRAINING.md`. The active
comparison root is recorded in `Danzero_plus/runs/fast-compare-run.txt`; its
`status.json` and eventual `REPORT.md` distinguish training from completed
1,000-round evaluation. This is a short-budget PPO pilot against the existing
12-hour FableDan checkpoint, not an equal-compute or convergence claim.

## Completed parallel PPO pilot (2026-11-05)

The corrected runner completed a fresh PPO training run and a 1,000-round
head-to-head. Earlier trials with the replay synchronization regression are
explicitly invalidated and are not the source of the checkpoint below.

### Controlled short throughput trials

Each trial used 120 seconds of learner runtime, the same random initial-model
seed, a 2,048 batch, a 32,768-entry replay pool, one GPU learner and one CPU
intra-op thread per seat actor. The opaque game executable cannot be seeded,
so these are noisy short trials rather than statistically replicated timings.

| Parallel tables | Received decisions/s | Learner updates/min | Actual optimizer steps in trial |
| ---: | ---: | ---: | ---: |
| 1 | 33.5 | 0.50 | 2 |
| 32 | 1,218.5 | 37.48 | 127 |
| 64 | 2,167.7 | 67.45 | 148 |

64 tables had both the highest collection throughput (64.63x single-table)
and the highest measured optimizer-step throughput among these configurations.
This does not mean every learner update applied a gradient: KL early stopping
frequently skips updates, which is why actual optimizer steps are counted.

### Training and head-to-head

- Configuration: 64 original DanZero+ tables, CPU candidate inference, one RTX
  3090 learner, fresh model initialization with seed 20261105.
- Training: 776.15 seconds (12.94 minutes), 2,098,175 received decisions,
  13,016 received seat trajectories, 1,000 learner updates and 3,250 actual
  optimizer steps. Collection averaged 2,703.3 decisions/s during training.
- The training replay pool was reduced to the upstream 2,048-entry default
  to limit policy lag; therefore full-run throughput is not a same-config
  speed comparison with the larger-pool trials.
- Evaluation: seed 20261112, exactly 1,000 rounds, alternating team seats,
  both selectors using the same frozen DMC Top-2 candidate filter.
- FableDan won **673/1,000 (67.3%)**; PPO won **327/1,000 (32.7%)**.
  FableDan mean reward was **+0.921**, with a Wilson 95% win-rate interval
  of **[64.3%, 70.1%]**.
- Both regression suites pass, including initial actor weight loading,
  trajectory-buffer clearing, IPv4/IPv6 port isolation, tensor PPO updates,
  concurrent replay-field alignment, evaluation seeds and shared model loading.
- Training, game-server and evaluation processes exited; the run-local packed
  server extraction directory was removed.

Artifacts:

- PPO weights: `ckpts/danzero-ppo-fast/final.pth`
- Model/optimizer snapshot: `ckpts/danzero-ppo-fast/training_state.pt`
- Training counters: `ckpts/danzero-ppo-fast/summary.json`
- Individual evaluation rewards: `ckpts/danzero-ppo-fast/head-to-head.json`
- Report: `ckpts/danzero-ppo-fast/REPORT.md`
- Full run, speed trials, hashes and logs:
  `Danzero_plus/runs/fast-valid-20261105-144030/`

This establishes that the combined FableDan selector can beat this particular
short-trained PPO checkpoint in the local evaluation. It does **not** establish
architectural superiority or reproduce the paper's trained DanZero+ strength:
FableDan had 12 hours / 31,018,859 collected decisions, versus about 13 minutes /
2.1 million for this PPO pilot. PPO also trains in the original multi-round
game executable, whereas FableDan trained and is evaluated in the local
single-round engine with approximated season/team context. Matching compute,
experience, environment and reward horizons is required for a method-level claim.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python -m fabledan.evaluate \
  --a ckpts/dmc-top2-selector/final.pt \
  --b danzero-ppo:ckpts/danzero-ppo-fast/final.pth \
  --filter-ckpt Danzero_plus/actor_torch/q_network.ckpt --candidate-k 2 \
  --games 1000 --seed 20261112 --workers 16 --log-every 100
```

## Data-budget and paper-reproduction audit (2026-11-05)

The local checkpoints and counters were inspected directly; this is an audit
of the completed runs, not a new training run.

| Quantity | DMC + FableDan selector | DMC + PPO pilot |
| --- | ---: | ---: |
| Collected / received decision records | 31,018,859 | 2,098,175 |
| Training wall time | About 12 hours | 776.15 seconds |
| Actual optimizer steps | 15,120 | 3,250 |
| GPUs used | 2 RTX 3090s (inference + learning) | 1 RTX 3090 learner |
| Training environment | Local single-round engine | Original multi-round executable |

FableDan's count and optimizer step are stored in
`ckpts/dmc-top2-selector/final.pt` (`meta.total_samples` and Adam state).
PPO's counters are in `ckpts/danzero-ppo-fast/summary.json`.
FableDan collected **14.7837 times** as many decision records; PPO received
about **6.76%** of that budget. These are experience counts, not game counts,
token counts, or the number of distinct records actually used in gradients.
Replay sampling can reuse records and eviction can discard records before use.
For example, PPO processed 3,250 * 2,048 = 6,656,000 batch rows in actual
optimizer steps, which must not be described as newly collected experience.
Optimizer steps also do not imply equal work across different architectures.
Both selectors reuse the same pre-trained DMC checkpoint; its original
training-data budget is not included in either selector's count and is unknown
from these local artifacts.

### Additional PPO-versus-DMC evaluation

To check the opponent pairing used in the paper, the completed PPO checkpoint
was evaluated against frozen DMC Top-1, using the existing local single-round
engine. This matches the type of opponent, **not** the paper's full evaluation
protocol. With 1,000 rounds, seed 20261113, alternating seats and 16 workers:

- PPO won 520/1,000 rounds: **52.0%**.
- Wilson 95% interval: **[48.9%, 55.1%]**; it includes 50%.
- PPO mean reward: **+0.114**.
- Full results: `ckpts/danzero-ppo-fast/ppo-vs-dmc.json`.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python -m fabledan.evaluate \
  --a danzero-ppo:ckpts/danzero-ppo-fast/final.pth --b dmc \
  --filter-ckpt Danzero_plus/actor_torch/q_network.ckpt --candidate-k 2 \
  --games 1000 --seed 20261113 --workers 16 --log-every 100 \
  --output ckpts/danzero-ppo-fast/ppo-vs-dmc.json
```

### What the paper actually reports

Primary source: *DanZero+: Dominating the GuanDan Game through Reinforcement
Learning*, arXiv:2312.02561v1, Sections IV-B and V-A/V-D, Tables III and V.

- Table III reports PPO with two candidates winning **55.13% against DMC**,
  and **92.70%, 86.22%, 90.83%** against competition baselines 1, 2, 4.
  Those competition bots are not the local FableDan rule agent.
- Its 1,000-game evaluation concerns multi-round GuanDan matches; our 1,000
  rounds omit full-match leveling, tribute and exact season/team context.
- The paper gives time-based learning curves and says satisfactory PPO
  performance is achieved in less than a day. It does not provide a cumulative
  PPO decision counter that can establish exact sample-budget equivalence.
- Table V lists Top-2, pool/batch 2,048, training frequency 13, Adam at 1e-4,
  gamma 0.99 and GAE lambda 0.95. These core settings agree with the pilot.
  However, it lists 40 actors and MLP sizes `512,512,512,256`, while the pilot
  uses 64 tables and the repository's default shared MLP
  `512,512,512,512,256`. Reusing released code is therefore not by itself an
  exact match to the paper's stated architecture and configuration.

Conclusion: the implementation is a runnable, short-trained DMC+PPO baseline,
**not a demonstrated reproduction of the paper's result**. Its local 52.0%
versus DMC is numerically below the paper's 55.13%, but different environments
and sampling uncertainty prevent attributing the gap solely to training or
declaring a statistically comparable reproduction failure. Likewise,
FableDan's existing 67.3% versus this PPO cannot establish equal-budget method
superiority. A selector comparison must align experience budgets, training
environments, reward horizons, candidate filtering and evaluation seeds; a
paper reproduction additionally needs the original full-match benchmark and
competition opponents.
