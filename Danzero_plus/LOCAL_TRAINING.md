# Local parallel PPO training

Run from the FableDan repository root, using its uv environment. The official
DanZero+ game executable, frozen DMC weights, PPO architecture, reward/GAE and
clipped PPO objective are retained. Independent tables feed one GPU learner.

```bash
TABLES=8 MAX_UPDATES=1000 MAX_SECONDS=1800 \
  POOL_SIZE=2048 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 \
  bash Danzero_plus/train_local.sh Danzero_plus/runs/my-ppo
```

`TABLES` defaults to one. Benchmark several values rather than assuming more
GPUs help: game simulation and CPU candidate inference dominate collection.
The runner requires Linux, GCC and available ports 5000, 23456 through
23456+TABLES-1, and 16000 through 16000+4*TABLES-1. `DATA_PORT` overrides 5000;
`ACTION_PORT_BASE` overrides 16000 (avoiding SSH's common X11 port 6010).
It compiles a small `LD_PRELOAD` library in the run directory to remap only
the game executable's fixed TCP bind port 23456 (IPv4 and IPv6). It does not
modify the packed game binary, cards, legal moves, game timing or reward logic.
Each table uses four original seat actors and four websocket clients.
Packed-server extraction uses the run-local `tmp/` directory on the data disk,
not the space-constrained system `/tmp`; cleanup removes these generated files
after stopping the run's process groups. Server stderr is kept per table.

Logs are unbuffered. `learner/final.pth` is a portable CPU state dictionary,
`learner/training_state.pt` additionally records the optimizer and update count,
and `learner/summary.json` records runtime, received decisions/trajectories,
update throughput and actual optimizer steps. `actor_ckpt/` retains five recent
atomic weight exports. Both update and time limits stop training and save a
final model; a limit of zero disables that limit. The runner cleans up its own
process groups when done, without invoking the upstream machine-specific kill
scripts. Use a new run directory each time.

`INIT_WEIGHTS=/path/to/final.pth` optionally initializes model weights, but does
not restore optimizer/RNG state and is not an exact training continuation.
Fresh runs seed learner and seat actors; the opaque game executable's random
seed is not controlled. Short speed trials are consequently noisy.

Two correctness fixes are necessary for long runs: actors must load learner
weights before their first decision, and trajectory state-cut buffers must
actually be cleared after transmission. Neither changes the intended PPO
algorithm. The upstream replay-pool sampling and per-match weight-refresh
schedule remain in use, so policy lag and early KL stopping can still occur;
inspect actual optimizer steps rather than counting only learner updates.
Replay insertion and sampling share a process condition: otherwise concurrent
deque eviction can misalign observations, actions and stored log probabilities.
The first PPO batch checks actor/learner log-probability agreement before a run
is accepted; regression tests also stress concurrent receive/sample alignment.
The upstream learner defaults to a 2,048-entry pool. The launcher retains the
earlier pilot's 32,768 default for compatibility; explicitly set `POOL_SIZE=2048`
for the smaller, fresher pool used in this training comparison.

## Head-to-head comparison

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python -m fabledan.evaluate \
  --a ckpts/dmc-top2-selector/final.pt \
  --b danzero-ppo:Danzero_plus/runs/my-ppo/learner/final.pth \
  --filter-ckpt Danzero_plus/actor_torch/q_network.ckpt --candidate-k 2 \
  --games 1000 --seed 20261112 --workers 16 --log-every 100 \
  --output Danzero_plus/runs/my-ppo/head-to-head.json
```

Both teams receive the same frozen DMC Top-2 candidate filter. Seats alternate
each round; the game seed sequence is independent of worker scheduling. Output
contains individual rewards, win rate, mean reward and a Wilson 95% interval.
Evaluation uses FableDan's single-round engine, whose season/team-rank fields
are approximated. It is not an official multi-round DanZero+ benchmark.
Training budgets must be reported: a short PPO run against the existing
12-hour FableDan checkpoint is a pilot, not an equal-compute algorithm claim.

Automate two-minute speed trials, training and the comparison in a new run root:

```bash
uv run python Danzero_plus/finish_comparison.py Danzero_plus/runs/my-comparison \
  --seconds 1800 --updates 1000 --workers 16
```

This benchmarks 1, 32 and 64 tables on this 64-CPU host (override with
`--tables 1 4 8 16 32 64`), reuses completed trials within that root, selects the
highest measured actual optimizer-step throughput (not empty update calls), trains a fresh
PPO with a 2,048-entry pool, and runs exactly 1,000 head-to-head rounds.
`status.json` tracks training/evaluation/completion or failure, `comparison.json`
records results and checkpoint hashes, and `REPORT.md` provides a short report.
After a successful comparison, artifacts are also copied to
`ckpts/danzero-ppo-fast/` for convenient access.

Regression checks:

```bash
uv run python tests/test_all.py
uv run python tests/test_danzero_training.py
```
