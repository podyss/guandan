uv run python -m fabledan.train_fast \
  --out ckpts/fabledan-top10 \
  --filter-ckpt Danzero_plus/actor_torch/q_network.ckpt \
  --candidate-k 10 \
  --actors 8 \
  --infer-threads 32 \
  --ring 16 \
  --buffer 262144 \
  --diversity 2 \
  --steps-per-cycle 16 \
  --batch 4096 \
  --micro-batch 512 \
  --ntp-weight 0.02 \
  --belief-weight 0 \
  --eval-cycles 100 \
  --eval-games 100 \
  --eval-workers 16 \
  --ckpt-cycles 5 \
  --device cuda:1 \
  --infer-device cuda:0




uv run python -u Danzero_plus/evaluate_paper_baselines.py \
  --baseline 2 \
  --policy fabledan \
  --checkpoint ckpts/fabledan-top10/latest.pt \
  --games 1000 \
  --parallel 48 \
  --timeout 7200 \
  --stall-timeout 180 \
  --output ckpts/fabledan-top10/evaluations/latest-vs-baseline2-1000-matches.json

