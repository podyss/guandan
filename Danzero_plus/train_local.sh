#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$ROOT/.." && pwd)"
RUN_DIR="${1:-$ROOT/runs/ppo-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$RUN_DIR/actors" "$RUN_DIR/actor_ckpt"
RUN_DIR="$(cd "$RUN_DIR" && pwd)"
mkdir -p "$RUN_DIR/tmp"
SERVER_TMPDIR="$RUN_DIR/tmp"
if [[ -e "$RUN_DIR/learner" ]]; then
    echo "Refusing to overwrite existing learner directory: $RUN_DIR/learner" >&2
    exit 1
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export LOG_FORMAT="${LOG_FORMAT:-stdout,csv}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
TABLES="${TABLES:-1}"
if ! [[ "$TABLES" =~ ^[1-9][0-9]*$ ]]; then
    echo "TABLES must be a positive integer" >&2
    exit 1
fi
chmod +x "$ROOT/actor_torch/danserver"
gcc -O2 -Wall -Wextra -shared -fPIC "$ROOT/actor_torch/server_port.c" -o "$RUN_DIR/server_port.so" -ldl
uv --project "$REPO" run python - "$TABLES" "${ACTION_PORT_BASE:-16000}" "${DATA_PORT:-5000}" <<'PY'
import socket
import sys

tables, action_base, data_port = map(int, sys.argv[1:])
ports = [data_port] + list(range(23456, 23456 + tables)) + list(range(action_base, action_base + 4 * tables))
if len(ports) != len(set(ports)):
    raise SystemExit('Training port ranges overlap')
for port in ports:
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(('127.0.0.1', port))
        except OSError as error:
            raise SystemExit(f'Training port {port} unavailable: {error}')
PY

processes=()
cleanup() {
    for pid in "${processes[@]:-}"; do
        kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
    done
    for pid in "${processes[@]:-}"; do
        wait "$pid" 2>/dev/null || true
    done
    rm -rf -- "$RUN_DIR/tmp"
}
trap cleanup EXIT INT TERM

learner_args=(
    --exp_path "$RUN_DIR/learner"
    --dmc_ckpt_dir "$RUN_DIR/actor_ckpt"
    --ckpt_save_freq "${CKPT_SAVE_FREQ:-100}"
    --batch_size "${BATCH_SIZE:-2048}"
    --pool_size "${POOL_SIZE:-32768}"
    --max_updates "${MAX_UPDATES:-0}"
    --max_seconds "${MAX_SECONDS:-0}"
    --seed "${SEED:-20261105}"
    --data_port "${DATA_PORT:-5000}"
)
if [[ -n "${INIT_WEIGHTS:-}" ]]; then
    learner_args+=(--init_weights "$INIT_WEIGHTS")
fi

setsid env PYTHONPATH="$ROOT/learner_torch" uv --project "$REPO" run python -u "$ROOT/learner_torch/learner.py" \
    "${learner_args[@]}" \
    >"$RUN_DIR/learner.log" 2>&1 &
processes+=("$!")
learner_pid="$!"
for attempt in $(seq 1 120); do
    [[ -f "$RUN_DIR/actor_ckpt/0.pth" ]] && break
    if ! kill -0 "$learner_pid" 2>/dev/null; then
        echo "Learner failed; inspect $RUN_DIR/learner.log" >&2
        exit 1
    fi
    sleep 1
done
[[ -f "$RUN_DIR/actor_ckpt/0.pth" ]] || exit 1

for ((table=0; table<TABLES; table++)); do
    mkdir -p "$RUN_DIR/actors/table$table"
    server_port=$((23456 + table))
    action_port=$((${ACTION_PORT_BASE:-16000} + 4 * table))
    setsid env TMPDIR="$SERVER_TMPDIR" LD_PRELOAD="$RUN_DIR/server_port.so" DANZERO_SERVER_PORT="$server_port" \
        "$ROOT/actor_torch/danserver" 100000 >"${SERVER_LOG:-/dev/null}" 2>"$RUN_DIR/server-$table.stderr.log" &
    processes+=("$!")
    setsid env PYTHONPATH="$ROOT/actor_torch" uv --project "$REPO" run python -u "$ROOT/actor_torch/actor.py" \
        --ip 127.0.0.1 --weights-dir "$RUN_DIR/actor_ckpt" \
        --dmc-ckpt "$ROOT/actor_torch/q_network.ckpt" \
        --data_port "${DATA_PORT:-5000}" --action_port "$action_port" \
        --seed "$((${SEED:-20261105} + table * 4))" \
        --exp_path "$RUN_DIR/actors/table$table" \
        >"$RUN_DIR/actor-$table.log" 2>&1 &
    processes+=("$!")
done
sleep 3
for ((table=0; table<TABLES; table++)); do
    setsid env PYTHONPATH="$ROOT/actor_torch" uv --project "$REPO" run python -u "$ROOT/actor_torch/game.py" \
        --server_port "$((23456 + table))" --action_port "$((${ACTION_PORT_BASE:-16000} + 4 * table))" \
        >"$RUN_DIR/game-$table.log" 2>&1 &
    processes+=("$!")
done

echo "DanZero+ PPO training started with $TABLES tables. Logs/checkpoints: $RUN_DIR"
while kill -0 "$learner_pid" 2>/dev/null; do
    for pid in "${processes[@]}"; do
        if [[ "$pid" != "$learner_pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
            echo "Training component $pid exited unexpectedly" >&2
            exit 1
        fi
    done
    sleep 2
done
wait "$learner_pid"
