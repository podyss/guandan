# -*- coding: utf-8 -*-
"""Dual-GPU DMC trainer: central batched GPU inference (DanLM-style).

Processes:
  - N actor processes (CPU): run `ring` concurrent rounds each via RingRunner,
    send batched decision requests to the inference server, push finished
    episodes to the learner.
  - 1 inference server process (GPU `--infer-device`): batches all pending
    decisions across actors into single forward passes; refreshes weights
    every cycle.
  - main process = learner (GPU `--device`): replay buffer + gradient steps,
    checkpointing, eval vs rule baseline & frozen snapshot, and automatic
    export/packaging of a ready-to-upload botzone zip whenever eval improves.

Typical run on a 2x5090 box:
    python -m fabledan.train_fast --out ckpts/fast1 --actors 24 ^
        --device cuda:1 --infer-device cuda:0
Resume:
    python -m fabledan.train_fast --out ckpts/fast1 --resume ckpts/fast1/latest.pt
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import math
import os
import queue as pyqueue
import random
import time
import zipfile

import numpy as np

try:
    import torch
    import torch.multiprocessing as mp
except ImportError as e:
    raise SystemExit("PyTorch required: pip install torch") from e

from .encode import FEAT_DIM
from .danzero_filter import DanZeroFilter
from .model_torch import FableDanNet, ModelConfig, export_npz, save_ckpt
from .train import Replay


# ---------------------------------------------------------------------------
# actor process
# ---------------------------------------------------------------------------

def actor_proc(actor_id, req_q, resp_q, sample_q, stop_ev, ring, eps, top_k, seed):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from .ring import RingRunner
    runner = RingRunner(ring=ring, seed=seed, eps=eps, top_k=top_k)
    while not stop_ev.is_set():
        reqs = runner.collect_requests()
        # one message per actor: (actor_id, [(slot, toks, feats), ...])
        req_q.put((actor_id, reqs))
        try:
            results = resp_q.get(timeout=60)
        except pyqueue.Empty:
            continue
        runner.step(dict(results))
        eps_out = runner.pop_episodes()
        if eps_out:
            sample_q.put(eps_out)


# ---------------------------------------------------------------------------
# inference server process
# ---------------------------------------------------------------------------

def infer_server(cfg_dict, req_q, resp_qs, weight_q, stop_ev, device,
                 max_decisions, stats_every, filter_ckpt=None,
                 candidate_k=2, infer_threads=16):
    torch.set_num_threads(infer_threads)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    use_amp = str(device).startswith("cuda")
    cfg = ModelConfig.from_dict(cfg_dict)
    model = FableDanNet(cfg).to(device).eval()
    filter_model = None
    if filter_ckpt:
        filter_model = DanZeroFilter.from_checkpoint(filter_ckpt)
    sd = weight_q.get()
    model.load_state_dict(sd)
    n_req = 0
    n_dec = 0
    n_fwd = 0
    t_last = time.time()
    cpu_last = time.process_time()
    while not stop_ev.is_set():
        # weight refresh (non-blocking)
        try:
            while True:
                sd = weight_q.get_nowait()
                model.load_state_dict(sd)
        except pyqueue.Empty:
            pass
        # gather a batch of actor messages
        msgs = []
        ndec = 0
        try:
            m = req_q.get(timeout=0.5)
            msgs.append(m)
            ndec += len(m[1])
            while ndec < max_decisions:
                m = req_q.get_nowait()
                msgs.append(m)
                ndec += len(m[1])
        except pyqueue.Empty:
            pass
        if not msgs:
            continue
        # build batch
        all_toks, all_feats, all_filter_feats, counts, route = [], [], [], [], []
        for actor_id, reqs in msgs:
            for slot, toks, feats, filter_feats in reqs:
                all_toks.append(toks)
                all_feats.append(feats)
                all_filter_feats.append(filter_feats)
                counts.append(feats.shape[0])
                route.append((actor_id, slot))
        B = len(all_toks)
        maxlen = max(len(t) for t in all_toks)
        T = np.zeros((B, maxlen), dtype=np.int64)
        L = np.zeros(B, dtype=np.int64)
        for j, t in enumerate(all_toks):
            T[j, :len(t)] = t
            L[j] = len(t)
        if filter_model is not None:
            flat_filter_feats = np.concatenate(all_filter_feats, axis=0)
            flat_filter_scores = filter_model.score(flat_filter_feats).numpy()
            filter_scores = np.split(flat_filter_scores,
                                     np.cumsum(counts)[:-1])
        else:
            filter_scores = None
        if filter_scores is not None:
            candidate_positions = [
                np.argsort(scores)[-min(candidate_k, len(scores)):]
                for scores in filter_scores]
        else:
            candidate_positions = [np.arange(count) for count in counts]
        candidate_counts = [len(indices) for indices in candidate_positions]
        Fcat = np.concatenate([
            feats[indices]
            for feats, indices in zip(all_feats, candidate_positions)], axis=0)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                             enabled=use_amp):
            tT = torch.from_numpy(T).to(device, non_blocking=True)
            tL = torch.from_numpy(L).to(device, non_blocking=True)
            tF = torch.from_numpy(Fcat).to(device, non_blocking=True)
            selector_ctx, _ = model.encode_seq(tT, tL)
            selector_q = model.q_head(torch.cat([
                selector_ctx.repeat_interleave(
                    torch.tensor(candidate_counts, device=device), dim=0),
                model.hand_mlp(tF)], dim=-1))[:, 0].float()
            selector_q = selector_q.cpu().numpy()
            result = []
            offset = 0
            for candidate_count, indices in zip(candidate_counts,
                                                candidate_positions):
                local_selector_q = selector_q[offset:offset + candidate_count]
                result.append((indices, local_selector_q))
                offset += candidate_count
        # split and route
        per_actor = {}
        for (actor_id, slot), decision_result in zip(route, result):
            per_actor.setdefault(actor_id, []).append((slot, decision_result))
        for actor_id, results in per_actor.items():
            resp_qs[actor_id].put(results)
        n_req += len(msgs)
        n_dec += B
        n_fwd += 1
        elapsed = time.time() - t_last
        if elapsed > stats_every:
            cpu_cores = (time.process_time() - cpu_last) / elapsed
            print("[infer] %.0f decisions/s, avg batch %.0f decisions "
                  "(%.1f actor msgs), CPU %.1f cores" % (
                      n_dec / elapsed, n_dec / max(n_fwd, 1),
                      n_req / max(n_fwd, 1), cpu_cores), flush=True)
            n_req = n_dec = n_fwd = 0
            t_last = time.time()
            cpu_last = time.process_time()


# ---------------------------------------------------------------------------
# packaging (auto botzone zip)
# ---------------------------------------------------------------------------

BOT_MODULES = ["__init__.py", "cards.py", "combos.py", "engine.py",
               "encode.py", "model_np.py", "agents.py", "train_demo.py",
               "evaluate.py"]


def pack_botzone_zip(weights_npz, out_zip):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bot_src = os.path.join(root, "botzone", "bot_fabledan.py")
    os.makedirs(os.path.dirname(out_zip), exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(bot_src, "__main__.py")
        for m in BOT_MODULES:
            z.write(os.path.join(root, "fabledan", m), "fabledan/" + m)
        z.write(weights_npz, "fabledan_weights.npz")
    return os.path.getsize(out_zip)


# ---------------------------------------------------------------------------
# learner / main
# ---------------------------------------------------------------------------

def run_eval(model, device, games, opponent="rule", opp_model=None, seed=123,
             filter_model=None, candidate_k=2):
    from .agents import RuleAgent, TorchAgent
    from .evaluate import evaluate
    make_b = (lambda: RuleAgent()) if opponent == "rule" else \
        (lambda: TorchAgent(opp_model, device=device))
    def make_a():
        agent = TorchAgent(model, device=device)
        if filter_model is not None:
            agent.with_danzero_filter(filter_model, candidate_k)
        return agent
    wr, _ = evaluate(make_a, make_b,
                     games=games, seed=seed)
    return wr


def run_parallel_eval(checkpoint, opponent, games, seed, workers,
                      filter_ckpt, candidate_k, output_path, cycle,
                      record_checkpoint):
    from .evaluate import evaluate_worker

    rng = random.Random(seed)
    seeds = [(index, rng.getrandbits(48)) for index in range(games)]
    chunk_size = max(1, math.ceil(games / (workers * 4)))
    chunks = [seeds[index:index + chunk_size]
              for index in range(0, games, chunk_size)]
    payloads = [(checkpoint, opponent, filter_ckpt, candidate_k, False, chunk)
                for chunk in chunks]
    rewards = []
    started = time.monotonic()
    last_reported = 0

    def report_progress(final=False):
        nonlocal last_reported
        completed = len(rewards)
        if not final and completed == games:
            return
        if not final and completed - last_reported < max(1, games // 10):
            return
        last_reported = completed
        wins = sum(reward > 0 for reward in rewards)
        win_rate = wins / completed if completed else 0.0
        denominator = 1 + 1.96 ** 2 / completed if completed else 1.0
        center = ((win_rate + 1.96 ** 2 / (2 * completed)) / denominator
                  if completed else 0.0)
        radius = (1.96 * math.sqrt(
            win_rate * (1 - win_rate) / completed +
            1.96 ** 2 / (4 * completed ** 2)) / denominator
            if completed else 0.0)
        event = {
            "event": "complete",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cycle": cycle,
            "opponent": "rule" if opponent == "rule" else "snapshot",
            "checkpoint": record_checkpoint,
            "seed": seed,
            "workers": min(workers, len(chunks)),
            "candidate_k": candidate_k,
            "filter_checkpoint": filter_ckpt,
            "games_completed": completed,
            "games_total": games,
            "wins": wins,
            "win_rate": win_rate,
            "win_rate_ci95": [center - radius, center + radius],
            "mean_reward": sum(rewards) / completed if completed else 0.0,
            "elapsed_s": time.monotonic() - started,
            "rewards": rewards,
        }
        print("[eval] cycle %d vs %s: %d/%d games, win %.1f%%, "
              "avg reward %.3f, %.1fs elapsed"
              % (cycle, event["opponent"], completed, games,
                 event["win_rate"] * 100, event["mean_reward"],
                 event["elapsed_s"]), flush=True)
        if final:
            with open(output_path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
                stream.flush()

    with ProcessPoolExecutor(max_workers=min(workers, len(chunks))) as executor:
        futures = [executor.submit(evaluate_worker, payload)
                   for payload in payloads]
        for future in as_completed(futures):
            rewards.extend(future.result())
            report_progress()

    report_progress(final=True)
    wins = sum(reward > 0 for reward in rewards)
    return {
        "wins": wins,
        "games": len(rewards),
        "win_rate": wins / len(rewards),
        "mean_reward": sum(rewards) / len(rewards),
        "elapsed_s": time.monotonic() - started,
    }


def backward_train_chunk(model, device, use_amp, tokens, lengths, features,
                         targets, beliefs, start, end, batch,
                         belief_weight):
    token_batch = torch.from_numpy(tokens[start:end]).to(device)
    length_batch = torch.from_numpy(lengths[start:end]).to(device)
    feature_batch = torch.from_numpy(features[start:end]).to(device).unsqueeze(1)
    target_batch = torch.from_numpy(targets[start:end]).to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
        q_values, hidden = model(token_batch, length_batch, feature_batch)
        q_loss = torch.nn.functional.mse_loss(
            q_values[:, 0].float(), target_batch)
        loss = q_loss
        if model.cfg.ntp_weight > 0:
            loss = loss + model.cfg.ntp_weight * model.ntp_loss(
                token_batch, hidden)
        if belief_weight > 0 and beliefs is not None:
            last_indices = (length_batch - 1).clamp(min=0)
            context = hidden[torch.arange(hidden.shape[0], device=device),
                             last_indices]
            belief_batch = torch.from_numpy(beliefs[start:end]).to(device)
            loss = loss + belief_weight * model.belief_loss(
                context.float(), belief_batch)
    ((end - start) / batch * loss).backward()
    return q_loss.detach() * ((end - start) / batch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ckpts/fast1")
    ap.add_argument("--resume", default="")
    ap.add_argument("--actors", type=int, default=min(8, os.cpu_count() or 8))
    ap.add_argument("--ring", type=int, default=16)
    ap.add_argument("--cycles", type=int, default=10**9)
    ap.add_argument("--buffer", type=int, default=262144)
    ap.add_argument("--diversity", type=int, default=2)
    ap.add_argument("--steps-per-cycle", type=int, default=16)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--micro-batch", type=int, default=512,
                    help="gradient-accumulation chunk size (memory bound)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eps", type=float, default=0.02)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--filter-ckpt", default="",
                    help="frozen DMC checkpoint; enables Top-K selector training")
    ap.add_argument("--candidate-k", type=int, default=2)
    ap.add_argument("--ntp-weight", type=float, default=0.02)
    ap.add_argument("--belief-weight", type=float, default=0.0,
                    help="legacy auxiliary loss (keep 0 for this experiment)")
    ap.add_argument("--n-blocks", type=int, default=4)
    ap.add_argument("--eval-cycles", type=int, default=100)
    ap.add_argument("--eval-games", type=int, default=100)
    ap.add_argument("--eval-workers", type=int,
                    default=min(16, os.cpu_count() or 1),
                    help="parallel worker processes for scheduled evaluations")
    ap.add_argument("--ckpt-cycles", type=int, default=5)
    ap.add_argument("--snapshot-cycles", type=int, default=500,
                    help="frozen self snapshot refresh for shadow eval")
    ap.add_argument("--device", default="cuda:1" if torch.cuda.device_count() > 1
                    else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--infer-device", default="cuda:0"
                    if torch.cuda.is_available() else "cpu")
    ap.add_argument("--infer-threads", type=int,
                    default=min(32, os.cpu_count() or 1),
                    help="CPU threads for the batched inference/filter worker")
    ap.add_argument("--infer-stats-every", type=float, default=10.0,
                    help="seconds between inference throughput/CPU reports")
    ap.add_argument("--max-decisions", type=int, default=1024,
                    help="max decisions per inference batch")
    ap.add_argument("--max-hours", type=float, default=0,
                    help="auto-stop after N hours: save ckpt, export npz, "
                         "pack upload zip (0 = run forever)")
    args = ap.parse_args()

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)

    if args.filter_ckpt and args.candidate_k < 1:
        ap.error("--candidate-k must be positive")
    if args.filter_ckpt and args.belief_weight != 0:
        ap.error("--belief-weight must be 0 for the DMC+NTP experiment")
    if args.eval_workers < 1:
        ap.error("--eval-workers must be positive")
    if args.infer_threads < 1 or args.infer_stats_every <= 0:
        ap.error("--infer-threads and --infer-stats-every must be positive")

    os.makedirs(args.out, exist_ok=True)
    device = args.device
    torch.backends.cuda.matmul.allow_tf32 = True
    use_amp = str(device).startswith("cuda")
    cfg = ModelConfig(n_blocks=args.n_blocks, ntp_weight=args.ntp_weight)
    model = FableDanNet(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_cycle, total_samples = 0, 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        if ck.get("optimizer"):
            opt.load_state_dict(ck["optimizer"])
        start_cycle = ck["meta"].get("cycle", 0)
        total_samples = ck["meta"].get("total_samples", 0)
        print("resumed cycle %d, %d samples" % (start_cycle, total_samples))

    mp.set_start_method("spawn", force=True)
    req_q = mp.Queue(maxsize=4096)
    sample_q = mp.Queue(maxsize=4096)
    weight_q = mp.Queue(maxsize=4)
    resp_qs = [mp.Queue(maxsize=8) for _ in range(args.actors)]
    stop_ev = mp.Event()

    server = mp.Process(target=infer_server,
                        args=(cfg.to_dict(), req_q, resp_qs, weight_q,
                        stop_ev, args.infer_device, args.max_decisions,
                              args.infer_stats_every, args.filter_ckpt or None,
                              args.candidate_k, args.infer_threads), daemon=True)
    server.start()

    actors = []
    for a in range(args.actors):
        p = mp.Process(target=actor_proc,
                       args=(a, req_q, resp_qs[a], sample_q, stop_ev,
                             args.ring, args.eps, args.top_k, 7000 + a),
                       daemon=True)
        p.start()
        actors.append(p)

    def push_weights():
        sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        try:
            weight_q.put_nowait(sd)
        except pyqueue.Full:
            pass

    weight_q.put({k: v.detach().cpu() for k, v in model.state_dict().items()})

    replay = Replay(args.buffer, belief_dim=45 if args.belief_weight > 0 else 0)
    rng = np.random.default_rng(0)
    cycle_budget = args.buffer // args.diversity
    best_wr = 0.0
    milestone_95 = False
    snapshot = None
    t_start = time.time()

    for cycle in range(start_cycle, args.cycles):
        got = 0
        t0 = time.time()
        while got < cycle_budget:
            ep = sample_q.get()
            for toks, feat, z, bel in ep:
                replay.add(toks, feat, z, bel)
            got += len(ep)
        total_samples += got
        t_collect = time.time() - t0

        t0 = time.time()
        model.train()
        losses = []
        micro_batch = max(1, min(args.micro_batch, args.batch))
        for _ in range(args.steps_per_cycle):
            T0, L0, F0, Z0, BEL0 = replay.sample(args.batch, rng)
            opt.zero_grad(set_to_none=True)
            chunk_count = (args.batch + micro_batch - 1) // micro_batch
            step_loss = 0.0
            for chunk_index in range(chunk_count):
                start = chunk_index * micro_batch
                end = min(start + micro_batch, args.batch)
                step_loss += backward_train_chunk(
                    model, device, use_amp, T0, L0, F0, Z0, BEL0,
                    start, end, args.batch, args.belief_weight)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(step_loss.item())
        model.eval()
        t_train = time.time() - t0
        push_weights()

        sps = total_samples / (time.time() - t_start + 1e-9)
        print("cycle %d  samples %s  loss %.4f  collect %.1fs train %.1fs  "
              "%.0f samples/s" % (cycle, f"{total_samples:,}",
                                  float(np.mean(losses)), t_collect, t_train,
                                  sps), flush=True)

        meta = {"cycle": cycle + 1, "total_samples": total_samples,
                "seed": 7000, "candidate_k": args.candidate_k,
                "filter_checkpoint": args.filter_ckpt,
                "cycles_budget": args.cycles, "epsilon": args.eps}
        if (cycle + 1) % args.ckpt_cycles == 0:
            save_ckpt(model, opt, meta, os.path.join(args.out, "latest.pt"))
        if args.snapshot_cycles and (cycle + 1) % args.snapshot_cycles == 0:
            snapshot = FableDanNet(cfg).to(device)
            snapshot.load_state_dict(model.state_dict())
            snapshot.eval()
        if (cycle + 1) % args.eval_cycles == 0:
            eval_checkpoint = os.path.join(
                args.out, ".eval-cycle-%d.pt" % (cycle + 1))
            snapshot_checkpoint = os.path.join(
                args.out, ".eval-snapshot-cycle-%d.pt" % (cycle + 1))
            eval_meta = dict(meta)
            eval_meta["evaluation_cycle"] = cycle + 1
            save_ckpt(model, opt, meta, os.path.join(args.out, "latest.pt"))
            checkpoint_dir = os.path.join(args.out, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            saved_checkpoint = os.path.join(
                checkpoint_dir, "%d.pt" % (cycle + 1))
            save_ckpt(model, opt, eval_meta, saved_checkpoint + ".tmp")
            os.replace(saved_checkpoint + ".tmp", saved_checkpoint)
            print("[checkpoint] saved %s before evaluation" % saved_checkpoint,
                  flush=True)
            save_ckpt(model, None, eval_meta, eval_checkpoint)
            try:
                rule_result = run_parallel_eval(
                    eval_checkpoint, "rule", args.eval_games, 123 + cycle,
                    args.eval_workers, args.filter_ckpt, args.candidate_k,
                    os.path.join(args.out, "evaluations.jsonl"), cycle + 1,
                    saved_checkpoint)
                wr = rule_result["win_rate"]
                print("  eval vs rule: %.1f%% (%d/%d), avg reward %.3f; "
                      "%.1fs" % (wr * 100, rule_result["wins"],
                                 rule_result["games"],
                                 rule_result["mean_reward"],
                                 rule_result["elapsed_s"]), flush=True)
                if snapshot is not None:
                    save_ckpt(snapshot, None, eval_meta, snapshot_checkpoint)
                    snapshot_result = run_parallel_eval(
                        eval_checkpoint, snapshot_checkpoint,
                        args.eval_games, 123 + cycle,
                        args.eval_workers, args.filter_ckpt,
                        args.candidate_k,
                        os.path.join(args.out, "evaluations.jsonl"),
                        cycle + 1, saved_checkpoint)
                    print("  eval vs snapshot(-%d cyc): %.1f%% (%d/%d), "
                          "avg reward %.3f; %.1fs"
                          % (args.snapshot_cycles,
                             snapshot_result["win_rate"] * 100,
                             snapshot_result["wins"],
                             snapshot_result["games"],
                             snapshot_result["mean_reward"],
                             snapshot_result["elapsed_s"]), flush=True)
            finally:
                for path in (eval_checkpoint, snapshot_checkpoint):
                    if os.path.exists(path):
                        os.remove(path)
            if wr >= best_wr:
                best_wr = wr
                save_ckpt(model, opt, meta, os.path.join(args.out, "best.pt"))
                if not args.filter_ckpt:
                    npz = os.path.join(args.out, "best.npz")
                    export_npz(model, npz)
                    zip_path = os.path.join(
                        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "dist", "fabledan_bot_ready.zip")
                    size = pack_botzone_zip(npz, zip_path)
                    print("  [打包] %s (%.1f MB) — 可直接上传 botzone (python3)"
                          % (zip_path, size / 1e6), flush=True)
                else:
                    print("  [保存] best.pt 依赖独立的 DanZero DMC 筛选器；"
                          "暂不打包为独立 Botzone bot。", flush=True)
            if wr >= 0.95 and not milestone_95:
                milestone_95 = True
                print("\n" + "=" * 64 +
                      "\n  里程碑: vs rule >= 95%%! 现在就值得上传第一版:\n"
                      "  dist/fabledan_bot_ready.zip -> botzone GuanDan, "
                      "编译器选 python3\n" + "=" * 64 + "\n", flush=True)

        if args.max_hours > 0 and time.time() - t_start >= args.max_hours * 3600:
            print("\n[到时收尾] 已运行 %.1f 小时，做最终评估与打包..."
                  % ((time.time() - t_start) / 3600), flush=True)
            save_ckpt(model, opt, meta, os.path.join(args.out, "latest.pt"))
            final_filter = (DanZeroFilter.from_checkpoint(args.filter_ckpt)
                            if args.filter_ckpt else None)
            wr = run_eval(model, device, max(args.eval_games, 100), "rule",
                          filter_model=final_filter,
                          candidate_k=args.candidate_k)
            save_ckpt(model, opt, meta, os.path.join(args.out, "final.pt"))
            if args.filter_ckpt:
                print("[完成] vs rule %.1f%% | final.pt 仍需外部 DMC 筛选器。"
                      "\n下次续训: uv run python -m fabledan.train_fast --out %s "
                      "--filter-ckpt %s --resume %s"
                      % (wr * 100, args.out, args.filter_ckpt,
                         os.path.join(args.out, "latest.pt")), flush=True)
            else:
                npz = os.path.join(args.out, "final.npz")
                export_npz(model, npz)
                zip_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "dist", "fabledan_bot_ready.zip")
                size = pack_botzone_zip(npz, zip_path)
                print("[完成] vs rule %.1f%% | 上传文件: %s (%.1f MB)\n"
                      "下次续训: uv run python -m fabledan.train_fast --out %s "
                      "--resume %s" % (wr * 100, zip_path, size / 1e6, args.out,
                                       os.path.join(args.out, "latest.pt")),
                      flush=True)
            break

    stop_ev.set()
    for p in actors + [server]:
        p.terminate()


if __name__ == "__main__":
    main()
