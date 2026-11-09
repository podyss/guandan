"""Shared, dynamically batched inference for native DanZero evaluations."""

import argparse
import json
from pathlib import Path
import pickle
import time
import traceback

import numpy as np
import torch
import zmq

from .danzero_filter import DanZeroFilter
from .model_torch import load_ckpt


class InferenceClient:
    def __init__(self, endpoint, timeout_ms=120000):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.socket.connect(endpoint)

    def request(self, operation, **payload):
        try:
            self.socket.send_pyobj({'operation': operation, **payload})
            response = self.socket.recv_pyobj()
        except zmq.Again as error:
            raise TimeoutError('shared FableDan inference timed out') from error
        if 'error' in response:
            raise RuntimeError(f"shared FableDan inference failed: {response['error']}")
        return response['result']

    def close(self):
        self.socket.close()
        self.context.term()


class BatchedEvaluator:
    def __init__(self, checkpoint, dmc_checkpoint, device, precision='fp32'):
        self.device = torch.device(device)
        if precision == 'bf16' and self.device.type != 'cuda':
            raise ValueError('bf16 inference requires a CUDA device')
        self.use_amp = precision == 'bf16'
        self.model, checkpoint_data = load_ckpt(checkpoint, device=self.device)
        self.model.eval()
        self.candidate_k = int(checkpoint_data.get('meta', {}).get('candidate_k', 2))
        if self.candidate_k < 1:
            raise ValueError('checkpoint candidate_k must be positive')
        del checkpoint_data
        self.filter = DanZeroFilter.from_checkpoint(dmc_checkpoint).to(self.device)

    @torch.inference_mode()
    def candidates(self, requests):
        counts = [len(request['features']) for request in requests]
        features = np.concatenate([request['features'] for request in requests])
        scores = np.concatenate([
            self.filter.score(features[start:start + 4096]).cpu().numpy()
            for start in range(0, len(features), 4096)
        ])
        scores = np.split(scores, np.cumsum(counts)[:-1])
        return [np.argsort(values)[-min(self.candidate_k, len(values)):][::-1]
                for values in scores]

    @torch.inference_mode()
    def select(self, requests):
        lengths = [min(len(request['tokens']), self.model.cfg.max_seq)
                   for request in requests]
        tokens = np.zeros((len(requests), max(lengths)), dtype=np.int64)
        for row, (request, length) in enumerate(zip(requests, lengths)):
            tokens[row, :length] = request['tokens'][:length]
        counts = [len(request['features']) for request in requests]
        features = np.concatenate([request['features'] for request in requests])
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            context, _ = self.model.encode_seq(
                torch.as_tensor(tokens, device=self.device),
                torch.tensor(lengths, device=self.device))
            hand = self.model.hand_mlp(torch.as_tensor(
                features, dtype=torch.float32, device=self.device))
            context = context.repeat_interleave(
                torch.tensor(counts, device=self.device), dim=0)
            scores = self.model.q_head(torch.cat([context, hand], dim=-1))[:, 0]
        scores = scores.float().cpu().numpy()
        return [int(np.argmax(values)) for values in
                np.split(scores, np.cumsum(counts)[:-1])]


def serve(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    evaluator = BatchedEvaluator(args.checkpoint, args.dmc_checkpoint,
                                 args.device, args.precision)
    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)
    socket.setsockopt(zmq.LINGER, 0)
    try:
        socket.bind(args.endpoint)
        metadata = {'device': str(evaluator.device),
                    'candidate_k': evaluator.candidate_k,
                    'precision': args.precision,
                    'batch_size': args.batch_size}
        args.ready_file.write_text(json.dumps(metadata))
        print(f'[inference] ready: {json.dumps(metadata)}', flush=True)
        started = time.monotonic()
        decisions = 0
        batches = 0
        while True:
            if not socket.poll(500):
                continue
            messages = []
            deadline = time.monotonic() + args.batch_wait_ms / 1000
            while True:
                route = socket.recv_multipart()
                request = pickle.loads(route[-1])
                messages.append((route[:-1], request))
                if len(messages) >= args.batch_size:
                    break
                remaining = max(0, int((deadline - time.monotonic()) * 1000))
                if not socket.poll(remaining):
                    break
            for operation in ('metadata', 'candidates', 'select'):
                group = [(route, request) for route, request in messages
                         if request.get('operation') == operation]
                if not group:
                    continue
                try:
                    requests = [request for _, request in group]
                    if operation == 'metadata':
                        results = [metadata] * len(group)
                    else:
                        results = getattr(evaluator, operation)(requests)
                    responses = [{'result': result} for result in results]
                except Exception as error:
                    traceback.print_exc()
                    responses = [{'error': str(error)}] * len(group)
                for (route, _), response in zip(group, responses):
                    socket.send_multipart([
                        *route, pickle.dumps(response)])
                if operation == 'select':
                    decisions += len(group)
                    batches += 1
            for route, request in messages:
                if request.get('operation') not in ('metadata', 'candidates', 'select'):
                    socket.send_multipart([*route, pickle.dumps(
                        {'error': 'unknown inference operation'})])
            elapsed = time.monotonic() - started
            if elapsed >= 10:
                print(f'[inference] {decisions / elapsed:.1f} decisions/s, '
                      f'average selector batch {decisions / max(batches, 1):.1f}',
                      flush=True)
                decisions = batches = 0
                started = time.monotonic()
    finally:
        socket.close()
        context.term()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dmc-checkpoint', required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--precision', choices=('fp32', 'bf16'), default='fp32')
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--ready-file', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--batch-wait-ms', type=float, default=2)
    args = parser.parse_args()
    if args.batch_size < 1 or args.batch_wait_ms < 0:
        parser.error('batch size must be positive and batch wait must be nonnegative')
    serve(args)


if __name__ == '__main__':
    main()
