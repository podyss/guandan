import concurrent.futures
import json
import os
from pathlib import Path
import pickle
import runpy
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'Danzero_plus/actor_torch'))

from fabledan.danzero_filter import DanZeroFilter
from fabledan.eval_inference import BatchedEvaluator, InferenceClient
from fabledan.model_torch import FableDanNet, ModelConfig
from fabledan_policy import FableDanPolicy


class EvaluationInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(19)
        cls.directory = tempfile.TemporaryDirectory(prefix='eval-inference-test-')
        cls.root = Path(cls.directory.name)
        cls.checkpoint = cls.root / 'selector.pt'
        cls.dmc_checkpoint = cls.root / 'dmc.ckpt'
        config = ModelConfig(d_model=16, n_blocks=1, n_heads=2, qk_dim=8,
                             v_dim=8, ffn_hidden=32, hand_hidden=32,
                             n_hand_layers=1, q_hidden=32, n_q_layers=1,
                             max_seq=64)
        model = FableDanNet(config)
        torch.save({'model': model.state_dict(), 'config': config.to_dict(),
                    'meta': {'candidate_k': 3}}, cls.checkpoint)
        weights = []
        for name, weight in DanZeroFilter().q.state_dict().items():
            array = weight.numpy()
            weights.append(array.T if name.endswith('weight') else array)
        with cls.dmc_checkpoint.open('wb') as checkpoint:
            pickle.dump(weights, checkpoint)
        cls.evaluator = BatchedEvaluator(cls.checkpoint, cls.dmc_checkpoint, 'cpu')
        generator = np.random.default_rng(31)
        cls.candidate_requests = [
            {'features': generator.normal(size=(count, 567)).astype(np.float32)}
            for count in (1, 7, 4)]
        cls.select_requests = [
            {'tokens': generator.integers(1, 48, size=length).tolist(),
             'features': generator.normal(size=(count, 80)).astype(np.float32)}
            for length, count in ((2, 1), (29, 4), (63, 2))]

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_batched_filter_preserves_candidate_order(self):
        results = self.evaluator.candidates(self.candidate_requests)
        for request, indices in zip(self.candidate_requests, results):
            scores = self.evaluator.filter.score(request['features']).numpy()
            expected = np.argsort(scores)[-min(3, len(scores)):][::-1]
            np.testing.assert_array_equal(indices, expected)

    def test_dynamic_padding_and_flat_actions_match_individual_inference(self):
        results = self.evaluator.select(self.select_requests)
        for request, selected in zip(self.select_requests, results):
            tokens = torch.zeros((1, 64), dtype=torch.long)
            length = len(request['tokens'])
            tokens[0, :length] = torch.tensor(request['tokens'])
            with torch.inference_mode():
                scores, _ = self.evaluator.model(
                    tokens, torch.tensor([length]),
                    torch.from_numpy(request['features'][None]))
            self.assertEqual(selected, int(scores[0].argmax()))

    def test_cpu_bf16_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'CUDA'):
            BatchedEvaluator(self.checkpoint, self.dmc_checkpoint, 'cpu', 'bf16')

    def test_remote_policy_does_not_load_models_in_clients(self):
        with patch('fabledan.eval_inference.InferenceClient') as client_class, \
                patch('fabledan_policy.load_ckpt') as loader, \
                patch.object(DanZeroFilter, 'from_checkpoint') as filter_loader:
            client_class.return_value.request.return_value = {'candidate_k': 10}
            policy = FableDanPolicy('unused.pt', 'unused.ckpt', 'ipc://unused')
            self.assertEqual(policy.candidate_k, 10)
            self.assertIsNone(policy.model)
            self.assertIsNone(policy.filter)
            loader.assert_not_called()
            filter_loader.assert_not_called()

    def test_local_policy_still_reads_candidate_k(self):
        with patch('fabledan_policy.load_ckpt', return_value=(Mock(), {})), \
                patch.object(DanZeroFilter, 'from_checkpoint', return_value=Mock()):
            policy = FableDanPolicy('unused.pt', 'unused.ckpt')
            self.assertEqual(policy.candidate_k, 2)
            self.assertIsNone(policy.inference_client)

    def test_worker_arguments_reuse_shared_service(self):
        from types import SimpleNamespace
        evaluator = runpy.run_path(str(REPO / 'Danzero_plus/evaluate_paper_baselines.py'))
        arguments = SimpleNamespace(policy='fabledan', device='cuda:2',
                                    infer_precision='fp32', infer_batch_size=16,
                                    inference_endpoint='ipc://shared')
        command = evaluator['inference_worker_args'](arguments)
        self.assertIn('ipc://shared', command)
        self.assertIn('cuda:2', command)
        arguments.device = 'cpu'
        arguments.inference_endpoint = None
        self.assertEqual(evaluator['inference_worker_args'](arguments)[:2],
                         ['--device', 'cpu'])

    def test_disconnected_service_times_out(self):
        client = InferenceClient(f'ipc://{self.root / "absent.sock"}', timeout_ms=50)
        try:
            with self.assertRaises(TimeoutError):
                client.request('metadata')
        finally:
            client.close()

    def test_shared_service_routes_concurrent_requests_and_errors(self):
        endpoint = f'ipc://{self.root / "server.sock"}'
        ready = self.root / 'ready.json'
        command = [sys.executable, '-m', 'fabledan.eval_inference',
                   '--checkpoint', str(self.checkpoint),
                   '--dmc-checkpoint', str(self.dmc_checkpoint), '--device', 'cpu',
                   '--endpoint', endpoint, '--ready-file', str(ready),
                   '--batch-size', '8', '--batch-wait-ms', '3']
        with (self.root / 'server.log').open('w+') as log:
            process = subprocess.Popen(command, cwd=REPO, stdout=log,
                                       stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 30
                while not ready.exists():
                    if process.poll() is not None or time.monotonic() > deadline:
                        log.seek(0)
                        self.fail(f'inference server failed: {log.read()}')
                    time.sleep(0.05)

                def request_pair(index):
                    client = InferenceClient(endpoint, timeout_ms=5000)
                    try:
                        metadata = client.request('metadata')
                        self.assertEqual(metadata['candidate_k'], 3)
                        position = index % len(self.select_requests)
                        candidates = client.request(
                            'candidates', **self.candidate_requests[position])
                        selected = client.request('select', **self.select_requests[position])
                        return position, candidates, selected
                    finally:
                        client.close()

                with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
                    results = list(pool.map(request_pair, range(12)))
                expected_candidates = self.evaluator.candidates(self.candidate_requests)
                expected_selected = self.evaluator.select(self.select_requests)
                for position, candidates, selected in results:
                    np.testing.assert_array_equal(candidates, expected_candidates[position])
                    self.assertEqual(selected, expected_selected[position])
                client = InferenceClient(endpoint, timeout_ms=5000)
                try:
                    with self.assertRaisesRegex(RuntimeError, 'unknown inference operation'):
                        client.request('unknown')
                    self.assertEqual(client.request('metadata')['device'], 'cpu')
                finally:
                    client.close()
            finally:
                process.terminate()
                process.wait(timeout=10)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_filter_and_selector_match_cpu_decisions(self):
        evaluator = BatchedEvaluator(self.checkpoint, self.dmc_checkpoint, 'cuda:0')
        self.assertEqual(next(evaluator.model.parameters()).device.type, 'cuda')
        self.assertEqual(next(evaluator.filter.parameters()).device.type, 'cuda')
        for cpu_indices, gpu_indices in zip(
                self.evaluator.candidates(self.candidate_requests),
                evaluator.candidates(self.candidate_requests)):
            np.testing.assert_array_equal(cpu_indices, gpu_indices)
        self.assertEqual(self.evaluator.select(self.select_requests),
                         evaluator.select(self.select_requests))
        evaluator.use_amp = True
        self.assertEqual(len(evaluator.select(self.select_requests)), 3)


if __name__ == '__main__':
    unittest.main()
