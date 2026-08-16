import math, unittest
import torch
from sglang.srt.layers.attention.dsa_backend import (
    DSA_TRTLLM_MAX_PSEUDO_BATCH,
    _pseudo_batch_slices,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDSATRTLLMPseudoBatch(CustomTestCase):
    def test_pseudo_batch_slices(self):
        for num_rows in [0, 1, 2, 65534, 65535, 65536, 65537, 131071, 131072, 196605]:
            with self.subTest(num_rows=num_rows):
                slices = _pseudo_batch_slices(num_rows)
                expected_len = math.ceil(num_rows / 65535) if num_rows > 0 else 0
                self.assertEqual(len(slices), expected_len)
                curr = 0
                for s in slices:
                    self.assertEqual(s.start, curr)
                    self.assertLessEqual(s.stop - s.start, DSA_TRTLLM_MAX_PSEUDO_BATCH)
                    curr = s.stop
                self.assertEqual(curr, num_rows)

    def test_chunked_launch_writes_every_row(self):
        for num_rows in [0, 1, 2, 65534, 65535, 65536, 65537, 131071, 131072, 196605]:
            with self.subTest(num_rows=num_rows):
                q = torch.arange(num_rows, dtype=torch.float32).view(num_rows, 1, 1, 1)
                out = torch.full((num_rows, 1, 1, 1), float("nan"), dtype=torch.float32)
                calls = 0

                def fake_kernel(query, out, **kw):
                    nonlocal calls
                    calls += 1
                    out.copy_(query * 2)

                slices = _pseudo_batch_slices(num_rows)
                for s in slices:
                    fake_kernel(q[s], out[s])

                expected_calls = math.ceil(num_rows / 65535) if num_rows > 0 else 0
                self.assertEqual(calls, expected_calls)
                if num_rows > 0:
                    self.assertFalse(torch.isnan(out).any().item())
                    self.assertTrue(torch.equal(out, q * 2))


if __name__ == "__main__":
    unittest.main()
