"""E2E: FP8 KV cache with baked scales must not emit the missing-scales warning.

Regression for https://github.com/sgl-project/sglang/issues/31224.

Launches a known FP8-KV checkpoint with --kv-cache-dtype fp8_e4m3 and no
--quantization-param-path, captures server logs, and asserts:

  * the false-positive "no scaling factors provided" warning is absent, and
  * the positive "per-layer scaling factors from the checkpoint" info is present.

Pattern matches sibling server-log e2e tests (e.g. test_post_capture_kv_sizing,
test_quark_mxfp4) that use popen_launch_server(..., return_stdout_stderr=...).
"""

from __future__ import annotations

import io
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=120, stage="extra-a", runner_config="1-gpu-large")

# Same FP8-KV model used by test_fp8kv_triton.py — checkpoint carries per-layer
# KV scales (modern baked path, not --quantization-param-path).
MODEL = "neuralmagic/Meta-Llama-3-8B-Instruct-FP8-KV"
WARN_SNIPPET = "no scaling factors"
INFO_SNIPPET = "per-layer scaling factors from the checkpoint"


class TestFp8KvBakedScalesNoFalseWarn(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        # Match test_quark_mxfp4 / test_soft_watchdog: StringIO avoids on-disk
        # log files and cwd collisions under parallel CI.
        cls.stdout = io.StringIO()
        cls.stderr = io.StringIO()
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--quantization",
                "fp8",
                "--kv-cache-dtype",
                "fp8_e4m3",
                "--attention-backend",
                "triton",
                "--log-level",
                "info",
            ],
            return_stdout_stderr=(cls.stdout, cls.stderr),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls.stdout.close()
        cls.stderr.close()

    def _server_logs(self) -> str:
        return self.stdout.getvalue() + self.stderr.getvalue()

    def test_no_false_missing_scales_warning(self):
        """Baked FP8 KV scales must not trigger the missing-scales warning."""
        logs = self._server_logs()
        self.assertNotIn(
            WARN_SNIPPET,
            logs,
            "Server logs still contain the #31224 false-positive warning "
            "despite baked per-layer KV scales in the checkpoint.",
        )
        self.assertIn(
            INFO_SNIPPET,
            logs,
            "Expected positive info that per-layer scales came from the "
            "checkpoint; not found in server logs.",
        )


if __name__ == "__main__":
    unittest.main()
