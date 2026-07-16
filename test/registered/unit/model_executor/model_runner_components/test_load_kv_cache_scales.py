"""Unit tests for FP8 KV cache scale loading warning gate (#31224).

Covers the false-positive warning when per-layer k_scale_float/v_scale_float
are already baked into the checkpoint (ModelOpt / compressed-tensors path)
and --quantization-param-path is not set.

Also covers a pipeline-level path: BaseKVCacheMethod.create_weights +
process_weights_after_loading → load_kv_cache_scales (the real load order in
ModelRunner), without launching a server.
"""

from __future__ import annotations

import logging
import unittest
from contextlib import contextmanager
from unittest.mock import MagicMock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.quantization.kv_cache import (  # noqa: E402
    BaseKVCacheMethod,
)
from sglang.srt.model_executor.model_runner_components.load_model_utils import (  # noqa: E402
    load_kv_cache_scales,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

LOGGER_NAME = (
    "sglang.srt.model_executor.model_runner_components.load_model_utils"
)
WARN_SNIPPET = "no scaling factors"
INFO_SNIPPET = "per-layer scaling factors from the checkpoint"


class FakeServerArgs:
    def __init__(self, kv_cache_dtype="fp8_e4m3", quantization_param_path=None):
        self.kv_cache_dtype = kv_cache_dtype
        self.quantization_param_path = quantization_param_path


class Attn(torch.nn.Module):
    def __init__(self, k_scale_float=None, v_scale_float=None):
        super().__init__()
        self.k_scale_float = k_scale_float
        self.v_scale_float = v_scale_float


class Model(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)


@contextmanager
def capture_logs(logger_name: str, level=logging.INFO):
    """Capture log records without requiring at least one emission (unlike assertLogs)."""
    logger = logging.getLogger(logger_name)
    records: list[logging.LogRecord] = []

    class _Handler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Handler(level=level)
    old_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(level)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)


def _messages(records) -> list[str]:
    return [r.getMessage() for r in records]


def _has_snippet(records, snippet: str) -> bool:
    return any(snippet in m for m in _messages(records))


class TestLoadKvCacheScales(CustomTestCase):
    """Regression coverage for issue #31224."""

    def test_no_warn_when_baked_non_default_scales_present(self):
        """Baked non-default scales must not emit the missing-scales warning.

        This is the #31224 false-positive case: ModelOpt / compressed-tensors
        checkpoints already set k_scale_float/v_scale_float during
        process_weights_after_loading, but quantization_param_path is None.
        """
        model = Model([Attn(0.028, 0.012), Attn(0.04, 0.02)])
        with capture_logs(LOGGER_NAME, logging.INFO) as records:
            load_kv_cache_scales(model=model, server_args=FakeServerArgs())
        self.assertFalse(
            _has_snippet(records, WARN_SNIPPET),
            f"unexpected missing-scales warning with baked scales: {_messages(records)}",
        )
        self.assertTrue(
            _has_snippet(records, INFO_SNIPPET),
            f"expected checkpoint-scales info log, got: {_messages(records)}",
        )

    def test_warn_when_no_scales_and_no_param_path(self):
        """True default case (None / 1.0 only) must still warn."""
        model = Model([Attn(None, None), Attn(1.0, 1.0)])
        with capture_logs(LOGGER_NAME, logging.WARNING) as records:
            load_kv_cache_scales(model=model, server_args=FakeServerArgs())
        self.assertTrue(
            _has_snippet(records, WARN_SNIPPET),
            f"expected missing-scales warning, got: {_messages(records)}",
        )

    def test_json_param_path_calls_model_loader_and_no_missing_warn(self):
        """Legacy --quantization-param-path path must call model loader."""

        class M(torch.nn.Module):
            def load_kv_cache_scales(self, path):
                self.called_with = path

        m = M()
        path = "/tmp/fake-scales.json"
        with capture_logs(LOGGER_NAME, logging.INFO) as records:
            load_kv_cache_scales(
                model=m,
                server_args=FakeServerArgs(quantization_param_path=path),
            )
        self.assertEqual(m.called_with, path)
        self.assertFalse(
            _has_snippet(records, WARN_SNIPPET),
            f"JSON path should not emit missing-scales warning: {_messages(records)}",
        )
        self.assertTrue(
            any("Loaded KV cache scaling factors" in m for m in _messages(records))
        )

    def test_non_fp8_kv_cache_dtype_silent(self):
        """Non-fp8_e4m3 dtype must not emit the FP8 KV warning."""
        model = Model([Attn(None, None)])
        with capture_logs(LOGGER_NAME, logging.WARNING) as records:
            load_kv_cache_scales(
                model=model, server_args=FakeServerArgs(kv_cache_dtype="auto")
            )
        self.assertFalse(
            _has_snippet(records, WARN_SNIPPET),
            f"non-fp8 dtype should be silent: {_messages(records)}",
        )

    def test_mixed_layers_any_non_default_suppresses_warn(self):
        """any() non-default is intentional: one baked layer suppresses global warn."""
        model = Model([Attn(0.028, 0.012), Attn(1.0, 1.0)])
        with capture_logs(LOGGER_NAME, logging.INFO) as records:
            load_kv_cache_scales(model=model, server_args=FakeServerArgs())
        self.assertFalse(
            _has_snippet(records, WARN_SNIPPET),
            f"mixed layers with one baked scale should not warn: {_messages(records)}",
        )


class TestLoadKvCacheScalesPipelineE2E(CustomTestCase):
    """Pipeline e2e: BaseKVCacheMethod weight processing → load_kv_cache_scales.

    Mirrors ModelRunner order: process_weights_after_loading finalizes
    k_scale_float/v_scale_float, then load_kv_cache_scales decides whether to
    warn. No server, no GPU — still end-to-end across the two modules.
    """

    def _make_attn_with_checkpoint_scales(self, k: float, v: float) -> torch.nn.Module:
        layer = torch.nn.Module()
        method = BaseKVCacheMethod(quant_config=MagicMock())
        method.create_weights(layer)
        # Simulate checkpoint load overwriting the -1.0 sentinel.
        with torch.no_grad():
            layer.k_scale.copy_(k)
            layer.v_scale.copy_(v)
        method.process_weights_after_loading(layer)
        return layer

    def test_pipeline_baked_scales_no_missing_warn(self):
        layers = [
            self._make_attn_with_checkpoint_scales(0.02804, 0.01186),
            self._make_attn_with_checkpoint_scales(0.04046, 0.02107),
        ]
        # Values from issue #31224 evidence table (Qwen3.5-MoE ModelOpt).
        for layer in layers:
            self.assertNotEqual(layer.k_scale_float, 1.0)
            self.assertNotEqual(layer.v_scale_float, 1.0)

        model = Model(layers)
        with capture_logs(LOGGER_NAME, logging.INFO) as records:
            load_kv_cache_scales(model=model, server_args=FakeServerArgs())
        self.assertFalse(
            _has_snippet(records, WARN_SNIPPET),
            f"pipeline with baked scales still warned: {_messages(records)}",
        )
        self.assertTrue(
            _has_snippet(records, INFO_SNIPPET),
            f"expected checkpoint-scales info log, got: {_messages(records)}",
        )

    def test_pipeline_missing_scales_still_warns(self):
        layers = []
        for _ in range(2):
            layer = torch.nn.Module()
            method = BaseKVCacheMethod(quant_config=MagicMock())
            method.create_weights(layer)
            # Leave sentinel -1.0 so process defaults to 1.0.
            method.process_weights_after_loading(layer)
            layers.append(layer)
            self.assertEqual(layer.k_scale_float, 1.0)
            self.assertEqual(layer.v_scale_float, 1.0)

        model = Model(layers)
        with capture_logs(LOGGER_NAME, logging.WARNING) as records:
            load_kv_cache_scales(model=model, server_args=FakeServerArgs())
        self.assertTrue(
            _has_snippet(records, WARN_SNIPPET),
            f"true missing scales should still warn: {_messages(records)}",
        )


if __name__ == "__main__":
    unittest.main()
