"""Regression tests for the EPLB + DeepSeek-V4 draft-model crash (#34974).

Enabling ``--enable-eplb`` alongside a DeepSeek-V4 draft model (DSPARK or NextN)
crashed every rank during draft CUDA graph capture:

    RuntimeError: Index tensor must have the same number of dimensions as self tensor

``--enable-eplb`` force-enables the expert distribution recorder and ``EPLBManager``
starts recording at construction, so the recorder's per-layer ``on_select_experts``
hook fires on every MoE layer. The hook takes its row index from an ambient
"current layer" value that the model's forward loop is responsible for setting.
The target model sets it; every other MoE draft model suppresses the hook with
``disable_this_region()``. The two DeepSeek-V4 draft models did neither, so the
hook ran with ``layer_idx=None`` and indexed the 2-D counter tensor with
``None`` -- which inserts an axis instead of selecting a row, leaving
``scatter_add_`` with a rank mismatch.

These tests drive the real recorder through the real draft ``forward`` methods.
The stand-in stages/decoder do exactly what a real MoE layer does at the point
that matters -- call ``on_select_experts`` the way ``select_experts`` does in
``layers/moe/topk.py`` -- so no weights, GPU, or server are needed.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.eplb.expert_distribution import (
    ExpertDistributionRecorder,
    get_global_expert_distribution_recorder,
    set_global_expert_distribution_recorder,
)
from sglang.srt.eplb.expert_location import ExpertLocationMetadata
from sglang.srt.models import deepseek_v4_nextn as nextn_module
from sglang.srt.models.deepseek_v4_dspark import DeepseekV4ForCausalLMDSpark
from sglang.srt.models.deepseek_v4_nextn import DeepseekV4ModelNextN
from sglang.srt.runtime_context import get_parallel
from sglang.test.test_utils import CustomTestCase

# The recorder's counter tensor is sized by the *target* model's layer count.
# Draft stages are indexed from 0, so they would alias onto target rows.
NUM_TARGET_LAYERS = 8
NUM_EXPERTS = 16
HIDDEN = 4
HC_MULT = 2


def _build_expert_location_metadata() -> ExpertLocationMetadata:
    """A trivial placement: logical expert i lives at physical slot i."""
    physical_to_logical_map = torch.arange(NUM_EXPERTS).repeat(NUM_TARGET_LAYERS, 1)
    logical_to_all_physical_map = torch.zeros(
        NUM_TARGET_LAYERS, NUM_EXPERTS, NUM_EXPERTS, dtype=torch.long
    )
    logical_to_all_physical_map[..., 0] = physical_to_logical_map
    return ExpertLocationMetadata(
        physical_to_logical_map=physical_to_logical_map,
        physical_to_logical_map_cpu=physical_to_logical_map.clone(),
        logical_to_all_physical_map=logical_to_all_physical_map,
        logical_to_all_physical_map_cpu=logical_to_all_physical_map.clone(),
        logical_to_all_physical_map_num_valid=torch.ones(
            NUM_TARGET_LAYERS, NUM_EXPERTS, dtype=torch.long
        ),
        ep_size=1,
        logical_to_rank_dispatch_physical_map=None,
    )


def _build_server_args() -> SimpleNamespace:
    """The recorder fields only. ``--enable-eplb`` forces mode to ``stat``."""
    return SimpleNamespace(
        expert_distribution_recorder_mode="stat",
        expert_distribution_recorder_buffer_size=4,
        enable_expert_distribution_metrics=False,
        moe_a2a_backend="none",
        deepep_mode="normal",
        elastic_ep_backend=None,
        device="cpu",
    )


class _MoeLayerStandIn:
    """Calls the recorder hook exactly as ``select_experts`` does.

    ``layers/moe/topk.py`` calls ``on_select_experts(topk_ids=...)`` with no
    ``is_nextn`` gate, so a draft MoE layer reaches it the same way a target one
    does. ``topk_ids`` may carry ``-1`` padding, which the gatherer masks off.
    """

    def __init__(self, topk_ids: torch.Tensor):
        self._topk_ids = topk_ids
        self.call_count = 0
        # None until called, so a layer the forward never reached is
        # distinguishable from one that ran with the recorder suppressed.
        self.saw_recording_enabled = None

    def __call__(self, positions, hidden_states, forward_batch, *args, **kwargs):
        recorder = get_global_expert_distribution_recorder()
        self.call_count += 1
        self.saw_recording_enabled = not recorder._disable_all
        recorder.on_select_experts(topk_ids=self._topk_ids)
        return hidden_states


class _NextnDecoderStandIn(_MoeLayerStandIn):
    """NextN's decoder returns ``(hidden_states, residual, post, comb)``."""

    def __call__(self, positions, hidden_states, forward_batch, *args, **kwargs):
        super().__call__(positions, hidden_states, forward_batch)
        return hidden_states, None, None, None


def _topk_ids(kind: str) -> torch.Tensor:
    if kind == "normal":
        return torch.tensor([0, 3, 7, 15], dtype=torch.int64)
    if kind == "padded":
        # -1 is the "no expert" pad the gatherer masks off.
        return torch.tensor([[0, -1], [5, -1]], dtype=torch.int64)
    if kind == "empty":
        return torch.zeros(0, dtype=torch.int64)
    raise ValueError(kind)


def _run_dspark_forward(stages) -> None:
    """Drive ``DeepseekV4ForCausalLMDSpark.forward`` over the given stages.

    Built with ``object.__new__`` because ``forward`` reads only ``self.stages``
    when ``input_embeds`` is supplied -- constructing the real module would need
    a DeepSeek-V4 checkpoint.
    """
    model = object.__new__(DeepseekV4ForCausalLMDSpark)
    model.stages = stages
    model.forward(
        input_ids=torch.zeros(2, dtype=torch.int64),
        positions=torch.arange(2, dtype=torch.int64),
        forward_batch=SimpleNamespace(),
        input_embeds=torch.zeros(2, HC_MULT, HIDDEN),
    )


class _Identity:
    def __call__(self, x):
        return x


class _IdentityLinear:
    """ReplicatedLinear returns ``(output, bias)``."""

    def __call__(self, x):
        return x, None


def _run_nextn_forward(decoders) -> None:
    """Drive ``DeepseekV4ModelNextN.forward`` over its single MoE decoder.

    The projections around the decoder are identity stand-ins; only the decoder
    call site matters here. ``dsa_use_prefill_cp`` and ``attn_dp_size`` need a
    published runtime config that a unit test has no reason to build, so the CP
    branch is pinned off -- it is orthogonal to the recorder.
    """
    (decoder,) = decoders
    model = object.__new__(DeepseekV4ModelNextN)
    model.config = SimpleNamespace(hidden_size=HIDDEN)
    model.hc_mult = HC_MULT
    model.rms_norm_eps = 1e-6
    model.hc_eps = 1e-6
    model.hnorm = _Identity()
    model.enorm = _Identity()
    model.h_proj = _IdentityLinear()
    model.e_proj = _IdentityLinear()
    model.decoder = decoder
    model.hc_head_fn = torch.zeros(HC_MULT, HC_MULT * HIDDEN)
    model.hc_head_base = torch.zeros(HC_MULT)
    model.hc_head_scale = torch.ones(1)
    model.shared_head = SimpleNamespace(norm=_Identity())

    n_tokens = 2
    forward_batch = SimpleNamespace(
        attn_cp_metadata=None,
        forward_mode=None,
        spec_info=SimpleNamespace(
            hidden_states=torch.zeros(n_tokens * HC_MULT, HIDDEN)
        ),
    )
    with (
        patch.object(nextn_module, "dsa_use_prefill_cp", return_value=False),
        get_parallel().override(attn_dp_size=1),
    ):
        model.forward(
            input_ids=torch.zeros(n_tokens, dtype=torch.int64),
            positions=torch.arange(n_tokens, dtype=torch.int64),
            forward_batch=forward_batch,
            input_embeds=torch.zeros(n_tokens, HIDDEN),
        )


# name -> (runner, MoE layer stand-in, layer count, topk_ids kind)
_CASES = [
    ("dspark_multi_stage", _run_dspark_forward, _MoeLayerStandIn, 3, "normal"),
    ("dspark_single_stage", _run_dspark_forward, _MoeLayerStandIn, 1, "normal"),
    ("dspark_padded_topk_ids", _run_dspark_forward, _MoeLayerStandIn, 2, "padded"),
    ("dspark_empty_topk_ids", _run_dspark_forward, _MoeLayerStandIn, 2, "empty"),
    ("nextn_decoder", _run_nextn_forward, _NextnDecoderStandIn, 1, "normal"),
    ("nextn_padded_topk_ids", _run_nextn_forward, _NextnDecoderStandIn, 1, "padded"),
]


class TestDraftModelSuppressesExpertRecorder(CustomTestCase):
    """The draft forward must not feed the target's expert statistics."""

    def setUp(self):
        self._previous_recorder = get_global_expert_distribution_recorder()
        # The gatherer allocates its counters on get_device(), which needs an
        # accelerator. Which device holds them is irrelevant here -- the bug is
        # in how the row is indexed -- so keep them on CPU.
        with patch(
            "sglang.srt.eplb.expert_distribution.get_device", return_value="cpu"
        ):
            self.recorder = ExpertDistributionRecorder.init_new(
                _build_server_args(), _build_expert_location_metadata(), rank=0
            )
        set_global_expert_distribution_recorder(self.recorder)
        # EPLBManager.start_record()s at construction, so recording is already
        # live by the time draft CUDA graph capture runs.
        self.recorder.start_record()

    def tearDown(self):
        set_global_expert_distribution_recorder(self._previous_recorder)

    def _gatherer(self):
        (gatherer,) = self.recorder._single_pass_gatherers.values()
        return gatherer

    def test_draft_forward_does_not_crash_or_record(self):
        for name, run_forward, layer_cls, num_layers, topk_kind in _CASES:
            with self.subTest(case=name):
                self._gatherer()._data.zero_()
                layers = [layer_cls(_topk_ids(topk_kind)) for _ in range(num_layers)]

                # Before the fix this raises:
                #   Index tensor must have the same number of dimensions as self
                run_forward(layers)

                for i, layer in enumerate(layers):
                    # assertFalse alone would pass for a layer the forward never
                    # reached, since saw_recording_enabled would still be None.
                    self.assertEqual(
                        layer.call_count, 1, f"layer {i} was not run by the forward"
                    )
                    self.assertIs(
                        layer.saw_recording_enabled,
                        False,
                        f"layer {i} ran with the recorder still active",
                    )
                self.assertEqual(
                    int(self._gatherer()._data.sum()),
                    0,
                    "draft routing leaked into the target's expert counters",
                )

    def test_recording_still_works_when_layer_context_is_set(self):
        """The fix must suppress the draft only, not disable recording globally.

        This drives a stand-in directly inside ``with_current_layer`` rather than
        a target forward: it pins the recorder's contract, not the target model's.
        """
        layer = _MoeLayerStandIn(_topk_ids("normal"))
        self._gatherer()._data.zero_()

        with self.recorder.with_current_layer(2):
            layer(None, None, None)

        data = self._gatherer()._data
        self.assertIs(layer.saw_recording_enabled, True)
        self.assertEqual(int(data[2].sum()), 4, "target layer counts were not recorded")
        self.assertEqual(int(data.sum()), 4, "counts landed outside the target layer")

    def test_hook_without_layer_context_is_the_underlying_defect(self):
        """Pins the root cause so the forward tests cannot pass for a bad reason."""
        with self.assertRaises(RuntimeError) as caught:
            self.recorder.on_select_experts(topk_ids=_topk_ids("normal"))
        self.assertIn("same number of dimensions", str(caught.exception))

        # ...and that suppressing the region is what makes it safe.
        with self.recorder.disable_this_region():
            self.recorder.on_select_experts(topk_ids=_topk_ids("normal"))
        self.assertEqual(int(self._gatherer()._data.sum()), 0)


if __name__ == "__main__":
    unittest.main()
