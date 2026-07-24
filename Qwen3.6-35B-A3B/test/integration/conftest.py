# SPDX-License-Identifier: Apache-2.0
"""Conftest for the Qwen3.6-35B-A3B integration tests.

Sets the validated recipe flags that must be live BEFORE `import vllm` (they
are read at model/platform import time). Mirrors
examples/vllm_neuron/models/qwen3_5_moe/run.py so the tests run with no manual
`export`; restores the prior environment on teardown.

  VLLM_GDN_SEQ_NKI         bounded-graph GatedDeltaNet prefill (required seq>=512)
  VLLM_UNIFIED_KV_GATHER   block-indexed .ap KV gather (else torch fallback)
  VLLM_MOE_TKG_ROUTER_FP32 fp32 decode router (correct top-8)
"""

import os

import pytest

_ENV_OVERRIDES = {
    "VLLM_GDN_SEQ_NKI": "1",
    "VLLM_UNIFIED_KV_GATHER": "1",
    "VLLM_MOE_TKG_ROUTER_FP32": "1",
}


@pytest.fixture(autouse=True, scope="module")
def _qwen36_recipe_env():
    """Apply the required recipe env vars for the module, restore on teardown."""
    old = {k: os.environ.get(k) for k in _ENV_OVERRIDES}
    for k, v in _ENV_OVERRIDES.items():
        # setdefault semantics: an explicit export still wins.
        os.environ.setdefault(k, v)
    yield
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
