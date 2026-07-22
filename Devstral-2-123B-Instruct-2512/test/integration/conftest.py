# SPDX-License-Identifier: Apache-2.0
"""Conftest for tiny E2E integration tests.

Sets env vars required for CPU-mode E2E inference:
- VLLM_ENABLE_V1_MULTIPROCESSING=0: V1 engine core subprocess hangs in CPU
  mode; in-process mode avoids this while TP workers still spawn safely.
- VLLM_NEURON_MIN_KV_BUDGET_GIB=0: no real HBM exists in CPU mode, so the
  computed budget is always near zero.
"""

import os

import pytest

_ENV_OVERRIDES = {
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    "VLLM_NEURON_MIN_KV_BUDGET_GIB": "0",
}


@pytest.fixture(autouse=True, scope="module")
def _tiny_e2e_env():
    """Set required env vars for tiny E2E tests, restore on teardown."""
    old = {k: os.environ.get(k) for k in _ENV_OVERRIDES}
    os.environ.update(_ENV_OVERRIDES)
    yield
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
