"""NKI dispatch for the GLM-5.2 MLA inner attention block (Phase 5a-ii).

Wraps `mla_block_kernel.glm52_mla_block_kernel` in the same way the framework wraps
its own NKI kernels (`nki.jit()` -> `wrap_nki` -> `wrapped[lnc](...)`, cf.
`vllm_neuron/functional/attention/attention_segmented_cte.py`), and gates it on
`can_run_kernel` so CPU mode falls back to torch.

🔴 **Opt-in, default OFF.** Set `GLM52_MLA_BLOCK_KERNEL=1` to enable.

Validated on the NKI CPU simulator against the torch path it replaces. Every figure is
reproduced by `equiv_glm_5_2/tests/test_08_mla_block_kernel_agreement.py`, which is the
source of these numbers rather than a transcription of them:

| stage | fp32 | **bf16 (production)** |
|---|---|---|
| inside `_mla_attend_tiled`, Sq=128 (1 query tile) | rel_fro 2.55e-07 | **3.44e-03** |
| inside `_mla_attend_tiled`, Sq=256 (2 query tiles) | rel_fro 2.63e-07 | **3.47e-03** |
| inside `_mla_attend_tiled`, Sq=200 (last tile 72 wide) | rel_fro 2.60e-07 | **3.46e-03** |
| all-masked segment leaves (m, l, acc) untouched | exact, 0.000e+00 | exact, 0.000e+00 |
| chained segments == one softmax over the concatenation | rel_fro 2.29e-07 | — |

⚠️ **Read the bf16 column, not the fp32 one.** The model runs bf16; fp32 agreement is
diagnostic only. Online softmax is exact, so both columns are accumulation order — bf16
carries ~8 mantissa bits against fp32's 24, and 3.4e-03 is what that costs.

🔴 An earlier version of this table gave **3.5e-03** for the end-to-end row and I recorded
that it "measured a path where the kernel never executed". The reachability bug was real
and measured (`q_tile=Sq` plus `_shape_ok`'s `Sq <= 128` meant the gate opened zero times,
fixed by `_Q_TILE = 128` in model.py). But the *explanation of the number* was wrong: with
the kernel never running, ON and OFF are the same code path and the difference is exactly
0, not 3.5e-03. The old figure is within 2% of the bf16 measurement above, so it was
almost certainly a legitimate **bf16** comparison — of what exactly is no longer
recoverable, since it had no source in the repo. The lesson that survives is the one about
provenance, not about that number being fabricated.

⚠️ Reproducing these needs BOTH `VLLM_NEURON_CPU_MODE=1` and `NKI_SIMULATOR=1`:
`can_run_kernel` only consults `NKI_SIMULATOR` inside its `VLLM_NEURON_CPU_MODE` branch,
so `NKI_SIMULATOR` alone leaves the gate shut and any "agreement" measured that way is
torch against torch. Plus `GLM52_MLA_BLOCK_KERNEL=1` for the flag below.

It has **not** been run on device, which is why it is not the default: that would
put an unexercised path in front of every request.

The flag is read from the environment rather than added to `vllm_neuron/envs.py` to
keep the shared-framework diff minimal, per ADD_MODEL_TO_FORK_INSTRUCTIONS.md §2c.
"""

import os

import torch

try:  # the wrapper machinery only exists on a Neuron-enabled install
    import nki
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from .mla_block_kernel import glm52_mla_block_kernel as _raw_kernel

    _JIT = nki.jit()(_raw_kernel)
    _WRAPPED = wrap_nki(_JIT)
except Exception:  # noqa: BLE001 - absence is a valid state; torch path is used
    _WRAPPED = None

from vllm_neuron.utils.neuron_utils import can_run_kernel

# exp() of this underflows to 0, so an all-masked block contributes nothing without
# producing the NaN that a true -inf would. Must match the kernel's sentinel.
MASK_NEG = -30000.0


def kernel_enabled() -> bool:
    return os.environ.get("GLM52_MLA_BLOCK_KERNEL", "") in ("1", "true", "True")


def _shape_ok(Sq: int, L: int, R: int) -> bool:
    """The kernel's own shape asserts: Sq <= 128 partitions, L % 128 == 0, R <= 128.

    Split out from `can_use_block_kernel` so the shape rule can be checked
    independently of device availability — `can_run_kernel` is False in CPU mode
    unless NKI_SIMULATOR=1, which would otherwise mask whether a shape is legal.
    """
    return Sq <= 128 and L % 128 == 0 and R <= 128


def can_use_block_kernel(q_lift: torch.Tensor, Sq: int, L: int, R: int) -> bool:
    """Whether the NKI inner block can serve this shape on this device."""
    if not kernel_enabled() or _WRAPPED is None:
        return False
    if not can_run_kernel(q_lift):
        return False
    return _shape_ok(Sq, L, R)


def mla_block(
    q_lift_t: torch.Tensor,   # [L, Sq]
    c_kv_t: torch.Tensor,     # [L, Sk]
    q_pe_t: torch.Tensor,     # [R, Sq]
    k_pe_t: torch.Tensor,     # [R, Sk]
    mask_add: torch.Tensor,   # [Sq, Sk] fp32, 0 visible / MASK_NEG masked
    m_in: torch.Tensor,       # [Sq, 1] fp32
    l_in: torch.Tensor,       # [Sq, 1] fp32
    acc_in: torch.Tensor,     # [Sq, L] fp32
    softmax_scale: float,
    lnc: int = 2,
):
    """One online-softmax step over a key segment. Returns (m, l, acc)."""
    return _WRAPPED[lnc](
        q_lift_t, c_kv_t, q_pe_t, k_pe_t, mask_add, m_in, l_in, acc_in, softmax_scale
    )
