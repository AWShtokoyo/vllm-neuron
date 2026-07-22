# Tutorials

End-to-end guided walkthroughs for specific deployment scenarios and performance optimization.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Disaggregated inference: 1P1D and xPyD
:link: tutorial-di-1p1d-xpyd
:link-type: doc

Configure disaggregated inference topologies.
:::

:::{grid-item-card} Deploy gpt-oss
:link: tutorial-gpt-oss
:link-type: doc

Deploy gpt-oss 20B and 120B, single-instance or disaggregated.
:::

:::{grid-item-card} Prefix caching benchmark
:link: tutorial-prefix-caching-gpt-oss-benchmarking
:link-type: doc

Measure TTFT improvement from prefix caching with GPT-OSS.
:::

:::{grid-item-card} Deploy Qwen3-VL-32B
:link: tutorial-qwen3-vl-32b
:link-type: doc

Serve the multimodal Qwen3-VL-32B model.
:::

:::{grid-item-card} Deploy Devstral-2-123B
:link: tutorial-devstral-2-123b
:link-type: doc

Serve the Devstral-2-123B dense FP8 model.
:::

::::

:::{toctree}
:maxdepth: 1
:hidden:

Disaggregated inference (1P1D and xPyD) <tutorial-di-1p1d-xpyd>
Deploying gpt-oss <tutorial-gpt-oss>
Benchmarking prefix caching (GPT-OSS) <tutorial-prefix-caching-gpt-oss-benchmarking>
Deploying Qwen3-VL-32B <tutorial-qwen3-vl-32b>
Deploying Devstral-2-123B <tutorial-devstral-2-123b>
:::
