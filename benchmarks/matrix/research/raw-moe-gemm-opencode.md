
> Sisyphus - ultraworker · glm-5.2

• Find AWQMoeMarlin support predicate in vLLM Librarian Agent
• Find Moe WNA16 fallback kernel in vLLM Librarian Agent
• Find all ROCm MoE backends and env vars in vLLM Librarian Agent
• Find AITER fmoe variants and dtype support Librarian Agent
• Find Triton fused MoE tuning configs in vLLM Librarian Agent
• Find local MoE / AITER context in mi210-llm-stack Explore Agent
I detect **research/investigation** intent - user wants an exhaustive structured report on vLLM MoE expert GEMM optimization for gfx90a, with no file writes. My approach: fire many parallel background agents (librarian for remote vLLM source via GitHub, explore for local repo context) + direct grep.app/web searches + context7, then synthesize.

Launching parallel sweep now.
✓ Find AITER fmoe variants and dtype support Librarian Agent
✓ Find Moe WNA16 fallback kernel in vLLM Librarian Agent
✓ Find local MoE / AITER context in mi210-llm-stack Explore Agent
✓ Find all ROCm MoE backends and env vars in vLLM Librarian Agent
✓ Find AWQMoeMarlin support predicate in vLLM Librarian Agent
⚙ grep_app_searchGitHub {"query":"Falling back to Moe WNA16 kernels"}
⚙ grep_app_searchGitHub {"query":"AWQMoeMarlin"}
⚙ grep_app_searchGitHub {"query":"is not supported by AWQMoeMarlin","language":["Python"]}
✓ Find Triton fused MoE tuning configs in vLLM Librarian Agent
Got strong initial hits. The predicate is `check_moe_marlin_supports_layer` in `vllm/model_executor/layers/quantization/utils/marlin_utils.py`. Let me fire more parallel searches for the predicate internals, web docs, and context7.
% WebFetch https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/layers/quantization/moe_wna16.py
% WebFetch https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/layers/quantization/utils/marlin_utils.py
✗ grep_app_searchGitHub {"query":"class FusedMoEConfigKernelConfig"} failed
Error: No results found for your query.

IMPORTANT: This tool searches for literal code patterns that would appear in files, not keywords or natural language queries. For example:
- Instead of 'react hook state', search for 'useState('
- Instead of 'error handling', search 'catch ('
⚙ grep_app_searchGitHub {"query":"moe_backend"}
⚙ grep_app_searchGitHub {"query":"fmoe_2stages"}
⚙ grep_app_searchGitHub {"query":"VLLM_ROCM_USE_AITER_MOE"}
⚙ grep_app_searchGitHub {"query":"class MoeWNA16Config"}
⚙ grep_app_searchGitHub {"query":"VLLM_USE_TRITON_AWQ"}
⚙ grep_app_searchGitHub {"query":"aiter_fmoe forward_default"}
⚙ grep_app_searchGitHub {"query":"def check_moe_marlin_supports_layer"}
⚙ grep_app_searchGitHub {"query":"def check_marlin_supports_layer"}
% WebFetch https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/utils/marlin_utils.py
⚙ grep_app_searchGitHub {"query":"def select_wna16_moe_backend"}
**Smoking gun for Q1 found**: `check_moe_marlin_supports_layer` at the top has `if current_platform.is_rocm(): return False` — an unconditional hard block. Let me fire additional parallel searches for the WNA16 oracle dispatcher, AITER gfx90a support, and Triton tuning paths.
⚙ grep_app_searchGitHub {"query":"available_backends WNA16MoEBackend TRITON_FUSED_MOE"}
⚙ grep_app_searchGitHub {"query":"VLLM_TEST_FUSED_MOE_TUNER"}
⚙ grep_app_searchGitHub {"query":"def get_default_config fused_moe"}
⚙ grep_app_searchGitHub {"query":"is_supported_config fmoe bf16"}
⚙ grep_app_searchGitHub {"query":"fmoe_v2_bf16"}
⚙ grep_app_searchGitHub {"query":"def tune_fused_moe"}
% WebFetch https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/layers/fused_moe/oracle/int_wna16.py
⚙ grep_app_searchGitHub {"query":"aiter fmoe gfx90a"}
⚙ grep_app_searchGitHub {"query":"gfx90a fused_moe config"}
⚙ grep_app_searchGitHub {"query":"class WNA16MoEBackend"}
⚙ websearch_web_search_exa {"query":"vLLM AMD gfx90a MI210 AWQ MoE expert GEMM performance bottleneck Triton fused_moe tuning","numResults":8}
Massive find: **vLLM PR #43389** is the exact thing that optimizes MoE int4 Triton on ROCm. Also confirmed: WNA16MoEBackend.TRITON is the ROCm path via `TritonWNA16Experts`. Let me check the local repo and wait for librarian agents.
→ Read .
⚙ grep_app_searchGitHub {"query":"get_moe_wna16_block_config"}
⚙ grep_app_searchGitHub {"query":"get_search_path | get_config_path fused_moe"}
⚙ grep_app_searchGitHub {"query":"def is_supported_config TritonWNA16Experts"}
⚙ grep_app_searchGitHub {"query":"def create_forward_function TritonWNA16Experts"}
⚙ grep_app_searchGitHub {"query":"class TritonWNA16Experts"}
Let me peek at the local project's key docs/JOURNAL while waiting for librarian agents.
→ Read docs
→ Read JOURNAL.md [limit=150]
→ Read README.md [limit=120]
The local repo confirms a lot. Let me check the AITER port matrix doc and PR #43389 status while waiting for librarian agents.
→ Read docs/19-aiter-operator-port-matrix.md [limit=100]
⚙ grep_app_searchGitHub {"query":"def get_default_config fused_moe M N K E"}
% WebFetch https://api.github.com/repos/vllm-project/vllm/pulls/43389
PR #43389 confirmed **OPEN, not merged** — so not in v0.23.1. Local docs confirm INT8 GEMM works on gfx90a from-source at 4.3× decode speedup. Now let me wait for librarian agents to finalize before synthesizing the report.
⚙ background_output librarian - Find AWQMoeMarlin support predicate in vLLM
Q1 fully answered. Now collecting the rest in parallel.
⚙ background_output librarian - Find AITER fmoe variants and dtype support
⚙ background_output librarian - Find all ROCm MoE backends and env vars in vLLM
⚙ background_output explore - Find local MoE / AITER context in mi210-llm-stack
⚙ background_output librarian - Find Triton fused MoE tuning configs in vLLM
⚙ background_output librarian - Find Moe WNA16 fallback kernel in vLLM
4 librarian tasks still running. The local explore task completed with rich context (confirms INT8 GEMM 4.3× decode win, FP8 bit-trick, AITER 242-kernel port ceiling). I'll end my turn and wait for the system notifications.

# Todos
[✓] Q1: AWQMoeMarlin predicate - DONE (smoking gun: `if current_platform.is_rocm(): return False` at marlin_utils.py:360-361)
[•] Q2: Moe WNA16 fallback path - waiting on bg_1f9b25b4 (Triton vs CUDA/HIP)
[•] Q3: Enumerate all ROCm MoE backends - waiting on bg_ad8695eb
[•] Q4: AITER fmoe variants/dtypes - waiting on bg_3510fba1
[•] Q5: Triton fused-MoE tuning configs - waiting on bg_b286ccdf
[ ] Q6: Highest-leverage change - synthesize from Q1-Q5 + local INT8 GEMM findings
[ ] Write final structured markdown report

Waiting for system reminders on `bg_1f9b25b4`, `bg_ad8695eb`, `bg_3510fba1`, `bg_b286ccdf`. Ending turn.
