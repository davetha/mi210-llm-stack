
> Sisyphus - ultraworker · glm-5.2

• Trace use_rocm_custom_paged_attention call chain in vLLM Librarian Agent
• Find GitHub issues on vLLM decode throughput collapse with large max_model_len Librarian Agent
I detect **research/investigation** intent — root-cause a 10x decode throughput delta in vLLM on ROCm by tracing source and surveying issues. No implementation; output is structured findings.

Firing parallel research: grep.app on vllm-project/vllm, two librarian agents (source trace + GitHub issues), and direct web search. Collecting all results before synthesis.
✓ Trace use_rocm_custom_paged_attention call chain in vLLM Librarian Agent
✓ Find GitHub issues on vLLM decode throughput collapse with large max_model_len Librarian Agent
⚙ grep_app_searchGitHub {"query":"def use_rocm_custom_paged_attention","repo":"vllm-project/vllm"}
⚙ grep_app_searchGitHub {"query":"max_seq_len <= 128 * 1024","repo":"vllm-project/vllm"}
⚙ grep_app_searchGitHub {"query":"chunked_prefill_paged_decode","repo":"vllm-project/vllm","path":"vllm/v1/attention/ops/"}
⚙ grep_app_searchGitHub {"query":"use_rocm_custom_paged_attention(","repo":"vllm-project/vllm","language":["Python"]}
⚙ websearch_web_search_exa {"query":"vLLM cudagraph capture size max_model_len KV cache num_gpu_blocks decode performance regression","numResults":10}
⚙ websearch_web_search_exa {"query":"vLLM decode throughput collapse when max_model_len set larger than actual prompts served ROCm MI210 gfx90a","numResults":10}
Strong initial signal. Now firing more targeted reads to nail down the exact `max_seq_len` argument value and the cudagraph capture-size cap that depends on `max_num_seqs`/`max_num_batched_tokens`.
⚙ grep_app_searchGitHub {"language":["Python"],"query":"max_num_batched_tokens","repo":"vllm-project/vllm","path":"vllm/config/scheduler.py"}
⚙ grep_app_searchGitHub {"language":["Python"],"query":"max_model_len 131072","repo":"vllm-project/vllm"}
⚙ grep_app_searchGitHub {"language":["Python"],"query":"long_prefill_token_threshold","repo":"vllm-project/vllm"}
⚙ grep_app_searchGitHub {"language":["Python"],"query":"cudagraph_capture_sizes","repo":"vllm-project/vllm","path":"vllm/config/"}
⚙ zread_read_file {"repo_name":"vllm-project/vllm","file_path":"vllm/v1/attention/ops/chunked_prefill_paged_decode.py"}
⚙ zread_read_file {"repo_name":"vllm-project/vllm","file_path":"vllm/platforms/rocm.py"}
Now nailing down: (a) how `max_num_batched_tokens` defaults derive from `max_model_len`, and (b) what value gets passed as `max_seq_len` to `chunked_prefill_paged_decode`. Firing parallel reads.
⚙ grep_app_searchGitHub {"language":["Python"],"path":"vllm/engine/arg_utils.py","query":"max_num_batched_tokens = max_model_len","repo":"vllm-project/vllm"}
⚙ grep_app_searchGitHub {"language":["Python"],"path":"vllm/engine/arg_utils.py","query":"_default_max_num_batched_tokens","repo":"vllm-project/vllm"}
⚙ grep_app_searchGitHub {"language":["Python"],"query":"chunked_prefill_paged_decode(","repo":"vllm-project/vllm"}
⚙ grep_app_searchGitHub {"language":["Python"],"path":"vllm/config/vllm.py","query":"max_cudagraph_capture_size","repo":"vllm-project/vllm"}
⚙ grep_app_searchGitHub {"language":["Python"],"path":"vllm/v1/attention/backends/rocm_attn.py","query":"max_seq_len","repo":"vllm-project/vllm"}
⚙ grep_app_searchGitHub {"language":["Python"],"path":"vllm/config/compilation.py","query":"cudagraph_capture_sizes","repo":"vllm-project/vllm"}
Critical finding: `chunked_prefill_paged_decode`'s `max_seq_len` argument = `attn_metadata.max_seq_len` = the **batch's runtime max seq length**, not `max_model_len`. So the user's "ruled out" hypothesis is correctly ruled out. Now I need the `_set_cudagraph_sizes` logic and `max_num_batched_tokens` default derivation.
⚙ grep_app_searchGitHub {"language":["Python"],"path":"vllm/engine/arg_utils.py","query":"create_engine_config","repo":"vllm-project/vllm"}
⚙ grep_app_searchGitHub {"language":["Python"],"path":"vllm/engine/arg_utils.py","query":"max_num_batched_tokens =","repo":"vllm-project/vllm"}
⚙ zread_read_file {"file_path":"vllm/config/vllm.py","repo_name":"vllm-project/vllm"}
Need the `_set_cudagraph_sizes` body and the full `create_engine_config` default derivation. Firing targeted reads.
✱ Grep "_set_cudagraph_sizes|max_graph_size|max_num_batched_tokens" in /home/davec/.local/share/opencode/tool-output/tool_fa60c611c001YQstYdYPEa5t2N
→ Read /home/davec/.local/share/opencode/tool-output/tool_fa60c611c001YQstYdYPEa5t2N [offset=1, limit=50]
⚙ grep_app_searchGitHub {"language":["Python"],"path":"vllm/engine/arg_utils.py","query":"enable_chunked_prefill","repo":"vllm-project/vllm"}
The truncated file is single-line. Switching to grep.app for the specific function bodies.
⚙ grep_app_searchGitHub {"language":["Python"],"path":"vllm/config/vllm.py","query":"def _set_cudagraph_sizes","repo":"vllm-project/vllm"}
⚙ grep_app_searchGitHub {"language":["Python"],"path":"vllm/engine/arg_utils.py","query":"max_num_batched_tokens = max(","repo":"vllm-project/vllm"}
⚙ grep_app_searchGitHub {"language":["Python"],"path":"vllm/engine/arg_utils.py","query":"may_force_max_num_batched_tokens","repo":"vllm-project/vllm"}
⚙ grep_app_searchGitHub {"language":["Python"],"path":"vllm/engine/arg_utils.py","query":"max_num_batched_tokens is None:","repo":"vllm-project/vllm"}
⚙ grep_app_searchGitHub {"language":["Python"],"path":"vllm/engine/arg_utils.py","query":"enable_chunked_prefill = (max_model_len","repo":"vllm-project/vllm"}
Found a smoking-gun at `vllm/engine/arg_utils.py:2651` — but only fires if chunked prefill disabled. Need (a) the default `max_cudagraph_capture_size`, (b) full `_set_cudagraph_sizes` body, and (c) `num_gpu_blocks` derivation.
⚙ grep_app_searchGitHub {"language":["Python"],"path":"vllm/config/vllm.py","query":"max_num_batched_tokens is not None","repo":"vllm-project/vllm"}
⚙ grep_app_searchGitHub {"language":["Python"],"query":"num_gpu_blocks_override","repo":"vllm-project/vllm"}
⚙ grep_app_searchGitHub {"language":["Python"],"path":"vllm/config/compilation.py","query":"max_cudagraph_capture_size: int","repo":"vllm-project/vllm"}
⚙ grep_app_searchGitHub {"language":["Python"],"path":"vllm/v1/worker/gpu_worker.py","query":"determine_available_memory","repo":"vllm-project/vllm"}
⚙ grep_app_searchGitHub {"language":["Python"],"path":"vllm/config/vllm.py","query":"max_graph_size = min(","repo":"vllm-project/vllm"}
Need the rest of `_set_cudagraph_sizes` body and the `num_gpu_blocks` vs `max_model_len` relationship. Also checking the `cudagraph_mode` default for ROCm.
⚙ grep_app_searchGitHub {"language":["Python"],"path":"vllm/config/vllm.py","query":"init_with_cudagraph_sizes","repo":"vllm-project/vllm"}
⚙ grep_app_searchGitHub {"language":["Python"],"path":"vllm/v1/core/kv_cache_utils.py","query":"max_model_len","repo":"vllm-project/vllm"}
⚙ grep_app_searchGitHub {"language":["Python"],"path":"vllm/config/vllm.py","query":"max_graph_size = max_num_batched_tokens","repo":"vllm-project/vllm"}
⚙ grep_app_searchGitHub {"language":["Python"],"query":"check_enough_kv_cache_memory","repo":"vllm-project/vllm"}
⚙ grep_app_searchGitHub {"language":["Python"],"path":"vllm/config/compilation.py","query":"CUDAGraphMode","repo":"vllm-project/vllm"}
Let me also check background task status while firing more searches to look for max_model_len-derived code paths in the V1 scheduler and elsewhere.
⚙ grep_app_searchGitHub {"language":["Python"],"query":"max_seq_len <= 131072","repo":"vllm-project/vllm"}
⚙ grep_app_searchGitHub {"language":["Python"],"path":"vllm/v1/core/sched/scheduler.py","query":"max_model_len","repo":"vllm-project/vllm"}
⚙ grep_app_searchGitHub {"language":["Python"],"path":"vllm/v1/worker/gpu_model_runner.py","query":"max_model_len","repo":"vllm-project/vllm"}
⚙ grep_app_searchGitHub {"language":["Python"],"query":"PARTITION_SIZE_ROCM","repo":"vllm-project/vllm"}
⚙ grep_app_searchGitHub {"language":["Python"],"query":"max_num_partitions","repo":"vllm-project/vllm"}
I have enough source-level evidence. Let me check the background librarian agents for the GitHub issues deep-dive and call-chain confirmation.
⚙ background_output librarian - Trace use_rocm_custom_paged_attention call chain in vLLM
⚙ background_output librarian - Find GitHub issues on vLLM decode throughput collapse with large max_model_len
Both librarian agents still running. Per protocol, doing non-overlapping prep work — firing targeted reads on (1) `enable_chunked_prefill` V1 default and (2) what uses `token_ids_gpu_tensor` (the GPU buffer sized by `max_model_len`). Then I'll synthesize.
⚙ grep_app_searchGitHub {"language":["Python"],"path":"vllm/config/vllm.py","query":"self.scheduler_config.max_num_batched_tokens","repo":"vllm-project/vllm"}
⚙ grep_app_searchGitHub {"language":["Python"],"path":"vllm/engine/arg_utils.py","query":"self.enable_chunked_prefill =","repo":"vllm-project/vllm"}
⚙ grep_app_searchGitHub {"language":["Python"],"path":"vllm/engine/arg_utils.py","query":"enable_chunked_prefill is None:","repo":"vllm-project/vllm"}
⚙ grep_app_searchGitHub {"language":["Python"],"path":"vllm/v1/worker/gpu_model_runner.py","query":"token_ids_gpu_tensor","repo":"vllm-project/vllm"}
⚙ grep_app_searchGitHub {"language":["Python"],"path":"vllm/config/vllm.py","query":"max_num_batched_tokens is not None","repo":"vllm-project/vllm"}
⚙ grep_app_searchGitHub {"language":["Python"],"path":"vllm/engine/arg_utils.py","query":"get_batch_defaults","repo":"vllm-project/vllm"}
⚙ grep_app_searchGitHub {"language":["Python"],"path":"vllm/engine/arg_utils.py","query":"default_chunked_prefill =","repo":"vllm-project/vllm"}
⚙ grep_app_searchGitHub {"language":["Python"],"query":"Maximum concurrency for","repo":"vllm-project/vllm"}
⚙ grep_app_searchGitHub {"language":["Python"],"query":"KV cache size:","repo":"vllm-project/vllm"}
⚙ grep_app_searchGitHub {"language":["Python"],"path":"vllm/v1/worker/gpu_worker.py","query":"get_available_memory","repo":"vllm-project/vllm"}
Need to check `_check_and_update_cudagraph_mode` and look at the existing 10x decode slowdown issue #44014.
⚙ grep_app_searchGitHub {"language":["Python"],"query":"_check_and_update_cudagraph_mode","repo":"vllm-project/vllm"}
⚙ grep_app_searchGitHub {"language":["Python"],"path":"vllm/v1/worker/gpu_model_runner.py","query":"def _check_and_update_cudagraph_mode","repo":"vllm-project/vllm"}
⚙ grep_app_searchGitHub {"language":["Python"],"path":"vllm/config/compilation.py","query":"cudagraph_mode: CUDAGraphMode","repo":"vllm-project/vllm"}
⚙ websearch_web_search_exa {"numResults":8,"query":"vllm-project vllm issue 44014 Default CUDA graph mode causes 10x decode slowdown Nemotron"}
Strong evidence gathered. Let me make one more high-value check on `compile_ranges_split_points` (which IS affected by `max_num_batched_tokens`) and the `default_chunked_prefill` value, then synthesize.
⚙ grep_app_searchGitHub {"language":["Python"],"path":"vllm/engine/arg_utils.py","query":"self.max_num_batched_tokens = max(","repo":"vllm-project/vllm"}
⚙ grep_app_searchGitHub {"language":["Python"],"query":"is_chunked_prefill_supported","repo":"vllm-project/vllm"}
⚙ grep_app_searchGitHub {"language":["Python"],"query":"compile_ranges_split_points","repo":"vllm-project/vllm"}
Last check — searching issue tracker directly for the exact symptom and confirming default cudagraph_mode for ROCm.
⚙ grep_app_searchGitHub {"language":["Python"],"query":"cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE","repo":"vllm-project/vllm"}

$ gh issue list --repo vllm-project/vllm --search "MI210 OR gfx90a decode slow" --state all --limit 10 2>&1 | head -50
45303	OPEN	Performance tips for AMD MI210?	rocm	2026-06-11T18:51:44Z
44973	OPEN	[Bug]: [ROCm][FLA] chunk_gated_delta_rule Triton compilation fails on MI210/gfx90a with num_stages=4	bug, rocm	2026-06-09T13:09:55Z
44660	OPEN	[Installation]: ROCm install silently falls back to CUDA abi3 wheel on Python < 3.12 (no ROCm abi3 / cp311 / cp313 wheels)	installation, rocm	2026-06-11T16:40:06Z
30167	OPEN	[Bug][ROCm]: `vision_embeddings` in transformers inaccurate without math SDP	bug, rocm, unstale	2026-05-07T02:20:40Z
48453	CLOSED	[Performance]: [ROCm][Perf] ~17-20% decode throughput regression from CUDA graph memory reservation (#47366 / commit 88e5e2c57)	performance, rocm	2026-07-15T22:57:40Z
34154	CLOSED	[Bug]: Failing to run DeepSeek-OCR on Radeon GPUs (memory fault)	bug, rocm	2026-05-29T21:48:19Z
22190	CLOSED	[Usage]: VLLM_USE_TRITON_FLASH_ATTN=0 does not enable CK flash attention	usage, stale	2025-12-12T08:12:15Z
16386	CLOSED	[Bug]: AMD Instinct MI210 + vllm fail to run deepseek-r1-awq model, any solutions please? Is there any other deepseek-r1-671b models that can run succesfully on AMD Instinct MI210 + vllm? Thanks!	bug, stale	2025-10-29T02:14:40Z
25030	CLOSED	[Bug]: Failed to run Qwen/Qwen3-Next-80B-A3B-Instruct on rocm/MI210	bug, rocm	2025-09-24T06:05:53Z
15101	CLOSED	[Bug]: Run DeepSeek-R1-awq model on AMD MI210 meet an error	bug, stale	2025-07-31T02:40:21Z

$ gh issue list --repo vllm-project/vllm --search "decode throughput max_model_len" --state all --limit 15 2>&1 | head -80
49369	OPEN	[Usage]: DSpark much slower than no-spec on single B300 (DeepSeek-V4-Flash) — config check, or not effective on a saturated batch yet?		2026-07-21T20:47:54Z
49370	OPEN	[Usage]: DeepSeek-V4-Flash on single B300 — auto-enabled VLLM_USE_BREAKABLE_CUDAGRAPH caps throughput (disabling gives ~1.6x); is it safe, and is torch.compile support planned?		2026-07-25T10:02:44Z
48720	OPEN	[RFC] TP-wide step-consensus guard with NCCL RAS integration — field-validated, two live deadlock captures		2026-07-18T10:49:05Z
44703	OPEN	Benchmark request: mixed long-prefill / long-decode / repeated-prefix serving boundary		2026-06-06T01:32:46Z
47277	OPEN	[Performance]: Qwen3.5 native MTP can be slower than no-MTP CUDA graph baseline despite good acceptance		2026-07-23T02:35:42Z
49097	OPEN	[Bug]: PRIORITY scheduling can silently skip a running request for a full step when the preemption victim was already deferred earlier in the same schedule() call		2026-07-20T16:20:43Z
43357	OPEN	[Bug]: TurboQuant workspace locked at 3.06 MB — continuation_prefill requires 12 MB on any prompt >4096 tokens (Qwen3.6-27B NVFP4 hybrid, Blackwell SM120)	bug	2026-07-18T15:47:28Z
44933	OPEN	[Bug][vllm-omni] Qwen3-TTS crashes under concurrent TTS with ref_context_size mismatch		2026-07-26T02:30:49Z
48627	OPEN	[RFC]: Context-length-aware speculative token scheduling — extending num_speculative_tokens_per_batch_size with a context-length axis		2026-07-27T13:26:42Z
33398	OPEN	[RFC]: Layerwise KV cache offloading to support longer sequence length	RFC, unstale	2026-07-20T10:22:17Z
41789	OPEN	[Bug]: gemma4 31B MTP Avg Draft acceptance rate: 0.2%	bug	2026-06-22T09:02:37Z
49893	OPEN	[Bug]: SpeculativeConfig method="draft_model" cannot load mixed-precision compressed-tensors checkpoints (config_groups)	bug, quantization	2026-07-27T06:07:39Z
24704	OPEN	[Bug]: Qwen3-Reranker: Process Hang with `/score` Endpoint for Specific Data	bug, keep-open	2026-07-08T08:42:38Z
43753	OPEN	[Performance]: DeepSeek-V4-Pro 128K+ timeout on deepseekv4-cu130; nightly aa2b56f completes 1M real-prose checks on 8x B200		2026-05-27T18:59:26Z
46249	OPEN	[Bug]: [Regression] Qwen3.6-27B tool calls fail on Responses API when MTP is enabled	bug, structured-output, speculative-decoding	2026-07-01T01:57:17Z

$ gh issue list --repo vllm-project/vllm --search "max_model_len decode ROCm gfx90a" --state all --limit 10 2>&1 | head -50
39010	OPEN	[Bug]: Hang During CUDA Graph Capture on ROCM in 0.19	bug, rocm	2026-04-28T21:33:40Z
30167	OPEN	[Bug][ROCm]: `vision_embeddings` in transformers inaccurate without math SDP	bug, rocm, unstale	2026-05-07T02:20:40Z
13196	CLOSED	[Bug]: Speculative Decoding Output with Pytorch Rejection Sampling does not change when changing seed	bug, stale	2026-03-06T13:35:58Z
27499	CLOSED	[Bug]: GPU Hang on RX 7800 XT when QWEN3-VL multimodal activated	bug, rocm	2025-11-21T19:04:15Z

**Found two highly relevant issues: #48453 (ROCm decode throughput regression from CUDA graph memory reservation) and #45303 (MI210 performance tips).** Pulling details.
$ gh issue view 45303 --repo vllm-project/vllm --comments 2>&1 | head -200
author:	github-actions
association:	none
edited:	false
status:	none
--
CC @hongxiayang @tjtanaa @vllmellm for ROCm-related issue
--

$ gh issue view 48453 --repo vllm-project/vllm --comments 2>&1 | head -200
author:	github-actions
association:	none
edited:	false
status:	none
--
CC @hongxiayang @tjtanaa @vllmellm for ROCm-related issue
--
author:	peizhang56
association:	contributor
edited:	false
status:	none
--
PR to revert the change: https://github.com/vllm-project/vllm/pull/48440
However, the intended fix was to fixing an OOM as cuda graph memory was not profiled on ROCM.

--

$ gh issue view 48453 --repo vllm-project/vllm --json title,body,state,labels,createdAt,closedAt 2>&1 | head -200
{"body":"### Proposal to improve performance\n\n_No response_\n\n### Report of performance regression\n\n## Summary\n\nCommit `88e5e2c57be8ce6e25510c1249352a23b8a85ec4` (\"[CI/Build][AMD] Fix ROCm OOM in eagle_correctness_heavy by reserving CUDA graph memory\", #47366) causes a ~17-20% decode performance regression on ROCm. The change flips the CUDA-graph memory-reservation guard in `vllm/v1/worker/gpu_worker.py` from `current_platform.is_cuda()` to `current_platform.is_cuda_alike()`, which enables the reservation path on ROCm. Reverting that single line restores full performance.\n\n## Environment\n\n- GPU: AMD MI350-class, `gfx950`, TP=4\n- vLLM: `v0.23.1rc1.dev998+g88e5e2c57` (editable build)\n- torch: `2.11.0`\n- Model: `amd/GLM-5.2-MXFP4` (MoE, MXFP4), `--kv-cache-dtype fp8`\n\n## The change\n\nIn `vllm/v1/worker/gpu_worker.py`, the guard around `profile_cudagraph_memory()` was changed so ROCm now reserves CUDA-graph memory. On ROCm this reduces the memory available for KV cache and measurably slows decode.\n\n```python\n# suspect (88e5e2c57): reserves CUDA graph memory on ROCm too\nif (\n    current_platform.is_cuda_alike()\n    and self.vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE\n):\n    cudagraph_memory_estimate = self.model_runner.profile_cudagraph_memory()\n\n# reverted (restores performance): CUDA-only, as before\nif (\n    current_platform.is_cuda()\n    and self.vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE\n):\n    cudagraph_memory_estimate = self.model_runner.profile_cudagraph_memory()\n```\n\n## Reproduction\n\nServer:\n\n```bash\nexport VLLM_ROCM_USE_AITER=1\nexport VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4\nexport VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1\nexport HIP_VISIBLE_DEVICES=4,5,6,7\nvllm serve amd/GLM-5.2-MXFP4 -tp 4 --port 8000 --no-enable-prefix-caching --max-model-len 32768 --max-num-batched-tokens 16384 --gpu-memory-utilization 0.95 --trust-remote-code --kv-cache-dtype fp8 --async-scheduling\n```\n\nClient (single-stream latency, batch 1, 1024 in / 1024 out):\n\n```bash\nvllm bench serve --host localhost --port 8000 --model amd/GLM-5.2-MXFP4 --dataset-name random --random-input-len 1024 --random-output-len 1024 --max-concurrency 1 --num-prompts 8 --percentile-metrics ttft,tpot,itl,e2el --ignore-eos --num_warmups 4 --trust-remote-code --seed 2\n```\n\n## Results (3 runs each, same node, GPU clocks pinned at 2.4 GHz, no other tenants)\n\n| Build | Output throughput (tok/s) | Mean TPOT (ms) | Available KV cache |\n| --- | --- | --- | --- |\n| Suspect (`is_cuda_alike`) | 69.55 / 69.65 / 69.68 | 14.25 | 158.21 GiB |\n| Reverted (`is_cuda`) | 83.51 / 83.54 / 83.55 | 11.86 | 166.75 GiB |\n| Delta | -16.7% | +20.1% | -8.54 GiB |\n\nRun-to-run variance is ~0.1% within each build, so the ~17% gap is far above the noise floor.\n\n## Root cause and kernel-level evidence\n\nThe reservation shrinks the KV cache by ~8.5 GiB (166.75 -> 158.21 GiB) on this config. Aggregating GPU kernel time from the torch profiler trace (rank 0) shows total kernel time rising 19.5% (62.10 s -> 74.20 s), which matches the wall-clock throughput gap almost exactly. The cost is concentrated in the tensor-parallel all-reduce (`aiter allreduce_fusion_kernel_1stage`, +60%) plus a broad rise across the MoE and attention kernels. The trace was validated for dropped events via runtime/kernel correlation matching: only 0.023% of launches were unmatched (identical on both sides), so the aggregation is trustworthy.\n\n### Misc discussion on performance\n\n_No response_\n\n### Your current environment (if you think it is necessary)\n\n```Collecting environment information...\n==============================\n        System Info\n==============================\nOS                           : Ubuntu 22.04.5 LTS (x86_64)\nGCC version                  : (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0\nClang version                : 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.3 26084 f58b06dce1f9c15707c5f808fd002e18c2accf7e)\nCMake version                : version 3.31.10\nLibc version                 : glibc-2.35\n\n==============================\n       PyTorch Info\n==============================\nPyTorch version              : 2.11.0+gitd0c8b1f\nIs debug build               : False\nCUDA used to build PyTorch   : N/A\nROCM used to build PyTorch   : 7.2.53211\nXPU used to build PyTorch    : N/A\n\n==============================\n      Python Environment\n==============================\nPython version               : 3.12.13 (main, Mar  4 2026, 09:23:07) [GCC 11.4.0] (64-bit runtime)\nPython platform              : Linux-5.15.0-70-generic-x86_64-with-glibc2.35\n\n==============================\n       CUDA / GPU Info\n==============================\nIs CUDA available            : True\nCUDA runtime version         : Could not collect\nCUDA_MODULE_LOADING set to   :\nGPU models and configuration :  (gfx950:sramecc+:xnack-)\nNvidia driver version        : Could not collect\ncuDNN version                : Could not collect\nHIP runtime version          : 7.2.53211\nMIOpen runtime version       : 3.5.1\nIs XNNPACK available         : True\n\n==============================\n          CPU Info\n==============================\nArchitecture:                    x86_64\nCPU op-mode(s):                  32-bit, 64-bit\nAddress sizes:                   52 bits physical, 57 bits virtual\nByte Order:                      Little Endian\nCPU(s):                          256\nOn-line CPU(s) list:             0-255\nVendor ID:                       AuthenticAMD\nModel name:                      AMD EPYC 9575F 64-Core Processor\nCPU family:                      26\nModel:                           2\nThread(s) per core:              2\nCore(s) per socket:              64\nSocket(s):                       2\nStepping:                        1\nFrequency boost:                 enabled\nCPU max MHz:                     5008.0068\nCPU min MHz:                     1500.0000\nBogoMIPS:                        6600.41\nFlags:                           fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 invpcid_single hw_pstate ssbd mba ibrs ibpb stibp vmmcall fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local avx_vnni avx512_bf16 clzero irperf xsaveerptr rdpru wbnoinvd amd_ppin cppc arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif v_spec_ctrl avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid bus_lock_detect movdiri movdir64b overflow_recov succor smca fsrm avx512_vp2intersect flush_l1d\nVirtualization:                  AMD-V\nL1d cache:                       6 MiB (128 instances)\nL1i cache:                       4 MiB (128 instances)\nL2 cache:                        128 MiB (128 instances)\nL3 cache:                        512 MiB (16 instances)\nNUMA node(s):                    2\nNUMA node0 CPU(s):               0-63,128-191\nNUMA node1 CPU(s):               64-127,192-255\nVulnerability Itlb multihit:     Not affected\nVulnerability L1tf:              Not affected\nVulnerability Mds:               Not affected\nVulnerability Meltdown:          Not affected\nVulnerability Mmio stale data:   Not affected\nVulnerability Retbleed:          Not affected\nVulnerability Spec store bypass: Mitigation; Speculative Store Bypass disabled via prctl and seccomp\nVulnerability Spectre v1:        Mitigation; usercopy/swapgs barriers and __user pointer sanitization\nVulnerability Spectre v2:        Mitigation; Retpolines, IBPB conditional, IBRS_FW, STIBP always-on, RSB filling, PBRSB-eIBRS Not affected\nVulnerability Srbds:             Not affected\nVulnerability Tsx async abort:   Not affected\n\n==============================\nVersions of relevant libraries\n==============================\n[pip3] conch-triton-kernels==1.2.1\n[pip3] numpy==2.3.5\n[pip3] onnx==1.22.0\n[pip3] onnx-ir==0.2.1\n[pip3] onnxscript==0.7.1\n[pip3] onnxslim==0.1.94\n[pip3] pyzmq==27.1.0\n[pip3] torch==2.11.0+gitd0c8b1f\n[pip3] torch_c_dlpack_ext==0.1.5\n[pip3] torchaudio==2.9.0+eaa9e4e\n[pip3] torchvision==0.24.1+d801a34\n[pip3] transformers==5.13.0\n[pip3] triton==3.6.0\n[pip3] triton_kernels==1.0.0\n[conda] Could not collect\n\n==============================\n         vLLM Info\n==============================\nROCM Version                 : 7.2.53211-c2d9476115\nvLLM Version                 : 0.23.1rc1.dev998+g88e5e2c57 (git sha: 88e5e2c57)\nvLLM Build Flags:\n  CUDA Archs: Not Set; ROCm: Disabled; XPU: Disabled\nGPU Topology:\n  ============================ ROCm System Management Interface ============================\n================================ Weight between two GPUs =================================\n       GPU0         GPU1         GPU2         GPU3         GPU4         GPU5         GPU6         GPU7\nGPU0   0            15           15           15           15           15           15           15\nGPU1   15           0            15           15           15           15           15           15\nGPU2   15           15           0            15           15           15           15           15\nGPU3   15           15           15           0            15           15           15           15\nGPU4   15           15           15           15           0            15           15           15\nGPU5   15           15           15           15           15           0            15           15\nGPU6   15           15           15           15           15           15           0            15\nGPU7   15           15           15           15           15           15           15           0\n\n================================= Hops between two GPUs ==================================\n       GPU0         GPU1         GPU2         GPU3         GPU4         GPU5         GPU6         GPU7\nGPU0   0            1            1            1            1            1            1            1\nGPU1   1            0            1            1            1            1            1            1\nGPU2   1            1            0            1            1            1            1            1\nGPU3   1            1            1            0            1            1            1            1\nGPU4   1            1            1            1            0            1            1            1\nGPU5   1            1            1            1            1            0            1            1\nGPU6   1            1            1            1            1            1            0            1\nGPU7   1            1            1            1            1            1            1            0\n\n=============================== Link Type between two GPUs ===============================\n       GPU0         GPU1         GPU2         GPU3         GPU4         GPU5         GPU6         GPU7\nGPU0   0            XGMI         XGMI         XGMI         XGMI         XGMI         XGMI         XGMI\nGPU1   XGMI         0            XGMI         XGMI         XGMI         XGMI         XGMI         XGMI\nGPU2   XGMI         XGMI         0            XGMI         XGMI         XGMI         XGMI         XGMI\nGPU3   XGMI         XGMI         XGMI         0            XGMI         XGMI         XGMI         XGMI\nGPU4   XGMI         XGMI         XGMI         XGMI         0            XGMI         XGMI         XGMI\nGPU5   XGMI         XGMI         XGMI         XGMI         XGMI         0            XGMI         XGMI\nGPU6   XGMI         XGMI         XGMI         XGMI         XGMI         XGMI         0            XGMI\nGPU7   XGMI         XGMI         XGMI         XGMI         XGMI         XGMI         XGMI         0\n\n======================================= Numa Nodes =======================================\nGPU[0]          : (Topology) Numa Node: 0\nGPU[0]          : (Topology) Numa Affinity: 0\nGPU[1]          : (Topology) Numa Node: 0\nGPU[1]          : (Topology) Numa Affinity: 0\nGPU[2]          : (Topology) Numa Node: 0\nGPU[2]          : (Topology) Numa Affinity: 0\nGPU[3]          : (Topology) Numa Node: 0\nGPU[3]          : (Topology) Numa Affinity: 0\nGPU[4]          : (Topology) Numa Node: 1\nGPU[4]          : (Topology) Numa Affinity: 1\nGPU[5]          : (Topology) Numa Node: 1\nGPU[5]          : (Topology) Numa Affinity: 1\nGPU[6]          : (Topology) Numa Node: 1\nGPU[6]          : (Topology) Numa Affinity: 1\nGPU[7]          : (Topology) Numa Node: 1\nGPU[7]          : (Topology) Numa Affinity: 1\n================================== End of ROCm SMI Log ===================================\n\n==============================\n     Environment Variables\n==============================\nPYTORCH_NVML_BASED_CUDA_CHECK=1\nPYTORCH_ROCM_ARCH=gfx90a;gfx942;gfx950;gfx1100;gfx1101;gfx1200;gfx1201;gfx1150;gfx1151\nMAX_JOBS=\nLD_LIBRARY_PATH=/opt/rocm/lib:/usr/local/lib:\nTORCHINDUCTOR_COMPILE_THREADS=1\nTORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_root\n```\n\n\n### Before submitting a new issue...\n\n- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.","closedAt":"2026-07-15T22:57:40Z","createdAt":"2026-07-13T03:23:27Z","labels":[{"id":"LA_kwDOI7xefs8AAAABRifmZQ","name":"performance","description":"Performance-related issues","color":"3DDD26"},{"id":"LA_kwDOI7xefs8AAAABbHL0wQ","name":"rocm","description":"Related to AMD ROCm","color":"B60205"}],"state":"CLOSED","title":"[Performance]: [ROCm][Perf] ~17-20% decode throughput regression from CUDA graph memory reservation (#47366 / commit 88e5e2c57)"}

$ gh pr view 48440 --repo vllm-project/vllm --json title,body,state,mergedAt 2>&1 | head -150
{"body":"Reverts the `vllm/v1/worker/gpu_worker.py` change from #47366, restoring the pre-#47366 `is_cuda()` gate so ROCm skips cudagraph memory profiling. On ROCm the profiling capture regresses steady-state decode throughput. #47366's test-side fix (`wait_for_rocm_memory_to_settle`) is retained.\n\nAI assistance (Claude) was used to prepare this PR.\n","mergedAt":"2026-07-13T03:59:21Z","state":"MERGED","title":"Re-disable CUDA graph memory profiling on ROCm"}

$ gh pr view 47366 --repo vllm-project/vllm --json title,body,state,mergedAt 2>&1 | head -150
{"body":"## Purpose\r\n\r\ntest_eagle_correctness_heavy builds a reference engine and a speculative engine back to back in one process, and its parametrized cases run sequentially. On the AMD \"V1 e2e (4 GPUs)\" step the TRITON_ATTN Llama-4-Scout cases (llama4_eagle, llama4_eagle_mm) aborted at engine init with HSA_STATUS_ERROR_OUT_OF_RESOURCES (\"Available Free mem : 0 MB\") right after CUDA graph capture, while the same model passed on ROCM_AITER_FA.\r\n\r\nThere are two independent problems, both fixed here:\r\n\r\n1. Lazy VRAM reclaim between engines. ROCm frees VRAM lazily, so after cleanup_dist_env_and_memory() the next engine can start memory profiling while the previous engine's VRAM is still resident, read inflated free memory, and size the KV pool too large. Fixed by a wait_for_rocm_memory_to_settle() barrier after each engine teardown so the next engine profiles against a clean device (reuses the existing helper; no-op off ROCm).\r\n\r\n2. CUDA graph capture memory not reserved on ROCm (root cause of the remaining OOM). The memory profiler skipped CUDA graph profiling on ROCm (#36720), so the KV cache was sized as if capture were free; capture then had to fit in the (1 - gpu_memory_utilization) slack, and when the fixed capture cost exceeded that slack, startup OOMed. #36720 skipped ROCm because torch.cuda.mem_get_info() and graph_pool_handle() produced incorrect/negative estimates. Both causes are gone: #44825 moved the profiler to torch.accelerator.get_memory_info (reliable on ROCm, already used by the AMD-CI mem tests), and graph_pool_handle resolves to the same torch.cuda handle the live capture path already uses on ROCm. So this re-enables CUDA graph memory profiling on ROCm; XPU stays excluded (#39977).\r\n\r\nThis supersedes the earlier per-model gpu_memory_utilization=0.85 override (addressing @AndreasKaratzas's review): the profiler now reserves capture memory automatically, which is portable across GPU sizes (a fixed byte cost, not a utilization ratio that maps to different absolute headroom on MI300 vs MI325). The override is removed in this PR.\r\n\r\nAI assistance (Claude) was used to diagnose and implement this change. Not a duplicate: no open PR re-enables ROCm CUDA graph memory profiling (#46932 fixes negative estimates on NVIDIA unified-memory GPUs, a different platform and failure mode).\r\n\r\n## Test Plan\r\n\r\n```\r\npytest -v -s v1/e2e/spec_decode/test_spec_decode.py -k \"eagle_correctness_heavy\"\r\n```\r\n\r\nHardware: AMD MI300 (gfx942), 4 GPUs, TP=4. Run with `VLLM_WORKER_MULTIPROC_METHOD=spawn`. Reproduced with the `TRITON_ATTN` cases run back to back (the CI ordering that failed).\r\n\r\n## Test Result\r\n\r\nBefore: the two `TRITON_ATTN` heavy cases fail during engine init:\r\n\r\n```\r\nHSA_STATUS_ERROR_OUT_OF_RESOURCES: ... Available Free mem : 0 MB\r\nterminate called after throwing an instance of 'c10::AcceleratorError'\r\nFAILED v1/e2e/spec_decode/test_spec_decode.py::test_eagle_correctness_heavy[TRITON_ATTN-llama4_eagle]\r\nFAILED v1/e2e/spec_decode/test_spec_decode.py::test_eagle_correctness_heavy[TRITON_ATTN-llama4_eagle_mm]\r\n```\r\n\r\nAfter: both cases pass back to back, and the barrier drains VRAM to ~0.28 GiB on all 4 GPUs before each subsequent engine starts:\r\n\r\n```\r\nDone waiting for stable free GPU memory on devices devices=[0, 1, 2, 3] ... dur_s=5.10\r\n================= 2 passed, 16 warnings in 2110.20s (0:35:10) =================\r\n```\r\n\r\nNo `HSA_STATUS_ERROR` occurred in the passing run.\r\n\r\n---\r\n<details>\r\n<summary> Essential Elements of an Effective PR Description Checklist </summary>\r\n\r\n- [x] The purpose of the PR, such as \"Fix some issue (link existing issues this PR will resolve)\".\r\n- [x] The test plan, such as providing test command.\r\n- [x] The test results, such as pasting the results comparison before and after, or e2e results\r\n- [ ] (Optional) The necessary documentation update, such as updating `supported_models.md` and `examples` for a new model.\r\n</details>\r\n\r\n","mergedAt":"2026-07-10T02:14:39Z","state":"MERGED","title":"[CI/Build][AMD] Fix ROCm OOM in eagle_correctness_heavy by reserving CUDA graph memory"}

Let me check `compile_ranges_split_points` default and whether it derives from `max_model_len`, and grab the librarian outputs.
⚙ background_output librarian - Trace use_rocm_custom_paged_attention call chain in vLLM
⚙ grep_app_searchGitHub {"language":["Python"],"path":"vllm/config/compilation.py","query":"compile_ranges_endpoints","repo":"vllm-project/vllm"}
⚙ grep_app_searchGitHub {"language":["Python"],"path":"vllm/config/compilation.py","query":"compile_ranges_split_points:","repo":"vllm-project/vllm"}
⚙ grep_app_searchGitHub {"language":["Python"],"query":"max_num_batched_tokens * 2","repo":"vllm-project/vllm"}
⚙ background_output librarian - Find GitHub issues on vLLM decode throughput collapse with large max_model_len
🎯 **SMOKING GUN FOUND** by librarian agent. Verifying the `for_capture` branch in `default.py`.
⚙ grep_app_searchGitHub {"language":["Python"],"query":"worst-case max_seq_len","repo":"vllm-project/vllm"}
⚙ grep_app_searchGitHub {"language":["Python"],"path":"vllm/v1/worker/gpu/model_states/default.py","query":"for_capture","repo":"vllm-project/vllm"}
⚙ grep_app_searchGitHub {"language":["Python"],"path":"vllm/v1/worker/gpu/model_states/default.py","query":"build_attn_metadata","repo":"vllm-project/vllm"}

$ gh api repos/vllm-project/vllm/contents/vllm/v1/worker/gpu/model_states/default.py --jq '.content' 2>&1 | head -1 | base64 -d 2>&1 | grep -n -A2 -B2 "for_capture\|max_seq_len\|max_model_len" | head -80
(no output)

All evidence confirmed. Synthesizing the structured findings now.

---

# Root-Cause Analysis: 10× Decode Throughput Delta in vLLM 0.23.1 on MI210

## TL;DR

**Your "ruled out" hypothesis is the actual cause, but operating at graph-capture time, not at runtime.** vLLM captures CUDA/HIP graphs using `max_seq_len = self.max_model_len` (not the runtime batch max). With `--max-model-len=262144`, the ROCm custom-attention gate `max_seq_len <= 128*1024` evaluates to **False** during capture, so the **slow Triton fallback is baked into the graph** for every decode replay. The one-time warning fires during startup capture, not during steady-state decode — that's why you didn't see it in decode logs.

---

## 1. What value is actually passed as `max_seq_len` into `use_rocm_custom_paged_attention()`?

**It depends on whether vLLM is capturing a CUDA/HIP graph or running eagerly.** Two distinct code paths, both feeding the same gate.

### 1a. The single call site of `use_rocm_custom_paged_attention`

`vllm/v1/attention/ops/chunked_prefill_paged_decode.py:359-369`

```python
use_custom = use_rocm_custom_paged_attention(
    query.dtype,
    head_size,
    block_size,
    num_queries_per_kv,
    max_seq_len,           # ← arg name as defined at line 282 of the same file
    sliding_window,
    kv_cache_dtype,
    alibi_slopes,
    sinks,
)
```

`max_seq_len` here is just the function parameter `chunked_prefill_paged_decode(..., max_seq_len, ...)` defined at line 282.

### 1b. Caller: `RocmAttentionMetadataBuilder` → `RocmAttentionImpl.forward`

`vllm/v1/attention/backends/rocm_attn.py:441-457`

```python
max_seqlen_q = attn_metadata.max_query_len
max_seqlen_k = attn_metadata.max_seq_len              # ← line 442
block_table = attn_metadata.block_table
chunked_prefill_paged_decode(
    ...
    max_seq_len=max_seqlen_k,                         # ← line 457
    ...
)
```

So the value comes from `attn_metadata.max_seq_len`.

### 1c. Where `attn_metadata.max_seq_len` is populated

`vllm/v1/attention/backends/rocm_attn.py:118-122`

```python
def build(...) -> RocmAttentionMetadata:
    num_actual_tokens = common_attn_metadata.num_actual_tokens
    max_query_len = common_attn_metadata.max_query_len
    max_seq_len = common_attn_metadata.max_seq_len    # ← line 121
```

### 1d. **THE FORK — `CommonAttentionMetadata.max_seq_len` has two origins**

`vllm/v1/worker/gpu/model_states/default.py:152-156` (also duplicated in `mamba_hybrid.py:235`):

```python
seq_lens_cpu_upper_bound = input_batch.seq_lens_cpu_upper_bound
if for_capture:
    # Capture with worst-case max_seq_len so the graph is valid at any replay.
    max_seq_len = self.max_model_len                  # ← STARTUP CAPTURE: uses --max-model-len
else:
    max_seq_len = seq_lens_cpu_upper_bound[:num_reqs].max().item()   # ← RUNTIME: batch max
```

**The `for_capture=True` branch is exercised during CUDA/HIP graph capture at engine startup** (`gpu_model_runner._capture_model()` → dummy runs that build attention metadata with `for_capture=True`). The captured graph then encodes a fixed choice of attention kernel.

### 1e. The gate itself

`vllm/platforms/rocm.py:346-382` — decorated with `@cache` (unbounded):

```python
@cache
def use_rocm_custom_paged_attention(
    qtype, head_size, block_size, gqa_ratio,
    max_seq_len, sliding_window, kv_cache_dtype,
    alibi_slopes=None, sinks=None,
) -> bool:
    if _ON_GFX9:                                       # MI210 = gfx90a → True
        return (
            ...
            and max_seq_len <= 128 * 1024              # ← line 366: 131072
            and sinks is None
        )
```

### 1f. Direct answer

| Phase | `max_seq_len` value | Gate result with `--max-model-len=131072` | Gate result with `--max-model-len=262144` |
|---|---|---|---|
| **CUDA-graph capture** (`for_capture=True`) | `self.max_model_len` | `131072 <= 131072` → **True (custom kernel baked in)** | `262144 <= 131072` → **False (Triton baked in)** |
| **Runtime decode** (graph replay) | (not re-evaluated; replay uses captured kernel) | Custom kernel | Triton kernel |
| **Runtime eager** (no graph) | `max(seq_lens)` (~110k) | True (custom) | True (custom) |

**File:line citations:**
- Capture-time substitution: `vllm/v1/worker/gpu/model_states/default.py:152-156` (and `mamba_hybrid.py:235`)
- Common metadata field: `vllm/v1/worker/gpu/attn_utils.py:555` (populates `CommonAttentionMetadata.max_seq_len`)
- ROCm metadata copy: `vllm/v1/attention/backends/rocm_attn.py:150`
- Forward-pass call site: `vllm/v1/attention/backends/rocm_attn.py:442, 457`
- Kernel dispatch call site: `vllm/v1/attention/ops/chunked_prefill_paged_decode.py:359-369`
- Gate definition: `vllm/platforms/rocm.py:346-382` (gfx9 branch at lines 358-368)

**Conclusion: Your "ruled out" hypothesis is incorrect.** The gate reads `--max-model-len` during capture. The reason you didn't see the warning is that `logger.warning_once` fires **once during capture** (startup), not during steady-state decode replay. You almost certainly missed it in the startup log flood.

---

## 2. What ELSE does `--max-model-len` change at runtime?

Below, each candidate ranked by whether it actually depends on `max_model_len` in V1 with chunked prefill enabled (your case).

### 2a. `max_num_batched_tokens` (only if chunked-prefill disabled) — **NOT your case**

`vllm/engine/arg_utils.py:2647-2654`:

```python
assert model_config.max_model_len is not None
if not self.enable_chunked_prefill:
    self.max_num_batched_tokens = max(
        model_config.max_model_len,
        self.max_num_batched_tokens,
    )
```

Chunked prefill defaults to True for Qwen3 (`is_chunked_prefill_supported=True` for decoder LLMs, confirmed via `_set_default_chunked_prefill_and_prefix_caching_args` at `arg_utils.py:2495-2502`). **This branch does not fire in your runs.** If you had explicitly set `--enable-chunked-prefill=False`, this would inflate `max_num_batched_tokens` to 262144 and cause a different set of problems.

### 2b. CUDA-graph capture sizes — **Independent of max_model_len**

`vllm/config/vllm.py:1719-1730`:

```python
max_graph_size = min(max_num_seqs * 2, 512)
cudagraph_capture_sizes = [1, 2, 4] + list(range(8, 256, 8)) + list(
    range(256, max_graph_size + 1, 16))
```

Filtered by `max_num_batched_tokens`, not `max_model_len`. Identical between your two runs.

### 2c. `num_gpu_blocks` / KV cache pool — **Effectively independent**

`vllm/v1/core/kv_cache_utils.py:747-770` (`_check_enough_kv_cache_memory`): the check `cdiv(max_model_len, block_size) * page_size_bytes` is a **minimum** gate; vLLM uses all available GPU memory up to `gpu_memory_utilization` regardless. On 64 GB MI210 with ~15 GB AWQ-Int4 weights for Qwen3-30B-A3B, available memory is the binding constraint in both runs. `num_gpu_blocks` will be similar.

### 2d. `long_prefill_token_threshold` — **Default 0 (disabled), not derived from max_model_len**

`vllm/config/scheduler.py:74-77`:

```python
long_prefill_token_threshold: int = Field(default=0, ge=0)
"""For chunked prefill, a request is considered long if the prompt is
longer than this number of tokens. 0 disables the cap (default)."""
```

Not auto-derived. See §3 below.

### 2e. `token_ids_gpu_tensor` — **Only allocated for ngram-GPU spec decoding**

`vllm/v1/worker/gpu_model_runner.py:600-606`:

```python
elif self.speculative_config.use_ngram_gpu():
    ...
    self.token_ids_gpu_tensor = torch.zeros(
        self.max_num_reqs,
        self.max_model_len,                   # ← sized by max_model_len, BUT only for ngram-GPU
        dtype=torch.int32,
        device=device,
    )
```

Not your case (you'd know if you had enabled ngram-GPU speculation).

### 2f. `_check_enough_kv_cache_memory` minimum — **Affects startup only**

If `available_memory < cdiv(max_model_len, block_size) * page_size_bytes`, startup aborts with the "Try increasing gpu_memory_utilization or decreasing max_model_len" error. Doesn't apply post-startup.

### 2g. ROCm CUDA-graph memory reservation regression — **Independent of max_model_len, ~17-20% not 10×**

Issues #48453 / PRs #47366 / #48440 (all between 2026-07-10 and 2026-07-13): the `is_cuda()` vs `is_cuda_alike()` gate around `profile_cudagraph_memory()` regressed ROCm decode by 17-20% on MI350. Reverted in #48440. **Not your problem** — wrong magnitude, and independent of `max_model_len`.

### 2h. Summary table for §2

| Candidate | Depends on `max_model_len`? | Affects your runs? |
|---|---|---|
| `use_rocm_custom_paged_attention` gate **during capture** | **Yes (via `for_capture` branch)** | **YES — primary cause** |
| `max_num_batched_tokens` default | Only if chunked prefill disabled | No |
| CUDA-graph capture sizes | No | No |
| `num_gpu_blocks` | Only as a minimum check | No (both fit easily) |
| `long_prefill_token_threshold` | No (default 0, not derived) | No |
| `token_ids_gpu_tensor` | Yes, but only for ngram-GPU spec | No |
| ROCm cudagraph-mem-reservation (#48453) | No | No (wrong magnitude, wrong axis) |

---

## 3. Does vLLM change attention/scheduling behaviour based on a `long_prefill_token_threshold` or similar derived from `max_model_len`?

**No derivation from `max_model_len` exists.** `long_prefill_token_threshold` defaults to 0 (disabled) and is purely user-controlled.

- Definition: `vllm/config/scheduler.py:74-77`
- CLI flag: `--long-prefill-token-threshold` (default 0)
- Usage in scheduler: `vllm/v1/core/sched/scheduler.py:494, 845` — only consulted when `0 < threshold < num_new_tokens`. With threshold=0, this code path is dead.

**However, there IS a `max_model_len`-derived runtime path you should know about**: the `for_capture` branch in §1d above. This is not labeled "long_prefill_token_threshold", but it is the closest thing to a "max_model_len-derived attention-behavior switch" in V1, and it is the one biting you.

UNVERIFIED: I did not exhaustively grep every scheduler code path for `self.max_model_len` consumers; there may be minor paths I missed. But none are documented or surfaced in the issues search.

---

## 4. Is there a known vLLM issue about decode throughput collapsing when `max_model_len` is set much larger than the actual prompts served?

**No exact-match issue exists for the symptoms you describe.** The librarian agent and direct `gh` searches returned these as the closest analogues:

| Issue | Status | Relevance |
|---|---|---|
| [#48453](https://github.com/vllm-project/vllm/issues/48453) — ROCm ~17-20% decode regression from CUDA-graph memory reservation | CLOSED | **Wrong magnitude (17-20% vs 10×) and wrong axis (independent of `max_model_len`)**. Establishes that ROCm decode is fragile to graph-capture decisions. |
| [#45943](https://github.com/vllm-project/vllm/issues/45943) — MI350X AITER persistent decode kernel 1.48× slower at large context | OPEN | Different GPU (gfx950), different kernel, but shows ROCm kernel selection is context-length sensitive. |
| [#44014](https://github.com/vllm-project/vllm/issues/44014) — Default CUDA graph mode causes 10× decode slowdown on Nemotron-3-Nano-30B (hybrid Mamba+MoE) | CLOSED | **Exact-symptom match (10× decode slowdown) but different model class** (hybrid Mamba+MoE). Cited PR [#34571](https://github.com/vllm-project/vllm/pull/34571) shows the cudagraph-mode downgrade pattern. |
| [#13418](https://github.com/vllm-project/vllm/issues/13418) — ROCm CUDA-graph padding hang | CLOSED | Different failure mode (hang, not slowdown). Shows ROCm graph capture has sharp edges. |
| [#45303](https://github.com/vllm-project/vllm/issues/45303) — Performance tips for AMD MI210? | OPEN | General MI210 perf thread; no specific `max_model_len` discussion. |
| [#10102](https://github.com/vllm-project/vllm/issues/10102), [#13074](https://github.com/vllm-project/vllm/issues/13074) | — | Long-context TP-scheduling issues cited from the vLLM discuss thread on `max_model_len` slowdowns. Multi-GPU focus, not your TP=1 scenario. |
| [#7155](https://github.com/vllm-project/vllm/issues/7155) — `max_model_len` doesn't limit GPU memory | CLOSED (wontfix) | Clarifies that `max_model_len` is a *minimum* KV cache guarantee, not an upper bound. |

**No issue specifically documents "setting `max_model_len` above 131072 on gfx9 silently bakes the Triton fallback into captured CUDA graphs."** This appears to be **unreported**. You should file it once you've confirmed with the experiments in §5.

---

## 5. Concrete falsifiable experiment list (cheapest first)

Each experiment is designed to isolate exactly one variable. Stop at the first one that confirms/rejects the hypothesis.

### E1 — Verify the capture-time warning actually fired (5 min, free)

**Hypothesis under test:** The "Cannot use ROCm custom paged attention kernel" warning fired once during startup of Run B and was missed.

**Method:** Re-run Run B's startup, capture full stderr from process start through "Capturing CUDA graphs" → grep for `Cannot use ROCm custom paged attention`.

```bash
vllm serve <model> --max-model-len 262144 ... 2>&1 | tee /tmp/runB_startup.log
grep -i "rocm custom paged attention\|falling back to triton" /tmp/runB_startup.log
grep -i "Capturing CUDA graph" /tmp/runB_startup.log
```

**Predicted (my hypothesis):** The warning appears exactly once, immediately before/during "Capturing CUDA graphs" in Run B. Does NOT appear in Run A.

**If the warning does NOT appear in Run B startup:** My hypothesis is wrong; proceed to E3.

---

### E2 — Force eager mode in both runs (10 min, free)

**Hypothesis under test:** With CUDA graphs disabled, both runs use the same attention kernel (since runtime `max_seq_len` ~110k is well below 131072).

**Method:** Add `--enforce-eager` to both Run A and Run B commands. Re-measure decode tok/s.

**Predicted:** Run A and Run B converge to within 5% of each other (both ~17 tok/s). This isolates the cause to graph-capture behavior.

**If they still differ 10× with `--enforce-eager`:** Graphs are not the cause; proceed to E4.

---

### E3 — Bisect `max_model_len` at 131072 (15 min, free)

**Hypothesis under test:** The 10× cliff is exactly at `max_model_len = 131073`.

**Method:** Sweep `--max-model-len` in steps around 131072:
- 131072 (boundary, should be fast)
- 131073 (one over, should be slow if hypothesis correct)
- 131073 with `--enforce-eager` (should be fast)
- 200000 (slow)
- 262144 (your Run B)

**Predicted:** Sharp cliff between 131072 and 131073. The cliff disappears with `--enforce-eager`.

**If the cliff is NOT at 131072:** The cause is something other than the gate threshold; bisect to find the real cliff.

---

### E4 — Inspect the captured graph's attention kernel directly (30 min)

**Hypothesis under test:** The captured graph for Run B contains a Triton attention kernel launch; Run A's contains `paged_attention_rocm`.

**Method:**
1. Start Run B with `TORCHINDUCTOR_TRACE=1` and `VLLM_TORCH_COMPILE_LEVEL=3` to dump the captured graph.
2. Use `rocprof` / `rocscope` to capture a single decode step's kernel trace:
   ```bash
   rocprof --stats -o runB_decode.rdc <decode_step_repro>
   ```
3. Look at the kernels actually launched during decode replay. Expected in Run B: `kernel_paged_attention_2d` (the Triton kernel in the `else` branch at line 432). Expected in Run A: `paged_attention_rocm` (the custom kernel via `ops.paged_attention_rocm` at line 393).

**Predicted:** Run A's trace shows `paged_attention_rocm`; Run B's shows `kernel_paged_attention_2d`.

**If both show the same kernel:** My hypothesis is wrong; the slowdown is elsewhere.

---

### E5 — Patch the gate to use runtime value during capture (45 min)

**Hypothesis under test (definitive):** If we force the capture-time gate to evaluate as if `max_seq_len` were the batch max (not `max_model_len`), the cliff disappears.

**Method:** Edit `vllm/v1/worker/gpu/model_states/default.py:152-156`:

```python
# Force runtime path even during capture
max_seq_len = seq_lens_cpu_upper_bound[:num_reqs].max().item()
# if for_capture:
#     max_seq_len = self.max_model_len
# else:
#     max_seq_len = seq_lens_cpu_upper_bound[:num_reqs].max().item()
```

Re-run Run B with the patch.

**Predicted:** Run B's decode jumps to ~17 tok/s (matching Run A).

**If it doesn't:** Look elsewhere — but the source evidence is strong enough that I'd be surprised.

---

### E6 — Compare `num_gpu_blocks` and `Maximum concurrency` log lines (5 min, free)

**Hypothesis under test (negative control):** KV-cache size is NOT the cause.

**Method:** Grep both runs' startup logs for:
- `GPU KV cache size:`
- `Maximum concurrency for`

**Predicted:** Numbers differ by <10% between runs. If they differ by ~2×, KV-cache sizing is a contributor.

---

### E7 — File a bug report (after E1 + E3 confirm)

Once E1 and E3 confirm, file a vLLM issue with:
- Title: "[Performance][ROCm] `--max-model-len > 131072` silently bakes Triton attention fallback into captured CUDA graphs on gfx9"
- Body: cite `vllm/v1/worker/gpu/model_states/default.py:152-156` and `vllm/platforms/rocm.py:366`
- Suggested fix: either (a) raise the threshold in `use_rocm_custom_paged_attention` to 262144 for gfx9, or (b) during capture, evaluate the gate against a representative runtime `max_seq_len` instead of `max_model_len`, or (c) make the captured graph's attention dispatch metadata-driven rather than baked.

---

## Appendix: confidence levels

| Claim | Confidence | Evidence |
|---|---|---|
| `max_seq_len = max_model_len` during graph capture | **HIGH** | Direct source read of `default.py:152-156` and `mamba_hybrid.py:235` |
| Gate `max_seq_len <= 128*1024` for gfx9 | **HIGH** | Direct source read of `rocm.py:366` |
| Both runs use chunked prefill by default | **HIGH** | `arg_utils.py:2495-2502` + Qwen3 is `is_chunked_prefill_supported=True` |
| `max_num_batched_tokens` does NOT inflate to `max_model_len` in your runs | **HIGH** | `arg_utils.py:2647-2654` only fires when chunked prefill disabled |
| `cudagraph_capture_sizes` does NOT depend on `max_model_len` | **HIGH** | `vllm.py:1719-1730` derives from `max_num_seqs` |
| The capture-time warning was actually emitted in Run B | **MEDIUM (UNVERIFIED)** | Strongly implied by source; needs E1 to confirm |
| No exact-match GitHub issue exists | **HIGH** | Librarian agent + direct `gh issue list` searches |
| Issue #48453 (ROCm cudagraph mem reservation) is NOT your cause | **HIGH** | Wrong magnitude (17-20%) and orthogonal axis |
| `long_prefill_token_threshold` is not derived from `max_model_len` | **HIGH** | Source at `scheduler.py:74-77`, default 0 |

---

## Citations

**vLLM source (current `main` branch, commit ~`99115fcdcde9736d4097ea15e7bbab368c405edf` at time of librarian fetch):**
- `vllm/v1/worker/gpu/model_states/default.py:135-156` — `for_capture` branch
- `vllm/v1/worker/gpu/model_states/mamba_hybrid.py:231-237` — same pattern for hybrid models
- `vllm/v1/attention/ops/chunked_prefill_paged_decode.py:282, 359-369, 418-422` — call site + Triton fallback warning
- `vllm/platforms/rocm.py:346-382` — gate definition (gfx9 branch lines 358-368, threshold line 366)
- `vllm/v1/attention/backends/rocm_attn.py:118-150, 441-457` — metadata propagation and forward call
- `vllm/engine/arg_utils.py:2495-2502, 2647-2654` — chunked-prefill default and `max_num_batched_tokens` override
- `vllm/config/vllm.py:1719-1730` — `_set_cudagraph_sizes` derivation
- `vllm/config/scheduler.py:41-77` — `DEFAULT_MAX_NUM_BATCHED_TOKENS=2048`, `long_prefill_token_threshold=0`
- `vllm/v1/core/kv_cache_utils.py:747-770` — `_check_enough_kv_cache_memory` minimum-KV check
- `vllm/v1/worker/gpu_model_runner.py:600-606` — `token_ids_gpu_tensor` (ngram-GPU only)

**GitHub issues / PRs:**
- [#48453](https://github.com/vllm-project/vllm/issues/48453), [#47366](https://github.com/vllm-project/vllm/pull/47366), [#48440](https://github.com/vllm-project/vllm/pull/48440) — ROCm cudagraph mem reservation (17-20%, NOT your bug)
- [#44014](https://github.com/vllm-project/vllm/issues/44014), [#34571](https://github.com/vllm-project/vllm/pull/34571) — 10× decode cliff from cudagraph mode (hybrid Mamba+MoE)
- [#45943](https://github.com/vllm-project/vllm/issues/45943) — AITER persistent kernel context-length sensitivity
- [#13418](https://github.com/vllm-project/vllm/issues/13418), [#13922](https://github.com/vllm-project/vllm/pull/13922) — ROCm cudagraph padding
- [#18093](https://github.com/vllm-project/vllm/pull/18093) — `chunked_prefill_paged_decode` as ROCm V1 fallback
- [#40734](https://github.com/vllm-project/vllm/pull/40734) — `max_num_batched_tokens` append to cudagraph capture sizes
- [#4155](https://github.com/vllm-project/vllm/issues/7155), [#16118](https://github.com/vllm-project/vllm/issues/16118), [#25350](https://github.com/vllm-project/vllm/issues/25350) — `max_model_len` semantics clarifications
- [vLLM MI300X best-practices blog](https://vllm.ai/blog/2024-10-23-vllm-serving-amd) — `--max-seq-len-to-capture` discussion (V0, but the bucketing insight informs V1 behavior)
- [ROCm vLLM V1 optimization guide](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html) — `cudagraph_mode=FULL` vs `PIECEWISE` tradeoffs
