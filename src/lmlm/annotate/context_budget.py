"""GPU-measured context limit and per-document token budget for vLLM annotation.

measure_max_model_len() finds the largest max_model_len the current GPU can serve.
budget_prompts() builds annotator prompts and drops, never truncates, documents that do not fit.

Settings these functions rely on, from the "llm" block of configs/llama/default.json:
- gpu_memory_utilization 0.95: the authors' value. vLLM sizes its KV cache to fill this fraction of total
  GPU memory (vllm/worker/worker.py:243-245). The remaining 5% covers memory allocated after that
  calculation: CUDA graph capture (vllm/worker/worker.py:256-287) and the CUDA context.
- dtype "bfloat16": without it vLLM runs a float32 checkpoint in float16 (vllm/config.py:1642-1644);
  Llama 3.1 is a bfloat16 model and LMLM-Annotator ships as float32.
- max_num_seqs 40: upper bound on documents generated at once. When their KV cache exceeds capacity,
  vLLM pauses and recomputes some of them (PreemptionMode.RECOMPUTE); that costs time and never truncates.
"""
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

# Ceiling on max_model_len. Above 32768, vLLM 0.6.3 switches on chunked prefill by itself
# (vllm/engine/arg_utils.py:940-956), which changes scheduling and memory profiling.
MAX_MODEL_LEN_CEILING = 32768

# max_model_len of the probe engine. vLLM refuses to start when max_model_len exceeds the KV cache
# capacity; 2048 passes on any GPU that can hold the model at all, so the probe does not fail on size.
PROBE_MAX_MODEL_LEN = 2048

# The measured capacity is rounded down to a multiple of this. The probe and the real engine profile
# the same shape (max_num_seqs dummy sequences, vllm/worker/model_runner.py:1252-1255) and the real
# engine profiles fewer tokens, so its capacity is at least the probe's; this is margin on top.
ROUND_DOWN_TO = 1024

# Output tokens expected per input text token. The annotator copies the text and inserts dblookup
# calls; over the 830 dwiki-eval1k reference docs the output/input token ratio peaks at 3.99 (p99 3.30).
# Used only to decide whether a document fits. Each output may use all remaining context.
OUTPUT_TOKENS_PER_TEXT_TOKEN = 3.99


def _probe_kv_capacity(model_path, llm_kwargs):
    from vllm import LLM  # imported in the child so the parent process never initializes CUDA

    llm = LLM(model=model_path, **{
        **llm_kwargs,
        "max_model_len": PROBE_MAX_MODEL_LEN,
        # vLLM's memory profiling runs a dummy batch of max_num_batched_tokens tokens before sizing the KV
        # cache (vllm/worker/worker.py:223-245). Sizing that batch at the ceiling measures the cache left
        # after the largest activations a real engine profiles; a real engine without chunked prefill
        # profiles max(max_model_len, 2048) tokens (vllm/config.py:1003). vLLM only rejects values below
        # max_model_len (vllm/config.py:1039-1047).
        "max_num_batched_tokens": MAX_MODEL_LEN_CEILING,
        # Random weights of the real shape and dtype: the same GPU memory, no checkpoint read.
        "load_format": "dummy",
        # CUDA graphs are captured after the KV cache is sized (vllm/worker/worker.py:256-287); skipping
        # them shortens the probe without changing the measurement.
        "enforce_eager": True,
    })
    cache = llm.llm_engine.cache_config
    return cache.num_gpu_blocks * cache.block_size


def measure_max_model_len(model_path, llm_kwargs):
    """Largest max_model_len, up to MAX_MODEL_LEN_CEILING, that this GPU can serve with llm_kwargs.

    Measured by vLLM's own memory profiling on the current GPU, so the result is specific to the GPU,
    model, dtype, and max_num_seqs. Other processes' GPU memory is not accounted for (vLLM's limitation).
    """
    # Separate process: vLLM 0.6.3 does not reliably release GPU memory when an engine is discarded, so
    # the probe runs in a child process that exits before the real engine starts.
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
        capacity = pool.submit(_probe_kv_capacity, model_path, llm_kwargs).result()
    max_model_len = min(MAX_MODEL_LEN_CEILING, capacity // ROUND_DOWN_TO * ROUND_DOWN_TO)
    if max_model_len < PROBE_MAX_MODEL_LEN:
        raise RuntimeError(f"KV cache holds {capacity} tokens; too small to annotate on this GPU.")
    print(f"KV cache capacity {capacity} tokens at {MAX_MODEL_LEN_CEILING}-token profiling; "
          f"max_model_len set to {max_model_len}")
    return max_model_len


def budget_prompts(texts, tokenizer, prompt, max_model_len):
    """Build one annotator prompt per text with its generation limit; drop texts that do not fit.

    Returns (kept, dropped). kept: list of (index, prompt, max_tokens). dropped: list of dicts (index,
    tokens_needed, max_model_len, reason) for each text whose prompt plus expected output exceeds
    max_model_len. Nothing is truncated.
    """
    kept, dropped = [], []
    for i, text in enumerate(texts):
        # The chat template writes <|begin_of_text|>; vLLM adds another when it encodes the string.
        p = tokenizer.apply_chat_template(prompt(text), tokenize=False, add_generation_prompt=True)
        p = p.removeprefix(tokenizer.bos_token)
        prompt_tokens = len(tokenizer.encode(p))  # encoded as vLLM encodes it, <|begin_of_text|> included
        text_tokens = len(tokenizer.encode(text, add_special_tokens=False))
        needed = prompt_tokens + math.ceil(OUTPUT_TOKENS_PER_TEXT_TOKEN * text_tokens)
        if needed > max_model_len:
            dropped.append({"index": i, "tokens_needed": needed, "max_model_len": max_model_len,
                            "reason": "prompt plus expected output exceeds max_model_len"})
        else:
            kept.append((i, p, max_model_len - prompt_tokens))
    return kept, dropped
