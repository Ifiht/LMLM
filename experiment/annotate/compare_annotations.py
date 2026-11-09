# Run the annotator on the 830 dwiki-eval1k reference inputs and compare runs per document.
# Run:     python experiment/annotate/compare_annotations.py --model-path PATH --run-name NAME
#          -> output/eval/annotator_check/NAME.json (raw outputs, timing); refuses to overwrite.
# Compare: python experiment/annotate/compare_annotations.py --compare NAME_A NAME_B
#          -> per-document agreement between two runs (two runs of one checkpoint = run-to-run noise).
# Self-check: --compare NAME NAME reports 100% identical.
import argparse
import difflib
import json
import os
import re
import time

from lmlm.constants import CONFIGS_DIR, DATA_DIR, ROOT_DIR

REF_PATH = os.path.join(DATA_DIR, "dwiki-eval1k_annotator_llama-v6.1_cleaned.json")
SAVE_DIR = os.path.join(ROOT_DIR, "output", "eval", "annotator_check")
# wozniak: non-greedy match ends at the first "]"; a looked-up value containing "]" is cut short.
# Upgrade path: parse with the dblookup validators in src/lmlm/training/utils/utils_filter.py.
DBLOOKUP = re.compile(r"\[dblookup\('(.*?)',\s*'(.*?)'\)\s*->\s*(.*?)\]")


def load_args():
    parser = argparse.ArgumentParser(description="Per-document annotator agreement between checkpoints.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--model-path", type=str, help="Annotator checkpoint directory to run.")
    mode.add_argument("--compare", nargs=2, metavar=("NAME_A", "NAME_B"), help="Two saved run names to compare.")
    parser.add_argument("--run-name", type=str, help="Names the results file (required with --model-path).")
    args = parser.parse_args()
    if args.model_path and not args.run_name:
        parser.error("--run-name is required with --model-path")
    return args


def run(args):
    # vLLM imports here so --compare runs without CUDA
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from lmlm.annotate.annotators import Prompt
    from lmlm.annotate.context_budget import budget_prompts, measure_max_model_len

    save_path = os.path.join(SAVE_DIR, f"{args.run_name}.json")
    if os.path.exists(save_path):
        raise FileExistsError(f"{save_path} exists; choose another --run-name.")
    with open(REF_PATH) as f:
        examples = json.load(f)["examples"]
    with open(os.path.join(CONFIGS_DIR, "llama", "default.json")) as f:
        configs = json.load(f)
    texts = [e["text"] for e in examples]

    t = time.time()
    max_model_len = measure_max_model_len(args.model_path, configs["llm"])
    probe_s = time.time() - t

    t = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    llm = LLM(model=args.model_path, max_model_len=max_model_len, **configs["llm"])
    load_s = time.time() - t

    # Same prompts and limits as LlamaAnnotator.annotate (src/lmlm/annotate/annotators.py), which returns
    # only text; calling vLLM directly also exposes token counts and finish reasons.
    kept, dropped = budget_prompts(texts, tokenizer, Prompt("llama-v6.1"), max_model_len)
    params = [SamplingParams(**{**configs["sampling"], "max_tokens": max_tokens}) for _, _, max_tokens in kept]
    t = time.time()
    responses = llm.generate([p for _, p, _ in kept], params)
    generate_s = time.time() - t

    outputs = [None] * len(texts)  # None where the input was dropped
    for (i, _, _), response in zip(kept, responses):
        outputs[i] = response.outputs[0]
    generated = [o for o in outputs if o is not None]
    summary = {
        "model_path": args.model_path,
        "llm_config": configs["llm"],
        "max_model_len": max_model_len,
        "n_docs": len(texts),
        "n_dropped_input": len(dropped),
        "dropped_input": dropped,
        "probe_s": probe_s,
        "load_s": load_s,
        "generate_s": generate_s,
        "output_tok_per_s": sum(len(o.token_ids) for o in generated) / generate_s,
        "n_hit_token_limit": sum(o.finish_reason == "length" for o in generated),
    }
    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(save_path, "w") as f:
        # outputs in reference order; original_dataset_ids repeats across different docs (820 unique of 830)
        json.dump({"summary": summary, "outputs": [o.text if o else None for o in outputs]}, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Saved {save_path}")


def triplet_jaccard(a, b):
    sa, sb = set(DBLOOKUP.findall(a)), set(DBLOOKUP.findall(b))
    return len(sa & sb) / len(sa | sb) if sa | sb else 1.0


def stats(values):
    s = sorted(values)
    return f"mean {sum(s) / len(s):.4f}  p10 {s[len(s) // 10]:.4f}  min {s[0]:.4f}"


def compare(name_a, name_b):
    runs = []
    for name in (name_a, name_b):
        with open(os.path.join(SAVE_DIR, f"{name}.json")) as f:
            runs.append(json.load(f)["outputs"])
    a, b = runs
    assert len(a) == len(b), f"{name_a} has {len(a)} outputs, {name_b} has {len(b)}"
    # documents dropped in either run (None) have nothing to compare
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    a, b = [x for x, _ in pairs], [y for _, y in pairs]

    similarity = [difflib.SequenceMatcher(None, x.split(), y.split(), autojunk=False).ratio() for x, y in zip(a, b)]
    print(f"{name_a} vs {name_b}  ({len(a)} docs; {len(runs[0]) - len(a)} dropped in either run)")
    print(f"identical outputs:   {sum(x == y for x, y in zip(a, b)) / len(a):.1%}")
    print(f"word similarity:     {stats(similarity)}")
    print(f"triplet jaccard:     {stats([triplet_jaccard(x, y) for x, y in zip(a, b)])}")


def main():
    args = load_args()
    if args.compare:
        compare(*args.compare)
    else:
        run(args)


if __name__ == "__main__":
    main()
