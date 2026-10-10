# Deterministic CPU checks for annotation prompts, drops, fine-tuning text, and run comparison.
# No model weights, tokenizer files, or GPU: FakeLlamaTokenizer and StubLLM stand in for them.
# Run: python -m pytest tests/
import json
import os
import re
import sys
import types

from lmlm.annotate.annotators import LlamaAnnotator, Prompt
from lmlm.annotate.context_budget import MAX_MODEL_LEN_CEILING, budget_prompts
from lmlm.constants import CONFIGS_DIR, ROOT_DIR
from lmlm.training.utils.load_sft_dataset import format_chat

sys.path.insert(0, os.path.join(ROOT_DIR, "experiment", "annotate"))
import compare_annotations  # noqa: E402


class FakeLlamaTokenizer:
    """The Llama 3 tokenizer behaviour the code relies on, without tokenizer files.

    apply_chat_template writes <|begin_of_text|> and role headers like the Llama 3.1 template; encode
    counts one token per special token or word and, by default, prepends <|begin_of_text|> as the real
    tokenizer does when vLLM or TRL encodes a string.
    """
    bos_token = "<|begin_of_text|>"
    bos_token_id = 0
    pieces = re.compile(r"<\|[a-z_]+\|>|[^\s<]+|<")

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = self.bos_token + "".join(
            f"<|start_header_id|>{m['role']}<|end_header_id|>\n\n{m['content'].strip()}<|eot_id|>" for m in messages)
        return text + ("<|start_header_id|>assistant<|end_header_id|>\n\n" if add_generation_prompt else "")

    def encode(self, text, add_special_tokens=True):
        ids = [self.bos_token_id if p == self.bos_token else 1 for p in self.pieces.findall(text)]
        return [self.bos_token_id] * add_special_tokens + ids

    def __call__(self, text):
        return {"input_ids": self.encode(text)}


class StubLLM:
    """Stands in for vllm.LLM: the first prompt stops at the length limit, the rest stop normally."""

    def generate(self, prompts, params):
        assert len(prompts) == len(params)
        return [types.SimpleNamespace(outputs=[types.SimpleNamespace(
            text=f"annotation {n}", finish_reason="length" if n == 0 else "stop")]) for n in range(len(prompts))]


TOKENIZER = FakeLlamaTokenizer()
PROMPT = Prompt("llama-v6.1")
ASSISTANT_HEADER = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
SHORT = "Ada Lovelace\n\nAda Lovelace (1815–1852) was an English mathematician."
LONG = "Long Article\n\n" + " ".join(["word"] * 600)  # needs ~3,100 tokens: over a 1,000 limit
LIMIT = 1000


def test_prompts_are_whole_with_one_bos():
    kept, dropped = budget_prompts([SHORT, LONG], TOKENIZER, PROMPT, MAX_MODEL_LEN_CEILING)
    assert dropped == []
    for (i, prompt, max_tokens), text in zip(kept, [SHORT, LONG]):
        assert text.strip() in prompt
        assert prompt.endswith(ASSISTANT_HEADER)
        ids = TOKENIZER.encode(prompt)  # vLLM encodes string prompts this way, adding <|begin_of_text|>
        assert ids[0] == TOKENIZER.bos_token_id and ids.count(TOKENIZER.bos_token_id) == 1
        assert max_tokens == MAX_MODEL_LEN_CEILING - len(ids)


def test_oversize_texts_are_dropped_not_shortened():
    kept, dropped = budget_prompts([SHORT, LONG], TOKENIZER, PROMPT, LIMIT)
    assert [i for i, _, _ in kept] == [0] and SHORT.strip() in kept[0][1]
    assert [d["index"] for d in dropped] == [1]
    assert dropped[0]["tokens_needed"] > LIMIT
    assert dropped[0]["reason"] == "prompt plus expected output exceeds max_model_len"


def test_annotate_aligns_results_and_reports_drops():
    annotator = LlamaAnnotator.__new__(LlamaAnnotator)  # skip __init__: no GPU probe, no model load
    annotator.tokenizer, annotator.prompt, annotator.llm, annotator.max_model_len = TOKENIZER, PROMPT, StubLLM(), LIMIT
    with open(os.path.join(CONFIGS_DIR, "llama", "default.json")) as f:
        annotator.configs = json.load(f)

    # SHORT reaches the LLM first and stops at the length limit; LONG is dropped before generation.
    results = annotator.annotate([SHORT, LONG, SHORT + " Second.", SHORT + " Third."])

    assert results == [None, None, "annotation 1", "annotation 2"]
    assert {d["index"]: d["reason"] for d in annotator.dropped} == {
        1: "prompt plus expected output exceeds max_model_len",
        0: "output reached max_model_len",
    }


def test_fine_tuning_text_has_one_bos():
    def prompt(text, annotation):
        return PROMPT(text) + [{"role": "assistant", "content": annotation}]

    example = {"text": SHORT, "annotated_text": "Ada Lovelace [dblookup('Ada Lovelace', 'Birth Year') -> 1815] 1815"}
    ids = TOKENIZER(format_chat(example, TOKENIZER, prompt)["formatted_text"])["input_ids"]  # as TRL's SFTTrainer
    assert ids[0] == TOKENIZER.bos_token_id and ids.count(TOKENIZER.bos_token_id) == 1


def test_compare_skips_dropped_docs(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(compare_annotations, "SAVE_DIR", str(tmp_path))
    runs = {
        "a": ["Born [dblookup('A', 'Birth Year') -> 1900] 1900.", None, "Same text."],
        "b": ["Born [dblookup('A', 'Birth Year') -> 1901] 1901.", "Kept in b only.", "Same text."],
    }
    for name, outputs in runs.items():
        (tmp_path / f"{name}.json").write_text(json.dumps({"outputs": outputs}))

    compare_annotations.compare("a", "a")
    out = capsys.readouterr().out
    assert "(2 docs; 1 dropped in either run)" in out and "identical outputs:   100.0%" in out

    compare_annotations.compare("a", "b")
    out = capsys.readouterr().out
    assert "(2 docs; 1 dropped in either run)" in out and "identical outputs:   50.0%" in out
