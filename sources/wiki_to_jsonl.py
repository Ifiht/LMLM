"""Convert the enwiki-20260901 dump into annotator input chunks.

Run: python sources/wiki_to_jsonl.py   (from the repo root; probes the GPU first, then runs on CPU)

Reads data/raw/enwiki-20260901/*.bz2 and keeps articles (namespace 0, redirects skipped).
Each article becomes plain text: mwparserfromhell strip_code() output, with no title or headings added.
Chunks grow section by section up to the largest text the annotator accepts on this GPU
(src/lmlm/annotate/context_budget.py). A section over that limit is split at paragraph boundaries;
a paragraph over the limit is dropped and logged. Chunks never join text across a dropped paragraph.

Writes:
  data/raw/enwiki-20260901.jsonl          {"id": "<page_id>_chunk<n>", "text": ...}
  data/raw/enwiki-20260901.dropped.jsonl  {"id": "<page_id>", "tokens": ..., "reason": ..., "text": ...}
  data/ids/enwiki-heldout-ids.json        chunk ids of HELDOUT_ARTICLES whole articles
"""
import bz2
import glob
import json
import math
import os
import random
import re
import xml.etree.ElementTree as ET
from multiprocessing import get_context

import mwparserfromhell
from transformers import AutoTokenizer

from lmlm.annotate.annotators import Prompt
from lmlm.annotate.context_budget import OUTPUT_TOKENS_PER_TEXT_TOKEN, budget_prompts, measure_max_model_len
from lmlm.constants import CONFIGS_DIR, DATA_DIR, ROOT_DIR

DUMP_DIR = os.path.join(DATA_DIR, "raw", "enwiki-20260901")
OUT_PATH = os.path.join(DATA_DIR, "raw", "enwiki-20260901.jsonl")
DROPPED_PATH = os.path.join(DATA_DIR, "raw", "enwiki-20260901.dropped.jsonl")
PARTS_DIR = os.path.join(DATA_DIR, "raw", "enwiki-20260901.parts")  # per-dump-file output; finished files are skipped on rerun
HELDOUT_PATH = os.path.join(DATA_DIR, "ids", "enwiki-heldout-ids.json")
MODEL_PATH = os.path.join(ROOT_DIR, "..", "weights", "LMLM-Annotator")
PROMPT_ID = "llama-v6.1"
HELDOUT_ARTICLES = 1000  # the authors' test set size (dwiki-test1k)
SEP = "\n\n"
# wozniak: half the logical CPUs bounds memory (one tokenizer and one parsed article per worker).
# Upgrade path: raise it if RAM allows.
WORKERS = max(1, os.cpu_count() // 2)

# Wikipedia's standard appendix sections (MOS:LAYOUT): citation and link lists, not article prose.
APPENDIX_HEADINGS = {"references", "notes", "footnotes", "citations", "sources", "bibliography",
                     "further reading", "external links", "see also", "notes and references"}
# Removed before strip_code(), which otherwise keeps their contents: reference text inline,
# image captions as "thumb|caption", category names, and table cells run together.
# wozniak: tables are dropped whole; their facts are lost. Upgrade path: render rows as sentences.
DROP_TAGS = {"ref", "references", "gallery", "table"}
DROP_LINK_NAMESPACES = {"file", "image", "category"}


def sections(wikitext):
    """Plain-text paragraphs of each section of an article, appendix sections and headings removed."""
    code = mwparserfromhell.parse(wikitext)
    junk = [t for t in code.filter_tags() if str(t.tag).strip().lower() in DROP_TAGS]
    junk += [l for l in code.filter_wikilinks()
             if str(l.title).strip().split(":")[0].strip().lower() in DROP_LINK_NAMESPACES]
    for node in junk:
        try:
            code.remove(node)
        except ValueError:  # already removed with an enclosing node
            pass
    result = []
    for section in code.get_sections(flat=True, include_lead=True):
        headings = section.filter_headings(recursive=False)
        if headings and headings[0].title.strip_code().strip().lower() in APPENDIX_HEADINGS:
            continue
        for h in headings:
            section.remove(h)
        paragraphs = []
        for block in re.split(r"\n\s*\n", section.strip_code()):
            lines = [line.strip() for line in block.splitlines()]
            if text := "\n".join(line for line in lines if line):
                paragraphs.append(text)
        if paragraphs:
            result.append(paragraphs)
    return result


def chunk_article(article_sections, n_tokens, fits):
    """Pack sections into chunks no larger than fits() allows.

    article_sections: list of sections, each a list of paragraphs. n_tokens(text) counts tokens;
    fits(n) says whether a chunk of n tokens is accepted. Returns (chunks, dropped), dropped being
    (paragraph, tokens) pairs. A section that does not fit whole contributes its paragraphs one by one.
    """
    sep_tokens = n_tokens(SEP)
    chunks, dropped, current, current_tokens = [], [], [], 0

    def flush():
        nonlocal current, current_tokens
        if current:
            chunks.append(SEP.join(current))
        current, current_tokens = [], 0

    for paragraphs in article_sections:
        whole = SEP.join(paragraphs)
        whole_tokens = n_tokens(whole)
        units = [(whole, whole_tokens)] if fits(whole_tokens) else [(p, n_tokens(p)) for p in paragraphs]
        for text, tokens in units:
            if not fits(tokens):
                flush()  # keep chunks contiguous: no chunk spans a dropped paragraph
                dropped.append((text, tokens))
            elif current and fits(current_tokens + sep_tokens + tokens):
                current.append(text)
                current_tokens += sep_tokens + tokens
            else:
                flush()
                current, current_tokens = [text], tokens
    flush()
    return chunks, dropped


def iter_articles(path):
    """(page_id, wikitext) for every non-redirect namespace-0 page in one dump file."""
    with bz2.open(path) as f:
        context = ET.iterparse(f, events=("start", "end"))
        _, root = next(context)
        ns = root.tag[:root.tag.index("}") + 1]
        for event, elem in context:
            if event != "end" or elem.tag != ns + "page":
                continue
            text = elem.findtext(f"{ns}revision/{ns}text")
            if elem.findtext(ns + "ns") == "0" and elem.find(ns + "redirect") is None and text:
                yield elem.findtext(ns + "id"), text
            root.clear()  # free parsed pages


_worker = {}


def _init_worker(max_model_len):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    prompt = Prompt(PROMPT_ID)
    empty = tokenizer.apply_chat_template(prompt(""), tokenize=False, add_generation_prompt=True)
    overhead = len(tokenizer.encode(empty.removeprefix(tokenizer.bos_token)))
    # Estimate of budget_prompts' test (context_budget.py), for packing; each chunk is then checked exactly.
    _worker.update(tokenizer=tokenizer, prompt=prompt, max_model_len=max_model_len, fits=lambda n:
                   overhead + n + math.ceil(OUTPUT_TOKENS_PER_TEXT_TOKEN * n) <= max_model_len)


def _convert_file(path):
    """Write one dump file's chunks and drops to PARTS_DIR. Returns [(page_id, n_chunks)]."""
    name = os.path.basename(path)
    out, drop, index = (os.path.join(PARTS_DIR, f"{name}.{k}") for k in ("jsonl", "dropped.jsonl", "index.json"))
    if os.path.exists(index):
        with open(index) as f:
            return json.load(f)
    tokenizer, prompt, max_model_len, fits = (_worker[k] for k in ("tokenizer", "prompt", "max_model_len", "fits"))
    n_tokens = lambda text: len(tokenizer.encode(text, add_special_tokens=False))
    articles = []
    with open(out + ".tmp", "w") as fo, open(drop + ".tmp", "w") as fd:
        for page_id, wikitext in iter_articles(path):
            chunks, dropped = chunk_article(sections(wikitext), n_tokens, fits)
            kept, rejected = budget_prompts(chunks, tokenizer, prompt, max_model_len)
            for d in rejected:
                dropped.append((chunks[d["index"]], n_tokens(chunks[d["index"]])))
            for text, tokens in dropped:
                fd.write(json.dumps({"id": page_id, "tokens": tokens, "reason":
                                     "prompt plus expected output exceeds max_model_len", "text": text}) + "\n")
            for n, (i, _, _) in enumerate(kept):
                fo.write(json.dumps({"id": f"{page_id}_chunk{n}", "text": chunks[i]}) + "\n")
            if kept:
                articles.append((page_id, len(kept)))
    os.replace(out + ".tmp", out)
    os.replace(drop + ".tmp", drop)
    with open(index, "w") as f:  # written last: marks the file finished
        json.dump(articles, f)
    print(f"{name}: {len(articles)} articles")
    return articles


def main():
    with open(os.path.join(CONFIGS_DIR, "llama", "default.json")) as f:
        max_model_len = measure_max_model_len(MODEL_PATH, json.load(f)["llm"])
    os.makedirs(PARTS_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(DUMP_DIR, "*.bz2")))
    with get_context("spawn").Pool(WORKERS, _init_worker, (max_model_len,)) as pool:
        per_file = pool.map(_convert_file, files, chunksize=1)

    for final, suffix in ((OUT_PATH, "jsonl"), (DROPPED_PATH, "dropped.jsonl")):
        with open(final, "wb") as fo:
            for path in files:
                with open(os.path.join(PARTS_DIR, f"{os.path.basename(path)}.{suffix}"), "rb") as fi:
                    while block := fi.read(1 << 24):
                        fo.write(block)

    articles = [a for file_articles in per_file for a in file_articles]
    heldout = random.Random(0).sample(sorted(articles), HELDOUT_ARTICLES)
    with open(HELDOUT_PATH, "w") as f:
        json.dump([f"{page_id}_chunk{n}" for page_id, n_chunks in heldout for n in range(n_chunks)], f, indent=4)

    for path in files:
        for suffix in ("jsonl", "dropped.jsonl", "index.json"):
            os.remove(os.path.join(PARTS_DIR, f"{os.path.basename(path)}.{suffix}"))
    os.rmdir(PARTS_DIR)
    print(f"{len(articles)} articles, {sum(n for _, n in articles)} chunks -> {OUT_PATH}; "
          f"{HELDOUT_ARTICLES} held-out articles -> {HELDOUT_PATH}")


if __name__ == "__main__":
    main()
