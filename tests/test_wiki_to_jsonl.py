# CPU checks for sources/wiki_to_jsonl.py: wikitext cleanup and section-aware chunking.
# Run: python -m pytest tests/
import os
import sys

from lmlm.constants import ROOT_DIR

sys.path.insert(0, os.path.join(ROOT_DIR, "sources"))
from wiki_to_jsonl import SEP, chunk_article, sections  # noqa: E402

WIKITEXT = """'''Ada Lovelace''' was a [[mathematician|mathematician]].<ref>Toole 1998, p. 12.</ref>
[[File:Ada.jpg|thumb|Portrait of [[Ada Lovelace]]]]

She wrote the first program.

== Early life ==
She was born in London.
{| class="wikitable"
|-
| 1815 || born
|}

== References ==
{{reflist}}
* Toole, Betty. ''Ada''.

[[Category:Mathematicians]]
"""


def test_sections_keep_prose_only():
    assert sections(WIKITEXT) == [
        ["Ada Lovelace was a mathematician.", "She wrote the first program."],
        ["She was born in London."],
    ]


def test_chunks_fit_and_keep_every_paragraph_in_order():
    words = lambda text: len(text.split())
    fits = lambda n: n <= 10
    article = [
        ["a1 a2 a3", "b1 b2 b3"],                     # section fits whole
        ["c1 c2 c3 c4 c5 c6", "d1 d2 d3 d4 d5 d6"],   # section too big whole: split at paragraphs
        [" ".join(f"e{i}" for i in range(11))],      # paragraph over the limit: dropped
        ["f1 f2"],
    ]
    chunks, dropped = chunk_article(article, words, fits)

    assert all(fits(words(c)) for c in chunks)
    assert [t for t, _ in dropped] == article[2]
    # Every paragraph appears once, in article order, and no chunk spans the dropped paragraph.
    kept = [p for c in chunks for p in c.split(SEP)]
    assert kept == [p for s in article if s is not article[2] for p in s]
    assert "f1 f2" in chunks and not any("d1" in c and "f1" in c for c in chunks)
