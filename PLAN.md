# PLAN

## SUMMARY

### NAME
LMLM-2026 — Limited Memory Language Model, full pipeline rebuilt on enwiki-20260901.

### STATUS (2026-10-09)
    environment     COMPLETE     lmlm conda env; matches requirements.lock.txt
    gpu             OPERATIONAL  1x RTX 3090 24GB, driver 580.178.04; CUDA verified
    corpus          COMPLETE     data/raw/enwiki-20260901; 71 files, 48GB; md5 verified
    annotator       COMPLETE     ../weights/LMLM-Annotator
    annotate.sh     OPEN         MODEL_ID -> ../weights/LMLM-Annotator
    converter       NOT STARTED  dump -> {id, text} loader
    factscore       NOT STARTED  pip install factscore --no-deps
    M5-M8           NOT STARTED  annotation, database, pretraining, evaluation

### CRITICAL PATH
    M1  Point scripts/annotate/annotate_annotator.sh at ../weights/LMLM-Annotator
        Exit: model loads from local path.
    M2  Annotator check on dwiki-eval1k (dolmino format).               1d
        Exit: dblookup call rate and verbatim-copy fidelity match the
        reference; tokens/sec measured for bf16 and fp8.
    M3  Dump converter + loader; hold out perplexity eval set.        1-2d
        Exit: text format matches dolmino side by side; assert check passes.
    M4  Pilot annotation, 1k docs from enwiki-20260901.                 1d
        Exit: M2 quality bar.
    M5  Full corpus annotation.                                  3wk-4mo
        Exit: all docs annotated; resumable checkpoints.
        Duration fixed by M2 throughput.
    M6  Build database; compressed FAISS index within 46GB RAM.       2-3d
        Exit: triplet count and spot check pass; index loads in memory.
    M7  Pretrain LMLM-176M and matching Standard-176M baseline.      6-8wk
        Exit: loss converges; LMLM perplexity beats baseline.
    M8  Evaluate: perplexity, NLU, T-REx, FactScore, TOFU.            3-5d
        Exit: results table, LMLM vs. baseline.

