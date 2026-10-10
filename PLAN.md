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
    M1  Load the annotator from local weights instead of HF.          <1h
        1. EDIT scripts/annotate/annotate_annotator.sh:5
           MODEL_ID=../weights/LMLM-Annotator
        Exit: model loads from local path.

    M2  Prove our annotator reproduces the authors' annotations,        1d
        and measure its speed.
        Reference: data/dwiki-eval1k_annotator_llama-v6.1_cleaned.json
        (830 docs; original `text` + their `annotated_text`).
        1. EDIT src/lmlm/annotate/dataloader.py
           add loader that reads `text` from the reference file
        2. RUN python -m lmlm.annotate.annotate (bf16, then fp8)
           -> output/annotation/
        3. CREATE_FILE experiment/annotate/compare_annotations.py
           per doc: dblookup count, verbatim-copy check; tokens/sec
        4. RUN experiment/annotate/compare_annotations.py
        Exit: dblookup call rate and verbatim-copy fidelity match the
        reference; tokens/sec measured for bf16 and fp8.

    M3  Split every article into plain-text chunks along its own     2-3d
        section structure, sized to fit one annotator call; set aside
        held-out eval articles.
        1. CREATE_FILE sources/wiki_to_jsonl.py
           Purpose: turn the raw dump into annotator input chunks that
           follow article structure instead of truncating articles.
           Function: stream data/raw/enwiki-20260901/*.bz2 (articles only,
           redirects skipped); parse wikitext with mwparserfromhell; split
           at section headings (get_sections); merge adjacent small
           sections up to the chunk token limit; split oversize sections
           at paragraph, then sentence boundaries; strip markup to plain
           text; prefix every chunk with the article title ("Title\n\nBody",
           dolmino format); measure sizes with the annotator tokenizer.
           Writes {id: "<page_id>_chunk<n>", text} rows to
           data/raw/enwiki-20260901.jsonl and held-out article ids (all
           chunks of an article on one side) to
           data/ids/enwiki-heldout-ids.json. Assert self-check: no chunk
           exceeds the limit; chunks of an article reassemble its full text.
           Chunk token limit derives from max_model_len (proposal pending).
        2. RUN python sources/wiki_to_jsonl.py
        3. EDIT src/lmlm/annotate/dataloader.py
           register an "enwiki" loader that reads
           data/raw/enwiki-20260901.jsonl into (texts, ids), without
           truncate_sample_length
        Exit: no chunk over the limit; spot-checked splits fall on section
        or paragraph boundaries; chunk format matches dolmino side by side.

    M4  Annotate 1k of our docs and confirm quality holds.              1d
        1. CREATE_FILE data/ids/enwiki-pilot1k-ids.json
        2. EDIT scripts/annotate/annotate_annotator.sh
           DATASET, MANAGER, --subset ids/enwiki-pilot1k-ids
        3. RUN scripts/annotate/annotate_annotator.sh
        4. RUN experiment/annotate/compare_annotations.py
        Exit: M2 quality bar.

    M5  Annotate the full corpus.                                3wk-4mo
        1. EDIT scripts/annotate/annotate_annotator.sh
           remove --subset
        2. RUN scripts/annotate/annotate_annotator.sh
           resumable: saves every 500 docs, skips annotated docs
        Exit: all docs annotated.
        Duration fixed by M2 throughput.

    M6  Extract the knowledge database from the annotations and      2-3d
        make it fit in 46GB RAM.
        1. EDIT scripts/train/extract_database.sh
           ANNOTATION_PATH, SAVE_PATH (hard-coded to squad-eval100)
        2. RUN scripts/train/extract_database.sh -> data/database/
        3. EDIT src/lmlm/database/topk_retriever.py
           compressed FAISS index
        Exit: triplet count and spot check pass; index loads in memory.

    M7  Train the LMLM and a no-memory baseline of the same size.    6-8wk
        1. CREATE_FILE experiment/train/prepare_pretrain_dataset.py
           annotations -> train/validation splits with `annotated_text`
           (format read by src/lmlm/training/utils/load_sft_dataset.py:22),
           using cleaning functions in src/lmlm/training/utils/utils_filter.py
        2. RUN experiment/train/prepare_pretrain_dataset.py
        3. EDIT scripts/train/pretrain.sh
           NUM_GPUs=1, CUDA_VISIBLE_DEVICES=0, DATASET_PATH;
           drop missing scripts/account/wandb_config.sh
        4. RUN scripts/train/pretrain.sh (LMLM-176M)
        5. EDIT src/lmlm/training/pretrain.py:68 or
           src/lmlm/training/utils/load_sft_dataset.py:21
           baseline flag conflict: plain_baseline requires special tokens
           off; load_sft_dataset asserts them on
        6. RUN scripts/train/pretrain.sh (Standard-176M, --plain_baseline True)
        Exit: loss converges; LMLM perplexity beats baseline.

    M8  Score both models on perplexity, NLU, knowledge, factuality, 3-5d
        and unlearning.
        Perplexity
        1. CREATE_FILE scripts/eval/eval_ppl.sh
           calls src/lmlm/training/eval_dataset.py on held-out docs
        2. RUN scripts/eval/eval_ppl.sh
        NLU
        3. EDIT scripts/eval/eval_nlu_task.sh  CHECKPOINTS
        4. RUN scripts/eval/eval_nlu_task.sh
        T-REx
        5. EDIT scripts/eval/eval_trex.sh  CHECKPOINTS, DATABASE_PATH
        6. RUN scripts/eval/eval_trex.sh
        FactScore
        7. RUN git clone FActScore to ../FActScore; follow its setup
           (believed to include its own Wikipedia DB; unverified)
        8. CREATE_FILE scripts/account/openai_key.sh
        9. EDIT scripts/eval/eval_factscore.sh  CHECKPOINTS, DATABASE_PATH
        10. RUN scripts/eval/eval_factscore.sh  (OpenAI API, billed)
        TOFU
        11. RUN git clone locuslab/open-unlearning (outside repo)
        12. CREATE_FILE integration of LMLM checkpoints with
            open-unlearning (not in this repo; scope unknown)
        Exit: results table, LMLM vs. baseline.

### DEFINITIONS
```
RUN          Execute a shell command. Must complete successfully for the
             project state it executes in. May depend on prior RUN, EDIT,
             or CREATE_FILE steps.

EDIT         Modify an existing file. Step shall state the exact change
             needed.

CREATE_FILE  Write a new file. Step shall state the purpose and function of
             the file.
```

> [!NOTE]
> All steps shall use paths relative to the repo root for every referenced file and command.
