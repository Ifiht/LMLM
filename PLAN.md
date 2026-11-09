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
    M1  COMPLETE  scripts/annotate/annotate_annotator.sh loads the annotator
        from ../weights/LMLM-Annotator.

    M2  COMPLETE  The bf16 annotator reproduces the authors' dwiki-eval1k
        annotations without truncation (experiment/annotate/compare_annotations.py,
        run bf16-y).

    M3  Fetch the rendered HTML of every article revision in the      4wk
        20260901 stub, split each article into plain-text chunks along its
        own section structure, sized to fit one annotator call; set aside
        held-out eval articles.
        1. RUN WIKIMEDIA_CONTACT=<email or URL> python sources/fetch_wiki_html.py
           reads data/raw/enwiki-20260901-stub-articles.xml.gz; writes
           data/raw/enwiki-20260901-html.sqlite; resumable; ~24 days at
           200 requests/min
        2. EDIT sources/wiki_to_jsonl.py
           iter_articles(): read (page_id, html) rows with status 200 from
           data/raw/enwiki-20260901-html.sqlite.
           sections(): split the HTML at its <section> elements; drop
           headings, reference lists, infoboxes, tables, navboxes, and
           appendix sections; return plain-text paragraphs per section.
           Chunking, budget check, drop log, and held-out split unchanged.
           tests/test_wiki_to_jsonl.py: sample input becomes HTML.
        3. RUN python sources/wiki_to_jsonl.py
           writes data/raw/enwiki-20260901.jsonl,
           data/raw/enwiki-20260901.dropped.jsonl,
           data/ids/enwiki-heldout-ids.json
        4. EDIT src/lmlm/annotate/dataloader.py
           register an "enwiki" loader that reads
           data/raw/enwiki-20260901.jsonl into (texts, ids)
        Exit: no chunk over the limit; spot-checked chunks read as the
        rendered article text, split at section or paragraph boundaries.

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
