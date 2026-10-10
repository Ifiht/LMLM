import re
import string
import torch
from collections import defaultdict, Counter
from typing import List, Tuple


def add_shared_context_ids(dataset):
    context_to_ids = defaultdict(list)
    for ex in dataset:
        context_to_ids[ex['context']].append(ex['id'])
    return dataset.add_column("shared_ids", [context_to_ids[ex['context']] for ex in dataset])


def chunk_wiki_text(texts: List[str], ids: List[str], max_len: int = 750) -> Tuple[List[str], List[str]]:
    chunks, chunk_ids = [], []
    for text, pid in zip(texts, ids):
        tokens = text.split()
        for i in range(0, len(tokens), max_len):
            chunk = " ".join(tokens[i:i + max_len])
            chunks.append(chunk)
            chunk_ids.append(f"{pid}_chunk{i // max_len}")
    return chunks, chunk_ids

import re

def get_save_name(args):
    """
    Generate a filename for saving based on model and annotator info.
    """
    # Extract model name from path
    model_name = args.model_id.rstrip('/').split('/')[-1]

    # Extract model size (e.g., '7B') if present
    size_match = re.search(r'(\d+B)', model_name, re.IGNORECASE)
    model_size = size_match.group(1).lower() if size_match else ""

    # Format annotator name
    if args.annotator in {"llama", "llama-lora-ft", "llama-lora-ft-hf"}:
        suffix = model_name.split('_', 1)[-1]
        if args.annotator == "llama":
            annotator = f"llama{model_size}-{suffix}"
        else:
            annotator = f"llama{model_size}-lora-ft-{suffix}"
    else:
        annotator = args.annotator

    return f"{args.manager}_{annotator}_{args.prompt_id}"
