"""
Train/test splitting utilities.

The test corpus must have a fixed number of texts per age and must NEVER
be used for training.  Two modes are supported:

  Option A – This corpus is the only data: split each age's 30 texts into
             train / test subsets (stratified).
  Option B – A separate training corpus exists: this entire corpus = test set.
"""

import os
import json
from collections import defaultdict
from typing import Dict, List, Tuple

from sklearn.model_selection import train_test_split

from src.data.parser import TextMetadata


def stratified_split(
    entries: List[TextMetadata],
    n_test_per_age: int = 3,
    random_seed: int = 42,
) -> Tuple[List[TextMetadata], List[TextMetadata]]:
    """
    Split the corpus into train and test sets, stratified by age.

    For each age, exactly `n_test_per_age` texts are assigned to the test set
    (e.g. 3 test, 27 train out of 30).

    Returns (train_entries, test_entries).
    """
    by_age: Dict[int, List[TextMetadata]] = defaultdict(list)
    for m in entries:
        by_age[m.age].append(m)

    train_all, test_all = [], []

    for age in sorted(by_age.keys()):
        texts = by_age[age]
        if len(texts) <= n_test_per_age:
            train_all.extend(texts)
            continue

        train_set, test_set = train_test_split(
            texts, test_size=n_test_per_age, random_state=random_seed
        )
        train_all.extend(train_set)
        test_all.extend(test_set)

    return train_all, test_all


def save_split(
    train_entries: List[TextMetadata],
    test_entries: List[TextMetadata],
    output_dir: str,
):
    """Save the train/test split to JSON files for reproducibility."""
    os.makedirs(output_dir, exist_ok=True)

    def to_records(entries):
        return [
            {
                "filepath": m.filepath,
                "filename": m.filename,
                "age": m.age,
                "author": f"{m.first_name} {m.last_name}",
                "title": m.title,
                "publication_year": m.publication_year,
            }
            for m in entries
        ]

    with open(os.path.join(output_dir, "train_split.json"), "w", encoding="utf-8") as f:
        json.dump(to_records(train_entries), f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_dir, "test_split.json"), "w", encoding="utf-8") as f:
        json.dump(to_records(test_entries), f, ensure_ascii=False, indent=2)

    print(f"Split saved: {len(train_entries)} train, {len(test_entries)} test -> {output_dir}")


def load_split(output_dir: str, corpus_dir: str) -> Tuple[List[str], List[str]]:
    """
    Load a previously saved split. Returns (train_filepaths, test_filepaths).
    Falls back to filepath stored in the JSON; if those don't exist, tries
    to reconstruct from filename + corpus_dir.
    """
    def _load(path):
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
        filepaths = []
        for r in records:
            fp = r["filepath"]
            if not os.path.exists(fp):
                fp = os.path.join(corpus_dir, r["filename"])
            filepaths.append(fp)
        return filepaths

    train_fps = _load(os.path.join(output_dir, "train_split.json"))
    test_fps = _load(os.path.join(output_dir, "test_split.json"))
    return train_fps, test_fps
