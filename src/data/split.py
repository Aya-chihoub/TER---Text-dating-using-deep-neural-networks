"""
Train/test splitting utilities.

Splits the corpus into three distinct sets:
  1. Train (85% of the 90% block) - Used to train the model.
  2. Validation (15% of the 90% block) - Used for early stopping.
  3. Test (10% overall) - NEVER used during training; final evaluation only.
"""

import os
import json
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from src.data.parser import TextMetadata

def split_train_val_test(
    entries: List[TextMetadata],
    test_size: float = 0.10,
    val_size: float = 0.15,
    random_seed: int = 42,
) -> Tuple[List[TextMetadata], List[TextMetadata], List[TextMetadata]]:
    """
    Split the corpus into Train, Validation, and Test sets, stratified by age.
    """
    # Extract ages to ensure we stratify (balance) the ages in all splits
    ages = [m.age for m in entries]

    # Step 1: Carve out the 10% Final Test Set
    train_val_entries, test_entries = train_test_split(
        entries, 
        test_size=test_size, 
        stratify=ages, 
        random_state=random_seed
    )

    # Step 2: Split the remaining 90% into 85% Train and 15% Validation
    train_val_ages = [m.age for m in train_val_entries]
    train_entries, val_entries = train_test_split(
        train_val_entries, 
        test_size=val_size, 
        stratify=train_val_ages, 
        random_state=random_seed
    )

    return train_entries, val_entries, test_entries

def save_split(
    train_entries: List[TextMetadata],
    val_entries: List[TextMetadata],
    test_entries: List[TextMetadata],
    output_dir: str,
):
    """Save the train/val/test splits to JSON files for reproducibility."""
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

    with open(os.path.join(output_dir, "val_split.json"), "w", encoding="utf-8") as f:
        json.dump(to_records(val_entries), f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_dir, "test_split.json"), "w", encoding="utf-8") as f:
        json.dump(to_records(test_entries), f, ensure_ascii=False, indent=2)

    print(f"Splits saved: {len(train_entries)} Train, {len(val_entries)} Val, {len(test_entries)} Test -> {output_dir}")