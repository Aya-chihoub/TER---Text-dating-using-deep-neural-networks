"""
PyTorch Dataset for the TER corpus.

Each text is:
  1. Tokenised (whitespace split).
  2. Encoded into a sequence of feature vectors via FormFeatureExtractor.
  3. Sliced into fixed-length windows (with stride) to produce multiple samples.
  4. Labelled with the author's age (30–70  →  class index 0–40).

The Dataset can be built from a list of TextMetadata objects or from a
previously saved train/test split.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.parser import TextMetadata, load_text
from src.data.features import FormFeatureExtractor
from src.utils.config import MIN_AGE, MAX_AGE


class TextAgeDataset(Dataset):
    """
    PyTorch Dataset that yields (feature_sequence, age_label) pairs.

    Each element is a fixed-length window of word-level feature vectors
    extracted from the corpus texts.

    Parameters
    ----------
    entries : list of TextMetadata
        The texts to include (train or test split).
    extractor : FormFeatureExtractor
        The feature extractor to use.
    sequence_length : int
        Number of tokens per sample window.
    stride : int
        Step size for the sliding window. If stride < sequence_length,
        windows overlap (data augmentation for training).
        If stride == sequence_length, no overlap.
    """

    def __init__(
        self,
        entries: List[TextMetadata],
        extractor: FormFeatureExtractor,
        sequence_length: int = 512,
        stride: int = 256,
    ):
        self.extractor = extractor
        self.sequence_length = sequence_length
        self.stride = stride

        # Pre-compute all samples: list of (feature_matrix, label)
        self.samples: List[Tuple[np.ndarray, int]] = []
        self._build(entries)

    def _build(self, entries: List[TextMetadata]):
        """Load texts, extract features, and slice into windows."""
        for meta in entries:
            text = load_text(meta.filepath)
            tokens = text.split()

            if len(tokens) < self.sequence_length:
                # Text too short: pad with zeros
                feats = self.extractor.extract(tokens)
                padded = np.zeros(
                    (self.sequence_length, self.extractor.feature_dim),
                    dtype=np.float32,
                )
                padded[:len(tokens)] = feats
                label = meta.age - MIN_AGE  # 30→0, 31→1, …, 70→40
                self.samples.append((padded, label))
                continue

            # Extract features for the full text
            feats = self.extractor.extract(tokens)

            # Sliding window
            label = meta.age - MIN_AGE
            for start in range(0, len(tokens) - self.sequence_length + 1, self.stride):
                window = feats[start : start + self.sequence_length]
                self.samples.append((window, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        feats, label = self.samples[idx]
        return torch.from_numpy(feats), label

    def get_feature_names(self) -> List[str]:
        """Return the feature names from the extractor."""
        return self.extractor.feature_names()


def build_datasets(
    train_entries: List[TextMetadata],
    test_entries: List[TextMetadata],
    extractor: Optional[FormFeatureExtractor] = None,
    sequence_length: int = 512,
    train_stride: int = 256,
    test_stride: Optional[int] = None,
) -> Tuple[TextAgeDataset, TextAgeDataset]:
    """
    Convenience function to build train and test datasets.

    For training, we use overlapping windows (stride < sequence_length)
    to augment data.  For testing, we use non-overlapping windows
    (stride == sequence_length) by default.
    """
    if extractor is None:
        extractor = FormFeatureExtractor()

    if test_stride is None:
        test_stride = sequence_length   # no overlap for test

    print(f"Building train dataset ({len(train_entries)} texts, "
          f"seq_len={sequence_length}, stride={train_stride}) ...")
    train_ds = TextAgeDataset(
        train_entries, extractor,
        sequence_length=sequence_length,
        stride=train_stride,
    )
    print(f"  → {len(train_ds)} train samples")

    print(f"Building test dataset ({len(test_entries)} texts, "
          f"seq_len={sequence_length}, stride={test_stride}) ...")
    test_ds = TextAgeDataset(
        test_entries, extractor,
        sequence_length=sequence_length,
        stride=test_stride,
    )
    print(f"  → {len(test_ds)} test samples")

    return train_ds, test_ds
