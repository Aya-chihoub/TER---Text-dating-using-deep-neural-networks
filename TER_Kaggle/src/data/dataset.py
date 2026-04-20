"""
PyTorch Dataset for the TER corpus.

Each text is:
  1. Tokenised (whitespace split).
  2. Encoded into a sequence of feature vectors via FormFeatureExtractor.
  3. (Optional) Globally scaled with a StandardScaler fit on TRAIN only.
  4. Sliced into fixed-length windows (with stride) to produce multiple samples.
  5. Labelled with the author's age (30–70  →  class index 0–40).
"""

from __future__ import annotations

import os
from collections import Counter
from typing import List, Optional, Tuple

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from src.data.parser import TextMetadata, load_text
from src.data.features import FormFeatureExtractor
from src.utils.config import MIN_AGE, MAX_AGE


def _doc_heartbeat_interval(n_docs: int) -> int:
    """How often to print progress (avoids Kaggle idle disconnect during long spaCy passes)."""
    raw = os.environ.get("TER_DATA_PROGRESS_EVERY", "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    if n_docs <= 20:
        return 1
    if n_docs <= 100:
        return 5
    return 10


def collect_train_matrix_for_scaler(
    train_entries: List[TextMetadata],
    extractor: FormFeatureExtractor,
    sequence_length: int,
    stride: int,
) -> np.ndarray:
    """
    Stack every token-row from all training windows (and all rows from short texts)
    so StandardScaler can be fit on the training distribution only.
    """
    blocks: List[np.ndarray] = []
    n_docs = len(train_entries)
    step = _doc_heartbeat_interval(n_docs)
    for i, meta in enumerate(train_entries):
        if i % step == 0 or i == n_docs - 1:
            print(f"  [scaler fit rows] document {i + 1}/{n_docs} …", flush=True)
        text = load_text(meta.filepath)
        tokens = text.split()
        if not tokens:
            continue
        feats = extractor.extract(tokens)
        n = len(tokens)
        if n < sequence_length:
            blocks.append(feats)
        else:
            for start in range(0, n - sequence_length + 1, stride):
                blocks.append(feats[start : start + sequence_length])
    if not blocks:
        d = extractor.feature_dim
        return np.empty((0, d), dtype=np.float64)
    return np.vstack(blocks).astype(np.float64, copy=False)


def _transform_features(scaler: Optional[StandardScaler], arr: np.ndarray) -> np.ndarray:
    if scaler is None:
        return np.asarray(arr, dtype=np.float32)
    out = scaler.transform(np.asarray(arr, dtype=np.float64))
    return np.asarray(out, dtype=np.float32)


class TextAgeDataset(Dataset):
    """
    PyTorch Dataset that yields (feature_sequence, age_label) pairs.

    If ``feature_scaler`` is set, it must be fitted on the training set only;
    val/test use ``transform`` (never ``fit``).
    """

    def __init__(
        self,
        entries: List[TextMetadata],
        extractor: FormFeatureExtractor,
        sequence_length: int = 512,
        stride: int = 256,
        feature_scaler: Optional[StandardScaler] = None,
        progress_tag: str = "dataset",
    ):
        self.extractor = extractor
        self.sequence_length = sequence_length
        self.stride = stride
        self.feature_scaler = feature_scaler
        self._progress_tag = progress_tag

        self.samples: List[Tuple[np.ndarray, int]] = []
        self._build(entries)

    def _build(self, entries: List[TextMetadata]):
        """Load texts, extract features, optional global scaling, slice into windows."""

        self.chunk_doc_ids: List[int] = []

        n_docs = len(entries)
        step = _doc_heartbeat_interval(n_docs)
        for doc_id, meta in enumerate(entries):
            if doc_id % step == 0 or doc_id == n_docs - 1:
                print(f"  [{self._progress_tag}] document {doc_id + 1}/{n_docs} …", flush=True)
            text = load_text(meta.filepath)
            tokens = text.split()

            if len(tokens) < self.sequence_length:
                feats = _transform_features(
                    self.feature_scaler, self.extractor.extract(tokens)
                )
                padded = np.zeros(
                    (self.sequence_length, self.extractor.feature_dim),
                    dtype=np.float32,
                )
                padded[: len(tokens)] = feats
                label = meta.age - MIN_AGE
                self.samples.append((padded, label))
                self.chunk_doc_ids.append(doc_id)
                continue

            feats = self.extractor.extract(tokens)
            if self.feature_scaler is not None:
                feats = _transform_features(self.feature_scaler, feats)

            label = meta.age - MIN_AGE
            for start in range(0, len(tokens) - self.sequence_length + 1, self.stride):
                window = feats[start : start + self.sequence_length]
                self.samples.append((window.astype(np.float32, copy=False), label))
                self.chunk_doc_ids.append(doc_id)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        feats, label = self.samples[idx]
        return torch.from_numpy(feats), label

    def get_feature_names(self) -> List[str]:
        return self.extractor.feature_names()


def build_datasets(
    train_entries: List[TextMetadata],
    test_entries: List[TextMetadata],
    extractor: Optional[FormFeatureExtractor] = None,
    sequence_length: int = 512,
    train_stride: int = 256,
    test_stride: Optional[int] = None,
) -> Tuple[TextAgeDataset, TextAgeDataset, StandardScaler]:
    """
    Build train / val (or test) datasets with the same global scaler.

    The scaler is fit on all training token-rows (from sliding windows and short texts),
    then applied via ``transform`` when building both splits.
    """
    if extractor is None:
        extractor = FormFeatureExtractor()

    if test_stride is None:
        test_stride = sequence_length

    print("Computing global word frequencies from training corpus ...")
    global_freq: Counter = Counter()
    n_tr = len(train_entries)
    g_step = _doc_heartbeat_interval(n_tr)
    for i, meta in enumerate(train_entries):
        if i % g_step == 0 or i == n_tr - 1:
            print(f"  [global freq] document {i + 1}/{n_tr} …", flush=True)
        text = load_text(meta.filepath)
        global_freq.update(t.lower() for t in text.split())
    extractor.set_global_freq(global_freq)
    print(f"  → {len(global_freq)} unique words, {sum(global_freq.values())} total tokens")

    print("Fitting global StandardScaler on training features ...")
    fit_mat = collect_train_matrix_for_scaler(
        train_entries, extractor, sequence_length, train_stride
    )
    scaler = StandardScaler()
    if fit_mat.shape[0] == 0:
        scaler.fit(np.zeros((1, extractor.feature_dim), dtype=np.float64))
    else:
        scaler.fit(fit_mat)
    print(f"  → fitted on {fit_mat.shape[0]} token-rows (train only)")
    if scaler.scale_ is not None and len(scaler.mean_) >= 3:
        print(f"  → mean (first 3): {scaler.mean_[:3]}")
        print(f"  → scale (first 3): {scaler.scale_[:3]}")

    print(f"Building train dataset ({len(train_entries)} texts, "
          f"seq_len={sequence_length}, stride={train_stride}) ...")
    train_ds = TextAgeDataset(
        train_entries,
        extractor,
        sequence_length=sequence_length,
        stride=train_stride,
        feature_scaler=scaler,
        progress_tag="train windows",
    )
    print(f"  → {len(train_ds)} train samples")

    print(f"Building val/test dataset ({len(test_entries)} texts, "
          f"seq_len={sequence_length}, stride={test_stride}) ...")
    test_ds = TextAgeDataset(
        test_entries,
        extractor,
        sequence_length=sequence_length,
        stride=test_stride,
        feature_scaler=scaler,
        progress_tag="val windows",
    )
    print(f"  → {len(test_ds)} samples")

    return train_ds, test_ds, scaler
