"""
Persist and reload the global normalization artifacts.

The fitted :class:`~sklearn.preprocessing.StandardScaler` and the training-
corpus word frequency table are computed once per run inside
:func:`src.data.dataset.build_datasets`. This module saves them to disk (both as
a human-readable CSV *table* and as a pickle ready for reuse) so that a brand
new, unseen test set can be normalized with the **exact** statistics used at
training time.

Contents written to ``<out_dir>/``:

- ``feature_normalization_stats.csv`` — one row per feature column with
  mean / scale / var / n_samples_seen (the "table" you can eyeball).
- ``feature_scaler.pkl`` — pickled ``StandardScaler`` for lossless reuse.
- ``global_word_frequencies.csv`` — ``word,frequency`` rows sorted by frequency
  descending (used by the ``global_freq`` feature at inference time).
- ``normalization_manifest.json`` — metadata (feature names, sequence_length,
  stride, n_fit_rows, total_tokens, vocab_size, created_at).

Load them back with :func:`load_normalization_artifacts` to rebuild an
extractor + scaler identical to the training-time one.
"""

from __future__ import annotations

import csv
import json
import pickle
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.data.features import FormFeatureExtractor
from src.utils.config import FeatureConfig


# ── Filenames ────────────────────────────────────────────────────────────────
STATS_CSV = "feature_normalization_stats.csv"
SCALER_PKL = "feature_scaler.pkl"
GLOBAL_FREQ_CSV = "global_word_frequencies.csv"
MANIFEST_JSON = "normalization_manifest.json"


def _stats_rows(
    scaler: StandardScaler,
    feature_names: List[str],
) -> List[dict]:
    """Build one dict per feature column describing its normalization stats."""
    mean = np.asarray(scaler.mean_, dtype=np.float64)
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    var = np.asarray(getattr(scaler, "var_", scale ** 2), dtype=np.float64)
    n_seen = int(getattr(scaler, "n_samples_seen_", 0))
    if len(feature_names) != len(mean):
        raise ValueError(
            f"feature_names length {len(feature_names)} != scaler dim {len(mean)}"
        )
    rows: List[dict] = []
    for i, name in enumerate(feature_names):
        rows.append(
            {
                "feature_index": i,
                "feature_name": name,
                "mean": float(mean[i]),
                "scale": float(scale[i]),
                "var": float(var[i]),
                "n_samples_seen": n_seen,
            }
        )
    return rows


def save_normalization_artifacts(
    scaler: StandardScaler,
    extractor: FormFeatureExtractor,
    out_dir: str | Path,
    *,
    sequence_length: int,
    stride: int,
    n_fit_rows: int,
    feature_names: Optional[List[str]] = None,
) -> Path:
    """
    Save the fitted scaler, the stats table, the training word-frequency table,
    and a small JSON manifest inside ``out_dir``. Returns the resolved path.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if feature_names is None:
        feature_names = extractor.feature_names()

    # 1. Stats CSV (the "table" of mean/std per feature column).
    rows = _stats_rows(scaler, feature_names)
    with (out / STATS_CSV).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["feature_index", "feature_name", "mean", "scale", "var",
                        "n_samples_seen"],
        )
        writer.writeheader()
        writer.writerows(rows)

    # 2. Pickle the exact sklearn scaler for lossless reuse.
    with (out / SCALER_PKL).open("wb") as f:
        pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)

    # 3. Global word-frequency table (word, frequency), sorted desc.
    global_freq = extractor.get_global_freq() or Counter()
    sorted_items = sorted(global_freq.items(), key=lambda kv: (-kv[1], kv[0]))
    with (out / GLOBAL_FREQ_CSV).open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "frequency"])
        writer.writerows(sorted_items)

    # 4. Manifest (small JSON so the evaluation script can sanity-check).
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "feature_names": feature_names,
        "feature_dim": len(feature_names),
        "sequence_length": int(sequence_length),
        "stride": int(stride),
        "n_fit_rows": int(n_fit_rows),
        "vocab_size": len(global_freq),
        "total_tokens": extractor.global_total_tokens,
        "files": {
            "stats_csv": STATS_CSV,
            "scaler_pkl": SCALER_PKL,
            "global_freq_csv": GLOBAL_FREQ_CSV,
        },
    }
    with (out / MANIFEST_JSON).open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return out


def load_global_frequencies(path: str | Path) -> Counter:
    """Load the global word-frequency table written by :func:`save_normalization_artifacts`."""
    c: Counter = Counter()
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header != ["word", "frequency"]:
            raise ValueError(f"Unexpected header in {path}: {header}")
        for row in reader:
            if len(row) != 2:
                continue
            try:
                c[row[0]] = int(row[1])
            except ValueError:
                continue
    return c


def load_scaler(path: str | Path) -> StandardScaler:
    """Load the pickled ``StandardScaler`` saved at training time."""
    with Path(path).open("rb") as f:
        return pickle.load(f)


def load_normalization_artifacts(
    in_dir: str | Path,
    feature_config: Optional[FeatureConfig] = None,
) -> Tuple[FormFeatureExtractor, StandardScaler, dict]:
    """
    Reload the fitted scaler + global word frequencies and return an extractor
    ready to transform a brand new test set using the *exact* training stats.

    Parameters
    ----------
    in_dir : str | Path
        Directory containing the artifacts written by
        :func:`save_normalization_artifacts`.
    feature_config : FeatureConfig, optional
        Feature flags. Must match the training-time configuration. Defaults to
        :class:`~src.utils.config.FeatureConfig` (the project default).

    Returns
    -------
    extractor, scaler, manifest
    """
    d = Path(in_dir)
    manifest_path = d / MANIFEST_JSON
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    cfg = feature_config if feature_config is not None else FeatureConfig()
    extractor = FormFeatureExtractor(config=cfg)

    # Sanity check: feature layout must match the training-time one.
    trained_names = manifest.get("feature_names", [])
    runtime_names = extractor.feature_names()
    if trained_names and trained_names != runtime_names:
        raise ValueError(
            "Feature configuration mismatch between runtime and saved artifacts.\n"
            f"  runtime: {runtime_names}\n"
            f"  trained: {trained_names}"
        )

    global_freq = load_global_frequencies(d / GLOBAL_FREQ_CSV)
    extractor.set_global_freq(global_freq)

    scaler = load_scaler(d / SCALER_PKL)
    return extractor, scaler, manifest


def preview_stats(
    scaler: StandardScaler,
    feature_names: List[str],
    stream: Optional["object"] = None,
) -> str:
    """Return a small ASCII table of the feature stats (also prints it)."""
    header = f"{'idx':>3}  {'feature':<22}  {'mean':>14}  {'scale':>14}"
    line = "-" * len(header)
    out_lines = [header, line]
    for i, name in enumerate(feature_names):
        out_lines.append(
            f"{i:>3}  {name:<22}  {scaler.mean_[i]:>14.6g}  {scaler.scale_[i]:>14.6g}"
        )
    msg = "\n".join(out_lines)
    if stream is not None:
        print(msg, file=stream)
    else:
        print(msg)
    return msg
