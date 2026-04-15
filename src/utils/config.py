"""
Configuration module for the TER project.
Centralizes all hyperparameters, paths, and settings.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CORPUS_DIR = os.path.join(PROJECT_ROOT, "corpus_age_etudiant")
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")

# ── Lists / external resources ────────────────────────────────────────────
LISTS_DIR = os.path.join(PROJECT_ROOT, "lists")
LEXIQUE4_PATH = os.path.join(LISTS_DIR, "Lexique4_syllabes", "Lexique4.tsv")
ACCENTED_CHARS_CSV = os.path.join(LISTS_DIR, "Characteres-accentues",
                                  "Characteres-accentues",
                                  "french_accented_characters_with_unicode.csv")
STOP_WORDS_JSON = os.path.join(LISTS_DIR, "mots-vides-countwordsfree",
                               "mots-vides-countwordsfree",
                               "stop_words_french.json")

# ── Age range ─────────────────────────────────────────────────────────────────

MIN_AGE = 30
MAX_AGE = 70
 #NUM_CLASSES = MAX_AGE - MIN_AGE + 1    before it was :  41 classes
NUM_CLASSES = 4  # 4 classes: 30-39, 40-49, 50-59, 60-70

# ── Data config ───────────────────────────────────────────────────────────────

@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    corpus_dir: str = CORPUS_DIR
    n_test_per_age: int = 3           # fixed number of texts per age held for test
    random_seed: int = 42
    sequence_length: int = 5000       # number of words per sample ( tokens ) 
    stride: int = 5000                # CHANGE THIS FROM 500 TO 1000               # sliding window stride (for extracting multiple samples per text)
    min_age: int = MIN_AGE
    max_age: int = MAX_AGE


# ── Feature config ────────────────────────────────────────────────────────────

@dataclass
class FeatureConfig:
    """
    Hand-crafted features: 4 active columns (reduced redundancy).

    Kept global_freq only among frequency variants; doc-level freq_count / log_freq
    disabled as highly correlated. Pair with per-document StandardScaler in dataset.py.
    """

    # 1. Frequency (only corpus-level signal)
    use_freq_count: bool = False
    use_log_freq: bool = False
    use_freq_rank: bool = False
    use_global_freq: bool = True

    # 2. Word length
    use_char_count: bool = True
    use_syllable_count: bool = False

    # 3. Char composition
    use_vowel_ratio: bool = False
    use_accent_type: bool = False

    # 4. Lexical
    use_punctuation_type: bool = True
    use_pos_tag: bool = True

    # 5. Positional
    use_pos_in_sent: bool = False
    use_sent_length: bool = False
    use_is_boundary: bool = False

    # 6. Context
    use_adj_period: bool = False

    context_window: int = 5           # window size for context features

    @property
    def feature_dim(self) -> int:
        """Calculates exact input size based on active features."""
        return sum([
            self.use_freq_count, self.use_log_freq, self.use_freq_rank, self.use_global_freq,
            self.use_char_count, self.use_syllable_count,
            self.use_vowel_ratio, self.use_accent_type,
            self.use_punctuation_type, self.use_pos_tag,
            self.use_pos_in_sent, self.use_sent_length, self.use_is_boundary,
            self.use_adj_period
        ])

# ── Model config ──────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    """Configuration for the multi-branch 1D CNN (3 branches × 3 layers)."""
    feature_dim: int = 14             # will be set from FeatureConfig.feature_dim
    num_classes: int = NUM_CLASSES
    kernel_sizes: List[int] = field(default_factory=lambda: [3, 7, 13])
    num_filters: int = 64
    pool_size: int = 2                # MaxPool1d takes 1 out of every 2
    num_conv_layers: int = 3          # Conv1d+MaxPool layers stacked per branch
    dropout: float = 0.5

# ── Training config ───────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    """Training loop: stronger regularization vs prior 3e-3 / 1e-4 / lighter dropout."""
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 5e-4
    num_epochs: int = 50
    patience: int = 10                # early stopping patience
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 5
    device: str = "cuda"              # "cuda" or "cpu"
    checkpoint_dir: str = field(
        default_factory=lambda: os.environ.get("TER_CHECKPOINT_DIR", EXPERIMENTS_DIR)
    )
    experiment_name: str = "global_norm_v1"


# ── Convenience: build all configs ────────────────────────────────────────────

def get_default_configs():
    """Return a tuple of all default configs."""
    data_cfg = DataConfig()
    feat_cfg = FeatureConfig()
    model_cfg = ModelConfig(feature_dim=feat_cfg.feature_dim)
    train_cfg = TrainingConfig()
    return data_cfg, feat_cfg, model_cfg, train_cfg