"""
Configuration module for the TER project.
Centralizes all hyperparameters, paths, and settings.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CORPUS_DIR = os.path.join(PROJECT_ROOT, "corpus_age_etudiant", "corpus_age_etudiant")
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
NUM_CLASSES = MAX_AGE - MIN_AGE + 1  # 41 classes


# ── Data config ───────────────────────────────────────────────────────────────

@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    corpus_dir: str = CORPUS_DIR
    n_test_per_age: int = 3           # fixed number of texts per age held for test
    random_seed: int = 42
    sequence_length: int = 1000       # number of words per sample
    stride: int = 500                 # sliding window stride (for extracting multiple samples per text)
    min_age: int = MIN_AGE
    max_age: int = MAX_AGE


# ── Feature config ────────────────────────────────────────────────────────────

@dataclass
class FeatureConfig:
    """Configuration for the hand-crafted feature extraction."""
    # Which feature groups to enable
    use_frequency: bool = True
    use_word_length: bool = True
    use_char_composition: bool = True
    use_lexical: bool = True
    use_positional: bool = True
    use_context: bool = True
    context_window: int = 5           # window size for context features

    @property
    def feature_dim(self) -> int:
        """Compute the total feature vector dimension based on enabled groups."""
        dim = 0
        if self.use_frequency:
            dim += 4    # freq_in_text, log_freq, freq_rank, global_freq
        if self.use_word_length:
            dim += 2    # word_length, syllable_count
        if self.use_char_composition:
            dim += 2    # vowel_ratio, accent_type
        if self.use_lexical:
            dim += 2    # punctuation_type, pos_tag
        if self.use_positional:
            dim += 3    # pos_in_sentence, sentence_length, is_sentence_boundary
        if self.use_context:
            dim += 1    # adjacent_to_period
        return dim


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
    dropout: float = 0.3


# ── Training config ───────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    """Configuration for the training loop."""
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_epochs: int = 50
    patience: int = 10                # early stopping patience
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 5
    device: str = "cuda"              # "cuda" or "cpu"
    checkpoint_dir: str = EXPERIMENTS_DIR
    experiment_name: str = "default"


# ── Convenience: build all configs ────────────────────────────────────────────

def get_default_configs():
    """Return a tuple of all default configs."""
    data_cfg = DataConfig()
    feat_cfg = FeatureConfig()
    model_cfg = ModelConfig(feature_dim=feat_cfg.feature_dim)
    train_cfg = TrainingConfig()
    return data_cfg, feat_cfg, model_cfg, train_cfg
