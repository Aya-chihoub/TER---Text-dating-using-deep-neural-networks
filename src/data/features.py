"""
Hand-crafted feature extraction for word forms.

Each word in a text is encoded as a fixed-size numerical vector
of measurable characteristics.  NO embeddings -- only explicit,
interpretable features.

Feature groups
--------------
1. Frequency        – frequency count, log frequency, frequency rank,
                      global frequency in training corpus
2. Word length      – character count, syllable count (Lexique4 lookup)
3. Char composition – vowel ratio, accent type code (CSV accentues)
4. Lexical          – punctuation type, POS tag (spaCy French model)
5. Positional       – absolute position in sentence, sentence length, boundary flag
6. Context          – adjacent to a period '.' (word after or before a full stop)

External resources
------------------
- Lexique4.tsv                  → syllable_count (col 26_SyllNb, ~190k words)
- french_accented_chars.csv     → accent_type + VOWELS set
- spaCy fr_core_news_sm model   → Universal POS tags (ADJ, NOUN, PROPN, VERB, …)
"""

from __future__ import annotations

import csv
import math
import os
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import spacy
from spacy.tokens import Doc

from src.utils.config import LEXIQUE4_PATH, ACCENTED_CHARS_CSV


# ── spaCy French model (lazy-loaded) ────────────────────────────────────────

_NLP: Optional[spacy.language.Language] = None


def _get_nlp() -> spacy.language.Language:
    """Lazy-load the spaCy French model, keeping only POS-relevant pipes."""
    global _NLP
    if _NLP is None:
        _NLP = spacy.load(
            "fr_core_news_sm",
            disable=["parser", "ner", "lemmatizer", "attribute_ruler"],
        )
    return _NLP


# ── POS tag mapping (Universal POS → numeric code) ──────────────────────────

POS_TAG_MAP = {
    "ADJ":   1,
    "NOUN":  2,
    "PROPN": 3,
    "VERB":  4,
    "ADV":   5,
    "ADP":   6,
    "DET":   7,
    "PRON":  8,
    "CCONJ": 9,
    "SCONJ": 10,
    "AUX":   11,
    "NUM":   12,
    "INTJ":  13,
    "PUNCT": 14,
}


# ── Lexique4 syllable lookup ─────────────────────────────────────────────────

def _load_lexique4_syllables(path: str) -> Dict[str, int]:
    """Build a word → syllable_count dict from Lexique4.tsv."""
    syll_map: Dict[str, int] = {}
    if not os.path.isfile(path):
        return syll_map
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # skip header
        for row in reader:
            if len(row) < 27:
                continue
            mot = row[0].lower().strip()
            try:
                nb = int(row[25])  # 26_SyllNb (0-indexed col 25)
            except (ValueError, IndexError):
                continue
            if mot and nb > 0:
                syll_map[mot] = nb
    return syll_map


SYLLABLE_LOOKUP = _load_lexique4_syllables(LEXIQUE4_PATH)


# ── Accent type map (from CSV) ───────────────────────────────────────────────

_ACCENT_TYPE_CODES = {
    "acute": 1,
    "grave": 2,
    "circumflex": 3,
    "diaeresis": 4,
    "cedilla": 5,
    "ligature": 6,
}


def _load_accent_map(path: str) -> Dict[str, int]:
    """Build char → accent_code dict from the accented characters CSV."""
    accent_map: Dict[str, int] = {}
    if not os.path.isfile(path):
        return accent_map
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            char = row["lowercase"]
            atype = row["accent_type"].strip()
            if atype in _ACCENT_TYPE_CODES:
                accent_map[char] = _ACCENT_TYPE_CODES[atype]
    return accent_map


ACCENT_MAP = _load_accent_map(ACCENTED_CHARS_CSV)

_VOWELS_BASE = frozenset("aeiouy")
_VOWEL_ACCENT_TYPES = frozenset({1, 2, 3, 4, 6})


def _build_vowels(accent_map: Dict[str, int]) -> frozenset:
    """Build complete French vowels set from base vowels + accented chars CSV."""
    vowels = set(_VOWELS_BASE)
    for char, code in accent_map.items():
        if code in _VOWEL_ACCENT_TYPES:
            vowels.add(char)
    return frozenset(vowels)


VOWELS = _build_vowels(ACCENT_MAP)


# ── Punctuation type map ──────────────────────────────────────────────────────

# Tirets : "-" (U+002D) ; U+2013 demi-cadratin ; U+2014 cadratin (dialogue, incises).
# Apostrophe : "'" (U+0027) ; U+2019 apostrophe typographique (élision, « l'auteur »).
PUNCTUATION_MAP = {
    ".": 1,
    ",": 2,
    ";": 3,
    ":": 4,
    "!": 5,
    "?": 6,
    '"': 7,
    "'": 8,
    "(": 9,
    ")": 10,
    "-": 11,
    "\u2013": 12,
    "…": 13,
    "«": 14,
    "»": 15,
    "\u2019": 16,
    "\u2014": 17,
}

PUNCTUATION_CHARS = frozenset(PUNCTUATION_MAP.keys())


# ── Helper functions ──────────────────────────────────────────────────────────

def _is_punctuation(token: str) -> bool:
    """Return True if the token is pure punctuation."""
    return all(ch in PUNCTUATION_CHARS for ch in token) and len(token) > 0


def _punctuation_type(token: str) -> int:
    """Return the punctuation type code (0 if not punctuation)."""
    if len(token) == 1 and token in PUNCTUATION_MAP:
        return PUNCTUATION_MAP[token]
    if _is_punctuation(token):
        return PUNCTUATION_MAP.get(token[0], 0)
    return 0


def _accent_type(word: str) -> int:
    """Return accent type code based on first accented char (0 if none)."""
    for ch in word.lower():
        if ch in ACCENT_MAP:
            return ACCENT_MAP[ch]
    return 0


def _vowel_ratio(word: str) -> float:
    """Proportion of vowels in the word (0.0 if empty)."""
    if not word:
        return 0.0
    return sum(1 for ch in word.lower() if ch in VOWELS) / len(word)


def _syllable_count(word: str) -> int:
    """
    Syllable count using Lexique4 lookup with heuristic fallback.
    For unknown words, falls back to counting vowel groups.
    """
    w = word.lower().strip()
    if w in SYLLABLE_LOOKUP:
        return SYLLABLE_LOOKUP[w]
    count = 0
    prev_vowel = False
    for ch in w:
        is_v = ch in VOWELS
        if is_v and not prev_vowel:
            count += 1
        prev_vowel = is_v
    return max(count, 1) if word else 0


# ── Sentence boundary detection ───────────────────────────────────────────────

def _find_sentence_boundaries(tokens: List[str]) -> List[int]:
    """
    Return a list of the same length as `tokens` where each element
    is the sentence index that token belongs to.

    Segmentation très simple : après chaque jeton exactement égal à
    '.', '!' ou '?' (séparé par des espaces dans le texte), l'indice
    de phrase augmente. Le jeton de ponctuation reste dans la phrase
    qu'il termine ; le mot suivant commence une nouvelle phrase.

    Exemple : ``Bonjour . Le`` → phrase 0 = [Bonjour, .], phrase 1 = [Le].
    """
    sentence_ids: List[int] = []
    sid = 0
    for tok in tokens:
        sentence_ids.append(sid)
        if tok in (".", "!", "?"):
            sid += 1
    return sentence_ids


# ── Main feature extractor ────────────────────────────────────────────────────

class FormFeatureExtractor:
    """
    Extracts a fixed-size feature vector for every word (form) in a text.

    Usage
    -----
    >>> extractor = FormFeatureExtractor()
    >>> extractor.set_global_freq(global_counter)   # from training corpus
    >>> tokens = text.split()
    >>> matrix = extractor.extract(tokens)           # shape (n, feature_dim)
    """

    def __init__(
        self,
        use_frequency: bool = True,
        use_word_length: bool = True,
        use_char_composition: bool = True,
        use_lexical: bool = True,
        use_positional: bool = True,
        use_context: bool = True,
        context_window: int = 5,
    ):
        self.use_frequency = use_frequency
        self.use_word_length = use_word_length
        self.use_char_composition = use_char_composition
        self.use_lexical = use_lexical
        self.use_positional = use_positional
        self.use_context = use_context
        self.context_window = context_window

        self._global_freq: Optional[Counter] = None
        self._global_total: int = 0

    # ── Global frequency (training corpus) ─────────────────────────────

    def set_global_freq(self, freq: Counter) -> None:
        """Set the global word frequency table (computed from training corpus)."""
        self._global_freq = freq
        self._global_total = sum(freq.values())

    # ── Dimension ──────────────────────────────────────────────────────

    @property
    def feature_dim(self) -> int:
        """Total number of features per word."""
        dim = 0
        if self.use_frequency:
            dim += 4    # freq_in_text, log_freq, freq_rank, global_freq
        if self.use_word_length:
            dim += 2
        if self.use_char_composition:
            dim += 2
        if self.use_lexical:
            dim += 2    # punctuation_type, pos_tag
        if self.use_positional:
            dim += 3
        if self.use_context:
            dim += 1    # adjacent_to_period
        return dim

    # ── Extraction ─────────────────────────────────────────────────────

    def extract(self, tokens: List[str]) -> np.ndarray:
        """
        Encode every token as a feature vector.

        Parameters
        ----------
        tokens : list of str
            The whitespace-split tokens of the text.

        Returns
        -------
        features : np.ndarray, shape (len(tokens), self.feature_dim)
        """
        n = len(tokens)
        if n == 0:
            return np.zeros((0, self.feature_dim), dtype=np.float32)

        # ── Pre-compute text-level stats ──────────────────────────────
        word_freq = Counter(t.lower() for t in tokens)

        sorted_freqs = sorted(set(word_freq.values()), reverse=True)
        freq_to_rank = {f: r + 1 for r, f in enumerate(sorted_freqs)}
        word_rank = {w: freq_to_rank[f] for w, f in word_freq.items()}

        sentence_ids = _find_sentence_boundaries(tokens)

        sent_start: dict = {}
        sent_end: dict = {}
        token_offset_in_sent = [0] * n
        offset_counter: dict = {}
        for i, sid in enumerate(sentence_ids):
            if sid not in sent_start:
                sent_start[sid] = i
                offset_counter[sid] = 0
            sent_end[sid] = i
            token_offset_in_sent[i] = offset_counter[sid]
            offset_counter[sid] += 1

        sent_lengths = {
            sid: sent_end[sid] - sent_start[sid] + 1 for sid in sent_start
        }

        # ── POS tagging via spaCy ─────────────────────────────────────
        pos_tags = [0] * n
        if self.use_lexical:
            nlp = _get_nlp()
            spaces = [True] * n
            spaces[-1] = False
            doc = Doc(nlp.vocab, words=tokens, spaces=spaces)
            for _, proc in nlp.pipeline:
                doc = proc(doc)
            for j, spacy_tok in enumerate(doc):
                pos_tags[j] = POS_TAG_MAP.get(spacy_tok.pos_, 0)
            # Pure punctuation tokens (apostrophe seule, tiret, etc.) : spaCy peut predire
            # des POS absurdes (VERB) car ces formes sont hors distribution apres split().
            # On aligne pos_tag sur la grammaire universelle : PUNCT.
            for j in range(n):
                if _punctuation_type(tokens[j]) > 0:
                    pos_tags[j] = POS_TAG_MAP["PUNCT"]

        # ── Build feature matrix ──────────────────────────────────────
        feats = np.zeros((n, self.feature_dim), dtype=np.float32)

        for i, tok in enumerate(tokens):
            col = 0
            tok_lower = tok.lower()

            # 1. Frequency features
            if self.use_frequency:
                feats[i, col] = word_freq[tok_lower]
                col += 1
                feats[i, col] = math.log1p(word_freq[tok_lower])
                col += 1
                feats[i, col] = word_rank[tok_lower]
                col += 1
                if self._global_freq is not None:
                    feats[i, col] = self._global_freq.get(tok_lower, 0)
                col += 1

            # 2. Word length features
            if self.use_word_length:
                feats[i, col] = len(tok)
                col += 1
                feats[i, col] = _syllable_count(tok)
                col += 1

            # 3. Character composition
            if self.use_char_composition:
                feats[i, col] = _vowel_ratio(tok)
                col += 1
                feats[i, col] = _accent_type(tok)
                col += 1

            # 4. Lexical features
            if self.use_lexical:
                feats[i, col] = _punctuation_type(tok)
                col += 1
                feats[i, col] = pos_tags[i]
                col += 1

            # 5. Positional features
            if self.use_positional:
                feats[i, col] = token_offset_in_sent[i]
                col += 1
                sid = sentence_ids[i]
                feats[i, col] = sent_lengths[sid]
                col += 1
                is_boundary = (i == sent_start[sid]) or (i == sent_end[sid])
                feats[i, col] = float(is_boundary)
                col += 1

            # 6. Context: adjacent to a period '.' only (not comma, etc.)
            #    If previous token is '.', this word is right after a sentence end →
            #    typically sentence-initial; if next token is '.', right before a period
            #    → typically sentence-final. (Distinct from ! / ? which do not trigger this bit.)
            if self.use_context:
                prev_is_period = i > 0 and tokens[i - 1] == "."
                next_is_period = i + 1 < n and tokens[i + 1] == "."
                feats[i, col] = 1.0 if (prev_is_period or next_is_period) else 0.0
                col += 1

        return feats

    def feature_names(self) -> List[str]:
        """Return human-readable names for each feature column."""
        names: List[str] = []
        if self.use_frequency:
            names += ["freq_in_text", "log_freq", "freq_rank", "global_freq"]
        if self.use_word_length:
            names += ["word_length", "syllable_count"]
        if self.use_char_composition:
            names += ["vowel_ratio", "accent_type"]
        if self.use_lexical:
            names += ["punctuation_type", "pos_tag"]
        if self.use_positional:
            names += ["pos_in_sentence", "sentence_length", "is_sentence_boundary"]
        if self.use_context:
            names += ["adjacent_to_period"]
        return names
