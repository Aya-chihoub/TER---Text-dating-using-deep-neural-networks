"""
Hand-crafted feature extraction for word forms.

Each word in a text is encoded as a fixed-size numerical vector
of measurable characteristics.  NO embeddings -- only explicit,
interpretable features.

Feature groups
--------------
1. Frequency        – frequency count, log frequency, frequency rank
2. Word length      – character count, syllable count (Lexique4 lookup)
3. Char composition – vowel ratio, accent type code (CSV accentues)
4. Lexical          – stop-word type (countwordsfree), punctuation type,
                      prefix code, suffix code
5. Positional       – absolute position in sentence, sentence length, boundary flag
6. Context          – local punctuation density, followed by punctuation

External resources (lists/ folder)
-----------------------------------
- Lexique4.tsv              → syllable_count (col 26_SyllNb, ~190k words)
- french_accented_chars.csv → accent_type + VOWELS set
- stop_words_french.json    → stop_word_type (~496 French stop words)
- MorphoLex-FR suffixes.csv → suffix_code
- MorphoLex-FR prefixes.csv + Demonext → prefix_code
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import Counter
from typing import Dict, List

import numpy as np

from src.utils.config import LEXIQUE4_PATH, ACCENTED_CHARS_CSV, STOP_WORDS_JSON


# ── Lexique4 syllable lookup ─────────────────────────────────────────────────
# Source: Lexique4 (New, Pallier et al., 2026) – ~190 000 French word forms
# http://www.lexique.org
# Column 26_SyllNb gives the exact number of phonological syllables.

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
# Source: lists/Characteres-accentues/french_accented_characters_with_unicode.csv
# Maps each accented character to a type code:
# 1=aigu, 2=grave, 3=circonflexe, 4=trema, 5=cedille, 6=ligature

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

# Build VOWELS set: base vowels + accented vowels from the CSV
_VOWELS_BASE = frozenset("aeiouy")

_VOWEL_ACCENT_TYPES = frozenset({1, 2, 3, 4, 6})  # aigu, grave, circ, trema, ligature

def _build_vowels(accent_map: Dict[str, int]) -> frozenset:
    """Build complete French vowels set from base vowels + accented chars CSV.
    All accented characters except cedilla (ç) are vowels or vowel ligatures."""
    vowels = set(_VOWELS_BASE)
    for char, code in accent_map.items():
        if code in _VOWEL_ACCENT_TYPES:
            vowels.add(char)
    return frozenset(vowels)

VOWELS = _build_vowels(ACCENT_MAP)


# ── French stop words (countwordsfree) ───────────────────────────────────────
# Source: lists/mots-vides-countwordsfree/stop_words_french.json (~496 words)
# Each stop word is categorized by grammatical function:
# 0=not stop word, 1=determinant, 2=preposition, 3=conjonction,
# 4=pronom, 5=adverbe/negation, 6=auxiliaire/other

def _load_stop_words(path: str) -> frozenset:
    """Load the stop word set from the JSON file."""
    if not os.path.isfile(path):
        return frozenset()
    with open(path, encoding="utf-8") as f:
        return frozenset(w.lower() for w in json.load(f))

_STOP_WORDS_SET = _load_stop_words(STOP_WORDS_JSON)

_CAT_DETERMINANT = frozenset([
    "au", "aux", "ce", "ces", "cet", "cette", "des", "du",
    "l", "la", "le", "les",
    "ma", "mes", "mon", "nos", "notre",
    "sa", "ses", "son", "ta", "tes", "ton",
    "un", "une", "vos", "votre", "leur", "leurs",
])

_CAT_PREPOSITION = frozenset([
    "dans", "par", "pour", "sur", "avec", "de", "d", "en",
    "depuis", "devant", "entre", "envers", "hormis", "malgré",
    "parmi", "pendant", "sans", "selon", "sous", "vers", "via",
    "avant", "après", "chez", "contre", "durant", "excepté",
    "moyennant", "outre", "touchant", "concernant", "sauf",
])

_CAT_CONJONCTION = frozenset([
    "et", "ou", "mais", "que", "qu", "car", "donc", "ni",
    "sinon", "lorsque", "puisque", "quoique", "quand",
])

_CAT_PRONOM = frozenset([
    "c", "elle", "elles", "eux", "il", "ils", "j", "je", "lui",
    "m", "me", "moi", "n", "nous", "on",
    "s", "se", "t", "te", "toi", "tu", "vous", "y", "qui",
    "celui", "celle", "ceux", "celles", "chacun", "quiconque",
    "soi", "rien", "personne", "nul",
])

_CAT_ADVERBE = frozenset([
    "ne", "pas", "plus", "moins", "bien", "mal", "peu", "trop",
    "très", "aussi", "encore", "toujours", "jamais", "non",
    "plutôt", "surtout", "seulement", "tant",
])

STOP_WORD_MAP: dict[str, int] = {}
for _w in _CAT_DETERMINANT & _STOP_WORDS_SET:
    STOP_WORD_MAP[_w] = 1
for _w in _CAT_PREPOSITION & _STOP_WORDS_SET:
    STOP_WORD_MAP[_w] = 2
for _w in _CAT_CONJONCTION & _STOP_WORDS_SET:
    STOP_WORD_MAP[_w] = 3
for _w in _CAT_PRONOM & _STOP_WORDS_SET:
    STOP_WORD_MAP[_w] = 4
for _w in _CAT_ADVERBE & _STOP_WORDS_SET:
    STOP_WORD_MAP[_w] = 5
# Remaining stop words not in any specific category → auxiliaire/other
for _w in _STOP_WORDS_SET:
    if _w not in STOP_WORD_MAP:
        STOP_WORD_MAP[_w] = 6

FRENCH_STOP_WORDS = frozenset(STOP_WORD_MAP.keys())

# ── Punctuation type map ──────────────────────────────────────────────────────
PUNCTUATION_MAP = {
    ".": 1, ",": 2, ";": 3, ":": 4, "!": 5, "?": 6,
    '"': 7, "'": 8, "(": 9, ")": 10, "-": 11, "–": 12,
    "…": 13, "«": 14, "»": 15,
}

PUNCTUATION_CHARS = frozenset(PUNCTUATION_MAP.keys())

# ── Suffix map – grammatical categories ───────────────────────────────────────
# Source: standard French morphology / MorphoLex-FR
# https://github.com/hugomailhot/morpholex-fr
# 1=nom, 2=adverbe, 3=adjectif, 4=verbe imparfait,
# 5=verbe infinitif, 6=participe present, 7=participe passe
SUFFIX_MAP = {
    "tion": 1, "sion": 1, "eur": 1, "euse": 1,
    "iste": 1, "istes": 1, "age": 1, "ance": 1, "ence": 1,
    "ment": 2,
    "eux": 3, "euses": 3, "able": 3, "ible": 3,
    "ique": 3, "iques": 3,
    "ait": 4, "ais": 4, "aient": 4,
    "er": 5, "ir": 5, "re": 5,
    "ant": 6, "ants": 6, "ante": 6,
    "é": 7, "ée": 7, "és": 7, "ées": 7,
}

_MAX_SUFFIX = max(len(s) for s in SUFFIX_MAP)

# ── Prefix map – grammatical categories ───────────────────────────────────────
# Source: standard French morphology / Demonette-2
# https://www.peren-revues.fr/lexique/index.php?id=1242
# 1=repetition, 2=negation/privation, 3=temporel, 4=spatial/degre,
# 5=opposition, 6=association, 7=transformation, 8=reflexif
PREFIX_MAP = {
    "ré": 1, "re": 1,
    "dés": 2, "dé": 2, "in": 2, "im": 2, "il": 2, "ir": 2,
    "pré": 3, "post": 3,
    "sur": 4, "sous": 4, "super": 4, "ultra": 4, "extra": 4,
    "anti": 5, "contre": 5,
    "entre": 6, "inter": 6, "co": 6, "com": 6, "con": 6,
    "trans": 7, "en": 7, "em": 7,
    "auto": 8,
}

_MAX_PREFIX = max(len(p) for p in PREFIX_MAP)


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


def _suffix_code(word: str) -> int:
    """
    Return grammatical category code based on suffix (0 if none).
    Checks longest suffixes first so '-tion' beats '-on'.
    """
    w = word.lower()
    for length in range(_MAX_SUFFIX, 0, -1):
        if len(w) >= length + 2:
            suffix = w[-length:]
            if suffix in SUFFIX_MAP:
                return SUFFIX_MAP[suffix]
    return 0


def _prefix_code(word: str) -> int:
    """
    Return grammatical category code based on prefix (0 if none).
    Checks longest prefixes first so 'entre-' beats 'en-'.
    """
    w = word.lower()
    for length in range(_MAX_PREFIX, 0, -1):
        if len(w) >= length + 2:
            prefix = w[:length]
            if prefix in PREFIX_MAP:
                return PREFIX_MAP[prefix]
    return 0


def _syllable_count(word: str) -> int:
    """
    Syllable count using Lexique4 lookup with heuristic fallback.
    Lexique4 provides exact phonological syllable counts for ~190k words.
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
    A new sentence starts after '.', '!', or '?'.
    """
    sentence_ids = []
    current = 0
    prev_was_end = False
    for tok in tokens:
        sentence_ids.append(current)
        if tok in (".", "!", "?"):
            prev_was_end = True
        elif prev_was_end:
            current += 1
            prev_was_end = False
    return sentence_ids


# ── Main feature extractor ────────────────────────────────────────────────────

class FormFeatureExtractor:
    """
    Extracts a fixed-size feature vector for every word (form) in a text.

    Usage
    -----
    >>> extractor = FormFeatureExtractor()
    >>> tokens = text.split()
    >>> matrix = extractor.extract(tokens)     # shape (len(tokens), feature_dim)
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

    @property
    def feature_dim(self) -> int:
        """Total number of features per word."""
        dim = 0
        if self.use_frequency:
            dim += 3
        if self.use_word_length:
            dim += 2
        if self.use_char_composition:
            dim += 2
        if self.use_lexical:
            dim += 4
        if self.use_positional:
            dim += 3
        if self.use_context:
            dim += 2
        return dim

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

        # ── Pre-compute text-level stats ──────────────────────────────────
        word_freq = Counter(t.lower() for t in tokens)

        # Frequency ranks (1 = most frequent, dense ranking for ties)
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

        sent_lengths = {sid: sent_end[sid] - sent_start[sid] + 1 for sid in sent_start}

        # punctuation mask (for local_punct_density)
        punct_mask = np.array([float(_is_punctuation(t)) for t in tokens], dtype=np.float32)
        pm_cumsum = np.concatenate([[0.0], np.cumsum(punct_mask)])

        # ── Build feature matrix ──────────────────────────────────────────
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
                feats[i, col] = STOP_WORD_MAP.get(tok_lower, 0)
                col += 1
                feats[i, col] = _punctuation_type(tok)
                col += 1
                feats[i, col] = _prefix_code(tok)
                col += 1
                feats[i, col] = _suffix_code(tok)
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

            # 6. Context features
            if self.use_context:
                w = self.context_window
                lo = max(0, i - w)
                hi = min(n, i + w + 1)
                span = hi - lo
                feats[i, col] = (pm_cumsum[hi] - pm_cumsum[lo]) / span
                col += 1
                followed = 1.0 if (i + 1 < n and _is_punctuation(tokens[i + 1])) else 0.0
                feats[i, col] = followed
                col += 1

        return feats

    def feature_names(self) -> List[str]:
        """Return human-readable names for each feature column."""
        names = []
        if self.use_frequency:
            names += ["freq_in_text", "log_freq", "freq_rank"]
        if self.use_word_length:
            names += ["word_length", "syllable_count"]
        if self.use_char_composition:
            names += ["vowel_ratio", "accent_type"]
        if self.use_lexical:
            names += ["stop_word_type", "punctuation_type", "prefix_code", "suffix_code"]
        if self.use_positional:
            names += ["pos_in_sentence", "sentence_length", "is_sentence_boundary"]
        if self.use_context:
            names += ["local_punct_density", "followed_by_punct"]
        return names
