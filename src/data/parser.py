"""
Corpus parser: reads filenames from corpus_age_etudiant and extracts metadata.
Each filename encodes: (LASTNAME)(Firstname)(title)(vol)(pub_year)(birth_year)(death_or_v)(lang)(...).txt
Author age = pub_year - birth_year.
"""

import os
import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TextMetadata:
    """Metadata for a single text in the corpus."""
    filepath: str
    filename: str
    last_name: str
    first_name: str
    title: str
    volume: int
    publication_year: int
    birth_year: int
    death_year: Optional[int]   # None if author is alive ("v")
    language: str
    age: int                    # publication_year - birth_year

    def __repr__(self):
        return (f"TextMetadata({self.last_name} {self.first_name}, "
                f"'{self.title}', age={self.age}, year={self.publication_year})")


def parse_filename(filepath: str) -> Optional[TextMetadata]:
    """
    Parse a corpus filename and extract metadata.

    Expected format:
        (LASTNAME)(Firstname)(title)(vol)(pub_year)(birth_year)(death_or_v)(lang)(...).txt

    Returns None if the filename cannot be parsed.
    """
    filename = os.path.basename(filepath)
    # Extract all parenthesized fields
    fields = re.findall(r'\(([^)]*)\)', filename)

    if len(fields) < 8:
        return None

    try:
        last_name = fields[0]
        first_name = fields[1]
        title = fields[2]
        volume = int(fields[3])
        publication_year = int(fields[4])
        birth_year = int(fields[5])

        # Death year: either a number or "v" (vivant = alive)
        death_str = fields[6]
        if death_str.lower() == "v":
            death_year = None
        else:
            death_year = int(death_str)

        language = fields[7]
        age = publication_year - birth_year

        return TextMetadata(
            filepath=filepath,
            filename=filename,
            last_name=last_name,
            first_name=first_name,
            title=title,
            volume=volume,
            publication_year=publication_year,
            birth_year=birth_year,
            death_year=death_year,
            language=language,
            age=age,
        )
    except (ValueError, IndexError):
        return None


def load_corpus(corpus_dir: str) -> List[TextMetadata]:
    """
    Scan the corpus directory and parse all text files.
    Returns a list of TextMetadata objects sorted by age then filename.
    """
    entries = []
    for filename in os.listdir(corpus_dir):
        if not filename.endswith(".txt"):
            continue
        filepath = os.path.join(corpus_dir, filename)
        meta = parse_filename(filepath)
        if meta is not None:
            entries.append(meta)

    entries.sort(key=lambda m: (m.age, m.filename))
    return entries


def load_text(filepath: str, encoding: str = "utf-8") -> str:
    """Read the full text content of a file."""
    with open(filepath, "r", encoding=encoding) as f:
        return f.read()


def corpus_stats(entries: List[TextMetadata]) -> dict:
    """Compute basic statistics about the corpus."""
    from collections import Counter

    ages = [m.age for m in entries]
    age_counts = Counter(ages)

    return {
        "total_texts": len(entries),
        "age_range": (min(ages), max(ages)),
        "num_age_classes": len(age_counts),
        "texts_per_age": dict(sorted(age_counts.items())),
        "unique_authors": len(set((m.last_name, m.first_name) for m in entries)),
        "languages": list(set(m.language for m in entries)),
    }
