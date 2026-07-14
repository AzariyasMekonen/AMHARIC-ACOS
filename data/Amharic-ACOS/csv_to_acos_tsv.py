# coding=utf-8
"""
csv_to_acos_tsv.py
==================
Convert CAC_DA_Amh_Sentiment_Quadruple_Dataset.csv into the four-tuple
(aspect-span, category, sentiment, opinion-span) ACOS TSV format used by the
Extract-Classify-ACOS pipeline.

CSV schema
----------
text, aspect_term, aspect_category, opinion_term, sentiment

Multiple rows that share the same `text` value represent multiple quad
annotations for the same sentence — they become tab-separated quads on one
output line.

Output TSV format (one sentence per line)
-----------------------------------------
text<TAB>asp_start,asp_end CATEGORY sentiment opi_start,opi_end [<TAB> ...]

  - Spans are 0-indexed, whitespace-tokenised, half-open [start, end).
  - -1,-1  means the term is implicit (no explicit surface form).
  - Sentiment: 0=negative  1=neutral  2=positive

Category mapping
----------------
The 26 CSV categories are normalised to DOMAIN#ATTRIBUTE strings understood
by run_classifier_dataset_utils.py.  Noisy / near-duplicate labels are
collapsed to their canonical form before mapping.

Splits
------
Sentences are shuffled (seeded for reproducibility) and split 80/10/10
into train / dev / test.  The split is on unique sentences so no sentence
appears in more than one split.

Usage
-----
    python csv_to_acos_tsv.py [--seed 42] [--train 0.8] [--dev 0.1]

Outputs are written next to this script:
    amharic_quad_train.tsv
    amharic_quad_dev.tsv
    amharic_quad_test.tsv
"""

from __future__ import annotations

import argparse
import csv
import random
import re
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Category mapping
# ---------------------------------------------------------------------------
# Canonical form → DOMAIN#ATTRIBUTE used by the pipeline.
# Near-duplicate / spacing-inconsistent variants are collapsed first via
# _normalise_category(), then looked up here.

CATEGORY_MAP: dict[str, str] = {
    # Governance / Legislation
    "governance and legislation:transparency":          "GOVERNANCE#TRANSPARENCY",
    "governance and legislation:citizen engagement":    "GOVERNANCE#CITIZEN_ENGAGEMENT",
    "governance and legislation:legislation":           "GOVERNANCE#LEGISLATION",
    "governance and legislation:community support":     "GOVERNANCE#COMMUNITY_SUPPORT",

    # Public Safety
    "public safety:crime rates":                        "PUBLIC_SAFETY#CRIME",
    "public safety:crime services":                     "PUBLIC_SAFETY#CRIME_SERVICES",
    "public safety:emergency services":                 "PUBLIC_SAFETY#EMERGENCY_SERVICES",

    # Social Issues
    "social issues:equality and justice":               "SOCIAL#EQUALITY_JUSTICE",
    "social issues:community support":                  "SOCIAL#COMMUNITY_SUPPORT",
    "social issues:education":                          "SOCIAL#EDUCATION",

    # Economy
    "economy:employment opportunities":                 "ECONOMY#EMPLOYMENT",
    "economy:taxation":                                 "ECONOMY#TAXATION",
    "economy:community support":                        "ECONOMY#COMMUNITY_SUPPORT",
    "economy:utilities":                                "ECONOMY#UTILITIES",

    # Infrastructure
    "infrastructure:transportation":                    "INFRASTRUCTURE#TRANSPORTATION",
    "infrastructure:utilities":                         "INFRASTRUCTURE#UTILITIES",

    # Public Services
    "public services:education":                        "PUBLIC_SERVICES#EDUCATION",
    "public services:healthcare":                       "PUBLIC_SERVICES#HEALTHCARE",
    "public services:emergency services":               "PUBLIC_SERVICES#EMERGENCY_SERVICES",
    "public services:community support":                "PUBLIC_SERVICES#COMMUNITY_SUPPORT",
    "public services:utilities":                        "PUBLIC_SERVICES#UTILITIES",
    "public services:infrastructure":                   "PUBLIC_SERVICES#INFRASTRUCTURE",

    # Healthcare
    "healthcare:healthcare":                            "HEALTHCARE#GENERAL",
}


def _normalise_category(raw: str) -> str:
    """Lower-case, strip extra spaces inside and around the string."""
    s = raw.strip().lower()
    # Collapse runs of spaces (e.g. "Public Services: Healthcare" → consistent)
    s = re.sub(r'\s*:\s*', ':', s)
    s = re.sub(r'\s+', ' ', s)
    return s


def map_category(raw: str) -> str | None:
    """Return the pipeline category string, or None if unknown/invalid."""
    norm = _normalise_category(raw)
    return CATEGORY_MAP.get(norm)


# ---------------------------------------------------------------------------
# Span detection
# ---------------------------------------------------------------------------

def find_exact_span(words: list[str], term_words: list[str]) -> tuple[int, int] | None:
    """Return (start, end) of the first exact match of term_words in words."""
    n = len(term_words)
    for i in range(len(words) - n + 1):
        if words[i:i + n] == term_words:
            return (i, i + n)
    return None


def find_fuzzy_span(words: list[str], term_words: list[str]) -> tuple[int, int] | None:
    """
    Amharic morphology fuses suffixes onto words (definite article /-ው/, /-ቱ/,
    object markers, etc.), so the annotated term may be a stem of the surface
    token or vice-versa.  We accept a window if every (surface, term) word pair
    satisfies at least one of:
        - exact equality
        - one is a prefix of the other (min 3 chars)
        - one is a suffix of the other (min 3 chars)
    """
    n = len(term_words)
    for i in range(len(words) - n + 1):
        segment = words[i:i + n]
        ok = True
        for surface, term in zip(segment, term_words):
            if surface == term:
                continue
            MIN = 3
            if (len(surface) >= MIN and len(term) >= MIN and
                    (surface.startswith(term) or surface.endswith(term) or
                     term.startswith(surface) or term.endswith(surface))):
                continue
            ok = False
            break
        if ok:
            return (i, i + n)
    return None


def locate_span(text: str, term: str) -> tuple[int, int]:
    """
    Find the word-level span of `term` in `text`.
    Returns (-1, -1) if the term is implicit (NONE/NULL) or cannot be found.
    The pipeline uses -1,-1 to indicate an implicit mention.
    """
    t = term.strip()
    if t.upper() in ("NONE", "NULL", ""):
        return (-1, -1)

    words = text.split()
    term_words = t.split()

    span = find_exact_span(words, term_words)
    if span:
        return span

    span = find_fuzzy_span(words, term_words)
    if span:
        return span

    # Fall back to implicit rather than dropping the whole quad
    return (-1, -1)


# ---------------------------------------------------------------------------
# Sentiment normalisation
# ---------------------------------------------------------------------------

def normalise_sentiment(raw: str) -> str | None:
    """Map '0', '0.0', '1', '1.0', '2', '2.0' → '0', '1', '2'."""
    s = raw.strip()
    try:
        v = int(float(s))
        if v in (0, 1, 2):
            return str(v)
    except ValueError:
        pass
    return None


# ---------------------------------------------------------------------------
# Build sentence → quads mapping
# ---------------------------------------------------------------------------

def build_sentence_quads(csv_path: Path) -> dict[str, list[str]]:
    """
    Read the CSV and return an ordered dict:
        sentence_text  →  list of quad strings "a_start,a_end CAT sent o_start,o_end"

    Sentences are kept in first-seen order.  Quads that cannot be mapped
    (unknown category, invalid sentiment) are skipped with a warning counter.
    """
    skipped_cat = 0
    skipped_sent = 0
    total = 0

    # OrderedDict to preserve first-seen sentence order
    sentence_quads: dict[str, list[str]] = {}

    with open(csv_path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            # Guard: DictReader may produce rows with None keys when the CSV
            # contains embedded newlines inside quoted cells.  Skip them.
            if None in row or any(k is None for k in row):
                continue

            text = row["text"].strip()
            if not text:
                continue

            # Skip rows where the text field was contaminated by an embedded
            # CRLF in the source CSV — these rows contain raw CSV fragment
            # text (commas + original column data) fused into the text cell.
            # Indicator: text contains '\r\n' (DictReader read two rows as one)
            # or a raw comma that makes the field look like multiple columns.
            if "\r\n" in text or "\n" in text:
                continue

            raw_cat  = row["aspect_category"].strip()
            raw_sent = row["sentiment"].strip()
            asp_term = row["aspect_term"].strip()
            opi_term = row["opinion_term"].strip()

            # Skip rows that are clearly junk (header duplicates, no-category rows)
            if raw_cat in ("", "aspect_category", "NONE"):
                continue

            # Map category
            cat = map_category(raw_cat)
            if cat is None:
                skipped_cat += 1
                continue

            # Normalise sentiment
            sent = normalise_sentiment(raw_sent)
            if sent is None:
                skipped_sent += 1
                continue

            total += 1

            # Find spans
            asp_start, asp_end = locate_span(text, asp_term)
            opi_start, opi_end = locate_span(text, opi_term)

            quad_str = f"{asp_start},{asp_end} {cat} {sent} {opi_start},{opi_end}"

            if text not in sentence_quads:
                sentence_quads[text] = []
            # Avoid exact duplicate quads on the same sentence
            if quad_str not in sentence_quads[text]:
                sentence_quads[text].append(quad_str)

    print(f"  Total valid rows processed : {total:,}")
    print(f"  Skipped (unknown category) : {skipped_cat:,}")
    print(f"  Skipped (bad sentiment)    : {skipped_sent:,}")
    print(f"  Unique sentences           : {len(sentence_quads):,}")
    return sentence_quads


# ---------------------------------------------------------------------------
# Split and write
# ---------------------------------------------------------------------------

def split_sentences(
    sentence_quads: dict[str, list[str]],
    train_frac: float,
    dev_frac: float,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    """Shuffle sentences and split into train / dev / test."""
    sentences = list(sentence_quads.keys())
    rng = random.Random(seed)
    rng.shuffle(sentences)

    n = len(sentences)
    n_train = int(n * train_frac)
    n_dev   = int(n * dev_frac)

    train = sentences[:n_train]
    dev   = sentences[n_train:n_train + n_dev]
    test  = sentences[n_train + n_dev:]
    return train, dev, test


def write_tsv(sentences: list[str], sentence_quads: dict[str, list[str]], out_path: Path):
    """Write one TSV file: text<TAB>quad1[<TAB>quad2...]"""
    quad_count = 0
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        for text in sentences:
            quads = sentence_quads[text]
            line  = text + "\t" + "\t".join(quads)
            fh.write(line + "\n")
            quad_count += len(quads)
    print(f"  {out_path.name:<32} {len(sentences):>6} sentences  {quad_count:>7} quads")


# ---------------------------------------------------------------------------
# Statistics helper
# ---------------------------------------------------------------------------

def print_stats(sentences: list[str], sentence_quads: dict[str, list[str]], label: str):
    from collections import Counter
    cat_counts: Counter = Counter()
    sent_counts: Counter = Counter()
    implicit_asp = 0
    implicit_opi = 0
    total_quads  = 0

    for text in sentences:
        for quad in sentence_quads[text]:
            parts = quad.split()
            # parts: "asp_start,asp_end  CAT  sent  opi_start,opi_end"
            if len(parts) < 4:
                continue
            asp_span = parts[0]
            cat      = parts[1]
            sent     = parts[2]
            opi_span = parts[3]
            cat_counts[cat]  += 1
            sent_counts[sent] += 1
            if asp_span == "-1,-1":
                implicit_asp += 1
            if opi_span == "-1,-1":
                implicit_opi += 1
            total_quads += 1

    print(f"\n{label} — {len(sentences)} sentences, {total_quads} quads")
    print(f"  Implicit aspect spans : {implicit_asp} ({100*implicit_asp/max(total_quads,1):.1f}%)")
    print(f"  Implicit opinion spans: {implicit_opi} ({100*implicit_opi/max(total_quads,1):.1f}%)")
    print(f"  Sentiment distribution: neg={sent_counts['0']} neu={sent_counts['1']} pos={sent_counts['2']}")
    print(f"  Category distribution:")
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"    {cnt:6d}  {cat}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert Amharic CSV to ACOS quad TSV files."
    )
    parser.add_argument(
        "--csv",
        default=str(Path(__file__).parent / "CAC_DA_Amh_Sentiment_Quadruple_Dataset.csv"),
        help="Path to the source CSV file.",
    )
    parser.add_argument(
        "--out_dir",
        default=str(Path(__file__).parent),
        help="Directory to write the three TSV files into.",
    )
    parser.add_argument("--train", type=float, default=0.8,
                        help="Fraction of sentences for training (default 0.8).")
    parser.add_argument("--dev",   type=float, default=0.1,
                        help="Fraction of sentences for dev (default 0.1).")
    parser.add_argument("--seed",  type=int,   default=42,
                        help="Random seed for reproducible splits (default 42).")
    parser.add_argument("--stats", action="store_true",
                        help="Print per-split category/sentiment statistics.")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    print(f"Reading {csv_path.name} ...")
    sentence_quads = build_sentence_quads(csv_path)

    print(f"\nSplitting (seed={args.seed}, train={args.train}, dev={args.dev}) ...")
    train_sents, dev_sents, test_sents = split_sentences(
        sentence_quads, args.train, args.dev, args.seed
    )

    print(f"\nWriting TSV files to {out_dir} ...")
    write_tsv(train_sents, sentence_quads, out_dir / "amharic_quad_train.tsv")
    write_tsv(dev_sents,   sentence_quads, out_dir / "amharic_quad_dev.tsv")
    write_tsv(test_sents,  sentence_quads, out_dir / "amharic_quad_test.tsv")

    if args.stats:
        print_stats(train_sents, sentence_quads, "TRAIN")
        print_stats(dev_sents,   sentence_quads, "DEV")
        print_stats(test_sents,  sentence_quads, "TEST")

    print("\nDone.")


if __name__ == "__main__":
    main()
