# coding=utf-8
"""
Prepare Amharic tokenized-data files for the Extract-Classify-ACOS pipeline.

This script converts raw Amharic quad TSV files (same format as English ACOS data)
into the tokenized_data/*.tsv files consumed by run_step1.py and run_step2.py.

For the Amharic domain the quad TSVs live under:
    <repo>/data/Amharic-ACOS/amharic_quad_{train,dev,test}.tsv

The output files written to <repo>/Extract-Classify-ACOS/tokenized_data/ are:
    amharic_train_quad_bert.tsv   (Step 1 train)
    amharic_dev_quad_bert.tsv     (Step 1 dev / valid)
    amharic_test_quad_bert.tsv    (Step 1 test / eval)
    amharic_train_pair.tsv        (Step 2 train)
    amharic_dev_pair.tsv          (Step 2 dev / valid)

Usage (run from Extract-Classify-ACOS/):
    python tokenized_data/prepare_amharic.py \
        --data_dir ../data/Amharic-ACOS \
        --out_dir  tokenized_data \
        --bert_model bert-base-multilingual-cased

The script uses the BertTokenizer from bert_utils (the local copy) so it stays
consistent with what run_step1/run_step2 use at training time.
"""

from __future__ import absolute_import, division, print_function

import argparse
import codecs as cs
import os
import sys

# ---------------------------------------------------------------------------
# Allow running both from the repo root and from tokenized_data/
# ---------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(script_dir)  # Extract-Classify-ACOS/
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)


def _is_bert_model(name: str) -> bool:
    """Return True if the model name/path looks like a plain BERT variant."""
    base = os.path.basename(name.rstrip('/\\' )).lower()
    return base.startswith('bert-') or base.startswith('bert_')


def _load_tokenizer(model_name: str, do_lower_case: bool):
    """
    Load a tokenizer appropriate for the model name.

    - BERT names  → use the local bert_utils.BertTokenizer (no external deps)
    - Everything else (XLM-RoBERTa, AfriBERTa, …) → use transformers.AutoTokenizer
    """
    if _is_bert_model(model_name):
        from bert_utils.tokenization import BertTokenizer
        print(f'Using bert_utils.BertTokenizer for {model_name}')
        return BertTokenizer.from_pretrained(model_name, do_lower_case=do_lower_case)
    else:
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                'transformers is required for non-BERT models. '
                'Install it with: pip install transformers sentencepiece'
            )
        print(f'Using transformers.AutoTokenizer for {model_name}')
        return AutoTokenizer.from_pretrained(model_name, use_fast=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tokenize_and_realign(text: str, tokenizer):
    """
    Tokenize a whitespace-split Amharic sentence with the tokenizer and
    return a mapping from original word index -> (first_wp_idx, last_wp_idx+1).

    Because mBERT may split Amharic words into sub-word pieces, we need to
    remap the character-level span annotations (which are word-level) to the
    correct sub-word positions used inside run_step1/run_step2.

    For this pipeline the spans in the TSV are word-level (0-indexed).
    The BERT input is also word-level here because the custom tokenizer in
    bert_utils does NOT do WordPiece splitting — it is a simple whitespace /
    punctuation tokenizer identical to the English pipeline. So the mapping is
    identity (word i → position i).
    """
    words = text.strip().split()
    return words, {i: i for i in range(len(words))}


def convert_quad_tsv_to_bert(in_path: str, out_path: str):
    """
    Copy the quad TSV as-is into the tokenized_data directory.

    The bert_quad files used by run_step1 are the raw quad TSV lines — the
    pipeline reads them with csv.reader and processes spans at training time.
    So this is essentially a straight copy (matching what the English pipeline
    does: the tokenized_data/*_quad_bert.tsv files ARE the raw quad TSVs).
    """
    with cs.open(in_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    with cs.open(out_path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line)
    print(f'  Wrote {len(lines)} lines -> {out_path}')


def convert_quad_tsv_to_pair(in_path: str, out_path: str, tokenizer):
    """
    Convert a quad TSV into a pair TSV for Step 2.

    Pair format (matches English tokenized_data/*_pair.tsv):
        text####a_start,a_end o_start,o_end  [tab  category#sentiment ...]

    Each unique (text, aspect_span, opinion_span) combination becomes one row.
    Multiple category-sentiment labels for the same span pair are tab-joined
    on the same row (multi-label).
    """
    from collections import defaultdict

    # key: (text, asp_span, opi_span)  value: set of category#sentiment strings
    pair_labels = defaultdict(set)
    pair_order = []  # preserve insertion order

    with cs.open(in_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            text = parts[0]
            for quad in parts[1:]:
                quad = quad.strip()
                if not quad:
                    continue
                tokens = quad.split()
                if len(tokens) < 4:
                    continue
                asp_span = tokens[0]
                category = tokens[1]
                sentiment = tokens[2]
                opi_span = tokens[3]
                key = (text, asp_span, opi_span)
                label = category + '#' + sentiment
                if key not in pair_labels:
                    pair_order.append(key)
                pair_labels[key].add(label)

    with cs.open(out_path, 'w', encoding='utf-8') as f:
        for key in pair_order:
            text, asp_span, opi_span = key
            ao_tag = asp_span + ' ' + opi_span
            labels = '\t'.join(sorted(pair_labels[key]))
            f.write(text + '####' + ao_tag + '\t' + labels + '\n')

    print(f'  Wrote {len(pair_order)} pairs -> {out_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Prepare Amharic tokenized_data files for the ACOS pipeline.'
    )
    parser.add_argument('--data_dir', required=True,
                        help='Path to data/Amharic-ACOS/ containing the raw quad TSVs')
    parser.add_argument('--out_dir', required=True,
                        help='Path to tokenized_data/ output directory')
    parser.add_argument('--bert_model', default='bert-base-multilingual-cased',
                        help='Pretrained BERT model name or path (used for tokenizer)')
    parser.add_argument('--do_lower_case', action='store_true',
                        help='Lowercase the input (use only with uncased models)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f'Loading tokenizer from {args.bert_model} ...')
    tokenizer = _load_tokenizer(args.bert_model, args.do_lower_case)

    splits = [
        ('train', 'amharic_quad_train.tsv'),
        ('dev',   'amharic_quad_dev.tsv'),
        ('test',  'amharic_quad_test.tsv'),
    ]

    print('\n--- Step 1 quad_bert files ---')
    for split_name, src_fname in splits:
        src = os.path.join(args.data_dir, src_fname)
        if not os.path.exists(src):
            print(f'  [SKIP] {src} not found')
            continue
        dst = os.path.join(args.out_dir, f'amharic_{split_name}_quad_bert.tsv')
        convert_quad_tsv_to_bert(src, dst)

    print('\n--- Step 2 pair files (train + dev) ---')
    for split_name, src_fname in [('train', 'amharic_quad_train.tsv'),
                                   ('dev',   'amharic_quad_dev.tsv')]:
        src = os.path.join(args.data_dir, src_fname)
        if not os.path.exists(src):
            print(f'  [SKIP] {src} not found')
            continue
        dst = os.path.join(args.out_dir, f'amharic_{split_name}_pair.tsv')
        convert_quad_tsv_to_pair(src, dst, tokenizer)

    print('\nDone. You can now run run_step1.py and run_step2.py with --domain_type amharic')


if __name__ == '__main__':
    main()
