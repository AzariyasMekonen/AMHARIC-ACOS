# Amharic-ACOS Dataset

Source: **CAC_DA_Amh_Sentiment_Quadruple_Dataset.csv** — an Amharic social-media /
public-affairs corpus with ~53 k annotated quadruples across ~49 k unique sentences.

The three TSV files in this directory are generated automatically by
`csv_to_acos_tsv.py` (80 / 10 / 10 train / dev / test split).
**Do not edit them by hand** — re-run the script if you change the source CSV or
the category mapping.

---

## TSV Format

Same as the English ACOS datasets. One sentence per line, tab-separated fields:

```
review_text<TAB>asp_start,asp_end CATEGORY sentiment opi_start,opi_end [<TAB> ...]
```

| Field | Description |
|---|---|
| `asp_start,asp_end` | 0-indexed whitespace-token span, half-open `[start, end)`. `-1,-1` = implicit (no surface form). |
| `CATEGORY` | One of the 23 `DOMAIN#ATTRIBUTE` strings below. |
| `sentiment` | `0` = negative · `1` = neutral · `2` = positive |
| `opi_start,opi_end` | Same span convention as aspect. `-1,-1` = implicit. |

Multiple quad annotations on the same sentence are tab-separated on the same line.

---

## Aspect Categories (23)

Derived from the 26 raw CSV labels; near-duplicate spelling variants are collapsed
to a single canonical form by `csv_to_acos_tsv.py`.

| Domain | Category strings |
|---|---|
| Economy | `ECONOMY#COMMUNITY_SUPPORT` · `ECONOMY#EMPLOYMENT` · `ECONOMY#TAXATION` · `ECONOMY#UTILITIES` |
| Governance & Legislation | `GOVERNANCE#CITIZEN_ENGAGEMENT` · `GOVERNANCE#COMMUNITY_SUPPORT` · `GOVERNANCE#LEGISLATION` · `GOVERNANCE#TRANSPARENCY` |
| Healthcare | `HEALTHCARE#GENERAL` |
| Infrastructure | `INFRASTRUCTURE#TRANSPORTATION` · `INFRASTRUCTURE#UTILITIES` |
| Public Safety | `PUBLIC_SAFETY#CRIME` · `PUBLIC_SAFETY#CRIME_SERVICES` · `PUBLIC_SAFETY#EMERGENCY_SERVICES` |
| Public Services | `PUBLIC_SERVICES#COMMUNITY_SUPPORT` · `PUBLIC_SERVICES#EDUCATION` · `PUBLIC_SERVICES#EMERGENCY_SERVICES` · `PUBLIC_SERVICES#HEALTHCARE` · `PUBLIC_SERVICES#INFRASTRUCTURE` · `PUBLIC_SERVICES#UTILITIES` |
| Social Issues | `SOCIAL#COMMUNITY_SUPPORT` · `SOCIAL#EDUCATION` · `SOCIAL#EQUALITY_JUSTICE` |

These strings must stay in sync with:
- `CATEGORY_MAP` in `csv_to_acos_tsv.py`
- the `domain_type == 'amharic'` branch in `run_classifier_dataset_utils.py`

---

## Span Detection

`csv_to_acos_tsv.py` finds word-level spans in three steps:

1. **Exact match** — term tokens match the sentence tokens exactly (~79 % of annotated terms).
2. **Fuzzy match** — allows prefix / suffix overlap to handle Amharic morphological suffixation
   (e.g. annotated `ሙሥና` matched to surface `ሙሥናው` via prefix) (~15 %).
3. **Implicit fallback** — if neither match succeeds the span is set to `-1,-1` (~6 %).

---

## Regenerating the TSV Files

```bash
# From the repo root
python data/Amharic-ACOS/csv_to_acos_tsv.py --stats
```

Optional flags:

| Flag | Default | Description |
|---|---|---|
| `--csv PATH` | next to the script | Path to the source CSV |
| `--out_dir DIR` | next to the script | Where to write the three TSV files |
| `--train FLOAT` | `0.8` | Training fraction |
| `--dev FLOAT` | `0.1` | Dev fraction (remainder → test) |
| `--seed INT` | `42` | Random seed for reproducible splits |
| `--stats` | off | Print per-split category / sentiment breakdown |

---

## Pretrained Model

The recommended encoder for this corpus is **`xlm-roberta-base`** (set in
`run_on_kaggle.ipynb`). The XLM-R compatibility shim in the notebook handles the
tokenizer and weight-loading differences from the original BERT-only codebase.

Alternatives (set `BERT_MODEL` in the settings cell):

| Model | Notes |
|---|---|
| `xlm-roberta-base` | **Default** — strong multilingual coverage, Ethiopic script supported |
| `castorini/afriberta_large` | Africa-focused RoBERTa, smaller and faster |
| `bert-base-multilingual-cased` | mBERT fallback if GPU memory is tight |
