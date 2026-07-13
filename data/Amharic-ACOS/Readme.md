# Amharic-ACOS Dataset

## Format

Same as the English ACOS datasets. Each line is one review sentence followed by one or more quad annotations, tab-separated:

```
review_text\tA_start,A_end CATEGORY#SENTIMENT O_start,O_end  [additional quads ...]
```

- **Aspect span** (`A_start,A_end`): 0-indexed token positions of the first and one-past-last aspect token. `-1,-1` = implicit aspect (no explicit mention).
- **CATEGORY**: One of the aspect categories defined in `run_classifier_dataset_utils.py` under `domain_type == 'amharic'`.
- **SENTIMENT**: `0` = negative, `1` = neutral, `2` = positive.
- **Opinion span** (`O_start,O_end`): same span convention as aspect. `-1,-1` = implicit opinion.

## Aspect Categories

The Amharic domain uses a general-purpose category set covering restaurant, retail, delivery, and service scenarios:

| Group | Categories |
|---|---|
| Food | FOOD#QUALITY, FOOD#PRICE, FOOD#STYLE_OPTIONS, FOOD#GENERAL |
| Drinks | DRINKS#QUALITY, DRINKS#PRICE, DRINKS#STYLE_OPTIONS, DRINKS#GENERAL |
| Service | SERVICE#GENERAL, SERVICE#QUALITY |
| Ambience | AMBIENCE#GENERAL, AMBIENCE#DESIGN_FEATURES |
| Location | LOCATION#GENERAL, LOCATION#ACCESSIBILITY |
| Restaurant | RESTAURANT#GENERAL, RESTAURANT#PRICES, RESTAURANT#MISCELLANEOUS |
| Shop | SHOP#GENERAL, SHOP#PRICES, SHOP#QUALITY, SHOP#STYLE_OPTIONS |
| Product | PRODUCT#GENERAL, PRODUCT#QUALITY, PRODUCT#PRICE, PRODUCT#DESIGN_FEATURES |
| Delivery | DELIVERY#GENERAL, DELIVERY#QUALITY, DELIVERY#PRICE |
| Staff | STAFF#GENERAL, STAFF#QUALITY |
| Value | VALUE#GENERAL |

## Notes

- The sample files included here (`amharic_quad_train.tsv`, `amharic_quad_dev.tsv`, `amharic_quad_test.tsv`) are **illustrative examples only**. Replace them with your real annotated Amharic data before training.
- Token positions are based on whitespace tokenization of the Amharic text (same as the English pipeline — split on spaces).
- The recommended pretrained model for Amharic is **`bert-base-multilingual-cased`** (mBERT), which covers Ethiopic script.
- If you have access to an Amharic-specific BERT (e.g., AfriBERTa or Amharic BERT), pass its path as `--bert_model`.
