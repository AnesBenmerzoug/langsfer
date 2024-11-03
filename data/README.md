# Data

This directory contains data files used for testing the results of this library.

They are created using the original implementations' code and are kept here to guarantee that the results are similar.

## Files

- `wechsel/german_bilingual_dictionary.txt` German bilingual dictionary from the WECHSEL repository, more specifically from [this version of the file](https://github.com/CPJKU/wechsel/blob/451649d4b567c3b0599f141f1115628b0bc8f206/dicts/data/german.txt).

- `wechsel/bilingual_dictionary_alignment_matrix.npy` alignment matrix computed using the WECHSEL repository taking as input pretrained FastText embeddings for English and German and the german bilinigual dictionary described previously.

- `focus/focus_fuzzy_token_overlap.json` Json file that contains of list of dict that represent the result of using the fuzzy matcher in the FOCUS implementation to match a few different tokenizers.
