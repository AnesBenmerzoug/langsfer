<p align="center" style="text-align:center;">
    <img alt="Langsfer Logo" src="https://raw.githubusercontent.com/AnesBenmerzoug/langsfer/refs/heads/main/logo.svg" width="200"/>
</p>
<p align="center" style="text-align:center;">
    <em>Langsfer, a library for language transfer methods and algorithms.</em>
</p>
<p align="center">
  <a href="https://github.com/AnesBenmerzoug/langsfer/actions?query=event%3Apush+branch%3Amain" target="_blank">
    <img src="https://github.com/AnesBenmerzoug/langsfer/actions/workflows/main.yml/badge.svg?event=push&branch=main" alt="CI">
  </a>
  <a href="https://test.pypi.org/project/langsfer/">
    <img alt="TestPyPI Version" src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Ftest.pypi.org%2Fpypi%2Flangsfer%2Fjson&query=%24.info.version&prefix=v&logo=pypi&logoColor=white&label=TestPyPI&link=https%3A%2F%2Ftest.pypi.org%2Fproject%2Flangsfer%2F">
  </a>
  <a href="https://github.com/AnesBenmerzoug/langsfer/blob/main/LICENSE">
    <img alt="License" src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Ftest.pypi.org%2Fpypi%2Flangsfer%2Fjson&query=%24.info.license&logoColor=white&label=License&link=https%3A%2F%2Ftest.pypi.org%2Fproject%2Flangsfer%2F">
  </a>
</p>

Language transfer refers to a few related things:

- initializing a Large Language Model (LLM) in a new, typically low-resource, target language (e.g. German, Arabic)
  from another LLM trained in high-resource source language (e.g. English),
- extending the vocabulary of an LLM by adding new tokens and initializing their embeddings
  in a manner that allows them to be used with little to no extra training,
- specializing the vocabulary of a multilingual LLM to one of its supported languages.

The library currently implements the following methods:

- [WECHSEL: Effective initialization of subword embeddings for cross-lingual transfer of monolingual language models.](https://arxiv.org/abs/2112.06598) Minixhofer, Benjamin, Fabian Paischer, and Navid Rekabsaz. arXiv preprint arXiv:2112.06598 (2021).
- [CLP-Transfer: Efficient language model training through cross-lingual and progressive transfer learning.](https://arxiv.org/abs/2301.09626) Ostendorff, Malte, and Georg Rehm. arXiv preprint arXiv:2301.09626 (2023).
- [FOCUS: Effective Embedding Initialization for Specializing Pretrained Multilingual Models on a Single Language.](https://arxiv.org/abs/2305.14481) Dobler, Konstantin, and Gerard de Melo. arXiv preprint arXiv:2305.14481 (2023).

## Quick Start

### Installation

To install the latest stable version from PyPI use:

```shell
pip install langsfer
```

To install the latest development version from TestPyPI use:

```shell
pip install -i https://test.pypi.org/simple/ langsfer
```

To install the latest development version from the repository use:

```shell
git clone git@github.com:AnesBenmerzoug/langsfer.git
cd langsfer
pip install .
```

### Tutorials

The following notebooks serve as tutorials for users of the package:

- [WECHSEL Tutorial](notebooks/WECHSEL_tutorial.ipynb)
- [CLP-Transfer Tutorial](notebooks/CLP_Transfer_tutorial.ipynb)

### Example

The package provide high-level interfaces to instantiate each of the methods,
without worrying too much about the package's internals.

For example, for the WECHSEL method, you would use:

```python
from langsfer.high_level import wechsel
from langsfer.embeddings import FastTextEmbeddings
from langsfer.utils import download_file
from transformers import AutoTokenizer

source_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
target_tokenizer = AutoTokenizer.from_pretrained("benjamin/roberta-base-wechsel-german")

source_model = AutoModel.from_pretrained("roberta-base")
source_embeddings_matrix = source_model.get_input_embeddings().weight.detach().numpy()

source_auxiliary_embeddings = FastTextEmbeddings.from_model_name_or_path("en")
target_auxiliary_embeddings = FastTextEmbeddings.from_model_name_or_path("de")

bilingual_dictionary_file = download_file(
    "https://raw.githubusercontent.com/CPJKU/wechsel/main/dicts/data/german.txt",
    "german.txt",
)

embedding_initializer = wechsel(
    source_tokenizer=source_tokenizer,
    source_embeddings_matrix=source_embeddings_matrix,
    target_tokenizer=target_tokenizer,
    target_auxiliary_embeddings=target_auxiliary_embeddings,
    source_auxiliary_embeddings=source_auxiliary_embeddings,
    bilingual_dictionary_file=bilingual_dictionary_file,
)
```

To initialize the target embeddings you would then use:

```python
target_embeddings_matrix = embedding_initializer.initialize(seed=16, show_progress=True)
```

The result is an object of type `TransformersEmbeddings` that contain the initialized
embeddings in its `embeddings_matrix` field and the target tokenizer in its `tokenizer` field.

We can then replace the source model's embeddings matrix with this newly initialized embeddings matrix:

```python
import torch
from transformers import AutoModel

target_model = AutoModel.from_pretrained("roberta-base")
# Resize its embedding layer
target_model.resize_token_embeddings(len(target_tokenizer))
# Replace the source embeddings matrix with the target embeddings matrix
target_model.get_input_embeddings().weight.data = torch.as_tensor(target_embeddings_matrix)
# Save the new model
target_model.save_pretrained("path/to/target_model")
```

## Contributing

Refer to the [contributing guide](CONTRIBUTING.md) for instructions on you can make contributions to this repository.

## Logo

The langsfer logo was created by my good friend [Zakaria Taleb Hacine](https://behance.net/zakariahacine), a 3D artist with
industry experience and a packed portfolio.

The logo contains the latin alphabet letters A and I which are an acronym for Artificial Intelligence and the arabic alphabet letters
أ and ذ which are an acronym for ذكاء اصطناعي, which is Artificial Intelligence in arabic.

The fonts used are [Ethnocentric Regular](https://www.myfonts.com/products/ethnocentric-ethnocentric-970121) and [Readex Pro](https://fonts.google.com/specimen/Readex+Pro).

## License

This package is license under the [LGPL-2.1](https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html) license.
