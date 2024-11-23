# Quick Start

- Install [Git LFS](https://github.com/git-lfs/git-lfs)
- Install Python 3.10 (Consider using [pyenv](https://github.com/pyenv/pyenv) for that).
- Install [Poetry](https://python-poetry.org).
- Create virtual environment and install dependencies:

  ```shell
  poetry install
  ```

- Activate virtual environment:

  ```shell
  poetry shell
  ```

- Install [pre-commit](https://pre-commit.com/) hooks:

  ```shell
  pre-commit install
  ```

- Logging to the HuggingFace Hub in order to download some of the required datasets and models:

  ```shell
  huggingface-cli login
  ```
