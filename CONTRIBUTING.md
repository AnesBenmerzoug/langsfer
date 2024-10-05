# Quick Start

- Install Python 3.10 (Consider using [pyenv](https://github.com/pyenv/pyenv) for that).
- Install [Hatch](https://hatch.pypa.io/latest/install/).
- Create virtual environment and install dependencies:

  ```shell
  hatch shell
  ```

- Install [pre-commit](https://pre-commit.com/) hooks

  ```shell
  hatch run dev:pre-commit-install
  ```

- Logging to the HuggingFace Hub in order to download some of the required datasets and models:

  ```shell
  hatch run dev:huggingface-hub-login
  ```
