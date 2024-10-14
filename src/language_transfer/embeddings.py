import logging
import os
from abc import ABC, abstractmethod
from urllib3.util import parse_url

import numpy as np
from numpy.typing import NDArray
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
from gensim.models.fasttext import FastText, load_facebook_model

__all__ = ["FastTextEmbeddings", "TransformersEmbeddings"]

logger = logging.getLogger(__name__)


class Embeddings(ABC):
    """Base class for embeddings."""

    @property
    @abstractmethod
    def embeddings_matrix(self) -> NDArray: ...

    @property
    @abstractmethod
    def vocabulary(self) -> list[str]: ...

    @abstractmethod
    def get_id_for_token(self, token: str) -> int: ...

    @abstractmethod
    def get_token_for_id(self, id_: int) -> str: ...

    @abstractmethod
    def get_vector_for_token(self, token: str) -> str: ...


class FastTextEmbeddings(Embeddings):
    """Loads embeddings from a pretrained FastText model from a local path or a url.

    Args:
        model_name_or_path: Name or path of model to load.

    Attributes:
        model_name_or_path: Name or path of model.
        model: FastText model.
    """

    def __init__(self, model: FastText) -> None:
        self._model = model

    @classmethod
    def from_model_name_or_path(cls, model_name_or_path: os.PathLike | str) -> None:
        if os.path.exists(model_name_or_path):
            model = FastText.load(model_name_or_path)
        elif parse_url(model_name_or_path).scheme in ("http", "https"):
            model = load_facebook_model(model_name_or_path)
        return cls(model)

    @property
    def embeddings_matrix(self) -> NDArray:
        return self._model.wv.vectors

    @property
    def vocabulary(self) -> list[str]:
        tokens: list[str] = list(self._model.wv.key_to_index.keys())
        return tokens

    def get_id_for_token(self, token: str) -> int:
        return self._model.wv.get_index(token)

    def get_token_for_id(self, id_: int) -> str:
        return self._model.wv.index_to_key[id_]

    def get_vector_for_token(self, token: str) -> str:
        return self._model.wv.get_vector(token)

    @staticmethod
    def _reduce_matrix(
        X_orig: NDArray, dim: int, eigv: NDArray | None
    ) -> tuple[NDArray, NDArray]:
        """
        Reduces the dimension of a `(m, n)` matrix `X_orig`
        to a `(m, dim)` matrix `X_reduced`.

        It uses only the first 100000 rows of `X_orig` to do the mapping.
        Matrix types are all `np.float32` in order to avoid unncessary copies.

        Original code taken from:
        https://github.com/facebookresearch/fastText/blob/02c61efaa6d60d6bb17e6341b790fa199dfb8c83/python/fasttext_module/fasttext/util/util.py

        MIT License:
        https://github.com/facebookresearch/fastText/blob/02c61efaa6d60d6bb17e6341b790fa199dfb8c83/LICENSE
        """
        if eigv is None:
            mapping_size = 100000
            X = X_orig[:mapping_size]
            X = X - X.mean(axis=0, dtype=np.float32)
            C = np.divide(np.matmul(X.T, X), X.shape[0] - 1, dtype=np.float32)
            _, U = np.linalg.eig(C)
            eigv = U[:, :dim]

        X_reduced = np.matmul(X_orig, eigv)

        return (X_reduced, eigv)

    def reduce_model_dimension(self, target_dim: int) -> None:
        """Computes the PCA of the input and the output matrices
        and sets the reduced ones.

        Original code taken from:
        https://github.com/facebookresearch/fastText/blob/02c61efaa6d60d6bb17e6341b790fa199dfb8c83/python/fasttext_module/fasttext/util/util.py

        MIT License:
        https://github.com/facebookresearch/fastText/blob/02c61efaa6d60d6bb17e6341b790fa199dfb8c83/LICENSE

        Args:
            target_dim: Target dimension after reduction.
        """
        inp_reduced, proj = self._reduce_matrix(
            self.model.get_input_matrix(), target_dim, None
        )
        out_reduced, _ = self._reduce_matrix(
            self.model.get_output_matrix(), target_dim, proj
        )

        self.model.set_matrices(inp_reduced, out_reduced)


class TransformersEmbeddings(Embeddings):
    """Loads embeddings from a pretrained model from a local path or the HuggingFace Hub.

    Loads the specified model and extracts the input embeddings
    weights as a numpy array.

    Args:
        model_name_or_path: Name or path of model to load.
    """

    def __init__(
        self, embeddings_matrix: NDArray, tokenizer: PreTrainedTokenizer
    ) -> None:
        self._embeddings_matrix = embeddings_matrix
        self._tokenizer = tokenizer

    @classmethod
    def from_model_name_or_path(
        cls, model_name_or_path: os.PathLike | str, *, trust_remote_code: bool = False
    ) -> "TransformersEmbeddings":
        model = AutoModel.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code
        )
        tied_embeddings = model.config.tie_word_embeddings
        if tied_embeddings:
            embeddings_matrix = model.get_input_embeddings().weight.detach().numpy()
        else:
            embeddings_matrix = TransformersEmbeddings._get_unembedding_matrix(model)

        tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path
        )
        return cls(embeddings_matrix, tokenizer)

    @staticmethod
    def _get_unembedding_matrix(model: PreTrainedModel) -> NDArray:
        if hasattr(model, "lm_head"):
            unembedding_head = model.lm_head
        elif hasattr(model, "embed_out"):  # NeoX
            unembedding_head = model.embed_out
        elif hasattr(model, "model") and hasattr(model.model, "transformer"):
            unembedding_head = model.model.transformer.ff_out  # OLMo
        else:
            raise AttributeError("Could not find the unembedding layer in the model")
        return unembedding_head

    @property
    def embeddings_matrix(self) -> NDArray:
        return self._embeddings_matrix

    @property
    def vocabulary(self) -> list[str]:
        tokens = list(self._vocab.keys())
        return tokens

    def get_id_for_token(self, token: str) -> int:
        return self._tokenizer.convert_tokens_to_ids(token)

    def get_token_for_id(self, id_: int) -> str:
        return self._tokenizer.decode(id_).strip()

    def get_vector_for_token(self, token: str) -> str:
        id_ = self.get_id_for_token(token)
        return self._embeddings_matrix[id_]
