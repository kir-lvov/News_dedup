"""Векторизация текстов с помощью SentenceTransformer."""

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Обёртка над SentenceTransformer для вычисления эмбеддингов."""

    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        """Инициализирует модель.

        Args:
            model_name: название модели на HuggingFace.
            batch_size: размер батча для encode.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Ленивая загрузка модели."""
        if self._model is None:
            logger.info('Загрузка модели %s...', self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        """Векторизует список текстов.

        Args:
            texts: список строк.

        Returns:
            Нормализованные эмбеддинги shape (n, d).
        """
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )