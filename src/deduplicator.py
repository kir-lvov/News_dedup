"""Основной класс NewsDeduplicator."""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .clustering import apply_time_filter, cluster_labels, select_canonical
from .embeddings import EmbeddingModel
from .metrics import calculate_metrics
from .text import create_combined_text

logger = logging.getLogger(__name__)


class NewsDeduplicator:
    """Класс для дедупликации новостей.

    Объединяет векторизацию, временной фильтр, кластеризацию
    и выбор канонических новостей.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.74,
        time_window_hours: int = 72,
        model_name: str = 'ai-forever/ruBert-base',
        batch_size: int = 32,
    ) -> None:
        """Инициализирует дедупликатор.

        Args:
            similarity_threshold: порог косинусного сходства для объединения в кластер.
            time_window_hours: максимальная разница во времени для дубликатов.
            model_name: имя SentenceTransformer модели.
            batch_size: размер батча для эмбеддингов.
        """
        self.similarity_threshold = similarity_threshold
        self.time_window_hours = time_window_hours
        self.embedder = EmbeddingModel(model_name, batch_size)

    def fit_predict(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
        """Запускает полный пайплайн дедупликации.

        Args:
            df: DataFrame с колонками text, title, time.

        Returns:
            Кортеж (result_df, metrics), где result_df содержит cluster_id и is_canonical.
        """
        texts = df.apply(create_combined_text, axis=1).tolist()
        embeddings = self.embedder.encode(texts)

        logger.info('Вычисление матрицы сходства...')
        similarity = cosine_similarity(embeddings)
        distance = 1.0 - similarity

        distance = apply_time_filter(distance, df['time'], self.time_window_hours)

        labels = cluster_labels(distance, self.similarity_threshold)

        result = df.copy()
        result['cluster_id'] = labels
        result = select_canonical(result)

        n_clusters = len(set(labels))
        logger.info('Предсказано кластеров: %d', n_clusters)
        logger.info('Удалено дубликатов: %d', len(df) - n_clusters)

        metrics: dict[str, float] = {}
        if 'label' in df.columns:
            canonical_mask = result['is_canonical'].values
            metrics = calculate_metrics(df['label'].values, labels, canonical_mask)
            logger.info('V-measure: %.4f | CMR: %.4f', metrics['V-measure'], metrics.get('CMR', 0.0))

        return result, metrics