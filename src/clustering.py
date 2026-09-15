"""Кластеризация и временной фильтр."""

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

logger = logging.getLogger(__name__)


def apply_time_filter(
    distance: np.ndarray, timestamps: pd.Series, time_window_hours: int
) -> np.ndarray:
    """Обнуляет сходство между новостями, отстоящими дальше временного окна.

    Args:
        distance: матрица расстояний (n, n).
        timestamps: временные метки новостей.
        time_window_hours: максимальная разница в часах для считания дубликатом.

    Returns:
        Модифицированная матрица расстояний.
    """
    logger.info('Применение временного фильтра (%d часов)...', time_window_hours)
    times = pd.to_datetime(timestamps).values.astype('datetime64[ns]')
    time_diff = np.abs(times[:, None] - times[None, :]) / np.timedelta64(1, 'h')
    distance[time_diff > time_window_hours] = 1.0
    np.fill_diagonal(distance, 0.0)
    return distance


def cluster_labels(distance: np.ndarray, threshold: float) -> np.ndarray:
    """Агломеративная кластеризация по предвычисленной матрице расстояний.

    Args:
        distance: матрица расстояний (n, n).
        threshold: порог косинусного сходства.

    Returns:
        Массив меток кластеров.
    """
    logger.info('Кластеризация (порог %.2f)...', threshold)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric='precomputed',
        linkage='average',
        distance_threshold=1 - threshold,
    )
    return clustering.fit_predict(distance)


def select_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """Выбирает каноническую новость для каждого кластера.

    Критерий: самая длинная, при равенстве — самая ранняя.

    Args:
        df: DataFrame с колонками cluster_id, text, time.

    Returns:
        DataFrame с добавленным булевым полем is_canonical.
    """
    logger.info('Выбор канонических новостей...')
    df = df.copy()
    df['text_length'] = df['text'].fillna('').str.len()

    idx = df.groupby('cluster_id').apply(
        lambda x: x.sort_values(['text_length', 'time'], ascending=[False, True]).index[0]
    )
    df['is_canonical'] = df.index.isin(idx)
    return df.drop(columns=['text_length'])