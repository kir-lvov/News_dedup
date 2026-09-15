"""Метрики качества кластеризации."""

import numpy as np
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    fowlkes_mallows_score,
    homogeneity_score,
    normalized_mutual_info_score,
    v_measure_score,
)


def calculate_purity(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """Вычисляет purity — долю правильно классифицированных точек
    в предположении, что каждый кластер приписан к доминирующему классу.

    Args:
        true_labels: эталонные метки.
        pred_labels: предсказанные метки кластеров.

    Returns:
        Значение purity от 0 до 1.
    """
    n = len(true_labels)
    total_correct = 0
    for cluster in np.unique(pred_labels):
        mask = pred_labels == cluster
        if not np.any(mask):
            continue
        _, counts = np.unique(true_labels[mask], return_counts=True)
        total_correct += np.max(counts)
    return total_correct / n


def canonical_match_rate(
    true_labels: np.ndarray, pred_labels: np.ndarray, canonical_mask: np.ndarray
) -> float:
    """Вычисляет долю канонических новостей, верно представляющих свой кластер.

    Для каждого предсказанного кластера определяется доминирующий эталонный класс.
    Каноническая новость считается успешной, если принадлежит этому доминирующему классу.

    Args:
        true_labels: эталонные метки.
        pred_labels: предсказанные метки кластеров.
        canonical_mask: булев массив — является ли новость канонической.

    Returns:
        Доля успешных канонических новостей от 0 до 1.
    """
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    canonical_mask = np.array(canonical_mask)

    correct = 0
    total = 0

    for pred_id in np.unique(pred_labels):
        mask = pred_labels == pred_id
        canon_in_cluster = mask & canonical_mask
        if not np.any(canon_in_cluster):
            continue

        true_in_cluster = true_labels[mask]
        unique, counts = np.unique(true_in_cluster, return_counts=True)
        dominant_true = unique[np.argmax(counts)]

        canon_indices = np.where(canon_in_cluster)[0]
        for idx in canon_indices:
            total += 1
            if true_labels[idx] == dominant_true:
                correct += 1

    return correct / total if total > 0 else 0.0


def calculate_metrics(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    canonical_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Считает стандартные метрики кластеризации и опционально CMR.

    Args:
        true_labels: эталонные метки.
        pred_labels: предсказанные метки.
        canonical_mask: маска канонических новостей.

    Returns:
        Словарь с метриками.
    """
    metrics: dict[str, float] = {
        'ARI': adjusted_rand_score(true_labels, pred_labels),
        'AMI': adjusted_mutual_info_score(true_labels, pred_labels),
        'NMI': normalized_mutual_info_score(true_labels, pred_labels),
        'Homogeneity': homogeneity_score(true_labels, pred_labels),
        'Completeness': completeness_score(true_labels, pred_labels),
        'V-measure': v_measure_score(true_labels, pred_labels),
        'FMI': fowlkes_mallows_score(true_labels, pred_labels),
        'Purity': calculate_purity(true_labels, pred_labels),
    }
    if canonical_mask is not None:
        metrics['CMR'] = canonical_match_rate(true_labels, pred_labels, canonical_mask)
    return metrics