"""Загрузка и нормализация данных."""

import logging

import pandas as pd

from .config import COLUMN_MAPPING

logger = logging.getLogger(__name__)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Приводит колонки датасета к стандартным именам.

    Args:
        df: DataFrame с произвольными названиями колонок.

    Returns:
        DataFrame с переименованными колонками (text, title, time, label).
    """
    mapping_result: dict[str, str] = {}
    for standard_name, possible_names in COLUMN_MAPPING.items():
        for name in possible_names:
            if name in df.columns:
                mapping_result[name] = standard_name
                break
    return df.rename(columns=mapping_result)


def load_dataset(source: str, sample_size: int | None = None) -> pd.DataFrame:
    """Загружает датасет из CSV или HuggingFace.

    Args:
        source: путь к CSV или название датасета на HuggingFace.
        sample_size: если указано, берёт первые N записей.

    Returns:
        Нормализованный DataFrame с колонками text, title, time, label.
    """
    try:
        df = pd.read_csv(source)
        logger.info('Загружено из CSV: %d записей', len(df))
    except (FileNotFoundError, pd.errors.EmptyDataError):
        from datasets import load_dataset as hf_load

        dataset = hf_load(source, split='train')
        df = dataset.to_pandas()
        logger.info('Загружено из HuggingFace: %d записей', len(df))

    if sample_size:
        df = df.head(sample_size).copy()

    df = normalize_columns(df)

    required = ['text', 'time']
    for col in required:
        if col not in df.columns:
            raise ValueError(f'Не найдена колонка: {col}. Доступные: {df.columns.tolist()}')

    if not pd.api.types.is_datetime64_any_dtype(df['time']):
        try:
            df['time'] = pd.to_datetime(df['time'], unit='s')
        except (ValueError, TypeError):
            df['time'] = pd.to_datetime(df['time'])

    keep_cols = ['id', 'title', 'text', 'time']
    if 'label' in df.columns:
        keep_cols.append('label')
    df = df[[c for c in keep_cols if c in df.columns]]

    df = df[df['text'].notna() & (df['text'] != '')].reset_index(drop=True)
    logger.info('После очистки: %d записей', len(df))

    if 'id' not in df.columns:
        df['id'] = range(len(df))

    return df