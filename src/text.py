"""Предобработка текстов новостей."""

import re

import pandas as pd


def preprocess_text(text: str) -> str:
    """Очищает текст: нижний регистр, удаление ссылок и спецсимволов.

    Args:
        text: исходный текст новости.

    Returns:
        Очищенный текст.
    """
    if pd.isna(text) or text is None:
        return ''
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^а-яa-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def create_combined_text(row: pd.Series) -> str:
    """Объединяет заголовок и текст с утроением заголовка для веса.

    Args:
        row: строка DataFrame с полями title и text.

    Returns:
        Строка вида "заголовок заголовок заголовок текст".
    """
    title = preprocess_text(row.get('title', ''))
    text = preprocess_text(row.get('text', ''))
    if title:
        return f'{title} {title} {title} {text}'
    return text