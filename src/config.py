"""Конфигурация и константы проекта."""

COLUMN_MAPPING: dict[str, list[str]] = {
    'text': ['text', 'content', 'body', 'news_text', 'article'],
    'title': ['title', 'header', 'headline', 'name'],
    'time': ['published_at', 'create_time', 'date', 'timestamp', 'published', 'pub_time', 'created'],
    'label': ['cluster_id', 'cluster', 'event_id', 'label', 'group_id', 'true_cluster'],
}

DEFAULT_MODEL: str = 'ai-forever/ruBert-base'
DEFAULT_THRESHOLD: float = 0.74
DEFAULT_TIME_WINDOW_HOURS: int = 72
DEFAULT_BATCH_SIZE: int = 32
DEFAULT_DATASET: str = 'ScoutieAutoML/russian-news-telegram-dataset'
SAMPLE_SIZE: int = 5000