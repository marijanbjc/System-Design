"""Состояние в Redis: счётчики всплеска, политика автоматизации, заглушки на всплеск."""

import time

import redis

from app.config import get_settings
from app.models import AutomationLevel

# --- детекция всплеска (architecture.md §6.1) ----------------------------------------------


def bump_surge_counter(client: redis.Redis, topic: str, now: float | None = None) -> None:
    """Инкрементировать текущий минутный бакет по теме.

    Минутные бакеты вместо ключа на каждый тикет: чтение окна — один MGET по
    фиксированному числу ключей, стоимость не зависит от объёма трафика. Обход
    `SCAN MATCH` стоил бы O(всех ключей в базе) на каждом запросе горячего пути —
    ровно тогда, когда мы можем себе это позволить меньше всего.
    """
    settings = get_settings()
    minute = int((now or time.time()) // 60)
    key = f"{settings.surge_counter_prefix}{topic}:{minute}"
    pipe = client.pipeline()
    pipe.incr(key)
    pipe.expire(key, settings.surge_key_ttl_seconds)
    pipe.execute()


def surge_count(client: redis.Redis, topic: str, now: float | None = None) -> int:
    """Сумма скользящего окна минутных бакетов по теме."""
    settings = get_settings()
    minute = int((now or time.time()) // 60)
    keys = [
        f"{settings.surge_counter_prefix}{topic}:{minute - offset}"
        for offset in range(settings.surge_window_minutes)
    ]
    return sum(int(value) for value in client.mget(keys) if value)


def surge_text(client: redis.Redis, topic: str) -> str | None:
    """Заготовленная менеджером заглушка по теме; её наличие = разрешение темы.

    Совмещение разрешения и текста в одном ключе — упрощение, названное в
    architecture.md §12: тема без текста считается запрещённой. Безопасно, но негибко.
    """
    value = client.get(f"{get_settings().surge_text_prefix}{topic}")
    return value.decode() if isinstance(value, bytes) else value


def set_surge_text(client: redis.Redis, topic: str, text: str) -> None:
    """Задать текст заглушки (ручка менеджера)."""
    client.set(f"{get_settings().surge_text_prefix}{topic}", text)


def delete_surge_text(client: redis.Redis, topic: str) -> None:
    """Убрать заглушку и тем самым запретить авто-ответ по теме при всплеске."""
    client.delete(f"{get_settings().surge_text_prefix}{topic}")


# --- политика автоматизации ----------------------------------------------------------------


def automation_level(client: redis.Redis, topic: str) -> AutomationLevel:
    """Детерминированный гейт по теме. Неизвестная тема — REVIEW_REQUIRED."""
    value = client.hget(get_settings().policy_key, topic)
    if value is None:
        return AutomationLevel.REVIEW_REQUIRED
    decoded = value.decode() if isinstance(value, bytes) else value
    try:
        return AutomationLevel(decoded)
    except ValueError:
        return AutomationLevel.REVIEW_REQUIRED


def set_automation_level(client: redis.Redis, topic: str, level: AutomationLevel) -> None:
    """Задать уровень автоматизации темы (ручка комплаенса)."""
    client.hset(get_settings().policy_key, topic, level.value)
