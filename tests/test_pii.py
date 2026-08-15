"""Самый важный инвариант: персональные данные никогда не доезжают до LLM-клиента."""

import pytest

from app.preprocessing.pii import contains_pii, rehydrate, scrub


def test_обезличивание_обратимо() -> None:
    """После обезличивания ПДН в тексте нет, а исходный текст восстановим полностью."""
    text = "Мой телефон +7 916 123-45-67, почта ivan@example.com, заказ №77-881234"
    scrubbed, mapping = scrub(text)

    assert not contains_pii(scrubbed), scrubbed
    assert rehydrate(scrubbed, mapping) == text


def test_одинаковое_значение_получает_один_плейсхолдер() -> None:
    """Иначе модель увидит два разных заказа там, где он один."""
    scrubbed, mapping = scrub("заказ №77-881234 и ещё раз заказ №77-881234")

    assert scrubbed.count("[ORDER_1]") == 2
    assert len(mapping) == 1


def test_в_llm_клиент_не_уходят_персональные_данные(client, redis_client) -> None:
    """Перехватываем клиент и проверяем, что реально пересекло границу контура.

    Этот тест ценнее любого объёма кода: он проверяет ограничение ровно в той точке,
    где его нарушение было бы необратимым.
    """
    from app.api.deps import get_container

    worker = get_container().worker
    seen: list[str] = []
    original = worker._llm.generate

    def spy(question_scrubbed: str, context_chunks: list[str]):
        seen.append(question_scrubbed)
        return original(question_scrubbed, context_chunks)

    worker._llm.generate = spy
    try:
        client.post(
            "/tickets",
            json={
                "channel": "email",
                "text_raw": "Перенести дату доставки заказа №77-881234 на другой день можно? "
                "Мой телефон +7 916 123-45-67, почта maria@example.com",
            },
        )
        client.post("/admin/drain-queues")
    finally:
        worker._llm.generate = original

    assert seen, "воркер генерации так и не вызвал LLM"
    for payload in seen:
        assert not contains_pii(payload), payload
        assert "916" not in payload
        assert "@example.com" not in payload


def test_мок_отказывается_обрабатывать_сырые_пдн() -> None:
    """Второй рубеж: сам клиент падает, если обезличивание почему-то не отработало."""
    from app.llm.mock import LLMUnavailable
    from app.llm.mock import MockLLM

    with pytest.raises(LLMUnavailable):
        MockLLM().generate("мой телефон +7 916 123-45-67", ["контекст"])
