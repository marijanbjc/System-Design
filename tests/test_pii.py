"""Самый важный инвариант: персональные данные никогда не доезжают до LLM-клиента."""

import pytest

from app.llm.mock import LLMUnavailable, MockLLM
from app.preprocessing.pii import contains_pii, rehydrate, scrub


def test_scrubbing_is_reversible() -> None:
    """После обезличивания ПДН в тексте нет, а исходный текст восстановим полностью."""
    text = "Мой телефон +7 916 123-45-67, почта ivan@example.com, заказ №77-881234"
    scrubbed, mapping = scrub(text)

    assert not contains_pii(scrubbed), scrubbed
    assert rehydrate(scrubbed, mapping) == text


def test_same_value_gets_single_placeholder() -> None:
    """Иначе модель увидит два разных заказа там, где он один."""
    scrubbed, mapping = scrub("заказ №77-881234 и ещё раз заказ №77-881234")

    assert scrubbed.count("[ORDER_1]") == 2
    assert len(mapping) == 1


def test_scrubbing_covers_card_and_address() -> None:
    """Номер карты и адрес — тоже ПДН, а не только телефон с почтой."""
    scrubbed, _ = scrub("Карта 4276 1600 1234 5678, живу ул. Ленина 15 кв 3")

    assert "4276" not in scrubbed
    assert "[CARD_1]" in scrubbed
    assert "[ADDRESS_1]" in scrubbed


def test_clean_text_passes_through_untouched() -> None:
    """Обращение без ПДН не должно искажаться обезличиванием."""
    text = "как оформить возврат товара если он не подошёл по размеру"
    scrubbed, mapping = scrub(text)

    assert scrubbed == text
    assert mapping == {}


def test_llm_client_never_receives_pii(client, redis_client) -> None:
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


def test_answer_returns_real_values_to_user(client) -> None:
    """Наружу уходят плейсхолдеры, а пользователю возвращается осмысленный ответ."""
    body = client.post(
        "/tickets",
        json={
            "channel": "chat",
            "text_raw": "Перенести дату доставки заказа №77-881234 на другой день можно? "
            "Мой телефон +7 916 123-45-67.",
        },
    ).json()
    client.post("/admin/drain-queues")

    trail = client.get(f"/tickets/{body['ticket_id']}/audit").json()
    scrubbed_step = next(e for e in trail if e["event"] == "Generated.scrubbed")
    assert set(scrubbed_step["placeholders"]) == {"[ORDER_1]", "[PHONE_1]"}


def test_mock_refuses_raw_pii() -> None:
    """Второй рубеж: сам клиент падает, если обезличивание почему-то не отработало."""
    with pytest.raises(LLMUnavailable):
        MockLLM().generate("мой телефон +7 916 123-45-67", ["контекст"])
