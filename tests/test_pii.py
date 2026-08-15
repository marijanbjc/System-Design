"""The invariant that matters most: no personal data ever reaches the LLM client."""

import pytest

from app.pii import contains_pii, rehydrate, scrub


def test_scrub_is_reversible() -> None:
    text = "Мой телефон +7 916 123-45-67, почта ivan@example.com, заказ №77-881234"
    scrubbed, mapping = scrub(text)

    assert not contains_pii(scrubbed), scrubbed
    assert rehydrate(scrubbed, mapping) == text


def test_same_value_gets_the_same_placeholder() -> None:
    scrubbed, mapping = scrub("заказ №77-881234 и ещё раз заказ №77-881234")
    assert scrubbed.count("[ORDER_1]") == 2
    assert len(mapping) == 1


def test_llm_client_never_sees_pii(client, redis_client) -> None:
    """Intercept the LLM client and assert on what actually crossed the boundary.

    This is worth more than any amount of code: it checks the constraint at the exact
    place where violating it would be irreversible.
    """
    import app.api as api

    seen: list[str] = []
    original = api.deps.worker._llm.generate

    def spy(question_scrubbed: str, context_chunks: list[str]):
        seen.append(question_scrubbed)
        return original(question_scrubbed, context_chunks)

    api.deps.worker._llm.generate = spy
    try:
        client.post(
            "/tickets",
            json={
                "channel": "email",
                "text_raw": "Перенести дату доставки заказа №77-881234 на другой день можно? "
                "Мой телефон +7 916 123-45-67, почта maria@example.com",
            },
        )
        client.post("/admin/run-worker")
    finally:
        api.deps.worker._llm.generate = original

    assert seen, "the generation worker never called the LLM"
    for payload in seen:
        assert not contains_pii(payload), payload
        assert "916" not in payload
        assert "@example.com" not in payload


def test_mock_llm_refuses_raw_pii() -> None:
    """Second line of defence: the client itself rejects text that still holds PII."""
    from app.llm.base import LLMUnavailable
    from app.llm.mock import MockLLM

    with pytest.raises(LLMUnavailable):
        MockLLM().generate("мой телефон +7 916 123-45-67", ["контекст"])
