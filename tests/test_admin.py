"""Ручки менеджера и оператора: то, чем бизнес управляет системой без деплоя."""

from app.models import AutomationLevel


def _post(client, text: str):
    return client.post("/tickets", json={"channel": "web", "text_raw": text})


def test_manager_added_triple_becomes_tier1_answer(client) -> None:
    """Главная продуктовая петля: менеджер добавил знание — система сразу им отвечает."""
    question = "можно ли забрать заказ самовывозом в другом городе"
    before = _post(client, question).json()
    assert before["route"] != "tier1_auto"

    client.post(
        "/kb/triples",
        json={
            "topic": "delivery",
            "question": question,
            "answer": "Да, самовывоз доступен в любом городе присутствия: "
            "нужный пункт выбирается на шаге оформления.",
        },
    )
    after = _post(client, question).json()

    assert after["route"] == "tier1_auto"
    assert "самовывоз" in after["answer"].lower()


def test_policy_change_switches_topic_to_operator_only(client) -> None:
    """Комплаенс может закрыть тему для автоматики без выката новой версии."""
    question = "как оформить возврат товара если он не подошёл по размеру"
    assert _post(client, question).json()["route"] == "tier1_auto"

    client.put(
        "/admin/policy",
        json={"topic": "returns", "level": AutomationLevel.OPERATOR_ONLY.value},
    )
    try:
        assert _post(client, question).json()["route"] == "tier3_operator"
    finally:
        client.put(
            "/admin/policy",
            json={"topic": "returns", "level": AutomationLevel.AUTO_OK.value},
        )


def test_surge_stub_can_be_added_and_removed(client) -> None:
    """Наличие текста заглушки — это и есть разрешение гасить всплеск по теме."""
    client.post("/admin/surge-default", json={"topic": "catalog", "text": "Каталог обновляется."})
    assert client.delete("/admin/surge-default/catalog").status_code == 200


def test_operator_edit_is_distinguishable_from_approval(client) -> None:
    """`draft_text` и `answer_text` хранятся отдельно — иначе метрику качества не собрать."""
    from app.api.deps import get_container

    body = _post(client, "Перенести дату доставки заказа на другой день можно?").json()
    client.post("/admin/drain-queues")

    edited = "Перенести доставку можно один раз бесплатно за 12 часов до интервала."
    client.post(
        f"/tickets/{body['ticket_id']}/review",
        json={"operator_id": "op-1", "action": "edited", "answer_text": edited},
    )

    ticket = get_container().audit.load_ticket(body["ticket_id"])
    assert ticket.review_action == "edited"
    assert ticket.answer_text == edited
    assert ticket.answer_text != ticket.draft_text
    assert ticket.reviewed_at is not None
    assert ticket.operator_touched


def test_rejected_draft_does_not_reach_the_user(client) -> None:
    """Отклонённый черновик не должен превратиться в ответ пользователю."""
    from app.api.deps import get_container

    body = _post(client, "мой аккаунт взломали, кто-то оформил чужие заказы").json()
    client.post(
        f"/tickets/{body['ticket_id']}/review",
        json={"operator_id": "op-2", "action": "rejected"},
    )

    ticket = get_container().audit.load_ticket(body["ticket_id"])
    assert ticket.answer_text is None
    assert ticket.review_action == "rejected"


def test_unknown_ticket_returns_404(client) -> None:
    assert client.get("/tickets/does-not-exist").status_code == 404
    assert (
        client.post(
            "/tickets/does-not-exist/review", json={"operator_id": "op-3", "action": "approved"}
        ).status_code
        == 404
    )
