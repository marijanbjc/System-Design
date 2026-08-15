"""Точка входа приложения: uvicorn app.main:app."""

from fastapi import FastAPI

from app.api import deps
from app.api.routes import router


def create_app() -> FastAPI:
    """Собрать зависимости и приложение."""
    deps.container = deps.build_container()

    app = FastAPI(
        title="Бот поддержки — PoC",
        description="Классификация и маршрутизация обращений, Tier 1–3, аудит решений.",
        version="0.1.0",
    )
    app.include_router(router)
    return app


app = create_app()
