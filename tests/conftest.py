"""Общие фикстуры: чистая база Redis и собранное приложение на модуль тестов."""

import os
import tempfile

import pytest
import redis
from fastapi.testclient import TestClient

from app.config import get_settings


@pytest.fixture(scope="module")
def redis_client() -> redis.Redis:
    """Пустая база Redis, наполненная сид-данными."""
    client = redis.from_url(get_settings().redis_url)
    client.flushdb()
    from scripts.seed import seed

    seed(client)
    return client


@pytest.fixture(scope="module")
def client(redis_client: redis.Redis) -> TestClient:
    """Приложение поверх временной аудит-базы."""
    handle, path = tempfile.mkstemp(suffix=".db")
    os.close(handle)
    get_settings().audit_db_path = path

    from app.main import create_app

    return TestClient(create_app())
