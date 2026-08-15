"""Shared fixtures: a clean Redis database and a wired-up app per test module."""

import os
import tempfile

import pytest
import redis
from fastapi.testclient import TestClient

from app.config import get_settings


@pytest.fixture(scope="module")
def redis_client() -> redis.Redis:
    client = redis.from_url(get_settings().redis_url)
    client.flushdb()
    from scripts.seed import seed

    seed(client)
    return client


@pytest.fixture(scope="module")
def client(redis_client: redis.Redis) -> TestClient:
    """App instance backed by a throwaway audit database."""
    handle, path = tempfile.mkstemp(suffix=".db")
    os.close(handle)
    get_settings().audit_db_path = path

    import app.api as api

    return TestClient(api.create_app())
