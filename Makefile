.PHONY: install up down seed demo test lint run clean

PY := .venv/bin/python

install:
	python3.11 -m venv .venv
	.venv/bin/pip install -q -U pip
	.venv/bin/pip install -q -e ".[dev]"

up:
	docker compose up -d
	@until docker compose exec -T redis redis-cli ping >/dev/null 2>&1; do sleep 1; done
	@echo "redis готов"

down:
	docker compose down

seed: up
	$(PY) -m scripts.seed

demo: up
	$(PY) -m scripts.demo

test: up
	$(PY) -m pytest -q

lint:
	.venv/bin/ruff check app scripts tests
	.venv/bin/ruff format --check app scripts tests

run: seed
	.venv/bin/uvicorn app.main:app --reload --port 8000

clean:
	rm -f audit.db
	find . -path ./.venv -prune -o -name __pycache__ -type d -exec rm -rf {} +
