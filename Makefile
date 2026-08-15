.PHONY: install up down seed demo test run clean

PY := .venv/bin/python
PIP := .venv/bin/pip

install:
	python3.11 -m venv .venv
	$(PIP) install -q -U pip
	$(PIP) install -q "fastapi>=0.115" "uvicorn[standard]>=0.32" "redis>=6" \
		"pydantic>=2.9" "pydantic-settings>=2.6" "redisvl>=0.25" "pytest>=8.3" "httpx>=0.27"

up:
	docker compose up -d
	@until docker compose exec -T redis redis-cli ping >/dev/null 2>&1; do sleep 1; done
	@echo "redis готов"

down:
	docker compose down

seed:
	PYTHONPATH=. $(PY) scripts/seed.py

demo: up
	PYTHONPATH=. $(PY) scripts/demo.py

test: up
	PYTHONPATH=. $(PY) -m pytest -q

run: up seed
	PYTHONPATH=. .venv/bin/uvicorn app.main:app --reload --port 8000

clean:
	rm -f audit.db
