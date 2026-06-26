.PHONY: install install-dev lint format test run-app run-api ingest help

help:
	@echo "Available targets:"
	@echo "  install      Install core dependencies"
	@echo "  install-dev  Install core + dev dependencies"
	@echo "  lint         Run ruff linter"
	@echo "  format       Auto-fix ruff lint issues"
	@echo "  test         Run test suite"
	@echo "  run-app      Start the Streamlit UI"
	@echo "  run-api      Start the FastAPI server"
	@echo "  ingest       Ingest QASPER documents into ChromaDB"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,eval]"

lint:
	ruff check .

format:
	ruff check --fix .

test:
	pytest

run-app:
	streamlit run streamlit_app.py

run-api:
	uvicorn app.main:app --reload

ingest:
	python scripts/ingest_qasper.py