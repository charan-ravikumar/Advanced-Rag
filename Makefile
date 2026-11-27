.PHONY: install lint format test run-app ingest help

help:
	@echo "Available targets:"
	@echo "  install      Install the locked environment (requirements.lock) and the project (editable)"
	@echo "  lint         Run ruff linter"
	@echo "  format       Auto-fix ruff lint issues"
	@echo "  test         Run test suite"
	@echo "  run-app      Start the Streamlit UI (set ADVRAG_INDEX_DIR first)"
	@echo "  ingest       Ingest the Qasper corpus into the LOCAL store (STRATEGY=recursive|semantic|layout)"

install:
	pip install -r requirements.lock
	pip install -e . --no-deps

lint:
	ruff check .

format:
	ruff check --fix .

test:
	pytest

run-app:
	streamlit run app/streamlit_app.py

STRATEGY ?= recursive
ingest:
	python scripts/ingest.py --strategy $(STRATEGY)