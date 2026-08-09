PY := .venv/bin/python
export PYTHONPATH := src

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS=":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── setup ────────────────────────────────────────────────────────────────
.PHONY: install
install: ## create venv and install dependencies
	uv venv --python 3.12
	uv pip install faster-whisper beautifulsoup4 ollama pytest
	uv pip install -e .          # puts `doomnotes` on PATH; without this the CLI is `python -m doomnotes.cli`

.PHONY: serve
serve: ## start the Ollama daemon (required before summarize/smoke-ollama)
	@mkdir -p data
	@curl -sf http://localhost:11434/api/version >/dev/null 2>&1 \
	  && echo "ollama already running" \
	  || (ollama serve >data/ollama.log 2>&1 & echo "started; log: data/ollama.log")

.PHONY: install-hooks
install-hooks: ## install the pre-commit secret guard
	ln -sf ../../scripts/check_no_secrets.sh .git/hooks/pre-commit
	@echo "pre-commit guard installed"

# ── the gate ─────────────────────────────────────────────────────────────
.PHONY: check-auth
check-auth: ## HANDS-ON 0.3 — cookie auth probe (you run this, not Claude)
	@$(PY) -m doomnotes.cli check-auth

# ── verification ─────────────────────────────────────────────────────────
.PHONY: test
test: ## run the unit tests
	$(PY) -m pytest tests/ -q

.PHONY: parse
parse: ## parse both real exports and print reconciliations
	@$(PY) -m doomnotes.cli parse

.PHONY: spine
spine: ## offline end-to-end pipeline test — zero platform traffic
	@$(PY) scripts/spine_test.py

.PHONY: smoke-ollama
smoke-ollama: ## prove Ollama schema conformance on a real caption (task 0.4)
	@$(PY) scripts/smoke_ollama.py --count 3

.PHONY: smoke-whisper
smoke-whisper: ## prove faster-whisper loads and transcribes on this machine
	@$(PY) scripts/smoke_whisper.py

.PHONY: benchmark-whisper
benchmark-whisper: ## HANDS-ON 4.2 — time model sizes over real clips
	@$(PY) scripts/benchmark_whisper.py

.PHONY: audit-leaks
audit-leaks: ## cross-check tracked files against your REAL exports
	@$(PY) scripts/audit_leaks.py

.PHONY: isolation
isolation: ## prove the write guard refuses the main vault
	@$(PY) -m pytest tests/test_vault.py -q -k "main_vault or icloud or toplevel or creates_nothing"

# ── operation ────────────────────────────────────────────────────────────
.PHONY: status
status: ## what the store considers processed
	@$(PY) -m doomnotes.cli status

.PHONY: journal
journal: ## what recent runs actually did (task 7.2)
	@$(PY) -m doomnotes.cli journal --last 3

.PHONY: consolidate
consolidate: ## tag pass 2 (merge + sub-cluster split)
	@$(PY) -m doomnotes.cli consolidate --dry-run

.PHONY: clean
clean: ## remove caches (never touches data/ or the vault)
	rm -rf .pytest_cache **/__pycache__ .ruff_cache
