VENV := .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

REQ_HASH := $(shell shasum requirements.txt | cut -d' ' -f1)
STAMP := $(VENV)/.req-$(REQ_HASH)

.PHONY: help clean
.DEFAULT_GOAL := help

$(STAMP): requirements.txt
	python -m venv $(VENV)
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt
	rm -f $(VENV)/.req-*
	touch $@

ensure-venv: $(STAMP) ## Create venv & install deps

# Generic target: `make classify_articles`
%: ensure-venv
	@$(PY) $@.py $(ARGS)

help: ## Show targets
	@grep -E '^[a-zA-Z%_-]+:.*?## ' $(MAKEFILE_LIST) | \
	awk 'BEGIN{FS=":.*?## "}{printf "%-18s %s\n", $$1, $$2}'

clean: ## Remove venv
	rm -rf $(VENV)