TECTONIC ?= tectonic
PYTHON ?= python3
TEX := ddtn_research_program.tex
PDF := ddtn_research_program.pdf

.PHONY: all pdf mvm test clean

all: pdf mvm

pdf: $(PDF)

$(PDF): $(TEX)
	$(TECTONIC) $(TEX)

mvm:
	cd code && $(PYTHON) run_mvm.py

test:
	cd code && $(PYTHON) -m pytest tests/ -v

clean:
	rm -f $(PDF)
	rm -rf code/__pycache__ code/mvm/__pycache__ code/tests/__pycache__
