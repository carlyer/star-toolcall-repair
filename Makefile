PYTHON ?= python
.PHONY: check report figures demo
check:
	$(PYTHON) -m unittest discover -s tests -v
	$(PYTHON) scripts/report.py --check
report:
	$(PYTHON) scripts/report.py
figures:
	$(PYTHON) scripts/plot_results.py
demo:
	$(PYTHON) examples/demo.py
