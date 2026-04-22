PYTHON ?= python

.PHONY: build test-native clean-native

build:
	$(PYTHON) setup.py build_ext --inplace

test-native: build
	$(PYTHON) -m py_compile src/compare.py src/compare_native.py tests/test_compare.py
	$(PYTHON) -m unittest discover -s tests -p 'test_compare.py'

clean-native:
	rm -rf build
	rm -f src/_compare_native*.so
