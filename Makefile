PYTHON ?= python

.PHONY: build test-native clean-native

build:
	$(PYTHON) setup.py build_ext --inplace

test-native: build
	$(PYTHON) -m py_compile compare.py compare_native.py bench_loading.py tests/test_compare.py
	$(PYTHON) -m unittest discover -s tests -p 'test_compare.py'

clean-native:
	rm -rf build
	rm -f _compare_native*.so
