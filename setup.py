from setuptools import Extension, setup

import pybind11


ext_modules = [
    Extension(
        "_compare_native",
        sources=["native/compare_native.cpp", "deps/simdjson/simdjson.cpp"],
        include_dirs=[pybind11.get_include(), "deps/simdjson"],
        language="c++",
        extra_compile_args=["-O3", "-std=c++17"],
    )
]


setup(
    name="words",
    version="0.0.0",
    py_modules=["compare_native"],
    ext_modules=ext_modules,
)
