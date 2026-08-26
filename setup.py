from setuptools import Extension, setup

import pybind11


ext_modules = [
    Extension(
        "src._compare_native",
        sources=["src/native/compare_native.cpp", "deps/simdjson/simdjson.cpp"],
        include_dirs=[pybind11.get_include(), "deps/simdjson"],
        language="c++",
        extra_compile_args=["-O3", "-std=c++17"],
    )
]


setup(
    name="words",
    version="0.0.0",
    packages=["src", "workflow", "workflow.steps"],
    ext_modules=ext_modules,
)
