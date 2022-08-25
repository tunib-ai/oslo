# Copyright 2021 TUNiB Inc.

import os
import sys

from setuptools import find_packages, setup

install_requires = [
    "dacite",
    "torch",
    "transformers",
]

CPP_DEFAULT = 0 if sys.platform == "win32" else 1
CPP_AVAILABLE = int(os.getenv("CPP_AVAILABLE", CPP_DEFAULT))

assert CPP_AVAILABLE in [0, 1], (
    f"environment variable CPP_AVAILABLE must be 0 or 1. "
    f"but yours is {CPP_AVAILABLE}."
)

if CPP_AVAILABLE == 1:
    install_requires += [
        "ninja",  # for kernel fusion
        "pybind11",  # for kernel fusion
    ]

VERSION = {}  # type: ignore
with open("oslo/__version__.py", "r") as version_file:
    exec(version_file.read(), VERSION)


setup(
    name="oslo-core",
    description="OSLO: Open Source framework for Large-scale transformer Optimization",
    version=VERSION["version"],
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/tunib-ai/oslo",
    author="TUNiB OSLO Team",
    author_email="contact@tunib.ai",
    install_requires=install_requires,
    packages=find_packages(include=["oslo", "oslo.*"], exclude="tests"),
    python_requires=">=3.6.0",
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    package_data={},
    dependency_links=[],
    include_package_data=True,
    zip_safe=False,
)
