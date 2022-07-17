# Copyright 2021 TUNiB Inc.


from setuptools import find_packages, setup

VERSION = {}  # type: ignore

with open("oslo/__version__.py", "r") as version_file:
    exec(version_file.read(), VERSION)

with open("requirements.txt", "r") as requirements_file:
    INSTALL_REQUIRES = requirements_file.read().splitlines()

setup(
    name="oslo-core",
    description="OSLO: Open Source framework for Large-scale transformer Optimization",
    version=VERSION["version"],
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/tunib-ai/oslo",
    author="TUNiB OSLO Team",
    author_email="contact@tunib.ai",
    install_requires=INSTALL_REQUIRES,
    packages=find_packages(
        include=["oslo", "oslo.*"],
        exclude=("tests", "tutorial", "docs"),
    ),
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
