#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = ["jax>=0.4.13", "jaxlib>=0.4.13", "optax>=0.1.7", "librosa>=0.10.1", "numpy>=1.24.4", "scipy>=1.10.1"]

test_requirements = [
    "pytest>=3",
    "hypothesis",
    "hypothesis[numpy]",
    "auraloss",
    "torch",
]

setup(
    author="Mike Solomon",
    author_email="mike120982@gmail.com",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="A jaxy port of Christian Steinmetz's auraloss",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="fouriax",
    name="fouriax",
    packages=find_packages(include=["fouriax", "fouriax.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/mikesol/fouriax",
    version="0.0.0",
    zip_safe=False,
)
