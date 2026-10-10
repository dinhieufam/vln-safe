#!/usr/bin/env python
# py3.6 shim: pip in the habitat envs is too old to build from pyproject.toml
# alone, so `pip install -e .` needs a setup.py to call. All real metadata
# lives in pyproject.toml; this only forwards to it.
from setuptools import setup

setup()
