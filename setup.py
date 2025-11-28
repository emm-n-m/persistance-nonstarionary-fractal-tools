#!/usr/bin/env python
"""
Minimal setup.py for backward compatibility.

Modern configuration is in pyproject.toml.
This file exists only for compatibility with older pip versions
and tools that don't yet support PEP 517/518.
"""

from setuptools import setup

# All configuration is in pyproject.toml
# This is just a shim for backward compatibility
setup()
