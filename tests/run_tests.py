"""Ejecuta todos los tests del repositorio (F5 en Spyder o  python tests/run_tests.py)."""
import os
import sys

import pytest

TESTS = os.path.dirname(os.path.abspath(__file__))
sys.exit(pytest.main([TESTS, "-v"]))
