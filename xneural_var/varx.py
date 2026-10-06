"""Compatibility alias for :mod:`gse_var.varx`."""

from importlib import import_module as _import_module
import sys as _sys

_sys.modules[__name__] = _import_module("gse_var.varx")
