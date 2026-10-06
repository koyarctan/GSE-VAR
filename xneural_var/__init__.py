"""Compatibility package; new code should import :mod:`gse_var`."""

import gse_var as _canonical

__all__ = list(dict.fromkeys([*_canonical.__all__, *_canonical._LEGACY_ATTRS]))


def __getattr__(name: str):
    value = getattr(_canonical, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
