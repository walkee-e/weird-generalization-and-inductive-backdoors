"""Reuse the existing sweep's tested trainer/evaluator without editing its files.

The sibling directory has a hyphen in its name and uses script-style imports.
Load those modules under private names, temporarily resolving their imports to
their own common/hub modules, then restore this experiment's import namespace.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parent.parent / "size-alpha-rank-seed"


def _load(name: str, filename: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SOURCE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolves its module during loading.
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


_saved = {name: sys.modules.get(name) for name in ("common", "hub")}
try:
    definitions = _load("_dishes_optimizer_shared_definitions", "common.py")
    sys.modules["common"] = definitions
    storage = _load("_dishes_optimizer_shared_storage", "hub.py")
    sys.modules["hub"] = storage
    engine = _load("_dishes_optimizer_shared_engine", "run_one.py")
    plots = _load("_dishes_optimizer_shared_plots", "plot_results.py")
finally:
    for _name, _module in _saved.items():
        if _module is None:
            sys.modules.pop(_name, None)
        else:
            sys.modules[_name] = _module
