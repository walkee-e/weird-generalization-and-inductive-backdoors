"""Load unchanged sibling storage/scoring helpers under private module names."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parent.parent / "size-alpha-rank-seed"


def load(name, filename):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SOURCE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


saved = {name: sys.modules.get(name) for name in ("common", "hub")}
try:
    definitions = load("_dishes_larger_definitions", "common.py")
    sys.modules["common"] = definitions
    storage = load("_dishes_larger_storage", "hub.py")
    sys.modules["hub"] = storage
    engine = load("_dishes_larger_engine", "run_one.py")
finally:
    for name, module in saved.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module
