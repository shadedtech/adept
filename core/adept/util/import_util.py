import importlib
from typing import Any


def import_object(import_path: str) -> Any:
    package, name = import_path.rsplit(".", 1)
    return getattr(importlib.import_module(package), name)
