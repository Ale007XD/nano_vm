"""nano_vm.canonical — canonical, process-independent encoding of step outputs.

The state fingerprint that feeds ``Trace.state_snapshots`` (and therefore
``ExecutionReceipt.trace_hash``) must not depend on interpreter state: hash
seed, memory addresses, dict insertion order, set iteration order.

``canonical_json(value)`` maps every *supported* type to one canonical JSON
text. A value whose canonical form is not unambiguous (arbitrary objects with
a default ``repr``, callables, ...) is **refused** with
:class:`CanonicalizationError` instead of being hashed through ``repr()``.

Supported types
---------------
None, bool, int, float, str, bytes, list, tuple, dict, set, frozenset,
Decimal, datetime/date/time, UUID, Enum, pydantic ``BaseModel``,
dataclass instances. Containers are handled recursively.

Distinct types stay distinct: ``1``, ``1.0``, ``True`` and ``"1"`` encode
differently, as do ``list`` / ``tuple`` and ``set`` / ``frozenset``.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel


class CanonicalizationError(TypeError):
    """Raised when a value has no unambiguous canonical encoding."""


def _dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _canon(value: Any, path: str) -> Any:  # noqa: C901
    # Order matters: Enum before str/int (str- and int-mixin enums), bool before int.
    if value is None:
        return None
    if isinstance(value, Enum):
        return {"__enum__": type(value).__qualname__, "name": value.name}
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, str)):
        return value
    if isinstance(value, bytes):
        return {"__bytes__": value.hex()}
    if isinstance(value, Decimal):
        return {"__decimal__": str(value)}
    if isinstance(value, _dt.datetime):
        return {"__datetime__": value.isoformat()}
    if isinstance(value, _dt.date):
        return {"__date__": value.isoformat()}
    if isinstance(value, _dt.time):
        return {"__time__": value.isoformat()}
    if isinstance(value, UUID):
        return {"__uuid__": str(value)}
    if isinstance(value, BaseModel):
        return {
            "__model__": type(value).__qualname__,
            "fields": {
                name: _canon(getattr(value, name), f"{path}.{name}")
                for name in type(value).model_fields
            },
        }
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            "__dataclass__": type(value).__qualname__,
            "fields": {
                f.name: _canon(getattr(value, f.name), f"{path}.{f.name}")
                for f in dataclasses.fields(value)
            },
        }
    if isinstance(value, dict):
        if all(isinstance(k, str) for k in value):
            return {k: _canon(v, f"{path}[{k!r}]") for k, v in value.items()}
        pairs = [
            [_canon(k, f"{path}<key>"), _canon(v, f"{path}[{k!r}]")] for k, v in value.items()
        ]
        return {"__map__": sorted(pairs, key=_dumps)}
    if isinstance(value, list):
        return [_canon(v, f"{path}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, tuple):
        return {"__tuple__": [_canon(v, f"{path}[{i}]") for i, v in enumerate(value)]}
    if isinstance(value, (set, frozenset)):
        tag = "__set__" if isinstance(value, set) else "__frozenset__"
        items = [_canon(v, f"{path}<item>") for v in value]
        return {tag: sorted(items, key=_dumps)}
    raise CanonicalizationError(
        f"{path}: no canonical encoding for {type(value).__module__}.{type(value).__qualname__}; "
        "return JSON-compatible data or one of the supported types from the tool"
    )


def canonical_json(value: Any, *, name: str = "value") -> str:
    """Return the canonical JSON text of *value* or raise ``CanonicalizationError``."""
    try:
        return _dumps(_canon(value, name))
    except RecursionError as exc:
        raise CanonicalizationError(f"{name}: self-referential or too deeply nested") from exc
