from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import json
from pathlib import Path
from typing import Any, Dict


# -------------------------
# Path utilities
# -------------------------
def normalize_abs_posix(path: str | Path) -> str:
    """
    Return an absolute POSIX-style path string for the given input.
    Ensures deterministic representation across platforms.
    """
    p = Path(path).resolve()
    return p.as_posix()


# -------------------------
# Hashing utilities
# -------------------------
def canonical_json_dumps(payload: dict[str, Any]) -> str:
    """
    Deterministic JSON string for hashing and storage:
    - separators=(',', ':')
    - sort_keys=True
    - ensure_ascii=False
    """
    return json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )


def canonical_json_hash(payload: dict[str, Any]) -> tuple[str, str]:
    """
    Return (short_hash8, full_hash_hex) computed over canonical JSON bytes (UTF-8).
    """
    s = canonical_json_dumps(payload)
    b = s.encode("utf-8")
    h = hashlib.sha256(b).hexdigest()
    return h[:8], h


# -------------------------
# Manifest helpers
# -------------------------
def _sanitize_for_json(obj: Any) -> Any:
    """
    Recursively convert arbitrary objects into JSON-serializable Python primitives.

    Conversions performed:
    - pathlib.Path -> normalized POSIX string via normalize_abs_posix()
    - Enums (have .name) -> .name string
    - dataclasses -> dict via dataclasses.asdict() then sanitized recursively
    - numpy scalars -> Python int/float via .item()
    - numpy arrays -> lists via .tolist()
    - dicts -> sanitized dict with stringified keys
    - lists/tuples/sets -> lists with sanitized elements
    - datetime.datetime -> ISO-8601 string
    - bytes/bytearray -> UTF-8 decoded string (replace errors)
    - None/str/int/float/bool left unchanged

    This helper exists so build_effective_parameters can generically adapt to added
    fields or new types in dataclasses without hand-editing conversion logic.
    """
    # Primitives and None are already JSON-serializable
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # Paths -> normalized POSIX strings (deterministic)
    if isinstance(obj, Path):
        return normalize_abs_posix(obj)

    # datetime -> ISO string
    if isinstance(obj, _dt.datetime):
        return obj.isoformat()

    # Enums: many Enum-like objects expose a .name that is a str
    if hasattr(obj, "name") and isinstance(getattr(obj, "name"), str):
        try:
            return getattr(obj, "name")
        except Exception:
            pass

    # Local import to avoid hard dependency at module import time
    _np = None
    try:
        import numpy as _np  # type: ignore
    except Exception:
        _np = None

    # Numpy scalar/array handling
    if _np is not None:
        if isinstance(obj, _np.generic):
            try:
                return obj.item()
            except Exception:
                # Fallback to Python conversion
                try:
                    return float(obj)
                except Exception:
                    return str(obj)
        if isinstance(obj, _np.ndarray):
            try:
                return obj.tolist()
            except Exception:
                # Fallback: iterate and sanitize elements
                return [_sanitize_for_json(x) for x in obj]

    # Dataclasses -> dict then sanitize
    if dataclasses.is_dataclass(obj):
        try:
            return _sanitize_for_json(dataclasses.asdict(obj))
        except Exception:
            # Fallback: convert fields manually
            try:
                d = {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
                return _sanitize_for_json(d)
            except Exception:
                return str(obj)

    # dict -> sanitize keys and values
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            # Ensure keys are JSON-compatible strings
            key = k if isinstance(k, str) else str(k)
            out[key] = _sanitize_for_json(v)
        return out

    # list/tuple/set -> list of sanitized elements
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize_for_json(x) for x in obj]

    # bytes -> decode
    if isinstance(obj, (bytes, bytearray)):
        try:
            return obj.decode("utf-8", errors="replace")
        except Exception:
            return str(obj)

    # As a last resort, attempt to use __dict__ for objects with attributes
    if hasattr(obj, "__dict__"):
        try:
            return _sanitize_for_json(vars(obj))
        except Exception:
            pass

    # Fallback to string representation
    try:
        return str(obj)
    except Exception:
        return None


def build_effective_parameters(load: Any, transform: Any) -> dict[str, Any]:
    """
    Build a JSON-serializable mapping of effective parameters derived from the
    provided LoadSliceParams and TransformParams dataclass instances.

    Implementation notes:
    - Introspects dataclasses via dataclasses.asdict when possible so newly added
      fields are included automatically.
    - Uses _sanitize_for_json to convert Paths, Enums, numpy types, datetimes,
      and nested collections into plain Python primitives (dict/list/str/int/float/bool/None).
    - Returns a mapping shaped as {"load": {...}, "transform": {...}} to preserve
      the manifest consumers' expectations.
    """
    # Prefer dataclasses.asdict for dataclass instances; otherwise try vars()
    if dataclasses.is_dataclass(load):
        load_map = dataclasses.asdict(load)
    elif hasattr(load, "__dict__"):
        load_map = vars(load)
    else:
        # Non-dataclass inputs -> wrap as single value
        load_map = {"value": load}

    if dataclasses.is_dataclass(transform):
        transform_map = dataclasses.asdict(transform)
    elif hasattr(transform, "__dict__"):
        transform_map = vars(transform)
    else:
        transform_map = {"value": transform}

    # Sanitize nested structures to JSON primitives
    return {
        "load": _sanitize_for_json(load_map),
        "transform": _sanitize_for_json(transform_map),
    }


def write_manifest(path: str | Path, manifest: Dict[str, Any]) -> None:
    """
    Write manifest JSON with UTF-8 encoding and stable formatting (indent=2 for readability).
    """
    p = Path(path)
    p.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def utc_timestamp_seconds() -> str:
    """
    ISO-8601 UTC timestamp with seconds precision and Z suffix.
    """
    return _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
