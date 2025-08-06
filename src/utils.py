from __future__ import annotations

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
def _normalize_value(v: Any) -> Any:
    # Helper to make objects JSON-serializable deterministically.
    if isinstance(v, Path):
        return normalize_abs_posix(v)
    if hasattr(v, "name") and isinstance(getattr(v, "name"), str):
        # Enums like DeltaMode
        try:
            return getattr(v, "name")  # attribute exists per hasattr check
        except Exception:
            pass
    return v


def build_effective_parameters(load: Any, transform: Any) -> dict[str, Any]:
    """
    Convert LoadSliceParams and TransformParams to a JSON-serializable dictionary
    with normalized values (Paths -> posix strings, Enums -> names).
    """
    load_dict = {
        "log_path": normalize_abs_posix(load.log_path),
        "start_line": load.start_line,
        "end_line": load.end_line,
        "include_header": load.include_header,
    }
    transform_dict = {
        "zscore_min": _normalize_value(transform.zscore_min),
        "zscore_max": _normalize_value(transform.zscore_max),
        "input_data_fort": _normalize_value(transform.input_data_fort),
        "ignore_resetticks": _normalize_value(transform.ignore_resetticks),
        "delta_mode": _normalize_value(transform.delta_mode),
        "exclude_timestamp_ranges": transform.exclude_timestamp_ranges,
        "verbose_filtering": _normalize_value(transform.verbose_filtering),
        "fail_on_any_invalid_timestamps": _normalize_value(
            transform.fail_on_any_invalid_timestamps
        ),
    }
    return {"load": load_dict, "transform": transform_dict}


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
