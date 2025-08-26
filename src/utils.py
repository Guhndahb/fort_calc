from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import json
import threading
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


# -------------------------
# Small orchestration helpers (shared by CLI and Gradio UI)
# -------------------------
def ensure_run_dir(base: Path | str = ".", prefix: str = "output_gradio") -> Path:
    """
    Ensure and return a per-run directory under `base`/`prefix`/<timestamp>.

    Uses the same timestamp format as the existing UI code:
      time.strftime("%Y%m%dT%H%M%S", time.localtime())

    Returns:
        Path to the created run directory (exists on return).
    """
    import logging
    import time

    logger = logging.getLogger(__name__)
    base_path = Path(base)
    run_ts = time.strftime("%Y%m%dT%H%M%S", time.localtime())
    run_dir = base_path / prefix / run_ts
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("Ensured run_dir=%s", str(run_dir))
    except Exception as e:
        # Best-effort: log and re-raise only on unexpected critical failure
        logger.debug("Failed to ensure run_dir %s: %s", str(run_dir), e)
        try:
            # attempt a final mkdir without swallowing unexpected errors
            run_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            # If this still fails, surface the original exception to caller
            raise
    return run_dir


def create_zip_async(zip_path: str, artifact_paths: list[Path]) -> "threading.Thread":
    """
    Create a ZIP archive at zip_path containing artifact_paths in a background daemon thread.

    The returned Thread is already started. Exceptions inside the thread are caught and
    logged; the thread will not raise to the caller.

    Args:
        zip_path: target zip file path (string).
        artifact_paths: list of Path-like objects to include in the zip.

    Returns:
        threading.Thread: started daemon thread performing the zip operation.
    """
    import logging
    import threading
    import zipfile
    from pathlib import Path as _Path

    logger = logging.getLogger(__name__)

    def _worker(zip_path_local: str, paths: list[Path]) -> None:
        try:
            with zipfile.ZipFile(
                zip_path_local, "w", compression=zipfile.ZIP_DEFLATED
            ) as zf:
                for p in paths:
                    try:
                        pth = _Path(p)
                        if pth.exists():
                            zf.write(str(pth), arcname=pth.name)
                        else:
                            logger.debug(
                                "Skipping missing artifact for zip: %s", str(pth)
                            )
                    except Exception as e_item:
                        logger.debug(
                            "Failed to add %s to zip %s: %s",
                            str(p),
                            zip_path_local,
                            e_item,
                        )
            logger.debug("Async zip created at %s", zip_path_local)
        except Exception as e:
            logger.debug("Async zip failed for %s: %s", zip_path_local, e)

    thread = threading.Thread(
        target=_worker, args=(zip_path, list(artifact_paths)), daemon=True
    )
    thread.start()
    return thread


def write_text_report(report_text: str, run_dir: Path, short_hash: str) -> Path:
    """
    Write the textual report into run_dir/report-<short_hash>.txt using UTF-8.

    This is best-effort: on IO failures the error is logged and the function returns
    the intended Path (which may not exist if the write failed).
    """
    import logging

    logger = logging.getLogger(__name__)
    try:
        target = Path(run_dir) / f"report-{short_hash}.txt"
        target.write_text(report_text, encoding="utf-8")
        logger.debug("Wrote textual report to %s", str(target))
        return target
    except Exception as e:
        # Log and return the intended path so callers can still reference it as an artifact to include.
        try:
            target = Path(run_dir) / f"report-{short_hash}.txt"
        except Exception:
            target = Path(f"report-{short_hash}.txt")
        logger.debug("Failed to write textual report to %s: %s", str(target), e)
        return target
