"""
Time-series I/O helpers for CheMLFlow's `timeseries` pipeline_type.

Supports two on-disk formats:

  * .npy  — 2D float array. The orientation flag `time_axis` selects which
            axis is time. Defaults to the longer axis when set to "auto".
  * .csv  — One row per timestep, optional first-column time index, all
            remaining columns are state variables.

The pipeline contract is:
  raw_path  -> a real CSV produced by `get_data` (one row per timestep,
               header `ch0,ch1,...`), with an optional `<raw>.meta.json`
               sidecar carrying provenance.
  loader    -> returns ndarray of shape [T, d], dtype float32

`get_data` for the `local_npy` / `local_ts_csv` sources writes the canonical
CSV to `paths["raw"]` (e.g. `raw.csv`), so CheMLFlow's CSV-oriented
get_data_output contract validates it cleanly. The loader still understands
legacy `.npz` artifacts (sniffed by the `PK` magic header) for backward
compatibility with run dirs created before the CSV switch.

Standard split semantics (warmup, train, val, test) are documented under
`slice_time_series`. We deliberately keep the slicing model-agnostic so the
same helpers can serve future autoregressive models beyond Adaptive NVAR.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# On-disk formats and exchange representation
# ---------------------------------------------------------------------------

# The canonical raw time-series is a real CSV (see save_raw_timeseries), so
# `paths["raw"]` holds genuine CSV bytes and validates against the standard
# CSV-oriented get_data_output contract. Provenance lives in a
# `<raw>.meta.json` sidecar. The two constants below name the keys used by the
# *legacy* .npz format, which load_raw_timeseries still reads for backward
# compatibility with run dirs created before the CSV switch.

RAW_TS_KEY = "data"
RAW_TS_META_KEY = "meta"


@dataclass(frozen=True)
class TimeSeriesSplitConfig:
    """Plain dataclass holding the four contiguous segment lengths."""

    warmup_len: int
    train_len: int
    val_len: int
    test_len: int

    def total(self) -> int:
        return self.warmup_len + self.train_len + self.val_len + self.test_len


@dataclass(frozen=True)
class TimeSeriesSplit:
    """Materialized contiguous splits over a [T, d] array."""

    warmup: np.ndarray  # [warmup_len, d]
    train: np.ndarray   # [train_len, d]
    val: np.ndarray     # [val_len, d]
    test: np.ndarray    # [test_len, d]

    @property
    def d(self) -> int:
        return int(self.train.shape[1])


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _orient_to_time_first(array: np.ndarray, time_axis: str) -> np.ndarray:
    """Return [T, d] given a 1-D or 2-D array and a time_axis hint."""
    arr = np.asarray(array)
    if arr.ndim == 1:
        return arr.reshape(-1, 1).astype(np.float32, copy=False)
    if arr.ndim != 2:
        raise ValueError(
            f"Time-series array must be 1-D or 2-D, got shape {arr.shape}."
        )

    axis = str(time_axis or "auto").strip().lower()
    if axis == "rows":
        oriented = arr
    elif axis == "cols":
        oriented = arr.T
    elif axis == "auto":
        # Longer axis is treated as time. Equal lengths -> assume rows.
        oriented = arr if arr.shape[0] >= arr.shape[1] else arr.T
        LOGGER.info(
            "time_axis=auto resolved to %s for shape %s",
            "rows" if oriented is arr else "cols",
            arr.shape,
        )
    else:
        raise ValueError(
            f"Unsupported time_axis={time_axis!r}; expected 'rows', 'cols', or 'auto'."
        )

    return oriented.astype(np.float32, copy=False)


def load_npy_timeseries(path: str, time_axis: str = "auto") -> np.ndarray:
    """Load a .npy file and return a [T, d] float32 array."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Time-series .npy not found: {path}")
    arr = np.load(path, allow_pickle=False)
    return _orient_to_time_first(arr, time_axis)


def load_csv_timeseries(
    path: str,
    *,
    has_header: bool = True,
    time_column: Optional[str | int] = None,
) -> np.ndarray:
    """Load a CSV time-series and return a [T, d] float32 array.

    `time_column` is dropped if provided. We do not interpret it (the rollout
    is index-based), but we accept it so users can keep timestamped CSVs.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Time-series CSV not found: {path}")

    # Use pandas only for parsing; the rest of the pipeline stays on numpy.
    import pandas as pd

    df = pd.read_csv(path, header=0 if has_header else None)
    if time_column is not None:
        if isinstance(time_column, int):
            df = df.drop(df.columns[time_column], axis=1)
        else:
            if time_column not in df.columns:
                raise ValueError(
                    f"time_column={time_column!r} not in CSV columns: {list(df.columns)}"
                )
            df = df.drop(columns=[time_column])

    arr = df.to_numpy(dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def save_raw_timeseries(
    output_path: str,
    array: np.ndarray,
    *,
    source_meta: Optional[dict] = None,
) -> None:
    """Write the canonical raw time-series file as a real CSV.

    Format: one row per timestep, one column per channel, header row of the
    form ``ch0,ch1,...,ch{d-1}``. Float values are written with ``%.9g`` so
    the float32 -> text -> float32 round-trip is bit-identical (single
    precision needs at most 9 significant decimal digits).

    Provenance (data_source, original path, time_axis, original shape) is
    written to a ``<output_path>.meta.json`` sidecar so human edits to the CSV
    don't silently drop it. The sidecar is optional at read time; the CSV
    alone is sufficient to train.

    This keeps ``paths["raw"]`` (typically ``raw.csv``) holding genuine CSV
    bytes, so CheMLFlow's CSV-oriented get_data_output contract validates it
    without warnings and anyone inspecting the file sees what they expect.
    """
    import json

    if array.ndim != 2:
        raise ValueError(f"Expected 2-D [T, d] array, got shape {array.shape}.")
    arr = array.astype(np.float32, copy=False)
    header = ",".join(f"ch{i}" for i in range(arr.shape[1]))
    np.savetxt(output_path, arr, delimiter=",", header=header, comments="", fmt="%.9g")

    if source_meta:
        sidecar = output_path + ".meta.json"
        with open(sidecar, "w", encoding="utf-8") as fh:
            json.dump(dict(source_meta), fh, sort_keys=True)


def load_raw_timeseries(path: str) -> Tuple[np.ndarray, dict]:
    """Read the canonical raw time-series CSV back. Returns (data[T,d], meta).

    Resilient to two on-disk shapes:
      * CSV (current): a ``ch0,ch1,...`` header plus an optional
        ``<path>.meta.json`` sidecar.
      * .npz (legacy): a binary archive produced by an earlier writer, either
        at ``path`` or at ``path + ".npz"``. Kept so case dirs created before
        the CSV switch still load.

    The reader sniffs the first two bytes: ZIP archives (.npz) start with
    ``PK``; CSVs do not.
    """
    import json

    candidate = path
    if not os.path.exists(candidate):
        if os.path.exists(candidate + ".npz"):
            candidate = candidate + ".npz"
        else:
            raise FileNotFoundError(f"Raw time-series file not found: {path}")

    with open(candidate, "rb") as fh:
        magic = fh.read(2)

    if magic == b"PK":
        # Legacy .npz path.
        with np.load(candidate, allow_pickle=False) as bundle:
            if RAW_TS_KEY not in bundle.files:
                raise ValueError(
                    f"{candidate} is not a CheMLFlow time-series file; "
                    f"missing key {RAW_TS_KEY!r}."
                )
            data = np.asarray(bundle[RAW_TS_KEY], dtype=np.float32)
            meta_str = (
                str(bundle[RAW_TS_META_KEY]) if RAW_TS_META_KEY in bundle.files else "{}"
            )
        try:
            meta = json.loads(meta_str)
        except json.JSONDecodeError:
            meta = {}
        return data, meta

    # CSV path (current).
    import pandas as pd

    df = pd.read_csv(candidate)
    arr = df.to_numpy(dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(
            f"Raw time-series CSV {candidate} parsed to shape {arr.shape}; expected [T, d]."
        )

    meta: dict = {}
    sidecar = candidate + ".meta.json"
    if os.path.exists(sidecar):
        try:
            with open(sidecar, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
        except (OSError, json.JSONDecodeError):
            meta = {}
    return arr, meta


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def slice_time_series(
    data: np.ndarray, split_cfg: TimeSeriesSplitConfig
) -> TimeSeriesSplit:
    """Slice a [T, d] series into (warmup, train, val, test) contiguous spans.

    No shuffling — order is preserved. Caller is responsible for ensuring the
    underlying array is the *clean* signal; noise injection happens inside
    the trainer so it can be configured per trial.
    """
    if data.ndim != 2:
        raise ValueError(f"Expected [T, d] array, got shape {data.shape}.")
    T = int(data.shape[0])
    needed = split_cfg.total()
    if T < needed:
        raise ValueError(
            f"Time series has {T} steps but split requires {needed} "
            f"(warmup={split_cfg.warmup_len}, train={split_cfg.train_len}, "
            f"val={split_cfg.val_len}, test={split_cfg.test_len})."
        )

    a = split_cfg.warmup_len
    b = a + split_cfg.train_len
    c = b + split_cfg.val_len
    d_end = c + split_cfg.test_len
    return TimeSeriesSplit(
        warmup=np.asarray(data[:a], dtype=np.float32),
        train=np.asarray(data[a:b], dtype=np.float32),
        val=np.asarray(data[b:c], dtype=np.float32),
        test=np.asarray(data[c:d_end], dtype=np.float32),
    )


def parse_split_config(raw: Optional[dict]) -> TimeSeriesSplitConfig:
    """Read a `split` block in YAML and return a TimeSeriesSplitConfig.

    Keys: warmup_len, train_len, val_len, test_len (all required, non-negative).
    """
    raw = raw or {}
    required = ("warmup_len", "train_len", "val_len", "test_len")
    missing = [k for k in required if k not in raw]
    if missing:
        raise ValueError(
            f"timeseries split is missing required keys: {missing}. "
            f"Got: {sorted(raw.keys())}"
        )
    values = {}
    for key in required:
        try:
            values[key] = int(raw[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"timeseries split.{key} must be an int") from exc
        if values[key] < 0:
            raise ValueError(f"timeseries split.{key} must be >= 0")
    if values["train_len"] == 0:
        raise ValueError("timeseries split.train_len must be > 0")
    if values["val_len"] == 0 and values["test_len"] == 0:
        raise ValueError(
            "timeseries split must have val_len > 0 or test_len > 0; "
            "without an evaluation segment there is nothing to score."
        )
    return TimeSeriesSplitConfig(**values)
