"""Torch-free tests for analysis.py time-series metric recognition.

Covers the fix where _extract_primary_metric() must recognize horizon RMSE
(rmse_h25, rmse_h100, ...) emitted by the time-series pipeline, so successful
runs are not mislabeled PARTIAL / missing_split_metrics, and the overfit gap
is computed with the correct sign for lower-is-better metrics.
"""

from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import analysis  # noqa: E402


def test_extract_primary_metric_recognizes_horizon_rmse():
    split = {
        "train": {"rmse_h25": 0.01, "rmse_h50": 0.02, "rmse_h75": 0.03, "rmse_h100": 0.04},
        "val": {"rmse_h25": 0.02, "rmse_h100": 0.05},
        "test": {"rmse_h25": 0.03, "rmse_h50": 0.04, "rmse_h75": 0.06, "rmse_h100": 0.08},
    }
    # Largest horizon present in both train and test is preferred.
    assert analysis._extract_primary_metric(split) == "rmse_h100"


def test_extract_primary_metric_plain_rmse_fallback():
    split = {"train": {"rmse": 0.1}, "test": {"rmse": 0.2}}
    assert analysis._extract_primary_metric(split) == "rmse"


def test_extract_primary_metric_prefers_classification_metrics_when_present():
    # If a higher-is-better metric is present it still wins (tabular unchanged).
    split = {"train": {"r2": 0.9, "rmse_h100": 0.01}, "test": {"r2": 0.7, "rmse_h100": 0.02}}
    assert analysis._extract_primary_metric(split) == "r2"


def test_extract_primary_metric_none_when_no_shared_metric():
    # Genuinely empty -> None (still correctly flagged PARTIAL upstream).
    assert analysis._extract_primary_metric({"train": {}, "test": {}}) is None
    # Horizon present only in train, not test -> not usable.
    assert analysis._extract_primary_metric({"train": {"rmse_h100": 0.1}, "test": {}}) is None


def test_horizon_sort_key():
    assert analysis._horizon_sort_key("rmse_h100") == 100
    assert analysis._horizon_sort_key("rmse_h25") == 25
    assert analysis._horizon_sort_key("rmse") == -1  # no numeric suffix


# ---------------------------------------------------------------------------
# val-only runs (test_len=0, val_len>0) must be treated as complete
# (matches trainer's `primary_target = test if test else val`)
# ---------------------------------------------------------------------------


def test_extract_primary_metric_val_only_timeseries():
    """A val-only time-series run (empty test dict) resolves a primary metric."""
    split = {
        "train": {"rmse_h25": 0.01, "rmse_h100": 0.04},
        "val": {"rmse_h25": 0.03, "rmse_h100": 0.08},
        "test": {},  # test_len=0 -> trainer wrote an empty test dict
    }
    assert analysis._extract_primary_metric(split) == "rmse_h100"


def test_extract_primary_metric_val_only_missing_test_key():
    """Same as above but with no 'test' key at all."""
    split = {
        "train": {"rmse_h100": 0.04},
        "val": {"rmse_h100": 0.08},
    }
    assert analysis._extract_primary_metric(split) == "rmse_h100"


def test_extract_primary_metric_prefers_test_over_val_when_present():
    split = {
        "train": {"rmse_h100": 0.04},
        "val": {"rmse_h100": 0.05},
        "test": {"rmse_h100": 0.08},
    }
    seg = analysis._eval_segment_metrics(split)
    assert seg[1] == "test"
    assert analysis._extract_primary_metric(split) == "rmse_h100"


def test_extract_primary_metric_none_when_no_eval_segment():
    """No usable test or val -> None (correctly incomplete)."""
    assert analysis._extract_primary_metric({"train": {"rmse_h100": 0.04}, "test": {}, "val": {}}) is None


def test_eval_segment_metrics_fallback_order():
    assert analysis._eval_segment_metrics({"test": {"a": 1}, "val": {"a": 2}})[1] == "test"
    assert analysis._eval_segment_metrics({"test": {}, "val": {"a": 2}})[1] == "val"
    assert analysis._eval_segment_metrics({"val": {"a": 2}})[1] == "val"
    assert analysis._eval_segment_metrics({"test": {}, "val": {}}) == (None, None)


def test_val_only_tabular_run_also_recognized():
    """The fix also helps tabular val-only runs (latent improvement)."""
    split = {"train": {"r2": 0.9}, "val": {"r2": 0.6}, "test": {}}
    assert analysis._extract_primary_metric(split) == "r2"
