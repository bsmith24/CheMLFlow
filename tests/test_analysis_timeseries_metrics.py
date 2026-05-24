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
