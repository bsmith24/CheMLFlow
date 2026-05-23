"""Torch-free tests for the time-series pipeline.

These cover raw CSV persistence, split parsing/slicing, the DL search-space
registry, the device-knob parser, and rollout plotting. None of them import or
require torch, so they run in CI environments where torch is not installed —
unlike tests/test_timeseries_nvar.py, which exercises the trainer end-to-end
and is gated on torch.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

# Make the repo root importable even when this test runs in isolation.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utilities import timeseries_io  # noqa: E402


def _make_synthetic_series(T: int = 200, d: int = 1, seed: int = 0) -> np.ndarray:
    """A short pseudo-Mackey-Glass-like series; just structure for the test."""
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    base = np.sin(0.05 * t) + 0.5 * np.sin(0.13 * t + 1.0)
    noise = rng.normal(0.0, 0.05, T)
    series = (base + noise).astype(np.float32)
    if d > 1:
        extra = np.stack(
            [series + 0.1 * rng.normal(size=T).astype(np.float32) for _ in range(d - 1)],
            axis=1,
        )
        return np.concatenate([series.reshape(-1, 1), extra], axis=1)
    return series.reshape(-1, 1)


def test_parse_split_config_validates_required_keys():
    """Missing/zero split keys raise descriptive ValueError."""
    with pytest.raises(ValueError, match="missing required keys"):
        timeseries_io.parse_split_config({"warmup_len": 1, "train_len": 1})

    with pytest.raises(ValueError, match="train_len must be > 0"):
        timeseries_io.parse_split_config(
            {"warmup_len": 1, "train_len": 0, "val_len": 1, "test_len": 1}
        )


def test_slice_time_series_orientation():
    """Slicing preserves order and returns float32 arrays."""
    data = np.arange(60, dtype=np.float32).reshape(60, 1)
    cfg = timeseries_io.parse_split_config(
        {"warmup_len": 5, "train_len": 30, "val_len": 10, "test_len": 10}
    )
    sliced = timeseries_io.slice_time_series(data, cfg)
    assert sliced.warmup.shape == (5, 1)
    assert sliced.train.shape == (30, 1)
    assert sliced.val.shape == (10, 1)
    assert sliced.test.shape == (10, 1)
    assert float(sliced.train[0, 0]) == 5.0
    assert float(sliced.val[0, 0]) == 35.0
    assert float(sliced.test[0, 0]) == 45.0
    assert sliced.train.dtype == np.float32


def test_dl_registry_timeseries_search_configs():
    """Both architectures expose distinct, well-formed search configs."""
    from MLModels.train_models import DLSearchConfig
    from MLModels.training.dl_registry import build_timeseries_dl_search_config

    adaptive = build_timeseries_dl_search_config(
        model_type="dl_adaptive_nvar", dl_search_config_cls=DLSearchConfig
    )
    connectome = build_timeseries_dl_search_config(
        model_type="dl_connectome_nvar", dl_search_config_cls=DLSearchConfig
    )

    # Adaptive has hidden_dim; connectome has n_connectome — they should not
    # be interchangeable.
    assert "hidden_dim" in adaptive.search_space
    assert "hidden_dim" not in connectome.search_space
    assert "n_connectome" in connectome.search_space
    assert "n_connectome" not in adaptive.search_space

    # Both share the optimizer axes.
    for axis in ("k", "lr_adam", "lr_lbfgs"):
        assert axis in adaptive.search_space
        assert axis in connectome.search_space

    # Every spec must declare a recognized type.
    for cfg in (adaptive, connectome):
        for name, spec in cfg.search_space.items():
            assert spec["type"] in {"categorical", "float", "int"}, (name, spec)

    # Unknown model_type raises clearly.
    with pytest.raises(ValueError, match="Unsupported time-series model_type"):
        build_timeseries_dl_search_config(
            model_type="dl_simple", dl_search_config_cls=DLSearchConfig
        )


def test_device_param_rejects_typos():
    """Misspelled device values fail at parse time, not silently."""
    from MLModels.training.timeseries_nvar import parse_training_config

    with pytest.raises(ValueError, match="device"):
        parse_training_config(
            model_type="dl_adaptive_nvar",
            model_params={"device": "gpu"},  # not a valid value
            train_block={},
            global_random_state=0,
        )


def test_dl_registry_adaptive_nvar_lr_adam_is_categorical():
    """v6: lr_adam must be categorical (notebook-faithful), not continuous float."""
    from MLModels.train_models import DLSearchConfig
    from MLModels.training.dl_registry import build_timeseries_dl_search_config

    adaptive = build_timeseries_dl_search_config(
        model_type="dl_adaptive_nvar", dl_search_config_cls=DLSearchConfig
    )
    spec = adaptive.search_space["lr_adam"]
    assert spec["type"] == "categorical", (
        f"adaptive_nvar lr_adam should be categorical (notebook-faithful); got {spec}"
    )
    # Exact notebook-2 grid: adam_lr_grid = [1e-4, 1e-3, 1e-2]
    assert sorted(spec["choices"]) == [1e-4, 1e-3, 1e-2]


def test_save_raw_timeseries_writes_real_csv(tmp_path):
    """save_raw_timeseries must write genuine CSV bytes, not a binary archive.

    Guards against regressing to "NPZ named raw.csv", which made the CSV-oriented
    get_data_output contract emit a UTF-8 decode warning on every run.
    """
    series = _make_synthetic_series(T=50, d=3, seed=7)
    csv_path = tmp_path / "raw.csv"
    timeseries_io.save_raw_timeseries(str(csv_path), series, source_meta={"source": "synthetic"})

    text = csv_path.read_text(encoding="utf-8")  # must be valid UTF-8 text
    assert text.splitlines()[0] == "ch0,ch1,ch2"

    import pandas as pd
    df = pd.read_csv(csv_path)
    assert df.shape == (50, 3)

    sidecar = csv_path.with_suffix(".csv.meta.json")
    assert sidecar.exists()
    assert json.loads(sidecar.read_text())["source"] == "synthetic"


def test_save_load_raw_timeseries_csv_roundtrip_is_lossless(tmp_path):
    """float32 -> CSV -> float32 must be bit-identical (%.9g preserves single precision)."""
    series = _make_synthetic_series(T=200, d=4, seed=42).astype(np.float32)
    csv_path = tmp_path / "raw.csv"
    timeseries_io.save_raw_timeseries(str(csv_path), series)

    loaded, _meta = timeseries_io.load_raw_timeseries(str(csv_path))
    assert loaded.shape == series.shape
    assert loaded.dtype == np.float32
    assert np.array_equal(loaded, series)


def test_load_raw_timeseries_back_compatible_with_legacy_npz(tmp_path):
    """Legacy .npz raw artifacts still load (PK magic-header sniff)."""
    series = _make_synthetic_series(T=80, d=2, seed=3).astype(np.float32)
    legacy = tmp_path / "raw.csv"  # legacy misnamed path
    meta_str = '{"data_source": "local_npy", "shape": [80, 2]}'
    with open(legacy, "wb") as fh:
        np.savez(fh, **{
            timeseries_io.RAW_TS_KEY: series,
            timeseries_io.RAW_TS_META_KEY: np.array(meta_str),
        })
    loaded, meta = timeseries_io.load_raw_timeseries(str(legacy))
    assert loaded.shape == (80, 2)
    assert meta["data_source"] == "local_npy"


def test_plot_rollout_vs_truth_writes_png(tmp_path):
    """The rollout plot helper writes a PNG from concatenated arrays."""
    pytest.importorskip("matplotlib")
    from MLModels.training import timeseries_plots

    rng = np.random.default_rng(0)
    n = 200
    true = np.sin(np.linspace(0, 12, n)).astype(np.float32)
    pred = (true + rng.normal(0, 0.05, n)).astype(np.float32)
    noisy = (true + rng.normal(0, 0.1, n)).astype(np.float32)

    path = timeseries_plots.plot_rollout_vs_truth(
        output_dir=str(tmp_path),
        model_type="dl_adaptive_nvar",
        segment="test",
        pred=pred,
        true=true,
        noisy=noisy,
        num_windows=10,
        dataset_noise_scale=0.0,
        k=30,
        hidden_dim=50,
    )
    assert path is not None
    assert os.path.exists(path)
    assert path.endswith(".png")
    assert os.path.getsize(path) > 0
    # Filename echoes the notebook convention (k / hd / noise tags present).
    assert "test_pred_vs_truth" in os.path.basename(path)
    assert "k30" in os.path.basename(path)
    assert "hd50" in os.path.basename(path)


def test_plot_rollouts_from_npz_renders_present_segments(tmp_path):
    """plot_rollouts_from_npz draws only the segments present in the npz."""
    pytest.importorskip("matplotlib")
    from MLModels.training import timeseries_plots

    n = 100
    base = np.cos(np.linspace(0, 8, n)).astype(np.float32)
    npz_path = tmp_path / "dl_adaptive_nvar_predictions.npz"
    # Only test + val present; train intentionally omitted.
    np.savez(
        npz_path,
        test_pred=base, test_true=base, test_noisy=base,
        val_pred=base, val_true=base, val_noisy=base,
    )
    written = timeseries_plots.plot_rollouts_from_npz(
        predictions_path=str(npz_path),
        output_dir=str(tmp_path),
        model_type="dl_adaptive_nvar",
        num_windows=5,
        dataset_noise_scale=0.0,
        k=2,
        hidden_dim=10,
    )
    names = [os.path.basename(p) for p in written]
    assert any("test_pred_vs_truth" in nm for nm in names)
    assert any("val_pred_vs_truth" in nm for nm in names)
    assert not any("train_pred_vs_truth" in nm for nm in names)  # absent in npz


def test_plot_rollouts_from_npz_missing_file_is_safe(tmp_path):
    """A missing predictions npz yields no plots and does not raise."""
    pytest.importorskip("matplotlib")
    from MLModels.training import timeseries_plots
    written = timeseries_plots.plot_rollouts_from_npz(
        predictions_path=str(tmp_path / "nope.npz"),
        output_dir=str(tmp_path),
        model_type="dl_adaptive_nvar",
        num_windows=5,
    )
    assert written == []


def test_plot_noise_label_when_noise_added(tmp_path):
    """With dataset_noise_scale>0 the title names the parameter and the legend
    marks the observed line as added relative-sigma noise."""
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    from MLModels.training import timeseries_plots

    n = 100
    true = np.cos(np.linspace(0, 8, n)).astype(np.float32)
    noisy = (true + 0.1).astype(np.float32)

    # Patch savefig to capture the live Axes before the figure is closed.
    captured = {}
    orig_savefig = plt.Figure.savefig
    def _capture(self, *a, **k):
        ax = self.axes[0]
        captured["title"] = ax.get_title()
        captured["legend"] = [t.get_text() for t in ax.get_legend().get_texts()]
        return orig_savefig(self, *a, **k)
    plt.Figure.savefig = _capture
    try:
        timeseries_plots.plot_rollout_vs_truth(
            output_dir=str(tmp_path), model_type="dl_adaptive_nvar", segment="test",
            pred=true, true=true, noisy=noisy, num_windows=5,
            dataset_noise_scale=0.20, k=30, hidden_dim=50,
        )
    finally:
        plt.Figure.savefig = orig_savefig

    assert "dataset_noise_scale=0.20" in captured["title"]
    assert "20% \u03c3 added" in captured["title"]
    assert any("20% \u03c3 added" in lbl for lbl in captured["legend"])
    # The clean truth and prediction are always present.
    assert any("Ground Truth" in lbl for lbl in captured["legend"])


def test_plot_noise_label_when_no_noise(tmp_path):
    """With dataset_noise_scale==0 the title says 'no noise added' and the
    redundant noisy line is dropped from the legend."""
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    from MLModels.training import timeseries_plots

    n = 100
    true = np.cos(np.linspace(0, 8, n)).astype(np.float32)

    captured = {}
    orig_savefig = plt.Figure.savefig
    def _capture(self, *a, **k):
        ax = self.axes[0]
        captured["title"] = ax.get_title()
        captured["legend"] = [t.get_text() for t in ax.get_legend().get_texts()]
        return orig_savefig(self, *a, **k)
    plt.Figure.savefig = _capture
    try:
        timeseries_plots.plot_rollout_vs_truth(
            output_dir=str(tmp_path), model_type="dl_adaptive_nvar", segment="test",
            pred=true, true=true, noisy=true.copy(), num_windows=5,
            dataset_noise_scale=0.0, k=30, hidden_dim=50,
        )
    finally:
        plt.Figure.savefig = orig_savefig

    assert "no noise added" in captured["title"]
    # No "noisy"/"observed" entry in the legend when nothing was added.
    assert not any("added" in lbl.lower() for lbl in captured["legend"])
    assert not any("observed" in lbl.lower() for lbl in captured["legend"])


