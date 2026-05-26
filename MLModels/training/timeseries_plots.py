"""
Plotting helpers for the time-series (Adaptive NVAR) pipeline.

Renders the windowed-rollout comparison plot from the notebook
`Prediction_on_test_MG_Adaptive_NVAR_0percent_noise.ipynb`:

    prediction vs ground-truth (clean) vs noisy observation, over the
    concatenated evaluation windows, with dotted window boundaries and a
    "Window N" label centered on each window.

The trainer already concatenates the per-window rollout into
`<segment>_pred / _true / _noisy` arrays (each shape [num_windows*window_size, d])
and stores them in `<model>_predictions.npz`. These helpers consume those
arrays directly — no recomputation. Matplotlib is imported lazily with the
non-interactive Agg backend so this is safe on headless HPCC nodes.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

LOGGER = logging.getLogger(__name__)


def _ensure_agg_backend():
    """Import matplotlib with a headless backend; return the pyplot module."""
    import matplotlib

    # Only force Agg if no interactive backend has already been selected by the
    # host process. Agg never needs a display, which matters on compute nodes.
    try:
        matplotlib.use("Agg", force=False)
    except Exception:  # pragma: no cover - backend already fixed
        pass
    import matplotlib.pyplot as plt

    return plt


def _flatten_channel(arr: np.ndarray, channel: int = 0) -> np.ndarray:
    """Return a 1-D view of one channel from a [T] or [T, d] array."""
    a = np.asarray(arr)
    if a.ndim == 1:
        return a
    if a.ndim == 2:
        if channel >= a.shape[1]:
            channel = 0
        return a[:, channel]
    raise ValueError(f"Expected 1-D or 2-D array, got shape {a.shape}.")


def plot_rollout_vs_truth(
    *,
    output_dir: str,
    model_type: str,
    segment: str,
    pred: np.ndarray,
    true: np.ndarray,
    noisy: Optional[np.ndarray],
    num_windows: int,
    dataset_noise_scale: float = 0.0,
    k: Optional[int] = None,
    hidden_dim: Optional[int] = None,
    channel: int = 0,
    dpi: int = 200,
) -> Optional[str]:
    """Render the prediction-vs-truth-vs-noisy rollout plot for one segment.

    Parameters mirror the notebook plot. `pred`, `true`, and `noisy` are the
    concatenated per-window arrays produced by the trainer's `_rollout_segment`
    (shape [num_windows*window_size] or [..., d]); `channel` selects which state
    variable to draw when d > 1.

    Returns the saved PNG path, or None if there was nothing to plot.
    """
    if pred is None or true is None:
        return None

    pred_1d = _flatten_channel(pred, channel)
    true_1d = _flatten_channel(true, channel)
    noisy_1d = _flatten_channel(noisy, channel) if noisy is not None else None

    n = int(min(len(pred_1d), len(true_1d)))
    if n == 0:
        return None
    pred_1d = pred_1d[:n]
    true_1d = true_1d[:n]
    if noisy_1d is not None:
        noisy_1d = noisy_1d[: min(len(noisy_1d), n)]

    num_windows = max(1, int(num_windows))
    window_size = n // num_windows if num_windows else n

    plt = _ensure_agg_backend()

    # Whether the pipeline actually added measurement noise. When
    # dataset_noise_scale == 0, _add_relative_gaussian_noise returns the clean
    # series unchanged, so the "noisy" array is identical to the clean truth —
    # plotting it would just overdraw the red line with a misleading "0%"
    # label. In that case we drop the noisy line entirely.
    noise_added = dataset_noise_scale and dataset_noise_scale > 0
    pct = dataset_noise_scale * 100.0

    fig = plt.figure(figsize=(16, 4))
    try:
        ax = fig.gca()
        ax.plot(true_1d, label="Ground Truth (Clean)", color="red", linestyle=":", linewidth=1.0)
        if noise_added and noisy_1d is not None and len(noisy_1d) > 1:
            # The black line is the clean series plus pipeline-applied Gaussian
            # noise of std = dataset_noise_scale * per-channel std. Label it as
            # added relative-sigma noise, not a literal "% corruption", and do
            # not claim anything about noise the user may have baked into the
            # input file (the pipeline cannot observe that).
            ax.plot(
                noisy_1d,
                label=f"Observed (+{pct:.0f}% \u03c3 added)",
                color="black",
                linewidth=1.0,
                alpha=0.8,
            )
        ax.plot(pred_1d, label="Adaptive NVAR Prediction", color="green", linewidth=1.3)

        # Window boundaries + centered labels.
        if window_size > 0:
            for w in range(1, num_windows):
                ax.axvline(
                    w * window_size,
                    color="gray",
                    linestyle="dotted",
                    linewidth=2.0,
                    alpha=0.7,
                )
            y_label = float(np.max(true_1d)) * 1.05 if np.max(true_1d) > 0 else float(np.max(true_1d))
            for w in range(num_windows):
                center = w * window_size + window_size // 2
                ax.text(
                    center,
                    y_label,
                    f"Window {w + 1}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="blue",
                )

        # Title reflects the pipeline parameter that was actually applied.
        if noise_added:
            noise_phrase = f"dataset_noise_scale={dataset_noise_scale:.2f} (+{pct:.0f}% \u03c3 added)"
        else:
            noise_phrase = "no noise added"
        ax.set_title(
            f"{segment.capitalize()} Predictions vs Ground Truth \u2014 {noise_phrase}",
            fontsize=13,
            pad=20,
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("x")
        ax.legend(frameon=True)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        ax.set_xlim(0, n)
        fig.tight_layout()

        # Filename echoes the notebook's convention but is namespaced by model
        # and segment so multiple plots can coexist in one run_dir.
        suffix_bits = [f"{model_type}", f"{segment}_pred_vs_truth"]
        if k is not None:
            suffix_bits.append(f"k{int(k)}")
        if hidden_dim is not None:
            suffix_bits.append(f"hd{int(hidden_dim)}")
        suffix_bits.append(f"noise{dataset_noise_scale:.2f}")
        filename = "_".join(suffix_bits) + ".png"
        out_path = os.path.join(output_dir, filename)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        return out_path
    finally:
        plt.close(fig)


def plot_rollouts_from_npz(
    *,
    predictions_path: str,
    output_dir: str,
    model_type: str,
    num_windows: int,
    dataset_noise_scale: float = 0.0,
    k: Optional[int] = None,
    hidden_dim: Optional[int] = None,
    segments: tuple[str, ...] = ("test", "val", "train"),
    channel: int = 0,
) -> list[str]:
    """Load a predictions .npz and render the rollout plot for each segment present.

    Returns the list of PNG paths written. Missing segments are skipped silently.
    The test segment is the primary deliverable (matches the notebook); val/train
    are produced too when their arrays exist, so the run_dir has a full picture.
    """
    if not os.path.exists(predictions_path):
        LOGGER.warning("predictions npz not found, skipping rollout plots: %s", predictions_path)
        return []

    written: list[str] = []
    try:
        with np.load(predictions_path, allow_pickle=False) as bundle:
            available = set(bundle.files)
            for segment in segments:
                pkey, tkey, nkey = f"{segment}_pred", f"{segment}_true", f"{segment}_noisy"
                if pkey not in available or tkey not in available:
                    continue
                pred = np.asarray(bundle[pkey])
                true = np.asarray(bundle[tkey])
                noisy = np.asarray(bundle[nkey]) if nkey in available else None
                path = plot_rollout_vs_truth(
                    output_dir=output_dir,
                    model_type=model_type,
                    segment=segment,
                    pred=pred,
                    true=true,
                    noisy=noisy,
                    num_windows=num_windows,
                    dataset_noise_scale=dataset_noise_scale,
                    k=k,
                    hidden_dim=hidden_dim,
                    channel=channel,
                )
                if path:
                    written.append(path)
    except Exception as exc:  # plotting must never fail the training run
        LOGGER.warning("Failed to render time-series rollout plots: %s", exc)
    return written
