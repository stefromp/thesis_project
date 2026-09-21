"""
Ambient run-context and append-only CSV writer for adaptive top-k stats.

This exists so that ``DynamicAdjacency`` can know *where* to write
``topk_stats.csv`` and *how often* without threading an output directory or a
logging cadence through ``scripts/train.py`` / ``scripts/sample.py``. It mirrors
the same ambient-signal idea those two files already give us for free via
``nn.Module.training`` (set by ``.train()`` / ``.eval()``): here,
``ablation_runner._run_one()`` calls :func:`set_context` once at the very start
of every combo (and :func:`set_tag` once per generation seed before each
``sample()`` call); ``DynamicAdjacency.forward`` calls :func:`log_rows` and
never needs to know who set the context or why.

Contract:
  * If no context has been set, :func:`log_rows` is a silent no-op.
  * :func:`log_rows` never raises -- a logging failure must not be able to
    kill a training or sampling run.
  * The file is opened, appended to, and closed on every call (no buffering),
    so a crash mid-run loses at most the one in-flight row.
"""

from __future__ import annotations

import csv
import os
import threading
from typing import Iterable, Optional, Tuple

CSV_COLUMNS = ["step", "block", "t_bucket", "k_mean", "n_samples", "phase", "tag"]

_lock = threading.Lock()
_state = {"output_dir": None, "log_every": 100, "tag": ""}


def set_context(output_dir: str, log_every: int = 100, tag: str = "") -> None:
    """Point logging at ``output_dir`` (the combo's ``parent_dir``).

    Called once at the start of every ``_run_one()`` -- for every combo,
    regardless of ``topk_mode`` -- so a later combo can never append to an
    earlier combo's ``topk_stats.csv``.
    """
    with _lock:
        _state["output_dir"] = output_dir
        _state["log_every"] = max(1, int(log_every))
        _state["tag"] = tag


def set_tag(tag: str) -> None:
    """Change only the tag (e.g. the generation seed) without touching the
    rest of the context. Used before each ``sample()`` call within one combo;
    training logs with tag="" (set by :func:`set_context`)."""
    with _lock:
        _state["tag"] = tag


def _csv_path() -> Optional[str]:
    output_dir = _state.get("output_dir")
    return os.path.join(output_dir, "topk_stats.csv") if output_dir else None


def log_rows(
    step: int,
    block,
    rows: Iterable[Tuple[int, float, int]],
    phase: str,
) -> None:
    """Append one CSV row per occupied timestep bucket.

    Args:
        step:  training step (phase="train") or a sampling-call counter
               (phase="sample"). Both are 0-indexed and phase-local.
        block: block index within the model.
        rows:  iterable of (t_bucket, k_mean, n_samples) for this forward call.
        phase: "train" or "sample".

    phase="train" rows are throttled to every ``log_every`` steps (set via
    :func:`set_context`); phase="sample" rows are written unconditionally.
    Never raises.
    """
    try:
        if _state.get("output_dir") is None:
            return
        if phase == "train" and (step % _state["log_every"] != 0):
            return

        path = _csv_path()
        if path is None:
            return

        rows = list(rows)
        if not rows:
            return

        write_header = not os.path.exists(path)
        tag = _state.get("tag", "")
        with _lock:
            with open(path, "a", newline="") as fh:
                writer = csv.writer(fh)
                if write_header:
                    writer.writerow(CSV_COLUMNS)
                for t_bucket, k_mean, n_samples in rows:
                    writer.writerow([step, block, t_bucket, k_mean, n_samples, phase, tag])
    except Exception:
        # A logging failure must never be able to kill a training/sampling run.
        pass
