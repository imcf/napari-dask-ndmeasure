"""Dock widget: pick an Image + Labels layer, measure, browse/export the table."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ._measure import DEFAULT_STATS, available_stats, iter_measure_labels

if TYPE_CHECKING:
    import napari
    import pandas as pd


def _level_data(layer, level: int):
    """Return one dask array from a (possibly multiscale) layer's data."""
    data = layer.data
    return data[level] if layer.multiscale else data


def _sequential_ids_hint(labels_layer, level: int) -> "np.ndarray | None":
    """Build the exact id array from a Labels layer's known-object-count hint.

    patchworks' ``view_in_napari`` sets ``layer.metadata["n_objects"]`` /
    ``["sequential_labels"]`` when the labels were renumbered to a
    contiguous ``1..N`` range at write time (``sequential_labels=True``
    during the merge) — the exact id set is then ``range(1, N + 1)`` by
    construction, with no need to scan the array for it.

    Only trusted at ``level == 0`` (full resolution): a coarser pyramid
    level's strided downsampling can drop small objects entirely, so the
    id set from level 0 doesn't necessarily hold there.

    Parameters
    ----------
    labels_layer : napari.layers.Labels
        The selected Labels layer.
    level : int
        The pyramid level about to be measured.

    Returns
    -------
    np.ndarray or None
        ``np.arange(1, n_objects + 1)`` if the hint is present and
        trustworthy at this level, else ``None``.
    """
    if level != 0:
        return None
    meta = labels_layer.metadata
    if not meta.get("sequential_labels") or "n_objects" not in meta:
        return None
    return np.arange(1, int(meta["n_objects"]) + 1, dtype="int64")


class MeasureWidget(QWidget):
    """Measure a Labels layer against an Image layer via a chunk-local map/merge.

    Works directly on the layers' backing dask arrays — never pulls a
    hundred-thousand-object volume into RAM the way
    ``skimage.measure.regionprops`` would.

    Every successful measurement (fresh or cached) is written to CSV
    automatically, with no save dialog — see :meth:`_auto_save_csv`. Use
    **Save CSV…** to choose a different location; that location is then
    remembered for subsequent automatic saves too.
    """

    def __init__(self, napari_viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = napari_viewer
        self._table = None  # last computed pandas.DataFrame
        self._worker = None  # keeps the running thread_worker alive
        # Session-only cache keyed by (id(image data), id(labels data),
        # level, stats, scale) -> result table. id()-based: valid as long as
        # the layer (and its underlying data object) is still alive, which
        # is also exactly the right invalidation — a reloaded/replaced layer
        # gets a new data object and a fresh id, so it's never a stale hit.
        # ponytail: doesn't survive closing the widget/viewer (in-memory
        # only) and never evicts — fine for a session's worth of measurement
        # results (tiny compared to the images). Upgrade path: a disk cache
        # keyed by (zarr store path, component, level, stats) if cross-
        # session reuse on very expensive measurements is ever needed.
        self._cache: dict[tuple, "pd.DataFrame"] = {}
        # Where auto-save (see _on_measured) writes CSVs. None -> cwd, until
        # the user manually picks a location once via "Save CSV…", which is
        # remembered for subsequent auto-saves too.
        self._save_dir: Path | None = None

        layout = QVBoxLayout()
        self.setLayout(layout)

        layers_box = QGroupBox("Layers")
        layers_layout = QVBoxLayout()
        layers_box.setLayout(layers_layout)

        row = QHBoxLayout()
        row.addWidget(QLabel("Image:"))
        self.image_combo = QComboBox()
        row.addWidget(self.image_combo)
        layers_layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Labels:"))
        self.labels_combo = QComboBox()
        row.addWidget(self.labels_combo)
        layers_layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Pyramid level:"))
        self.level_spin = QSpinBox()
        self.level_spin.setMinimum(0)
        self.level_spin.setToolTip(
            "Multiscale layers only: 0 = full resolution. Coarser levels are "
            "faster but less precise — a quick preview before a full run."
        )
        row.addWidget(self.level_spin)
        layers_layout.addLayout(row)

        layout.addWidget(layers_box)

        stats_box = QGroupBox("Measurements")
        stats_layout = QVBoxLayout()
        stats_box.setLayout(stats_layout)
        self.stats_list = QListWidget()
        for name in available_stats():
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(
                Qt.Checked if name in DEFAULT_STATS else Qt.Unchecked
            )
            self.stats_list.addItem(item)
        stats_layout.addWidget(self.stats_list)
        layout.addWidget(stats_box)

        row = QHBoxLayout()
        row.addWidget(QLabel("Workers:"))
        self.workers_spin = QSpinBox()
        self.workers_spin.setMinimum(1)
        self.workers_spin.setMaximum(max(1, os.cpu_count() or 1))
        self.workers_spin.setValue(min(4, os.cpu_count() or 1))
        self.workers_spin.setToolTip(
            "Threads for the measurement. More = faster, but more decoded "
            "chunks held in memory at once. Lower this (even to 1) if "
            "measuring uses too much RAM."
        )
        row.addWidget(self.workers_spin)
        layout.addLayout(row)

        self.measure_btn = QPushButton("Measure")
        self.measure_btn.clicked.connect(self._on_measure_clicked)
        layout.addWidget(self.measure_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        self.results_table = QTableWidget()
        layout.addWidget(self.results_table)

        self.save_btn = QPushButton("Save CSV…")
        self.save_btn.clicked.connect(self._on_save_clicked)
        self.save_btn.setEnabled(False)
        layout.addWidget(self.save_btn)

        self._viewer.layers.events.inserted.connect(self._refresh_layer_choices)
        self._viewer.layers.events.removed.connect(self._refresh_layer_choices)
        self.labels_combo.currentIndexChanged.connect(self._update_level_range)
        self._refresh_layer_choices()

    def _update_level_range(self):
        _, labels_layer = self._selected_layers()
        if labels_layer is not None and labels_layer.multiscale:
            self.level_spin.setMaximum(len(labels_layer.data) - 1)
        else:
            self.level_spin.setMaximum(0)

    def _refresh_layer_choices(self, event=None):
        from napari.layers import Image, Labels

        prev_image = self.image_combo.currentText()
        prev_labels = self.labels_combo.currentText()
        self.image_combo.clear()
        self.labels_combo.clear()
        for layer in self._viewer.layers:
            if isinstance(layer, Image):
                self.image_combo.addItem(layer.name)
            elif isinstance(layer, Labels):
                self.labels_combo.addItem(layer.name)
        _restore(self.image_combo, prev_image)
        _restore(self.labels_combo, prev_labels)
        self._update_level_range()

    def _selected_layers(self):
        image_name = self.image_combo.currentText()
        labels_name = self.labels_combo.currentText()
        if not image_name or not labels_name:
            return None, None
        return self._viewer.layers[image_name], self._viewer.layers[labels_name]

    def _selected_stats(self) -> tuple[str, ...]:
        stats = []
        for i in range(self.stats_list.count()):
            item = self.stats_list.item(i)
            if item.checkState() == Qt.Checked:
                stats.append(item.text())
        return tuple(stats)

    def _on_measure_clicked(self):
        """Validate the current selection, then run (or reuse a cached) measurement.

        A cache hit populates the results table immediately, with no
        background thread. Otherwise a threaded, progress-reporting
        measurement is started — see :meth:`_on_progress`/:meth:`_on_measured`.
        """
        from napari.qt.threading import thread_worker

        image_layer, labels_layer = self._selected_layers()
        if image_layer is None or labels_layer is None:
            QMessageBox.warning(
                self, "Missing layers", "Pick an Image and a Labels layer."
            )
            return
        stats = self._selected_stats()
        if not stats:
            QMessageBox.warning(
                self, "No measurements selected", "Check at least one."
            )
            return

        level = self.level_spin.value()
        image_data = _level_data(image_layer, level)
        labels_data = _level_data(labels_layer, level)
        scale = tuple(labels_layer.scale[-labels_data.ndim :])

        cache_key = (id(image_data), id(labels_data), level, stats, scale)
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._on_measured(cached, cache_key=cache_key, from_cache=True)
            return

        ids_hint = _sequential_ids_hint(labels_layer, level)

        self.measure_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        # min == max is Qt's native "busy" (indeterminate) mode — start here
        # rather than a determinate 0/N, since the first real work
        # (iter_measure_labels' whole-volume object scan) has no natural
        # sub-progress and can take a while on a huge volume; a literal 0/N
        # bar would look frozen for that whole stretch.
        self.progress_bar.setRange(0, 0)
        if ids_hint is not None:
            self.status_label.setText(
                f"Measuring… using known object count "
                f"({ids_hint.size}) — skipping scan."
            )
        else:
            self.status_label.setText("Measuring…")

        n_workers = self.workers_spin.value()

        @thread_worker
        def _run():
            result = yield from iter_measure_labels(
                image_data,
                labels_data,
                stats=stats,
                scale=scale,
                n_workers=n_workers,
                ids=ids_hint,
            )
            return result

        # Keep a reference on self — an unreferenced worker can be garbage
        # collected mid-run, silently killing the thread before it finishes.
        self._worker = _run()
        self._worker.yielded.connect(self._on_progress)
        self._worker.returned.connect(
            lambda table: self._on_measured(table, cache_key=cache_key)
        )
        self._worker.errored.connect(self._on_measure_error)
        self._worker.start()

    def _on_progress(self, progress: tuple[int, int, str]) -> None:
        """Update the progress bar from an ``iter_measure_labels`` yield.

        Parameters
        ----------
        progress : tuple of (int, int, str)
            ``(done, total, stat_name)`` as yielded by
            :func:`napari_chunked_regionprops._measure.iter_measure_labels`.
        """
        done, total, stat_name = progress
        self.progress_bar.setMaximum(
            total
        )  # total=0 -> indeterminate (Qt: min==max)
        self.progress_bar.setValue(done)
        if total == 0:
            self.status_label.setText(f"Measuring… {stat_name}…")
        else:
            self.status_label.setText(
                f"Measuring… {stat_name} ({done}/{total})"
            )

    def _on_measure_error(self, exc: Exception):
        self.measure_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("")
        QMessageBox.critical(self, "Measurement failed", str(exc))

    def _on_measured(
        self,
        table: "pd.DataFrame",
        *,
        cache_key: tuple,
        from_cache: bool = False,
    ) -> None:
        """Finish a measurement: cache it, populate the table, update the layer.

        Parameters
        ----------
        table : pandas.DataFrame
            The measurement result, indexed by label id.
        cache_key : tuple
            Key this result should be (or already is) stored under in
            :attr:`_cache`.
        from_cache : bool, optional
            Whether *table* came from the cache rather than a fresh
            computation (skips re-inserting it). Default ``False``.
        """
        self.measure_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        if not from_cache:
            self._cache[cache_key] = table
        self._table = table
        suffix = " (cached)" if from_cache else ""
        status = f"{len(table)} objects measured.{suffix}"
        self._populate_table(table)
        self.save_btn.setEnabled(not table.empty)

        if not table.empty:
            saved_to = self._auto_save_csv(table)
            status += f" Saved to {saved_to}."
        self.status_label.setText(status)

        _, labels_layer = self._selected_layers()
        if labels_layer is not None and not table.empty:
            # feeds napari's built-in per-label feature display/coloring.
            features = table.reset_index()
            features["label"] = features["label"].astype(int)
            labels_layer.features = features

    def _auto_save_csv(self, table: "pd.DataFrame") -> Path:
        """Write *table* to CSV without prompting, every time a measurement finishes.

        Parameters
        ----------
        table : pandas.DataFrame
            The measurement result to save.

        Returns
        -------
        Path
            Where it was written: :attr:`_save_dir` (the last manually
            chosen "Save CSV…" directory) if set, else the current working
            directory, using :meth:`_default_csv_name`.
        """
        directory = self._save_dir or Path.cwd()
        path = directory / self._default_csv_name()
        table.to_csv(path)
        return path

    def _populate_table(self, table):
        self.results_table.clear()
        self.results_table.setRowCount(len(table))
        columns = ["label", *table.columns]
        self.results_table.setColumnCount(len(columns))
        self.results_table.setHorizontalHeaderLabels(columns)
        for row, (label_id, values) in enumerate(table.iterrows()):
            self.results_table.setItem(row, 0, QTableWidgetItem(str(label_id)))
            for col, value in enumerate(values, start=1):
                self.results_table.setItem(
                    row, col, QTableWidgetItem(f"{value:.4g}")
                )

    def _default_csv_name(self) -> str:
        """Suggested CSV filename, derived from the selected Labels layer.

        Returns
        -------
        str
            ``"<labels layer name>_measurements.csv"``, or
            ``"measurements.csv"`` if no Labels layer is selected.
        """
        _, labels_layer = self._selected_layers()
        stem = labels_layer.name if labels_layer is not None else "measurements"
        return f"{stem}_measurements.csv"

    def _on_save_clicked(self) -> None:
        """Prompt for a path, write the CSV there, and remember the folder.

        The chosen folder becomes the target for future automatic saves
        (see :meth:`_auto_save_csv`) too, instead of the default cwd.
        """
        if self._table is None or self._table.empty:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save measurements", self._default_csv_name(), "CSV (*.csv)"
        )
        if path:
            self._table.to_csv(path)
            self._save_dir = Path(path).parent


def _restore(combo: QComboBox, previous: str) -> None:
    idx = combo.findText(previous)
    if idx >= 0:
        combo.setCurrentIndex(idx)
