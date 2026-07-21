"""Dock widget: pick an Image + Labels layer, measure, browse/export the table."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
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

#: Stats that depend only on the Labels layer, not any intensity image —
#: computed once regardless of how many Image channels are checked, unlike
#: every other stat (measured per channel, see _on_measure_clicked).
_GEOMETRIC_STATS = frozenset({"area_voxels", "centroid"})


class _NumericTableWidgetItem(QTableWidgetItem):
    """A results-table cell that sorts by numeric value, not cell text.

    Plain ``QTableWidgetItem`` sorts lexicographically ("10" < "2"), which
    is wrong for label ids and measurement columns.
    """

    def __lt__(self, other: QTableWidgetItem) -> bool:
        try:
            return float(self.text()) < float(other.text())
        except ValueError:
            return super().__lt__(other)


def _level_data(layer, level: int):
    """Return one dask array from a (possibly multiscale) layer's data."""
    data = layer.data
    return data[level] if layer.multiscale else data


def _channel_slices(name: str, data, labels_shape: tuple[int, ...]):
    """Split one checked Image layer into ``(display_name, array)`` channels.

    Ordinarily an Image layer's data already matches the Labels layer shape
    exactly -- this plugin's "check every channel" list expects one napari
    layer per channel. Some readers instead load every channel into a
    single layer with a leading channel axis (shape ``(C, *labels_shape)``),
    which used to surface as an opaque "shape mismatch" error from
    :func:`iter_measure_labels`. Slice that axis into *C* per-channel
    arrays here instead, so it measures the same as *C* separate layers.
    """
    if data.shape == labels_shape:
        return [(name, data)]
    if data.ndim == len(labels_shape) + 1 and data.shape[1:] == labels_shape:
        return [(f"{name}_C{i}", data[i]) for i in range(data.shape[0])]
    raise ValueError(
        f"image layer {name!r} has shape {data.shape}, incompatible with "
        f"labels shape {labels_shape} (expected an exact match, or a "
        f"leading channel axis of shape ({data.shape[0] if data.ndim else '?'}, "
        f"{', '.join(map(str, labels_shape))}))"
    )


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

    A disk-backed manifest (see :meth:`_disk_cache_key`) also lets a fresh
    widget instance — e.g. after closing and reopening napari — reuse a
    prior measurement's CSV instead of recomputing it.
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
        # results (tiny compared to the images). Cross-session reuse is
        # handled separately by the disk manifest (_disk_cache_key et al.).
        self._cache: dict[tuple, "pd.DataFrame"] = {}
        # Where auto-save (see _on_measured) writes CSVs. None -> cwd, until
        # the user manually picks a location once via "Save CSV…", which is
        # remembered for subsequent auto-saves too.
        self._save_dir: Path | None = None
        # The Labels layer currently wearing a highlight colormap (see
        # _apply_highlight) and its original colormap, so Clear can put it
        # back exactly. None when no highlight is active.
        self._highlighted_labels_layer = None
        self._orig_colormap = None
        # The Labels layer currently recolored by a measurement column (see
        # _apply_measurement_colormap) and its colormap from just before
        # that, so "Reset colors" can restore it exactly.
        self._measurement_colored_layer = None
        self._pre_measurement_colormap = None

        layout = QVBoxLayout()
        self.setLayout(layout)

        layers_box = QGroupBox("Layers")
        layers_layout = QVBoxLayout()
        layers_box.setLayout(layers_layout)

        layers_layout.addWidget(
            QLabel("Image(s) — check every channel to measure:")
        )
        self.image_list = QListWidget()
        layers_layout.addWidget(self.image_list)

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

        reload_box = QGroupBox("Reload previous results")
        reload_layout = QHBoxLayout()
        reload_box.setLayout(reload_layout)
        self.reload_path_edit = QLineEdit()
        self.reload_path_edit.setPlaceholderText(
            "path to a previous *_measurements.csv"
        )
        reload_layout.addWidget(self.reload_path_edit)
        self.reload_browse_btn = QPushButton("Browse…")
        self.reload_browse_btn.clicked.connect(self._on_reload_browse_clicked)
        reload_layout.addWidget(self.reload_browse_btn)
        self.reload_btn = QPushButton("Load")
        self.reload_btn.clicked.connect(self._on_reload_clicked)
        reload_layout.addWidget(self.reload_btn)
        layout.addWidget(reload_box)

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
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.results_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.results_table.cellClicked.connect(self._on_result_row_clicked)
        self.results_table.itemSelectionChanged.connect(
            self._on_result_selection_changed
        )
        layout.addWidget(self.results_table)

        self.clear_selection_btn = QPushButton("Clear selection")
        self.clear_selection_btn.clicked.connect(
            self._on_clear_selection_clicked
        )
        self.clear_selection_btn.setEnabled(False)
        layout.addWidget(self.clear_selection_btn)

        colormap_box = QGroupBox("Color by measurement")
        colormap_layout = QHBoxLayout()
        colormap_box.setLayout(colormap_layout)
        colormap_layout.addWidget(QLabel("Column:"))
        self.colormap_column_combo = QComboBox()
        colormap_layout.addWidget(self.colormap_column_combo)
        colormap_layout.addWidget(QLabel("LUT:"))
        self.colormap_name_combo = QComboBox()
        from napari.utils.colormaps import AVAILABLE_COLORMAPS

        self.colormap_name_combo.addItems(sorted(AVAILABLE_COLORMAPS))
        self.colormap_name_combo.setCurrentText("viridis")
        colormap_layout.addWidget(self.colormap_name_combo)
        self.apply_colormap_btn = QPushButton("Apply")
        self.apply_colormap_btn.clicked.connect(
            self._on_apply_colormap_clicked
        )
        colormap_layout.addWidget(self.apply_colormap_btn)
        self.reset_colors_btn = QPushButton("Reset colors")
        self.reset_colors_btn.clicked.connect(self._on_reset_colors_clicked)
        self.reset_colors_btn.setEnabled(False)
        colormap_layout.addWidget(self.reset_colors_btn)
        layout.addWidget(colormap_box)

        self.save_btn = QPushButton("Save CSV…")
        self.save_btn.clicked.connect(self._on_save_clicked)
        self.save_btn.setEnabled(False)
        layout.addWidget(self.save_btn)

        self._viewer.layers.events.inserted.connect(self._refresh_layer_choices)
        self._viewer.layers.events.removed.connect(self._refresh_layer_choices)
        self.labels_combo.currentIndexChanged.connect(self._update_level_range)
        # On the viewer, not the Labels layer: a layer's own mouse_drag_
        # callbacks only fire while it's napari's *active* layer (see
        # napari/_vispy/canvas.py), so clicking the image would silently do
        # nothing whenever some other layer happened to be selected in the
        # layers list. The viewer's callbacks fire unconditionally; we look
        # up the current Labels selection ourselves inside the handler.
        self._viewer.mouse_drag_callbacks.append(self._on_image_clicked)
        self._refresh_layer_choices()

    def _update_level_range(self):
        _, labels_layer = self._selected_layers()
        if labels_layer is not None and labels_layer.multiscale:
            self.level_spin.setMaximum(len(labels_layer.data) - 1)
        else:
            self.level_spin.setMaximum(0)

    def _refresh_layer_choices(self, event=None):
        from napari.layers import Image, Labels

        prev_checked = {
            self.image_list.item(i).text()
            for i in range(self.image_list.count())
            if self.image_list.item(i).checkState() == Qt.Checked
        }
        first_population = self.image_list.count() == 0
        prev_labels = self.labels_combo.currentText()

        self.image_list.clear()
        self.labels_combo.clear()
        for layer in self._viewer.layers:
            if isinstance(layer, Image):
                item = QListWidgetItem(layer.name)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                # first time populating: default to "measure every channel"
                # rather than none, so a fresh widget's Measure button just
                # works. After that, respect what the user already chose.
                checked = first_population or layer.name in prev_checked
                item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
                self.image_list.addItem(item)
            elif isinstance(layer, Labels):
                self.labels_combo.addItem(layer.name)
        _restore(self.labels_combo, prev_labels)
        self._update_level_range()

    def _selected_image_layers(self) -> list:
        names = [
            self.image_list.item(i).text()
            for i in range(self.image_list.count())
            if self.image_list.item(i).checkState() == Qt.Checked
        ]
        return [
            self._viewer.layers[name]
            for name in names
            if name in self._viewer.layers
        ]

    def _selected_layers(self):
        """Return ``(image_layers, labels_layer)``.

        *image_layers* is every checked entry in the Image list (possibly
        several channels), *labels_layer* the current Labels selection.
        """
        labels_name = self.labels_combo.currentText()
        labels_layer = self._viewer.layers[labels_name] if labels_name else None
        return self._selected_image_layers(), labels_layer

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

        With multiple Image layers checked, intensity stats (everything
        except ``area_voxels``/``centroid``) are measured once per channel and
        suffixed with the channel's layer name (e.g.
        ``mean_intensity_DAPI``) — see the ``_run`` worker below. A checked
        layer whose data already bundles every channel into one array
        (shape ``(C, *labels_shape)``, e.g. a reader that doesn't split
        channels into separate layers) is expanded into *C* channels the
        same way — see :func:`_channel_slices`.
        """
        from napari.qt.threading import thread_worker

        image_layers, labels_layer = self._selected_layers()
        if not image_layers or labels_layer is None:
            QMessageBox.warning(
                self,
                "Missing layers",
                "Check at least one Image layer and pick a Labels layer.",
            )
            return
        stats = self._selected_stats()
        if not stats:
            QMessageBox.warning(
                self, "No measurements selected", "Check at least one."
            )
            return

        level = self.level_spin.value()
        labels_data = _level_data(labels_layer, level)
        scale = tuple(labels_layer.scale[-labels_data.ndim :])
        try:
            channels = [
                channel
                for layer in image_layers
                for channel in _channel_slices(
                    layer.name, _level_data(layer, level), labels_data.shape
                )
            ]
        except ValueError as exc:
            QMessageBox.critical(self, "Shape mismatch", str(exc))
            return

        cache_key = (
            tuple(id(data) for _, data in channels),
            id(labels_data),
            level,
            stats,
            scale,
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._on_measured(
                cached, cache_key=cache_key, cache_note="session cache"
            )
            return

        disk_key = self._disk_cache_key(
            image_layers,
            labels_layer,
            channels,
            labels_data,
            level,
            stats,
            scale,
        )
        disk_entry = self._load_disk_cache().get(disk_key)
        if disk_entry is not None and Path(disk_entry).exists():
            try:
                table = pd.read_csv(disk_entry, index_col="label")
            except (OSError, ValueError):
                table = None
            if table is not None:
                self._cache[cache_key] = table
                self._on_measured(
                    table, cache_key=cache_key, cache_note="disk cache"
                )
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
        geometric_stats = tuple(s for s in stats if s in _GEOMETRIC_STATS)
        intensity_stats = tuple(s for s in stats if s not in _GEOMETRIC_STATS)
        multi_channel = len(channels) > 1

        @thread_worker
        def _run():
            # ids discovered by whichever call runs first (the whole-volume
            # scan, if not skipped via ids_hint) are reused for every
            # subsequent call — no need to rescan per channel.
            ids = ids_hint
            tables = []

            def _run_one(data, stats_subset, label_prefix):
                nonlocal ids
                gen = iter_measure_labels(
                    data,
                    labels_data,
                    stats=stats_subset,
                    scale=scale,
                    n_workers=n_workers,
                    ids=ids,
                )
                try:
                    while True:
                        done, total, phase = next(gen)
                        yield (
                            done,
                            total,
                            (
                                f"{label_prefix}: {phase}"
                                if label_prefix
                                else phase
                            ),
                        )
                except StopIteration as stop:
                    table = stop.value
                if ids is None:
                    ids = table.index.to_numpy()
                return table

            if geometric_stats:
                tables.append(
                    (yield from _run_one(channels[0][1], geometric_stats, None))
                )
            if intensity_stats:
                for name, data in channels:
                    channel_table = yield from _run_one(
                        data, intensity_stats, name if multi_channel else None
                    )
                    if multi_channel:
                        channel_table = channel_table.add_suffix(f"_{name}")
                    tables.append(channel_table)

            result = tables[0]
            for extra in tables[1:]:
                result = result.join(extra)
            return result

        # Keep a reference on self — an unreferenced worker can be garbage
        # collected mid-run, silently killing the thread before it finishes.
        self._worker = _run()
        self._worker.yielded.connect(self._on_progress)
        self._worker.returned.connect(
            lambda table: self._on_measured(
                table, cache_key=cache_key, disk_key=disk_key
            )
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
        cache_note: str | None = None,
        disk_key: str | None = None,
    ) -> None:
        """Finish a measurement: cache it, populate the table, update the layer.

        Parameters
        ----------
        table : pandas.DataFrame
            The measurement result, indexed by label id.
        cache_key : tuple
            Key this result should be (or already is) stored under in
            :attr:`_cache`.
        cache_note : str, optional
            If *table* came from a cache rather than a fresh computation,
            a short label for the status text (e.g. ``"session cache"``)
            and a signal to skip re-inserting it into :attr:`_cache`
            (already there). ``None`` (default) means this was a fresh
            computation.
        disk_key : str, optional
            Disk-manifest key (see :meth:`_disk_cache_key`) to record
            *table*'s saved CSV path under, so a future widget instance
            can reuse it. Only meaningful for a fresh computation.
        """
        self.measure_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        if cache_note is None:
            self._cache[cache_key] = table
        self._table = table
        suffix = f" ({cache_note})" if cache_note else ""
        status = f"{len(table)} objects measured.{suffix}"
        self._populate_table(table)
        self._refresh_colormap_columns(table)
        self.save_btn.setEnabled(not table.empty)

        if not table.empty:
            saved_to = self._auto_save_csv(table)
            status += f" Saved to {saved_to}."
            if disk_key is not None:
                self._update_disk_cache(disk_key, saved_to)
        self.status_label.setText(status)

        _, labels_layer = self._selected_layers()
        self._wire_labels_features(labels_layer, table)

    def _wire_labels_features(
        self, labels_layer, table: "pd.DataFrame"
    ) -> None:
        """Feed *table* into the Labels layer's per-label features/coloring."""
        if labels_layer is None or table.empty:
            return
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

    def _disk_cache_key(
        self,
        image_layers,
        labels_layer,
        channels,
        labels_data,
        level,
        stats,
        scale,
    ) -> str:
        """Build a cross-session cache key from stable layer identity.

        Unlike the in-memory :attr:`_cache` key (``id()``-based, dies with
        the layer object), this survives closing and reopening napari: a
        layer's ``source.path`` (the file it was read from) is stable
        across sessions. Falls back to layer name + array shape/dtype for
        in-memory-only layers with no source path.

        *image_layers*/*channels* are parallel sequences (layer objects,
        (name, data) pairs) for every checked Image channel — order
        matters (it's part of the key), which is fine since it always
        comes from the same stable Image-list iteration order.

        ponytail: the fallback identifies by name, not content — a
        different array reusing the same layer name would false-hit.
        Fine for the common case (reload the same file); upgrade path
        would hash a data sample if that ever bites someone.
        """

        def source_id(layer, data) -> str:
            path = layer.source.path
            return (
                str(path)
                if path
                else f"name:{layer.name}:{data.shape}:{data.dtype}"
            )

        return json.dumps(
            [
                [
                    source_id(layer, data)
                    for layer, (_, data) in zip(image_layers, channels)
                ],
                source_id(labels_layer, labels_data),
                level,
                list(stats),
                list(scale),
            ]
        )

    def _disk_cache_path(self) -> Path:
        """Manifest file location: alongside where CSVs are auto-saved."""
        return (self._save_dir or Path.cwd()) / ".regionprops_cache.json"

    def _load_disk_cache(self) -> dict:
        path = self._disk_cache_path()
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}

    def _update_disk_cache(self, key: str, csv_path: Path) -> None:
        cache = self._load_disk_cache()
        cache[key] = str(csv_path.resolve())
        try:
            self._disk_cache_path().write_text(json.dumps(cache))
        except OSError:
            pass

    def _populate_table(self, table):
        # Sorting must be off while populating: with it on, Qt re-sorts
        # after every setItem call, so a row can move out from under us
        # mid-population and later columns land in the wrong row.
        self.results_table.setSortingEnabled(False)
        self.results_table.clear()
        self.results_table.setRowCount(len(table))
        columns = ["label", *table.columns]
        self.results_table.setColumnCount(len(columns))
        self.results_table.setHorizontalHeaderLabels(columns)
        for row, (label_id, values) in enumerate(table.iterrows()):
            self.results_table.setItem(
                row, 0, _NumericTableWidgetItem(str(label_id))
            )
            for col, value in enumerate(values, start=1):
                self.results_table.setItem(
                    row, col, _NumericTableWidgetItem(f"{value:.4g}")
                )
        self.results_table.setSortingEnabled(True)

    def _refresh_colormap_columns(self, table: "pd.DataFrame") -> None:
        """Repopulate the "Color by measurement" column picker with
        *table*'s numeric columns, keeping the previous selection if it's
        still there."""
        previous = self.colormap_column_combo.currentText()
        self.colormap_column_combo.clear()
        self.colormap_column_combo.addItems(
            list(table.select_dtypes(include="number").columns)
        )
        _restore(self.colormap_column_combo, previous)

    def _on_apply_colormap_clicked(self) -> None:
        if self._table is None or self._table.empty:
            return
        column = self.colormap_column_combo.currentText()
        if not column:
            QMessageBox.warning(
                self, "No column", "Pick a measurement column first."
            )
            return
        _, labels_layer = self._selected_layers()
        if labels_layer is None:
            return
        self._apply_measurement_colormap(
            labels_layer, column, self.colormap_name_combo.currentText()
        )
        self.reset_colors_btn.setEnabled(True)

    def _apply_measurement_colormap(
        self, labels_layer, column: str, colormap_name: str
    ) -> None:
        """Recolor every labelled object along *colormap_name* by its
        *column* value in :attr:`_table` — the smallest value gets the
        low end of the LUT, the largest the high end — replacing napari's
        default random per-label colors. Objects tied at the same value
        (or if every value is equal) get the LUT's midpoint.

        Clears any active row-selection highlight first (see
        :meth:`_apply_highlight`): recoloring the base view while a
        highlight overlay is active would otherwise leave that
        highlight's "restore to" colormap pointing at stale colors.
        """
        from napari.utils.colormaps import DirectLabelColormap, ensure_colormap

        if labels_layer is self._highlighted_labels_layer:
            self.results_table.clearSelection()
            self._highlighted_labels_layer = None
            self._orig_colormap = None

        if labels_layer is not self._measurement_colored_layer:
            self._pre_measurement_colormap = labels_layer.colormap
            self._measurement_colored_layer = labels_layer

        values = self._table[column].astype(float)
        vmin, vmax = values.min(), values.max()
        span = vmax - vmin
        normalized = (
            np.full(len(values), 0.5)
            if span == 0
            else ((values - vmin) / span).to_numpy()
        )
        rgba = ensure_colormap(colormap_name).map(normalized)

        color_dict = {
            int(label_id): rgba[i]
            for i, label_id in enumerate(self._table.index)
        }
        color_dict[0] = np.zeros(4, dtype="float32")
        color_dict[None] = np.zeros(4, dtype="float32")
        labels_layer.colormap = DirectLabelColormap(color_dict=color_dict)

    def _on_reset_colors_clicked(self) -> None:
        """Undo :meth:`_apply_measurement_colormap`, restoring whatever
        coloring the Labels layer had right before it was first applied."""
        if self._measurement_colored_layer is not None:
            self._measurement_colored_layer.colormap = (
                self._pre_measurement_colormap
            )
            self._measurement_colored_layer = None
            self._pre_measurement_colormap = None
        self.reset_colors_btn.setEnabled(False)

    def _on_result_row_clicked(self, row: int, column: int) -> None:
        """Clicking a results-table row centers/zooms the camera on that
        object (the row just clicked, even if it's part of a larger
        multi-row selection — see :meth:`_on_result_selection_changed` for
        the highlight itself).

        The zoom matters: with tens of thousands of objects in a volume,
        recoloring the selected label(s) alone is invisible unless the
        camera is already looking straight at one of them.
        """
        item = self.results_table.item(row, 0)
        if item is None:
            return
        _, labels_layer = self._selected_layers()
        if labels_layer is None:
            return
        self._center_camera_on_label(labels_layer, int(item.text()))

    def _on_result_selection_changed(self) -> None:
        """Highlight every currently-selected row's object in the image,
        dimming everything else — supports selecting several rows at once
        (Ctrl/Shift-click, standard Qt multi-select). Restores normal
        coloring the moment the selection becomes empty (see
        :meth:`_on_clear_selection_clicked`).
        """
        _, labels_layer = self._selected_layers()
        if labels_layer is None:
            return
        label_ids = {
            int(item.text())
            for item in self.results_table.selectedItems()
            if item.column() == 0
        }
        self.clear_selection_btn.setEnabled(bool(label_ids))
        if label_ids:
            self._apply_highlight(labels_layer, label_ids)
        elif self._highlighted_labels_layer is not None:
            self._highlighted_labels_layer.colormap = self._orig_colormap
            self._highlighted_labels_layer = None
            self._orig_colormap = None

    def _apply_highlight(self, labels_layer, label_ids: set) -> None:
        """Recolor *labels_layer* so only *label_ids* show at full color;
        every other object (and the background) fades out.

        Remembers *labels_layer*'s original colormap the first time this
        runs for it, so :meth:`_on_result_selection_changed` (on an empty
        selection) or :meth:`_on_clear_selection_clicked` can restore it
        exactly.

        ponytail: only ever remembers one layer's original colormap at a
        time — switching the Labels selection to a *different* layer
        while one is already highlighted leaves that first layer stuck
        dimmed. Not worth a dict of originals for what's normally a
        single-Labels-layer workflow.
        """
        from napari.utils.colormaps import DirectLabelColormap

        if labels_layer is not self._highlighted_labels_layer:
            self._orig_colormap = labels_layer.colormap
            self._highlighted_labels_layer = labels_layer

        highlight = np.array([1.0, 1.0, 0.0, 1.0], dtype="float32")
        dim = np.array([0.3, 0.3, 0.3, 0.15], dtype="float32")
        # Explicit per-label entry for every known object, rather than
        # relying on DirectLabelColormap's `None`-key fallback for
        # "everything else" — that fallback wasn't reliably dimming
        # unselected objects in practice (they all rendered highlighted).
        color_dict = {
            int(label_id): (highlight if label_id in label_ids else dim)
            for label_id in self._table.index
        }
        color_dict[0] = np.array([0.0, 0.0, 0.0, 0.0], dtype="float32")
        # Safety net for any id not in the table (shouldn't happen, but a
        # stale/mismatched ids hint could produce one) -- dim rather than
        # napari's own default of fully transparent for unlisted keys.
        color_dict[None] = dim
        labels_layer.colormap = DirectLabelColormap(color_dict=color_dict)

    def _on_clear_selection_clicked(self) -> None:
        """Deselect every row — triggers :meth:`_on_result_selection_changed`,
        which restores the Labels layer's original coloring since the
        selection becomes empty."""
        self.results_table.clearSelection()

    def _center_camera_on_label(self, labels_layer, label_id: int) -> None:
        """Center and zoom the camera on *label_id*'s object, jumping to
        its z-slice too if the labels array is 3D.

        Needs the ``centroid`` and ``area_voxels`` stats to have been measured
        (or reloaded) for this label — silently does nothing otherwise.

        ponytail: "linear size" is a rough ``area_voxels ** (1/ndim)`` estimate
        (voxel count -> a cube-root/sqrt length), not the object's real
        bounding box, so very elongated objects won't be framed tightly.
        Canvas size comes from the private ``_qt_viewer.canvas`` (no
        public napari accessor for it) — upgrade path if that ever breaks
        across a napari version bump. Z-slice jump only handles plain
        ``(z, y, x)`` volumes (``ndim == 3``), matching the rest of this
        module's dimensionality support — a 4th (e.g. time) axis isn't
        stepped.
        """
        if self._table is None or label_id not in self._table.index:
            return
        row = self._table.loc[label_id]
        ndim = labels_layer.ndim
        centroid_cols = [f"centroid_{ax}" for ax in "zyx"[-ndim:]]
        if "area_voxels" not in row.index or not all(
            c in row.index for c in centroid_cols
        ):
            return

        world_coord = labels_layer.data_to_world(
            [row[c] for c in centroid_cols]
        )
        self._viewer.camera.center = tuple(world_coord)

        if ndim == 3:
            current_step = list(self._viewer.dims.current_step)
            if len(current_step) >= 3:
                current_step[-3] = int(round(row["centroid_z"]))
                self._viewer.dims.current_step = current_step

        linear_size = row["area_voxels"] ** (1 / ndim) * np.mean(
            labels_layer.scale[-ndim:]
        )
        if linear_size <= 0:
            return
        canvas_px = min(self._viewer.window._qt_viewer.canvas.size)
        padding = 6  # object spans roughly 1/padding of the shorter canvas edge
        self._viewer.camera.zoom = canvas_px / (linear_size * padding)

    def _on_image_clicked(self, viewer, event):
        """Clicking a labelled object in the image selects its table row
        (replacing any existing selection) — the row-selection handler
        (:meth:`_on_result_selection_changed`) then applies the highlight.

        Registered on the *viewer's* ``mouse_drag_callbacks`` (see
        :meth:`__init__`), not the Labels layer's — a layer's own callbacks
        only fire while it's napari's active layer, which broke this
        entirely whenever some other layer was selected in the layers list.
        Looks up the current Labels selection itself instead of trusting
        napari's active-layer state.

        A generator (napari's standard click-vs-drag idiom): napari calls
        it once on mouse press, then resumes it on every subsequent
        ``mouse_move``/``mouse_release`` as long as it keeps yielding. A
        canvas pan/zoom *starts* with the same mouse press this callback
        sees, so treating press alone as "clicked an object" turned every
        pan that started on top of a label into a single-object selection —
        dimming every other label and making them appear to vanish as soon
        as the user moved the view. Waiting for release-without-movement
        distinguishes a real click from a drag.

        Note on 3D + multiscale: napari always renders 3D volumes at the
        coarsest pyramid level (``ScalarFieldBase._update_level_and_corners``
        forces this — vispy's Volume visual can't stream tiles the way the
        2D canvas can). Small objects that vanish at that downsampled level
        are therefore unreachable by ray-cast picking in 3D regardless of
        this fix; that's a napari rendering constraint, not something this
        handler can work around.
        """
        _, labels_layer = self._selected_layers()
        if labels_layer is None:
            return
        dragged = False
        yield
        while event.type == "mouse_move":
            dragged = True
            yield
        if dragged:
            return
        value = labels_layer.get_value(
            event.position,
            view_direction=event.view_direction,
            dims_displayed=event.dims_displayed,
            world=True,
        )
        # Multiscale layers return (data_level, value), not a bare value --
        # this plugin's whole point is multiscale/pyramid support, so
        # that's the common case here, not an edge case. Indexing the wrong
        # element (or int()-ing the tuple directly) picks up the pyramid
        # level instead of the label id, which "succeeds" silently and
        # selects the wrong row -- or no row at all, since the pyramid
        # level index rarely matches a real label id in the results table.
        if isinstance(value, tuple):
            value = value[1]
        if not value:
            return
        target = str(int(value))
        for row in range(self.results_table.rowCount()):
            item = self.results_table.item(row, 0)
            if item is not None and item.text() == target:
                self.results_table.setCurrentCell(row, 0)
                self.results_table.scrollToItem(item)
                break

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

    def _on_reload_browse_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load measurements", "", "CSV (*.csv)"
        )
        if path:
            self.reload_path_edit.setText(path)

    def _on_reload_clicked(self) -> None:
        """Load a CSV written by a *previous* run (this or another machine).

        Bypasses the disk-manifest cache entirely — no key matching, no
        assumption about paths or cwd being the same as when it was
        written. Point it at any ``*_measurements.csv`` and it repopulates
        the table and (if a matching Labels layer is selected) the
        image-click/row-click wiring, without recomputing anything.
        """
        path = self.reload_path_edit.text().strip()
        if not path:
            return
        try:
            table = pd.read_csv(path, index_col="label")
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            return

        self._table = table
        self._populate_table(table)
        self._refresh_colormap_columns(table)
        self.save_btn.setEnabled(not table.empty)
        self.status_label.setText(f"{len(table)} objects loaded from {path}.")

        _, labels_layer = self._selected_layers()
        self._wire_labels_features(labels_layer, table)


def _restore(combo: QComboBox, previous: str) -> None:
    idx = combo.findText(previous)
    if idx >= 0:
        combo.setCurrentIndex(idx)
