import dask.array as da
import numpy as np

from napari_dask_ndmeasure._widget import MeasureWidget, _sequential_ids_hint


def _add_layers(viewer):
    img = np.arange(16, dtype="float32").reshape(4, 4)
    lab = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 2, 2],
            [0, 0, 2, 2],
        ],
        dtype="int32",
    )
    viewer.add_image(img, name="image", scale=(0.5, 0.2))
    viewer.add_labels(lab, name="labels", scale=(0.5, 0.2))


def test_widget_populates_layer_choices(make_napari_viewer):
    viewer = make_napari_viewer()
    _add_layers(viewer)
    widget = MeasureWidget(viewer)

    assert [
        widget.image_combo.itemText(i)
        for i in range(widget.image_combo.count())
    ] == ["image"]
    assert [
        widget.labels_combo.itemText(i)
        for i in range(widget.labels_combo.count())
    ] == ["labels"]


def test_widget_measure_end_to_end(qtbot, make_napari_viewer, tmp_path):
    viewer = make_napari_viewer()
    _add_layers(viewer)
    widget = MeasureWidget(viewer)
    widget._save_dir = tmp_path  # keep the auto-save CSV out of the repo

    widget._on_measure_clicked()
    qtbot.waitUntil(lambda: widget._table is not None, timeout=5000)

    assert widget._table is not None
    assert list(widget._table.index) == [1, 2]
    assert widget.results_table.rowCount() == 2
    assert widget.save_btn.isEnabled()
    # progress bar was indeterminate throughout (stats are now computed
    # together, not one at a time) and got hidden again once done
    assert not widget.progress_bar.isVisible()
    assert "(cached)" not in widget.status_label.text()

    labels_layer = viewer.layers["labels"]
    assert "area" in labels_layer.features.columns


def test_widget_measure_auto_saves_csv(qtbot, make_napari_viewer, tmp_path):
    viewer = make_napari_viewer()
    _add_layers(viewer)
    widget = MeasureWidget(viewer)
    widget._save_dir = tmp_path

    widget._on_measure_clicked()
    qtbot.waitUntil(lambda: widget._table is not None, timeout=5000)

    saved = tmp_path / "labels_measurements.csv"
    assert saved.exists()
    assert "area" in saved.read_text()
    assert f"Saved to {saved}" in widget.status_label.text()


def test_widget_measure_second_run_is_a_cache_hit(
    qtbot, make_napari_viewer, monkeypatch, tmp_path
):
    viewer = make_napari_viewer()
    _add_layers(viewer)
    widget = MeasureWidget(viewer)
    widget._save_dir = tmp_path

    widget._on_measure_clicked()
    qtbot.waitUntil(lambda: widget._table is not None, timeout=5000)
    first_table = widget._table
    assert len(widget._cache) == 1

    # Break the underlying computation so a second *real* run would error —
    # a cache hit must not touch it at all. monkeypatch auto-reverts this
    # after the test, unlike a raw module-attribute assignment (an earlier
    # version of this test left iter_measure_labels broken for every test
    # that ran afterward, which hung them instead of failing cleanly).
    def _boom(*a, **k):
        raise AssertionError("should not recompute on a cache hit")

    monkeypatch.setattr(
        "napari_dask_ndmeasure._widget.iter_measure_labels", _boom
    )

    widget._table = None
    widget._on_measure_clicked()  # same layers/stats/level -> cache hit, synchronous

    assert widget._table is not None
    assert widget._table.equals(first_table)
    assert "(cached)" in widget.status_label.text()
    assert len(widget._cache) == 1  # unchanged, no new entry


def test_widget_default_csv_name_uses_labels_layer(make_napari_viewer):
    # Deliberately doesn't touch QFileDialog at all — mocking a Qt static
    # dialog method is fragile (ponytail: an earlier version of this test
    # hung forever because the mock silently didn't take effect and the
    # real, un-clickable headless dialog opened instead). _default_csv_name
    # is pure/dialog-free by design specifically so this is safely testable.
    viewer = make_napari_viewer()
    _add_layers(viewer)
    widget = MeasureWidget(viewer)

    assert widget._default_csv_name() == "labels_measurements.csv"


def test_widget_save_csv_writes_table_and_remembers_dir(
    qtbot, make_napari_viewer, monkeypatch, tmp_path
):
    viewer = make_napari_viewer()
    _add_layers(viewer)
    widget = MeasureWidget(viewer)
    widget._save_dir = tmp_path  # keep the auto-save out of the repo

    widget._on_measure_clicked()
    qtbot.waitUntil(lambda: widget._table is not None, timeout=5000)

    target = tmp_path / "manual" / "out.csv"
    target.parent.mkdir()
    monkeypatch.setattr(
        "napari_dask_ndmeasure._widget.QFileDialog.getSaveFileName",
        staticmethod(lambda *a, **k: (str(target), "CSV (*.csv)")),
    )

    widget._on_save_clicked()

    assert target.exists()
    assert "area" in target.read_text()
    assert widget._save_dir == target.parent


def test_sequential_ids_hint_present_at_level_zero(make_napari_viewer):
    viewer = make_napari_viewer()
    _add_layers(viewer)
    labels_layer = viewer.layers["labels"]
    labels_layer.metadata["sequential_labels"] = True
    labels_layer.metadata["n_objects"] = 2

    ids = _sequential_ids_hint(labels_layer, level=0)
    assert np.array_equal(ids, np.array([1, 2]))


def test_sequential_ids_hint_ignored_at_higher_level(make_napari_viewer):
    # A coarser pyramid level's strided downsampling can drop small objects
    # entirely, so the exact id set from level 0 doesn't necessarily hold
    # at level > 0 — the hint must not be trusted there.
    viewer = make_napari_viewer()
    _add_layers(viewer)
    labels_layer = viewer.layers["labels"]
    labels_layer.metadata["sequential_labels"] = True
    labels_layer.metadata["n_objects"] = 2

    assert _sequential_ids_hint(labels_layer, level=1) is None


def test_sequential_ids_hint_absent_without_metadata(make_napari_viewer):
    viewer = make_napari_viewer()
    _add_layers(viewer)
    labels_layer = viewer.layers["labels"]

    assert _sequential_ids_hint(labels_layer, level=0) is None


def test_widget_measure_uses_metadata_hint_to_skip_scan(
    qtbot, make_napari_viewer, monkeypatch, tmp_path
):
    viewer = make_napari_viewer()
    _add_layers(viewer)
    labels_layer = viewer.layers["labels"]
    labels_layer.metadata["sequential_labels"] = True
    labels_layer.metadata["n_objects"] = 2

    widget = MeasureWidget(viewer)
    widget._save_dir = tmp_path

    def _boom(*a, **k):
        raise AssertionError("da.unique should not run when the hint applies")

    monkeypatch.setattr("napari_dask_ndmeasure._measure.da.unique", _boom)

    widget._on_measure_clicked()
    # the "using known object count" status is set synchronously, before the
    # (threaded) measurement starts — check it right away, since it's
    # overwritten by the final "N objects measured" status once done, which
    # for this tiny synthetic array can happen almost instantly.
    assert "using known object count (2)" in widget.status_label.text()

    qtbot.waitUntil(lambda: widget._table is not None, timeout=5000)
    assert list(widget._table.index) == [1, 2]
    assert widget._table.loc[1, "area"] == 4


def test_widget_level_range_updates_for_multiscale(make_napari_viewer):
    viewer = make_napari_viewer()
    lab0 = da.zeros((8, 8), dtype="int32", chunks=(4, 4))
    lab1 = da.zeros((4, 4), dtype="int32", chunks=(4, 4))
    viewer.add_labels([lab0, lab1], name="pyramid_labels", multiscale=True)
    viewer.add_image(np.zeros((4, 4), dtype="float32"), name="image")

    widget = MeasureWidget(viewer)
    widget.labels_combo.setCurrentText("pyramid_labels")
    assert widget.level_spin.maximum() == 1
