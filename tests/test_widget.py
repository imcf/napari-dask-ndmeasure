import dask.array as da
import numpy as np
from qtpy.QtCore import Qt

from napari_chunked_regionprops._widget import (
    MeasureWidget,
    _sequential_ids_hint,
)


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
        widget.image_list.item(i).text()
        for i in range(widget.image_list.count())
    ] == ["image"]
    # first population defaults every channel to checked
    assert widget.image_list.item(0).checkState() == Qt.Checked
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
        "napari_chunked_regionprops._widget.iter_measure_labels", _boom
    )

    widget._table = None
    widget._on_measure_clicked()  # same layers/stats/level -> cache hit, synchronous

    assert widget._table is not None
    assert widget._table.equals(first_table)
    assert "(session cache)" in widget.status_label.text()
    assert len(widget._cache) == 1  # unchanged, no new entry


def test_widget_disk_cache_survives_new_widget_instance(
    qtbot, make_napari_viewer, monkeypatch, tmp_path
):
    """A fresh widget (e.g. after closing/reopening the dock) reuses the
    manifest-recorded CSV from a prior instance instead of recomputing."""
    viewer = make_napari_viewer()
    _add_layers(viewer)
    widget = MeasureWidget(viewer)
    widget._save_dir = tmp_path

    widget._on_measure_clicked()
    qtbot.waitUntil(lambda: widget._table is not None, timeout=5000)
    first_table = widget._table
    assert (tmp_path / ".regionprops_cache.json").exists()

    fresh_widget = MeasureWidget(viewer)
    fresh_widget._save_dir = tmp_path
    assert fresh_widget._cache == {}  # no in-memory cache in a new instance

    def _boom(*a, **k):
        raise AssertionError("should not recompute on a disk-cache hit")

    monkeypatch.setattr(
        "napari_chunked_regionprops._widget.iter_measure_labels", _boom
    )

    fresh_widget._on_measure_clicked()  # same layers/stats/level -> disk hit

    assert fresh_widget._table is not None
    assert fresh_widget._table.equals(first_table)
    assert "(disk cache)" in fresh_widget.status_label.text()


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
        "napari_chunked_regionprops._widget.QFileDialog.getSaveFileName",
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

    monkeypatch.setattr("napari_chunked_regionprops._measure.da.unique", _boom)

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


def test_widget_row_click_selects_label_in_image(
    qtbot, make_napari_viewer, tmp_path
):
    viewer = make_napari_viewer()
    _add_layers(viewer)
    widget = MeasureWidget(viewer)
    widget._save_dir = tmp_path

    widget._on_measure_clicked()
    qtbot.waitUntil(lambda: widget._table is not None, timeout=5000)

    default_center = tuple(viewer.camera.center)
    default_zoom = viewer.camera.zoom
    default_step = tuple(viewer.dims.current_step)

    widget._on_result_row_clicked(1, 0)  # second row -> label 2

    # camera actually moved/zoomed onto the object, not left untouched --
    # with tens of thousands of objects, just recoloring one is invisible.
    assert tuple(viewer.camera.center) != default_center
    assert viewer.camera.zoom != default_zoom
    # 2D test data has no z axis (ndim == 2) -> current_step untouched.
    assert tuple(viewer.dims.current_step) == default_step


def test_widget_row_click_jumps_z_slice_for_3d_labels(
    qtbot, make_napari_viewer, tmp_path
):
    viewer = make_napari_viewer()
    img = np.arange(2 * 4 * 4, dtype="float32").reshape(2, 4, 4)
    lab = np.zeros((2, 4, 4), dtype="int32")
    lab[1, 2:, 2:] = 1  # label 1 only exists on z-slice 1
    viewer.add_image(img, name="image")
    viewer.add_labels(lab, name="labels")
    widget = MeasureWidget(viewer)
    widget._save_dir = tmp_path

    widget._on_measure_clicked()
    qtbot.waitUntil(lambda: widget._table is not None, timeout=5000)

    viewer.dims.current_step = (0, 0, 0)
    widget._on_result_row_clicked(0, 0)  # only row -> label 1

    assert viewer.dims.current_step[0] == 1


def test_widget_multi_select_highlights_and_clear_restores(
    qtbot, make_napari_viewer, tmp_path
):
    # 3 objects, not 2 -- so a selection covering only some of them has an
    # actual "unselected but known" label (3) to check stays dim. A 2-object
    # dataset can't catch a regression where every *known* label ends up
    # highlighted (only unknown/background would be dim, masking the bug).
    viewer = make_napari_viewer()
    lab = np.array(
        [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 0, 0], [3, 3, 0, 0]],
        dtype="int32",
    )
    viewer.add_image(np.zeros((4, 4), dtype="float32"), name="image")
    viewer.add_labels(lab, name="labels")
    widget = MeasureWidget(viewer)
    widget._save_dir = tmp_path

    widget._on_measure_clicked()
    qtbot.waitUntil(lambda: widget._table is not None, timeout=5000)

    labels_layer = viewer.layers["labels"]
    original_colormap = labels_layer.colormap
    assert not widget.clear_selection_btn.isEnabled()

    widget.results_table.item(0, 0).setSelected(True)  # label 1
    widget.results_table.item(1, 0).setSelected(True)  # label 2

    assert widget.clear_selection_btn.isEnabled()
    assert labels_layer.colormap is not original_colormap
    # check the actual rendered colors, not just the color_dict structure --
    # the original bug was the dict looking right but the render being wrong.
    rendered = labels_layer.colormap.map(lab)
    assert rendered[0, 0][0] == 1.0  # label 1 highlighted (yellow -> R=1)
    assert rendered[0, 2][0] == 1.0  # label 2 highlighted too
    assert tuple(rendered[2, 0]) == (0.3, 0.3, 0.3, 0.15)  # label 3: dim
    assert tuple(rendered[2, 2]) == (0.0, 0.0, 0.0, 0.0)  # background

    widget._on_clear_selection_clicked()

    assert not widget.results_table.selectedItems()
    assert not widget.clear_selection_btn.isEnabled()
    assert labels_layer.colormap is original_colormap


class _FakeClickEvent:
    """Minimal stand-in for napari's mouse-click event object.

    Only the attributes ``_on_image_clicked`` reads; the values are
    irrelevant here since ``get_value`` itself is monkeypatched in the
    test that uses this.
    """

    position = (0, 0)
    view_direction = None
    dims_displayed = (0, 1)


def test_widget_image_click_selects_table_row(
    qtbot, make_napari_viewer, monkeypatch, tmp_path
):
    viewer = make_napari_viewer()
    _add_layers(viewer)
    widget = MeasureWidget(viewer)
    widget._save_dir = tmp_path

    widget._on_measure_clicked()
    qtbot.waitUntil(lambda: widget._table is not None, timeout=5000)

    labels_layer = viewer.layers["labels"]
    monkeypatch.setattr(labels_layer, "get_value", lambda *a, **k: 2)

    widget._on_image_clicked(labels_layer, _FakeClickEvent())

    assert (
        widget.results_table.item(widget.results_table.currentRow(), 0).text()
        == "2"
    )
    # selecting the row (see _on_result_selection_changed) applied the
    # highlight too, same mechanism as a direct table click.
    assert widget.clear_selection_btn.isEnabled()
    assert labels_layer.colormap.color_dict[2][0] == 1.0


def test_widget_reload_csv_from_path(
    qtbot, make_napari_viewer, monkeypatch, tmp_path
):
    """A CSV from a prior run (possibly another machine/OS) loads via the
    path box, without recomputing — the whole point being it works even
    when the disk-cache manifest's key wouldn't match."""
    viewer = make_napari_viewer()
    _add_layers(viewer)
    widget = MeasureWidget(viewer)
    widget._save_dir = tmp_path

    widget._on_measure_clicked()
    qtbot.waitUntil(lambda: widget._table is not None, timeout=5000)
    first_table = widget._table
    saved_path = tmp_path / "labels_measurements.csv"
    assert saved_path.exists()

    fresh_widget = MeasureWidget(viewer)

    def _boom(*a, **k):
        raise AssertionError("reload must not recompute")

    monkeypatch.setattr(
        "napari_chunked_regionprops._widget.iter_measure_labels", _boom
    )

    fresh_widget.reload_path_edit.setText(str(saved_path))
    fresh_widget._on_reload_clicked()

    assert fresh_widget._table is not None
    assert list(fresh_widget._table.index) == list(first_table.index)
    assert "loaded from" in fresh_widget.status_label.text()
    assert "area" in viewer.layers["labels"].features.columns


def test_widget_measures_every_checked_channel(
    qtbot, make_napari_viewer, tmp_path
):
    """With several Image layers checked, intensity stats are measured per
    channel (suffixed by layer name); geometric stats (area/centroid) are
    computed once, not once per channel."""
    viewer = make_napari_viewer()
    lab = np.array(
        [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 2, 2], [0, 0, 2, 2]],
        dtype="int32",
    )
    viewer.add_image(np.full((4, 4), 10, dtype="float32"), name="dapi")
    viewer.add_image(np.full((4, 4), 20, dtype="float32"), name="gfp")
    viewer.add_labels(lab, name="labels")

    widget = MeasureWidget(viewer)
    widget._save_dir = tmp_path
    # first population defaults every channel to checked -- confirm that,
    # since it's what makes "just click Measure" measure both by default.
    assert all(
        widget.image_list.item(i).checkState() == Qt.Checked
        for i in range(widget.image_list.count())
    )

    widget._on_measure_clicked()
    qtbot.waitUntil(lambda: widget._table is not None, timeout=5000)

    table = widget._table
    assert "mean_intensity_dapi" in table.columns
    assert "mean_intensity_gfp" in table.columns
    assert "mean_intensity" not in table.columns  # only unsuffixed if 1 channel
    assert table.loc[1, "mean_intensity_dapi"] == 10
    assert table.loc[1, "mean_intensity_gfp"] == 20
    # geometric stats present exactly once, not per channel
    assert "area" in table.columns
    assert "centroid_y" in table.columns
    assert "area_dapi" not in table.columns


def test_widget_single_checked_channel_keeps_unsuffixed_columns(
    qtbot, make_napari_viewer, tmp_path
):
    """Backward compatibility: with exactly one Image checked (the
    pre-multi-channel default), column names stay unsuffixed."""
    viewer = make_napari_viewer()
    _add_layers(viewer)
    widget = MeasureWidget(viewer)
    widget._save_dir = tmp_path

    widget._on_measure_clicked()
    qtbot.waitUntil(lambda: widget._table is not None, timeout=5000)

    assert "mean_intensity" in widget._table.columns
    assert "mean_intensity_image" not in widget._table.columns
