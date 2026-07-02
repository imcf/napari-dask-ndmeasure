import dask.array as da
import numpy as np
import pytest

from napari_chunked_regionprops._measure import (
    _compute_with_progress,
    _ensure_chunked,
    _lazy_measure,
    _needs_rechunk,
    available_stats,
    iter_measure_labels,
    measure_labels,
)


def _synthetic():
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
    return da.from_array(img, chunks=(2, 2)), da.from_array(lab, chunks=(2, 2))


def test_measure_labels_default_stats():
    img, lab = _synthetic()
    table = measure_labels(img, lab)

    assert list(table.index) == [1, 2]
    assert table.loc[1, "area"] == 4
    assert table.loc[2, "area"] == 4
    # label 1 = pixels 0,1,4,5 -> mean 2.5; label 2 = pixels 10,11,14,15 -> mean 12.5
    assert table.loc[1, "mean_intensity"] == pytest.approx(2.5)
    assert table.loc[2, "mean_intensity"] == pytest.approx(12.5)
    # geometric centroid of label 1 (rows 0-1, cols 0-1) is (0.5, 0.5)
    assert table.loc[1, "centroid_y"] == pytest.approx(0.5)
    assert table.loc[1, "centroid_x"] == pytest.approx(0.5)


def test_measure_labels_only_requested_stats():
    img, lab = _synthetic()
    table = measure_labels(img, lab, stats=("area",))
    assert list(table.columns) == ["area"]


def test_measure_labels_scale_adds_um_columns():
    img, lab = _synthetic()
    table = measure_labels(
        img, lab, stats=("area", "centroid"), scale=(0.5, 0.2)
    )
    assert table.loc[1, "area_um2"] == pytest.approx(4 * 0.5 * 0.2)
    assert table.loc[1, "centroid_y_um"] == pytest.approx(0.5 * 0.5)
    assert table.loc[1, "centroid_x_um"] == pytest.approx(0.5 * 0.2)


def test_measure_labels_shape_mismatch_raises():
    img = da.zeros((4, 4))
    lab = da.zeros((4, 5))
    with pytest.raises(ValueError, match="shape mismatch"):
        measure_labels(img, lab)


def test_measure_labels_unknown_stat_raises():
    img, lab = _synthetic()
    with pytest.raises(ValueError, match="unknown stats"):
        measure_labels(img, lab, stats=("bogus",))


def test_measure_labels_no_objects_returns_empty():
    img = da.zeros((4, 4), dtype="float32")
    lab = da.zeros((4, 4), dtype="int32")
    table = measure_labels(img, lab)
    assert table.empty


def test_available_stats_matches_measure_labels_options():
    img, lab = _synthetic()
    # every advertised stat should actually be usable
    table = measure_labels(img, lab, stats=available_stats())
    assert not table.empty


def test_measure_labels_rechunks_a_pathologically_huge_single_chunk():
    # A bare numpy array, wrapped naively, becomes ONE chunk covering the
    # whole array. For a genuinely huge layer that's the RAM-blowup bug
    # (dask_image.ndmeasure then needs the whole thing in RAM). Use a lazy
    # da.zeros of realistic size instead of a real numpy array -> no RAM used
    # by the test itself, but the single-chunk pathology is real.
    huge_labels = da.zeros(
        (37716, 27277, 115), dtype="int32", chunks=(37716, 27277, 115)
    )
    huge_image = da.zeros(
        (37716, 27277, 115), dtype="float32", chunks=(37716, 27277, 115)
    )
    assert huge_labels.numblocks == (1, 1, 1)  # precondition: single chunk
    assert _needs_rechunk(huge_labels)

    image, labels = _ensure_chunked(huge_image, huge_labels)
    assert labels.numblocks != (1, 1, 1)
    assert image.chunks == labels.chunks


def test_measure_labels_leaves_already_chunked_arrays_alone():
    # Regression test: an earlier version unconditionally rechunked even
    # well-chunked arrays (e.g. a real OME-ZARR pyramid layer), which was
    # itself a RAM/perf regression — real data movement + a denser task
    # graph for no reason. A sanely-chunked array must come back untouched.
    image = da.zeros((16, 1024, 1024), dtype="float32", chunks=(16, 1024, 1024))
    labels = da.zeros((16, 1024, 1024), dtype="int32", chunks=(16, 1024, 1024))
    assert not _needs_rechunk(labels)
    assert not _needs_rechunk(image)

    out_image, out_labels = _ensure_chunked(image, labels)
    assert out_image.chunks == image.chunks
    assert out_labels.chunks == labels.chunks


def test_measure_labels_small_single_chunk_input_still_correct():
    # Small arrays are legitimately fine as a single chunk (well under the
    # rechunk threshold) — measure_labels must still work correctly, it
    # just won't (and shouldn't) trigger a rechunk.
    img = da.asarray(np.arange(400 * 400, dtype="float32").reshape(400, 400))
    lab = da.asarray(np.ones((400, 400), dtype="int32"))
    assert not _needs_rechunk(lab)

    table = measure_labels(img, lab, stats=("area",))
    assert table.loc[1, "area"] == 400 * 400


def _drain(gen):
    """Collect every yielded value from *gen*, plus its final return value."""
    progress = []
    try:
        while True:
            progress.append(next(gen))
    except StopIteration as stop:
        return progress, stop.value


def test_iter_measure_labels_yields_progress_then_returns_table():
    # All requested stats are now computed together in one dask.compute()
    # call (see the Notes in iter_measure_labels' docstring for why), so
    # there are two phases: "scanning for objects" then "computing N
    # measurement(s)" — each with real, growing (done, total) dask task
    # counts (not a fixed count — exact task counts are a dask-graph
    # implementation detail we shouldn't pin to a magic number).
    img, lab = _synthetic()
    gen = iter_measure_labels(img, lab, stats=("area", "mean_intensity"))
    progress, table = _drain(gen)

    assert progress, "expected at least one progress update"
    phases = [p[2] for p in progress]
    assert set(phases) == {"scanning for objects", "computing 2 measurement(s)"}
    # everything from the scanning phase comes before everything from the
    # computing phase (no interleaving between the two)
    scan_end = max(
        i for i, p in enumerate(phases) if p == "scanning for objects"
    )
    compute_start = min(
        i for i, p in enumerate(phases) if p == "computing 2 measurement(s)"
    )
    assert scan_end < compute_start

    for done, total, _ in progress:
        assert 0 <= done <= total

    assert table.loc[1, "area"] == 4


def test_lazy_measure_graph_scales_with_chunks_not_objects():
    # The whole point of the map/merge rewrite: an earlier version wrapped
    # dask_image.ndmeasure, whose labeled_comprehension builds one task
    # *per requested object id* (times every chunk) — for a high object
    # count that made the task graph itself the bottleneck. The new engine
    # builds one map task per chunk (regardless of how many ids are inside
    # it) plus one merge task, so graph size must track chunk count, not
    # object count, even with thousands of objects.
    n_objects = 2000
    size, chunk = 400, 40  # 10x10 = 100 chunks
    rng = np.random.default_rng(0)
    lab_np = np.zeros((size, size), dtype="int32")
    ys = rng.integers(0, size, n_objects)
    xs = rng.integers(0, size, n_objects)
    lab_np[ys, xs] = np.arange(1, n_objects + 1)

    image = da.zeros((size, size), dtype="float32", chunks=(chunk, chunk))
    labels = da.from_array(lab_np, chunks=(chunk, chunk))
    ids = np.arange(1, n_objects + 1)
    n_chunks = labels.numblocks[0] * labels.numblocks[1]

    graph = _lazy_measure(image, labels, ids, ("area", "mean_intensity"))
    n_tasks = len(graph.__dask_graph__())

    # A per-object-id design would need on the order of n_chunks * n_objects
    # tasks (200,000+ here) just for the label-matching step. A per-chunk
    # design needs a small constant multiple of n_chunks.
    assert n_tasks < 10 * n_chunks
    assert n_tasks < n_objects  # sanity: nowhere near per-object scaling


def test_compute_with_progress_yield_count_bounded_not_per_task():
    # Regression test: an earlier version pushed one progress yield per
    # finished dask task. On a graph with thousands of fine-grained tasks
    # (e.g. measuring 100k+ objects), that floods a cross-thread Qt signal
    # faster than a GUI thread can repaint, so the displayed progress falls
    # further and further behind and can look permanently stuck even after
    # the real computation finished. Progress is now sampled on a timer
    # instead, so the yield count must stay small regardless of how many
    # actual tasks the graph has.
    many_tasks = da.zeros((2000,), chunks=(1,)).sum()
    gen = _compute_with_progress([many_tasks], num_workers=4)
    progress = []
    try:
        while True:
            progress.append(next(gen))
    except StopIteration as stop:
        (result,) = stop.value

    assert result == 0
    final_total = progress[-1][1]
    assert final_total > 100  # the graph genuinely had many tasks
    assert len(progress) < 50  # but yields stayed bounded, not per-task


def test_iter_measure_labels_return_value_matches_measure_labels():
    img, lab = _synthetic()
    _, table_from_iter = _drain(iter_measure_labels(img, lab))
    table_from_wrapper = measure_labels(img, lab)
    assert table_from_iter.equals(table_from_wrapper)


def test_iter_measure_labels_ids_skips_the_scan(monkeypatch):
    # When the caller already knows the id set (e.g. a producer guarantees
    # sequential 1..N labels), the whole-volume scan must not run at all —
    # that scan is itself a full pass over the data, the whole point of
    # accepting ids= is to skip paying for it.
    def _boom(*a, **k):
        raise AssertionError(
            "da.unique should not be called when ids= is given"
        )

    monkeypatch.setattr("napari_chunked_regionprops._measure.da.unique", _boom)

    img, lab = _synthetic()
    gen = iter_measure_labels(img, lab, stats=("area",), ids=np.array([1, 2]))
    progress, table = _drain(gen)

    assert all(p[2] != "scanning for objects" for p in progress)
    assert table.loc[1, "area"] == 4
    assert table.loc[2, "area"] == 4


def test_measure_labels_ids_matches_scanned_result():
    img, lab = _synthetic()
    table_scanned = measure_labels(img, lab, stats=available_stats())
    table_with_ids = measure_labels(
        img, lab, stats=available_stats(), ids=np.array([1, 2])
    )
    assert table_scanned.equals(table_with_ids)


def test_measure_labels_n_workers_does_not_change_result():
    # n_workers only bounds concurrency for the combined compute() call —
    # correctness must not depend on it.
    img, lab = _synthetic()
    table_1 = measure_labels(img, lab, stats=available_stats(), n_workers=1)
    table_4 = measure_labels(img, lab, stats=available_stats(), n_workers=4)
    assert table_1.equals(table_4)
