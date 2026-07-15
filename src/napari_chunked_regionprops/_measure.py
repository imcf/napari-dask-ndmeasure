"""Out-of-core, chunk-parallel regionprops-style measurements.

Measures a labelled OME-ZARR volume with hundreds of thousands of objects
directly from disk/dask — unlike ``skimage.measure.regionprops``, which
needs the full labelled + intensity array in RAM. Each chunk is aggregated
locally (one ``scipy.ndimage`` call per chunk, over every id actually
present in it), then merged across chunks in a single numpy pass — task
count scales with chunk count, not with object count (see the Notes in
:func:`iter_measure_labels` for why that distinction matters).
"""

from __future__ import annotations

import math
import os
import threading
from typing import Any, Generator, Sequence

import dask
import dask.array as da
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from dask import delayed
from dask.diagnostics import Callback

#: stat name -> whether it needs the real intensity image (False = purely
#: geometric, derived from label positions alone).
_STATS: dict[str, bool] = {
    "area_voxels": False,
    "centroid": False,
    "mean_intensity": True,
    "std_intensity": True,
    "min_intensity": True,
    "max_intensity": True,
    "weighted_centroid": True,
}

DEFAULT_STATS = ("area_voxels", "centroid", "mean_intensity", "std_intensity")

# Target ~128 MB dask chunks (dask's own default) when an input isn't already
# chunked sensibly. Without this, a plain numpy array (or a dask array
# wrapped without chunks=) becomes ONE giant chunk, and the per-chunk
# aggregation in _chunk_partial then has to hold the whole thing in RAM at
# once — silently defeating the entire point of this plugin on exactly the
# huge images it's meant for.
_CHUNK_TARGET = "auto"
# Only rechunk when a chunk is pathologically large (the single-giant-chunk
# case above). A layer that's already a well-chunked OME-ZARR pyramid (the
# common case via patchworks' view_in_napari) must be left alone —
# unconditionally rechunking to "auto" was itself the cause of a RAM/perf
# regression: it forces a full rechunk pass (real data movement, extra graph
# complexity) even when the existing chunking was already fine.
_MAX_CHUNK_BYTES = 512 * 1024**2  # 512 MiB


def _needs_rechunk(arr: da.Array) -> bool:
    """Whether any chunk in *arr* exceeds a sane in-memory ceiling.

    Parameters
    ----------
    arr : da.Array
        Array to check.

    Returns
    -------
    bool
        True if *arr*'s largest chunk exceeds :data:`_MAX_CHUNK_BYTES`.
    """
    return math.prod(arr.chunksize) * arr.dtype.itemsize > _MAX_CHUNK_BYTES


class _ProgressState:
    """Mutable ``(done, total)`` box, written in place from dask worker threads.

    Plain attribute assignment rather than a queue — CPython's GIL makes a
    single attribute write atomic enough for a poller on another thread to
    always see *some* consistent, if possibly one-write-stale, pair. See
    :func:`_compute_with_progress` for why this (state + poll) shape is
    used instead of pushing an event per task.
    """

    __slots__ = ("done", "total")

    def __init__(self):
        self.done = 0
        self.total = 0


class _StateProgress(Callback):
    """Dask diagnostics callback that writes live task counts into a state box.

    Mirrors ``dask.diagnostics.progress.ProgressBar``'s own bookkeeping
    (``len(state["finished"])`` vs. every task across ``ready``/``waiting``/
    ``running``/``finished``).
    """

    def __init__(self, state: "_ProgressState"):
        self._state = state

    def _start_state(self, dsk, state):
        self._update(state)

    def _posttask(self, key, result, dsk, state, worker_id):
        self._update(state)

    def _update(self, state):
        done = len(state["finished"])
        total = (
            sum(len(state[k]) for k in ("ready", "waiting", "running")) + done
        )
        self._state.done = done
        self._state.total = total


def _compute_with_progress(
    lazy_values: Sequence[Any],
    num_workers: int,
    poll_interval: float = 0.15,
) -> Generator[tuple[int, int], None, tuple]:
    """Run ``dask.compute(*lazy_values)`` in a background thread, yielding progress.

    Parameters
    ----------
    lazy_values : sequence
        Dask collections to compute together (shares work across them the
        same way a single :func:`dask.compute` call always does).
    num_workers : int
        Thread count for the computation.
    poll_interval : float, optional
        Seconds between progress yields (default 0.15). Progress is sampled
        on a timer rather than pushed once per finished task — a graph with
        tens of thousands of fine-grained tasks (e.g. a very high object
        count) would otherwise fire a cross-thread Qt signal per task
        faster than a GUI thread can repaint, so the displayed progress
        falls further and further behind the real state as the backlog
        grows, eventually looking frozen at a stale value even though the
        computation itself finished. Sampling at a fixed, small rate keeps
        the number of yields bounded by wall-clock time instead of by task
        count.

    Yields
    ------
    tuple of (int, int)
        ``(done, total)`` dask task counts, sampled at *poll_interval*
        (plus a final yield for whatever the state was the instant the
        computation finished, even if that's less than *poll_interval*
        since the last one). ``total`` can grow between yields early on, as
        dask discovers more of the graph — treat it as "at least this
        many", not a fixed denominator from the first yield.

    Returns
    -------
    tuple
        The computed results, in the same order as *lazy_values* — exactly
        what ``dask.compute(*lazy_values)`` would have returned directly.
    """
    state = _ProgressState()
    result_box: dict[str, tuple] = {}
    error_box: dict[str, BaseException] = {}
    done_event = threading.Event()

    def _run():
        try:
            with _StateProgress(state):
                result_box["value"] = dask.compute(
                    *lazy_values, num_workers=num_workers
                )
        except BaseException as exc:  # re-raised on the caller's thread below
            error_box["error"] = exc
        finally:
            done_event.set()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    last = None
    while not done_event.is_set():
        current = (state.done, state.total)
        if current != last:
            yield current
            last = current
        done_event.wait(poll_interval)
    thread.join()
    final = (state.done, state.total)
    if final != last:
        yield final
    if "error" in error_box:
        raise error_box["error"]
    return result_box["value"]


def available_stats() -> tuple[str, ...]:
    """Return the stat names accepted by ``stats=`` on measurement functions.

    Returns
    -------
    tuple of str
        Every valid entry for :func:`measure_labels`'s and
        :func:`iter_measure_labels`'s ``stats`` argument.

    Examples
    --------
    >>> available_stats()
    ('area_voxels', 'centroid', 'mean_intensity', 'std_intensity', 'min_intensity', 'max_intensity', 'weighted_centroid')
    """
    return tuple(_STATS)


def _ensure_chunked(image: Any, labels: Any) -> tuple[da.Array, da.Array]:
    """Coerce *image*/*labels* to dask arrays with matching, bounded chunks.

    Handles plain numpy arrays and dask arrays that happen to be a single
    giant chunk (both common when a napari layer isn't a chunked OME-ZARR
    pyramid) by rechunking to ``_CHUNK_TARGET``. Arrays that are already
    reasonably chunked (e.g. a real OME-ZARR pyramid) are left untouched —
    rechunking an already-fine array is itself expensive (real data
    movement, a denser task graph), not free insurance.

    Parameters
    ----------
    image : array-like
        Intensity image — a numpy array, or a dask array with any chunking
        (including a single chunk covering the whole array).
    labels : array-like
        Integer label array, same shape as *image*.

    Returns
    -------
    tuple of (da.Array, da.Array)
        ``(image, labels)`` as dask arrays sharing an identical chunk grid.
        Only rechunked (to ``_CHUNK_TARGET``, or to match each other) when
        :func:`_needs_rechunk` says the input actually needs it.

    Examples
    --------
    >>> import numpy as np
    >>> image, labels = _ensure_chunked(
    ...     np.zeros((4, 4), dtype="float32"), np.zeros((4, 4), dtype="int32")
    ... )
    >>> image.chunks == labels.chunks
    True
    """
    labels = da.asarray(labels)
    image = da.asarray(image)
    if _needs_rechunk(labels):
        labels = labels.rechunk(_CHUNK_TARGET)
    if _needs_rechunk(image):
        image = image.rechunk(_CHUNK_TARGET)
    if image.chunks != labels.chunks:
        image = image.rechunk(labels.chunks)
    return image, labels


#: stat name -> the raw per-chunk/merge quantities it needs (see
#: :func:`_chunk_partial`/:func:`_merge_partials`). Union these across every
#: requested stat to know what a given call actually has to compute.
_STAT_WANTS: dict[str, tuple[str, ...]] = {
    "area_voxels": ("count",),
    "centroid": ("count", "pos_sum"),
    "mean_intensity": ("count", "sum"),
    "std_intensity": ("count", "sum", "sumsq"),
    "min_intensity": ("min",),
    "max_intensity": ("max",),
    "weighted_centroid": ("sum", "wpos_sum"),
}


def _chunk_partial(
    label_chunk: np.ndarray,
    image_chunk: "np.ndarray | None",
    offset: tuple[int, ...],
    want: frozenset[str],
) -> "dict[str, np.ndarray] | None":
    """Aggregate one chunk's voxels into small per-local-id partial sums.

    The fix for the per-object-id task blowup an earlier version of this
    module had: every ``scipy.ndimage`` call here runs **once per chunk**,
    over whichever ids are actually present in *that* chunk (typically a
    small fraction of the total object count) — scipy already vectorizes
    internally over an ``index=`` array, so there was never a need to call
    it once per id.

    Parameters
    ----------
    label_chunk : np.ndarray
        One chunk's worth of labels.
    image_chunk : np.ndarray or None
        The matching intensity chunk, or ``None`` if nothing in *want*
        needs real intensity data (skips reading/passing it entirely).
    offset : tuple of int
        This chunk's start position along each axis in the full array —
        needed so position sums (for centroids) are in global, not
        chunk-local, coordinates.
    want : frozenset of str
        Which raw quantities to compute: any of ``"count"``, ``"sum"``,
        ``"sumsq"``, ``"min"``, ``"max"``, ``"pos_sum"``, ``"wpos_sum"``.

    Returns
    -------
    dict of str to np.ndarray, or None
        ``None`` if the chunk has no non-background labels. Otherwise a
        dict with an ``"ids"`` array (the chunk's local non-background ids,
        ascending) plus one array per requested quantity in *want*, each
        aligned index-for-index with ``"ids"``.
    """
    local_ids = np.unique(label_chunk)
    local_ids = local_ids[local_ids > 0]
    if local_ids.size == 0:
        return None

    partial: dict[str, np.ndarray] = {"ids": local_ids}
    if "count" in want:
        ones = np.ones(label_chunk.shape, dtype="float64")
        partial["count"] = np.atleast_1d(
            ndi.sum_labels(ones, label_chunk, local_ids)
        )
    if "sum" in want:
        partial["sum"] = np.atleast_1d(
            ndi.sum_labels(image_chunk, label_chunk, local_ids)
        )
    if "sumsq" in want:
        partial["sumsq"] = np.atleast_1d(
            ndi.sum_labels(
                image_chunk.astype("float64") ** 2, label_chunk, local_ids
            )
        )
    if "min" in want:
        partial["min"] = np.atleast_1d(
            ndi.minimum(image_chunk, label_chunk, local_ids)
        )
    if "max" in want:
        partial["max"] = np.atleast_1d(
            ndi.maximum(image_chunk, label_chunk, local_ids)
        )
    if "pos_sum" in want or "wpos_sum" in want:
        grids = np.indices(label_chunk.shape, dtype="float64")
        for axis in range(label_chunk.ndim):
            grids[axis] += offset[axis]
        if "pos_sum" in want:
            partial["pos_sum"] = np.stack(
                [
                    np.atleast_1d(
                        ndi.sum_labels(grids[axis], label_chunk, local_ids)
                    )
                    for axis in range(label_chunk.ndim)
                ],
                axis=1,
            )
        if "wpos_sum" in want:
            partial["wpos_sum"] = np.stack(
                [
                    np.atleast_1d(
                        ndi.sum_labels(
                            grids[axis] * image_chunk, label_chunk, local_ids
                        )
                    )
                    for axis in range(label_chunk.ndim)
                ],
                axis=1,
            )
    return partial


def _merge_partials(
    partials: "list[dict[str, np.ndarray] | None]",
    ids: np.ndarray,
    stats: Sequence[str],
    ndim: int,
) -> dict[str, np.ndarray]:
    """Combine every chunk's partial sums into the final per-stat columns.

    Runs once, after every :func:`_chunk_partial` call has finished. The
    gathered data here is bounded by how many (chunk, id) pairs exist —
    objects × however many chunks each one spans (usually 1, occasionally a
    few near boundaries) — not by voxel count, so one eager numpy merge is
    fine even for a huge volume.

    Parameters
    ----------
    partials : list of (dict or None)
        One entry per chunk, as returned by :func:`_chunk_partial`.
    ids : np.ndarray
        Every non-background label id the final table must have a row for.
        Any order — this doesn't assume *ids* is sorted.
    stats : sequence of str
        Which output columns to build (see :data:`_STAT_WANTS`).
    ndim : int
        Spatial dimensionality of the labels array (for axis-suffixed
        centroid column names).

    Returns
    -------
    dict of str to np.ndarray
        One entry per output column (e.g. ``"area_voxels"``, ``"centroid_y"``,
        ``"mean_intensity"``), each aligned index-for-index with *ids*.
    """
    want = frozenset(w for s in stats for w in _STAT_WANTS[s])
    n = ids.size
    # Map an arbitrary (not-necessarily-sorted) id to its position in `ids`
    # via one sort, rather than assuming callers hand us sorted ids.
    order = np.argsort(ids)
    sorted_ids = ids[order]

    count = np.zeros(n) if "count" in want else None
    sum_ = np.zeros(n) if "sum" in want else None
    sumsq = np.zeros(n) if "sumsq" in want else None
    min_ = np.full(n, np.inf) if "min" in want else None
    max_ = np.full(n, -np.inf) if "max" in want else None
    pos_sum = np.zeros((n, ndim)) if "pos_sum" in want else None
    wpos_sum = np.zeros((n, ndim)) if "wpos_sum" in want else None

    for p in partials:
        if p is None:
            continue
        pos = order[np.searchsorted(sorted_ids, p["ids"])]
        if count is not None:
            count[pos] += p["count"]
        if sum_ is not None:
            sum_[pos] += p["sum"]
        if sumsq is not None:
            sumsq[pos] += p["sumsq"]
        if min_ is not None:
            min_[pos] = np.minimum(min_[pos], p["min"])
        if max_ is not None:
            max_[pos] = np.maximum(max_[pos], p["max"])
        if pos_sum is not None:
            pos_sum[pos] += p["pos_sum"]
        if wpos_sum is not None:
            wpos_sum[pos] += p["wpos_sum"]

    axis_names = "zyx"[-ndim:]
    columns: dict[str, np.ndarray] = {}
    for stat in stats:
        if stat == "area_voxels":
            columns["area_voxels"] = count
        elif stat == "mean_intensity":
            columns["mean_intensity"] = sum_ / count
        elif stat == "std_intensity":
            mean = sum_ / count
            columns["std_intensity"] = np.sqrt(sumsq / count - mean**2)
        elif stat == "min_intensity":
            columns["min_intensity"] = min_
        elif stat == "max_intensity":
            columns["max_intensity"] = max_
        elif stat == "centroid":
            for i, ax in enumerate(axis_names):
                columns[f"centroid_{ax}"] = pos_sum[:, i] / count
        elif stat == "weighted_centroid":
            for i, ax in enumerate(axis_names):
                columns[f"weighted_centroid_{ax}"] = wpos_sum[:, i] / sum_
    return columns


def _lazy_measure(
    image: da.Array, labels: da.Array, ids: np.ndarray, stats: Sequence[str]
) -> "dask.delayed.Delayed":
    """Build the lazy chunk-map + single-merge graph for *stats*.

    Parameters
    ----------
    image : da.Array
        Intensity image, chunk-aligned with *labels*.
    labels : da.Array
        Integer label array.
    ids : np.ndarray
        Every non-background label id to produce a row for.
    stats : sequence of str
        Which stats to compute — see :data:`_STAT_WANTS`.

    Returns
    -------
    dask.delayed.Delayed
        Computes to the same ``dict[str, np.ndarray]`` shape
        :func:`_merge_partials` returns. One delayed object, so
        :func:`_compute_with_progress` sees one task graph: a map task per
        chunk plus one final merge task — task count scales with chunk
        count, not object count.
    """
    want = frozenset(w for s in stats for w in _STAT_WANTS[s])
    need_intensity = bool(want & {"sum", "sumsq", "min", "max", "wpos_sum"})

    label_blocks = labels.to_delayed().ravel()
    image_blocks = (
        image.to_delayed().ravel()
        if need_intensity
        else [None] * label_blocks.size
    )
    offsets = [
        tuple(sl.start for sl in slc)
        for slc in da.core.slices_from_chunks(labels.chunks)
    ]

    partials = [
        delayed(_chunk_partial)(lbl, img, offset, want)
        for lbl, img, offset in zip(label_blocks, image_blocks, offsets)
    ]
    return delayed(_merge_partials)(partials, ids, stats, labels.ndim)


def iter_measure_labels(
    image: da.Array,
    labels: da.Array,
    *,
    stats: Sequence[str] = DEFAULT_STATS,
    scale: Sequence[float] | None = None,
    n_workers: int | None = None,
    ids: np.ndarray | None = None,
) -> Generator[tuple[int, int, str], None, pd.DataFrame]:
    """Generator version of :func:`measure_labels` — yields progress.

    Identical contract to :func:`measure_labels`, except it yields progress
    markers and returns (rather than just computes) the final table. This
    is the shape napari's ``napari.qt.threading.thread_worker`` expects for
    a progress-reporting background task: drive it with ``result = yield
    from iter_measure_labels(...)`` inside a ``@thread_worker``-decorated
    generator function, and its ``.yielded``/``.returned`` Qt signals fire
    with exactly these values.

    Parameters
    ----------
    image : da.Array
        Intensity image, same shape as *labels*. Rechunked automatically if
        it isn't already sensibly chunked (see :func:`_ensure_chunked`).
    labels : da.Array
        Integer label array (0 = background).
    stats : sequence of str, optional
        Which measurements to compute — see :func:`available_stats`.
        Default: area_voxels, centroid, mean/std intensity.
    scale : sequence of float or None, optional
        Physical size per axis. See :func:`measure_labels`.
    n_workers : int or None, optional
        Threads used for the combined stat computation. Default
        ``min(4, cpu_count)`` — deliberately modest: more concurrent chunk
        tasks means more decoded chunks held in memory at once, so this
        trades a bit of wall-clock speed for a bounded RAM ceiling. Raise it
        if you have RAM to spare and want it faster; lower it (even to 1)
        if measuring is still using too much memory.
    ids : np.ndarray or None, optional
        The exact set of non-background label ids present in *labels*, if
        already known (e.g. a producer that renumbers labels to a
        contiguous ``1..N`` range, like patchworks with
        ``sequential_labels=True``, can hand you ``np.arange(1, N + 1)``
        directly). When given, the whole-volume scan that would otherwise
        find this (the "scanning for objects" phase) is skipped entirely —
        for a huge volume that scan is itself a full pass over the data, so
        this can save as much time as the measurement itself. Wrong/stale
        ids here silently produce a table for the wrong id set — only pass
        this when you're certain it's still accurate for the exact array
        given.

    Yields
    ------
    tuple of (int, int, str)
        ``(done, total, phase)`` — real dask task counts, updated as
        computation progresses within each phase (``phase`` is
        ``"scanning for objects"``, skipped entirely when *ids* is given,
        or ``"computing N measurement(s)"``). ``total`` can grow between
        yields early in a phase as dask discovers more of the graph, so
        it's "at least this many so far", not a value fixed from the first
        yield of that phase.

    Returns
    -------
    pandas.DataFrame
        The final measurement table — identical to what
        :func:`measure_labels` returns for the same arguments. Retrieve it
        via the generator's ``StopIteration.value``, or transparently via
        ``yield from`` (see above).

    Notes
    -----
    Every requested stat is aggregated together in one chunk-local map +
    single-merge graph (see :func:`_lazy_measure`): each chunk is visited
    once regardless of how many stats or ids are requested (one
    ``scipy.ndimage`` call per stat-quantity per chunk, over only the ids
    actually present there), then every chunk's small partial sums are
    combined in one final numpy pass. Task count scales with chunk count,
    not with object count — an earlier version of this module built one
    task *per requested object id* internally; for a huge object count
    that made the task graph itself the bottleneck, independent of data
    size or I/O speed.

    Examples
    --------
    >>> import dask.array as da
    >>> import numpy as np
    >>> img = da.from_array(np.arange(16).reshape(4, 4).astype("float32"))
    >>> lab = da.from_array(
    ...     np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 2, 2], [0, 0, 2, 2]])
    ... )
    >>> gen = iter_measure_labels(img, lab, stats=("area_voxels", "mean_intensity"))
    >>> progress = []
    >>> try:
    ...     while True:
    ...         progress.append(next(gen))
    ... except StopIteration as stop:
    ...     table = stop.value
    >>> sorted({p[2] for p in progress})
    ['computing 2 measurement(s)', 'scanning for objects']
    >>> int(table.loc[1, "area_voxels"])
    4
    """
    if image.shape != labels.shape:
        raise ValueError(
            f"shape mismatch: image={image.shape} labels={labels.shape}"
        )
    unknown = set(stats) - set(_STATS)
    if unknown:
        raise ValueError(
            f"unknown stats {sorted(unknown)}; choose from {available_stats()}"
        )

    image, labels = _ensure_chunked(image, labels)
    nw = n_workers if n_workers is not None else min(4, os.cpu_count() or 1)

    if ids is not None:
        ids = np.asarray(ids)
    else:
        # da.unique(labels[labels > 0]) looked more direct, but boolean-
        # masked fancy indexing on a dask array produces unknown chunk
        # sizes, which makes the unique()/reduction tree behind it
        # materially more expensive than necessary. da.unique(labels) has a
        # dedicated, well-optimized chunk-wise-unique + tree-reduce
        # implementation; dropping 0 from the (tiny, already-in-RAM) result
        # afterwards is essentially free.
        progress_gen = _compute_with_progress([da.unique(labels)], nw)
        try:
            while True:
                done, total = next(progress_gen)
                yield done, total, "scanning for objects"
        except StopIteration as stop:
            (ids,) = stop.value
        ids = ids[ids > 0]

    if ids.size == 0:
        return pd.DataFrame(index=pd.Index([], name="label"))

    # One lazy chunk-map + single-merge graph for every requested stat
    # together — see the Notes section above for why "together" matters.
    graph = _lazy_measure(image, labels, ids, stats)
    progress_gen = _compute_with_progress([graph], nw)
    phase = f"computing {len(stats)} measurement(s)"
    try:
        while True:
            done, total = next(progress_gen)
            yield done, total, phase
    except StopIteration as stop:
        (columns,) = stop.value

    table = pd.DataFrame(columns, index=pd.Index(ids, name="label"))

    if scale is not None:
        scale_arr = np.asarray(scale, dtype="float64")
        if "area_voxels" in table.columns:
            table["area_um3" if len(scale_arr) == 3 else "area_um2"] = table[
                "area_voxels"
            ] * np.prod(scale_arr)
        for prefix in ("centroid", "weighted_centroid"):
            cols = [c for c in table.columns if c.startswith(f"{prefix}_")]
            for c, s in zip(cols, scale_arr):
                table[f"{c}_um"] = table[c] * s

    return table


def measure_labels(
    image: da.Array,
    labels: da.Array,
    *,
    stats: Sequence[str] = DEFAULT_STATS,
    scale: Sequence[float] | None = None,
    n_workers: int | None = None,
    ids: np.ndarray | None = None,
) -> pd.DataFrame:
    """Measure every non-background object in *labels* against *image*.

    Parameters
    ----------
    image : da.Array
        Intensity image, same shape as *labels*. Rechunked automatically if
        it isn't already sensibly chunked (see :func:`_ensure_chunked`).
    labels : da.Array
        Integer label array (0 = background).
    stats : sequence of str, optional
        Which measurements to compute — see :func:`available_stats`.
        Default: area_voxels, centroid, mean/std intensity.
    scale : sequence of float or None, optional
        Physical size per axis (e.g. from the napari layer's ``scale``).
        When given, ``area_voxels`` gets an ``area_um3``/``area_um2`` twin
        reported in physical units (voxel count × voxel volume), and
        centroid columns get a ``_um`` twin alongside the pixel-coordinate
        one.
    n_workers : int or None, optional
        Threads used for the combined stat computation. Default
        ``min(4, cpu_count)`` — see :func:`iter_measure_labels` for the
        speed/RAM trade-off this controls.
    ids : np.ndarray or None, optional
        The exact set of non-background label ids, if already known — see
        :func:`iter_measure_labels` for when this is safe to pass and what
        it saves.

    Returns
    -------
    pandas.DataFrame
        One row per label, indexed by label id.

    Examples
    --------
    >>> import dask.array as da
    >>> import numpy as np
    >>> img = da.from_array(np.arange(16).reshape(4, 4).astype("float32"))
    >>> lab = da.from_array(
    ...     np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 2, 2], [0, 0, 2, 2]])
    ... )
    >>> table = measure_labels(img, lab, stats=("area_voxels", "mean_intensity"))
    >>> int(table.loc[1, "area_voxels"])
    4
    """
    gen = iter_measure_labels(
        image,
        labels,
        stats=stats,
        scale=scale,
        n_workers=n_workers,
        ids=ids,
    )
    try:
        while True:
            next(gen)
    except StopIteration as stop:
        return stop.value
