"""Out-of-core, chunk-parallel regionprops-style measurements.

Wraps ``dask_image.ndmeasure`` so a labelled OME-ZARR volume with hundreds of
thousands of objects can be measured directly from disk/dask — unlike
``skimage.measure.regionprops``, which needs the full labelled + intensity
array in RAM.
"""

from __future__ import annotations

import math
import os
import queue
import threading
from typing import Any, Generator, Sequence

import dask
import dask.array as da
import numpy as np
import pandas as pd
from dask.diagnostics import Callback

#: stat name -> dask_image.ndmeasure function name, and whether it needs the
#: real intensity image (False = geometric only, computed on a dummy image).
_STATS: dict[str, tuple[str, bool]] = {
    "area": ("area", False),
    "centroid": ("center_of_mass", False),
    "mean_intensity": ("mean", True),
    "std_intensity": ("standard_deviation", True),
    "min_intensity": ("minimum", True),
    "max_intensity": ("maximum", True),
    "weighted_centroid": ("center_of_mass", True),
}

DEFAULT_STATS = ("area", "centroid", "mean_intensity", "std_intensity")

# Target ~128 MB dask chunks (dask's own default) when an input isn't already
# chunked sensibly. Without this, a plain numpy array (or a dask array
# wrapped without chunks=) becomes ONE giant chunk, and dask_image.ndmeasure
# then has to hold the whole thing in RAM during .compute() — silently
# defeating the entire point of this plugin on exactly the huge images it's
# meant for.
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


class _QueueProgress(Callback):
    """Dask diagnostics callback that pushes ``(done, total)`` task counts.

    Mirrors ``dask.diagnostics.progress.ProgressBar``'s own bookkeeping
    (``len(state["finished"])`` vs. every task across ``ready``/``waiting``/
    ``running``/``finished``) but pushes onto a queue instead of drawing a
    terminal bar, so a caller on a different thread can consume real,
    per-task progress from a running ``dask.compute()`` call.
    """

    def __init__(self, q: "queue.Queue[tuple[int, int]]"):
        self._queue = q

    def _start_state(self, dsk, state):
        self._emit(state)

    def _posttask(self, key, result, dsk, state, worker_id):
        self._emit(state)

    def _emit(self, state):
        done = len(state["finished"])
        total = (
            sum(len(state[k]) for k in ("ready", "waiting", "running")) + done
        )
        self._queue.put((done, total))


def _compute_with_progress(
    lazy_values: Sequence[Any], num_workers: int
) -> Generator[tuple[int, int], None, tuple]:
    """Run ``dask.compute(*lazy_values)`` in a background thread, yielding progress.

    Parameters
    ----------
    lazy_values : sequence
        Dask collections to compute together (shares work across them the
        same way a single :func:`dask.compute` call always does).
    num_workers : int
        Thread count for the computation.

    Yields
    ------
    tuple of (int, int)
        ``(done, total)`` dask task counts, pushed by :class:`_QueueProgress`
        every time a task finishes. ``total`` can grow between yields early
        on, as dask discovers more of the graph — treat it as "at least
        this many", not a fixed denominator from the first yield.

    Returns
    -------
    tuple
        The computed results, in the same order as *lazy_values* — exactly
        what ``dask.compute(*lazy_values)`` would have returned directly.
    """
    q: "queue.Queue[tuple[int, int] | None]" = queue.Queue()
    result_box: dict[str, tuple] = {}
    error_box: dict[str, BaseException] = {}

    def _run():
        try:
            with _QueueProgress(q):
                result_box["value"] = dask.compute(
                    *lazy_values, num_workers=num_workers
                )
        except BaseException as exc:  # re-raised on the caller's thread below
            error_box["error"] = exc
        finally:
            q.put(None)  # sentinel: no more progress coming

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    while True:
        item = q.get()
        if item is None:
            break
        yield item
    thread.join()
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
    ('area', 'centroid', 'mean_intensity', 'std_intensity', 'min_intensity', 'max_intensity', 'weighted_centroid')
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
        Default: area, centroid, mean/std intensity.
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
    Every requested stat is built as a lazy dask array first, then computed
    together in one ``dask.compute()`` call. Each
    ``dask_image.ndmeasure`` function builds its own graph rooted at the
    same ``image``/``labels`` chunks; computing them one at a time would
    mean dask re-reads and re-decodes every chunk once *per stat* (4
    default stats -> 4 full passes over the data). Computing them together
    lets the scheduler recognize the shared chunk-read tasks and run each
    one once, regardless of how many stats need it.

    Examples
    --------
    >>> import dask.array as da
    >>> import numpy as np
    >>> img = da.from_array(np.arange(16).reshape(4, 4).astype("float32"))
    >>> lab = da.from_array(
    ...     np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 2, 2], [0, 0, 2, 2]])
    ... )
    >>> gen = iter_measure_labels(img, lab, stats=("area", "mean_intensity"))
    >>> progress = []
    >>> try:
    ...     while True:
    ...         progress.append(next(gen))
    ... except StopIteration as stop:
    ...     table = stop.value
    >>> sorted({p[2] for p in progress})
    ['computing 2 measurement(s)', 'scanning for objects']
    >>> int(table.loc[1, "area"])
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

    import dask_image.ndmeasure as ndm

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

    # Build every stat as a *lazy* dask array first, then compute them all
    # together — see the Notes section above for why.
    dummy = da.ones_like(image, dtype="float32")
    lazy = {
        stat: getattr(ndm, _STATS[stat][0])(
            image if _STATS[stat][1] else dummy, labels, ids
        )
        for stat in stats
    }
    progress_gen = _compute_with_progress(list(lazy.values()), nw)
    phase = f"computing {len(stats)} measurement(s)"
    try:
        while True:
            done, total = next(progress_gen)
            yield done, total, phase
    except StopIteration as stop:
        computed = stop.value

    columns: dict[str, np.ndarray] = {}
    for stat, result in zip(lazy.keys(), computed):
        if result.ndim == 1:
            columns[stat] = result
        else:
            # center_of_mass: one column per spatial axis
            axis_names = "zyx"[-result.shape[1] :]
            for i, ax in enumerate(axis_names):
                columns[f"{stat}_{ax}"] = result[:, i]

    table = pd.DataFrame(columns, index=pd.Index(ids, name="label"))

    if scale is not None:
        scale_arr = np.asarray(scale, dtype="float64")
        if "area" in table.columns:
            table["area_um3" if len(scale_arr) == 3 else "area_um2"] = table[
                "area"
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
        Default: area, centroid, mean/std intensity.
    scale : sequence of float or None, optional
        Physical size per axis (e.g. from the napari layer's ``scale``).
        When given, ``area`` is reported in physical units (voxel count ×
        voxel volume) and centroid columns get a ``_um`` twin alongside the
        pixel-coordinate one.
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
    >>> table = measure_labels(img, lab, stats=("area", "mean_intensity"))
    >>> int(table.loc[1, "area"])
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
