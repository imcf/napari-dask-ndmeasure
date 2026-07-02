# napari-dask-ndmeasure

[![PyPI](https://img.shields.io/pypi/v/napari-dask-ndmeasure.svg)](https://pypi.org/project/napari-dask-ndmeasure/)
[![Python versions](https://img.shields.io/pypi/pyversions/napari-dask-ndmeasure.svg)](https://pypi.org/project/napari-dask-ndmeasure/)
[![License: GPLv3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Out-of-core, chunk-parallel regionprops-style measurements for **huge**
labelled images inside napari.

`skimage.measure.regionprops` needs the full labelled + intensity array in
RAM — fine for one tile, not for a hundred-thousand-object OME-ZARR volume.
This plugin measures a `Labels` layer directly against an `Image` layer via
[`dask-image`](https://image.dask.org)'s `ndmeasure`, working straight off
the layers' backing dask/zarr arrays, chunk by chunk, without ever
materializing the whole volume.

## Install

```bash
pip install napari-dask-ndmeasure
```

## Use

1. Open your image + labels in napari (e.g. via
   [patchworks](https://github.com/imcf/patchworks)'s
   `view_in_napari(...)`, which also sets each layer's physical `scale`, so
   measurements come out in µm — not just pixels).
2. `Plugins → Dask ndmeasure → Measure (dask-image)`.
3. Pick the Image and Labels layer, check the measurements you want, hit
   **Measure**. The UI stays responsive throughout (runs in a background
   thread) with a progress bar; every measurement is also written to CSV
   automatically (no dialog) — see *Output* below.
4. Browse the table in the dock widget, or **Save CSV…** to pick a different
   location (which then becomes the target for future automatic saves too).

The measured table is also written to the Labels layer's `.features`, so you
can immediately color the layer by any measured value via napari's built-in
"color by feature" for Labels layers.

Re-clicking **Measure** with the same layers/level/stats is instant — results
are cached for the session (in memory; cleared when the widget/viewer
closes).

### Output

Every successful measurement is written to CSV automatically, no dialog:
`<save folder>/<labels layer name>_measurements.csv`, where the save folder
is the last one you picked via **Save CSV…**, or the current working
directory if you never have.

### Keeping RAM and time bounded

Two things matter for a huge volume, both handled automatically:

- **Chunking.** Every array is rechunked to a bounded size before measuring
  *only if it needs it* — a plain numpy-backed layer (or any dask array that
  isn't already sensibly chunked) would otherwise become **one giant
  chunk**, silently forcing the whole volume into RAM. A layer that's
  already a well-chunked OME-ZARR pyramid is left untouched — rechunking an
  already-fine array is itself expensive (real data movement), not free
  insurance.
- **One combined pass.** All requested stats are computed together in a
  single `dask.compute()` call, not one at a time. Each `dask-image`
  measurement function builds its own graph over the same image/labels
  chunks; computing them separately means every chunk gets read and decoded
  once *per stat* (4 default stats = 4 full passes over the data). Computing
  them together lets dask's scheduler share the chunk-read work across all
  of them. The trade-off: progress reporting is two coarse phases (scanning,
  then computing) rather than one tick per stat — the fine-grained version
  needed the separate, slower calls to work.

The **Workers** spin box controls how many threads that combined computation
uses (default `min(4, cpu count)`). More workers can mean more decoded
chunks held in memory at once — turn it down (even to 1) if a measurement is
still using too much RAM; turn it up if you have RAM to spare and want it
faster.

## Measurements

| Stat | Needs intensity image | Notes |
|---|---|---|
| `area` | no | voxel count (+ `area_um2`/`area_um3` if the layer is calibrated) |
| `centroid` | no | geometric centroid, one column per axis (+ `_um` twins) |
| `weighted_centroid` | yes | intensity-weighted centroid |
| `mean_intensity` | yes | |
| `std_intensity` | yes | |
| `min_intensity` | yes | |
| `max_intensity` | yes | |

## Multiscale / pyramid layers

A "Pyramid level" spin box lets you measure a coarser level first — much
faster, handy to sanity-check the setup before running the full-resolution
pass on a very large volume.

## Development

```bash
git clone https://github.com/imcf/napari-dask-ndmeasure
cd napari-dask-ndmeasure
pip install -e ".[dev]"
pytest
```

## License

GNU General Public License v3.0 (GPL-3.0). See [LICENSE](LICENSE).
