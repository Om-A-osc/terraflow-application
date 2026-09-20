#!/usr/bin/env python
"""
Clip the GCN250 global curve-number raster to India.

    python scripts/prepare_gcn250.py

GCN250 (Jaafar et al. 2019, CC BY 4.0) gives a curve number for average
antecedent conditions everywhere on land at 250 m.  With it the runoff
calculation reads a measured curve number for the catchment; without it the
service falls back to the OpenStreetMap land-use classes and the CGWB table,
which is coarser but works.

The source is a 640 MB tiled BigTIFF on Figshare.  Figshare honours HTTP range
requests, so rasterio reads only the window over India (about 60 MB) rather
than downloading the whole thing.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

# ARC II (average antecedent conditions) from the Figshare record
GCN250_ARCII_URL = "https://ndownloader.figshare.com/files/15377363"

# Generous bounds around India, including the islands
INDIA_BOUNDS = (67.0, 6.0, 98.5, 37.5)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(config.GCN250_TIF))
    parser.add_argument("--url", default=GCN250_ARCII_URL)
    parser.add_argument("--bounds", nargs=4, type=float, default=list(INDIA_BOUNDS),
                        metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    args = parser.parse_args()

    try:
        import rasterio
        from rasterio.windows import from_bounds
    except ImportError:
        print("rasterio is required: pip install 'rasterio<1.5'")
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    west, south, east, north = args.bounds
    url = f"/vsicurl/{args.url}"
    print("Clipping GCN250 (ARC II) to India")
    print(f"  source: {args.url}")
    print(f"  bounds: {west}, {south} to {east}, {north}")

    started = time.time()
    try:
        with rasterio.open(url) as src:
            print(f"  full raster: {src.width} x {src.height}, {src.dtypes[0]}, crs {src.crs}")
            window = from_bounds(west, south, east, north, src.transform)
            window = window.round_offsets().round_lengths()
            print(f"  reading window {int(window.width)} x {int(window.height)}…")

            data = src.read(1, window=window)
            profile = src.profile.copy()
            profile.update(
                driver="GTiff",
                height=int(window.height),
                width=int(window.width),
                transform=src.window_transform(window),
                compress="deflate",
                tiled=True,
                blockxsize=512,
                blockysize=512,
                BIGTIFF="IF_SAFER",
            )
    except Exception as exc:  # noqa: BLE001
        print(f"\nCould not read the source: {exc}")
        print("The service will use the OpenStreetMap land-use table instead.")
        return 1

    with rasterio.open(out, "w", **profile) as dst:
        dst.write(data, 1)

    valid = data[(data >= 1) & (data <= 100)]
    size_mb = out.stat().st_size / 1e6
    print(f"\nDone in {time.time() - started:.0f}s: {size_mb:.0f} MB at {out}")
    if valid.size:
        print(f"  curve numbers {valid.min()} to {valid.max()}, mean {valid.mean():.1f}")
    print("Restart the API; analyses will report cn_source = gcn250.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
