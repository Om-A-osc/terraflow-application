#!/usr/bin/env python
"""
Download the IMD 0.25 degree gridded daily rainfall archive.

    python scripts/download_imd.py --start 1991 --end 2025

Rainfall is the one input the analysis cannot fall back on cheaply.  Without
this archive the service uses NASA POWER, which is reanalysis: it smooths daily
intensity and so biases curve-number runoff low.  The IMD grid is
gauge-interpolated and is what Indian practice actually designs against.

Each year is a 25 MB classic-NetCDF file (about 50 seconds on a decent link).
They are stacked into one int16 memmap of tenths of a millimetre, which is
about 0.43 GB for 35 years and is read in a microsecond at analysis time.

The file is a plain HTTP POST with no key and no registration.  Note that the
page's own text lags the data: it says 1901 to 2024 while 2025 downloads fine.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402


def fetch_year(year: int, retries: int = 2) -> np.ndarray | None:
    """Download one year and return a (days, lat, lon) float32 array."""
    from scipy.io import netcdf_file

    for attempt in range(retries + 1):
        try:
            print(f"  {year}: downloading…", end="", flush=True)
            started = time.time()
            response = requests.post(
                config.IMD_RF25_POST,
                data={"RF25": str(year)},
                timeout=(10, 600),
                headers={"User-Agent": config.HTTP_USER_AGENT},
            )
            response.raise_for_status()
            payload = response.content
            if len(payload) < 1_000_000:
                print(f" only {len(payload)} bytes, skipping")
                return None

            with netcdf_file(io.BytesIO(payload), "r", mmap=False) as handle:
                variable = None
                for name in ("RAINFALL", "rf", "rain", "RF"):
                    if name in handle.variables:
                        variable = handle.variables[name]
                        break
                if variable is None:
                    print(f" unexpected variables {list(handle.variables)}")
                    return None
                data = np.array(variable.data, dtype=np.float32)

            print(f" {len(payload) / 1e6:.1f} MB, {data.shape} in {time.time() - started:.0f}s")
            return data
        except Exception as exc:  # noqa: BLE001
            print(f" failed ({exc})")
            if attempt < retries:
                time.sleep(5)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=1991)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--out-dir", default=str(config.RAINFALL_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    years = list(range(args.start, args.end + 1))
    print(f"IMD 0.25 degree daily rainfall, {years[0]} to {years[-1]}")
    print(f"About {len(years) * 25} MB to download, {len(years) * 0.0125:.1f} GB stored\n")

    chunks, kept = [], []
    for year in years:
        data = fetch_year(year)
        if data is None:
            continue
        # Files are (time, lat, lon); a few years ship (time, lon, lat)
        if data.shape[1] == config.IMD_NLON and data.shape[2] == config.IMD_NLAT:
            data = np.transpose(data, (0, 2, 1))
        if data.shape[1:] != (config.IMD_NLAT, config.IMD_NLON):
            print(f"  {year}: unexpected grid {data.shape[1:]}, skipping")
            continue
        chunks.append(data)
        kept.append(year)

    if not chunks:
        print("\nNothing downloaded. The service will use NASA POWER instead.")
        return 1

    stacked = np.concatenate(chunks, axis=0)
    del chunks

    # Tenths of a millimetre in int16 halves the file against float32 and still
    # resolves far finer than the data's own accuracy.  -999 stays the fill.
    scaled = np.where(stacked < -900, -999, np.round(stacked * 10.0))
    scaled = np.clip(scaled, -999, 32767).astype(np.int16)

    memmap_path = out_dir / config.RAINFALL_MEMMAP.name
    memmap = np.memmap(memmap_path, dtype="int16", mode="w+", shape=scaled.shape)
    memmap[:] = scaled
    memmap.flush()
    del memmap

    meta = {
        "shape": list(scaled.shape),
        "dtype": "int16",
        "scale": 10.0,
        "start_date": f"{kept[0]}-01-01",
        "years": kept,
        "lat0": config.IMD_LAT0,
        "lon0": config.IMD_LON0,
        "step": config.IMD_STEP,
        "source": "India Meteorological Department, 0.25 degree gridded daily rainfall",
        "url": config.IMD_RF25_POST,
    }
    (out_dir / config.RAINFALL_META.name).write_text(json.dumps(meta, indent=2))

    size_gb = memmap_path.stat().st_size / 1e9
    print(f"\nDone: {scaled.shape[0]:,} days, {len(kept)} years, {size_gb:.2f} GB")
    print(f"  {memmap_path}")
    print("Restart the API to pick it up; /health will show local_rainfall true.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
