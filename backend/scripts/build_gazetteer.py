#!/usr/bin/env python
"""
Build the local village gazetteer from the GeoNames India dump.

    python scripts/build_gazetteer.py

Downloads IN.zip (about 16 MB, CC BY 4.0) plus the administrative-code tables,
keeps the populated places, joins the state / district / subdistrict names, and
writes a SQLite database with an FTS5 index.  The result is roughly 150 MB and
answers a prefix search in a millisecond, with no rate limit and no key.

Run it once per machine; the file can also simply be copied between the nodes.
"""

from __future__ import annotations

import argparse
import io
import sqlite3
import sys
import time
import unicodedata
import zipfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

GEONAMES = "https://download.geonames.org/export/dump"
PLACE_CODES = {"PPL", "PPLA", "PPLA2", "PPLA3", "PPLA4", "PPLC", "PPLL", "PPLQ", "PPLS", "PPLX"}


def fetch(url: str) -> bytes:
    print(f"  downloading {url}")
    t0 = time.time()
    resp = requests.get(url, timeout=(10, 300), headers={"User-Agent": config.HTTP_USER_AGENT})
    resp.raise_for_status()
    print(f"    {len(resp.content) / 1e6:.1f} MB in {time.time() - t0:.1f}s")
    return resp.content


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    return "".join(c for c in text if not unicodedata.combining(c)).lower().strip()


def load_admin_names() -> tuple:
    """admin1 (state) and admin2 (district) code -> name."""
    states, districts = {}, {}
    for line in fetch(f"{GEONAMES}/admin1CodesASCII.txt").decode("utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and parts[0].startswith("IN."):
            states[parts[0]] = parts[1]
    for line in fetch(f"{GEONAMES}/admin2Codes.txt").decode("utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and parts[0].startswith("IN."):
            districts[parts[0]] = parts[1]
    print(f"  {len(states)} states, {len(districts)} districts")
    return states, districts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(config.GAZETTEER_DB))
    parser.add_argument("--keep-all", action="store_true",
                        help="keep every feature class, not just populated places")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print("Building the village gazetteer from GeoNames (CC BY 4.0)")
    states, districts = load_admin_names()

    payload = fetch(f"{GEONAMES}/IN.zip")
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        raw = zf.read("IN.txt").decode("utf-8")
    print(f"  IN.txt: {len(raw) / 1e6:.1f} MB")

    # First pass: subdistrict (ADM3) names, so villages can be labelled with them
    subdistricts = {}
    for line in raw.splitlines():
        f = line.split("\t")
        if len(f) < 15:
            continue
        if f[7] == "ADM3":
            subdistricts[f"IN.{f[10]}.{f[11]}.{f[12]}"] = f[1]
    print(f"  {len(subdistricts)} subdistricts")

    if out.exists():
        out.unlink()
    conn = sqlite3.connect(out)
    conn.executescript(
        """
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        CREATE TABLE villages (
            geonameid   INTEGER PRIMARY KEY,
            name        TEXT NOT NULL,
            asciiname   TEXT,
            lat         REAL NOT NULL,
            lon         REAL NOT NULL,
            fcode       TEXT,
            state       TEXT,
            district    TEXT,
            subdistrict TEXT,
            population  INTEGER DEFAULT 0
        );
        CREATE VIRTUAL TABLE village_fts USING fts5(
            name, asciiname, alternatenames, district, subdistrict, state,
            content='', tokenize='unicode61 remove_diacritics 2'
        );
        """
    )

    rows, fts_rows = [], []
    kept = 0
    for line in raw.splitlines():
        f = line.split("\t")
        if len(f) < 15:
            continue
        fclass, fcode = f[6], f[7]
        if not args.keep_all and (fclass != "P" or fcode not in PLACE_CODES):
            continue

        geonameid = int(f[0])
        state = states.get(f"IN.{f[10]}", "")
        district = districts.get(f"IN.{f[10]}.{f[11]}", "")
        subdistrict = subdistricts.get(f"IN.{f[10]}.{f[11]}.{f[12]}", "")
        population = int(f[14] or 0)

        kept += 1
        rows.append((geonameid, f[1], f[2], float(f[4]), float(f[5]), fcode,
                     state, district, subdistrict, population))
        # The FTS rowid must be the geonameid: because `geonameid` is declared
        # INTEGER PRIMARY KEY it *is* villages.rowid, and the join is on rowid.
        fts_rows.append((geonameid, normalise(f[1]), normalise(f[2]),
                         normalise(f[3])[:400], normalise(district),
                         normalise(subdistrict), normalise(state)))

    print(f"  inserting {kept:,} places")
    conn.executemany(
        "INSERT INTO villages VALUES (?,?,?,?,?,?,?,?,?,?)", rows
    )
    conn.executemany(
        "INSERT INTO village_fts(rowid, name, asciiname, alternatenames, district, subdistrict, state) "
        "VALUES (?,?,?,?,?,?,?)",
        fts_rows,
    )
    conn.executescript(
        """
        CREATE INDEX idx_villages_latlon ON villages(lat, lon);
        CREATE INDEX idx_villages_name ON villages(asciiname);
        """
    )
    conn.commit()

    count = conn.execute("SELECT COUNT(*) FROM villages").fetchone()[0]
    sample = conn.execute(
        """SELECT v.name, v.district, v.state FROM village_fts
           JOIN villages v ON v.rowid = village_fts.rowid
           WHERE village_fts MATCH '"jeora"*' LIMIT 3"""
    ).fetchall()
    conn.execute("VACUUM")
    conn.close()

    size_mb = out.stat().st_size / 1e6
    print(f"Done: {count:,} places, {size_mb:.0f} MB at {out}")
    if sample:
        print("  sample match for 'jeora':", sample)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
