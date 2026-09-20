"""
Village search.

The interactive search box must not depend on an external geocoder: Nominatim's
usage policy forbids autocomplete outright, and Photon's public instance
throttles heavy use.  So the primary index is local, a SQLite FTS5 table built
from the GeoNames India dump by ``scripts/build_gazetteer.py`` (about 558,000
populated places, roughly 82 percent of the villages in the Local Government
Directory).  Photon is kept as an online fallback for the long tail.

Village names repeat constantly in India, so every suggestion carries its
tehsil, district and state, and results are biased towards the current map view.
"""

from __future__ import annotations

import logging
import math
import sqlite3
import unicodedata
from functools import lru_cache

import config
from core.cache import bump, cached_call, make_key
from core.resilience import call_source, get

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Local index
# ─────────────────────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def _connection() -> sqlite3.Connection | None:
    if not config.GAZETTEER_DB.exists():
        logger.info(
            "No local gazetteer at %s; village search will use Photon. "
            "Run scripts/build_gazetteer.py to build it.",
            config.GAZETTEER_DB,
        )
        return None
    conn = sqlite3.connect(f"file:{config.GAZETTEER_DB}?mode=ro", uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def has_local_index() -> bool:
    return _connection() is not None


def _normalise(text: str) -> str:
    """Fold accents and case so 'Pipariyā' matches a typed 'pipariya'."""
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c)).lower().strip()


def _fts_query(text: str) -> str:
    """Turn typed text into a prefix query, escaping the FTS operators."""
    tokens = [t for t in _normalise(text).replace('"', " ").split() if t]
    if not tokens:
        return '""'
    return " ".join(f'"{t}"*' for t in tokens)


def _search_local(query: str, lat: float | None, lon: float | None, limit: int) -> list:
    conn = _connection()
    if conn is None:
        return []

    try:
        rows = conn.execute(
            """
            SELECT v.geonameid, v.name, v.asciiname, v.lat, v.lon, v.fcode,
                   v.state, v.district, v.subdistrict, v.population,
                   bm25(village_fts) AS rank
            FROM village_fts
            JOIN villages v ON v.rowid = village_fts.rowid
            WHERE village_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (_fts_query(query), limit * 12),
        ).fetchall()
    except sqlite3.OperationalError as exc:
        logger.warning("Gazetteer query failed: %s", exc)
        return []

    scored = []
    for row in rows:
        # Rank: text match, then size, then distance from where the user is looking
        score = -float(row["rank"] or 0.0)
        score += math.log1p(float(row["population"] or 0)) * 0.35
        if lat is not None and lon is not None:
            d_km = _haversine_km(lat, lon, row["lat"], row["lon"])
            score += 3.0 * math.exp(-d_km / 50.0)
        if _normalise(row["asciiname"] or "") == _normalise(query):
            score += 2.0
        if (row["fcode"] or "").startswith("PPL"):
            score += 0.5
        scored.append((score, row))

    scored.sort(key=lambda pair: -pair[0])
    out = []
    for _, row in scored[:limit]:
        out.append({
            "id": f"gn:{row['geonameid']}",
            "name": row["name"],
            "subdistrict": row["subdistrict"],
            "district": row["district"],
            "state": row["state"],
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "population": int(row["population"] or 0),
            "kind": row["fcode"],
            "source": "geonames",
        })
    if out:
        bump("geocode_local")
    return out


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


# ─────────────────────────────────────────────────────────────────────────────
# Photon fallback
# ─────────────────────────────────────────────────────────────────────────────


def _search_photon(query: str, lat: float | None, lon: float | None, limit: int) -> list:
    params = [
        ("q", query), ("limit", str(limit)), ("lang", "en"),
        # countrycode is the documented filter; a bbox alone leaks Nepal and
        # Bangladesh results for Indian-sounding prefixes.
        ("countrycode", "IN"),
        ("osm_tag", "place:village"), ("osm_tag", "place:hamlet"),
        ("osm_tag", "place:town"), ("osm_tag", "place:city"),
    ]
    if lat is not None and lon is not None:
        params += [("lat", str(lat)), ("lon", str(lon))]

    def _call():
        return get(config.PHOTON_URL, params=params, timeout=(3.05, 8.0)).json()

    data = call_source("photon", _call)
    if not data:
        return []

    out = []
    for feat in data.get("features", [])[:limit]:
        props = feat.get("properties", {})
        coords = (feat.get("geometry") or {}).get("coordinates") or [None, None]
        if coords[0] is None:
            continue
        out.append({
            "id": f"osm:{props.get('osm_type', 'N')}{props.get('osm_id')}",
            "name": props.get("name") or query,
            "subdistrict": props.get("district") or props.get("locality"),
            "district": props.get("county"),
            "state": props.get("state"),
            "lat": float(coords[1]),
            "lon": float(coords[0]),
            "population": 0,
            "kind": props.get("osm_value") or "village",
            "source": "photon",
        })
    if out:
        bump("geocode_photon")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def search(query: str, lat: float | None = None, lon: float | None = None, limit: int = 8) -> dict:
    """Village suggestions for a typed prefix."""
    query = (query or "").strip()
    if len(query) < 2:
        return {"results": [], "source": "none"}

    results = _search_local(query, lat, lon, limit)
    source = "geonames"
    if not results:
        key = make_key("photon", query.lower(), round(lat, 1) if lat else None,
                       round(lon, 1) if lon else None, limit)
        results = cached_call(
            key, lambda: _search_photon(query, lat, lon, limit),
            expire=config.CACHE_TTL_GEOCODE,
        )
        source = "photon"

    return {"results": results or [], "source": source if results else "none"}


def get_village(village_id: str) -> dict | None:
    """Resolve an id from the search results back to a place."""
    if village_id.startswith("gn:"):
        conn = _connection()
        if conn is None:
            return None
        row = conn.execute(
            """SELECT geonameid, name, lat, lon, fcode, state, district, subdistrict, population
               FROM villages WHERE geonameid = ?""",
            (village_id[3:],),
        ).fetchone()
        if row is None:
            return None
        return {
            "id": village_id,
            "name": row["name"],
            "subdistrict": row["subdistrict"],
            "district": row["district"],
            "state": row["state"],
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "population": int(row["population"] or 0),
            "kind": row["fcode"],
            "source": "geonames",
        }

    if village_id.startswith("osm:"):
        # Photon results carry their coordinates in the id lookup cache; the
        # client also sends them back, so this is only a fallback path.
        return None
    return None


def boundary_for(village: dict, radius_m: float | None = None) -> dict:
    """
    A polygon for the village.

    Real administrative boundaries for Indian villages are thin on the ground:
    OSM has only about 57,600 village-level relations for 677,662 villages, and
    the open Census polygons carry a non-commercial licence.  So unless an OSM
    relation is found, this returns a disc around the point and labels it as
    synthesized, because the polygon the user draws is what actually gets
    analysed.
    """
    lat, lon = village["lat"], village["lon"]
    osm = _osm_boundary(lat, lon, village.get("name", ""))
    if osm is not None:
        return osm

    # Default extent: a disc of the median Indian village area, about 250 ha
    radius_m = radius_m or 900.0
    return {
        "type": "Feature",
        "properties": {
            "source": "synthesized",
            "note": "Approximate extent around the village point; draw your own area to analyse",
            "radius_m": radius_m,
        },
        "geometry": _disc(lat, lon, radius_m),
    }


def _osm_boundary(lat: float, lon: float, name: str) -> dict | None:
    """Look for an administrative relation around the point via Overpass."""
    query = (
        f'[out:json][timeout:20];'
        f'is_in({lat},{lon})->.a;'
        f'relation(pivot.a)["boundary"="administrative"]["admin_level"~"^(9|10)$"];'
        f'out geom;'
    )

    def _call():
        from core.resilience import http

        resp = http().post(config.OVERPASS_ENDPOINTS[0], data={"data": query}, timeout=(3.05, 25.0))
        resp.raise_for_status()
        return resp.json()

    key = make_key("boundary", round(lat, 4), round(lon, 4))
    data = cached_call(
        key, lambda: call_source("overpass_boundary", _call), expire=config.CACHE_TTL_GEOCODE
    )
    if not data or not data.get("elements"):
        return None

    element = data["elements"][0]
    rings = []
    for member in element.get("members", []):
        if member.get("role") not in ("outer", "", None):
            continue
        coords = [(p["lon"], p["lat"]) for p in member.get("geometry", []) or []]
        if len(coords) >= 4:
            rings.append(coords)
    if not rings:
        return None

    rings.sort(key=len, reverse=True)
    ring = rings[0]
    if ring[0] != ring[-1]:
        ring = ring + [ring[0]]

    return {
        "type": "Feature",
        "properties": {
            "source": "osm",
            "name": (element.get("tags") or {}).get("name", name),
            "admin_level": (element.get("tags") or {}).get("admin_level"),
        },
        "geometry": {"type": "Polygon", "coordinates": [[[x, y] for x, y in ring]]},
    }


def _disc(lat: float, lon: float, radius_m: float, points: int = 48) -> dict:
    dlat = radius_m / 111_320.0
    dlon = radius_m / (111_320.0 * max(0.2, math.cos(math.radians(lat))))
    ring = []
    for i in range(points + 1):
        angle = 2 * math.pi * i / points
        ring.append([lon + dlon * math.cos(angle), lat + dlat * math.sin(angle)])
    return {"type": "Polygon", "coordinates": [ring]}
