"""
Curve number for a catchment.

Preferred source is the GCN250 global raster (Jaafar et al. 2019), clipped to
India once by ``scripts/prepare_gcn250.py`` and read here with a windowed
rasterio call.  Without it the curve number is built from the OSM land-use grid
and a hydrologic soil group, using the CGWB Table 4.14 lookup in config.

Note on the lambda convention: GCN250 and the NRCS tables were calibrated with
Ia = 0.2 S, while the CGWB manual prescribes 0.3 S for India (0.1 S for black
soils in wetter antecedent conditions).  The value used is returned with the
result so the report can state it.
"""

from __future__ import annotations

import logging

import numpy as np

import config

logger = logging.getLogger(__name__)


def catchment_curve_number(
    mask: np.ndarray,
    transform,
    epsg: int,
    landuse_class: np.ndarray | None = None,
    hsg: str = config.DEFAULT_HSG,
    override: float | None = None,
) -> dict:
    """
    Area-weighted curve number (antecedent condition II) over a catchment mask.

    Returns ``{cn, source, hsg, breakdown}``.
    """
    if override is not None:
        return {
            "cn": float(np.clip(override, 30.0, 100.0)),
            "source": "user_override",
            "hsg": hsg,
            "breakdown": {},
        }

    from_raster = _cn_from_gcn250(mask, transform, epsg)
    if from_raster is not None:
        return from_raster

    return _cn_from_landuse(mask, landuse_class, hsg)


def _cn_from_gcn250(mask: np.ndarray, transform, epsg: int) -> dict | None:
    """Zonal mean of the GCN250 ARC II raster over the catchment."""
    if not config.GCN250_TIF.exists():
        return None
    try:
        import rasterio
        from rasterio.warp import Resampling, reproject

        with rasterio.open(config.GCN250_TIF) as src:
            dest = np.full(mask.shape, np.nan, dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=dest,
                dst_transform=transform,
                dst_crs=f"EPSG:{epsg}",
                resampling=Resampling.nearest,
                src_nodata=src.nodata,
                dst_nodata=np.nan,
            )
        values = dest[mask & np.isfinite(dest)]
        values = values[(values >= 1) & (values <= 100)]
        if values.size == 0:
            return None
        return {
            "cn": round(float(values.mean()), 1),
            "source": "gcn250",
            "hsg": "from_raster",
            "breakdown": {},
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("GCN250 read failed, falling back to the lookup table: %s", exc)
        return None


def _cn_from_landuse(mask: np.ndarray, landuse_class: np.ndarray | None, hsg: str) -> dict:
    hsg = hsg if hsg in ("A", "B", "C", "D") else config.DEFAULT_HSG
    if landuse_class is None:
        cn = config.CN_TABLE_AMC2["unknown"][hsg]
        return {
            "cn": float(cn),
            "source": "default_table",
            "hsg": hsg,
            "breakdown": {"unknown": 1.0},
        }

    classes = landuse_class[mask]
    if classes.size == 0:
        cn = config.CN_TABLE_AMC2["unknown"][hsg]
        return {"cn": float(cn), "source": "default_table", "hsg": hsg, "breakdown": {}}

    unique, counts = np.unique(classes.astype(str), return_counts=True)
    total = counts.sum()
    weighted, breakdown = 0.0, {}
    for cls, count in zip(unique, counts):
        table = config.CN_TABLE_AMC2.get(cls, config.CN_TABLE_AMC2["unknown"])
        share = count / total
        weighted += table[hsg] * share
        breakdown[cls] = round(float(share), 3)

    return {
        "cn": round(float(weighted), 1),
        "source": "osm_landuse_table",
        "hsg": hsg,
        "breakdown": breakdown,
    }


def hsg_from_cn(cn: float) -> str:
    """
    Rough inverse used only to score soil suitability when no soil layer exists:
    a high curve number on ordinary land implies a tighter soil.
    """
    if cn >= 88:
        return "D"
    if cn >= 80:
        return "C"
    if cn >= 70:
        return "B"
    return "A"


def convert_cn(cn2: float, amc: int) -> float:
    """
    Convert an antecedent-condition-II curve number to condition I or III
    (Chow et al., as reproduced in the Indian literature).
    """
    cn2 = float(np.clip(cn2, 1.0, 100.0))
    if amc == 1:
        return float(cn2 / (2.281 - 0.01281 * cn2))
    if amc == 3:
        return float(cn2 / (0.427 + 0.00573 * cn2))
    return cn2
