"""
Configuration for the TerraFlow backend.

Every tunable in the plan lives here so the analysis can be reproduced and the
report can quote the exact defaults that produced a figure.  Anything that a
deployment might need to change is also readable from an environment variable.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("POND_DATA_DIR", BASE_DIR / "data"))
CACHE_DIR = Path(os.getenv("POND_CACHE_DIR", DATA_DIR / "cache"))
GAZETTEER_DB = Path(os.getenv("POND_GAZETTEER_DB", DATA_DIR / "gazetteer.sqlite"))
RAINFALL_DIR = Path(os.getenv("POND_RAINFALL_DIR", DATA_DIR / "rainfall"))
RAINFALL_MEMMAP = RAINFALL_DIR / "imd_rf25.dat"
RAINFALL_META = RAINFALL_DIR / "imd_rf25.json"
GCN250_TIF = Path(os.getenv("POND_GCN250_TIF", DATA_DIR / "gcn250" / "GCN250_india.tif"))
NUMBA_CACHE_DIR = Path(os.getenv("NUMBA_CACHE_DIR", DATA_DIR / "numba_cache"))

for _d in (DATA_DIR, CACHE_DIR, RAINFALL_DIR, NUMBA_CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# numba must see this before pyflwdir is imported, so that its @njit(cache=True)
# functions are compiled once and reused across worker restarts.
os.environ.setdefault("NUMBA_CACHE_DIR", str(NUMBA_CACHE_DIR))

# GDAL / rasterio environment for anonymous windowed reads of public COGs.
os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "3")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "1")
os.environ.setdefault("VSI_CACHE", "TRUE")
os.environ.setdefault("GDAL_CACHEMAX", "512")

# ─────────────────────────────────────────────────────────────────────────────
# Versioning — bump to invalidate every cached bundle / result
# ─────────────────────────────────────────────────────────────────────────────

ALGO_VERSION = "3.3.0"

# ─────────────────────────────────────────────────────────────────────────────
# Server
# ─────────────────────────────────────────────────────────────────────────────

HOST = os.getenv("POND_HOST", "0.0.0.0")
PORT = int(os.getenv("POND_PORT", "8000"))
CORS_ORIGINS = os.getenv(
    "POND_CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://localhost:4173",
).split(",")

MAX_UPLOAD_SIZE_MB = 50
# A cold village pays for an elevation fetch plus an Overpass query, which
# together can run to half a minute on a slow network; a warm one answers in
# about a second.  The budget has to cover the cold case or the first request
# in every village fails.
COMPUTE_TIMEOUT_S = float(os.getenv("POND_COMPUTE_TIMEOUT", "75"))
POOL_WORKERS = int(os.getenv("POND_POOL_WORKERS", str(min(8, (os.cpu_count() or 4)))))

# Per-source HTTP timeouts: (connect, read)
HTTP_TIMEOUT = (3.05, 20.0)
HTTP_USER_AGENT = os.getenv(
    "POND_USER_AGENT", "TerraFlow/3.2 (course project; contact: omanand8132@gmail.com)"
)

# Circuit breaker: 3 consecutive failures open the breaker for 60 s.
BREAKER_FAIL_MAX = 3
BREAKER_RESET_TIMEOUT_S = 60

# Cache lifetimes (seconds); None = never expires (keyed by ALGO_VERSION).
CACHE_TTL_BUNDLE = None
CACHE_TTL_DEM = None
CACHE_TTL_RESULT = 7 * 86400
CACHE_TTL_OSM = 30 * 86400
CACHE_TTL_CLIMATE = 30 * 86400
CACHE_TTL_GEOCODE = 30 * 86400
CACHE_SIZE_LIMIT_BYTES = int(os.getenv("POND_CACHE_SIZE", str(8 * 10**9)))
CACHE_SHARDS = 8


# ─────────────────────────────────────────────────────────────────────────────
# Analysis parameters
# ─────────────────────────────────────────────────────────────────────────────


class AnalysisConfig(BaseModel):
    """Terrain-analysis parameters (plan sections 5 and 7)."""

    # ── Grid ────────────────────────────────────────────────────────────────
    dem_resolution_m: float = 30.0
    fine_resolution_m: float = 10.0          # used below fine_resample_max_km2
    fine_resample_max_km2: float = 5.0
    pad_m: float = 2000.0                    # buffer around the selection
    pad_growth: tuple = (2000.0, 4000.0, 8000.0, 16000.0)
    max_cells: int = 4_000_000

    # ── Selection limits ────────────────────────────────────────────────────
    min_area_ha: float = 1.0
    max_area_km2: float = 25.0
    max_vertices: int = 200

    # ── Depressions / candidates ────────────────────────────────────────────
    min_depression_depth_m: float = 0.5
    min_depression_area_ha: float = 0.1
    candidate_min_spacing_m: float = 50.0
    num_candidates: int = 5
    max_candidates: int = 20

    # ── Streams ─────────────────────────────────────────────────────────────
    stream_threshold_ha: float = 2.0
    snap_distance_cells: int = 5

    # ── Embankment (DamSite-style) search ───────────────────────────────────
    embankment_min_upstream_ha: float = 1.0
    embankment_max_upstream_ha: float = 50.0
    embankment_heights_m: tuple = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
    # A village structure impounds a few hundred metres of valley; water that
    # reaches the edge of this window is leaking somewhere else.
    embankment_max_extent_m: float = 400.0
    # Sanity cap: anything larger is a reservoir, not a farm pond or check dam
    max_site_capacity_m3: float = 2_000_000.0
    # Below this the natural hollow is not worth calling a pond on its own;
    # the site is offered as an excavated pond instead.
    min_useful_capacity_m3: float = 300.0
    # A sheet of water a few centimetres deep over a hectare is not a pond: it
    # evaporates in weeks and nobody would build it.  Below this average depth
    # the location is offered as something to excavate instead.
    min_mean_depth_m: float = 0.3
    # MGNREGA-style excavated farm pond used where terrain offers no basin
    dugout_side_m: float = 20.0
    dugout_depth_m: float = 3.0
    dugout_side_slope_factor: float = 0.6   # trapezoidal section vs a box

    # ── Stage–storage ───────────────────────────────────────────────────────
    stage_step_m: float = 0.1
    freeboard_m: float = 0.5
    dead_storage_m: float = 0.3
    dem_vertical_uncertainty_m: float = 2.0

    # ── AHP weights (appendix pairwise matrix, CR < 0.10) ───────────────────
    weights: dict = Field(
        default_factory=lambda: {
            "runoff": 0.29,
            "slope": 0.16,
            "stream": 0.16,
            "depression": 0.16,
            "soil": 0.09,
            "landuse": 0.09,
            "settlement": 0.05,
        }
    )

    # ── Legacy contour-upload path (phase 2) ────────────────────────────────
    # Kept so the KML demo keeps working against the same config object.
    max_contour_vertices: int = 50000
    rbf_kernel: str = "thin_plate_spline"
    min_depression_area_cells: int = 20
    min_catchment_area_km2: float = 0.01
    score_weights: dict = Field(
        default_factory=lambda: {
            "depression": 0.35, "twi": 0.25, "slope": 0.20, "accumulation": 0.20,
        }
    )

    # ── Constraint buffers (metres) ─────────────────────────────────────────
    buffer_building_m: float = 100.0
    buffer_road_m: float = 30.0
    buffer_railway_m: float = 50.0
    buffer_water_m: float = 50.0
    max_slope_pct_pond: float = 10.0
    max_slope_pct_embankment: float = 15.0


class HydrologyParams(BaseModel):
    """Runoff / water-balance parameters (plan section 6)."""

    lambda_ia: float = 0.3                  # CGWB: 0.3 (0.1 black soils AMC II/III)
    black_soil: bool = False
    cn_override: float | None = None
    seepage_mm_day: float = 24.0            # unlined pond (CRIDA rule of thumb)
    seepage_mm_day_lined: float = 2.0
    lined: bool = False
    open_water_kc: float = 1.05             # FAO-56 Table 12, shallow tropical
    dependability_pct: float = 75.0
    record_start_year: int = 1991
    record_end_year: int = 2025

    # AMC thresholds on 5-day antecedent rainfall (mm), CGWB Table 4.13
    amc_dormant_low: float = 12.5
    amc_dormant_high: float = 27.5
    amc_growing_low: float = 35.0
    amc_growing_high: float = 52.5
    growing_months: tuple = (6, 7, 8, 9, 10)


class CostConfig(BaseModel):
    """Earthwork unit costs in rupees (CPWD DSR 2023 chapter 2 unless noted)."""

    excavation_per_m3: float = 177.50       # item 2.6.1 mechanical, all soils
    fill_per_m3: float = 197.90             # item 2.3.1 banking, 20 cm layers
    haul_per_m3_per_50m: float = 33.70      # state SoR lead slab
    haul_per_m3_per_km: float = 674.00      # beyond 1 km (20 x 50 m slabs)
    borrow_per_m3: float = 700.50           # item 2.25(a)
    waste_per_m3: float = 20.00             # disposal beyond free lead
    lining_per_m2: float = 98.00            # HDPE 500 micron, optional
    free_lead_m: float = 50.0
    manual_rates: bool = False              # MGNREGS labour-intensive toggle
    manual_excavation_per_m3: float = 620.0

    def resolved(self) -> "CostConfig":
        """Apply the manual-labour toggle to the excavation rate."""
        if not self.manual_rates:
            return self
        return self.model_copy(
            update={"excavation_per_m3": self.manual_excavation_per_m3}
        )


class GeometryConfig(BaseModel):
    """Bund / pond geometry defaults (FAO pond construction manual)."""

    bund_top_width_m: float = 2.0
    bund_side_slope: float = 2.0            # m:1 horizontal:vertical
    freeboard_m: float = 0.5
    settlement_allowance: float = 0.10      # SA, of construction height
    shrink: float = 0.10                    # sigma, compacted -> bank
    swell: float = 0.25                     # truck trips only
    max_dig_depth_m: float = 3.0
    max_spill_raise_m: float = 2.5
    # An excavated pond is a designed shape, so the budget can also widen it.
    # 5 m is as deep as an unlined village pond is normally taken; a 100 m
    # side is a hectare, beyond which it is a tank rather than a farm pond.
    dugout_max_depth_m: float = 5.0
    dugout_max_side_m: float = 100.0
    bund_height_warn_m: float = 3.0
    bund_height_max_m: float = 5.0


# ─────────────────────────────────────────────────────────────────────────────
# Curve-number lookup fallback (CGWB Table 4.14, AMC II) by land use x HSG
# ─────────────────────────────────────────────────────────────────────────────

CN_TABLE_AMC2: dict[str, dict[str, int]] = {
    #                     A    B    C    D
    "cropland":         {"A": 67, "B": 78, "C": 85, "D": 89},
    "cropland_bunded":  {"A": 59, "B": 69, "C": 76, "D": 79},
    "paddy":            {"A": 95, "B": 95, "C": 95, "D": 95},
    "fallow":           {"A": 77, "B": 86, "C": 91, "D": 94},
    "orchard":          {"A": 39, "B": 53, "C": 67, "D": 71},
    "forest":           {"A": 26, "B": 40, "C": 58, "D": 61},
    "scrub":            {"A": 33, "B": 47, "C": 64, "D": 67},
    "grass":            {"A": 39, "B": 61, "C": 74, "D": 80},
    "wasteland":        {"A": 71, "B": 80, "C": 85, "D": 88},
    "builtup":          {"A": 77, "B": 85, "C": 90, "D": 92},
    "water":            {"A": 100, "B": 100, "C": 100, "D": 100},
    "unknown":          {"A": 62, "B": 74, "C": 82, "D": 86},
}

DEFAULT_HSG = "C"

# Land-use suitability score for the siting model (1.0 = most suitable)
LANDUSE_SUITABILITY: dict[str, float] = {
    "wasteland": 1.0,
    "scrub": 1.0,
    "grass": 0.9,
    "fallow": 0.8,
    "cropland": 0.6,
    "cropland_bunded": 0.6,
    "paddy": 0.5,
    "unknown": 0.6,
    "orchard": 0.4,
    "forest": 0.0,
    "builtup": 0.0,
    "water": 0.0,
}

# Hydrologic-soil-group suitability for surface storage (D holds water best)
HSG_SUITABILITY: dict[str, float] = {"D": 1.0, "C": 0.8, "B": 0.4, "A": 0.2}

# ─────────────────────────────────────────────────────────────────────────────
# Structure rules (Ramakrishnan et al. 2009, after IMSD 1995 / INCOH)
# ─────────────────────────────────────────────────────────────────────────────

STRUCTURE_RULES = {
    "farm_pond": {
        "label": "Farm pond",
        "slope_pct": (0.0, 5.0),
        "catchment_ha": (0.5, 5.0),
        "water_level_m": (2.0, 2.5),
        "storage_m3": (2000, 5000),
    },
    "check_dam": {
        "label": "Check dam / embankment",
        "slope_pct": (0.0, 15.0),
        "catchment_ha": (5.0, 50.0),
        "water_level_m": (4.0, 5.0),
        "storage_m3": (5000, 7000),
    },
    "percolation_tank": {
        "label": "Percolation tank",
        "slope_pct": (0.0, 10.0),
        "catchment_ha": (25.0, 40.0),
        "water_level_m": (6.0, 7.0),
        "storage_m3": (5000, 10000),
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# External endpoints
# ─────────────────────────────────────────────────────────────────────────────

COPERNICUS_BUCKET = "https://copernicus-dem-30m.s3.amazonaws.com"
TERRAIN_TILES_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
TERRAIN_TILE_ZOOM = 13

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
OSM_MAP_API_URLS = [
    "https://www.openstreetmap.org/api/0.6/map",
    "https://api.openstreetmap.org/api/0.6/map",
]

PHOTON_URL = "https://photon.komoot.io/api/"
NOMINATIM_LOOKUP_URL = "https://nominatim.openstreetmap.org/lookup"

NASA_POWER_DAILY = "https://power.larc.nasa.gov/api/temporal/daily/point"
NASA_POWER_CLIMATOLOGY = "https://power.larc.nasa.gov/api/temporal/climatology/point"

IMD_RF25_POST = "https://www.imdpune.gov.in/cmpg/Griddata/RF25.php"

# IMD 0.25 degree grid geometry
IMD_LAT0, IMD_LON0, IMD_STEP = 6.5, 66.5, 0.25
IMD_NLAT, IMD_NLON = 129, 135
