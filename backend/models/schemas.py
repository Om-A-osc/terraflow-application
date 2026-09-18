"""
Request and response models.

Responses are deliberately loose about the nested analysis blocks (they are
plain dicts assembled by the services) but strict about everything the client
sends, so a bad polygon or a negative budget is rejected at the boundary with a
message the user can act on.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

import config


# ─────────────────────────────────────────────────────────────────────────────
# Shared
# ─────────────────────────────────────────────────────────────────────────────


class ErrorResponse(BaseModel):
    status: str = "error"
    detail: str
    error_code: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    service: str = "TerraFlow API"
    version: str
    algo_version: str
    engine: str
    local_gazetteer: bool
    local_rainfall: bool


class MetricsResponse(BaseModel):
    counters: dict
    breakers: dict
    pool: dict
    cache: dict


# ─────────────────────────────────────────────────────────────────────────────
# Village search
# ─────────────────────────────────────────────────────────────────────────────


class VillageSuggestion(BaseModel):
    id: str
    name: str
    subdistrict: Optional[str] = None
    district: Optional[str] = None
    state: Optional[str] = None
    lat: float
    lon: float
    population: int = 0
    kind: Optional[str] = None
    source: str


class VillageSearchResponse(BaseModel):
    results: list[VillageSuggestion]
    source: str


class VillageDetail(BaseModel):
    village: VillageSuggestion
    boundary: Optional[dict] = None
    suggested_area: Optional[dict] = Field(
        None, description="A default polygon to analyse if the user does not draw one"
    )
    rainfall: dict = Field(default_factory=dict)
    bundle_ready: bool = False


class WarmResponse(BaseModel):
    status: Literal["ready", "started", "failed"]
    village_id: str
    job_id: Optional[str] = None
    detail: Optional[str] = None
    elapsed_s: Optional[float] = None


class JobStatus(BaseModel):
    job_id: str
    status: Literal["running", "done", "failed"]
    stage: Optional[str] = None
    detail: Optional[str] = None
    elapsed_s: Optional[float] = None


# ─────────────────────────────────────────────────────────────────────────────
# Analysis
# ─────────────────────────────────────────────────────────────────────────────


class HydrologyOverrides(BaseModel):
    """User-visible assumptions; every one of these changes the headline number."""

    lambda_ia: Optional[float] = Field(
        None, ge=0.05, le=0.4,
        description="Initial abstraction ratio Ia/S. CGWB prescribes 0.3 for India "
                    "(0.1 for black soils in wetter conditions); NRCS tables use 0.2.",
    )
    black_soil: bool = False
    curve_number: Optional[float] = Field(None, ge=30, le=100)
    seepage_mm_day: Optional[float] = Field(None, ge=0, le=200)
    lined: bool = False
    hsg: Literal["A", "B", "C", "D"] = config.DEFAULT_HSG
    dependability_pct: float = Field(75.0, ge=50, le=95)

    def to_params(self) -> config.HydrologyParams:
        params = config.HydrologyParams()
        if self.lambda_ia is not None:
            params.lambda_ia = self.lambda_ia
        params.black_soil = self.black_soil
        params.cn_override = self.curve_number
        if self.seepage_mm_day is not None:
            params.seepage_mm_day = self.seepage_mm_day
        params.lined = self.lined
        params.dependability_pct = self.dependability_pct
        return params


class AnalyzeRequest(BaseModel):
    geometry: dict = Field(..., description="GeoJSON Polygon in WGS84")
    num_sites: int = Field(5, ge=1, le=20)
    resolution_m: Optional[float] = Field(None, ge=5, le=90)
    dem_source: Literal["copernicus", "terrain"] = "copernicus"
    hydrology: HydrologyOverrides = Field(default_factory=HydrologyOverrides)
    village_id: Optional[str] = None

    @field_validator("geometry")
    @classmethod
    def _check_geometry(cls, value: dict) -> dict:
        if not isinstance(value, dict):
            raise ValueError("geometry must be a GeoJSON object")
        if value.get("type") == "Feature":
            value = value.get("geometry") or {}
        if value.get("type") not in ("Polygon", "MultiPolygon"):
            raise ValueError("geometry must be a Polygon or MultiPolygon")
        coords = value.get("coordinates")
        if not coords:
            raise ValueError("geometry has no coordinates")
        ring = coords[0] if value["type"] == "Polygon" else coords[0][0]
        if len(ring) < 4:
            raise ValueError("a polygon needs at least three distinct corners")
        if len(ring) > config.AnalysisConfig().max_vertices:
            raise ValueError(
                f"the polygon has {len(ring)} vertices; the maximum is "
                f"{config.AnalysisConfig().max_vertices}. Simplify it and try again."
            )
        return value


# ─────────────────────────────────────────────────────────────────────────────
# Earthwork
# ─────────────────────────────────────────────────────────────────────────────


class CostOverrides(BaseModel):
    """Unit rates in rupees; defaults come from CPWD DSR 2023 chapter 2."""

    excavation_per_m3: Optional[float] = Field(None, ge=0, le=5000)
    fill_per_m3: Optional[float] = Field(None, ge=0, le=5000)
    haul_per_m3_per_50m: Optional[float] = Field(None, ge=0, le=1000)
    haul_per_m3_per_km: Optional[float] = Field(None, ge=0, le=20000)
    borrow_per_m3: Optional[float] = Field(None, ge=0, le=5000)
    waste_per_m3: Optional[float] = Field(None, ge=0, le=5000)
    manual_rates: bool = Field(
        False, description="Use labour-intensive MGNREGS rates instead of mechanical ones"
    )

    def to_config(self) -> config.CostConfig:
        costs = config.CostConfig()
        for field_name in (
            "excavation_per_m3", "fill_per_m3", "haul_per_m3_per_50m",
            "haul_per_m3_per_km", "borrow_per_m3", "waste_per_m3",
        ):
            value = getattr(self, field_name)
            if value is not None:
                setattr(costs, field_name, value)
        costs.manual_rates = self.manual_rates
        return costs


class GeometryOverrides(BaseModel):
    bund_top_width_m: Optional[float] = Field(None, ge=0.5, le=10)
    bund_side_slope: Optional[float] = Field(None, ge=1, le=4)
    freeboard_m: Optional[float] = Field(None, ge=0, le=2)
    max_dig_depth_m: Optional[float] = Field(None, ge=0.5, le=8)
    max_spill_raise_m: Optional[float] = Field(None, ge=0, le=5)

    def to_config(self) -> config.GeometryConfig:
        geometry = config.GeometryConfig()
        for field_name in (
            "bund_top_width_m", "bund_side_slope", "freeboard_m",
            "max_dig_depth_m", "max_spill_raise_m",
        ):
            value = getattr(self, field_name)
            if value is not None:
                setattr(geometry, field_name, value)
        return geometry


class EarthworkRequest(BaseModel):
    analysis_id: str
    site_id: str
    budget: float = Field(..., ge=0, le=1e10, description="Budget in rupees")
    costs: CostOverrides = Field(default_factory=CostOverrides)
    geometry: GeometryOverrides = Field(default_factory=GeometryOverrides)
    include_haul_plan: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Legacy contour upload (phase 2) — kept so the original demo still runs
# ─────────────────────────────────────────────────────────────────────────────


class Extent(BaseModel):
    min_lon: float
    max_lon: float
    min_lat: float
    max_lat: float


class ElevationRange(BaseModel):
    min: float
    max: float


class DEMSize(BaseModel):
    rows: int
    cols: int


class AnalysisMetadata(BaseModel):
    filename: str
    contour_count: int
    elevation_range: ElevationRange
    extent: Extent
    dem_resolution_m: float
    dem_size: DEMSize


class DEMStats(BaseModel):
    mean_elevation: float
    std_elevation: float
    mean_slope_deg: float


class CatchmentInfo(BaseModel):
    area_km2: float
    polygon: dict


class PondCandidate(BaseModel):
    rank: int
    score: float
    location: dict
    elevation_m: float
    depression_depth_m: float
    estimated_volume_m3: float
    estimated_surface_area_m2: float
    catchment: CatchmentInfo
    pond_footprint: Optional[dict] = None
    twi: float
    slope_deg: float


class AnalysisResponse(BaseModel):
    status: str = "success"
    metadata: AnalysisMetadata
    contours_geojson: dict
    candidates: list[PondCandidate]
    dem_stats: DEMStats
    water_bodies_geojson: Optional[dict] = None
