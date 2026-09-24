"""
Tests for runoff, the water balance and the earthwork optimiser.

The curve-number cases are checked against figures worked by hand from the
CGWB manual, so a change in the formula shows up here rather than in a report.
"""

from __future__ import annotations

import numpy as np
import pytest
from affine import Affine

import config
from services.earthwork_service import optimise
from services.rainfall_service import _with_calendar, weibull_dependable
from services.runoff_service import annual_inflow, daily_runoff, water_balance
from services.storage_service import stage_storage_curve

RES = 10.0


def synthetic_rainfall(years: int = 20, seed: int = 7) -> dict:
    """A monsoon-shaped daily series: most rain falls June to September."""
    rng = np.random.default_rng(seed)
    start = np.datetime64("2000-01-01", "D")
    n = years * 365
    dates = start + np.arange(n, dtype="timedelta64[D]")
    months = dates.astype("datetime64[M]").astype(int) % 12 + 1

    values = np.zeros(n)
    monsoon = np.isin(months, (6, 7, 8, 9))
    # Rain on about a third of monsoon days, exponentially distributed
    wet = monsoon & (rng.random(n) < 0.35)
    values[wet] = rng.exponential(22.0, wet.sum())
    return _with_calendar({
        "dates": dates, "values": values.astype(np.float32),
        "source": "synthetic", "label": "synthetic monsoon",
    })


# ── Curve number ─────────────────────────────────────────────────────────────


def _one_storm(depth_mm: float, antecedent_mm: float = 0.0, month: int = 7) -> dict:
    """Five antecedent days then the storm, so the AMC class is controlled."""
    dates = np.array(
        [f"2010-{month:02d}-{day:02d}" for day in range(10, 16)], dtype="datetime64[D]"
    )
    values = np.full(6, antecedent_mm / 5.0, dtype=np.float32)
    values[-1] = depth_mm
    return _with_calendar({
        "dates": dates, "values": values, "source": "test", "label": "test",
    })


def test_scs_cn_matches_a_hand_worked_figure():
    """
    CN 80 at average antecedent moisture, lambda 0.3, a 60 mm storm:
    S = 25400/80 - 254 = 63.5 mm, Ia = 19.05 mm,
    Q = (60 - 19.05)^2 / (60 - 19.05 + 63.5) = 16.1 mm.

    The five preceding days carry 40 mm so the growing-season thresholds
    (35 and 52.5 mm) put the day in condition II and the curve number is used
    as given.
    """
    series = _one_storm(60.0, antecedent_mm=40.0)
    params = config.HydrologyParams(lambda_ia=0.3)
    result = daily_runoff(series, 80.0, params)
    assert result["runoff_mm"][-1] == pytest.approx(16.1, abs=0.3)


def test_dry_antecedent_conditions_cut_the_runoff():
    """
    The same storm on dry ground is condition I: CN falls to about 64, S rises
    to 145 mm and the yield drops by an order of magnitude.  This is the
    adjustment the CGWB table prescribes, and it is easy to lose in a refactor.
    """
    wet = daily_runoff(_one_storm(60.0, antecedent_mm=60.0), 80.0)["runoff_mm"][-1]
    dry = daily_runoff(_one_storm(60.0, antecedent_mm=0.0), 80.0)["runoff_mm"][-1]
    assert dry < wet / 5
    assert dry == pytest.approx(1.7, abs=0.4)


def test_no_runoff_below_the_initial_abstraction():
    # Ia = 0.3 * 63.5 = 19.05 mm at condition II, so a 10 mm day yields nothing
    series = _one_storm(10.0, antecedent_mm=40.0)
    assert daily_runoff(series, 80.0)["runoff_mm"][-1] == 0.0


def test_lambda_changes_the_answer_materially():
    """The CGWB value and the NRCS value are not interchangeable."""
    series = synthetic_rainfall()
    strict = daily_runoff(series, 80.0, config.HydrologyParams(lambda_ia=0.3))
    nrcs = daily_runoff(series, 80.0, config.HydrologyParams(lambda_ia=0.2))
    assert nrcs["runoff_mm"].sum() > strict["runoff_mm"].sum() * 1.1


def test_daily_beats_lumping_the_totals():
    """
    The curve-number equation is convex, so feeding it a monthly total instead
    of the daily series inflates runoff.  This guards the ordering.
    """
    series = synthetic_rainfall(years=1)
    daily_total = daily_runoff(series, 80.0)["runoff_mm"].sum()

    monthly_sums = []
    for month in range(1, 13):
        sel = series["month"] == month
        monthly_sums.append(float(series["values"][sel].sum()))
    lumped_dates = np.array([f"2000-{m:02d}-15" for m in range(1, 13)], dtype="datetime64[D]")
    lumped = _with_calendar({
        "dates": lumped_dates, "values": np.array(monthly_sums, dtype=np.float32),
        "source": "test", "label": "test",
    })
    lumped_total = daily_runoff(lumped, 80.0)["runoff_mm"].sum()

    assert lumped_total > daily_total * 1.5


def test_weibull_dependability_ordering():
    values = np.array([100, 200, 300, 400, 500], dtype=float)
    p75 = weibull_dependable(values, 75.0)
    p50 = weibull_dependable(values, 50.0)
    # The value exceeded in 75 percent of years is smaller than the median
    assert p75 < p50 < values.max()
    assert values.min() <= p75 <= values.max()


# ── Water balance ────────────────────────────────────────────────────────────


def flat_curve(capacity: float, area: float) -> dict:
    """A simple prism: constant area, storage linear in depth."""
    depth = capacity / area
    return {
        "level_m": [0.0, depth],
        "area_m2": [0.0, area],
        "volume_m3": [0.0, capacity],
        "capacity_m3": capacity,
        "usable_level_m": depth,
        "spill_level_m": depth,
        "bed_level_m": 0.0,
    }


def test_small_pond_spills_and_captures_less_than_it_receives():
    series = synthetic_rainfall()
    runoff = daily_runoff(series, 80.0)
    catchment = 500_000.0                      # 50 ha

    evaporation = {"open_water_mm_month": [150] * 12, "annual_mm": 1800}
    small = water_balance(runoff, catchment, flat_curve(2000, 1000), 2000, evaporation)
    large = water_balance(runoff, catchment, flat_curve(200000, 40000), 200000, evaporation)

    inflow = annual_inflow(runoff, catchment)
    assert small["harvestable_mean_m3"] < inflow["mean_m3"]
    assert small["spill_mean_m3"] > 0
    # A bigger pond captures more of the same inflow
    assert large["harvestable_mean_m3"] > small["harvestable_mean_m3"]


def test_lining_reduces_losses():
    series = synthetic_rainfall()
    runoff = daily_runoff(series, 80.0)
    evaporation = {"open_water_mm_month": [120] * 12, "annual_mm": 1440}
    curve = flat_curve(20000, 8000)

    unlined = water_balance(runoff, 300_000, curve, 20000, evaporation,
                            config.HydrologyParams(lined=False))
    lined = water_balance(runoff, 300_000, curve, 20000, evaporation,
                          config.HydrologyParams(lined=True))

    assert lined["losses_mean_m3"] < unlined["losses_mean_m3"]
    assert lined["mean_wet_months"] >= unlined["mean_wet_months"]


# ── Earthwork optimiser ──────────────────────────────────────────────────────


def bowl_dem(n: int = 80, depth: float = 2.0) -> np.ndarray:
    y, x = np.mgrid[0:n, 0:n]
    r = np.hypot(x - n / 2, y - n / 2)
    dem = np.full((n, n), 100.0, dtype=np.float32)
    inside = r < 14
    dem[inside] = 100.0 - depth * (1 - r[inside] / 14.0)
    return dem.astype(np.float32)


def optimise_bowl(budget: float, **kwargs):
    dem = bowl_dem()
    seed = (40, 40)
    curve = stage_storage_curve(dem, seed, RES * RES, spill_level=100.0, step_m=0.1)
    footprint = np.zeros(dem.shape, dtype=bool)
    yy, xx = np.mgrid[0:80, 0:80]
    footprint[np.hypot(xx - 40, yy - 40) < 14] = True
    return optimise(
        dem=dem, footprint_mask=footprint, curve=curve,
        cell_area_m2=RES * RES, resolution_m=RES, budget=budget, seed=seed, **kwargs,
    )


def test_a_bigger_budget_never_buys_less_water():
    added = [optimise_bowl(b)["storage"]["added_m3"] for b in (100_000, 500_000, 2_000_000)]
    assert added[0] <= added[1] <= added[2]


def test_the_design_stays_within_budget():
    for budget in (100_000, 750_000, 3_000_000):
        result = optimise_bowl(budget)
        if result["within_budget"]:
            assert result["cost"]["total"] <= budget * 1.0001


def test_manual_rates_buy_less_for_the_same_money():
    """
    Labour-intensive MGNREGS rates are roughly three times the mechanical rate,
    so the same budget must buy materially less storage.

    The raise is pinned to zero here: a design that only builds a bund is paid
    for out of borrowed earth and is genuinely insensitive to the excavation
    rate, which would make the comparison vacuous.
    """
    # The bowl floor is about 6 ha, so a tenth of a metre of deepening is
    # already 6,000 cubic metres; the budget has to match that scale or the
    # optimiser correctly answers "you cannot afford to dig at all".
    dig_only = config.GeometryConfig(max_spill_raise_m=0.0)
    budget = 5_000_000
    machine = optimise_bowl(budget, geometry=dig_only)
    manual = optimise_bowl(
        budget, geometry=dig_only, costs=config.CostConfig(manual_rates=True)
    )
    assert machine["storage"]["added_m3"] > 0
    assert manual["storage"]["added_m3"] < machine["storage"]["added_m3"] * 0.6


def test_a_budget_too_small_to_dig_is_reported_honestly():
    """A budget that buys nothing must say so rather than invent storage."""
    dig_only = config.GeometryConfig(max_spill_raise_m=0.0)
    result = optimise_bowl(50_000, geometry=dig_only)
    assert result["storage"]["added_m3"] == 0.0
    assert result["design"]["extra_depth_m"] == 0.0


def test_zero_budget_buys_nothing():
    result = optimise_bowl(0.0)
    assert result["storage"]["added_m3"] == 0.0
    assert result["cost"]["total"] == 0.0


def test_cost_breakdown_adds_up():
    result = optimise_bowl(1_500_000)
    cost = result["cost"]
    parts = cost["excavation"] + cost["placement"] + cost["haulage"] + cost["borrow"] + cost["disposal"]
    assert parts == pytest.approx(cost["total"], rel=1e-6)


def test_bund_height_is_capped_and_flagged():
    """The optimiser must not push the bund to an unbuildable height."""
    result = optimise_bowl(500_000_000)          # effectively unlimited
    geometry = config.GeometryConfig()
    assert result["design"]["spill_raise_m"] <= geometry.max_spill_raise_m + 1e-9
    assert result["design"]["extra_depth_m"] <= geometry.max_dig_depth_m + 1e-9
    if result["design"]["bund_max_height_m"] > geometry.bund_height_warn_m:
        assert result["warnings"], "a tall bund must carry a warning"


def test_dugout_budget_buys_storage_and_never_less_for_more_money():
    """An excavated pond grows with the budget: deeper first, then wider."""
    from services.earthwork_service import optimise_dugout

    def run(budget):
        return optimise_dugout(331.0, 3.0, 20.0, 0.6, budget, resolution_m=30.0)

    one_lakh = run(100_000)
    assert one_lakh["storage"]["added_m3"] > 0
    assert one_lakh["cost"]["total"] <= 100_000
    assert one_lakh["design"]["new_depth_m"] > 3.0 or one_lakh["design"]["new_side_m"] > 20.0

    five_lakh = run(500_000)
    assert five_lakh["storage"]["added_m3"] >= one_lakh["storage"]["added_m3"]
    assert five_lakh["cost"]["total"] <= 500_000
    # Volume and spoil agree: what is dug is what is stored
    assert abs(five_lakh["design"]["excavation_m3"] - five_lakh["storage"]["added_m3"]) < 1.0

    nothing = run(0.0)
    assert nothing["storage"]["added_m3"] == 0
    assert nothing["cost"]["total"] == 0
