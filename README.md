# TerraFlow

**Read the land, find the water.**

Search any Indian village, draw an area on the map, and get back where a pond
should go, how much water it can hold, how much it will collect in a year, and
how much more it could store if you paid to move soil.

Everything it uses is free and needs no API key.

---

## What it does

1. **Search a village.** A local index of 557,960 Indian places answers a
   prefix in about two milliseconds, with tehsil, district and state shown
   because village names repeat constantly.
2. **Prepare in the background.** Selecting a village starts fetching its
   elevation, drainage and map features so the first area you draw is answered
   from local disk.
3. **Draw an area.** Polygon or rectangle, editable, with the area in hectares
   shown live and oversized or self-crossing shapes rejected before they are
   sent.
4. **Get results.** The catchment that drains into the selection, ranked pond
   sites with their own catchment and footprint, the storage each can hold, and
   the water each collects in an average year and in three years out of four.
5. **Apply a budget.** Enter rupees and the rates for excavating, hauling and
   placing soil, and the optimiser reports the extra storage the money buys,
   the depth to dig, the bund to build, and the full cost breakdown.

Everything is drawn on the map: catchment, drainage lines, pond footprints, the
footprint after the works, the bund, and optionally the soil haul routes.

### Reading the map

Overlays sit on satellite imagery, which is busy and mid-toned, so the styling
follows two rules. A colour means exactly one thing, and every bright line is
drawn over a dark casing that keeps it legible against fields and water.

| What you see | Meaning |
| --- | --- |
| White outline | The area you drew |
| Pink outline, faint fill | Everything that drains into your area. Dashed where it runs past the analysed window, so the figure is a lower bound. Deliberately not blue: a catchment is a boundary, not water, and in blue it was mistaken for the drainage running under it |
| Pale pink dashed | The catchment of the one pond you selected. Off until you ask for it, from the button on the site card or the layer switch |
| Cyan lines | Natural drainage inside your area, the only blue on the map because it is the only water. Heavier as the stream order rises |
| Green fill, white edge | The chosen pond. Only one is drawn at a time; the numbered pins show where the others are |
| Amber dashed, orange ring | The pond after the proposed works, and the bund that holds it |
| Numbered pin | A suggested pond, by rank. The chosen one is amber and larger |

Everything outside the area you drew is dimmed, so the answer reads against the
ground it is about. The dimming follows the viewport and switches itself off
once you zoom out far enough that the area is a small part of the view, because
past that point the sheet is all there would be to look at. The catchment is
the one thing deliberately drawn beyond the boundary, because it is upstream of
your area by definition.

Every suggested pond, and every pond footprint, lies inside the area you drew.
A cell counts as inside only when its centre is, each pond is confined to the
selection when its storage is worked out, and the drawn shapes are clipped to
the boundary, so a 30 m cell cannot overhang it.

Sites are told apart by their rank number rather than by colour, so the palette
stays free to mean something. The layer switches at the bottom left double as
the legend: each row carries the swatch it turns on.

---

## Running it

```bash
./deploy/run-local.sh          # API on :8000, frontend on :5173
```

The script creates the virtual environment and installs packages on first run.
Open <http://localhost:5173>. Interactive API docs are at
<http://localhost:8000/docs>.

To run the two halves separately:

```bash
cd backend && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cd backend && .venv/bin/python -m uvicorn main:app --port 8000

cd frontend && npm install && npm run dev
```

### Optional local datasets

The app works immediately without these; each one makes it better or faster.
Check which are in use at `/health`.

| Script | Downloads | Size | What it improves |
| --- | --- | --- | --- |
| `scripts/build_gazetteer.py` | GeoNames India | 100 MB | Village search offline in milliseconds instead of through Photon |
| `scripts/download_imd.py` | IMD daily rainfall, 1991 to 2025 | 0.43 GB | Gauge-measured rainfall instead of NASA POWER reanalysis, which smooths daily intensity and biases runoff low |
| `scripts/prepare_gcn250.py` | GCN250 curve numbers for India | ~60 MB | A measured curve number instead of the land-use lookup table |

```bash
cd backend
.venv/bin/python scripts/build_gazetteer.py
.venv/bin/python scripts/download_imd.py --start 1991 --end 2025
.venv/bin/python scripts/prepare_gcn250.py
```

---

## How it works

### Data

| Need | Source | Fallback |
| --- | --- | --- |
| Elevation | Copernicus GLO-30 on AWS Open Data, read as windowed COG | Tilezen terrain tiles (SRTM 30 m), decoded with OpenCV |
| Rainfall | IMD 0.25 degree daily grid, local memmap | NASA POWER daily point |
| Evaporation | NASA POWER monthly climatology, FAO-56 Penman-Monteith computed locally | Hargreaves, then a generic Indian climatology |
| Curve number | GCN250 at 250 m | OpenStreetMap land use with the CGWB table |
| Land use, water, roads, buildings | Overpass, with mirrors | The OSM map API, then no constraints with a warning |
| Village index | GeoNames India | Photon |
| Basemaps | Esri World Imagery, OpenStreetMap, OpenTopoMap | — |

Each source sits behind its own circuit breaker: three failures open it for a
minute, so a host that is down or blocked costs one timeout rather than one per
request. Every degraded path is reported in the response and lowers the
confidence shown against each site.

### Algorithms

**Terrain.** The elevation window is reprojected to UTM, buildings and woodland
are levelled to their surroundings because a global DEM is a surface model, and
then pyflwdir runs Priority-Flood depression filling, D8 flow directions,
upstream area, slope and Strahler stream ordering. On this machine that is 3.0
seconds for 4 million cells against 18.4 seconds for the pure-Python code it
replaced.

**Catchment.** The outlets of the drawn polygon are the cells inside it whose
downstream neighbour is outside. Sorting them by upstream area and keeping
enough to cover 95 percent of the outflow gives the catchment in one pass. A
catchment that reaches the edge of the window is reported as a lower bound.

**Storage.** For each water level the wetted set is flood-filled from the bed
and integrated on the raw elevation, giving the stage, area and storage curve
that the capacity, the chart and the budget filter all read from.

**Runoff.** The Indian form of the SCS Curve Number method from the CGWB manual,
run on the daily rainfall series with the antecedent moisture class set from the
previous five days. Daily matters: the equation is convex, so feeding it monthly
totals inflates runoff several times over, and there is a test for that.
Strange's monsoon table is computed alongside, because the accepted Indian
methods disagree by up to a factor of two and one number would be false
precision.

**Water balance.** Each hydrological year is stepped month by month, losing
evaporation and seepage over the current water spread and spilling anything
above capacity, which gives the volume actually captured rather than the volume
that fell.

**Siting.** Natural depressions, embankment sites swept across drainage lines,
and excavated ponds on flat ground. A Boolean mask removes anything within
100 m of a building, 30 m of a road, 50 m of water or railway, in forest, or too
steep. The survivors are scored by a weighted sum under AHP weights, and the
weights are perturbed 200 times to report how robust each ranking is.

**Budget.** For a pond in a natural basin, a search over excavation depth and
bund height, each evaluated by flooding the DEM inside a containment window so
that the volume claimed and the footprint drawn always agree. For an excavated
pond, which is a designed hole rather than a basin, a search over extra depth
and width instead: every cubic metre costs the same to dig, so the planner
takes the largest pond the money buys and, among near-equal designs, the
deepest, which loses least to evaporation. An optional second stage solves the
soil haul as a transportation problem with HiGHS.

### The four machines

One edge and three compute nodes. The edge serves the frontend and hashes
`/api` requests on the village key so a village's cached data stays on one node;
if that node fails, nginx retries the next one.

```
Browser ──▶ Machine A: nginx (static build, /api proxy, rate limit, failover)
                 │  consistent hash on village id
                 ├──▶ Machine B: FastAPI, 2 workers, 8-process pool, disk cache
                 ├──▶ Machine C: same
                 └──▶ Machine D: same
```

Install `deploy/nginx.conf` on the edge and `deploy/terraflow.service` on each
compute node. The gazetteer, rainfall archive and curve-number raster are
copied to every node; everything else is fetched once and cached.

---

## Measured performance

Measured on one machine (16 threads, 15 GB), 20 concurrent users with a 20
second think time, two minutes against a prepared village. 40 analyses, no
failures, no timeouts.

| Endpoint | Median | 75th | 95th | Slowest |
| --- | --- | --- | --- | --- |
| Analyse an area | 1.1 s | 1.3 s | 4.1 s | 4.2 s |
| Apply a budget | 38 ms | 40 ms | 450 ms | 450 ms |
| Village search | 15 ms | 53 ms | 88 ms | 150 ms |

Where the time goes in a single analysis of a 185,000 cell window:

| Step | Warm | Cold |
| --- | --- | --- |
| Elevation window | 0 | 2 to 4 s |
| OpenStreetMap features | 0 | 4 to 25 s |
| Fill, D8, accumulation, slope | 0.1 s | 0.3 s |
| Rainfall, runoff, water balance | 0.02 s | 2 to 3 s |
| Siting and scoring | 0.7 s | 0.7 s |
| Overlays | 0.03 s | 0.03 s |

Preparing a village took 36 seconds in the run above, almost all of it waiting
on the elevation bucket and Overpass, which is why selecting a village starts
that work in the background and the app stays usable meanwhile. Because the
analysis window is snapped to a fixed grid of whole cells, every area drawn in
one village reuses a single cached bundle: those 40 analyses triggered exactly
one elevation fetch and one Overpass query, and the cache served 80 percent of
all lookups.

Run the load test yourself:

```bash
cd backend
.venv/bin/locust -f tests/locustfile.py --host http://127.0.0.1:8000 \
    --users 20 --spawn-rate 4 --run-time 2m --headless WarmVillageUser
```

---

## Tests

```bash
cd backend && .venv/bin/python -m pytest tests/ -q     # 25 tests
cd frontend && node e2e-check.mjs /tmp/pond_shots      # drives a real browser
```

The Python tests use synthetic terrain with a known answer: a plane with a pit
of a known volume, a bowl whose storage gradient must equal its water spread, a
curve number worked by hand from the CGWB manual. The browser test searches a
village, draws a rectangle, analyses it, applies a budget and screenshots each
step.

---

## Limits

- A 30 m grid cannot resolve a 20 by 20 m farm pond. Where the terrain offers no
  usable basin the app says so and proposes an excavated pond sized to the
  MGNREGA model, which the budget filter then grows. Upload a surveyed contour
  file for small ponds.
- Copernicus GLO-30 is a surface model. Buildings and woodland are levelled
  against their surroundings, but dense canopy still adds error.
- Runoff methods disagree by up to a factor of two, and the initial abstraction
  ratio alone moves the answer by about 37 percent. Both figures are shown, and
  the ratio, curve number and seepage are editable.
- Village boundaries exist in OpenStreetMap for only about 57,600 of India's
  677,662 villages. Otherwise a circle is drawn and labelled as approximate; the
  polygon you draw is what gets analysed.
- Selections are capped at 25 km² and 4 million cells, coarsening the grid
  automatically rather than exhausting memory.

## Layout

```
backend/
  config.py              every tunable and default, with its source
  core/                  cache, circuit breakers, process pool
  services/              DEM, hydrology, storage, runoff, siting, earthwork
  routes/                villages, analyze, earthwork, health and metrics
  scripts/               one-time dataset builders
  tests/                 pytest suite and the Locust load test
frontend/src/
  components/            search, map, draw control, results, budget, charts
  services/api.js        API client
deploy/                  nginx config, systemd unit, local run script
```

## Licences

GeoNames CC BY 4.0, Copernicus DEM free for any use, OpenStreetMap ODbL,
GCN250 CC BY 4.0, IMD data free with citation. All fine for a course project;
check the terms before any commercial use.
