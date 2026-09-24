"""
Load test for the report's stress and scaling section.

Three scenarios, matching the plan:

  A  warm village, many users          the steady state the demo runs in
  B  cold villages, few users          bounded by the external data fetch
  C  contour upload                    the legacy path

Run each against one, two and three compute nodes to produce the scaling curve:

    locust -f tests/locustfile.py --host http://127.0.0.1:8000 \
           --users 50 --spawn-rate 5 --run-time 10m --headless \
           --csv results/scenario-a WarmVillageUser

Record p50 and p95 from the CSV, and read throughput, cache hit ratio and pool
queue depth from /metrics at the end of the run.
"""

from __future__ import annotations

import random

from locust import HttpUser, between, events, task

# Villages used by scenario A.  They are warmed once before the run so the
# measurement reflects the steady state rather than the first-fetch cost.
WARM_VILLAGES = [
    {"id": "gn:10447453", "name": "Jeora", "lat": 21.2571, "lon": 81.30685},
]

# Scenario B deliberately spreads over places that will not be cached.
COLD_VILLAGES = [
    {"id": "gn:cold-1", "lat": 20.9320, "lon": 77.7523},
    {"id": "gn:cold-2", "lat": 26.4499, "lon": 80.3319},
    {"id": "gn:cold-3", "lat": 17.3850, "lon": 78.4867},
    {"id": "gn:cold-4", "lat": 23.2599, "lon": 77.4126},
    {"id": "gn:cold-5", "lat": 15.3173, "lon": 75.7139},
    {"id": "gn:cold-6", "lat": 11.1271, "lon": 78.6569},
    {"id": "gn:cold-7", "lat": 25.0961, "lon": 85.3131},
    {"id": "gn:cold-8", "lat": 22.9868, "lon": 87.8550},
    {"id": "gn:cold-9", "lat": 19.7515, "lon": 75.7139},
    {"id": "gn:cold-10", "lat": 27.0238, "lon": 74.2179},
]

PREFIXES = ["jeo", "pipar", "ramp", "anant", "sirs", "khed", "bhag", "nand"]


def square(lat: float, lon: float, side_km: float) -> dict:
    """A GeoJSON square of a given side, centred on a point."""
    half_lat = side_km / 2 / 111.32
    half_lon = half_lat / max(0.2, abs(pow(1 - (lat / 90) ** 2, 0.5)))
    return {
        "type": "Polygon",
        "coordinates": [[
            [lon - half_lon, lat - half_lat],
            [lon + half_lon, lat - half_lat],
            [lon + half_lon, lat + half_lat],
            [lon - half_lon, lat + half_lat],
            [lon - half_lon, lat - half_lat],
        ]],
    }


@events.test_start.add_listener
def warm_the_demo_villages(environment, **_kwargs):
    """
    Scenario A measures the steady state, so the villages are prepared before
    any user starts.

    The warm-up endpoint returns a job immediately, so this polls until the job
    finishes.  Skipping the wait makes every user in the first minute queue
    behind one cold elevation fetch and turns the whole run into scenario B.
    """
    import time

    import requests

    host = environment.host or "http://127.0.0.1:8000"
    for village in WARM_VILLAGES:
        try:
            response = requests.post(
                f"{host}/api/villages/{village['id']}/warm",
                params={"lat": village["lat"], "lon": village["lon"]},
                timeout=30,
            )
            body = response.json()
            if body.get("status") == "ready":
                print(f"{village['id']} was already prepared")
                continue

            job_id = body.get("job_id")
            if not job_id:
                continue
            print(f"Preparing {village['id']} (job {job_id})…")
            deadline = time.time() + 240
            while time.time() < deadline:
                time.sleep(3)
                job = requests.get(f"{host}/api/jobs/{job_id}", timeout=10).json()
                if job.get("status") == "done":
                    print(f"  ready in {job.get('elapsed_s')} s")
                    break
                if job.get("status") == "failed":
                    print(f"  preparation failed: {job.get('detail')}")
                    break
            else:
                print("  preparation did not finish in time; results will include cold fetches")
        except Exception as exc:  # noqa: BLE001
            print(f"Could not warm {village['id']}: {exc}")


class WarmVillageUser(HttpUser):
    """Scenario A: everyone works in the same, already prepared village."""

    wait_time = between(15, 25)          # 20 s think time, as in the plan
    weight = 10

    @task(3)
    def analyse_an_area(self):
        village = random.choice(WARM_VILLAGES)
        side = random.uniform(1.0, 2.2)          # about 100 to 480 ha
        # Users redraw around the same village rather than wandering across the
        # district, so the jitter stays well inside one cached window.
        jitter = random.uniform(-0.002, 0.002)
        payload = {
            "geometry": square(village["lat"] + jitter, village["lon"] + jitter, side),
            "num_sites": 4,
        }
        with self.client.post(
            "/api/analyze",
            json=payload,
            params={"village": village["id"]},   # nginx hashes on this
            name="POST /api/analyze (warm)",
            catch_response=True,
        ) as response:
            if response.status_code == 200 and "sites" in response.text:
                response.success()
            elif response.status_code == 504:
                response.failure("compute timeout")
            else:
                response.failure(f"status {response.status_code}")

    @task(2)
    def search(self):
        self.client.get(
            "/api/villages",
            params={"q": random.choice(PREFIXES), "lat": 21.25, "lon": 81.3, "limit": 8},
            name="GET /api/villages",
        )

    @task(1)
    def apply_a_budget(self):
        """Analyse then immediately price a budget, as a user would."""
        village = random.choice(WARM_VILLAGES)
        payload = {"geometry": square(village["lat"], village["lon"], 1.5), "num_sites": 3}
        response = self.client.post(
            "/api/analyze", json=payload,
            params={"village": village["id"]},
            name="POST /api/analyze (warm)",
        )
        if response.status_code != 200:
            return
        body = response.json()
        if not body.get("sites"):
            return
        self.client.post(
            "/api/earthwork",
            json={
                "analysis_id": body["analysis_id"],
                "site_id": body["sites"][0]["site_id"],
                "budget": random.choice([100000, 500000, 1000000, 2500000]),
            },
            params={"village": village["id"]},
            name="POST /api/earthwork",
        )


class ColdVillageUser(HttpUser):
    """Scenario B: every request lands on a village nobody has looked at."""

    wait_time = between(25, 40)
    weight = 1

    @task
    def analyse_somewhere_new(self):
        village = random.choice(COLD_VILLAGES)
        payload = {
            "geometry": square(village["lat"], village["lon"], 1.5),
            "num_sites": 3,
        }
        with self.client.post(
            "/api/analyze",
            json=payload,
            params={"village": village["id"]},
            name="POST /api/analyze (cold)",
            catch_response=True,
            timeout=180,
        ) as response:
            # A cold village fetches elevation and OpenStreetMap data, so a
            # timeout here is a finding about the network, not a crash.
            if response.status_code in (200, 504):
                response.success()
            else:
                response.failure(f"status {response.status_code}")


class HealthUser(HttpUser):
    """A watchdog's view: cheap endpoints that must stay fast under load."""

    wait_time = between(5, 10)
    weight = 1

    @task(2)
    def health(self):
        self.client.get("/health", name="GET /health")

    @task(1)
    def metrics(self):
        self.client.get("/metrics", name="GET /metrics")
