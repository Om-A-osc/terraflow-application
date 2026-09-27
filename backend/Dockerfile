# TerraFlow backend for Railway (or any container host).
#
# The gazetteer is built during the image build from the GeoNames India dump,
# so the container starts with local village search and no data volume.  The
# disk cache lives inside the container and is rebuilt on each deploy, which
# is fine: the first analysis in a village is cold, later ones are warm.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# rasterio's wheel links against libexpat, which the slim image leaves out
RUN apt-get update \
 && apt-get install -y --no-install-recommends libexpat1 \
 && rm -rf /var/lib/apt/lists/*

# Dependencies first, so code changes do not re-install them
COPY requirements.txt .
RUN grep -v -E '^(locust|pytest|httpx)' requirements.txt > /tmp/req.txt \
 && pip install -r /tmp/req.txt

COPY . .

# Village gazetteer: about 16 MB downloaded, 100 MB built
RUN python scripts/build_gazetteer.py

# A container host provides PORT; the app's own setting is the fallback.
# One uvicorn process; the compute pool inside it carries the parallelism.
ENV POND_POOL_WORKERS=3
EXPOSE 8000
CMD ["sh", "-c", "python -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --log-level info"]
