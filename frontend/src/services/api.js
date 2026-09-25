/**
 * API client for TerraFlow.
 *
 * The base URL is relative by default so the production build can be served
 * from the same nginx that proxies /api to the compute nodes.  In development
 * Vite proxies /api to the local backend (see vite.config.js).
 */
import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_BASE ?? '/api';

const client = axios.create({
  baseURL: API_BASE,
  timeout: 120000,
});

/** Turn an axios failure into a message worth showing a user. */
function describe(error) {
  if (axios.isCancel?.(error) || error.code === 'ERR_CANCELED') return null;
  const detail = error.response?.data?.detail;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail) && detail.length) {
    return detail.map((d) => d.msg ?? JSON.stringify(d)).join('; ');
  }
  if (error.code === 'ECONNABORTED') {
    return 'The request timed out. Try a smaller area.';
  }
  if (error.response?.status === 504) {
    return 'The analysis took too long. Select a smaller area.';
  }
  if (!error.response) {
    return 'Cannot reach the analysis server. Is the backend running?';
  }
  return error.message || 'Something went wrong';
}

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

async function call(promise) {
  try {
    const response = await promise;
    return response.data;
  } catch (error) {
    const message = describe(error);
    if (message === null) throw error; // cancelled on purpose
    throw new ApiError(message, error.response?.status);
  }
}

// ── Villages ────────────────────────────────────────────────────────────────

export function searchVillages(q, center, limit = 8, signal) {
  const params = { q, limit };
  if (center) {
    params.lat = center.lat;
    params.lon = center.lng ?? center.lon;
  }
  return call(client.get('/villages', { params, signal, timeout: 15000 }));
}

export function getVillage(village) {
  const params = {};
  if (village.source !== 'geonames') {
    params.lat = village.lat;
    params.lon = village.lon;
    params.name = village.name;
  }
  return call(client.get(`/villages/${encodeURIComponent(village.id)}`, { params }));
}

export function warmVillage(village) {
  return call(
    client.post(`/villages/${encodeURIComponent(village.id)}/warm`, null, {
      params: { lat: village.lat, lon: village.lon },
      timeout: 15000,
    }),
  );
}

export function getJob(jobId) {
  return call(client.get(`/jobs/${jobId}`, { timeout: 10000 }));
}

// ── Analysis ────────────────────────────────────────────────────────────────

export function analyzeArea({ geometry, numSites = 5, demSource = 'copernicus', hydrology = {}, villageId }) {
  return call(
    client.post('/analyze', {
      geometry,
      num_sites: numSites,
      dem_source: demSource,
      hydrology,
      village_id: villageId ?? null,
    }),
  );
}

export function runEarthwork({ analysisId, siteId, budget, costs = {}, geometry = {}, includeHaulPlan = false }) {
  return call(
    client.post('/earthwork', {
      analysis_id: analysisId,
      site_id: siteId,
      budget,
      costs,
      geometry,
      include_haul_plan: includeHaulPlan,
    }),
  );
}

// ── Legacy contour upload (phase 2 demo) ────────────────────────────────────

export function analyzeContour(file, resolution = 15, numCandidates = 5, onProgress) {
  const formData = new FormData();
  // The backend parameter is named contour_map; sending "file" was the bug
  // that made every upload fail with 422.
  formData.append('contour_map', file);

  return call(
    client.post(`/analyzeContour?resolution=${resolution}&num_candidates=${numCandidates}`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 300000,
      onUploadProgress: (event) => {
        if (onProgress && event.total) {
          onProgress(Math.round((event.loaded / event.total) * 100));
        }
      },
    }),
  );
}

// ── System ──────────────────────────────────────────────────────────────────

export function getHealth() {
  return call(client.get('/../health', { timeout: 8000 }));
}

export default client;
