/**
 * API client for TerraFlow.
 *
 * The base URL is relative by default so the production build can be served
 * from the same nginx that proxies /api to the compute nodes.  In development
 * Vite proxies /api to the local backend (see vite.config.js).
 */
import axios from 'axios';

/**
 * Backends in order of preference.
 *
 * VITE_API_BASES is a comma-separated list, e.g. the backend on this laptop,
 * then the cloud one, then the lab machine.  On first use every candidate's
 * /health is probed at once and the first in list order that answers is
 * used; if a request later fails to connect at all, the next one is tried.
 * With a single entry (the default, a relative /api behind a proxy) nothing
 * is probed.
 */
const CANDIDATES = (import.meta.env.VITE_API_BASES ?? import.meta.env.VITE_API_BASE ?? '/api')
  .split(',')
  .map((entry) => entry.trim())
  .filter(Boolean);
const PROBE_TIMEOUT_MS = 5000;

const client = axios.create({
  baseURL: CANDIDATES[0],
  timeout: 120000,
});

let chosen = null;
let choosing = null;
const listeners = new Set();

function labelFor(base) {
  if (base.startsWith('/')) return 'this server';
  if (/^https?:\/\/(127\.0\.0\.1|localhost)/.test(base)) return 'your laptop';
  if (/onrender\.com/.test(base)) return 'Render';
  if (/^http:\/\/10\./.test(base)) return 'lab machine';
  try {
    return new URL(base).host;
  } catch {
    return base;
  }
}

async function probe(base) {
  const health = `${base.replace(/\/api\/?$/, '')}/health`;
  const response = await axios.get(health, { timeout: PROBE_TIMEOUT_MS });
  return response.status === 200;
}

async function pickBackend(exclude = new Set()) {
  const order = CANDIDATES.filter((base) => !exclude.has(base));
  let base = order[0] ?? CANDIDATES[0];
  let reachable = true;
  if (CANDIDATES.length > 1) {
    const results = await Promise.allSettled(order.map(probe));
    const index = results.findIndex((r) => r.status === 'fulfilled' && r.value);
    reachable = index >= 0;
    if (reachable) base = order[index];
  }
  chosen = { base, label: labelFor(base), reachable };
  client.defaults.baseURL = base;
  listeners.forEach((listener) => listener(chosen));
  return chosen;
}

/** The backend in use, probing the candidates the first time. */
export function ensureBackend() {
  if (chosen) return Promise.resolve(chosen);
  if (!choosing) {
    choosing = pickBackend().finally(() => {
      choosing = null;
    });
  }
  return choosing;
}

/** Subscribe to backend changes; the callback also fires at once if one is chosen. */
export function onBackendChange(listener) {
  listeners.add(listener);
  if (chosen) listener(chosen);
  return () => listeners.delete(listener);
}

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

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * A connection that never reached the server is retried, twice.
 *
 * The lab host's port forwarding drops the odd connection outright (the
 * browser reports ERR_CONNECTION_TIMED_OUT while the next attempt answers in
 * milliseconds), so a request with no response at all is worth repeating.
 * Every endpoint here is safe to repeat: reads are reads, and the two POSTs
 * are deterministic and cached on the server by their inputs.
 */
async function call(makeRequest, retries = 2) {
  const request = typeof makeRequest === 'function' ? makeRequest : () => makeRequest;
  await ensureBackend();
  for (let attempt = 0; ; attempt += 1) {
    try {
      const response = await request();
      return response.data;
    } catch (error) {
      const message = describe(error);
      if (message === null) throw error; // cancelled on purpose
      const neverConnected = !error.response && error.code !== 'ECONNABORTED';
      if (neverConnected && attempt < retries) {
        // The backend may have gone away: fall through to the next candidate
        if (CANDIDATES.length > 1 && chosen) await pickBackend(new Set([chosen.base]));
        await sleep(600 * (attempt + 1));
        continue;
      }
      throw new ApiError(message, error.response?.status);
    }
  }
}

// ── Villages ────────────────────────────────────────────────────────────────

export function searchVillages(q, center, limit = 8, signal) {
  const params = { q, limit };
  if (center) {
    params.lat = center.lat;
    params.lon = center.lng ?? center.lon;
  }
  return call(() => client.get('/villages', { params, signal, timeout: 15000 }));
}

export function getVillage(village) {
  const params = {};
  if (village.source !== 'geonames') {
    params.lat = village.lat;
    params.lon = village.lon;
    params.name = village.name;
  }
  return call(() => client.get(`/villages/${encodeURIComponent(village.id)}`, { params }));
}

export function warmVillage(village) {
  return call(() =>
    client.post(`/villages/${encodeURIComponent(village.id)}/warm`, null, {
      params: { lat: village.lat, lon: village.lon },
      timeout: 15000,
    }),
  );
}

export function getJob(jobId) {
  return call(() => client.get(`/jobs/${jobId}`, { timeout: 10000 }));
}

// ── Analysis ────────────────────────────────────────────────────────────────

export function analyzeArea({ geometry, numSites = 5, demSource = 'copernicus', hydrology = {}, villageId }) {
  return call(() =>
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
  return call(() =>
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

  return call(() =>
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
  return call(() => client.get('/../health', { timeout: 8000 }));
}

export default client;
