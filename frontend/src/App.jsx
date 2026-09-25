import { useCallback, useEffect, useRef, useState } from 'react';
import './index.css';
import 'leaflet/dist/leaflet.css';

import BudgetPanel from './components/BudgetPanel';
import FileUpload from './components/FileUpload';
import MapViewer from './components/MapViewer';
import ResultsPanel from './components/ResultsPanel';
import VillageSearch from './components/VillageSearch';
import { analyzeArea, getJob, getVillage, warmVillage } from './services/api';

const DEFAULT_LAYERS = {
  sites: true,
  catchment: true,
  // The chosen pond's own catchment is a second, smaller boundary in the same
  // area as the first.  Showing both at once was the main source of confusion,
  // so it starts off and the user turns it on when they want it.
  siteCatchment: false,
  streams: true,
  constraints: false,
  village: true,
  // Everything outside the drawn area is dimmed, so the answer reads against
  // the ground it is about.
  focus: true,
};

/**
 * The TerraFlow mark: contour rings with a watercourse cutting through them.
 *
 * Inline rather than an <img> so the rings can draw themselves in on load and
 * so it inherits the page's colours if the theme ever changes.
 */
function BrandMark() {
  return (
    <svg
      className="brand-mark"
      viewBox="0 0 32 32"
      role="img"
      aria-label="TerraFlow"
      focusable="false"
    >
      <rect width="32" height="32" rx="7" fill="#0C1317" />
      <g fill="none" stroke="#E0A458" strokeLinecap="round">
        <path className="ring ring-1" d="M6.5 22.5c2.2-4.4 5.2-7.4 9.5-7.4s7.3 3 9.5 7.4"
          strokeWidth="2.1" opacity="0.95" />
        <path className="ring ring-2" d="M9 26.2c1.6-3.1 3.9-5.2 7-5.2s5.4 2.1 7 5.2"
          strokeWidth="1.9" opacity="0.6" />
        <path className="ring ring-3" d="M10 12.2c1.4-2.7 3.4-4.6 6-4.6s4.6 1.9 6 4.6"
          strokeWidth="1.9" opacity="0.45" />
      </g>
      <path
        className="drop"
        d="M16 4.5c0 6.2-3.4 8.6-3.4 13.2A3.4 3.4 0 0 0 16 21a3.4 3.4 0 0 0 3.4-3.3c0-4.6-3.4-7-3.4-13.2Z"
        fill="#48B7A8"
      />
    </svg>
  );
}

export default function App() {
  const [village, setVillage] = useState(null);
  const [villageBoundary, setVillageBoundary] = useState(null);
  const [seedGeometry, setSeedGeometry] = useState(null);
  const [preparing, setPreparing] = useState(null);

  const [geometry, setGeometry] = useState(null);
  const [areaM2, setAreaM2] = useState(0);
  const [numSites, setNumSites] = useState(4);

  const [analysis, setAnalysis] = useState(null);
  const [selectedSiteId, setSelectedSiteId] = useState(null);
  const [earthwork, setEarthwork] = useState(null);

  const [busy, setBusy] = useState(false);
  const [stage, setStage] = useState('');
  const [error, setError] = useState(null);
  const [notice, setNotice] = useState(null);
  const [tab, setTab] = useState('results');
  const [layers, setLayers] = useState(DEFAULT_LAYERS);
  const toggleLayer = useCallback(
    (key) => setLayers((prev) => ({ ...prev, [key]: !prev[key] })),
    [],
  );
  const [mapCenter, setMapCenter] = useState(null);
  const [flyTo, setFlyTo] = useState(null);

  const pollRef = useRef(null);

  useEffect(() => () => clearInterval(pollRef.current), []);

  // ── Village selection ─────────────────────────────────────────────────────

  const handleVillage = useCallback(async (picked) => {
    setVillage(picked);
    setError(null);
    setNotice(null);
    setFlyTo({ lat: picked.lat, lon: picked.lon });

    try {
      const detail = await getVillage(picked);
      setVillageBoundary(detail.boundary ?? null);
      if (detail.boundary?.geometry) setSeedGeometry(detail.boundary.geometry);
      if (detail.boundary?.properties?.source === 'synthesized') {
        setNotice(
          'No official boundary exists for this village, so the dashed circle is an approximation. Draw the area you actually care about.',
        );
      }
    } catch {
      setVillageBoundary(null);
    }

    // Prepare the terrain in the background while the user looks around
    try {
      const warm = await warmVillage(picked);
      if (warm.status === 'ready') {
        setPreparing(null);
        return;
      }
      if (warm.job_id) {
        setPreparing('Preparing elevation and map data…');
        clearInterval(pollRef.current);
        pollRef.current = setInterval(async () => {
          try {
            const job = await getJob(warm.job_id);
            if (job.status === 'done') {
              clearInterval(pollRef.current);
              setPreparing(null);
            } else if (job.status === 'failed') {
              clearInterval(pollRef.current);
              setPreparing(null);
            }
          } catch {
            clearInterval(pollRef.current);
            setPreparing(null);
          }
        }, 3000);
      }
    } catch {
      setPreparing(null);
    }
  }, []);

  // ── Drawing ───────────────────────────────────────────────────────────────

  const handleGeometry = useCallback((geom, area) => {
    setGeometry(geom);
    setAreaM2(area ?? 0);
    setError(null);
    if (geom) setNotice(null);
  }, []);

  const handleInvalid = useCallback((message) => {
    setError(message);
  }, []);

  // ── Analysis ──────────────────────────────────────────────────────────────

  const runAnalysis = useCallback(async () => {
    if (!geometry) {
      setError('Draw an area on the map first, using the polygon or rectangle tool.');
      return;
    }
    setBusy(true);
    setError(null);
    setEarthwork(null);
    setStage(preparing ? 'Fetching elevation and map data…' : 'Analysing the terrain…');

    try {
      const data = await analyzeArea({
        geometry,
        numSites,
        villageId: village?.id,
      });
      setAnalysis(data);
      setSelectedSiteId(data.sites?.[0]?.site_id ?? null);
      setTab('results');
      if (!data.sites?.length) {
        setNotice('No workable pond site in that area. Try somewhere with a drainage line running through it.');
      }
    } catch (err) {
      setError(err.message);
      setAnalysis(null);
    } finally {
      setBusy(false);
      setStage('');
    }
  }, [geometry, numSites, village, preparing]);

  const handleContourResult = useCallback((data) => {
    setNotice(
      `Contour file analysed: ${data.candidates?.length ?? 0} candidate sites from ${data.metadata?.contour_count ?? 0} contour lines. ` +
        'This is the original upload path; village search gives the full analysis.',
    );
  }, []);

  const selectedSite = analysis?.sites?.find((s) => s.site_id === selectedSiteId) ?? null;
  const areaHa = areaM2 / 10000;

  return (
    <div className="app">
      <main className="app-body">
        <aside className="sidebar">
          {/* A drawing-sheet title block: the mark, the name, and the one
              control that starts everything.  There is no top bar, so the map
              runs the full height of the window. */}
          <header className="masthead">
            <div className="brand">
              <BrandMark />
              <div>
                <h1>
                  Terra<em>Flow</em>
                </h1>
                <p className="tagline">Read the land, find the water</p>
              </div>
            </div>
            <VillageSearch mapCenter={mapCenter} onSelect={handleVillage} disabled={busy} />
          </header>

          <section className="sidebar-section">
            <h2>1 · Choose an area</h2>
            {village ? (
              <div className="village-chip">
                <strong>{village.name}</strong>
                <span>
                  {[village.subdistrict, village.district, village.state].filter(Boolean).join(' · ')}
                </span>
              </div>
            ) : (
              <p className="hint">Search for a village above, or just pan the map to where you know.</p>
            )}

            {preparing && (
              <div className="prepare-banner">
                <span className="spinner-inline" aria-hidden="true" />
                {preparing}
              </div>
            )}

            <p className="hint">
              Use the polygon or rectangle tool on the left of the map to outline the land to study.
            </p>

            <div className="selection-state">
              {geometry ? (
                <>
                  <span className="dot dot-ok" /> {areaHa.toFixed(1)} ha selected
                </>
              ) : (
                <>
                  <span className="dot" /> nothing selected yet
                </>
              )}
            </div>

            <label className="field-label" htmlFor="num-sites">
              Suggestions to return
            </label>
            <input
              id="num-sites"
              type="range"
              min={1}
              max={10}
              value={numSites}
              onChange={(e) => setNumSites(Number(e.target.value))}
            />
            <div className="range-value">{numSites} sites</div>

            <button
              type="button"
              className="btn btn-primary btn-full"
              onClick={runAnalysis}
              disabled={busy || !geometry}
            >
              {busy ? 'Analysing…' : 'Analyse this area'}
            </button>

            {error && <div className="error-banner">{error}</div>}
            {notice && <div className="notice-banner">{notice}</div>}
          </section>

          {analysis && (
            <>
              <nav className="tabs" role="tablist">
                <button
                  type="button"
                  role="tab"
                  aria-selected={tab === 'results'}
                  className={tab === 'results' ? 'tab active' : 'tab'}
                  onClick={() => setTab('results')}
                >
                  Results
                </button>
                <button
                  type="button"
                  role="tab"
                  aria-selected={tab === 'budget'}
                  className={tab === 'budget' ? 'tab active' : 'tab'}
                  onClick={() => setTab('budget')}
                >
                  Budget filter
                </button>
              </nav>

              <div className="sidebar-scroll">
                {tab === 'results' && (
                  <ResultsPanel
                    analysis={analysis}
                    selectedSiteId={selectedSiteId}
                    onSelectSite={setSelectedSiteId}
                    showSiteCatchment={layers.siteCatchment}
                    onToggleSiteCatchment={() => toggleLayer('siteCatchment')}
                  />
                )}
                {tab === 'budget' && (
                  <BudgetPanel
                    analysisId={analysis.analysis_id}
                    site={selectedSite}
                    result={earthwork}
                    onResult={setEarthwork}
                    disabled={busy}
                  />
                )}
              </div>
            </>
          )}

          {!analysis && (
            <section className="sidebar-section">
              <h2>Have a contour survey?</h2>
              <p className="hint">
                A surveyed contour file resolves small ponds far better than the 30 m global
                elevation model can.
              </p>
              <FileUpload onResult={handleContourResult} />
            </section>
          )}
        </aside>

        <div className="map-pane">
          {busy && (
            <div className="loading-overlay">
              <div className="spinner" />
              <div className="loading-text">{stage || 'Working…'}</div>
              <div className="loading-sub">
                The first area in a village takes longer while the data is fetched
              </div>
            </div>
          )}
          <MapViewer
            analysis={analysis}
            earthwork={earthwork}
            selectedSiteId={selectedSiteId}
            onSelectSite={setSelectedSiteId}
            onGeometryChange={handleGeometry}
            onInvalidGeometry={handleInvalid}
            onMapMove={setMapCenter}
            flyTo={flyTo}
            villageBoundary={villageBoundary}
            seedGeometry={seedGeometry}
            layers={layers}
            onToggleLayer={toggleLayer}
          />
        </div>
      </main>
    </div>
  );
}
