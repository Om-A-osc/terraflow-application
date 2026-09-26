import { MonthlyStorageChart, RainfallChart, StageStorageChart } from './StageStorageChart';
import { siteColor } from './mapTheme';

const cubic = (value) =>
  value == null ? '—' : `${Math.round(value).toLocaleString('en-IN')} m³`;

/**
 * Everything the analysis found, in the order a reader needs it: what the
 * selection drains, then the ranked sites, then the assumptions behind the
 * numbers.
 */
export default function ResultsPanel({
  analysis, selectedSiteId, onSelectSite,
  showSiteCatchment, onToggleSiteCatchment,
}) {
  if (!analysis) return null;

  const { catchment, rainfall, runoff, sites, data_quality: quality } = analysis;
  const scs = runoff?.scs_cn ?? {};

  return (
    <div className="results fade-in">
      <section className="sidebar-section">
        <h2>What this area drains</h2>
        <div className="stat-grid">
          <Stat label="Selected" value={`${analysis.selection.area_ha.toFixed(1)}`} unit="ha" />
          <Stat label="Catchment" value={catchment.area_ha.toFixed(1)} unit="ha" />
          <Stat label="Rainfall a year" value={rainfall?.mean_annual_mm ?? '—'} unit="mm" />
          <Stat label="Curve number" value={runoff?.curve_number ?? '—'} />
          <Stat
            label="Runoff a year"
            value={scs.mean_m3 ? Math.round(scs.mean_m3 / 1000) : '—'}
            unit="k m³"
          />
          <Stat
            label="3 years in 4"
            value={scs.dependable_m3 ? Math.round(scs.dependable_m3 / 1000) : '—'}
            unit="k m³"
          />
        </div>

        {catchment.truncated && (
          <div className="warn-banner">
            The catchment runs past the edge of the analysed window, so its area and the runoff are
            lower bounds.
          </div>
        )}

        {runoff?.strange_cross_check?.mean_m3 > 0 && (
          <p className="hint">
            Curve-number runoff is {cubic(scs.mean_m3)} a year. Strange&apos;s table, the other method
            Indian practice uses, gives {cubic(runoff.strange_cross_check.mean_m3)}. Treat the truth
            as lying between them.
          </p>
        )}

        <RainfallChart rainfall={rainfall} />
      </section>

      <section className="sidebar-section">
        <h2>Suggested ponds</h2>
        {sites.length > 0 && (
          <p className="hint">
            Pick one to draw it on the map. The numbered pins show where the others are.
          </p>
        )}
        {sites.length === 0 ? (
          <p className="panel-empty">
            No workable site in this area. Everything is either too steep, already water, or built on.
            Try a different area.
          </p>
        ) : (
          <div className="site-list">
            {sites.map((site, index) => (
              <SiteCard
                key={site.site_id}
                site={site}
                index={index}
                active={site.site_id === selectedSiteId}
                onSelect={() => onSelectSite(site.site_id)}
                showCatchment={showSiteCatchment}
                onToggleCatchment={onToggleSiteCatchment}
              />
            ))}
          </div>
        )}
      </section>

      <section className="sidebar-section">
        <h2>How reliable is this?</h2>
        <div className="quality-block">
          <Row label="Elevation" value={prettySource(quality.dem_source)} />
          <Row label="Rainfall" value={prettySource(quality.rainfall_source)} />
          <Row label="Curve number" value={prettySource(quality.cn_source)} />
          <Row label="Map features" value={quality.osm_available ? 'OpenStreetMap' : 'unavailable'} />
          {quality.notes?.map((note) => (
            <p key={note} className="quality-note">
              {note}
            </p>
          ))}
        </div>
      </section>

      <section className="sidebar-section">
        <h2>Assumptions</h2>
        <div className="quality-block">
          <Row label="Initial abstraction (Ia/S)" value={analysis.assumptions.lambda_ia} />
          <Row label="Curve number" value={analysis.assumptions.curve_number} />
          <Row label="Seepage" value={`${analysis.assumptions.seepage_mm_day} mm/day`} />
          <Row label="Open-water coefficient" value={analysis.assumptions.open_water_kc} />
          <Row label="Freeboard" value={`${analysis.assumptions.freeboard_m} m`} />
          <Row
            label="Elevation uncertainty"
            value={`±${analysis.assumptions.dem_vertical_uncertainty_m} m`}
          />
          <Row label="Grid resolution" value={`${analysis.grid.resolution_m} m`} />
          <Row label="Analysis took" value={`${analysis.timings.total_s} s`} />
        </div>
      </section>
    </div>
  );
}

function SiteCard({ site, index, active, onSelect, showCatchment, onToggleCatchment }) {
  const y = site.yield ?? {};
  const storage = site.storage ?? {};
  return (
    <article
      className={`site-card ${active ? 'active' : ''}`}
      onClick={onSelect}
      style={{ borderLeftColor: siteColor(index, active) }}
    >
      <header className="site-head">
        <span className="site-rank" style={{ background: siteColor(index, active) }}>
          {site.rank}
        </span>
        <div className="site-title">
          <strong>{site.structure?.label ?? site.kind}</strong>
          <span className="site-sub">{describeKind(site.kind)}</span>
        </div>
        <span className={`confidence confidence-${site.confidence_label}`}>
          {site.confidence_label}
        </span>
      </header>

      <div className="site-stats">
        <Stat small label="Holds" value={Math.round(storage.capacity_m3).toLocaleString('en-IN')} unit="m³" />
        <Stat small label="Depth" value={storage.max_depth_m} unit="m" />
        <Stat small label="Catchment" value={site.catchment.area_ha.toFixed(1)} unit="ha" />
        <Stat
          small
          label="Collects a year"
          value={Math.round(y.harvestable_mean_m3 ?? 0).toLocaleString('en-IN')}
          unit="m³"
        />
        <Stat small label="Holds water" value={y.mean_wet_months ?? '—'} unit="months" />
        <Stat small label="Slope" value={site.slope_pct} unit="%" />
      </div>

      {storage.note && <p className="site-note">{storage.note}</p>}

      {active && (
        <div className="site-detail">
          <p className="hint">
            Capacity is between {cubic(storage.capacity_low_m3)} and {cubic(storage.capacity_high_m3)}
            , allowing for the accuracy of the elevation data.
          </p>
          <StageStorageChart curve={site.stage_storage} />
          <MonthlyStorageChart yieldBlock={y} capacity={storage.capacity_m3} />
          <button
            type="button"
            className="link-button catchment-toggle"
            onClick={(event) => {
              event.stopPropagation();
              onToggleCatchment?.();
            }}
          >
            {showCatchment ? 'Hide' : 'Show'} the {site.catchment.area_ha.toFixed(1)} ha that
            drains into this pond
          </button>

          <div className="coords">
            {site.location.lat.toFixed(5)}°N, {site.location.lon.toFixed(5)}°E
          </div>
        </div>
      )}
    </article>
  );
}

function describeKind(kind) {
  if (kind === 'depression') return 'natural hollow';
  if (kind === 'embankment') return 'bund across a drainage line';
  if (kind === 'dugout') return 'excavated pond';
  return kind;
}

function prettySource(source) {
  const names = {
    copernicus_glo30: 'Copernicus 30 m',
    terrain_tiles: 'SRTM terrain tiles',
    contour_kml: 'uploaded contour survey',
    'imd_0.25deg': 'IMD gauge grid',
    nasa_power: 'NASA POWER reanalysis',
    gcn250: 'GCN250 global grid',
    osm_landuse_table: 'land use plus CGWB table',
    default_table: 'CGWB default table',
    user_override: 'your override',
  };
  return names[source] ?? source ?? 'unknown';
}

function Stat({ label, value, unit, small }) {
  return (
    <div className={small ? 'stat-mini' : 'stat-card'}>
      <div className="stat-label">{label}</div>
      <div className="stat-value">
        {value}
        {unit ? <span className="stat-unit">{unit}</span> : null}
      </div>
    </div>
  );
}

function Row({ label, value }) {
  return (
    <div className="stat-row">
      <span className="label">{label}</span>
      <span className="value">{value}</span>
    </div>
  );
}
