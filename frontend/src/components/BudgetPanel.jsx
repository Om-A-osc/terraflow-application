import { useState } from 'react';
import { runEarthwork } from '../services/api';

const PRESETS = [50000, 200000, 500000, 1000000, 2500000];

const rupees = (value) =>
  value == null ? '—' : `₹${Math.round(value).toLocaleString('en-IN')}`;
const cubic = (value) =>
  value == null ? '—' : `${Math.round(value).toLocaleString('en-IN')} m³`;

/**
 * The advanced budget filter.
 *
 * Give it a budget and the unit rates for excavating, hauling and placing soil,
 * and it reports how much more water the selected site could hold, what the
 * work would cost, and what the pond would look like afterwards.
 */
export default function BudgetPanel({ analysisId, site, onResult, result, disabled }) {
  const [budget, setBudget] = useState(500000);
  const [costs, setCosts] = useState({
    excavation_per_m3: 177.5,
    fill_per_m3: 197.9,
    haul_per_m3_per_50m: 33.7,
    borrow_per_m3: 700.5,
    manual_rates: false,
  });
  const [haulPlan, setHaulPlan] = useState(false);
  const [showRates, setShowRates] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  const setCost = (key, value) => setCosts((prev) => ({ ...prev, [key]: value }));

  const submit = async () => {
    if (!analysisId || !site) return;
    setBusy(true);
    setError(null);
    try {
      const data = await runEarthwork({
        analysisId,
        siteId: site.site_id,
        budget: Number(budget),
        costs,
        includeHaulPlan: haulPlan,
      });
      onResult(data);
    } catch (err) {
      setError(err.message);
      onResult(null);
    } finally {
      setBusy(false);
    }
  };

  if (!site) {
    return (
      <div className="panel-empty">Select a suggested site to work out what a budget would buy.</div>
    );
  }

  const storage = result?.storage;
  const design = result?.design;
  const cost = result?.cost;

  return (
    <div className="budget-panel">
      <p className="panel-intro">
        How much more water could <strong>site #{site.rank}</strong> hold if you paid to move soil?
      </p>

      <label className="field-label" htmlFor="budget-input">
        Budget (₹)
      </label>
      <input
        id="budget-input"
        type="number"
        className="text-input"
        min={0}
        step={10000}
        value={budget}
        onChange={(e) => setBudget(e.target.value)}
      />
      <div className="preset-row">
        {PRESETS.map((preset) => (
          <button
            key={preset}
            type="button"
            className={`chip ${Number(budget) === preset ? 'chip-active' : ''}`}
            onClick={() => setBudget(preset)}
          >
            {preset >= 100000 ? `₹${preset / 100000}L` : `₹${preset / 1000}k`}
          </button>
        ))}
      </div>

      <button
        type="button"
        className="link-button"
        onClick={() => setShowRates((v) => !v)}
        aria-expanded={showRates}
      >
        {showRates ? 'Hide' : 'Show'} unit rates
      </button>

      {showRates && (
        <div className="rates-grid">
          <label className="toggle-row">
            <input
              type="checkbox"
              checked={costs.manual_rates}
              onChange={(e) => setCost('manual_rates', e.target.checked)}
            />
            <span>
              Manual labour rates (MGNREGS, about ₹620/m³) instead of machine excavation
            </span>
          </label>
          <NumberField
            label="Excavate (₹/m³)"
            value={costs.excavation_per_m3}
            disabled={costs.manual_rates}
            onChange={(v) => setCost('excavation_per_m3', v)}
          />
          <NumberField
            label="Place and compact (₹/m³)"
            value={costs.fill_per_m3}
            onChange={(v) => setCost('fill_per_m3', v)}
          />
          <NumberField
            label="Haul (₹/m³ per 50 m)"
            value={costs.haul_per_m3_per_50m}
            onChange={(v) => setCost('haul_per_m3_per_50m', v)}
          />
          <NumberField
            label="Borrowed earth (₹/m³)"
            value={costs.borrow_per_m3}
            onChange={(v) => setCost('borrow_per_m3', v)}
          />
          <p className="rates-source">
            Defaults from CPWD Delhi Schedule of Rates 2023, chapter 2.
          </p>
        </div>
      )}

      <label className="toggle-row">
        <input type="checkbox" checked={haulPlan} onChange={(e) => setHaulPlan(e.target.checked)} />
        <span>Also work out where the soil should go (slower)</span>
      </label>

      <button
        type="button"
        className="btn btn-primary btn-full"
        onClick={submit}
        disabled={busy || disabled}
      >
        {busy ? 'Working it out…' : 'Apply budget'}
      </button>

      {error && <div className="error-banner">{error}</div>}

      {result && storage && (
        <div className="budget-result fade-in">
          <div className="headline-stat">
            <div className="headline-value">+{cubic(storage.added_m3)}</div>
            <div className="headline-label">
              extra storage, between {cubic(storage.added_low_m3)} and {cubic(storage.added_high_m3)}
            </div>
          </div>

          <div className="stat-rows">
            <Row label="Storage now" value={cubic(storage.base_capacity_m3)} />
            <Row label="Storage after the work" value={cubic(storage.new_capacity_m3)} strong />
            <Row label="Deepen the bed by" value={`${design.extra_depth_m} m`} />
            {design.designed ? (
              // An excavated pond is a designed hole: it grows down or out,
              // and there is no water level to raise and no bund to build
              <Row
                label="Pond after the work"
                value={`${design.new_side_m} m × ${design.new_side_m} m · ${design.new_depth_m} m deep`}
              />
            ) : (
              <Row label="Raise the water level by" value={`${design.spill_raise_m} m`} />
            )}
            <Row label="Soil to excavate" value={cubic(design.excavation_m3)} />
            {!design.designed && (
              <Row label="Bund to build" value={`${cubic(design.bund_volume_m3)} · ${design.bund_max_height_m} m high`} />
            )}
          </div>

          <div className="cost-breakdown">
            <div className="chart-title">Where the money goes</div>
            <Row label="Excavation" value={rupees(cost.excavation)} />
            <Row label="Placing and compacting" value={rupees(cost.placement)} />
            <Row label="Haulage" value={rupees(cost.haulage)} />
            {cost.borrow > 0 && <Row label="Borrowed earth" value={rupees(cost.borrow)} />}
            {cost.disposal > 0 && <Row label="Disposing of surplus" value={rupees(cost.disposal)} />}
            <Row label="Total" value={rupees(cost.total)} strong />
            <Row
              label="Cost per m³ of new storage"
              value={result.cost_per_m3_stored ? `₹${result.cost_per_m3_stored}` : '—'}
            />
          </div>

          {result.balanced_design?.added_storage_m3 > 0 && (
            <p className="hint">
              A balanced design that digs exactly as much soil as the bund needs would add{' '}
              {cubic(result.balanced_design.added_storage_m3)} for {rupees(result.balanced_design.cost)},
              buying and dumping nothing.
            </p>
          )}

          {result.haul_plan?.solved && (
            <p className="hint">
              Haul plan: {result.haul_plan.arrows.length} soil movements, average distance{' '}
              {result.haul_plan.mean_haul_m} m. The yellow lines on the map show them.
            </p>
          )}

          {!result.within_budget && (
            <div className="warn-banner">This is the cheapest workable design and it still exceeds the budget.</div>
          )}

          {result.warnings?.map((warning) => (
            <div key={warning} className="warn-banner">
              {warning}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function Row({ label, value, strong }) {
  return (
    <div className={`stat-row ${strong ? 'strong' : ''}`}>
      <span className="label">{label}</span>
      <span className="value">{value}</span>
    </div>
  );
}

function NumberField({ label, value, onChange, disabled }) {
  return (
    <label className="number-field">
      <span>{label}</span>
      <input
        type="number"
        min={0}
        step={0.5}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </label>
  );
}
