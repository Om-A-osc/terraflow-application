import {
  Area, AreaChart, CartesianGrid, Line, LineChart, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from 'recharts';

// Charts sit inside the panel, so they use the panel's palette
const AXIS = '#7f8a91';
const GRID = 'rgba(20,27,31,0.09)';
const RAIN = '#1d8478';

const tooltipStyle = {
  background: '#fcfaf5',
  border: '1px solid rgba(182,121,43,0.45)',
  borderRadius: 8,
  fontSize: 12,
  fontFamily: "'IBM Plex Mono', ui-monospace, monospace",
  color: '#141b1f',
};

/** Storage against water level, the curve every volume figure is read from. */
export function StageStorageChart({ curve, color = '#b6792b' }) {
  if (!curve?.level_m?.length || curve.level_m.length < 2) return null;

  const bed = curve.bed_level_m ?? curve.level_m[0];
  const data = curve.level_m.map((level, index) => ({
    depth: Number((level - bed).toFixed(2)),
    volume: curve.volume_m3[index],
    area: curve.area_m2[index],
  }));

  return (
    <div className="chart-block">
      <div className="chart-title">Storage against depth</div>
      <ResponsiveContainer width="100%" height={150}>
        <AreaChart data={data} margin={{ top: 6, right: 8, bottom: 2, left: 0 }}>
          <CartesianGrid stroke={GRID} vertical={false} />
          <XAxis
            dataKey="depth"
            tick={{ fill: AXIS, fontSize: 10, fontFamily: "'IBM Plex Mono', monospace" }}
            stroke={AXIS}
            label={{ value: 'Depth (m)', position: 'insideBottom', offset: -4, fill: AXIS, fontSize: 10 }}
            height={30}
          />
          <YAxis
            tick={{ fill: AXIS, fontSize: 10, fontFamily: "'IBM Plex Mono', monospace" }}
            stroke={AXIS}
            width={52}
            tickFormatter={(v) => (v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v)}
          />
          <Tooltip
            contentStyle={tooltipStyle}
            formatter={(value, name) =>
              name === 'volume'
                ? [`${Math.round(value).toLocaleString()} m³`, 'Storage']
                : [`${Math.round(value).toLocaleString()} m²`, 'Water spread']
            }
            labelFormatter={(d) => `Depth ${d} m`}
          />
          <Area type="monotone" dataKey="volume" stroke={color} fill={color} fillOpacity={0.22} strokeWidth={2} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

/** Average water held through the year, June to May. */
export function MonthlyStorageChart({ yieldBlock, capacity, color = '#b6792b' }) {
  const values = yieldBlock?.monthly_storage_m3;
  const labels = yieldBlock?.month_labels;
  if (!values?.length || !labels?.length) return null;

  const data = labels.map((month, index) => ({
    month,
    storage: values[index] ?? 0,
  }));

  return (
    <div className="chart-block">
      <div className="chart-title">
        Water held through an average year
        {capacity ? <span className="chart-note"> · capacity {Math.round(capacity).toLocaleString()} m³</span> : null}
      </div>
      <ResponsiveContainer width="100%" height={140}>
        <LineChart data={data} margin={{ top: 6, right: 8, bottom: 2, left: 0 }}>
          <CartesianGrid stroke={GRID} vertical={false} />
          <XAxis dataKey="month" tick={{ fill: AXIS, fontSize: 10, fontFamily: "'IBM Plex Mono', monospace" }} stroke={AXIS} interval={0} height={24} />
          <YAxis
            tick={{ fill: AXIS, fontSize: 10, fontFamily: "'IBM Plex Mono', monospace" }}
            stroke={AXIS}
            width={52}
            tickFormatter={(v) => (v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v)}
          />
          <Tooltip
            contentStyle={tooltipStyle}
            formatter={(value) => [`${Math.round(value).toLocaleString()} m³`, 'Stored']}
          />
          <Line type="monotone" dataKey="storage" stroke={color} strokeWidth={2} dot={{ r: 2.5 }} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

/** Long-term monthly rainfall for the village. */
export function RainfallChart({ rainfall }) {
  const monthly = rainfall?.monthly_mm;
  if (!monthly?.length) return null;
  const labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  const data = labels.map((month, index) => ({ month, rain: monthly[index] ?? 0 }));

  return (
    <div className="chart-block">
      <div className="chart-title">
        Average monthly rainfall
        <span className="chart-note"> · {rainfall.years} years</span>
      </div>
      <ResponsiveContainer width="100%" height={130}>
        <AreaChart data={data} margin={{ top: 6, right: 8, bottom: 2, left: 0 }}>
          <CartesianGrid stroke={GRID} vertical={false} />
          <XAxis dataKey="month" tick={{ fill: AXIS, fontSize: 10, fontFamily: "'IBM Plex Mono', monospace" }} stroke={AXIS} interval={0} height={24} />
          <YAxis tick={{ fill: AXIS, fontSize: 10, fontFamily: "'IBM Plex Mono', monospace" }} stroke={AXIS} width={44} />
          <Tooltip contentStyle={tooltipStyle} formatter={(value) => [`${value} mm`, 'Rainfall']} />
          <Area type="monotone" dataKey="rain" stroke={RAIN} fill={RAIN} fillOpacity={0.22} strokeWidth={2} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
