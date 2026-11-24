import { useMemo } from "react";
import {
  ComposedChart, Area, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from "recharts";
import { fmt, needsWarning } from "./scales.js";

const C = { band: "#86b6ef", line: "#2a78d6", obs: "#0b0b0b", grid: "#e1e0d9", axis: "#898781" };

function toRows(series) {
  // recharts wants one object per point; "band" is a [lower, upper] pair for the range area
  return series.date.map((d, i) => ({
    date: d, observed: series.observed[i], forecast: series.forecast[i],
    band: [series.lower[i], series.upper[i]],
  }));
}

function ChartTip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  return (
    <div className="chart-tip">
      <div className="tip-date">{label}</div>
      <div>observed {fmt.m(p.observed)}</div>
      <div>forecast {fmt.m(p.forecast)}</div>
      <div>90% interval {fmt.num(p.band[0])} to {fmt.m(p.band[1])}</div>
    </div>
  );
}

function SeriesChart({ series }) {
  const rows = useMemo(() => toRows(series), [series]);
  return (
    <ResponsiveContainer width="100%" height={260}>
      <ComposedChart data={rows} margin={{ top: 8, right: 12, bottom: 4, left: 4 }}>
        <CartesianGrid stroke={C.grid} vertical={false} />
        <XAxis dataKey="date" tick={{ fontSize: 11, fill: C.axis }} minTickGap={40}
               tickFormatter={(d) => d.slice(0, 7)} stroke={C.axis} />
        <YAxis tick={{ fontSize: 11, fill: C.axis }} stroke={C.axis} width={52}
               domain={["auto", "auto"]} tickFormatter={(v) => v.toFixed(1)}
               label={{ value: "level (m)", angle: -90, position: "insideLeft", fontSize: 11, fill: C.axis }} />
        <Tooltip content={<ChartTip />} />
        <Area dataKey="band" stroke="none" fill={C.band} fillOpacity={0.45} isAnimationActive={false} />
        <Line dataKey="forecast" stroke={C.line} strokeWidth={2} dot={false} isAnimationActive={false} />
        <Line dataKey="observed" stroke="none" isAnimationActive={false}
              dot={{ r: 1.6, fill: C.obs, stroke: "none" }} activeDot={{ r: 4 }} />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

function MetricsTable({ m }) {
  const rows = [
    ["Test RMSE, model", fmt.m(m.rmse_model)],
    ["Test RMSE, persistence", fmt.m(m.rmse_persistence)],
    ["Skill (1 - RMSE ratio)", fmt.num(m.skill)],
    ["90% interval width", `${fmt.m(m.width_m, 2)} (forecast +/- ${fmt.num(m.q_m, 2)} m)`],
    ["Test coverage", fmt.pct(m.coverage)],
    ["Test error / calibration error", fmt.num(m.err_ratio, 2)],
    ["Test samples (n_test)", m.n_test],
    ["Calibration samples", m.n_calib],
  ];
  return (
    <table className="metrics">
      <tbody>{rows.map(([k, v]) => <tr key={k}><th>{k}</th><td>{v}</td></tr>)}</tbody>
    </table>
  );
}

export default function StationPanel({ station, horizon, metrics, series, onClose }) {
  return (
    <aside className="panel">
      <button className="close" onClick={onClose} aria-label="Close panel">Close</button>
      <h2>{station.name.replaceAll("_", " ")}</h2>
      <div className="code">{station.code} | {horizon}-day forecast | test period</div>
      {!metrics ? (
        <p className="empty">No test data for this station: it has no observations after the
          calibration period, so no test forecasts could be scored.</p>
      ) : (
        <>
          {needsWarning(metrics) && (
            <p className="warning">Intervals under-covered here: this station's test errors were
              larger than in the calibration year.</p>
          )}
          <div className="chart-key">
            <span><i className="k-dot" />observed</span>
            <span><i className="k-line" />forecast</span>
            <span><i className="k-band" />90% interval</span>
          </div>
          <SeriesChart series={series} />
          <MetricsTable m={metrics} />
        </>
      )}
    </aside>
  );
}
