import { SCALES, fmt } from "./scales.js";

function Toggle({ options, value, onChange, label }) {
  return (
    <div className="toggle" role="group" aria-label={label}>
      <span className="toggle-label">{label}</span>
      {options.map(([key, text]) => (
        <button key={key} className={key === value ? "on" : ""} onClick={() => onChange(key)}>
          {text}
        </button>
      ))}
    </div>
  );
}

function HeadlineRow({ h, hl, active }) {
  return (
    <tr className={active ? "active" : ""}>
      <th>{h} days</th>
      <td>{fmt.m(hl.rmse_model)}</td>
      <td>{fmt.m(hl.rmse_persistence)}</td>
      <td>{fmt.pct(hl.coverage)}</td>
      <td>{fmt.m(hl.mean_width_m, 2)}</td>
      <td>{hl.n_test}</td>
    </tr>
  );
}

export default function Header({ data, horizon, setHorizon, scale, setScale, onAbout }) {
  const { meta, headline } = data;
  return (
    <header className="header">
      <div className="header-top">
        <div>
          <h1>Groundwater level forecasts, Bangalore ({meta.model})</h1>
          <p className="disclaimer">
            Test-period results from one time split and one training run. Differences between
            models are small; not a live forecast.
          </p>
        </div>
        <table className="headline">
          <thead>
            <tr>
              <th>Pooled test</th><th>RMSE model</th><th>RMSE persistence</th>
              <th>Coverage (target {fmt.pct(meta.coverage_target)})</th><th>Mean width</th><th>n</th>
            </tr>
          </thead>
          <tbody>
            {meta.horizons.map((h) => (
              <HeadlineRow key={h} h={h} hl={headline[h]} active={String(h) === horizon} />
            ))}
          </tbody>
        </table>
      </div>
      <div className="controls">
        <Toggle label="Horizon" value={horizon} onChange={setHorizon}
                options={meta.horizons.map((h) => [String(h), `${h} days`])} />
        <Toggle label="Colour by" value={scale} onChange={setScale}
                options={Object.entries(SCALES).map(([k, s]) => [k, s.label])} />
        <button className="about-btn" onClick={onAbout}>About and limitations</button>
      </div>
    </header>
  );
}
