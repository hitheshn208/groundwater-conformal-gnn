export default function About({ meta, onClose }) {
  const s = meta.split;
  return (
    <div className="overlay" onClick={onClose}>
      <div className="about" onClick={(e) => e.stopPropagation()} role="dialog" aria-label="About">
        <button className="close" onClick={onClose}>Close</button>
        <h2>About this demo</h2>
        <p>
          {meta.model} forecasts of groundwater level at {meta.horizons.join(" and ")} days ahead,
          with {Math.round(meta.coverage_target * 100)}% split-conformal intervals calibrated per
          station. {meta.note}
        </p>
        <p>
          Split by target date: training before {s.val_start}, early-stopping validation
          {" "}{s.val_start} to {s.train_end}, calibration to {s.calib_end}, test after that.
          Persistence means "the level stays at today's value".
        </p>
        <h3>Limitations</h3>
        <ul>
          <li>Results come from one time split and one training run; differences between models
            are small and may not hold for another split or seed.</li>
          <li>Some spike bursts survived data cleaning at a few stations (for example CGWBNG0007),
            which inflates errors there.</li>
          <li>The validation window used for early stopping is short ({s.val_start} to {s.train_end}).</li>
          <li>Rainfall data for 2026 is provisional.</li>
          <li>The station graph gave no reliable accuracy gain over the same model without it.</li>
          <li>Conformal intervals assume test errors resemble calibration-year errors. Where they
            did not, coverage fell below target (ringed stations on the map).</li>
          <li>Station CGWBNG0018 has no test observations and is shown without results.</li>
        </ul>
      </div>
    </div>
  );
}
