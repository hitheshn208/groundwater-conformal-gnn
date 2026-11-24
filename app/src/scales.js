// Fixed colour bins, the same for both horizons so the two maps can be compared.
// Bins only colour the markers; every number shown in text comes straight from demo_data.json.

export const NO_DATA = "#b9b8b2";
export const RING = "#0b0b0b";

export const SCALES = {
  skill: {
    label: "Skill vs persistence",
    note: "1 - RMSE(model) / RMSE(persistence), test period",
    value: (m) => m.skill,
    bins: [
      { max: 0, color: "#e34948", label: "below 0 (worse than persistence)" },
      { max: 0.05, color: "#9ec5f4", label: "0 to 0.05" },
      { max: 0.1, color: "#5598e7", label: "0.05 to 0.10" },
      { max: 0.2, color: "#256abf", label: "0.10 to 0.20" },
      { max: Infinity, color: "#0d366b", label: "0.20 and above" },
    ],
  },
  width: {
    label: "90% interval width",
    note: "full width 2q in metres, from the station's calibration year",
    value: (m) => m.width_m,
    bins: [
      { max: 1, color: "#d9d4f7", label: "under 1 m" },
      { max: 2, color: "#ab9fee", label: "1 to 2 m" },
      { max: 5, color: "#7f6de2", label: "2 to 5 m" },
      { max: 10, color: "#5241b8", label: "5 to 10 m" },
      { max: Infinity, color: "#2d2178", label: "10 m and above" },
    ],
  },
  coverage: {
    label: "Test coverage of 90% interval",
    note: "share of test observations inside the interval",
    value: (m) => m.coverage,
    bins: [
      { max: 0.85, color: "#d03b3b", label: "under 85% (low)" },
      { max: 0.9, color: "#fab219", label: "85% to 90%" },
      { max: 0.95, color: "#7cc47c", label: "90% to 95%" },
      { max: Infinity, color: "#0ca30c", label: "95% and above" },
    ],
  },
};

export function colourFor(scaleKey, metrics) {
  if (!metrics) return NO_DATA;
  const v = SCALES[scaleKey].value(metrics);
  return SCALES[scaleKey].bins.find((b) => v < b.max).color;
}

export const fmt = {
  m: (x, d = 3) => `${x.toFixed(d)} m`,
  pct: (x) => `${(100 * x).toFixed(1)}%`,
  num: (x, d = 3) => x.toFixed(d),
};

// Same rule as the brief: low coverage, or test errors well above calibration errors.
export const needsWarning = (m) => m && (m.low_coverage || m.err_ratio > 1.5);
