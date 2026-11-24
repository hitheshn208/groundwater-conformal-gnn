import { SCALES, NO_DATA, RING } from "./scales.js";

function Swatch({ color, dashed }) {
  return <span className="swatch" style={{ background: color, borderStyle: dashed ? "dashed" : "solid" }} />;
}

export default function Legend({ scale }) {
  const s = SCALES[scale];
  return (
    <div className="legend">
      <div className="legend-title">{s.label}</div>
      <div className="legend-note">{s.note}</div>
      {s.bins.map((b) => (
        <div key={b.label} className="legend-row"><Swatch color={b.color} />{b.label}</div>
      ))}
      <div className="legend-row"><Swatch color={NO_DATA} dashed />no test data</div>
      <div className="legend-row">
        <span className="ring" style={{ borderColor: RING }} />coverage under 85% (any colouring)
      </div>
    </div>
  );
}
