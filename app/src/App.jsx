import { useEffect, useState } from "react";
import Header from "./Header.jsx";
import StationMap from "./StationMap.jsx";
import Legend from "./Legend.jsx";
import StationPanel from "./StationPanel.jsx";
import About from "./About.jsx";

export default function App() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [horizon, setHorizon] = useState("7");
  const [scale, setScale] = useState("skill");
  // optional ?station=CODE opens that station's panel, so a link can point at one station
  const [selected, setSelected] = useState(new URLSearchParams(window.location.search).get("station"));
  const [showAbout, setShowAbout] = useState(false);

  useEffect(() => {
    fetch("/demo_data.json")
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then(setData)
      .catch((e) => setError(e.message));
  }, []);

  if (error) return <div className="status">Could not load demo_data.json: {error}</div>;
  if (!data) return <div className="status">Loading...</div>;

  const station = data.stations.find((s) => s.code === selected);
  return (
    <div className="app">
      <Header data={data} horizon={horizon} setHorizon={setHorizon}
              scale={scale} setScale={setScale} onAbout={() => setShowAbout(true)} />
      <main className="main">
        <StationMap data={data} horizon={horizon} scale={scale}
                    selected={selected} onSelect={setSelected} />
        <Legend scale={scale} />
        {station && (
          <StationPanel station={station} horizon={horizon}
                        metrics={data.metrics[station.code][horizon]}
                        series={data.series[station.code][horizon]}
                        onClose={() => setSelected(null)} />
        )}
      </main>
      {showAbout && <About meta={data.meta} onClose={() => setShowAbout(false)} />}
    </div>
  );
}
