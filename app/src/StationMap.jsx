import { Fragment } from "react";
import { MapContainer, TileLayer, CircleMarker, Tooltip } from "react-leaflet";
import { colourFor, RING } from "./scales.js";

const R = 8;

export default function StationMap({ data, horizon, scale, selected, onSelect }) {
  // fit the view to the stations themselves rather than a hand-picked centre
  const bounds = data.stations.map((s) => [s.lat, s.lon]);
  // draw no-data stations, then the selected one, last: CGWBNG0018 sits ~2 km from CGWBNG0019
  // and was hidden under it at city zoom
  const rank = (s) => (s.code === selected ? 2 : data.metrics[s.code][horizon] ? 0 : 1);
  const ordered = [...data.stations].sort((a, b) => rank(a) - rank(b));
  return (
    <MapContainer className="map" bounds={bounds} boundsOptions={{ padding: [40, 40] }}>
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      {ordered.map((s) => {
        const m = data.metrics[s.code][horizon];
        const isSel = s.code === selected;
        return (
          <Fragment key={s.code}>
            {m?.low_coverage && (
              // the ring is drawn under the marker and ignores clicks so the marker stays clickable
              <CircleMarker center={[s.lat, s.lon]} radius={R + 5} interactive={false}
                            pathOptions={{ color: RING, weight: 2.5, fill: false, dashArray: "4 3" }} />
            )}
            <CircleMarker center={[s.lat, s.lon]} radius={m ? R : R - 2}
                          eventHandlers={{ click: () => onSelect(s.code) }}
                          pathOptions={{ color: isSel ? RING : "#ffffff", weight: isSel ? 3 : 2,
                                         fillColor: colourFor(scale, m), fillOpacity: 0.95,
                                         dashArray: m ? null : "2 2" }}>
              <Tooltip direction="top" offset={[0, -R]}>
                {s.name.replaceAll("_", " ")} ({s.code}){m ? "" : " - no test data"}
              </Tooltip>
            </CircleMarker>
          </Fragment>
        );
      })}
    </MapContainer>
  );
}
