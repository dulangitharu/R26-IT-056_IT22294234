import { useMemo } from 'react';
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import { Map as MapIcon, MapPinOff } from 'lucide-react';
import { markerIconForLevel } from '../mapIcons';
import { groupByLocation } from '../locations';
import LevelBadge from './LevelBadge';
import MapReady from './MapReady';

const SRI_LANKA_CENTER = [7.8731, 80.7718];
const SRI_LANKA_ZOOM = 7;

export default function LocationsMap({ records }) {
  const locations = useMemo(() => {
    const grouped = groupByLocation(records);
    return [...grouped.entries()].map(([name, recs]) => ({ name, latest: recs[0] }));
  }, [records]);

  const center = useMemo(() => {
    if (locations.length === 0) return SRI_LANKA_CENTER;
    const avgLat = locations.reduce((sum, l) => sum + l.latest.gps_lat, 0) / locations.length;
    const avgLng = locations.reduce((sum, l) => sum + l.latest.gps_lng, 0) / locations.length;
    return [avgLat, avgLng];
  }, [locations]);

  const zoom = locations.length === 0 ? SRI_LANKA_ZOOM : 9;

  return (
    <div className="table-card">
      <div className="panel-header">
        <span><MapIcon size={16} />Location Map</span>
        <span style={{ fontSize: 12, color: 'var(--ink-500)' }}>{locations.length} location{locations.length !== 1 ? 's' : ''}</span>
      </div>
      <div className="map-card-wrap">
        <MapContainer center={center} zoom={zoom} scrollWheelZoom style={{ height: '100%', width: '100%' }}>
          <TileLayer
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          />
          <MapReady />
          {locations.map(({ name, latest }) => (
            <Marker key={name} position={[latest.gps_lat, latest.gps_lng]} icon={markerIconForLevel(latest.traffic_level)}>
              <Popup>
                <div className="map-popup">
                  <div className="map-popup-title">{name}</div>
                  <div className="map-popup-road">{latest.road}</div>
                  <div className="map-popup-badge"><LevelBadge level={latest.traffic_level} /></div>
                  <div className="map-popup-row"><span>Vehicles</span><strong>{latest.vehicle_count}</strong></div>
                  <div className="map-popup-row"><span>Suggested Green</span><strong>{latest.suggested_green_s}s</strong></div>
                </div>
              </Popup>
            </Marker>
          ))}
        </MapContainer>

        {locations.length === 0 && (
          <div className="map-empty-overlay">
            <div className="map-empty-text"><MapPinOff size={14} />No locations analyzed yet</div>
            <div className="map-empty-sub">Run an analysis to see it appear here.</div>
          </div>
        )}
      </div>
    </div>
  );
}
