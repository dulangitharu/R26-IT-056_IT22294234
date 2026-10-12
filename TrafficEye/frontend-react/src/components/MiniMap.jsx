import { useEffect, useMemo, useState } from 'react';
import { MapContainer, TileLayer, Marker, useMap } from 'react-leaflet';
import { Map as MapIcon, ChevronDown, ChevronRight } from 'lucide-react';
import { pointMarkerIcon } from '../mapIcons';
import MapReady from './MapReady';

const SRI_LANKA_CENTER = [7.8731, 80.7718];
const SRI_LANKA_ZOOM = 7;

function RecenterOnChange({ center, zoom }) {
  const map = useMap();
  useEffect(() => {
    map.setView(center, zoom);
  }, [center, zoom, map]);
  return null;
}

export default function MiniMap({ lat, lng }) {
  const [expanded, setExpanded] = useState(true);

  const hasCoords = Number.isFinite(lat) && Number.isFinite(lng)
    && lat >= -90 && lat <= 90 && lng >= -180 && lng <= 180;

  const center = useMemo(
    () => (hasCoords ? [lat, lng] : SRI_LANKA_CENTER),
    [hasCoords, lat, lng]
  );
  const zoom = hasCoords ? 14 : SRI_LANKA_ZOOM;

  return (
    <div className="mini-map-card">
      <button type="button" className="mini-map-toggle" onClick={() => setExpanded(e => !e)}>
        <span><MapIcon size={14} />Location Preview</span>
        <span>{expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}</span>
      </button>

      {expanded && (
        <div className="mini-map-wrap">
          <MapContainer center={center} zoom={zoom} scrollWheelZoom={false} style={{ height: '100%', width: '100%' }}>
            <TileLayer
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              attribution='&copy; OpenStreetMap contributors'
            />
            <MapReady />
            <RecenterOnChange center={center} zoom={zoom} />
            {hasCoords && <Marker position={center} icon={pointMarkerIcon()} />}
          </MapContainer>

          {!hasCoords && (
            <div className="mini-map-hint">Enter or capture GPS coordinates to preview the location.</div>
          )}
        </div>
      )}
    </div>
  );
}
