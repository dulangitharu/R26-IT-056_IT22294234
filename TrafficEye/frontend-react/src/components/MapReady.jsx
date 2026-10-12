import { useEffect } from 'react';
import { useMap } from 'react-leaflet';

// THE MAP BUG FIX: react-leaflet's MapContainer can end up with a stale
// (sometimes zero) cached container size when mounted under React 18
// StrictMode, which double-invokes mount effects in development — the
// Leaflet instance exists but never re-measures its container, so it never
// requests tiles for the real viewport. Only the pane's background color
// shows, which reads as an empty tinted rectangle. One invalidateSize()
// pass right after mount forces Leaflet to re-measure and fetch tiles.
export default function MapReady() {
  const map = useMap();
  useEffect(() => {
    const id = requestAnimationFrame(() => map.invalidateSize());
    return () => cancelAnimationFrame(id);
  }, [map]);
  return null;
}
