import { divIcon } from 'leaflet';

// Leaflet's *default* marker icon references image files (marker-icon.png,
// marker-shadow.png) via relative paths that Vite's bundler doesn't resolve
// correctly out of the box — a well-known Leaflet+Vite/webpack gotcha. We
// sidestep it entirely by never using L.icon()/the default icon: every
// marker below is an inline SVG divIcon, so there are no external image
// assets for the bundler to mishandle in the first place.

const MARKER_COLORS = {
  CLEAR: '#16A34A',
  LOW: '#16A34A',
  MEDIUM: '#D97706',
  HIGH: '#B91C1C',
};

function pinSvg(color) {
  return `
    <svg width="26" height="36" viewBox="0 0 26 36" xmlns="http://www.w3.org/2000/svg">
      <path d="M13 0C5.8 0 0 5.8 0 13c0 9.8 13 23 13 23s13-13.2 13-23C26 5.8 20.2 0 13 0z" fill="${color}" stroke="#fff" stroke-width="1.5"/>
      <circle cx="13" cy="13" r="5" fill="#fff"/>
    </svg>
  `;
}

function pinIcon(color) {
  return divIcon({
    className: 'map-pin-icon',
    html: pinSvg(color),
    iconSize: [26, 36],
    iconAnchor: [13, 36],
    popupAnchor: [0, -32],
  });
}

export function markerIconForLevel(level) {
  return pinIcon(MARKER_COLORS[level] || MARKER_COLORS.CLEAR);
}

export function pointMarkerIcon() {
  return pinIcon('#2563EB');
}
