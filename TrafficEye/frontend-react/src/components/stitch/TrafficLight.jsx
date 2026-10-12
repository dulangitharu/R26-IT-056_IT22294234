// Three stacked traffic-light dots, bottom (green) lit for CLEAR/LOW, middle
// (amber) for MEDIUM, top (red) for HIGH — a real, direct mapping of the
// backend's traffic_level value, not a decorative default.
const LIT = {
  CLEAR: 'bottom', LOW: 'bottom', MEDIUM: 'middle', HIGH: 'top',
};

export default function TrafficLight({ level, size = 12 }) {
  const lit = LIT[level] || 'bottom';
  const dot = (which, color, glow) => (
    <div
      className="rounded-full border border-outline/20"
      style={{
        width: size, height: size,
        background: lit === which ? color : 'var(--color-surface-variant)',
        boxShadow: lit === which ? `0 0 15px ${glow}` : undefined,
      }}
    />
  );
  return (
    <div className="flex flex-col gap-2 bg-surface p-2 rounded-xl border border-outline-variant/30">
      {dot('top', '#EF4444', 'rgba(239,68,68,0.6)')}
      {dot('middle', '#F59E0B', 'rgba(245,158,11,0.6)')}
      {dot('bottom', '#2ccb63', 'rgba(44,203,99,0.6)')}
    </div>
  );
}
