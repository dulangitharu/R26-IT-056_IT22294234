export function SkeletonStats() {
  return (
    <div className="skeleton-stats">
      {[0, 1, 2, 3].map(i => <div className="skeleton skeleton-stat-card" key={i} />)}
    </div>
  );
}

export function SkeletonRows({ rows = 4 }) {
  return (
    <div style={{ padding: 16 }}>
      {Array.from({ length: rows }).map((_, i) => (
        <div className="skeleton skeleton-row" key={i} style={{ opacity: 1 - i * 0.12 }} />
      ))}
    </div>
  );
}

export function SkeletonBlock({ height = 200 }) {
  return <div className="skeleton skeleton-block" style={{ height }} />;
}
