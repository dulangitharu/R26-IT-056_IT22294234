import { TrendingUp } from 'lucide-react';

function levelClass(level) {
  if (level === 'HIGH') return 'lvl-critical';
  if (level === 'MEDIUM') return 'lvl-warning';
  return 'lvl-good';
}

export default function TrendChart({ records }) {
  if (!records || records.length === 0) return null;

  // Oldest → newest, left to right (standard reading order for a trend).
  const recent = [...records].slice(0, 8).reverse();
  const max = Math.max(...recent.map(r => r.vehicle_count), 1);

  return (
    <div className="table-card trend-card">
      <div className="panel-header"><span><TrendingUp size={16} />Recent Vehicle Count Trend</span></div>
      <div className="trend-body">
        <div className="trend-bars">
          {recent.map((r, i) => {
            const heightPct = Math.max((r.vehicle_count / max) * 100, 4);
            return (
              <div className="trend-bar-col" key={r._id}>
                <div
                  className={`trend-bar ${levelClass(r.traffic_level)}`}
                  style={{ height: `${heightPct}%`, animationDelay: `${i * 0.05}s` }}
                  data-tooltip={`${r.vehicle_count} vehicles`}
                  aria-label={`${r.location_name}: ${r.vehicle_count} vehicles, ${r.traffic_level}`}
                />
                <div className="trend-bar-label">{r.recorded_time}</div>
              </div>
            );
          })}
        </div>
        <div className="trend-legend">
          <div className="trend-legend-item"><span className="trend-legend-dot" style={{ background: 'var(--good)' }} />CLEAR / LOW</div>
          <div className="trend-legend-item"><span className="trend-legend-dot" style={{ background: 'var(--warning)' }} />MEDIUM</div>
          <div className="trend-legend-item"><span className="trend-legend-dot" style={{ background: 'var(--critical)' }} />HIGH</div>
        </div>
      </div>
    </div>
  );
}
