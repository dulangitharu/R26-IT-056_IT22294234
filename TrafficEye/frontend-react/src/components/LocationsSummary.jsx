import { MapPin } from 'lucide-react';
import { LEVEL_META } from '../levels';
import { groupByLocation } from '../locations';

export default function LocationsSummary({ records }) {
  if (!records || records.length === 0) return null;

  const byLocation = groupByLocation(records);

  return (
    <div className="table-card">
      <div className="panel-header">
        <span><MapPin size={16} />GPS-Tagged Locations</span>
        <span style={{ fontSize: 12, color: 'var(--ink-500)' }}>{byLocation.size} unique</span>
      </div>
      <div className="loc-grid">
        {[...byLocation.entries()].map(([name, recs]) => {
          const latest = recs[0];
          const meta = LEVEL_META[latest.traffic_level] || LEVEL_META.CLEAR;
          return (
            <div className="loc-item" key={name}>
              <span className="loc-dot" style={{ background: meta.color }} />
              <div className="loc-info">
                <div className="loc-name">{name}</div>
                <div className="loc-sub">
                  {latest.road} · {latest.gps_lat.toFixed(4)}, {latest.gps_lng.toFixed(4)}
                </div>
              </div>
              <div className="loc-count">
                <div className="loc-count-val">{latest.vehicle_count}</div>
                <div className="loc-count-lbl">{recs.length} analysis{recs.length !== 1 ? 'es' : ''}</div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
