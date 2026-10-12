import { useEffect, useRef, useState } from 'react';
import { MapPin, Car, Ruler, Timer, TimerReset, Layers, Inbox, AlertTriangle } from 'lucide-react';
import LevelBadge from './LevelBadge';

export default function QueueIntelligence({ record }) {
  const [remaining, setRemaining] = useState(0);
  const totalRef = useRef(0);

  useEffect(() => {
    if (!record) return undefined;
    const total = Math.max(0, Math.round(record.suggested_green_s));
    totalRef.current = total;
    setRemaining(total);
    const id = setInterval(() => {
      setRemaining(r => Math.max(0, r - 1));
    }, 1000);
    return () => clearInterval(id);
  }, [record]);

  if (!record) {
    return (
      <div className="no-det">
        <Inbox size={26} />
        No analysis recorded yet. Run one from &quot;New Analysis&quot;.
      </div>
    );
  }

  const pct = totalRef.current > 0 ? (remaining / totalRef.current) * 100 : 0;

  return (
    <>
      {record.emergency_detected && (
        <div style={{
          background: 'var(--critical-tint)', border: '1px solid var(--critical-border)', borderRadius: 10,
          padding: '14px 16px', marginBottom: 16,
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: 'var(--critical)', fontWeight: 700, fontSize: 13, letterSpacing: '.03em' }}>
            <AlertTriangle size={16} />
            EMERGENCY VEHICLE DETECTED
          </div>
          <div style={{ marginTop: 8, fontSize: 13, color: 'var(--ink-700)', lineHeight: 1.7 }}>
            <div>Vehicle: <strong>{record.emergency_vehicle_type}</strong></div>
            <div>Queue Position: <strong>{record.emergency_queue_position}</strong></div>
            <div>Position Group: <strong>{record.emergency_position_group}</strong></div>
            <div>Priority: <strong>{record.emergency_priority}</strong></div>
            <div>Estimated Emergency Clearance: <strong>{record.emergency_clearance_time_s}s</strong></div>
            <div>Priority Green Recommendation: <strong>{record.emergency_priority_green_s}s</strong></div>
          </div>
          <div style={{ marginTop: 8, fontSize: 12.5, fontStyle: 'italic', color: 'var(--ink-500)' }}>
            {record.emergency_recommendation}
          </div>
        </div>
      )}
      <div className="qi-countdown">
        <div className="qi-ring">
          <div className="num">{remaining}</div>
          <div className="unit-label">SEC GREEN</div>
        </div>
        <div className="qi-info">
          <div className="qi-info-title">
            <MapPin size={15} style={{ verticalAlign: -2, marginRight: 6, opacity: .7 }} />
            {record.location_name} — {record.road} ({record.direction})
          </div>
          <div style={{ marginTop: 9 }}><LevelBadge level={record.traffic_level} /></div>
          <div className="qi-bar"><div className="qi-bar-fill" style={{ width: `${pct}%` }} /></div>
        </div>
      </div>
      <div className="qi-grid">
        <div className="qi-metric"><div className="stat-label"><Car size={11} style={{ verticalAlign: -1, marginRight: 4 }} />VEHICLES</div><div className="stat-value">{record.vehicle_count}</div></div>
        <div className="qi-metric"><div className="stat-label"><Ruler size={11} style={{ verticalAlign: -1, marginRight: 4 }} />QUEUE LENGTH</div><div className="stat-value">{record.queue_length_m}<span className="unit">m</span></div></div>
        <div className="qi-metric"><div className="stat-label"><Timer size={11} style={{ verticalAlign: -1, marginRight: 4 }} />CLEARANCE TIME</div><div className="stat-value">{record.clearance_time_s}<span className="unit">s</span></div></div>
        <div className="qi-metric"><div className="stat-label"><TimerReset size={11} style={{ verticalAlign: -1, marginRight: 4 }} />SUGGESTED GREEN</div><div className="stat-value">{record.suggested_green_s}<span className="unit">s</span></div></div>
        {record.safety_buffer_s != null && (
          <div className="qi-metric"><div className="stat-label"><TimerReset size={11} style={{ verticalAlign: -1, marginRight: 4 }} />SAFETY BUFFER</div><div className="stat-value">{record.safety_buffer_s}<span className="unit">s</span></div></div>
        )}
      </div>
      {record.vehicle_breakdown && (
        <div style={{ marginTop: 16 }}>
          <div className="stat-label" style={{ marginBottom: 7 }}><Layers size={11} style={{ verticalAlign: -1, marginRight: 4 }} />VEHICLE BREAKDOWN</div>
          {Object.keys(record.vehicle_breakdown).length === 0
            ? <div className="no-det"><Inbox size={20} />No vehicles detected</div>
            : Object.entries(record.vehicle_breakdown).map(([label, count]) => (
                <div className="det-item" key={label}>
                  <span>{label.toUpperCase()}</span>
                  <span className="det-badge">× {count}</span>
                </div>
              ))}
        </div>
      )}
    </>
  );
}
