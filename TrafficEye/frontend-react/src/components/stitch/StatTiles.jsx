import Icon from '../Icon';

const LEVEL_COLOR = {
  CLEAR: { text: 'text-tertiary-fixed-dim', dot: 'bg-tertiary-fixed-dim', glow: 'rgba(74,225,118,0.8)' },
  LOW: { text: 'text-tertiary-fixed-dim', dot: 'bg-tertiary-fixed-dim', glow: 'rgba(74,225,118,0.8)' },
  MEDIUM: { text: 'text-orange-400', dot: 'bg-orange-500', glow: 'rgba(249,115,22,0.8)' },
  HIGH: { text: 'text-error', dot: 'bg-error', glow: 'rgba(255,180,171,0.8)' },
};

// The stat tiles used on Analysis Results, reused wherever a compact
// summary of one record's core real fields is needed.
export default function StatTiles({ record }) {
  const lvl = LEVEL_COLOR[record.traffic_level] || LEVEL_COLOR.LOW;
  // Older History records were saved before total_detected_vehicles existed -
  // show an honest "not tracked" placeholder rather than fabricating a value.
  const hasTotalDetected = record.total_detected_vehicles !== undefined && record.total_detected_vehicles !== null;
  return (
    <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
      <div className="bg-surface-container p-4 rounded-xl border border-outline-variant/50 flex flex-col justify-between">
        <div className="flex items-start justify-between mb-2">
          <span className="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wider">Total Detected Vehicles</span>
          <Icon name="visibility" className="text-secondary/70 text-lg" />
        </div>
        <span className="font-headline-lg text-headline-lg text-on-surface">{hasTotalDetected ? record.total_detected_vehicles : '—'}</span>
      </div>
      <div className="bg-surface-container p-4 rounded-xl border border-outline-variant/50 flex flex-col justify-between">
        <div className="flex items-start justify-between mb-2">
          <span className="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wider">ROI Queue Vehicles</span>
          <Icon name="directions_car" className="text-primary/70 text-lg" />
        </div>
        <span className="font-headline-lg text-headline-lg text-on-surface">{record.vehicle_count}</span>
      </div>
      <div className="bg-surface-container p-4 rounded-xl border border-outline-variant/50 flex flex-col justify-between">
        <div className="flex items-start justify-between mb-2">
          <span className="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wider">Queue Length</span>
          <Icon name="straighten" className="text-secondary/70 text-lg" />
        </div>
        <span className="font-headline-lg text-headline-lg text-on-surface">{record.queue_length_m}<span className="font-body-md text-on-surface-variant ml-1">m</span></span>
      </div>
      <div className="bg-surface-container p-4 rounded-xl border border-outline-variant/50 flex flex-col justify-between">
        <div className="flex items-start justify-between mb-2">
          <span className="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wider">Clearance Est.</span>
          <Icon name="timer" className="text-primary-container/70 text-lg" />
        </div>
        <span className="font-headline-lg text-headline-lg text-on-surface">{record.clearance_time_s}<span className="font-body-md text-on-surface-variant ml-1">s</span></span>
      </div>
      <div className="bg-surface-container p-4 rounded-xl border border-outline-variant/50 flex flex-col justify-between relative overflow-hidden">
        <div className="flex items-start justify-between mb-2 relative z-10">
          <span className="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wider">Traffic Level</span>
          <Icon name="traffic" className={`${lvl.text} text-lg`} />
        </div>
        <div className="flex items-center gap-2 relative z-10 mt-1">
          <span className={`w-2.5 h-2.5 rounded-full ${lvl.dot}`} style={{ boxShadow: `0 0 8px ${lvl.glow}` }} />
          <span className={`font-headline-md text-[20px] ${lvl.text} font-bold tracking-wide`}>{record.traffic_level}</span>
        </div>
      </div>
    </div>
  );
}
