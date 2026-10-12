import Icon from './Icon';

const LEVEL_BADGE = {
  CLEAR: 'bg-tertiary-container/10 border-tertiary-container/30 text-tertiary-fixed',
  LOW: 'bg-tertiary-container/10 border-tertiary-container/30 text-tertiary-fixed',
  MEDIUM: 'bg-secondary-container/30 border-outline-variant text-secondary-fixed',
  HIGH: 'bg-error/20 border-error/30 text-error',
};

// Columns reproduced from the verified Stitch "Analysis History" screen.
// emergency_detected is now persisted per-record (see main.py/models.py):
// true -> DETECTED, false -> NONE. Records saved before this field existed
// simply lack the key, so r.emergency_detected is undefined/null for them -
// shown as "—" rather than guessing/backfilling a value the backend never
// actually computed for that record.
export default function RecordsTable({ records, onViewDetails }) {
  if (!records || records.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center gap-2 text-on-surface-variant py-16">
        <Icon name="inbox" />
        No analysis records yet.
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left border-collapse">
        <thead>
          <tr className="bg-surface-container border-b border-outline-variant">
            <th className="py-3 px-4 font-label-caps text-label-caps text-on-surface-variant uppercase font-bold tracking-wider">Date</th>
            <th className="py-3 px-4 font-label-caps text-label-caps text-on-surface-variant uppercase font-bold tracking-wider">Time</th>
            <th className="py-3 px-4 font-label-caps text-label-caps text-on-surface-variant uppercase font-bold tracking-wider text-right">Vehicle Count</th>
            <th className="py-3 px-4 font-label-caps text-label-caps text-on-surface-variant uppercase font-bold tracking-wider">Traffic Level</th>
            <th className="py-3 px-4 font-label-caps text-label-caps text-on-surface-variant uppercase font-bold tracking-wider text-right">Queue Length</th>
            <th className="py-3 px-4 font-label-caps text-label-caps text-on-surface-variant uppercase font-bold tracking-wider text-right">Clearance Time</th>
            <th className="py-3 px-4 font-label-caps text-label-caps text-on-surface-variant uppercase font-bold tracking-wider">Suggested Green</th>
            <th className="py-3 px-4 font-label-caps text-label-caps text-on-surface-variant uppercase font-bold tracking-wider text-center">Emergency</th>
            <th className="py-3 px-4 font-label-caps text-label-caps text-on-surface-variant uppercase font-bold tracking-wider">Status</th>
            {onViewDetails && <th className="py-3 px-4 font-label-caps text-label-caps text-on-surface-variant uppercase font-bold tracking-wider text-right">Action</th>}
          </tr>
        </thead>
        <tbody className="font-body-md text-body-md text-on-surface divide-y divide-outline-variant/50">
          {records.map(r => (
            <tr key={r._id} className="hover:bg-surface-variant/30 transition-colors group">
              <td className="py-4 px-4 whitespace-nowrap font-medium">{r.recorded_date}</td>
              <td className="py-4 px-4 text-on-surface-variant">{r.recorded_time}</td>
              <td className="py-4 px-4 text-right font-display-metrics text-[20px] font-semibold text-tertiary-container">{r.vehicle_count}</td>
              <td className="py-4 px-4">
                <span className={`inline-flex items-center px-2.5 py-1 rounded-full text-label-sm font-label-caps border ${LEVEL_BADGE[r.traffic_level] || LEVEL_BADGE.LOW}`}>
                  {r.traffic_level}
                </span>
              </td>
              <td className="py-4 px-4 text-right">{r.queue_length_m} m</td>
              <td className="py-4 px-4 text-right">{r.clearance_time_s} s</td>
              <td className="py-4 px-4 font-bold text-tertiary-fixed">{r.suggested_green_s} s</td>
              <td className="py-4 px-4 text-center">
                {r.emergency_detected === true ? (
                  <span className="inline-flex items-center px-2.5 py-1 rounded-full text-label-sm font-label-caps border bg-error/20 border-error/30 text-error">DETECTED</span>
                ) : r.emergency_detected === false ? (
                  <span className="text-on-surface-variant">NONE</span>
                ) : (
                  <span className="text-on-surface-variant/50" title="Not recorded for this older record">—</span>
                )}
              </td>
              <td className="py-4 px-4">
                <div className="flex items-center gap-1.5 text-on-surface-variant">
                  <div className="w-1.5 h-1.5 rounded-full bg-outline" />
                  {r.status || 'completed'}
                </div>
              </td>
              {onViewDetails && (
                <td className="py-4 px-4 text-right">
                  <button onClick={() => onViewDetails(r._id)} className="text-primary hover:text-primary-fixed transition-colors font-title-sm text-title-sm flex items-center gap-1 ml-auto">
                    View Details <Icon name="arrow_forward" className="text-[18px]" />
                  </button>
                </td>
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
