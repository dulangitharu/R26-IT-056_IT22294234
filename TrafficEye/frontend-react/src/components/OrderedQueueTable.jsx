import Icon from './Icon';

// Renders the backend's per-vehicle queue-position breakdown exactly as
// returned by build_queue_result() (traffic_calculations.py) — no value
// here is recalculated in React. recommended_speed_mps is deliberately
// labeled "Effective Discharge Speed" per the approved UI copy: it is the
// calibrated discharge speed used in the FIFO clearance calculation, not a
// live/current vehicle speed reading.
//
// Display-only: the Pos (queue_position), Group (position_group) and
// FIFO-Adjusted Clear Time (fifo_adjusted_clear_time_s) columns are hidden
// from this table per the approved UI copy. Those fields are STILL present in
// every backend response / record and are still used by the calculations and
// recommendations — this component just does not draw them.
export default function OrderedQueueTable({ orderedQueue }) {
  if (!orderedQueue || orderedQueue.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center gap-2 text-on-surface-variant text-body-md py-10">
        <Icon name="inbox" />
        No ordered-queue detail available.
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left border-collapse">
        <thead>
          <tr className="border-b border-outline-variant/50 bg-surface-container-lowest/30">
            <th className="py-3 px-4 font-label-caps text-label-caps text-on-surface-variant tracking-wider">Type</th>
            <th className="py-3 px-4 font-label-caps text-label-caps text-on-surface-variant tracking-wider text-right">Length (m)</th>
            <th className="py-3 px-4 font-label-caps text-label-caps text-on-surface-variant tracking-wider text-right">Cumulative Dist. (m)</th>
            <th className="py-3 px-4 font-label-caps text-label-caps text-on-surface-variant tracking-wider text-right">Effective Discharge Speed (m/s)</th>
            <th className="py-3 px-4 font-label-caps text-label-caps text-on-surface-variant tracking-wider text-right">Raw Clear Time (s)</th>
          </tr>
        </thead>
        <tbody className="font-body-md divide-y divide-outline-variant/20">
          {orderedQueue.map((v, i) => (
            <tr key={v.queue_position} className={'hover:bg-surface-variant/20 transition-colors' + (i % 2 === 1 ? ' bg-surface-container-lowest/10' : '')}>
              <td className="py-3 px-4 text-on-surface">{v.vehicle_type}</td>
              <td className="py-3 px-4 text-on-surface-variant text-right">{v.vehicle_length_m}</td>
              <td className="py-3 px-4 text-on-surface-variant text-right">{v.cumulative_distance_m}</td>
              <td className="py-3 px-4 text-on-surface-variant text-right">{v.recommended_speed_mps}</td>
              <td className="py-3 px-4 text-on-surface font-mono text-right">{v.raw_clear_time_s}s</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
