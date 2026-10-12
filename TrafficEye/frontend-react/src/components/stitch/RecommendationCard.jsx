import Icon from '../Icon';
import TrafficLight from './TrafficLight';

// "AI Recommendation" card — reproduces the Stitch pattern used on both
// Operations Overview and Analysis Results. suggested_green_s / queue
// clearance / vehicle count are the same real backend fields either way;
// no "Apply Phase" action is offered — there is no backend endpoint that
// actually pushes a signal change, so a button here would be non-functional.
export default function RecommendationCard({ record }) {
  return (
    <div className="bg-surface-container rounded-xl border border-outline-variant flex flex-col overflow-hidden relative shadow-lg shadow-black/20">
      <div className="p-5 border-b border-outline-variant relative z-10">
        <h3 className="font-title-sm text-title-sm text-on-surface flex items-center gap-2">
          <Icon name="tips_and_updates" className="text-tertiary-fixed" />
          AI Recommendation
        </h3>
      </div>
      <div className="p-6 flex-1 flex flex-col items-center justify-center relative z-10">
        <div className="bg-surface-container-lowest border border-outline-variant/50 rounded-2xl p-8 flex flex-col items-center gap-4 w-full relative">
          <div className="absolute -top-6 bg-surface-container border border-outline-variant px-4 py-1 rounded-full text-on-surface-variant font-label-caps text-[10px] tracking-widest shadow-md">
            OPTIMAL GREEN PHASE
          </div>
          <div className="flex items-center gap-6">
            <TrafficLight level={record.traffic_level} size={32} />
            <div className="text-center">
              <span className="font-display-metrics text-display-metrics text-on-surface leading-none block">
                {record.suggested_green_s}<span className="text-2xl text-on-surface-variant ml-1">s</span>
              </span>
              <span className="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wider block mt-2">Recommended Duration</span>
            </div>
          </div>
        </div>
        <div className="w-full grid grid-cols-2 gap-3 mt-6">
          <div className="bg-surface-container/50 rounded-lg p-3 border border-outline-variant/50 text-center">
            <div className="text-label-sm text-on-surface-variant mb-1">Queue Clearance</div>
            <div className="font-title-sm text-on-surface">{record.clearance_time_s}s</div>
          </div>
          <div className="bg-surface-container/50 rounded-lg p-3 border border-outline-variant/50 text-center">
            <div className="text-label-sm text-on-surface-variant mb-1">Vehicles Detected</div>
            <div className="font-title-sm text-on-surface">{record.vehicle_count}</div>
          </div>
        </div>
        {record.safety_buffer_s != null && (
          <div className="text-[11px] text-on-surface-variant/70 text-center mt-4">
            Includes {record.safety_buffer_s}s calibrated safety buffer.
          </div>
        )}
      </div>
    </div>
  );
}
