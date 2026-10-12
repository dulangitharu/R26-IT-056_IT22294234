import { useMemo } from 'react';
import { useAnalysis } from '../context/AnalysisContext';
import Icon from '../components/Icon';
import RecommendationCard from '../components/stitch/RecommendationCard';

// Structure reproduced from the verified Stitch "Operations Overview"
// screen (70e19e2fbea741718b39eaa4f2c7664d): four KPI cards, then a
// two-column "Recent Traffic Analysis" + "AI Recommendation" layout.
// The old Trend Chart / Location Map / Locations Summary panels (which have
// no counterpart in the verified Stitch screen) have been dropped.
function KpiCard({ label, value, sub, icon, unavailable }) {
  return (
    <div className="bg-surface-container border border-outline-variant rounded-xl p-5 relative overflow-hidden group">
      <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
        <Icon name={icon} className="text-6xl text-primary" />
      </div>
      <div className="flex flex-col h-full relative z-10">
        <span className="font-label-caps text-label-caps text-on-surface-variant uppercase tracking-wider mb-2">{label}</span>
        <div className="flex items-baseline gap-2 mt-auto">
          <span className={'font-display-metrics text-display-metrics ' + (unavailable ? 'text-on-surface-variant/50' : 'text-on-surface')}>{value}</span>
          {sub && <span className="font-label-sm text-label-sm text-on-surface-variant">{sub}</span>}
        </div>
      </div>
    </div>
  );
}

const LEVEL_BADGE = {
  CLEAR: 'bg-tertiary-container/20 text-tertiary-fixed-dim border-tertiary-container/30',
  LOW: 'bg-tertiary-container/20 text-tertiary-fixed-dim border-tertiary-container/30',
  MEDIUM: 'bg-secondary-container text-on-secondary-container border-outline-variant',
  HIGH: 'bg-error/20 text-error border-error/30',
};

export default function Overview({ onOpenResults }) {
  const { records, stats, loading, error, refresh } = useAnalysis();

  const totalVehiclesDetected = useMemo(
    () => records.reduce((sum, r) => sum + (r.vehicle_count || 0), 0),
    [records]
  );
  const latest = records[0];

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      <div className="flex justify-between items-end">
        <div>
          <h2 className="font-headline-lg text-headline-lg font-semibold text-on-surface tracking-tight">Traffic Operations Overview</h2>
          <p className="text-on-surface-variant mt-1">Real-time command and control metrics.</p>
        </div>
        <button onClick={refresh} className="text-label-sm font-label-sm text-on-surface-variant flex items-center gap-1.5 bg-surface-container-high px-3 py-1.5 rounded-full border border-outline-variant hover:text-on-surface transition-colors">
          <Icon name="refresh" className="text-[16px]" />
          Refresh
        </button>
      </div>

      {loading ? (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {[0, 1, 2, 3].map(i => <div key={i} className="h-[104px] rounded-xl bg-surface-container animate-pulse" />)}
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <KpiCard label="Total Analyses" value={stats ? stats.total_cases : '--'} icon="data_usage" />
          <KpiCard label="Total Vehicles Detected" value={records.length ? totalVehiclesDetected : '--'} icon="directions_car" />
          <KpiCard label="Latest Vehicle Count" value={latest?.vehicle_count ?? '--'} icon="speed" />
          <KpiCard
            label="Emergency Detections"
            value={stats?.emergency_detection_count ?? 0}
            sub={stats?.latest_emergency_date ? `Latest: ${stats.latest_emergency_date}${stats.latest_emergency_time ? ' ' + stats.latest_emergency_time : ''}` : undefined}
            icon="warning"
          />
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 bg-[#162032] border border-[#1E293B] rounded-xl flex flex-col">
          <div className="p-5 border-b border-[#1E293B] flex justify-between items-center">
            <div className="flex items-center gap-2">
              <Icon name="view_list" className="text-primary" />
              <h3 className="font-title-sm text-title-sm text-on-surface">Recent Traffic Analysis</h3>
            </div>
          </div>
          <div className="flex-1 overflow-hidden flex flex-col">
            <div className="grid grid-cols-12 gap-4 px-5 py-3 bg-surface-container-highest/30 border-b border-[#1E293B] font-label-caps text-label-caps text-on-surface-variant uppercase">
              <div className="col-span-4">Date</div>
              <div className="col-span-3">Time</div>
              <div className="col-span-3">Level</div>
              <div className="col-span-2 text-right">Action</div>
            </div>
            <div className="flex-1 overflow-y-auto max-h-[420px]">
              {error ? (
                <div className="p-5 text-error text-body-md">{error}</div>
              ) : records.length === 0 ? (
                <div className="p-8 text-center text-on-surface-variant flex flex-col items-center gap-2">
                  <Icon name="inbox" />
                  No analysis records yet.
                </div>
              ) : records.slice(0, 8).map(r => (
                <div key={r._id} className="grid grid-cols-12 gap-4 px-5 py-4 border-b border-[#1E293B] hover:bg-surface-variant/20 transition-colors items-center group">
                  <div className="col-span-4 flex items-center gap-3">
                    <div className="w-8 h-8 rounded bg-surface-container-highest flex items-center justify-center flex-shrink-0">
                      <Icon name="location_on" className="text-on-surface-variant text-sm" />
                    </div>
                    <span className="font-body-md text-on-surface truncate">{r.recorded_date}</span>
                  </div>
                  <div className="col-span-3 font-body-md text-on-surface-variant">{r.recorded_time}</div>
                  <div className="col-span-3">
                    <span className={`inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold border uppercase tracking-wide ${LEVEL_BADGE[r.traffic_level] || LEVEL_BADGE.LOW}`}>
                      {r.traffic_level}
                    </span>
                  </div>
                  <div className="col-span-2 text-right opacity-0 group-hover:opacity-100 transition-opacity">
                    {onOpenResults && (
                      <button onClick={() => onOpenResults(r._id)} className="text-label-sm font-label-sm text-primary border border-primary/30 px-3 py-1 rounded hover:bg-primary/10 transition-colors">
                        View
                      </button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div>
          {loading ? (
            <div className="h-full min-h-[400px] rounded-xl bg-surface-container animate-pulse" />
          ) : latest ? (
            <RecommendationCard record={latest} />
          ) : (
            <div className="bg-surface-container rounded-xl border border-outline-variant h-full min-h-[300px] flex flex-col items-center justify-center gap-2 text-on-surface-variant p-6 text-center">
              <Icon name="tips_and_updates" />
              No analysis recorded yet. Run one from &quot;New Analysis&quot;.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
