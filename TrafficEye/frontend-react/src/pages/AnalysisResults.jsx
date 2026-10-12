import Icon from '../components/Icon';
import { API_BASE } from '../api';
import RecommendationCard from '../components/stitch/RecommendationCard';
import StatTiles from '../components/stitch/StatTiles';
import VehicleBreakdownCard from '../components/stitch/VehicleBreakdownCard';
import OrderedQueueTable from '../components/OrderedQueueTable';

// Structure reproduced from the verified Stitch "Analysis Results" screen
// (71b3c968b2a4489b87dc80714c7f97be). Stitch's left panel is a fabricated
// "Traffic Intelligence Feed" camera view with invented bounding boxes and
// confidence percentages (e.g. "CAR • 98%") — that's replaced here with the
// REAL annotated recording the backend now generates (real per-frame YOLO
// boxes + class labels, no confidence, no LIVE/ROI-ACTIVE chrome).
// `annotated_video_url` is response-only (like ordered_queue/emergency_*
// below) — it only exists on the response returned at submission time, so
// it is never available for History → View Details (GET
// /api/analysis/{id}), which is why that path hides this whole region via
// `hideSessionSummary` rather than showing a broken/empty video panel.
//
// Structural gap (unchanged from prior phases): ordered_queue,
// average_gap_m, safety_buffer_s, and every emergency_* field are computed
// per-request in main.py but are NOT part of the persisted AnalysisRecord —
// they only exist on the response returned at submission time. A record
// reloaded later via GET /api/analysis/{id} (History → View Details) will
// not have them; this page detects that and says so rather than fabricating
// a table.
export default function AnalysisResults({ record, onBack, hideSessionSummary = false }) {
  if (!record) {
    return (
      <div className="max-w-7xl mx-auto">
        <div className="bg-surface-container-low rounded-xl border border-outline-variant/50 p-10 flex flex-col items-center gap-2 text-on-surface-variant">
          <Icon name="info" />
          No record selected.
        </div>
      </div>
    );
  }

  const hasQueueDetail = Array.isArray(record.ordered_queue) && record.ordered_queue.length > 0;
  const createdAt = record.created_at ? new Date(record.created_at).toLocaleString() : null;

  return (
    <div className="max-w-[1800px] mx-auto space-y-6">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 bg-surface-container-low p-6 rounded-xl border border-outline-variant/50">
        <div>
          {onBack && (
            <button onClick={onBack} className="flex items-center gap-1 text-label-sm font-label-sm text-on-surface-variant hover:text-on-surface transition-colors mb-2">
              <Icon name="arrow_back" className="text-[16px]" /> Back
            </button>
          )}
          <h1 className="font-headline-lg text-headline-lg text-on-surface mb-1 flex items-center gap-3">
            Traffic Analysis <Icon name="check_circle" fill className="text-tertiary-fixed" />
          </h1>
          <p className="font-body-lg text-body-lg text-on-surface-variant">
            {record.location_name} — {record.road} ({record.direction}){createdAt ? ` • ${createdAt}` : ''}
          </p>
        </div>
        <div className="flex items-center gap-3 bg-tertiary-container/10 border border-tertiary-container/30 px-4 py-2 rounded-lg">
          <Icon name="model_training" className="text-tertiary-container" />
          <span className="font-title-sm text-title-sm text-tertiary-container font-semibold">Recommendation Ready</span>
        </div>
      </div>

      {hideSessionSummary ? (
        <div className="max-w-md">
          <RecommendationCard record={record} />
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          <div className="lg:col-span-8 bg-surface-container rounded-xl border border-outline-variant overflow-hidden flex flex-col shadow-lg shadow-black/20">
            <div className="p-4 border-b border-outline-variant/30 flex justify-between items-center bg-surface-container-lowest/50">
              <h2 className="font-title-sm text-title-sm text-on-surface flex items-center gap-2">
                <Icon name="videocam" className="text-primary" />
                Analyzed Traffic Video
              </h2>
            </div>
            {record.annotated_video_url ? (
              <video
                controls
                className="w-full aspect-video bg-black"
                src={`${API_BASE}${record.annotated_video_url}`}
              />
            ) : (
              <div className="aspect-video flex flex-col items-center justify-center gap-2 text-on-surface-variant text-body-md text-center px-6">
                <Icon name="videocam_off" />
                Annotated video is not available for this analysis.
              </div>
            )}
          </div>

          <div className="lg:col-span-4">
            <RecommendationCard record={record} />
          </div>
        </div>
      )}

      <StatTiles record={record} />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <VehicleBreakdownCard breakdown={record.vehicle_breakdown} />

        <div className="lg:col-span-2 bg-surface-container rounded-xl border border-outline-variant overflow-hidden flex flex-col">
          <div className="p-4 border-b border-outline-variant/50 bg-surface-container-lowest/50 flex items-center justify-between">
            <h3 className="font-title-sm text-title-sm text-on-surface flex items-center gap-2">
              <Icon name="format_list_numbered" className="text-secondary" />
              Ordered Queue
            </h3>
            {hasQueueDetail && record.average_gap_m != null && (
              <span className="text-label-sm font-label-sm text-on-surface-variant">
                Avg. Gap: {record.average_gap_m} m{record.safety_buffer_s != null && ` · Safety Buffer: ${record.safety_buffer_s} s`}
              </span>
            )}
          </div>
          {hasQueueDetail ? (
            <OrderedQueueTable orderedQueue={record.ordered_queue} />
          ) : (
            <div className="flex flex-col items-center justify-center gap-2 text-on-surface-variant text-body-md py-10 px-6 text-center">
              <Icon name="info" />
              Detailed per-vehicle queue data is only returned at the moment an analysis is submitted and is not
              persisted for later retrieval — not available for this historical record.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
