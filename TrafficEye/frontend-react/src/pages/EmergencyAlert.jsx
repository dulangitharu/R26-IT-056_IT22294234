import { useEffect, useRef, useState } from 'react';
import Icon from '../components/Icon';
import { API_BASE } from '../api';
import StatTiles from '../components/stitch/StatTiles';
import VehicleBreakdownCard from '../components/stitch/VehicleBreakdownCard';
import OrderedQueueTable from '../components/OrderedQueueTable';

// The annotated overlay is produced by a backend background render. Poll
// frequently AND check once immediately on mount, so a clip that already
// finished rendering (or finishes mid-wait) is shown with no artificial delay.
// OVERLAY_POLL_INTERVAL_MS * OVERLAY_POLL_MAX ≈ 6 min ceiling: the backend
// caps a single render at ANNOTATED_RENDER_DEADLINE_S (240s), so if nothing
// has appeared well past that the render genuinely failed - fall through to
// the "could not be prepared" state instead of spinning forever.
const OVERLAY_POLL_INTERVAL_MS = 700;
const OVERLAY_POLL_MAX = 520;

// Structure reproduced from the verified Stitch "Emergency Alert" screen
// (6be914821a2e481e88cb28e164c49256), with everything not backed by real
// data or real backend capability deliberately stripped:
//   - no live camera feed / bounding-box overlay beyond the REAL annotated
//     recording below (same response-only annotated_video_url/API_BASE
//     pattern Analysis Results uses - no second video, no re-run YOLO)
//   - no emergency vehicle type / queue position / position group
//   - no confidence score, no "CAM_04 / 720p 60fps" stream metadata
//   - no fabricated GPS overlay
//   - no "Acknowledge Advisory" action — nothing persists an acknowledgment
//     server-side, so a button here would do nothing real
// The Incident Timeline keeps Stitch's step labels (they describe the real
// pipeline that produced this record) but drops the implication of
// per-step timestamps, since none are stored.
const TIMELINE = [
  'Traffic analysis started',
  'Emergency vehicle detected',
  'Emergency clearance calculated',
  'Priority green-time recommendation generated',
  'Analysis completed for officer review',
];

export default function EmergencyAlert({ record, onBack, onViewResults }) {
  // ── Annotated-overlay readiness ──────────────────────────────────────────
  // The POST /api/analysis response RESERVES record.annotated_video_url before
  // the background task has finished rendering the MP4 (record.annotated_video_
  // pending === true). Mounting <video src=that> immediately gives a broken /
  // 404 element stuck at 0:00 that never recovers. Instead: mount NO <video>
  // until GET /api/analysis-video-ready/{filename} confirms the finished file
  // exists, then mount it once (fresh key + .load()). Same real backend MP4
  // and the SAME existing readiness endpoint NewAnalysis already uses - no
  // second video, no re-detection, no extra POST, no backend change.
  const annotatedName = record?.annotated_video_url
    ? record.annotated_video_url.split('/').pop()
    : null;
  const [annotatedReady, setAnnotatedReady] = useState(
    () => !!record?.annotated_video_url && !record?.annotated_video_pending,
  );
  const [overlayFailed, setOverlayFailed] = useState(false);
  const videoRef = useRef(null);

  useEffect(() => {
    if (!annotatedName || annotatedReady || overlayFailed) return undefined;
    let cancelled = false;
    let attempts = 0;
    let id;

    const checkReady = async () => {
      attempts += 1;
      try {
        const res = await fetch(`${API_BASE}/api/analysis-video-ready/${annotatedName}`);
        if (!cancelled && res.ok) {
          const body = await res.json();
          if (body.ready) {
            setAnnotatedReady(true);
            clearInterval(id);
            return;
          }
        }
      } catch {
        /* transient network blip — try again on the next tick */
      }
      if (!cancelled && attempts >= OVERLAY_POLL_MAX) {
        setOverlayFailed(true);
        clearInterval(id);
      }
    };

    // Check straight away — no dead first interval — then keep polling.
    checkReady();
    id = setInterval(checkReady, OVERLAY_POLL_INTERVAL_MS);
    return () => { cancelled = true; clearInterval(id); };
  }, [annotatedName, annotatedReady, overlayFailed]);

  // Force the freshly-mounted <video> to (re)fetch the now-existing file the
  // instant readiness flips — defeats any stale/earlier failed media request.
  useEffect(() => {
    if (annotatedReady && videoRef.current) videoRef.current.load();
  }, [annotatedReady]);

  // Constant once ready (NOT continuously cache-busted): a distinct URL from
  // any bare-URL 404 the browser might have cached from an earlier attempt.
  const emergencyVideoSrc = record?.annotated_video_url
    ? `${API_BASE}${record.annotated_video_url}?ready=1`
    : null;

  // Render only when the backend genuinely flagged an emergency. Every value
  // below comes straight from the real backend POST response on `record`;
  // nothing here mutates record.emergency_detected or any research field.
  if (!record || !record.emergency_detected) {
    return (
      <div className="max-w-5xl mx-auto space-y-6">
        {onBack && (
          <button onClick={onBack} className="flex items-center gap-1 text-label-sm font-label-sm text-on-surface-variant hover:text-on-surface transition-colors">
            <Icon name="arrow_back" className="text-[16px]" /> Back
          </button>
        )}
        <div className="bg-surface-container-low rounded-xl border border-outline-variant/50 p-10 flex flex-col items-center gap-2 text-on-surface-variant">
          <Icon name="info" />
          No emergency detected for this record.
        </div>
      </div>
    );
  }

  // Vehicle breakdown and ordered queue are shown exactly as the backend
  // returned them - no label is renamed, added, or removed here.
  const breakdown = record.vehicle_breakdown || {};
  const orderedQueue = Array.isArray(record.ordered_queue) ? record.ordered_queue : [];
  const showQueueDetails = Object.keys(breakdown).length > 0 || orderedQueue.length > 0;

  const hasTiming = record.emergency_clearance_time_s != null && record.emergency_priority_green_s != null;
  const diff = hasTiming && record.suggested_green_s != null
    ? Math.round((record.suggested_green_s - record.emergency_priority_green_s) * 100) / 100
    : null;

  // ── Video-first reveal gate ──────────────────────────────────────────────
  // The annotated emergency video (green boxes + vehicle / ambulance labels)
  // is the headline of this page for the demo, so every analysis-detail
  // section below (Incident Timeline, Priority Advisory, Timing Comparison,
  // Safety Margin, breakdown / ordered queue) stays behind a small loading
  // state until that video is actually ready to play. This ONLY delays
  // rendering of sections that already existed — nothing is recomputed,
  // removed, or changed, and it reuses the SAME annotatedReady / overlayFailed
  // signals the video card already computes (GET /api/analysis-video-ready).
  // Records with no annotated video to wait for (e.g. a genuine emergency
  // reloaded from History, where annotated_video_url is not persisted) and
  // the ~10-min overlay-timeout fallback both reveal the details immediately,
  // so real analysis data is never permanently withheld.
  const detailsRevealed = annotatedReady || overlayFailed || !record.annotated_video_url;

  return (
    <div className="max-w-[1600px] mx-auto">
      <div className="bg-error-container text-on-error-container border border-error rounded-xl py-3 px-6 flex items-center justify-center gap-4 mb-6 relative overflow-hidden">
        <Icon name="warning" fill className="text-error text-3xl animate-pulse-fast" />
        <h2 className="font-headline-md text-headline-md font-bold tracking-widest uppercase">Emergency Vehicle Detected</h2>
        <Icon name="warning" fill className="text-error text-3xl animate-pulse-fast" />
      </div>

      <div className="flex flex-col md:flex-row justify-between items-start md:items-end mb-6 border-b border-outline-variant pb-4 gap-3">
        {onBack && (
          <button onClick={onBack} className="flex items-center gap-1 text-label-sm font-label-sm text-on-surface-variant hover:text-on-surface transition-colors">
            <Icon name="arrow_back" className="text-[16px]" /> Back
          </button>
        )}
        {onViewResults && (
          <button
            onClick={() => onViewResults(record)}
            className="bg-surface-container-high border border-outline-variant text-on-surface px-4 py-2 rounded-lg font-title-sm text-title-sm flex items-center gap-2 hover:bg-surface-bright transition-colors"
          >
            <Icon name="tune" /> View Analysis Details
          </button>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-8 flex flex-col gap-6">
          <div className="bg-surface-container rounded-xl border border-outline-variant overflow-hidden flex flex-col shadow-lg shadow-black/20">
            <div className="p-4 border-b border-outline-variant/30 flex justify-between items-center bg-surface-container-lowest/50">
              <h2 className="font-title-sm text-title-sm text-on-surface flex items-center gap-2">
                <Icon name="videocam" className="text-primary" />
                Analyzed Traffic Video
              </h2>
            </div>
            {!record.annotated_video_url ? (
              <div className="aspect-video flex flex-col items-center justify-center gap-2 text-on-surface-variant text-body-md text-center px-6">
                <Icon name="videocam_off" />
                Annotated video is not available for this analysis.
              </div>
            ) : annotatedReady ? (
              <video
                key={emergencyVideoSrc}
                ref={videoRef}
                controls
                playsInline
                autoPlay
                muted
                preload="auto"
                className="w-full aspect-video bg-black"
                src={emergencyVideoSrc}
              />
            ) : overlayFailed ? (
              <div className="aspect-video flex flex-col items-center justify-center gap-2 text-on-surface-variant text-body-md text-center px-6">
                <Icon name="videocam_off" />
                Emergency detection overlay could not be prepared for this analysis.
              </div>
            ) : (
              <div className="aspect-video flex flex-col items-center justify-center gap-2 text-on-surface-variant text-body-md text-center px-6">
                <Icon name="progress_activity" className="animate-spin" />
                Preparing emergency detection overlay…
                <span className="text-label-sm font-label-sm text-on-surface-variant/70">
                  green boxes + vehicle names — no refresh needed
                </span>
              </div>
            )}
          </div>

          {!detailsRevealed && (
            <div className="bg-surface-container rounded-xl border border-outline-variant p-5 flex items-center gap-3 text-on-surface-variant text-body-md">
              <Icon name="progress_activity" className="animate-spin" />
              Emergency analysis details will appear once the detection video is ready…
            </div>
          )}

          {detailsRevealed && (
          <div className="bg-surface-container rounded-xl border border-outline-variant p-5">
            <h3 className="font-title-sm text-title-sm text-on-surface mb-4 flex items-center gap-2">
              <Icon name="timeline" className="text-outline" /> Incident Timeline
            </h3>
            <div className="flex flex-col gap-3 relative before:absolute before:inset-y-2 before:left-[11px] before:w-0.5 before:bg-outline-variant before:z-0">
              {TIMELINE.map((step, i) => (
                <div key={step} className="flex gap-4 relative z-10">
                  <div className={
                    'w-6 h-6 rounded-full border-2 flex items-center justify-center shrink-0 mt-0.5 ' +
                    (i === 1 ? 'bg-error-container border-error' : i >= 3 ? 'bg-primary-container/20 border-primary' : 'bg-surface-container-highest border-outline-variant')
                  }>
                    {i === 1
                      ? <Icon name="emergency" fill className="text-[12px] text-error" />
                      : <div className={'w-2 h-2 rounded-full ' + (i >= 3 ? 'bg-primary' : 'bg-outline')} />}
                  </div>
                  <p className={'text-sm ' + (i === 1 ? 'text-on-error-container font-medium' : 'text-on-surface')}>{step}</p>
                </div>
              ))}
            </div>
          </div>
          )}
        </div>

        {detailsRevealed && (
        <div className="lg:col-span-4 flex flex-col gap-6">
          <div className="bg-surface-container-highest rounded-xl border border-error p-6 shadow-[0_8px_24px_rgba(0,0,0,0.4)] relative overflow-hidden">
            <div className="absolute top-0 right-0 p-4 opacity-10">
              <Icon name="policy" fill className="text-8xl text-error" />
            </div>
            <div className="flex items-center gap-3 mb-2 relative z-10">
              <Icon name="notifications_active" className="text-error text-2xl animate-pulse-fast" />
              <h2 className="font-headline-md text-headline-md font-bold text-error">Priority Advisory</h2>
            </div>
            {hasTiming ? (
              <div className="relative z-10 mb-2 flex flex-col gap-1">
                <p className="text-on-surface font-medium">Emergency Clearance: {record.emergency_clearance_time_s} s</p>
                <p className="text-on-surface font-medium">Emergency Priority Green: {record.emergency_priority_green_s} s</p>
                {record.emergency_priority && <p className="text-on-surface-variant text-sm">Priority: {record.emergency_priority}</p>}
              </div>
            ) : (
              <p className="relative z-10 mb-2 text-on-surface-variant text-body-md">
                Emergency timing values are only returned at the moment an analysis is submitted and are not
                persisted for later retrieval — not available for this record.
              </p>
            )}
            <p className="font-body-lg text-body-lg text-on-surface relative z-10">
              {record.emergency_recommendation || 'Priority green-time advisory generated for officer review.'}
            </p>
          </div>

          {hasTiming && (
            <div className="bg-surface-container rounded-xl border border-outline-variant p-6 flex flex-col gap-6">
              <h3 className="font-title-sm text-title-sm text-on-surface flex items-center gap-2">
                <Icon name="compare_arrows" className="text-outline" /> Timing Comparison
              </h3>
              <div className="bg-error-container/10 border-l-4 border-error p-4 rounded-r-lg">
                <div className="flex justify-between items-start">
                  <div>
                    <p className="font-label-caps text-label-caps text-error mb-1">Emergency Priority Green</p>
                    <div className="flex items-baseline gap-1">
                      <span className="font-display-metrics text-display-metrics text-error">{record.emergency_priority_green_s}</span>
                      <span className="font-label-sm text-label-sm text-error/80">sec</span>
                    </div>
                  </div>
                  <Icon name="timer" className="text-error opacity-80" />
                </div>
              </div>
              {record.suggested_green_s != null && (
                <div className="bg-surface-container-high border-l-4 border-outline-variant p-4 rounded-r-lg">
                  <div className="flex justify-between items-start">
                    <div>
                      <p className="font-label-caps text-label-caps text-on-surface-variant mb-1">Normal Suggested Green</p>
                      <div className="flex items-baseline gap-1">
                        <span className="font-display-metrics text-display-metrics text-on-surface">{record.suggested_green_s}</span>
                        <span className="font-label-sm text-label-sm text-on-surface-variant">sec</span>
                      </div>
                    </div>
                    <Icon name="hourglass_empty" className="text-outline" />
                  </div>
                  {diff != null && (
                    <div className="mt-3 flex items-center gap-1 text-xs text-primary bg-primary/10 px-2 py-1 rounded w-fit">
                      <Icon name="difference" className="text-[14px]" />
                      Difference: {diff} s
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {record.safety_buffer_s != null && (
            <div className="bg-surface-container rounded-lg border border-outline-variant p-4">
              <p className="font-label-caps text-label-caps text-on-surface-variant mb-1">Safety Margin</p>
              <p className="font-title-sm text-title-sm text-tertiary-fixed font-mono flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-tertiary-fixed inline-block" /> {record.safety_buffer_s} s
              </p>
            </div>
          )}
        </div>
        )}
      </div>

      {detailsRevealed && showQueueDetails && (
        <div className="mt-6 flex flex-col gap-6">
          <StatTiles record={record} />

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <VehicleBreakdownCard breakdown={breakdown} />

            <div className="lg:col-span-2 bg-surface-container rounded-xl border border-outline-variant overflow-hidden flex flex-col">
              <div className="p-4 border-b border-outline-variant/50 bg-surface-container-lowest/50 flex items-center justify-between">
                <h3 className="font-title-sm text-title-sm text-on-surface flex items-center gap-2">
                  <Icon name="format_list_numbered" className="text-secondary" />
                  Ordered Queue
                </h3>
                {record.average_gap_m != null && (
                  <span className="text-label-sm font-label-sm text-on-surface-variant">
                    Avg. Gap: {record.average_gap_m} m{record.safety_buffer_s != null && ` · Safety Buffer: ${record.safety_buffer_s} s`}
                  </span>
                )}
              </div>
              {orderedQueue.length > 0 ? (
                <OrderedQueueTable orderedQueue={orderedQueue} />
              ) : (
                <div className="flex flex-col items-center justify-center gap-2 text-on-surface-variant text-body-md py-10 px-6 text-center">
                  <Icon name="info" />
                  Detailed per-vehicle queue data is only returned at the moment an analysis is submitted.
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
