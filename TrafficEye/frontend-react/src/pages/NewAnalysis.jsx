import { useEffect, useRef, useState } from 'react';
import { apiPostForm, API_BASE } from '../api';
import { useAnalysis } from '../context/AnalysisContext';
import { useAuth } from '../context/AuthContext';
import Icon from '../components/Icon';
import RecommendationCard from '../components/stitch/RecommendationCard';
import StatTiles from '../components/stitch/StatTiles';
import VehicleBreakdownCard from '../components/stitch/VehicleBreakdownCard';
import OrderedQueueTable from '../components/OrderedQueueTable';

// Structure reproduced from the verified Stitch "New Analysis" screen
// (e2037fda4d774d4ba76a7c06cc5f73ca): a large upload zone (left/main) and a
// compact "Analysis Information" panel (right) holding just date/time.
//
// CORRECTIVE CHANGE: Location Name, Road, Direction, GPS Latitude/Longitude,
// "Use My Current Location", and the Location Preview map have been removed
// from this screen — they don't exist in the verified Stitch design.
//
// Analysis Date/Time are no longer officer-editable inputs: the moment a
// video is selected, the machine's LOCAL date/time is captured and shown as
// a read-only display (re-captured on every new selection). Built from
// getFullYear()/getMonth()/getDate()/getHours()/getMinutes() - deliberately
// not toISOString(), which is UTC and would show the wrong calendar date
// for local times behind UTC.
//
// IN-PLACE, TWO-STAGE FLOW (frontend only):
//   1. select video → raw local preview + "Waiting for analysis" placeholders
//   2. "Analyze Traffic" → exactly ONE POST /api/analysis
//   3. as SOON as the research response returns → metric cards
//      (total_detected_vehicles / vehicle_count / queue_length_m /
//      clearance_time_s / suggested_green_s / traffic_level /
//      vehicle_breakdown / ordered_queue) fill immediately. They are NEVER
//      held waiting for the annotated video.
//   4. the annotated (green-box + class-name) MP4 is rendered by a backend
//      BACKGROUND task; while it renders, the RAW preview stays visible with
//      "Preparing detection overlay…". The page polls
//      GET /api/analysis-video-ready/{filename} (~1s) — NEVER re-POSTs — and
//      swaps in the annotated video the instant the finished file exists.
// NORMAL videos stay on this page; an upload the backend flags with
// emergency_detected === true AUTO-opens the Emergency Alert page. No research
// value is computed in React.
function captureLocalTimestamp() {
  const d = new Date();
  const year = d.getFullYear();
  const month = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  const hours = String(d.getHours()).padStart(2, '0');
  const minutes = String(d.getMinutes()).padStart(2, '0');
  return {
    // Sent to the backend as recorded_date - kept as YYYY-MM-DD (the same
    // format the old <input type="date"> always produced) so existing
    // stored records, and History's date-range string comparison, stay
    // consistent with newly-created ones.
    isoDate: `${year}-${month}-${day}`,
    // Shown on screen only, per the required DD/MM/YYYY display format.
    displayDate: `${day}/${month}/${year}`,
    // Sent to the backend as recorded_time, and also what's displayed
    // (HH:MM already matches the required display format).
    time: `${hours}:${minutes}`,
  };
}

// Poll ceiling for annotated-video readiness: ~1s interval, so 600 = 10 min.
// The background encode runs a fresh YOLO pass per sampled frame, so a long
// source clip can take a few minutes; this stays generous without polling
// forever if the render truly failed.
const OVERLAY_POLL_MAX = 600;

export default function NewAnalysis({ onOpenResults, onOpenEmergency }) {
  const fileInputRef = useRef(null);
  const { refresh } = useAnalysis();
  const { token } = useAuth();

  const [fileName, setFileName] = useState('');
  const [dragOver, setDragOver] = useState(false);
  const [captured, setCaptured] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);
  // researchResultReady: the POST /api/analysis response (real metrics).
  const [result, setResult] = useState(null);
  // annotatedVideoReady: the background-rendered overlay MP4 is confirmed on
  // disk. Deliberately separate from `result` — metric cards never wait on it.
  const [annotatedReady, setAnnotatedReady] = useState(false);
  const [overlayFailed, setOverlayFailed] = useState(false);
  // Object URL for an instant local preview of the selected file, shown until
  // the annotated video is confirmed ready. Revoked in the effect below.
  const [previewUrl, setPreviewUrl] = useState(null);
  // Seconds elapsed since the current analysis POST started — shown in the
  // "Analyzing…" banner so a long wait on a big clip reads as progress rather
  // than a hang. Display-only: never affects the request, the response, or any
  // computed value.
  const [analysisElapsedS, setAnalysisElapsedS] = useState(0);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  // Tick the elapsed-time readout only while the single POST /api/analysis is
  // in flight. Stops the moment the result arrives (or the request ends).
  useEffect(() => {
    if (!submitting || result) return undefined;
    setAnalysisElapsedS(0);
    const startedAt = Date.now();
    const id = setInterval(() => {
      setAnalysisElapsedS(Math.floor((Date.now() - startedAt) / 1000));
    }, 1000);
    return () => clearInterval(id);
  }, [submitting, result]);

  // Poll ONLY for annotated-video readiness (never re-calls /api/analysis).
  // Runs once per completed analysis; stops on ready, on timeout, on a new
  // file selection (result → null), and on unmount.
  const annotatedName = result?.annotated_video_url
    ? result.annotated_video_url.split('/').pop()
    : null;
  useEffect(() => {
    if (!annotatedName || annotatedReady || overlayFailed) return undefined;
    let attempts = 0;
    let cancelled = false;
    let id;
    // One readiness probe. Runs immediately once (so a quickly-rendered overlay
    // swaps in without waiting a full poll interval first) and then on each
    // ~1s tick. This only affects when the annotated video appears; the metric
    // cards are already rendered from the POST response by this point.
    const probe = async () => {
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
          // Backend signalled the render gave up (unreadable source, ffmpeg
          // failure, or an interrupted render it could not resume). Stop
          // polling now and fall back to the raw upload instead of spinning
          // for the full ~10-minute poll ceiling. Older backends omit this
          // field, so this stays a no-op there.
          if (body.failed) {
            setOverlayFailed(true);
            clearInterval(id);
            return;
          }
        }
      } catch {
        /* transient — try again on the next tick */
      }
      if (!cancelled && attempts >= OVERLAY_POLL_MAX) {
        setOverlayFailed(true);
        clearInterval(id);
      }
    };
    probe();
    id = setInterval(probe, 1000);
    return () => { cancelled = true; clearInterval(id); };
  }, [annotatedName, annotatedReady, overlayFailed]);

  function resetOutputs() {
    setResult(null);
    setAnnotatedReady(false);
    setOverlayFailed(false);
  }

  function pickFile(file) {
    if (!file) return;
    setFileName(file.name);
    // A new selection invalidates any previous result / preview / overlay.
    resetOutputs();
    setError(null);
    setPreviewUrl(URL.createObjectURL(file));
    const dt = new DataTransfer();
    dt.items.add(file);
    fileInputRef.current.files = dt.files;
    // Re-captured on every selection, including replacing a previously
    // selected video with a different one.
    setCaptured(captureLocalTimestamp());
  }

  async function handleSubmit(e) {
    e.preventDefault();
    setError(null);

    if (!fileInputRef.current.files[0] || !captured) {
      setError('Please select a video file.');
      return;
    }

    setSubmitting(true);
    resetOutputs();

    const uploadedFile = fileInputRef.current.files[0];

    try {
      const fd = new FormData();
      fd.append('file', uploadedFile);
      fd.append('recorded_date', captured.isoDate);
      fd.append('recorded_time', captured.time);
      // EXACTLY ONE backend request for the whole flow.
      const record = await apiPostForm('/api/analysis', fd, token);
      // Research metrics are ready NOW — render them immediately. The
      // annotated video is still rendering in the backend background task;
      // the readiness effect above polls for it separately.
      setResult(record);
      refresh();

      // Open the Emergency Alert screen when the backend's own emergency
      // analysis found a real ROI Ambulance (record.emergency_detected).
      if (record?.emergency_detected === true && onOpenEmergency) {
        onOpenEmergency(record);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setSubmitting(false);
    }
  }

  const hasQueueDetail = Array.isArray(result?.ordered_queue) && result.ordered_queue.length > 0;
  const showAnnotated = !!result?.annotated_video_url && annotatedReady;

  // Loading-feedback text only. Adds a running seconds count once past a couple
  // of seconds, and a reassurance line for long clips, so the officer can see
  // it is still working. Purely cosmetic.
  const analyzingMsg =
    analysisElapsedS >= 20
      ? `Analyzing traffic — running the calculation… (${analysisElapsedS}s · longer videos take more time)`
      : `Analyzing traffic — running the calculation…${analysisElapsedS >= 3 ? ` (${analysisElapsedS}s)` : ''}`;

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <div>
        <h2 className="font-headline-lg text-headline-lg text-on-surface mb-1">New Traffic Analysis</h2>
      </div>

      <form onSubmit={handleSubmit} className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-8 flex flex-col gap-4">
          {/* Kept mounted at all times so fileInputRef stays valid whether the
              dropzone, the raw preview, or the annotated video is showing. */}
          <input ref={fileInputRef} type="file" accept="video/*" className="hidden" onChange={e => pickFile(e.target.files[0])} />

          {showAnnotated ? (
            <div className="bg-surface-container rounded-lg border border-[#1E293B] overflow-hidden">
              <div className="p-3 border-b border-[#1E293B] flex items-center justify-between">
                <span className="font-title-sm text-title-sm text-on-surface flex items-center gap-2">
                  <Icon name="movie" className="text-primary" /> Annotated Analysis Video
                </span>
                <button type="button" onClick={() => fileInputRef.current?.click()} className="text-label-sm font-label-sm text-primary hover:text-primary-fixed flex items-center gap-1">
                  <Icon name="cached" className="text-[16px]" /> New Video
                </button>
              </div>
              <video key={result.annotated_video_url} src={`${API_BASE}${result.annotated_video_url}`} controls preload="metadata" className="w-full aspect-video bg-black" />
              <div className="p-2 text-center text-label-sm font-label-sm text-tertiary-fixed flex items-center justify-center gap-2">
                <Icon name="check_circle" className="text-[16px]" /> Detection overlay ready — green boxes + vehicle class names
              </div>
            </div>
          ) : previewUrl ? (
            <div className="bg-surface-container rounded-lg border border-[#1E293B] overflow-hidden">
              <div className="p-3 border-b border-[#1E293B] flex items-center justify-between">
                <span className="font-title-sm text-title-sm text-on-surface flex items-center gap-2">
                  <Icon name="videocam" className="text-primary" /> Raw Video Preview
                </span>
                <button type="button" onClick={() => fileInputRef.current?.click()} className="text-label-sm font-label-sm text-primary hover:text-primary-fixed flex items-center gap-1">
                  <Icon name="cached" className="text-[16px]" /> Change
                </button>
              </div>
              <video key={previewUrl} src={previewUrl} controls muted preload="metadata" className="w-full aspect-video bg-black" />
              {submitting && !result && (
                <div className="p-2 text-center text-label-sm font-label-sm text-on-surface-variant flex items-center justify-center gap-2">
                  <Icon name="progress_activity" className="animate-spin text-[16px]" />
                  {analyzingMsg}
                </div>
              )}
              {result && !annotatedReady && !overlayFailed && (
                <div className="p-2 text-center text-label-sm font-label-sm text-on-surface-variant flex items-center justify-center gap-2">
                  <Icon name="progress_activity" className="animate-spin text-[16px]" />
                  Preparing detection overlay… (green boxes + vehicle names)
                </div>
              )}
              {result && overlayFailed && (
                <div className="p-2 text-center text-label-sm font-label-sm text-on-surface-variant flex items-center justify-center gap-2">
                  <Icon name="info" className="text-[16px]" />
                  Detection overlay could not be prepared — showing the raw upload.
                </div>
              )}
            </div>
          ) : (
            <div
              onClick={() => fileInputRef.current?.click()}
              onDragOver={e => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={e => {
                e.preventDefault();
                setDragOver(false);
                if (e.dataTransfer.files[0]) pickFile(e.dataTransfer.files[0]);
              }}
              className={
                'bg-surface-container border-2 border-dashed rounded-lg p-6 flex flex-col items-center justify-center text-center h-96 transition-colors cursor-pointer group ' +
                (dragOver ? 'border-primary' : fileName ? 'border-tertiary-container' : 'border-[#1E293B] hover:border-primary')
              }
            >
              <div className="w-20 h-20 rounded-full bg-surface-container-high flex items-center justify-center mb-4 group-hover:bg-primary/10 transition-colors">
                <Icon name={fileName ? 'movie' : 'videocam'} className={'text-4xl transition-colors ' + (fileName ? 'text-tertiary-container' : 'text-on-surface-variant group-hover:text-primary')} />
              </div>
              <h3 className="font-title-sm text-title-sm text-on-surface mb-2">{fileName ? 'Video Ready' : 'Provide Traffic Video'}</h3>
              <p className="font-body-md text-body-md text-on-surface-variant">{fileName || 'MP4, AVI, MOV, MKV supported — click or drop to upload'}</p>
            </div>
          )}
        </div>

        <div className="lg:col-span-4 flex flex-col gap-6">
          <div className="bg-surface-container border border-[#1E293B] rounded-lg p-6 flex-1">
            <h3 className="font-title-sm text-title-sm text-on-surface mb-6 flex items-center gap-2">
              <Icon name="event" className="text-primary text-xl" />
              Analysis Information
            </h3>
            <div className="space-y-4">
              {/* Read-only display, not a form input: the officer cannot
                  open a picker, type, or otherwise edit these — they're
                  fully derived from captureLocalTimestamp() on file
                  selection. */}
              <div className="bg-surface-container-low border border-outline-variant rounded p-4">
                <div className="font-label-caps text-label-caps text-on-surface-variant mb-1 uppercase tracking-wider">Analysis Date</div>
                <div className="font-title-sm text-title-sm text-on-surface">{captured ? captured.displayDate : '—'}</div>
              </div>
              <div className="bg-surface-container-low border border-outline-variant rounded p-4">
                <div className="font-label-caps text-label-caps text-on-surface-variant mb-1 uppercase tracking-wider">Analysis Time</div>
                <div className="font-title-sm text-title-sm text-on-surface">{captured ? captured.time : '—'}</div>
              </div>
            </div>
          </div>
        </div>

        <div className="lg:col-span-12 bg-surface-container-low border border-[#1E293B] rounded-lg p-4 flex items-start gap-4">
          <Icon name="info" className="text-tertiary mt-0.5" />
          <p className="font-body-md text-body-md text-on-surface-variant">
            The uploaded video will be analyzed for vehicle type, queue position, queue clearance and recommended green duration. Metrics appear as soon as the calculation finishes; the annotated overlay video follows a few seconds later.
          </p>
        </div>

        {error && (
          <div className="lg:col-span-12 flex items-center gap-2 bg-error-container/20 border border-error/40 text-error rounded-lg px-4 py-3 text-body-md font-body-md">
            <Icon name="warning" className="text-[18px]" />
            {error}
          </div>
        )}

        <div className="lg:col-span-12 flex justify-end pt-4 border-t border-outline-variant">
          <button
            type="submit"
            disabled={submitting || !!result}
            className="bg-[#38BDF8] hover:bg-[#7bd0ff] disabled:opacity-60 disabled:cursor-not-allowed text-[#08111F] font-title-sm text-title-sm px-6 py-3 rounded flex items-center gap-2 transition-colors"
          >
            <Icon name={submitting ? 'progress_activity' : 'qr_code_2'} className={submitting ? 'animate-spin' : ''} />
            {submitting ? 'Analyzing…' : result ? 'Analysis complete' : 'Analyze Traffic'}
          </button>
        </div>
      </form>

      {/* Waiting-for-analysis placeholder: a file is selected but the single
          POST hasn't produced a result yet. */}
      {fileName && !result && (
        <div className="space-y-4">
          <div className="flex items-center gap-3 bg-surface-container-low border border-[#1E293B] px-4 py-3 rounded-lg text-on-surface-variant">
            <Icon name={submitting ? 'progress_activity' : 'hourglass_empty'} className={submitting ? 'animate-spin text-primary' : 'text-on-surface-variant'} />
            <span className="font-title-sm text-title-sm">{submitting ? analyzingMsg : 'Waiting for analysis'}</span>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
            {['Total Detected Vehicles', 'ROI Queue Vehicles', 'Queue Length', 'Clearance Est.', 'Suggested Green'].map(l => (
              <div key={l} className="bg-surface-container p-4 rounded-xl border border-outline-variant/50 flex flex-col justify-between gap-2 opacity-60">
                <span className="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wider">{l}</span>
                <span className="font-headline-lg text-headline-lg text-on-surface-variant">—</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {result && (
        <div className="space-y-6">
          <div className="flex items-center gap-3 bg-tertiary-container/10 border border-tertiary-container/30 px-4 py-3 rounded-lg">
            <Icon name={annotatedReady || overlayFailed ? 'check_circle' : 'model_training'} className="text-tertiary-container" />
            <span className="font-title-sm text-title-sm text-tertiary-container font-semibold">
              Analysis complete
              {!annotatedReady && !overlayFailed && ' · Preparing detection overlay…'}
              {annotatedReady && ' · Detection overlay ready'}
              {overlayFailed && ' · Detection overlay unavailable'}
            </span>
          </div>

          {result.emergency_detected === true && onOpenEmergency && (
            <button
              onClick={() => onOpenEmergency(result)}
              className="w-full flex items-center justify-center gap-2 bg-error text-on-error py-3 rounded-lg font-title-sm text-title-sm font-bold animate-pulse-fast"
            >
              <Icon name="emergency" fill />
              Emergency Vehicle Detected — View Emergency Alert
            </button>
          )}

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-1">
              <RecommendationCard record={result} />
            </div>
            <div className="lg:col-span-2 flex flex-col gap-6">
              <StatTiles record={result} />
              <VehicleBreakdownCard breakdown={result.vehicle_breakdown} />
            </div>
          </div>

          <div className="bg-surface-container rounded-xl border border-outline-variant overflow-hidden flex flex-col">
            <div className="p-4 border-b border-outline-variant/50 bg-surface-container-lowest/50 flex items-center justify-between">
              <h3 className="font-title-sm text-title-sm text-on-surface flex items-center gap-2">
                <Icon name="format_list_numbered" className="text-secondary" />
                Ordered Queue
              </h3>
              {hasQueueDetail && result.average_gap_m != null && (
                <span className="text-label-sm font-label-sm text-on-surface-variant">
                  Avg. Gap: {result.average_gap_m} m{result.safety_buffer_s != null && ` · Safety Buffer: ${result.safety_buffer_s} s`}
                </span>
              )}
            </div>
            {hasQueueDetail ? (
              <OrderedQueueTable orderedQueue={result.ordered_queue} />
            ) : (
              <div className="flex flex-col items-center justify-center gap-2 text-on-surface-variant text-body-md py-10 px-6 text-center">
                <Icon name="info" />
                Detailed per-vehicle queue data is not available for this analysis.
              </div>
            )}
          </div>

          {onOpenResults && (
            <button
              onClick={() => onOpenResults(result)}
              className="text-primary hover:text-primary-fixed transition-colors font-title-sm text-title-sm flex items-center gap-1"
            >
              View Full Results <Icon name="arrow_forward" className="text-[18px]" />
            </button>
          )}
        </div>
      )}
    </div>
  );
}
