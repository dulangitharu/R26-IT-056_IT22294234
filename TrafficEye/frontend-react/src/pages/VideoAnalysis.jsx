import { useEffect, useRef, useState } from 'react';
import { Video, UploadCloud, CheckCircle2, Paperclip, Play, Pause, Square, Gauge, TrendingUp, Car } from 'lucide-react';
import { API_BASE, WS_BASE } from '../api';
import { LEVEL_META } from '../levels';
import ErrorState from '../components/ErrorState';

const MAX_CHART = 50;

export default function VideoAnalysis() {
  const fileInputRef = useRef(null);
  const wsRef = useRef(null);
  const canvasRef = useRef(null);
  const chartDataRef = useRef([]);

  const [fileName, setFileName] = useState('');
  const [dragOver, setDragOver] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadReady, setUploadReady] = useState(false);
  const [uploadError, setUploadError] = useState(null);

  const [live, setLive] = useState(false);
  const [paused, setPaused] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [wsError, setWsError] = useState(null);

  const [frame, setFrame] = useState(null);
  const [fps, setFps] = useState(0);
  const [progress, setProgress] = useState({ frame_idx: 0, total: 0 });
  const [level, setLevel] = useState({ count: 0, level: 'CLEAR' });
  const [detections, setDetections] = useState({});

  const [frameCount, setFrameCount] = useState(0);
  const [peakCount, setPeakCount] = useState(0);
  const [totalCount, setTotalCount] = useState(0);

  // Close the socket on unmount so we never leak a connection when the
  // officer navigates away mid-stream.
  useEffect(() => {
    return () => {
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  function pickFile(file) {
    if (!file) return;
    setFileName(file.name);
    uploadFile(file);
  }

  async function uploadFile(file) {
    setUploading(true);
    setUploadError(null);
    setUploadReady(false);
    try {
      const fd = new FormData();
      fd.append('file', file);
      let res;
      try {
        res = await fetch(`${API_BASE}/upload`, { method: 'POST', body: fd });
      } catch {
        throw new Error(`Cannot reach backend at ${API_BASE} — is the server running?`);
      }
      const data = await res.json();
      if (!res.ok || !data.ok) throw new Error(data.detail || 'Upload failed.');
      setUploadReady(true);
    } catch (err) {
      setUploadError(err.message);
    } finally {
      setUploading(false);
    }
  }

  function drawChart() {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.offsetWidth || 300, H = 100;
    canvas.width = W;
    canvas.height = H;
    ctx.clearRect(0, 0, W, H);

    const data = chartDataRef.current;
    if (data.length < 2) return;

    const max = Math.max(...data, 1);
    const step = W / (MAX_CHART - 1);

    const grad = ctx.createLinearGradient(0, 0, 0, H);
    grad.addColorStop(0, 'rgba(37,99,235,.32)');
    grad.addColorStop(1, 'rgba(37,99,235,.02)');

    ctx.beginPath();
    data.forEach((v, i) => {
      const x = i * step;
      const y = H - (v / max) * (H - 8) - 4;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.lineTo((data.length - 1) * step, H);
    ctx.lineTo(0, H);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();

    ctx.beginPath();
    data.forEach((v, i) => {
      const x = i * step;
      const y = H - (v / max) * (H - 8) - 4;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.strokeStyle = '#2563EB';
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  useEffect(() => {
    window.addEventListener('resize', drawChart);
    return () => window.removeEventListener('resize', drawChart);
  }, []);

  function startStream() {
    if (wsRef.current) wsRef.current.close();

    setPaused(false);
    setFrameCount(0);
    setPeakCount(0);
    setTotalCount(0);
    setDetections({});
    setWsError(null);
    chartDataRef.current = [];

    const ws = new WebSocket(`${WS_BASE}/ws/stream`);
    wsRef.current = ws;

    ws.onopen = () => {
      setLive(true);
      ws.send(JSON.stringify({ action: 'start', speed }));
    };

    ws.onmessage = (e) => {
      const d = JSON.parse(e.data);
      if (d.error) {
        setWsError(d.error);
        stopStream();
        return;
      }

      setFrame(d.frame);
      setFps(d.fps);
      setProgress({ frame_idx: d.frame_idx, total: d.total });
      setLevel({ count: d.count, level: d.level });

      setFrameCount(c => c + 1);
      setPeakCount(p => Math.max(p, d.count));
      setTotalCount(t => t + d.count);

      if (d.detections && d.detections.length > 0) {
        const counts = {};
        d.detections.forEach(det => { counts[det.label] = (counts[det.label] || 0) + 1; });
        setDetections(counts);
      } else {
        setDetections({});
      }

      chartDataRef.current.push(d.count);
      if (chartDataRef.current.length > MAX_CHART) chartDataRef.current.shift();
      drawChart();
    };

    ws.onclose = () => setLive(false);
    ws.onerror = () => setWsError('WebSocket connection failed.');
  }

  function togglePause() {
    if (!wsRef.current) return;
    const next = !paused;
    setPaused(next);
    wsRef.current.send(JSON.stringify({ action: next ? 'pause' : 'resume' }));
  }

  function stopStream() {
    if (wsRef.current) {
      wsRef.current.send(JSON.stringify({ action: 'stop' }));
      wsRef.current.close();
      wsRef.current = null;
    }
    setLive(false);
    setPaused(false);
    setFrame(null);
    setProgress({ frame_idx: 0, total: 0 });
  }

  function changeSpeed(v) {
    const num = +v;
    setSpeed(num);
    if (wsRef.current) {
      wsRef.current.send(JSON.stringify({ action: 'resume', speed: num }));
    }
  }

  const meta = LEVEL_META[level.level] || LEVEL_META.CLEAR;
  const progressPct = progress.total > 0 ? Math.min((progress.frame_idx / progress.total) * 100, 100) : 0;
  const avgCount = frameCount > 0 ? (totalCount / frameCount).toFixed(1) : '0.0';

  return (
    <div className="container">
      <div className="video-panel">
        <div className="panel-header">
          <span><Video size={16} />Live Feed</span>
          <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
            <span className="live-status">
              <span className={'dot' + (live ? ' live' : '')} />
              {live ? (paused ? 'PAUSED' : 'ANALYZING') : 'OFFLINE'}
            </span>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--ink-500)' }}>
              {fps ? `${fps} FPS` : '-- FPS'}
            </span>
          </div>
        </div>

        <div
          className={'upload-zone' + (dragOver ? ' dragover' : '') + (uploadReady ? ' has-file' : '')}
          onDragOver={e => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={e => {
            e.preventDefault();
            setDragOver(false);
            if (e.dataTransfer.files[0]) pickFile(e.dataTransfer.files[0]);
          }}
        >
          <input type="file" ref={fileInputRef} accept="video/*" onChange={e => pickFile(e.target.files[0])} />
          <div className="upload-icon">
            {uploading ? <span className="spinner" /> : uploadReady ? <CheckCircle2 size={32} /> : <UploadCloud size={32} />}
          </div>
          <div className="upload-title">{uploading ? 'Uploading...' : uploadReady ? 'Video Ready' : 'Drop a Traffic Video Here'}</div>
          <div className="upload-sub">MP4, AVI, MOV, MKV supported · Camera footage recommended</div>
          {fileName && <div className="upload-filename"><Paperclip size={13} />{fileName}</div>}
        </div>

        {uploadError && <div style={{ margin: '0 18px 18px' }}><ErrorState message={uploadError} /></div>}

        <div className="video-wrap">
          {frame ? (
            <img id="feed" src={`data:image/jpeg;base64,${frame}`} alt="Live analysis feed" />
          ) : (
            <div className="placeholder">
              <Video size={40} />
              {uploadReady ? <>Video ready — press START<br />to begin analysis</> : <>Upload a video to begin</>}
            </div>
          )}
        </div>

        <div className="progress-wrap">
          <div className="progress-bar"><div className="progress-fill" style={{ width: `${progressPct}%` }} /></div>
        </div>

        <div className="controls">
          <button className="btn primary" onClick={startStream} disabled={!uploadReady || live}><Play size={14} />Start</button>
          <button className="btn" onClick={togglePause} disabled={!live}>{paused ? <Play size={14} /> : <Pause size={14} />}{paused ? 'Resume' : 'Pause'}</button>
          <button className="btn" onClick={stopStream} disabled={!live}><Square size={14} />Stop</button>
          <span className="speed-label">SPEED</span>
          <select className="speed-select" value={speed} onChange={e => changeSpeed(e.target.value)}>
            <option value="0.5">0.5×</option>
            <option value="1">1×</option>
            <option value="2">2×</option>
            <option value="4">4×</option>
          </select>
        </div>

        {wsError && <div style={{ margin: '0 18px 18px' }}><ErrorState message={wsError} /></div>}
      </div>

      <div className="side-panel">
        <div className="traffic-card">
          <div className="panel-header"><span><Gauge size={16} />Traffic Status</span></div>
          <div className="traffic-level-display">
            <div
              className="level-ring"
              style={{
                borderColor: meta.color,
                boxShadow: live ? `0 0 22px ${meta.color}44, inset 0 0 12px ${meta.color}11` : undefined,
              }}
            >
              <div className="level-count" style={{ color: meta.color }}>{level.count}</div>
              <div className="level-label-sub">VEHICLES</div>
            </div>
            <div className="level-text" style={{ color: meta.color }}>{level.level}</div>
            <div className="level-desc">{meta.desc}</div>
          </div>
          <div className="stat-grid">
            <div className="stat-cell"><div className="stat-label">PEAK COUNT</div><div className="stat-value">{peakCount}</div></div>
            <div className="stat-cell"><div className="stat-label">AVG COUNT</div><div className="stat-value">{avgCount}</div></div>
            <div className="stat-cell"><div className="stat-label">FRAMES PROC</div><div className="stat-value">{frameCount}</div></div>
            <div className="stat-cell"><div className="stat-label">PROC FPS</div><div className="stat-value">{fps || 0}</div></div>
          </div>
        </div>

        <div className="detection-card">
          <div className="panel-header"><span><Car size={16} />Detections</span></div>
          <div className="det-list">
            {Object.keys(detections).length === 0
              ? <div className="no-det"><Car size={22} />No vehicles in frame</div>
              : Object.entries(detections).map(([label, cnt]) => (
                  <div className="det-item" key={label}>
                    <span>{label.toUpperCase()}</span>
                    <span className="det-badge">× {cnt}</span>
                  </div>
                ))}
          </div>
        </div>

        <div className="chart-card">
          <div className="panel-header"><span><TrendingUp size={16} />Vehicle Count History</span></div>
          <canvas ref={canvasRef} height="100" />
        </div>
      </div>
    </div>
  );
}
