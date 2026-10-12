import { useMemo, useState } from 'react';
import Icon from '../components/Icon';
import { useAnalysis } from '../context/AnalysisContext';
import RecordsTable from '../components/RecordsTable';

const PAGE_SIZE = 10;

// Structure reproduced from the verified Stitch "Analysis History" screen
// (b72f5b9ff5c043199d78495393d3ce6a): filter bar, data table, pagination.
// The backend's GET /api/analysis has no filter/pagination/export
// parameters, so all three are implemented client-side over the already-
// fetched real records — no server call is fabricated, nothing here invents
// data. "Emergency Active" will always show zero rows, honestly reflecting
// that emergency_detected is never persisted (see integration report) —
// that's not a bug in this filter.
function toCSV(records) {
  const cols = ['recorded_date', 'recorded_time', 'vehicle_count', 'traffic_level', 'queue_length_m', 'clearance_time_s', 'suggested_green_s', 'status'];
  const header = cols.join(',');
  const rows = records.map(r => cols.map(c => JSON.stringify(r[c] ?? '')).join(','));
  return [header, ...rows].join('\n');
}

export default function History({ onOpenResults }) {
  const { records, loading, error, refresh } = useAnalysis();
  const [dateFrom, setDateFrom] = useState('');
  const [dateTo, setDateTo] = useState('');
  const [levelFilter, setLevelFilter] = useState('All Levels');
  const [emergencyFilter, setEmergencyFilter] = useState('All Statuses');
  const [page, setPage] = useState(1);

  const filtered = useMemo(() => {
    return records.filter(r => {
      if (dateFrom && r.recorded_date < dateFrom) return false;
      if (dateTo && r.recorded_date > dateTo) return false;
      if (levelFilter !== 'All Levels' && r.traffic_level !== levelFilter) return false;
      if (emergencyFilter === 'Emergency Active' && !r.emergency_detected) return false;
      if (emergencyFilter === 'No Emergency' && r.emergency_detected) return false;
      return true;
    });
  }, [records, dateFrom, dateTo, levelFilter, emergencyFilter]);

  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const pageRecords = filtered.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);

  function clearFilters() {
    setDateFrom(''); setDateTo(''); setLevelFilter('All Levels'); setEmergencyFilter('All Statuses'); setPage(1);
  }

  function exportLog() {
    const csv = toCSV(filtered);
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `trafficeye_history_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      <div className="flex justify-between items-end">
        <div>
          <h2 className="font-headline-lg text-headline-lg text-on-surface tracking-tight">Analysis History</h2>
          <p className="font-body-md text-body-md text-on-surface-variant mt-1">Review past traffic flow analyses and AI recommendations.</p>
        </div>
        <div className="flex gap-2">
          <button onClick={refresh} className="py-2 px-4 border border-outline-variant rounded font-title-sm text-title-sm text-on-surface hover:bg-surface-container-high transition-colors flex items-center gap-2">
            <Icon name="refresh" className="text-[18px]" /> Refresh
          </button>
          <button onClick={exportLog} className="py-2 px-4 border border-outline-variant rounded font-title-sm text-title-sm text-on-surface hover:bg-surface-container-high transition-colors flex items-center gap-2">
            <Icon name="download" className="text-[18px]" /> Export Log
          </button>
        </div>
      </div>

      <div className="bg-surface-container-low border border-outline-variant rounded-lg p-4 flex flex-wrap gap-4 items-end shadow-sm">
        <div className="flex flex-col gap-1">
          <label className="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wider">From</label>
          <input type="date" value={dateFrom} onChange={e => { setDateFrom(e.target.value); setPage(1); }}
            className="bg-surface-dim border border-outline-variant rounded py-2 px-3 font-body-md text-body-md text-on-surface focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary [color-scheme:dark]" />
        </div>
        <div className="flex flex-col gap-1">
          <label className="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wider">To</label>
          <input type="date" value={dateTo} onChange={e => { setDateTo(e.target.value); setPage(1); }}
            className="bg-surface-dim border border-outline-variant rounded py-2 px-3 font-body-md text-body-md text-on-surface focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary [color-scheme:dark]" />
        </div>
        <div className="flex flex-col gap-1 flex-1 min-w-[150px]">
          <label className="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wider">Traffic Level</label>
          <select value={levelFilter} onChange={e => { setLevelFilter(e.target.value); setPage(1); }}
            className="w-full bg-surface-dim border border-outline-variant rounded py-2 px-3 font-body-md text-body-md text-on-surface focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary">
            <option>All Levels</option>
            <option>LOW</option>
            <option>MEDIUM</option>
            <option>HIGH</option>
          </select>
        </div>
        <div className="flex flex-col gap-1 flex-1 min-w-[150px]">
          <label className="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wider">Emergency Status</label>
          <select value={emergencyFilter} onChange={e => { setEmergencyFilter(e.target.value); setPage(1); }}
            className="w-full bg-surface-dim border border-outline-variant rounded py-2 px-3 font-body-md text-body-md text-on-surface focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary">
            <option>All Statuses</option>
            <option>Emergency Active</option>
            <option>No Emergency</option>
          </select>
        </div>
        <button onClick={clearFilters} className="py-2 px-5 bg-surface-container-highest hover:bg-surface-variant text-on-surface font-title-sm text-title-sm rounded border border-outline-variant transition-colors h-[42px]">
          Clear Filters
        </button>
      </div>

      <div className="bg-surface-container-low border border-outline-variant rounded-lg overflow-hidden shadow-sm">
        {loading ? (
          <div className="p-6 space-y-2">
            {[0, 1, 2, 3].map(i => <div key={i} className="h-11 rounded bg-surface-container animate-pulse" />)}
          </div>
        ) : error ? (
          <div className="p-6 flex items-center justify-between text-error">
            <span className="flex items-center gap-2"><Icon name="warning" />{error}</span>
            <button onClick={refresh} className="text-label-sm font-label-sm border border-error/40 px-3 py-1 rounded hover:bg-error/10">Retry</button>
          </div>
        ) : (
          <>
            <RecordsTable records={pageRecords} onViewDetails={onOpenResults} />
            <div className="border-t border-outline-variant bg-surface-dim p-4 flex items-center justify-between text-body-md text-on-surface-variant">
              <div>Showing {filtered.length === 0 ? 0 : (page - 1) * PAGE_SIZE + 1} to {Math.min(page * PAGE_SIZE, filtered.length)} of {filtered.length} records</div>
              <div className="flex gap-2">
                <button disabled={page <= 1} onClick={() => setPage(p => p - 1)} className="w-8 h-8 flex items-center justify-center rounded border border-outline-variant hover:bg-surface-variant transition-colors disabled:opacity-50">
                  <Icon name="chevron_left" className="text-[18px]" />
                </button>
                <span className="w-8 h-8 flex items-center justify-center rounded border border-primary bg-primary/10 text-primary font-medium">{page}</span>
                <span className="flex items-center px-1 text-outline">/ {totalPages}</span>
                <button disabled={page >= totalPages} onClick={() => setPage(p => p + 1)} className="w-8 h-8 flex items-center justify-center rounded border border-outline-variant hover:bg-surface-variant transition-colors disabled:opacity-50">
                  <Icon name="chevron_right" className="text-[18px]" />
                </button>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
