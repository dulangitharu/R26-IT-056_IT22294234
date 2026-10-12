import { createContext, useCallback, useContext, useEffect, useState } from 'react';
import { apiGet } from '../api';

const AnalysisContext = createContext(null);

export function AnalysisProvider({ children }) {
  const [records, setRecords] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [recs, st] = await Promise.all([apiGet('/api/analysis'), apiGet('/api/stats')]);
      setRecords(recs);
      setStats(st);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  return (
    <AnalysisContext.Provider value={{ records, stats, loading, error, refresh }}>
      {children}
    </AnalysisContext.Provider>
  );
}

export function useAnalysis() {
  const ctx = useContext(AnalysisContext);
  if (!ctx) throw new Error('useAnalysis must be used within an AnalysisProvider');
  return ctx;
}
