import { useState } from 'react';
import { AuthProvider, useAuth } from './context/AuthContext';
import { AnalysisProvider } from './context/AnalysisContext';
import { apiGet } from './api';
import Login from './components/Login';
import StitchSidebar from './components/StitchSidebar';
import StitchTopbar from './components/StitchTopbar';
import Icon from './components/Icon';
import Overview from './pages/Overview';
import NewAnalysis from './pages/NewAnalysis';
import VideoAnalysis from './pages/VideoAnalysis';
import History from './pages/History';
import Settings from './pages/Settings';
import AnalysisResults from './pages/AnalysisResults';
import EmergencyAlert from './pages/EmergencyAlert';
import './App.css';

function BrandedLoadingScreen() {
  return (
    <div className="h-screen w-screen flex flex-col items-center justify-center gap-4 bg-background text-on-surface">
      <div className="w-14 h-14 rounded-xl bg-primary-container flex items-center justify-center text-on-primary shadow-[0_0_20px_rgba(56,189,248,0.4)] animate-pulse">
        <Icon name="traffic" fill className="text-[28px]" />
      </div>
      <div className="font-headline-md text-headline-md font-bold">TrafficEye</div>
      <div className="text-body-md text-on-surface-variant">Verifying session...</div>
      <Icon name="progress_activity" className="animate-spin text-primary" />
    </div>
  );
}

function Dashboard() {
  const { token } = useAuth();
  const [tab, setTab] = useState('overview');
  const [returnTab, setReturnTab] = useState('overview');
  const [detailRecord, setDetailRecord] = useState(null);
  const [detailError, setDetailError] = useState(null);
  // Tracks whether the currently-open Analysis Results record came from the
  // immediate New Analysis submission ('new') or was reloaded from History/
  // Overview via GET /api/analysis/{id} ('history') — used only to decide
  // whether AnalysisResults shows its Session Summary panel; it does not
  // affect what data is fetched or how it's fetched.
  const [resultsSource, setResultsSource] = useState('new');

  // Navigate into a full-page view of a record already held in memory
  // (e.g. the response just returned from POST /api/analysis) — no refetch,
  // no data invented.
  function openResults(record) {
    setDetailError(null);
    setDetailRecord(record);
    setResultsSource('new');
    setReturnTab(tab);
    setTab('results');
  }
  function openEmergency(record) {
    setDetailError(null);
    setDetailRecord(record);
    setReturnTab(tab);
    setTab('emergency');
  }
  function closeDetail() {
    setTab(returnTab);
    setDetailRecord(null);
  }

  // Used by History/Overview "View Details" — fetches the existing,
  // previously-unused GET /api/analysis/{record_id} endpoint. No backend
  // change was needed for this to work.
  async function viewRecordDetails(recordId) {
    setDetailError(null);
    try {
      const record = await apiGet(`/api/analysis/${recordId}`, token);
      setDetailRecord(record);
      setResultsSource('history');
      setReturnTab(tab);
      setTab('results');
    } catch (err) {
      setDetailError(err.message);
    }
  }

  return (
    <AnalysisProvider>
      <div className="min-h-screen bg-background text-on-surface font-body-md">
        <StitchSidebar active={tab} onSelect={setTab} />
        <div className="ml-64 flex flex-col min-h-screen">
          <StitchTopbar />
          <main className="flex-1 p-6 overflow-y-auto">
            {detailError && (
              <div className="max-w-7xl mx-auto mb-6 flex items-center justify-between gap-3 bg-error-container/20 border border-error/40 text-error rounded-lg px-4 py-3 text-body-md font-body-md">
                <span className="flex items-center gap-2"><Icon name="warning" className="text-[18px]" />{detailError}</span>
                <button onClick={() => setDetailError(null)} className="text-label-sm font-label-sm border border-error/40 px-3 py-1 rounded hover:bg-error/10">Dismiss</button>
              </div>
            )}
            {tab === 'overview' && <Overview onOpenResults={viewRecordDetails} />}
            {tab === 'new' && <NewAnalysis onOpenResults={openResults} onOpenEmergency={openEmergency} />}
            {tab === 'live' && <VideoAnalysis />}
            {tab === 'history' && <History onOpenResults={viewRecordDetails} />}
            {tab === 'settings' && <Settings />}
            {tab === 'results' && <AnalysisResults record={detailRecord} onBack={closeDetail} hideSessionSummary={resultsSource === 'history'} />}
            {tab === 'emergency' && <EmergencyAlert record={detailRecord} onBack={closeDetail} onViewResults={openResults} />}
          </main>
        </div>
      </div>
    </AnalysisProvider>
  );
}

function AppShell() {
  const { status } = useAuth();

  if (status === 'checking') return <BrandedLoadingScreen />;
  if (status === 'unauthenticated') return <Login />;
  return <Dashboard />;
}

export default function App() {
  return (
    <AuthProvider>
      <AppShell />
    </AuthProvider>
  );
}
