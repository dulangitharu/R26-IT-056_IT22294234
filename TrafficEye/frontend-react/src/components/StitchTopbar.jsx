import { useEffect, useState } from 'react';
import { API_BASE } from '../api';
import { useAuth } from '../context/AuthContext';
import { colorForName, initialsForName } from '../avatar';
import Icon from './Icon';

// Minimal, real backend-connectivity check (GET / — already existed,
// previously unused). No uptime/bandwidth/confidence is fabricated: this
// only reflects what that endpoint actually returns. Kept small and
// separate from the page body per "do not invent a large new System Status
// page during this correction."
function useBackendStatus() {
  const [status, setStatus] = useState('checking');

  useEffect(() => {
    let cancelled = false;
    async function check() {
      try {
        const res = await fetch(API_BASE);
        if (!res.ok) throw new Error();
        if (!cancelled) setStatus('ok');
      } catch {
        if (!cancelled) setStatus('error');
      }
    }
    check();
    const id = setInterval(check, 20000);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  return status;
}

export default function StitchTopbar() {
  const { officer } = useAuth();
  const name = officer?.full_name || '';
  const backend = useBackendStatus();
  const dotColor = backend === 'ok' ? 'bg-tertiary-fixed-dim' : backend === 'error' ? 'bg-error' : 'bg-outline';
  const statusLabel = backend === 'ok' ? 'Backend online' : backend === 'error' ? 'Backend unreachable' : 'Checking backend...';

  return (
    <header className="bg-surface-dim sticky top-0 z-30 border-b border-outline-variant">
      <div className="flex justify-between items-center h-16 px-6">
        <nav className="flex gap-6 h-full items-end pt-2">
          <span className="font-title-sm text-title-sm text-on-surface-variant pb-3 px-2 cursor-default select-none">Network Health</span>
          <span className="font-title-sm text-title-sm text-primary border-b-2 border-primary pb-3 px-2 cursor-default select-none">Analysis</span>
        </nav>
        <div className="flex items-center gap-4">
          <span className="flex items-center gap-1.5 text-label-sm font-label-sm text-on-surface-variant" title={statusLabel}>
            <span className={`w-2 h-2 rounded-full ${dotColor}`} />
          </span>
          <span className="text-on-surface-variant p-2">
            <Icon name="notifications" />
          </span>
          <span className="text-on-surface-variant p-2">
            <Icon name="settings" />
          </span>
          <div className="w-px h-6 bg-outline-variant mx-1" />
          <div
            className="w-8 h-8 rounded-full border border-outline-variant flex items-center justify-center text-[11px] font-bold text-white flex-shrink-0"
            style={{ background: colorForName(name) }}
          >
            {initialsForName(name)}
          </div>
        </div>
      </div>
    </header>
  );
}
