import { useEffect, useRef, useState } from 'react';
import { ChevronDown, LogOut } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { colorForName, initialsForName } from '../avatar';
import { API_BASE } from '../api';

function formatClock(d) {
  return [d.getHours(), d.getMinutes(), d.getSeconds()].map(n => String(n).padStart(2, '0')).join(':');
}

// Real backend health check (GET / on the FastAPI app — already existed,
// unused by the frontend). No uptime/bandwidth/confidence is fabricated:
// this only reflects what that endpoint actually returns.
function useBackendStatus() {
  const [state, setState] = useState({ status: 'checking', yolo: null });

  useEffect(() => {
    let cancelled = false;
    async function check() {
      try {
        const res = await fetch(API_BASE);
        if (!res.ok) throw new Error();
        const data = await res.json();
        if (!cancelled) setState({ status: 'ok', yolo: data.yolo ?? null });
      } catch {
        if (!cancelled) setState({ status: 'error', yolo: null });
      }
    }
    check();
    const id = setInterval(check, 20000);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  return state;
}

export default function Topbar({ title }) {
  const { officer, logout } = useAuth();
  const [time, setTime] = useState(() => formatClock(new Date()));
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef(null);
  const backend = useBackendStatus();

  useEffect(() => {
    const id = setInterval(() => setTime(formatClock(new Date())), 1000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    function onClickOutside(e) {
      if (menuRef.current && !menuRef.current.contains(e.target)) setMenuOpen(false);
    }
    document.addEventListener('mousedown', onClickOutside);
    return () => document.removeEventListener('mousedown', onClickOutside);
  }, []);

  const name = officer?.full_name || '';

  return (
    <div className="topbar">
      <div className="tb-left">
        <div className="tb-title">{title}</div>
      </div>
      <div className="tb-right">
        <div className="tb-status">
          <div className={'pulse' + (backend.status === 'ok' ? ' dot-ok' : backend.status === 'error' ? ' dot-error' : ' dot-checking')} />
          {backend.status === 'checking' && 'Checking backend...'}
          {backend.status === 'ok' && (backend.yolo === false ? 'Backend online — YOLO unavailable' : 'Backend online')}
          {backend.status === 'error' && 'Backend unreachable'}
        </div>
        <div className="tb-time">{time}</div>

        <div className="tb-officer" data-open={menuOpen} ref={menuRef} onClick={() => setMenuOpen(o => !o)}>
          <div className="tb-avatar" style={{ background: colorForName(name) }}>
            {initialsForName(name)}
          </div>
          <span className="tb-officer-name">{name}</span>
          <span className="tb-caret"><ChevronDown size={14} /></span>

          {menuOpen && (
            <div className="tb-menu">
              <div className="tb-menu-item" onClick={logout}><LogOut size={14} />Sign Out</div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
