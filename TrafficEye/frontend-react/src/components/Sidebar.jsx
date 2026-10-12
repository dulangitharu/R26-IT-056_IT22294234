import { LayoutDashboard, Video, History as HistoryIcon, Settings as SettingsIcon, LogOut, Radar } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { colorForName, initialsForName } from '../avatar';

// 'live' (VideoAnalysis.jsx) is intentionally not in primary nav — it has no
// counterpart in the verified Stitch design and is outside the current
// fixed-video prototype scope. The page/route itself is preserved untouched.
const NAV = [
  {
    group: 'Analysis',
    items: [
      { id: 'overview', label: 'Overview', icon: LayoutDashboard },
      { id: 'new', label: 'New Analysis', icon: Video },
    ],
  },
  {
    group: 'Reports',
    items: [
      { id: 'history', label: 'History', icon: HistoryIcon },
    ],
  },
  {
    group: 'System',
    items: [
      { id: 'settings', label: 'Settings', icon: SettingsIcon },
    ],
  },
];

export default function Sidebar({ active, onSelect }) {
  const { officer, logout } = useAuth();
  const name = officer?.full_name || '';

  return (
    <nav className="sb">
      <div className="sb-top">
        <div className="sb-logo">
          <div className="sb-logo-mark"><Radar size={18} /></div>
          <div>
            <div className="sb-logo-text">TrafficEye</div>
            <div className="sb-logo-sub">Vehicle Density Monitor</div>
          </div>
        </div>
      </div>

      <div className="sb-user">
        <div className="sb-av" style={{ background: colorForName(name) }}>{initialsForName(name)}</div>
        <div>
          <div className="sb-uname">{name || 'Officer'}</div>
          <div className="sb-urole">Officer Panel</div>
        </div>
      </div>

      <div className="sb-nav">
        {NAV.map(section => (
          <div key={section.group}>
            <div className="sb-label">{section.group}</div>
            {section.items.map(item => {
              const Icon = item.icon;
              return (
                <button
                  key={item.id}
                  className={'tab-btn' + (active === item.id ? ' active' : '')}
                  onClick={() => onSelect(item.id)}
                >
                  <Icon size={16} />{item.label}
                </button>
              );
            })}
          </div>
        ))}
      </div>

      <div className="sb-btm">
        <div className="sb-logout" onClick={logout}><LogOut size={16} />Sign out</div>
      </div>
    </nav>
  );
}
