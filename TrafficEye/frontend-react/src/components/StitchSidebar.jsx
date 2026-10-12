import { useAuth } from '../context/AuthContext';
import { colorForName, initialsForName } from '../avatar';
import Icon from './Icon';

// Reproduces the verified Stitch sidebar structure (shared across all six
// screens' HTML exports) as React. The Stitch mockup also shows a "Live
// Monitor" CTA button here — intentionally omitted: that would point at the
// old WebSocket live-streaming feature, which stays out of primary
// navigation per the approved plan (VideoAnalysis.jsx is preserved but not
// linked from the UI). "Help Center" is likewise omitted — it has no real
// destination in this app, and this rebuild only ships nav items that
// actually do something.
const NAV = [
  { id: 'overview', label: 'Overview', icon: 'dashboard' },
  { id: 'new', label: 'New Analysis', icon: 'analytics' },
  { id: 'history', label: 'History', icon: 'history' },
];

export default function StitchSidebar({ active, onSelect }) {
  const { officer, logout } = useAuth();
  const name = officer?.full_name || '';

  return (
    <nav className="h-screen w-64 fixed left-0 top-0 bg-surface-container-low border-r border-outline-variant z-40 flex flex-col py-6">
      <div className="px-6 mb-8 flex items-center gap-3">
        <div className="w-10 h-10 rounded-lg bg-primary-container flex items-center justify-center text-on-primary font-bold shadow-[0_0_15px_rgba(56,189,248,0.3)]">
          <Icon name="traffic" fill />
        </div>
        <div>
          <h1 className="font-headline-md text-headline-md font-bold text-primary leading-none">TrafficEye</h1>
          <p className="font-label-caps text-label-caps uppercase tracking-wider text-on-surface-variant mt-1">Operational Unit</p>
        </div>
      </div>

      <div className="flex-1 px-3 space-y-2">
        {NAV.map(item => (
          <button
            key={item.id}
            onClick={() => onSelect(item.id)}
            className={
              'w-full flex items-center gap-3 px-4 py-3 transition-colors duration-200 active:scale-95 transition-transform text-left ' +
              (active === item.id
                ? 'border-l-4 border-primary bg-primary/10 text-primary font-bold rounded-r-lg'
                : 'text-on-surface-variant hover:text-on-surface hover:bg-surface-container-high rounded-lg border-l-4 border-transparent')
            }
          >
            <Icon name={item.icon} fill={active === item.id} />
            <span className="font-title-sm text-title-sm">{item.label}</span>
          </button>
        ))}
      </div>

      <div className="px-3 border-t border-outline-variant pt-4">
        <button
          onClick={logout}
          className="w-full flex items-center gap-3 px-4 py-3 text-on-surface-variant hover:text-on-surface hover:bg-surface-container-high transition-colors duration-200 rounded-lg text-left"
        >
          <Icon name="logout" />
          <span className="font-title-sm text-title-sm">Logout</span>
        </button>
      </div>

      <div className="px-6 mt-4 flex items-center gap-3">
        <div
          className="w-8 h-8 rounded-full border border-outline-variant flex items-center justify-center text-[11px] font-bold text-white flex-shrink-0"
          style={{ background: colorForName(name) }}
        >
          {initialsForName(name)}
        </div>
        <div className="flex flex-col min-w-0">
          <span className="text-xs text-on-surface font-semibold truncate">{name || 'Officer'}</span>
          <span className="text-[10px] text-on-surface-variant">Officer Panel</span>
        </div>
      </div>
    </nav>
  );
}
