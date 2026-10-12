import { useState } from 'react';
import { useAuth } from '../context/AuthContext';
import Icon from './Icon';

// Structure reproduced from the verified Stitch "Officer Login" screen
// (94b55e357dd14dcaacabb1168b6bef6c). Only the presentation changed —
// auth logic (login(), error handling, submitting state) is untouched from
// the previous implementation.
const BADGES = [
  { icon: 'smart_toy', color: 'text-primary-container', label: 'AI Vehicle Detection' },
  { icon: 'route', color: 'text-tertiary-container', label: 'Queue Intelligence' },
  { icon: 'local_hospital', color: 'text-error', label: 'Emergency Priority' },
];

export default function Login() {
  const { login } = useAuth();

  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);
  const [shake, setShake] = useState(false);

  async function handleSubmit(e) {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      await login(username, password);
    } catch (err) {
      setError(err.message);
      setShake(true);
      setTimeout(() => setShake(false), 500);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="h-screen w-screen overflow-hidden flex bg-background text-on-surface font-body-md">
      {/* Left: Branding */}
      <div className="hidden lg:flex flex-col w-1/2 relative bg-surface-dim overflow-hidden justify-between p-12">
        <div
          className="absolute inset-0 z-0 opacity-10 pointer-events-none"
          style={{ backgroundSize: '40px 40px', backgroundImage: 'linear-gradient(to right, #38bdf8 1px, transparent 1px), linear-gradient(to bottom, #38bdf8 1px, transparent 1px)' }}
        />
        <div
          className="absolute inset-0 z-0 pointer-events-none"
          style={{ background: 'radial-gradient(circle at 20% 20%, rgba(56,189,248,0.18) 0%, transparent 45%), radial-gradient(circle at 85% 85%, rgba(82,232,124,0.12) 0%, transparent 50%)' }}
        />

        <div className="relative z-10 flex flex-col gap-6 max-w-lg mt-12">
          <div className="flex items-center gap-4">
            <div className="w-16 h-16 rounded-xl bg-primary-container flex items-center justify-center text-on-primary shadow-[0_0_20px_rgba(56,189,248,0.35)]">
              <Icon name="traffic" fill className="text-[32px]" />
            </div>
            <h1 className="font-headline-lg text-headline-lg text-primary-container font-black tracking-tight">TrafficEye</h1>
          </div>
          <div>
            <h2 className="font-headline-md text-headline-md text-on-surface leading-tight mb-2">AI-Based Traffic Officer<br />Decision Support</h2>
            <p className="font-body-lg text-body-lg text-on-surface-variant">Real-time situational awareness and predictive analytics for urban mobility command centers.</p>
          </div>
        </div>

        <div className="relative z-10 flex flex-wrap gap-4 mt-auto mb-12">
          {BADGES.map(b => (
            <div key={b.label} className="flex items-center gap-2 bg-surface-container/50 border border-outline-variant/30 px-4 py-2 rounded-lg backdrop-blur-sm">
              <Icon name={b.icon} className={`${b.color} text-[20px]`} />
              <span className="font-label-caps text-label-caps text-on-surface">{b.label}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Right: Login form */}
      <div className="w-full lg:w-1/2 flex items-center justify-center p-6 bg-surface-container-lowest relative overflow-y-auto">
        <div className="absolute top-8 left-8 lg:hidden flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-primary-container flex items-center justify-center text-on-primary">
            <Icon name="traffic" fill className="text-[18px]" />
          </div>
          <span className="font-title-sm text-title-sm text-primary-container font-bold">TrafficEye</span>
        </div>

        <div className={'w-full max-w-md bg-[#111827] rounded-xl border border-outline-variant/50 p-8 shadow-2xl relative z-10 flex flex-col gap-8' + (shake ? ' animate-shake' : '')}>
          <div className="text-center">
            <div className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-primary-container/10 text-primary-container mb-4 ring-1 ring-primary-container/30">
              <Icon name="admin_panel_settings" className="text-[24px]" />
            </div>
            <h2 className="font-headline-md text-headline-md text-on-surface mb-2">Officer Sign In</h2>
            <p className="font-body-md text-body-md text-on-surface-variant">Access the TrafficEye operations console</p>
          </div>

          <form className="flex flex-col gap-5" onSubmit={handleSubmit}>
            <div className="flex flex-col gap-2">
              <label className="font-label-caps text-label-caps text-on-surface-variant uppercase tracking-wider" htmlFor="username">Officer ID / Username</label>
              <div className="relative">
                <span className="absolute inset-y-0 left-0 flex items-center pl-3 text-outline">
                  <Icon name="badge" className="text-[20px]" />
                </span>
                <input
                  className="w-full bg-[#0B1426] border border-outline-variant text-on-surface rounded-lg pl-10 pr-4 py-3 focus:outline-none focus:border-primary-container focus:ring-1 focus:ring-primary-container transition-colors placeholder:text-outline/50 font-body-md"
                  id="username" name="username" placeholder="Enter credential ID" type="text" autoComplete="username" required
                  value={username} onChange={e => setUsername(e.target.value)}
                />
              </div>
            </div>

            <div className="flex flex-col gap-2">
              <label className="font-label-caps text-label-caps text-on-surface-variant uppercase tracking-wider" htmlFor="password">Password</label>
              <div className="relative">
                <span className="absolute inset-y-0 left-0 flex items-center pl-3 text-outline">
                  <Icon name="key" className="text-[20px]" />
                </span>
                <input
                  className="w-full bg-[#0B1426] border border-outline-variant text-on-surface rounded-lg pl-10 pr-12 py-3 focus:outline-none focus:border-primary-container focus:ring-1 focus:ring-primary-container transition-colors placeholder:text-outline/50 font-body-md"
                  id="password" name="password" placeholder="••••••••" type={showPassword ? 'text' : 'password'} autoComplete="current-password" required
                  value={password} onChange={e => setPassword(e.target.value)}
                />
                <button
                  type="button"
                  className="absolute inset-y-0 right-0 flex items-center pr-3 text-outline hover:text-on-surface transition-colors focus:outline-none"
                  onClick={() => setShowPassword(s => !s)}
                  aria-label={showPassword ? 'Hide password' : 'Show password'}
                >
                  <Icon name={showPassword ? 'visibility_off' : 'visibility'} className="text-[20px]" />
                </button>
              </div>
            </div>

            {error && (
              <div className="flex items-center gap-2 bg-error-container/20 border border-error/40 text-error rounded-lg px-3 py-2 text-body-md font-body-md">
                <Icon name="warning" className="text-[18px]" />
                {error}
              </div>
            )}

            <button
              type="submit"
              disabled={submitting}
              className="w-full bg-[#38BDF8] hover:bg-[#7bd0ff] text-[#08111F] font-title-sm text-title-sm py-3 px-4 rounded-lg flex items-center justify-center gap-2 transition-colors duration-200 mt-2 disabled:opacity-70 disabled:cursor-not-allowed"
            >
              {submitting
                ? <Icon name="progress_activity" className="text-[20px] animate-spin" />
                : <Icon name="shield" className="text-[20px]" />}
              {submitting ? 'Signing in...' : 'Sign In to TrafficEye'}
            </button>
          </form>

          <div className="pt-6 border-t border-outline-variant/30 text-center flex flex-col items-center gap-2">
            <Icon name="lock" className="text-outline/50 text-[18px]" />
            <p className="font-label-sm text-label-sm text-outline">Authorized traffic personnel only. All access is logged.</p>
          </div>
        </div>
      </div>
    </div>
  );
}
