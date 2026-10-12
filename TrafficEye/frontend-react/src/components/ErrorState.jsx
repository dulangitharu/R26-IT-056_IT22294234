import { AlertTriangle, RotateCcw } from 'lucide-react';

export default function ErrorState({ message, onRetry }) {
  return (
    <div className="error-banner" style={{ margin: 0, justifyContent: 'space-between' }}>
      <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}><AlertTriangle size={15} />{message}</span>
      {onRetry && <button className="btn btn-sm" onClick={onRetry}><RotateCcw size={12} />Retry</button>}
    </div>
  );
}
