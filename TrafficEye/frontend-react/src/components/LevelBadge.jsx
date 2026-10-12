import { CheckCircle2, AlertTriangle, AlertOctagon } from 'lucide-react';
import { LEVEL_META } from '../levels';

const LEVEL_ICON = {
  CLEAR: CheckCircle2,
  LOW: CheckCircle2,
  MEDIUM: AlertTriangle,
  HIGH: AlertOctagon,
};

export default function LevelBadge({ level }) {
  const meta = LEVEL_META[level] || LEVEL_META.CLEAR;
  const Icon = LEVEL_ICON[level] || CheckCircle2;
  return (
    <span
      className="level-badge"
      style={{ background: meta.bg, color: meta.color, border: `1px solid ${meta.bd}` }}
    >
      <Icon size={12} />{level}
    </span>
  );
}
