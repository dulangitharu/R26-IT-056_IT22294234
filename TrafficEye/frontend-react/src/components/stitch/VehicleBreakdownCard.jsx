import Icon from '../Icon';

const BAR_COLOR = ['bg-primary', 'bg-tertiary-fixed', 'bg-secondary', 'bg-primary-container', 'bg-tertiary'];

export default function VehicleBreakdownCard({ breakdown }) {
  const entries = Object.entries(breakdown || {});
  const total = entries.reduce((s, [, c]) => s + c, 0);

  return (
    <div className="bg-surface-container rounded-xl border border-outline-variant p-6 flex flex-col">
      <h3 className="font-title-sm text-title-sm text-on-surface mb-6 flex items-center gap-2">
        <Icon name="pie_chart" className="text-secondary" />
        Vehicle Breakdown
      </h3>
      {entries.length === 0 ? (
        <div className="flex-1 flex flex-col items-center justify-center gap-2 text-on-surface-variant text-body-md py-6">
          <Icon name="inbox" />
          No vehicles detected
        </div>
      ) : (
        <div className="space-y-5 flex-1 flex flex-col justify-center">
          {entries.map(([label, count], i) => (
            <div key={label}>
              <div className="flex justify-between text-sm mb-1.5">
                <span className="text-on-surface font-medium capitalize">{label}</span>
                <span className="text-on-surface-variant font-mono">{count}</span>
              </div>
              <div className="w-full bg-surface-container-highest rounded-full h-2">
                <div className={`${BAR_COLOR[i % BAR_COLOR.length]} h-2 rounded-full`} style={{ width: `${total > 0 ? (count / total) * 100 : 0}%` }} />
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
