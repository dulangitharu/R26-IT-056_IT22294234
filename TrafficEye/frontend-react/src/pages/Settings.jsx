import { Settings as SettingsIcon, Wrench } from 'lucide-react';

export default function Settings() {
  return (
    <div className="container container-full">
      <div className="table-card" style={{ maxWidth: 460 }}>
        <div className="panel-header"><span><SettingsIcon size={16} />System Settings</span></div>
        <div className="qi-body">
          <div className="no-det"><Wrench size={26} />Settings will be available in a future update.</div>
        </div>
      </div>
    </div>
  );
}
