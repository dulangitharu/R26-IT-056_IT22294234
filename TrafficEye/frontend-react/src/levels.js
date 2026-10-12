// Validated 3-state semantic palette (checked with the data-viz skill's
// color validator against CVD separation + WCAG contrast — not eyeballed).
// CLEAR folds into the same "good" green as LOW: both read as "no action
// needed," distinguished by icon/label, never by hue alone.
export const LEVEL_META = {
  CLEAR:  { desc: 'Road is free',        color: '#16A34A', bg: '#E8F8ED', bd: '#B9EAC6' },
  LOW:    { desc: 'Light traffic',       color: '#16A34A', bg: '#E8F8ED', bd: '#B9EAC6' },
  MEDIUM: { desc: 'Moderate congestion', color: '#D97706', bg: '#FDF3E2', bd: '#F5D9A3' },
  HIGH:   { desc: 'Heavy congestion',    color: '#B91C1C', bg: '#FBEAE9', bd: '#F2C2BE' },
};
