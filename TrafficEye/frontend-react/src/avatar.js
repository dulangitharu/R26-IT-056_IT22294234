const AVATAR_COLORS = ['#2563EB', '#16A34A', '#D97706', '#7C3AED', '#0891B2', '#DB2777'];

export function colorForName(name = '') {
  let hash = 0;
  for (let i = 0; i < name.length; i++) {
    hash = name.charCodeAt(i) + ((hash << 5) - hash);
  }
  return AVATAR_COLORS[Math.abs(hash) % AVATAR_COLORS.length];
}

export function initialsForName(name = '') {
  const initials = name
    .split(' ')
    .filter(Boolean)
    .map(w => w[0])
    .slice(0, 2)
    .join('')
    .toUpperCase();
  return initials || '?';
}
