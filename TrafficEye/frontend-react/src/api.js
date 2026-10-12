export const API_BASE = 'http://localhost:8000';
export const WS_BASE = API_BASE.replace(/^http/, 'ws');

async function handle(res) {
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    // FastAPI validation errors return `detail` as an array of
    // {loc, msg} objects rather than a string — format those into a
    // readable message instead of letting Error() stringify the array.
    if (Array.isArray(body.detail)) {
      const fields = body.detail.map(d => d.loc?.[d.loc.length - 1]).filter(Boolean).join(', ');
      throw new Error(`Request rejected by server — missing/invalid field(s): ${fields || 'unknown'}`);
    }
    throw new Error(body.detail || `Request failed (${res.status})`);
  }
  return res.json();
}

function authHeaders(token) {
  return token ? { Authorization: `Bearer ${token}` } : {};
}

export async function apiGet(path, token) {
  let res;
  try {
    res = await fetch(`${API_BASE}${path}`, { headers: authHeaders(token) });
  } catch {
    throw new Error(`Cannot reach backend at ${API_BASE} — is the server running?`);
  }
  return handle(res);
}

export async function apiPostForm(path, formData, token) {
  let res;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      method: 'POST',
      body: formData,
      headers: authHeaders(token),
    });
  } catch {
    throw new Error(`Cannot reach backend at ${API_BASE} — is the server running?`);
  }
  return handle(res);
}

export async function login(username, password) {
  const body = new URLSearchParams();
  body.append('username', username);
  body.append('password', password);

  let res;
  try {
    res = await fetch(`${API_BASE}/api/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body,
    });
  } catch {
    throw new Error(`Cannot reach backend at ${API_BASE} — is the server running?`);
  }
  return handle(res);
}
