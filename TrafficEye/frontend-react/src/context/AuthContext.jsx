import { createContext, useCallback, useContext, useEffect, useState } from 'react';
import { apiGet, login as apiLogin } from '../api';

const AuthContext = createContext(null);
const TOKEN_KEY = 'trafficeye_token';

export function AuthProvider({ children }) {
  const [token, setToken] = useState(() => sessionStorage.getItem(TOKEN_KEY));
  const [officer, setOfficer] = useState(null);
  // 'checking' | 'authenticated' | 'unauthenticated'
  const [status, setStatus] = useState('checking');

  const verify = useCallback(async (t) => {
    try {
      const me = await apiGet('/api/auth/me', t);
      setOfficer(me);
      setStatus('authenticated');
    } catch {
      sessionStorage.removeItem(TOKEN_KEY);
      setToken(null);
      setOfficer(null);
      setStatus('unauthenticated');
    }
  }, []);

  useEffect(() => {
    if (token) verify(token);
    else setStatus('unauthenticated');
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function login(username, password) {
    const data = await apiLogin(username, password);
    sessionStorage.setItem(TOKEN_KEY, data.access_token);
    setToken(data.access_token);
    await verify(data.access_token);
  }

  function logout() {
    sessionStorage.removeItem(TOKEN_KEY);
    setToken(null);
    setOfficer(null);
    setStatus('unauthenticated');
  }

  return (
    <AuthContext.Provider value={{ token, officer, status, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within an AuthProvider');
  return ctx;
}
