import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

function Login({ onLogin }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [isRegistering, setIsRegistering] = useState(false);
  const navigate = useNavigate(); // Initialize the navigate function

  // Handle login request
  const handleLogin = async (e) => {
    e.preventDefault();
    try {
      const response = await fetch('https://distinctly-thorough-pelican.ngrok-free.app/token', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: new URLSearchParams({
          username,
          password,
        }),
      });

      const data = await response.json();
      if (response.ok) {
        onLogin(data.access_token);
        setError('');
        navigate('/app'); // Redirect to App.jsx route
      } else {
        setError(data.detail || 'Login failed');
      }
    } catch (error) {
      setError('Failed to connect to the server');
    }
  };

  // Handle registration request
  const handleRegister = async (e) => {
    e.preventDefault();
    try {
      const response = await fetch('https://distinctly-thorough-pelican.ngrok-free.app/register', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          username,
          password,
        }),
      });

      const data = await response.json();
      if (response.ok) {
        setIsRegistering(false); // Switch back to login on successful registration
        setError('Registration successful! Please log in.');
      } else {
        setError(data.detail || 'Registration failed');
      }
    } catch (error) {
      setError('Failed to connect to the server');
    }
  };

  return (
    <div className="login-container">
      <form onSubmit={isRegistering ? handleRegister : handleLogin}>
        <h2>{isRegistering ? 'Register' : 'Login'}</h2>
        {error && <p className="error">{error}</p>}
        <div>
          <label>Username</label>
          <input
            type="text"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            required
          />
        </div>
        <div>
          <label>Password</label>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
        </div>
        <button type="submit">{isRegistering ? 'Register' : 'Login'}</button>
        <p className="toggle-form">
          {isRegistering ? (
            <>
              Already have an account?{' '}
              <button type="button" onClick={() => setIsRegistering(false)}>
                Login
              </button>
            </>
          ) : (
            <>
              Don't have an account?{' '}
              <button type="button" onClick={() => setIsRegistering(true)}>
                Register
              </button>
            </>
          )}
        </p>
      </form>
    </div>
  );
}

export default Login;
