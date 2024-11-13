// SignIn.js
import React, { useState } from 'react';
import axios from 'axios';

function SignIn({ onSignIn }) {
  const [signInInput, setSignInInput] = useState('');

  const handleSignIn = async () => {
    const trimmedUsername = signInInput.trim();
    if (!trimmedUsername) {
      alert('Please enter a valid username.');
      return;
    }

    try {
      const response = await axios.post('http://127.0.0.1:8000/signin', {
        username: trimmedUsername,
      });

      if (response.data.message) {
        onSignIn(trimmedUsername);
      }
    } catch (error) {
      console.error('Sign-in error:', error);
      alert('Error signing in. Please try again.');
    }
  };

  return (
    <div className="signin-container">
      <h2>Sign In</h2>
      <input
        type="text"
        value={signInInput}
        onChange={(e) => setSignInInput(e.target.value)}
        placeholder="Enter your username"
      />
      <button onClick={handleSignIn}>Sign In</button>
    </div>
  );
}

export default SignIn;
