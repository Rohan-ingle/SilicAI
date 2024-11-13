import React from 'react';
import { Link } from 'react-router-dom';  // Make sure Link is imported correctly
import './welcome.css';  // Import Welcome page styles

function Welcome() {
  return (
    <div className="welcome-container">
      <h1>Welcome to the Chatbot!</h1>
      <p>Click below to start chatting with the bot.</p>
      <Link to="/chatbot">
        <button className="go-to-bot">Go to Bot</button>
      </Link>
    </div>
  );
}

export default Welcome;
