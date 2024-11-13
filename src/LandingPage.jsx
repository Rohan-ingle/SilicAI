import React from 'react';
import { useNavigate } from 'react-router-dom';
import './LandingPage.css'; // Import the CSS file

const LandingPage = () => {
  const navigate = useNavigate();

  const goToApp = () => {
    navigate('/app');
  };

  return (
    <div>

    <div class="loader">
    <span class="loader-text">Welcome</span>
    </div>

    <button className='landing-button' onClick={goToApp}>
        <span>Go to App</span>
    </button>
    </div>
  );
};

export default LandingPage;
