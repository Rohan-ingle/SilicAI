import React, { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import './App.css';
import Login from './Login';
import { marked } from 'marked';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

// Function to convert markdown to HTML
const parseMarkdown = (markdownText) => marked(markdownText);

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [image, setImage] = useState(null);
  const [isBotTyping, setIsBotTyping] = useState(false);
  const [action, setAction] = useState('chat');
  const [token, setToken] = useState(localStorage.getItem('token'));
  const chatHistoryRef = useRef(null);
  const scrollButtonRef = useRef(null);
  const sendButtonRef = useRef(null);
  const navigate = useNavigate();

  // Handle Login and Logout
  const handleLogin = (accessToken) => {
    setToken(accessToken);
    localStorage.setItem('token', accessToken);
    navigate('/app');
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    setToken(null);
    navigate('/');
  };

  const purgeHistory = async () => {
    try {
      // Call the backend API to purge history
      const response = await fetch('https://distinctly-thorough-pelican.ngrok-free.app/history?purge=True', {
        method: 'GET',
      });
  
      // Check if the response is okay
      if (!response.ok) {
        throw new Error('Failed to purge history');
      }
  
      // Show success toast
      toast.success('Successfully purged memory.', {
        position: "top-right",
        autoClose: 2500,
        hideProgressBar: false,
        closeOnClick: true,
        pauseOnHover: true,
        draggable: true,
      });
    } catch (error) {
      console.error('Error during history purge:', error);
      toast.error('Error Purging History.', {
        position: "top-right",
        autoClose: 2500,
        hideProgressBar: false,
        closeOnClick: true,
        pauseOnHover: true,
        draggable: true,
      });
      throw error;
    }
  };

  const scrollToBottom = () => {
    chatHistoryRef.current?.scrollTo(0, chatHistoryRef.current.scrollHeight);
  };

  const disableSendButton = () => sendButtonRef.current && (sendButtonRef.current.disabled = true);
  const enableSendButton = () => sendButtonRef.current && (sendButtonRef.current.disabled = false);

  const handleSendMessage = async () => {
    if ((action === 'chat' && !input.trim()) || ((action === 'classify' || action === 'segment') && !image)) {
      // Use toast to notify the user instead of alert
      toast.error('Please enter a message or upload an image based on the selected action.', {
        position: "top-right",
        autoClose: 2500,
        hideProgressBar: false,
        closeOnClick: true,
        pauseOnHover: true,
        draggable: true,
      });
      return;
    }

    const userMessage = { role: 'user', content: input, image: image ? URL.createObjectURL(image) : null };
    setMessages((prevMessages) => [...prevMessages, userMessage, { role: 'bot', content: '' }]);

    setIsBotTyping(true);
    disableSendButton();

    try {
      let result = '';
      if (action === 'chat' && input.trim()) {
        const chatHistory = messages.map((msg) => ({ role: msg.role, content: msg.content }));
        const response = await fetch('https://distinctly-thorough-pelican.ngrok-free.app/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question: input, chat_history: chatHistory }),
        });

        if (!response.ok) {
          throw new Error(`Server error: ${response.statusText}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let done = false;

        while (!done) {
          const { value, done: doneReading } = await reader.read();
          done = doneReading;
          if (value) {
            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split('\n').filter((line) => line.trim() !== '');
            for (const line of lines) {
              try {
                const parsed = JSON.parse(line);
                if (parsed.token) {
                  result += parsed.token;
                  setMessages((prevMessages) => {
                    const updatedMessages = [...prevMessages];
                    updatedMessages[updatedMessages.length - 1].content = parseMarkdown(result);
                    return updatedMessages;
                  });
                  scrollToBottom();
                }
                if (parsed.error) throw new Error(parsed.error);
              } catch (err) {
                console.error('Error parsing chunk:', err);
                toast.error(`Error parsing response: ${err.message}`, {
                  position: "top-right",
                  autoClose: 2500,
                  hideProgressBar: false,
                  closeOnClick: true,
                  pauseOnHover: true,
                  draggable: true,
                });
              }
            }
          }
        }
      } else if ((action === 'classify' || action === 'segment') && image) {
        const formData = new FormData();
        formData.append('image', image);
        const endpoint = action === 'classify'
          ? 'https://distinctly-thorough-pelican.ngrok-free.app/ml'
          : 'https://distinctly-thorough-pelican.ngrok-free.app/dl';

        const response = await fetch(endpoint, { method: 'POST', body: formData });

        if (!response.ok) {
          throw new Error(`Server error: ${response.statusText}`);
        }

        const result = await response.json();

        if (result.error) {
          throw new Error(result.error);
        }

        setMessages((prevMessages) => {
          const updatedMessages = [...prevMessages];
          updatedMessages[updatedMessages.length - 1].content = parseMarkdown(`**${result.message || result.defect_name}**`);
          if (result.segmented_image) updatedMessages[updatedMessages.length - 1].segmentedImage = `data:image/png;base64,${result.segmented_image}`;
          return updatedMessages;
        });
      }
    } catch (error) {
      console.error('Error fetching response:', error);
      toast.error('Error fetching response. Please try again.', {
        position: "top-right",
        autoClose: 2500,
        hideProgressBar: false,
        closeOnClick: true,
        pauseOnHover: true,
        draggable: true,
      });
      setMessages((prevMessages) => [...prevMessages, { role: 'bot', content: parseMarkdown('**Error fetching response. Please try again.**') }]);
    } finally {
      setInput('');
      setImage(null);
      setAction('chat');
      setIsBotTyping(false);
      enableSendButton();
      scrollToBottom();
    }
  };

  const handleKeyPress = (e) => e.key === 'Enter' && handleSendMessage();

  useEffect(() => {
    const checkIfScrolledToBottom = () => {
      const chatHistory = chatHistoryRef.current;
      if (!chatHistory) return;
      if (chatHistory.scrollTop + chatHistory.clientHeight < chatHistory.scrollHeight - 10) {
        scrollButtonRef.current.classList.add('visible');
      } else {
        scrollButtonRef.current.classList.remove('visible');
      }
    };

    chatHistoryRef.current?.addEventListener('scroll', checkIfScrolledToBottom);
    return () => chatHistoryRef.current?.removeEventListener('scroll', checkIfScrolledToBottom);
  }, [messages]);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file && (file.type === 'image/jpeg' || file.type === 'image/png' || file.type === 'image/jpg')) {
      setImage(file);
    } else {
      // Use toast to notify the user instead of alert
      toast.error('Please upload a valid image file (JPG, PNG, JPEG).', {
        position: "top-right",
        autoClose: 5000,
        hideProgressBar: false,
        closeOnClick: true,
        pauseOnHover: true,
        draggable: true,
      });
    }
  };

  if (!token) return <Login onLogin={handleLogin} />;

  return (
    <div className="App">
      {/* Toast Container */}
      <ToastContainer />

      <div className="button-container">
        <button onClick={handleLogout} className="logout-button">Logout</button>

        <button onClick={purgeHistory} className="purge-button">Purge Memory</button>
      </div>

      <div className="chat-container">
        <button className="scroll-to-bottom" ref={scrollButtonRef} onClick={scrollToBottom}>
          <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="none" viewBox="0 0 24 24">
            <path d="M12 16l-4-4h3V4h2v8h3l-4 4z" fill="white" />
          </svg>
        </button>

        <div className="chat-history" ref={chatHistoryRef}>
          {messages.map((msg, idx) => (
            <div key={idx} className={`message ${msg.role}`}>
              {msg.image && <img src={msg.image} alt="Uploaded" className="message-image" />}
              {msg.content && <div dangerouslySetInnerHTML={{ __html: msg.content }} />}
              {msg.segmentedImage && <img src={msg.segmentedImage} alt="Segmented" className="message-image" />}
            </div>
          ))}
          {isBotTyping && <div className="message bot">...</div>}
        </div>

        <div className="chat-input">
          <div className="input-container">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Type your message..."
            />
            <label htmlFor="file-upload" className="clip-logo">🔗</label>
          </div>

          <select value={action} onChange={(e) => setAction(e.target.value)} className="action-selector">
            <option value="chat">Chat</option>
            <option value="classify">Classify</option>
            <option value="segment">Segment</option>
          </select>
          <input id="file-upload" type="file" accept="image/*" onChange={handleFileChange} style={{ display: 'none' }} />
          <button onClick={handleSendMessage} ref={sendButtonRef} disabled={isBotTyping}>Send</button>
        </div>
      </div>
    </div>
  );
}

export default App;
