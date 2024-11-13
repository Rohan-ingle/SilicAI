import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import './BotPage.css'; // Import BotPage-specific styles

function BotPage() {
  const [messages, setMessages] = useState([]); // Holds the chat history
  const [input, setInput] = useState(''); // User input
  const chatHistoryRef = useRef(null); // Reference to chat history container
  const scrollButtonRef = useRef(null); // Reference to scroll button

  // Scroll to the bottom of the chat history
  const scrollToBottom = () => {
    if (chatHistoryRef.current) {
      chatHistoryRef.current.scrollTop = chatHistoryRef.current.scrollHeight;
    }
  };

  // Handle sending a message to the backend and receiving the bot's response
  const handleSendMessage = async () => {
    if (!input.trim()) return;

    const userMessage = { role: 'user', content: input };
    setMessages([...messages, userMessage]);

    try {
      const response = await axios.post('http://127.0.0.1:8000/chat', { question: input });
      const botMessage = { role: 'bot', content: response.data.answer };
      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        { role: 'bot', content: 'Error fetching response. Please try again.' },
      ]);
    } finally {
      setInput(''); // Clear the input after sending the message
      scrollToBottom(); // Scroll to the bottom after sending the message
    }
  };

  // Handle pressing "Enter" key to send message
  const handleKeyPress = (e) => {
    if (e.key === 'Enter') handleSendMessage();
  };

  // Scroll to the bottom when messages change
  useEffect(() => {
    scrollToBottom();
    if (chatHistoryRef.current.scrollTop < chatHistoryRef.current.scrollHeight) {
      scrollButtonRef.current.style.display = 'block'; // Show scroll-to-bottom button if scrolled up
    } else {
      scrollButtonRef.current.style.display = 'none'; // Hide button when at the bottom
    }
  }, [messages]);

  return (
    <div className="chat-container">
      {/* Scroll to bottom button */}
      <button
        className="scroll-to-bottom"
        ref={scrollButtonRef}
        onClick={scrollToBottom}
        style={{ display: 'none' }} // Initially hidden
      >
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="none" viewBox="0 0 24 24">
          <path d="M12 16l-4-4h3V4h2v8h3l-4 4z" fill="white" />
        </svg>
      </button>

      <div className="chat-history" ref={chatHistoryRef}>
        {messages.map((msg, idx) => (
          <div key={idx} className={`message ${msg.role}`}>
            {msg.content}
          </div>
        ))}
      </div>

      <div className="chat-input">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Type your message..."
        />
        <button onClick={handleSendMessage}>Send</button>
      </div>
    </div>
  );
}

export default BotPage;