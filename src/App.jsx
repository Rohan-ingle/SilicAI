import React, { useState, useRef, useEffect } from 'react';
import './App.css';
import { marked } from 'marked'; // Import marked library for markdown parsing

// Function to convert markdown to HTML
const parseMarkdown = (markdownText) => {
  return marked(markdownText);
};

function App() {
  const [messages, setMessages] = useState([]); // Holds the chat history
  const [input, setInput] = useState(''); // User input
  const [image, setImage] = useState(null); // Store the uploaded image
  const [isBotTyping, setIsBotTyping] = useState(false); // To track if the bot is typing
  const [action, setAction] = useState('chat'); // User selected action: 'chat', 'classify', 'segment'
  const chatHistoryRef = useRef(null); // Reference to chat history container
  const scrollButtonRef = useRef(null); // Reference to scroll button
  const sendButtonRef = useRef(null); // Reference to send button

  // Scroll to the bottom of the chat history
  const scrollToBottom = () => {
    if (chatHistoryRef.current) {
      chatHistoryRef.current.scrollTop = chatHistoryRef.current.scrollHeight;
    }
  };

  // Function to disable the send button
  const disableSendButton = () => {
    if (sendButtonRef.current) {
      sendButtonRef.current.disabled = true;
    }
  };

  // Function to enable the send button
  const enableSendButton = () => {
    if (sendButtonRef.current) {
      sendButtonRef.current.disabled = false;
    }
  };

  const handleSendMessage = async () => {
    // Validation based on action
    if (
      (action === 'chat' && !input.trim()) ||
      ((action === 'classify' || action === 'segment') && !image)
    ) {
      alert('Please enter a message or upload an image based on the selected action.');
      return;
    }

    let imageUrl = null;
    if (image) {
      imageUrl = URL.createObjectURL(image);
    }

    let userMessage = { role: 'user', content: input, image: imageUrl };
    let botMessage = { role: 'bot', content: '' }; // Initialize bot's placeholder message

    // Combine user and bot messages into a single state update
    setMessages((prevMessages) => [...prevMessages, userMessage, botMessage]);

    setIsBotTyping(true); // Show typing indicator
    disableSendButton(); // Disable send button while bot is responding

    try {
      if (action === 'chat' && input.trim()) {
        // Prepare chat history to send (excluding images)
        const chatHistory = messages
          .filter(msg => msg.role === 'user' || msg.role === 'bot')
          .map(msg => ({ role: msg.role, content: msg.content }));

        // Send the chat history along with the current question
        const response = await fetch('https://distinctly-thorough-pelican.ngrok-free.app/chat', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ question: input, chat_history: chatHistory }),
        });

        if (!response.body) {
          throw new Error('ReadableStream not supported in this browser.');
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let done = false;
        let result = '';

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
                if (parsed.error) {
                  setMessages((prevMessages) => {
                    const updatedMessages = [...prevMessages];
                    updatedMessages[updatedMessages.length - 1].content += `\nError: ${parsed.error}`;
                    return updatedMessages;
                  });
                  break;
                }
              } catch (err) {
                console.error('Error parsing chunk:', err);
              }
            }
          }
        }

        setIsBotTyping(false); // Hide typing indicator
      } else if ((action === 'classify' || action === 'segment') && image) {
        // Handle image requests (classification or segmentation)
        const formData = new FormData();
        formData.append('image', image);

        let endpoint = '';
        if (action === 'classify') {
          endpoint = 'https://distinctly-thorough-pelican.ngrok-free.app/ml';
        } else if (action === 'segment') {
          endpoint = 'https://distinctly-thorough-pelican.ngrok-free.app/dl';
        }

        const response = await fetch(endpoint, {
          method: 'POST',
          body: formData,
        });

        const result = await response.json();

        if (action === 'classify') {
          if (result.defect_name) {
            // Update bot message to show the defect type
            setMessages((prevMessages) => {
              const updatedMessages = [...prevMessages];
              updatedMessages[updatedMessages.length - 1].content = parseMarkdown(`**Defect Type:** ${result.defect_name}`);
              return updatedMessages;
            });
          } else if (result.error) {
            // Update bot message to show the error
            setMessages((prevMessages) => {
              const updatedMessages = [...prevMessages];
              updatedMessages[updatedMessages.length - 1].content = parseMarkdown(`**Error:** ${result.error}`);
              return updatedMessages;
            });
          }
        } else if (action === 'segment') {
          if (result.segmented_image) {
            // Display the segmented image
            setMessages((prevMessages) => {
              const updatedMessages = [...prevMessages];
              updatedMessages[updatedMessages.length - 1].content = parseMarkdown(`**${result.message}**`);
              updatedMessages[updatedMessages.length - 1].segmentedImage = `data:image/png;base64,${result.segmented_image}`;
              return updatedMessages;
            });
          } else if (result.error) {
            // Update bot message to show the error
            setMessages((prevMessages) => {
              const updatedMessages = [...prevMessages];
              updatedMessages[updatedMessages.length - 1].content = parseMarkdown(`**Error:** ${result.error}`);
              return updatedMessages;
            });
          }
        }

        setIsBotTyping(false); // Hide typing indicator
      }
    } catch (error) {
      console.error('Error fetching response:', error);
      setMessages((prevMessages) => [
        ...prevMessages,
        { role: 'bot', content: parseMarkdown('**Error fetching response. Please try again.**') },
      ]);
      setIsBotTyping(false); // Hide typing indicator
    } finally {
      setInput(''); // Clear input field
      setImage(null); // Clear uploaded image
      setAction('chat'); // Reset action to 'chat'
      scrollToBottom(); // Ensure the latest messages are visible
      enableSendButton(); // Re-enable send button
    }
  };

  // Function to handle pressing the "Enter" key to send a message
  const handleKeyPress = (e) => {
    if (e.key === 'Enter') handleSendMessage();
  };

  // Effect to handle scrolling and scroll button visibility
  useEffect(() => {
    const checkIfScrolledToBottom = () => {
      const chatHistory = chatHistoryRef.current;
      if (!chatHistory) return;

      // If the user is not at the bottom, show the scroll-to-bottom button
      if (chatHistory.scrollTop + chatHistory.clientHeight < chatHistory.scrollHeight - 10) {
        scrollButtonRef.current.classList.add('visible');
      } else {
        scrollButtonRef.current.classList.remove('visible');
      }
    };

    const chatHistoryElement = chatHistoryRef.current;
    if (chatHistoryElement) {
      chatHistoryElement.addEventListener('scroll', checkIfScrolledToBottom);
      // Initial check
      checkIfScrolledToBottom();
    }

    return () => {
      if (chatHistoryElement) {
        chatHistoryElement.removeEventListener('scroll', checkIfScrolledToBottom);
      }
    };
  }, [messages]);

  // Handle file upload (clip logo inside the input field)
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file && (file.type === 'image/jpeg' || file.type === 'image/png' || file.type === 'image/jpg')) {
      setImage(file); // Store the File object
    } else {
      alert('Please upload a valid image file (JPG, PNG, JPEG).');
    }
  };

  // Handle action change
  const handleActionChange = (e) => {
    setAction(e.target.value);
  };

  return (
    <div className="chat-container">
      {/* Scroll to bottom button */}
      <button
        className="scroll-to-bottom"
        ref={scrollButtonRef}
        onClick={scrollToBottom}
      >
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="none" viewBox="0 0 24 24">
          <path d="M12 16l-4-4h3V4h2v8h3l-4 4z" fill="white" />
        </svg>
      </button>

      {/* Chat history */}
      <div className="chat-history" ref={chatHistoryRef}>
        {messages.map((msg, idx) => (
          <div key={idx} className={`message ${msg.role}`}>
            {/* Display uploaded image if present */}
            {msg.image && <img src={msg.image} alt="Uploaded" className="message-image" />}
            {/* Display message content with markdown rendering */}
            {msg.content && <div dangerouslySetInnerHTML={{ __html: msg.content }} />}
            {/* Display segmented image if present */}
            {msg.segmentedImage && <img src={msg.segmentedImage} alt="Segmented" className="message-image" />}
          </div>
        ))}
        {/* Typing indicator */}
        {isBotTyping && <div className="message bot">...</div>}
      </div>

      {/* Chat input area */}
      <div className="chat-input">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Type your message..."
        />
        {/* Action selector */}
        <select value={action} onChange={handleActionChange} className="action-selector">
          <option value="chat">Chat</option>
          <option value="classify">Classify</option>
          <option value="segment">Segment</option>
        </select>
        {/* Clip logo for file upload */}
        <label htmlFor="file-upload" className="clip-logo">
          📎
        </label>
        <input
          id="file-upload"
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          style={{ display: 'none' }}
        />
        <button onClick={handleSendMessage} ref={sendButtonRef} disabled={isBotTyping}>
          Send
        </button>
      </div>
    </div>
  );
}

export default App;