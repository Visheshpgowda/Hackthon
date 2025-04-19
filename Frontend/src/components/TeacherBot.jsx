import React from 'react';
import { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Send, Loader, User, Bot, Search } from 'lucide-react';

const TeacherBot = () => {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([
    { type: 'bot', content: 'Hello! I can help you find student information. Try asking about a student by name, email, or roll number.' }
  ]);
  const [loading, setLoading] = useState(false);
  const [showTyping, setShowTyping] = useState(false);
  const [typingText, setTypingText] = useState('');
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const typingTimeoutRef = useRef(null);

  // Auto-scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, showTyping]);

  // Focus input on mount
  useEffect(() => {
    inputRef.current?.focus();
    
    // Greeting animation on first load
    const timer = setTimeout(() => {
      setShowTyping(true);
      simulateRealisticTyping("Hello! I can help you find student information. Try asking about a student by name, email, or roll number.");
    }, 500);
    
    return () => clearTimeout(timer);
  }, []);

  // Clean up typing animation on unmount
  useEffect(() => {
    return () => {
      if (typingTimeoutRef.current) {
        clearTimeout(typingTimeoutRef.current);
      }
    };
  }, []);

  const formatStudentData = (student) => {
    if (!student) return "No student found with that information.";
    
    return (
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-4 rounded-lg border border-blue-100 shadow-sm transition-all hover:shadow-md">
        <h3 className="font-bold text-lg text-blue-800 mb-2 flex items-center">
          <span className="bg-blue-100 p-1 rounded-full mr-2 animate-pulse">👤</span> 
          {student.name}
        </h3>
        <div className="grid grid-cols-2 gap-3 mt-3">
          <div className="bg-white bg-opacity-60 p-2 rounded transform transition-all hover:scale-105">
            <p className="text-sm font-medium text-indigo-600">Roll No</p>
            <p className="font-semibold">{student.roll_no}</p>
          </div>
          <div className="bg-white bg-opacity-60 p-2 rounded transform transition-all hover:scale-105">
            <p className="text-sm font-medium text-indigo-600">Class</p>
            <p className="font-semibold">{student.class}</p>
          </div>
          <div className="bg-white bg-opacity-60 p-2 rounded transform transition-all hover:scale-105">
            <p className="text-sm font-medium text-indigo-600">Email</p>
            <p className="text-sm break-all">{student.email}</p>
          </div>
          <div className="bg-white bg-opacity-60 p-2 rounded transform transition-all hover:scale-105">
            <p className="text-sm font-medium text-indigo-600">Phone</p>
            <p>{student.phone_no}</p>
          </div>
          <div className="bg-white bg-opacity-60 p-2 rounded transform transition-all hover:scale-105">
            <p className="text-sm font-medium text-indigo-600">Date of Birth</p>
            <p>{new Date(student.dob).toLocaleDateString()}</p>
          </div>
          <div className="bg-white bg-opacity-60 p-2 rounded transform transition-all hover:scale-105">
            <p className="text-sm font-medium text-indigo-600">Address</p>
            <p className="text-sm">{student.address}</p>
          </div>
        </div>
      </div>
    );
  };

  const simulateRealisticTyping = (text) => {
    let charIndex = 0;
    setTypingText('');
    
    const typeNextChar = () => {
      if (charIndex < text.length) {
        setTypingText(prev => prev + text.charAt(charIndex));
        charIndex++;
        
        // Random typing speed to make it feel more human
        const randomDelay = Math.floor(Math.random() * 35) + 25; // 25-60ms per character
        typingTimeoutRef.current = setTimeout(typeNextChar, randomDelay);
      } else {
        // Done typing, show the full message
        setTimeout(() => {
          setShowTyping(false);
          setMessages(prev => prev.slice(0, -1).concat({ type: 'bot', content: text }));
        }, 300);
      }
    };
    
    typeNextChar();
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;
    
    // Add user message to chat with popping animation
    const userMessage = { type: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    
    // Clear input with transition effect
    setInput('');
    inputRef.current?.focus();
    setLoading(true);
    
    // Show typing indicator with pre-response pause
    setTimeout(() => {
      setShowTyping(true);
    }, 700);
    
    try {
      const response = await axios.post("http://localhost:8000/query", {
        query: input
      }, {
        headers: {
          "Content-Type": "application/json"
        }
      });
      
      const students = response.data.results;
      
      // Realistic delay based on content length
      setTimeout(() => {
        if (students && students.length > 0) {
          // Show typing for first student
          const firstStudentMessage = "I found the student you're looking for:";
          
          // Start realistic typing for text response
          simulateRealisticTyping(firstStudentMessage);
          
          // After typing first message, show all results
          setTimeout(() => {
            students.forEach((student, index) => {
              setTimeout(() => {
                setMessages(prev => [...prev, { 
                  type: 'bot', 
                  content: formatStudentData(student)
                }]);
              }, index * 300); // Stagger multiple results
            });
            setLoading(false);
          }, firstStudentMessage.length * 40 + 500);
          
        } else {
          // No students found
          const notFoundMessage = "I couldn't find any student matching your query. Please try again with a different name, email, or roll number.";
          simulateRealisticTyping(notFoundMessage);
          setLoading(false);
        }
      }, 1500);
    } catch (err) {
      console.error("Error fetching data:", err);
      setTimeout(() => {
        const errorMessage = "Sorry, I encountered an error while searching for student information. Please try again later.";
        simulateRealisticTyping(errorMessage);
        setLoading(false);
      }, 1500);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Header with subtle animation */}
      <div className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white p-4 shadow-lg animate-gradient">
        <div className="max-w-3xl mx-auto flex items-center">
          <div className="bg-white bg-opacity-20 p-2 rounded-full mr-3 pulse-slow">
            <Bot className="h-6 w-6" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight">Student Information Assistant</h1>
            <p className="text-blue-100 text-sm">Ask me about any student</p>
          </div>
        </div>
      </div>
      
      {/* Chat Messages Area */}
      <div className="flex-1 p-4 overflow-y-auto">
        <div className="max-w-3xl mx-auto space-y-4">
          {messages.map((message, index) => (
            <div 
              key={index} 
              className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
              style={{ 
                animationDelay: `${index * 0.1}s`,
                opacity: 0,
                animation: 'fadeIn 0.4s ease-out forwards'
              }}
            >
              <div className={`flex max-w-xs md:max-w-md lg:max-w-lg ${message.type === 'user' ? 'flex-row-reverse' : ''}`}>
                <div className={`flex-shrink-0 h-8 w-8 rounded-full flex items-center justify-center ${
                  message.type === 'user' ? 'bg-blue-500 ml-2 animate-pop' : 'bg-gray-200 mr-2 animate-pop'
                }`}>
                  {message.type === 'user' ? 
                    <User className="h-4 w-4 text-white" /> : 
                    <Bot className="h-4 w-4 text-gray-600" />
                  }
                </div>
                <div 
                  className={`p-3 rounded-2xl ${
                    message.type === 'user' 
                      ? 'bg-blue-500 text-white rounded-tr-none message-pop-right' 
                      : 'bg-white text-gray-800 rounded-tl-none shadow message-pop-left'
                  }`}
                >
                  {typeof message.content === 'string' 
                    ? message.content 
                    : message.content}
                </div>
              </div>
            </div>
          ))}
          {showTyping && (
            <div className="flex justify-start animate-fadeIn">
              <div className="flex">
                <div className="flex-shrink-0 h-8 w-8 rounded-full bg-gray-200 flex items-center justify-center mr-2 animate-pop">
                  <Bot className="h-4 w-4 text-gray-600" />
                </div>
                <div className="bg-white text-gray-800 rounded-2xl rounded-tl-none p-3 shadow flex items-center message-pop-left">
                  {typingText ? (
                    <div>{typingText}<span className="typing-cursor">|</span></div>
                  ) : (
                    <div className="typing-indicator">
                      <span></span>
                      <span></span>
                      <span></span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>
      
      {/* Input Area */}
      <div className="border-t border-blue-100 bg-white p-4 shadow-lg animate-slide-up">
        <form onSubmit={handleSubmit} className="max-w-3xl mx-auto flex">
          <div className="relative flex-1">
            <input
              ref={inputRef}
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask about a student (e.g., 'Show me information for Kunal Deshmukh')"
              className="w-full border border-blue-200 rounded-l-full pl-4 pr-10 py-3 focus:outline-none focus:ring-2 focus:ring-blue-400 transition-all duration-300"
              disabled={loading}
            />
            <div className={`absolute right-3 top-3 transition-all duration-300 ${input ? 'opacity-100' : 'opacity-0'}`}>
              <Search className="text-gray-400 h-5 w-5" />
            </div>
          </div>
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className={`bg-gradient-to-r from-blue-500 to-indigo-600 text-white px-6 py-3 rounded-r-full hover:opacity-90 focus:outline-none focus:ring-2 focus:ring-blue-400 flex items-center justify-center transition-all duration-300 transform ${
              !input.trim() ? 'opacity-70 cursor-not-allowed' : 'hover:scale-105 active:scale-95'
            }`}
          >
            {loading ? (
              <Loader className="animate-spin h-5 w-5" />
            ) : (
              <Send className="h-5 w-5 animate-pulse-subtle" />
            )}
          </button>
        </form>
      </div>
    </div>
  );
};

// Add required animations with more detailed styles
const style = document.createElement('style');
style.textContent = `
  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
  }
  
  @keyframes appear {
    from { opacity: 0; transform: scale(0.95); }
    to { opacity: 1; transform: scale(1); }
  }
  
  @keyframes pop {
    0% { transform: scale(0.8); opacity: 0.5; }
    50% { transform: scale(1.05); }
    100% { transform: scale(1); opacity: 1; }
  }
  
  @keyframes slideUp {
    from { transform: translateY(20px); opacity: 0; }
    to { transform: translateY(0); opacity: 1; }
  }
  
  @keyframes messagePop {
    0% { transform: scale(0); opacity: 0; }
    70% { transform: scale(1.05); }
    100% { transform: scale(1); opacity: 1; }
  }
  
  @keyframes messageFade {
    from { opacity: 0; max-height: 0; }
    to { opacity: 1; max-height: 1000px; }
  }
  
  @keyframes pulseSubtle {
    0% { transform: scale(1); }
    50% { transform: scale(1.05); }
    100% { transform: scale(1); }
  }
  
  @keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
  }
  
  @keyframes pulseSlow {
    0% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.1); opacity: 0.9; }
    100% { transform: scale(1); opacity: 1; }
  }
  
  @keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0; }
  }
  
  .animate-fadeIn {
    animation: fadeIn 0.4s ease-out;
  }
  
  .animate-appear {
    animation: appear 0.3s ease-out;
  }
  
  .animate-pop {
    animation: pop 0.3s ease-out forwards;
  }
  
  .animate-slide-up {
    animation: slideUp 0.5s ease-out forwards;
  }
  
  .animate-pulse-subtle {
    animation: pulseSubtle 2s infinite ease-in-out;
  }
  
  .animate-gradient {
    background-size: 200% 200%;
    animation: gradientShift 15s ease infinite;
  }
  
  .pulse-slow {
    animation: pulseSlow 3s infinite ease-in-out;
  }
  
  .message-pop-left {
    transform-origin: top left;
    animation: messagePop 0.3s forwards;
  }
  
  .message-pop-right {
    transform-origin: top right;
    animation: messagePop 0.3s forwards;
  }
  
  .typing-indicator {
    display: flex;
    align-items: center;
  }
  
  .typing-indicator span {
    height: 8px;
    width: 8px;
    background: #ddd;
    border-radius: 50%;
    display: block;
    margin: 0 2px;
    opacity: 0.6;
  }
  
  .typing-indicator span:nth-child(1) {
    animation: bounce 1s infinite;
    animation-delay: 0s;
  }
  
  .typing-indicator span:nth-child(2) {
    animation: bounce 1s infinite;
    animation-delay: 0.2s;
  }
  
  .typing-indicator span:nth-child(3) {
    animation: bounce 1s infinite;
    animation-delay: 0.4s;
  }
  
  @keyframes bounce {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-5px); }
  }
  
  .typing-cursor {
    display: inline-block;
    width: 2px;
    height: 16px;
    background-color: #666;
    margin-left: 2px;
    animation: blink 1s infinite;
    vertical-align: middle;
  }
`;
document.head.appendChild(style);

export default TeacherBot;