import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import AdminPage from './AdminPage';

function ChatPage() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    if (!input) return;
    
    const newMsgs = [...messages, { role: 'user', text: input }];
    setMessages(newMsgs);
    setInput("");
    setLoading(true);

    try {
      // Docker 내부가 아닌 브라우저에서 접근하므로 localhost:8000 사용
      const res = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: input })
      });
      
      const data = await res.json();
      setMessages([...newMsgs, { role: 'bot', text: data.answer }]);
    } catch (error) {
      console.error(error);
      setMessages([...newMsgs, { role: 'bot', text: "Error: 백엔드 연결 실패 (CORS 또는 서버 꺼짐)" }]);
    }
    setLoading(false);
  };

  return (
    <div style={{ padding: '20px', maxWidth: '800px', margin: '0 auto', fontFamily: 'Arial' }}>
      <h1>🧩 Hybrid RAG Chatbot</h1>
      <div style={{ border: '1px solid #ddd', height: '500px', overflowY: 'scroll', padding: '10px', borderRadius: '8px', background: '#f9f9f9', marginBottom: '20px' }}>
        {messages.map((m, i) => (
          <div key={i} style={{ textAlign: m.role === 'user' ? 'right' : 'left', margin: '10px 0' }}>
            <span style={{ 
              background: m.role === 'user' ? '#007bff' : '#ffffff', 
              color: m.role === 'user' ? 'white' : 'black',
              padding: '10px 15px', 
              borderRadius: '15px',
              border: m.role === 'bot' ? '1px solid #ddd' : 'none',
              display: 'inline-block',
              maxWidth: '80%'
            }}>
              {m.text}
            </span>
          </div>
        ))}
        {loading && <div style={{textAlign: 'left', fontStyle: 'italic', color: '#666'}}>답변 생성 중...</div>}
      </div>
      <div style={{ display: 'flex', gap: '10px' }}>
        <input 
          value={input} 
          onChange={(e) => setInput(e.target.value)} 
          onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
          placeholder="질문을 입력하세요..."
          style={{ flex: 1, padding: '10px', borderRadius: '5px', border: '1px solid #ccc' }}
        />
        <button onClick={sendMessage} style={{ padding: '10px 20px', background: '#28a745', color: 'white', border: 'none', borderRadius: '5px', cursor: 'pointer' }}>Send</button>
      </div>
    </div>
  );
}

function App() {
  return (
    <Router>
      <div style={{ padding: '10px 20px', background: '#f0f2f5', borderBottom: '1px solid #ccc', marginBottom: '20px' }}>
        <Link to="/" style={{ marginRight: '20px', textDecoration: 'none', fontWeight: 'bold', color: '#333' }}>💬 Chat</Link>
        <Link to="/admin" style={{ textDecoration: 'none', fontWeight: 'bold', color: '#333' }}>🛠️ Admin</Link>
      </div>
      
      <Routes>
        <Route path="/" element={<ChatPage />} />
        <Route path="/admin" element={<AdminPage />} />
      </Routes>
    </Router>
  );
}

export default App;