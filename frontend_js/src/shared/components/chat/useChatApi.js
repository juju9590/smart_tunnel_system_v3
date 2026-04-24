import { useState } from 'react';
 
export function useChatApi(host) {
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [modalData, setModalData] = useState(null);
 
  const sendMessage = async (input, userName) => {
    if (!input.trim()) return;
 
    setMessages(prev => [...prev, { role: 'user', content: input, userName }]);
    setLoading(true);
 
    try {
      const res = await fetch(`http://${host}:5000/api/chat/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: input, userName: userName }),
      });
      const data = await res.json();
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.answer,
        dbData: data.data,
      }]);
 
      if (data.data && data.data.length > 0) {
        setModalData(data.data);
      }
 
    } catch {
      setMessages(prev => [...prev, { role: 'assistant', content: '연결 오류가 발생했습니다.' }]);
    } finally {
      setLoading(false);
    }
  };
 
  return { messages, loading, sendMessage, modalData, setModalData };
}