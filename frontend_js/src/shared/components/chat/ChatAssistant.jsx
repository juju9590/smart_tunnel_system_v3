import { useState, useEffect, useRef } from 'react';
import { useChatApi } from './useChatApi';
import DataModal from './DataModal';

// 1. 컴포넌트 외부로 분리 (중복 정의 제거)
const ExpandableText = ({ text, limit = 150, forceFull = false }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  // 사용자의 질문(forceFull)인 경우 줄이지 않고 그대로 반환
  if (forceFull) return <div style={{ whiteSpace: 'pre-wrap' }}>{text}</div>;

  const shouldTruncate = text.length > limit || text.split('\n').length > 4;

  if (!shouldTruncate || isExpanded) {
    return (
      <div style={{ whiteSpace: 'pre-wrap' }}>
        {text}
        {shouldTruncate && isExpanded && (
          <div style={{ textAlign: 'right' }}>
            <button onClick={() => setIsExpanded(false)} style={expandBtnStyle}>접기 ▴</button>
          </div>
        )}
      </div>
    );
  }

  return (
    <div style={{ whiteSpace: 'pre-wrap' }}>
      {text.slice(0, limit)}...
      <div style={{ marginTop: '5px' }}>
        <button onClick={() => setIsExpanded(true)} style={expandBtnStyle}>
          전체 답변 보기 (더보기) ▾
        </button>
      </div>
    </div>
  );
};

export default function ChatAssistant({ host, user }) {
  const [isOpen, setIsOpen] = useState(false);
  const [input, setInput] = useState('');
  const [selectedData, setSelectedData] = useState(null);
  const { messages, loading, sendMessage } = useChatApi(host);
  const scrollRef = useRef(null);

  // ✅ 해결 1: 스크롤 로직이 메시지 추가 시마다 '최하단'을 향하도록 수정
  useEffect(() => {
    if (scrollRef.current) {
      // scrollIntoView 대신 scrollTop 조절을 사용하여 확실하게 맨 아래로 보냄
      scrollRef.current.scrollTo({
        top: scrollRef.current.scrollHeight,
        behavior: 'smooth',
      });
    }
  }, [messages, loading, isOpen]);

  const handleSend = () => {
    if (!input.trim()) return;
    sendMessage(input, user?.name);
    setInput('');
  };

  return (
    <>
      <div style={floatingContainer}>
        {isOpen && (
          <div style={chatBox}>
            <div style={chatHeader}>
              <span>TADS AI 어시스턴트</span>
              <button onClick={() => setIsOpen(false)} style={closeHeaderBtn}>✕</button>
            </div>

            {/* ✅ 해결 2: ref={scrollRef}를 여기에 꼭 넣어야 스크롤이 작동합니다! */}
            <div style={msgArea} ref={scrollRef}>
              {messages.length === 0 && (
                <div style={welcomeMsg}>안녕하세요! TADS 데이터에 대해 무엇이든 물어보세요.</div>
              )}
              {messages.map((m, i) => (
                <div key={i} style={{ 
                  display: 'flex', 
                  flexDirection: 'column', 
                  alignItems: m.role === 'user' ? 'flex-end' : 'flex-start', 
                  marginBottom: '20px' 
                }}>
                  <div style={m.role === 'user' ? userMsgStyle : assistantMsgStyle}>
                    <ExpandableText text={m.content} forceFull={m.role === 'user'} />
                    
                    {/* ✅ 해결 3: IMAGE_PATH 별칭을 확실히 잡도록 로직 강화 */}
                    {m.dbData && m.dbData.map((row, idx) => {
                      const imgKey = Object.keys(row).find(key => 
                        key.toUpperCase() === 'IMAGE_PATH' || // SQL 별칭 대응
                        key.toLowerCase().includes('img') || 
                        key.toLowerCase().includes('path')
                      );

                      if (imgKey && row[imgKey]) {
                        return (
                          <div key={idx} style={{ marginTop: '12px' }}>
                            <img 
                              src={`${host.replace(/\/$/, '')}${row[imgKey]}`} // 슬래시 중복 방지
                              alt="관제 캡처"
                              style={{ width: '100%', borderRadius: '8px', border: '1px solid #475569', display: 'block' }}
                              onError={(e) => {
                                console.log("이미지 로드 실패:", e.target.src);
                                e.target.style.display = 'none';
                              }}
                            />
                          </div>
                        );
                      }
                      return null;
                    })}

                    {m.dbData && m.dbData.length > 0 &&
                      !(m.dbData.length === 1 && Object.values(m.dbData[0])[0] === 0) && (
                        <button onClick={() => setSelectedData(m.dbData)} style={viewDataBtn}>
                          📊 상세 결과 표로 보기 ({m.dbData.length}건)
                        </button>
                    )}
                  </div>
                </div>
              ))}
              {loading && <div style={assistantMsgStyle}>분석 중...</div>}
            </div>

            <div style={inputArea}>
              <input
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyPress={e => e.key === 'Enter' && handleSend()}
                placeholder="질문을 입력하세요..."
                style={inputStyle}
              />
              <button onClick={handleSend} style={btnStyle}>전송</button>
            </div>
          </div>
        )}
        {!isOpen && (
          <button onClick={() => setIsOpen(true)} style={toggleBtn}>💬</button>
        )}
      </div>
      <DataModal data={selectedData} onClose={() => setSelectedData(null)} host={host} />
    </>
  );
}


const closeHeaderBtn = {
  background: 'none',
  border: 'none',
  color: '#fff',
  fontSize: '20px',
  cursor: 'pointer',
  padding: '5px',
  lineHeight: '1'
};
// 스타일 정의 (기존과 동일)
const expandBtnStyle = { background: 'none', border: 'none', color: '#60a5fa', cursor: 'pointer', fontSize: '12px', padding: '4px 0', fontWeight: 'bold', textDecoration: 'underline' };
const floatingContainer = { position: 'fixed', bottom: '20px', right: '20px', zIndex: 9999 };
const chatBox = { 
  width: '450px', // 더 넓게 확장
  height: '700px', // 더 높게 확장
  background: '#0f172a', 
  borderRadius: '16px', // 곡률 조금 더 부드럽게
  display: 'flex', 
  flexDirection: 'column', 
  border: '1px solid #334155', 
  boxShadow: '0 20px 50px rgba(0,0,0,0.6)', 
  overflow: 'hidden' 
};
const chatHeader = { 
  padding: '15px 20px', 
  background: '#2563eb', 
  color: '#fff', 
  fontWeight: 'bold', 
  fontSize: '17px',
  display: 'flex',
  justifyContent: 'space-between', // 제목과 X 버튼 양옆 배치
  alignItems: 'center'
};
const msgArea = { 
  flex: 1, 
  overflowY: 'auto', 
  padding: '20px', // 패딩 확대
  display: 'flex', 
  flexDirection: 'column',
  gap: '10px' // 메시지 내부 요소 간 간격
};
const welcomeMsg = { textAlign: 'center', color: '#64748b', fontSize: '13px', marginTop: '20px', padding: '10px', border: '1px dashed #334155', borderRadius: '8px' };
const userMsgStyle = { background: '#2563eb', color: '#fff', padding: '10px 14px', borderRadius: '15px 15px 2px 15px', maxWidth: '85%', fontSize: '14px', lineHeight: '1.4' };
const assistantMsgStyle = { 
  background: '#1e293b', 
  color: '#e2e8f0', 
  padding: '12px 16px', 
  borderRadius: '15px 15px 15px 2px', 
  maxWidth: '90%', // 너비 더 확보
  fontSize: '14.5px', // 글자 크기 미세 조정
  lineHeight: '1.6', // 줄 간격 확보 (글자 붙어 보이지 않게)
  border: '1px solid #334155',
  letterSpacing: '-0.2px' // 자간 살짝 조절
};
const viewDataBtn = { marginTop: '10px', padding: '6px 12px', background: '#3b82f6', border: 'none', color: '#fff', borderRadius: '6px', cursor: 'pointer', fontSize: '12px', width: '100%', fontWeight: 'bold' };
const inputArea = { display: 'flex', padding: '15px', background: '#1e293b', gap: '8px' };
const inputStyle = { flex: 1, background: '#020617', border: '1px solid #334155', color: '#fff', padding: '10px', borderRadius: '6px', outline: 'none' };
const btnStyle = { background: '#2563eb', border: 'none', color: '#fff', padding: '0 15px', borderRadius: '6px', cursor: 'pointer', fontWeight: 'bold' };
const toggleBtn = { width: '60px', height: '60px', borderRadius: '50%', background: '#2563eb', color: '#fff', fontSize: '24px', cursor: 'pointer', border: 'none', boxShadow: '0 4px 15px rgba(37,99,235,0.4)', transition: 'transform 0.2s' };