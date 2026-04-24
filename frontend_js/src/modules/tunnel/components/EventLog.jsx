// src/modules/tunnel/compinents/EventLog.jsx
// 역할: 터널 탭 내 이벤트 로그 패널 (단일 컴포넌트로 분리)

export default function EventLog({ events }) {
  return (
    <div style={{ background: "#1a1a2e", padding: 20, borderRadius: 10 }}>
      <h3>🚨 이벤트 로그</h3>

      {events.length === 0 ? (
        <div style={{ color: "#aaa" }}>
          이상 징후 없음 (정상 흐름 유지)
        </div>
      ) : (
        <ul>
          {events.slice(0, 5).map((e, i) => (
            <li key={i}>{e}</li>
          ))}
        </ul>
      )}
    </div>
  );
}