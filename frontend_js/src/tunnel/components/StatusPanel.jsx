export default function StatusPanel({
  state,
  avgSpeed,
  running,
  connected,
  accident,
  accidentLabel,
  sourceName,
  frameId,
  vehicleCount,
  laneCount,
  error
}) {
  const getColor = () => {
    if (state === "ACCIDENT") return "red";
    if (state === "JAM") return "orange";
    if (state === "CONGESTION") return "yellow";
    return "lightgreen";
  };

  return (
    <div style={{ background: "#1a1a2e", padding: 20, borderRadius: 10 }}>
      <h3>🚦 상태</h3>
      <div style={{ color: getColor(), fontSize: 24 }}>
        ● {state}
      </div>

      <h3 style={{ marginTop: 20 }}>⚡ 평균 속도</h3>
      <div>{avgSpeed} px/s</div>

      <div style={{ marginTop: 16, lineHeight: 1.8, fontSize: 14 }}>
        <div><b>서비스:</b> {running ? "RUNNING" : "STOPPED"}</div>
        <div><b>연결:</b> {connected ? "CONNECTED" : "DISCONNECTED"}</div>
        <div><b>사고:</b> {accident ? accidentLabel : "NONE"}</div>
        <div><b>차량 수:</b> {vehicleCount}</div>
        <div><b>차선 수:</b> {laneCount}</div>
        <div><b>프레임:</b> {frameId}</div>
        <div><b>CCTV:</b> {sourceName}</div>
        {error && <div><b>에러:</b> {error}</div>}
      </div>

      <div
        style={{
          marginTop: 10,
          padding: 10,
          background: "#111",
          fontSize: 12
        }}
      >
        <b>기준</b><br />
        px/s: &lt;3 정체 / 4~10 혼잡 / 10↑ 정상
      </div>
    </div>
  );
}