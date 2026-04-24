// src/modules/tunnel/components/DwellChart.jsx
// 역할: 터널 탭 내 차량 체류시간 차트 패널 (단일 컴포넌트로 분리)

import { BarChart, Bar, XAxis, YAxis, Tooltip } from "recharts";

export default function DwellChart({ dwell }) {
  const data = Object.keys(dwell).slice(0, 10).map(id => ({
    name: `ID${id}`,
    time: dwell[id]
  }));

  const avg =
    data.reduce((sum, d) => sum + d.time, 0) / (data.length || 1);

  return (
    <div style={{ background: "#1a1a2e", padding: 20, borderRadius: 10 }}>
      <h3>📊 체류시간</h3>
      <div>평균 체류시간: {avg.toFixed(1)} sec</div>

      <BarChart width={400} height={200} data={data}>
        <XAxis dataKey="name" />
        <YAxis />
        <Tooltip />
        <Bar dataKey="time" />
      </BarChart>
    </div>
  );
}