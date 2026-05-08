import { useMemo } from "react";

export default function VideoPanel({ host, active }) {
  const baseUrl = `http://${host || window.location.hostname}:5000/api/tunnel/video_feed`;

  const videoUrl = useMemo(() => {
    return `${baseUrl}?t=${Date.now()}`;
  }, [baseUrl, active]);

  return (
    <div style={{ background: "#111", padding: 12, borderRadius: 12 }}>
      {!active ? (
        <div
          style={{
            width: "100%",
            height: 360,
            borderRadius: 8,
            background: "#000",
            color: "#aaa",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: 18
          }}
        >
          터널 탭 비활성화 또는 서비스 정지 상태
        </div>
      ) : (
        <img
          src={videoUrl}
          alt="Tunnel Video"
          style={{ width: "100%", borderRadius: 8 }}
        />
      )}
    </div>
  );
}