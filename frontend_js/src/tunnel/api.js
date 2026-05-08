import axios from "axios";

/* =========================================================
 * tunnel status
 * ========================================================= */
export const fetchTunnelStatus = async (BACKEND_URL) => {
  const res = await axios.get(`${BACKEND_URL}/api/tunnel/status`);
  return res.data;
};

/* =========================================================
 * CCTV 리스트 저장
 * ========================================================= */
export const setTunnelCctvList = async (BACKEND_URL, items) => {
  const res = await axios.post(`${BACKEND_URL}/api/tunnel/set-cctv-list`, {
    items,
  });
  return res.data;
};

/* =========================================================
 * CCTV 리스트 조회
 * ========================================================= */
export const getTunnelCctvList = async (BACKEND_URL) => {
  const res = await axios.get(`${BACKEND_URL}/api/tunnel/cctv-list`);
  return res.data;
};

/* =========================================================
 * 랜덤 CCTV 선택
 * ========================================================= */
export const selectRandomCctv = async (BACKEND_URL) => {
  const res = await axios.get(`${BACKEND_URL}/api/tunnel/select-random`);
  return res.data;
};

/* =========================================================
 * 이름으로 CCTV 선택
 * ========================================================= */
export const selectCctvByName = async (BACKEND_URL, name) => {
  const res = await axios.get(`${BACKEND_URL}/api/tunnel/select-cctv`, {
    params: { name },
  });
  return res.data;
};

/* =========================================================
 * 시연 영상 목록 조회
 * ========================================================= */
export const getDemoVideos = async (BACKEND_URL) => {
  const res = await axios.get(`${BACKEND_URL}/api/tunnel/demo-videos`);
  return res.data;
};

/* =========================================================
 * 시연 영상 선택
 * ========================================================= */
export const selectDemoVideo = async (BACKEND_URL, keyOrFilename) => {
  const body = String(keyOrFilename || "").endsWith(".mp4")
    ? { filename: keyOrFilename }
    : { key: keyOrFilename };

  const res = await axios.post(`${BACKEND_URL}/api/tunnel/select-demo-video`, body);
  return res.data;
};

export const pauseDemoVideo = async (BACKEND_URL) => {
  const res = await axios.post(`${BACKEND_URL}/api/tunnel/demo-video/pause`);
  return res.data;
};

export const resumeDemoVideo = async (BACKEND_URL) => {
  const res = await axios.post(`${BACKEND_URL}/api/tunnel/demo-video/resume`);
  return res.data;
};

/* =========================================================
 * 다른탭 이동시 중단
 * ========================================================= */
export const stopTunnelStream = async (BACKEND_URL) => {
  const res = await axios.post(`${BACKEND_URL}/api/tunnel/stream/stop`);
  return res.data;
};

/* =========================================================
 * 목표 차선 수 설정
 * ========================================================= */
export const setTunnelTargetLaneCount = async (BACKEND_URL, laneCount) => {
  const res = await axios.post(`${BACKEND_URL}/api/tunnel/lane/target-count`, {
    lane_count: laneCount,
  });
  return res.data;
};

/* =========================================================
 * 다른탭에서 터널탭 돌아오면 다시시작
 * ========================================================= */

export const restartTunnelStreamRandom = async (BACKEND_URL) => {
  const res = await axios.post(`${BACKEND_URL}/api/tunnel/stream/restart-random`);
  return res.data;
};

/* =========================================================
 * 백엔드 기반 최신 CCTV 목록 조회
 * 우선순위:
 * 1) 백엔드에서 ITS 최신 조회
 * 2) 실패하면 백엔드 캐시
 * 3) 그것도 실패하면 fallback
 * ========================================================= */
export const fetchTunnelCctvUrl = async (host) => {
  const BACKEND_URL = `http://${host}:5000`;

  try {
    const res = await getTunnelCctvList(BACKEND_URL);

    if (res?.ok && Array.isArray(res?.items) && res.items.length > 0) {
      return res;
    }

    throw new Error("백엔드 CCTV 목록이 비어 있습니다.");
  } catch (error) {
    console.warn("터널 CCTV 목록 조회 실패, fallback 사용:", error);

    return {
      ok: true,
      source: "fallback",
      items: [
        {
          name: "테스트 채널 1",
          url: "https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8",
        },
        {
          name: "테스트 채널 2",
          url: "https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8",
        },
        {
          name: "테스트 채널 3",
          url: "https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8",
        },
        {
          name: "테스트 채널 4",
          url: "https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8",
        },
      ],
    };
  }
};
