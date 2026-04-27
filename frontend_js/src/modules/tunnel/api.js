import axios from "axios";

/* =========================================================
 * 우선 연결할 검증 터널 리스트
 * - URL 고정 X
 * - 이름 기준으로 ITS 최신 목록에서 매칭
 * ========================================================= */
const PRIORITY_TUNNELS = [
  "[수도권제2순환선] 필봉산터널(동탄)",
  "[수도권제1순환선] 광암터널(1)",
  "[수도권제1순환선] 광암터널(2/3)",
  "[수원광명선] 성채터널(수원)",
  "[영동선] 둔내터널(강릉)-7/18",
  "[남해선] 창원1터널(부산)-7/12",
  "[영동선] [인천1]광교터널(인천1 1 고정)",
];

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
 * 유틸: 배열 셔플
 * ========================================================= */
function shuffleArray(arr) {
  return [...arr].sort(() => Math.random() - 0.5);
}

/* =========================================================
 * 유틸: 내부 터널 후보 필터
 * - "터널" 포함
 * - 외부/입구/출구/진입부/진출부 제외
 * ========================================================= */
function filterTunnelCandidates(list) {
  return list.filter((item) => {
    const name = String(item?.cctvname || item?.name || "");

    return (
      name.includes("터널") &&
      !name.includes("외부") &&
      !name.includes("입구") &&
      !name.includes("출구") &&
      !name.includes("진입부") &&
      !name.includes("진출부")
    );
  });
}

/* =========================================================
 * 유틸: ITS item -> tunnel item 변환
 * ========================================================= */
function mapItsItem(item) {
  return {
    name: item?.cctvname || "터널 CCTV",
    url: item?.cctvurl || "",
  };
}

/* =========================================================
 * 유틸: 우선 후보 이름 매칭
 * - 정확 일치 우선
 * - 없으면 includes 부분 일치
 * - 중복 제거
 * ========================================================= */
function findPriorityItems(apiItems, priorityNames) {
  const matched = [];
  const usedNames = new Set();

  for (const target of priorityNames) {
    const found =
      apiItems.find((item) => String(item?.cctvname || "") === target) ||
      apiItems.find((item) =>
        String(item?.cctvname || "").includes(target)
      );

    if (!found) continue;

    const foundName = String(found?.cctvname || "");
    if (usedNames.has(foundName)) continue;

    usedNames.add(foundName);
    matched.push(found);
  }

  return matched;
}

/* =========================================================
 * 유틸: ITS 목록 -> 최종 저장 목록 생성
 * - 우선 후보 먼저
 * - 부족한 부분은 랜덤 백업으로 채움
 * ========================================================= */
function buildFinalTunnelItems(rawList) {
  const list = Array.isArray(rawList) ? rawList : [];
  if (list.length === 0) return [];

  const tunnelCandidates = filterTunnelCandidates(list);
  if (tunnelCandidates.length === 0) return [];

  const priorityMatched = findPriorityItems(tunnelCandidates, PRIORITY_TUNNELS);

  const priorityItems = priorityMatched
    .map(mapItsItem)
    .filter((item) => item.name && item.url);

  const priorityNameSet = new Set(priorityItems.map((item) => item.name));

  const backupItems = shuffleArray(tunnelCandidates)
    .map(mapItsItem)
    .filter((item) => item.name && item.url)
    .filter((item) => !priorityNameSet.has(item.name))
    .slice(0, 5);

  return [...priorityItems, ...backupItems].slice(0, 8);
}

/* =========================================================
 * ITS OpenAPI 기반 최신 CCTV 목록 조회
 * 우선순위:
 * 1) ITS 최신 조회
 * 2) 실패하면 백엔드 캐시
 * 3) 그것도 실패하면 fallback
 * ========================================================= */
export const fetchTunnelCctvUrl = async (host) => {
  const BACKEND_URL = `http://${host}:5000`;

  /* -----------------------------------------------------
   * 1) ITS 최신 조회 우선
   * ----------------------------------------------------- */
  try {
    const res = await axios.get("https://openapi.its.go.kr:9443/cctvInfo", {
      params: {
        apiKey: "9241caeb859d43b0aaadf26b6b64988a",
        type: "ex",
        cctvType: "1",
        minX: "126.8",
        maxX: "127.89",
        minY: "36.8",
        maxY: "37.9",
        getType: "json",
      },
      timeout: 10000,
    });

    const list = res?.data?.response?.data || [];
    const finalItems = buildFinalTunnelItems(list);

    if (finalItems.length > 0) {
      await setTunnelCctvList(BACKEND_URL, finalItems);

      const priorityNames = new Set(PRIORITY_TUNNELS);
      const hasPriority = finalItems.some((item) => priorityNames.has(item.name));

      return {
        ok: true,
        items: finalItems,
        source: hasPriority ? "priority" : "its_random",
      };
    }

    throw new Error("ITS API 응답에서 유효한 터널 CCTV를 구성하지 못했습니다.");
  } catch (itsError) {
    console.warn("터널 ITS 최신 조회 실패, 캐시 사용 시도:", itsError);

    /* -----------------------------------------------------
     * 2) ITS 실패 시 백엔드 캐시 사용
     * ----------------------------------------------------- */
    try {
      const cachedRes = await getTunnelCctvList(BACKEND_URL);

      if (
        cachedRes?.ok &&
        Array.isArray(cachedRes?.items) &&
        cachedRes.items.length > 0
      ) {
        return {
          ok: true,
          items: cachedRes.items,
          source: "cache",
        };
      }

      throw new Error("캐시된 CCTV 목록이 비어 있습니다.");
    } catch (cacheError) {
      console.warn("터널 캐시 조회 실패, fallback 사용:", cacheError);

      /* ---------------------------------------------------
       * 3) 최종 fallback
       * --------------------------------------------------- */
      const fallbackItems = [
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
      ];

      return {
        ok: true,
        items: fallbackItems,
        source: "fallback",
      };
    }
  }
};
