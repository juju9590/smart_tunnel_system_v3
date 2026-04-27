import React, { useEffect, useMemo, useRef, useState } from "react";
import "./index.css";
import {
  fetchTunnelStatus,
  fetchTunnelCctvUrl,
  selectRandomCctv,
  setTunnelTargetLaneCount,
  stopTunnelStream,
} from "./api";

/* =========================================================
 * 공통 유틸
 * ========================================================= */
const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
const clamp01 = (value) => clamp(Number(value || 0), 0, 1);

function getSpeedHintText(avgSpeed) {
  const speed = Number(avgSpeed || 0);

  if (speed <= 1.3) return "실제속도 30km/h 이하 추정";
  if (speed <= 2.6) return "실제속도 30~50km/h 추정";
  return "실제속도 50km/h 이상 추정";
}

function getLaneReestimateText(status) {
  const s = status?.lane_reestimate_status || "idle";
  const count = Number(status?.lane_reestimate_frame_count || 0);
  const total = Number(status?.lane_reestimate_window || 50);

  if (s === "reestimating") return `재추정 중 (${count}/${total})`;
  if (s === "reestimated") return "재추정 완료";
  if (s === "confirmed") return "확정";
  return "대기";
}

function getStateClass(state) {
  if (state === "NORMAL") return "normal";
  if (state === "CONGESTION") return "congestion";
  if (state === "JAM") return "jam";
  if (state === "ACCIDENT") return "accident";
  if (state === "SUSPECT") return "suspect";
  if (state === "CONFIRMED") return "accident";
  if (state === "ERROR") return "error";
  return "ready";
}

function getVentRiskInfo(level) {
  switch (level) {
    case "NORMAL":
      return { label: "정상", emoji: "🟢", className: "risk-normal" };
    case "CAUTION":
      return { label: "주의", emoji: "🟡", className: "risk-caution" };
    case "WARNING":
      return { label: "경고", emoji: "🟠", className: "risk-warning" };
    case "DANGER":
      return { label: "위험", emoji: "🔴", className: "risk-danger" };
    default:
      return { label: "정상", emoji: "🟢", className: "risk-normal" };
  }
}

function getRiskLevelByScore(score) {
  const s = Number(score || 0);

  if (s >= 0.85) return "DANGER";
  if (s >= 0.6) return "WARNING";
  if (s >= 0.35) return "CAUTION";
  return "NORMAL";
}

function getAirMessage(level) {
  switch (level) {
    case "NORMAL":
      return "공기질 상태 정상";
    case "CAUTION":
      return "공기질 주의 단계";
    case "WARNING":
      return "공기질 경고 단계";
    case "DANGER":
      return "환기 대응 필요";
    default:
      return "공기질 상태 정상";
  }
}

/* =========================================================
 * 현재 상태가 유지된다고 가정한 5분 후 추정치
 * - 프론트 표시용 추정값
 * ========================================================= */
function buildFutureVentPreview(vent, status) {
  const currentScore = clamp01(vent?.risk_score_final ?? 0);
  const density = Number(vent?.traffic_density ?? 0);
  const dwell = Number(vent?.avg_dwell_time_roi ?? 0);
  const avgSpeed = Number(vent?.avg_speed_roi ?? status?.avg_speed_roi ?? 0);
  const vehicleCount = Number(vent?.vehicle_count_roi ?? 0);
  const trafficState = String(status?.state || "NORMAL").toUpperCase();
  const accident = Boolean(status?.accident);

  let stateBonus = 0.04;
  if (trafficState === "CONGESTION") stateBonus = 0.10;
  if (trafficState === "JAM") stateBonus = 0.18;
  if (trafficState === "ACCIDENT" || accident) stateBonus = 0.22;

  const lowSpeedBonus =
    avgSpeed <= 1.2 ? 0.12 : avgSpeed <= 2.0 ? 0.08 : avgSpeed <= 3.0 ? 0.04 : 0.01;

  const densityBonus = density * 0.12;
  const dwellBonus = clamp(dwell / 60, 0, 1) * 0.10;

  const predictedScore = clamp01(
    currentScore + stateBonus + lowSpeedBonus + densityBonus + dwellBonus
  );

  const predictedLevel = getRiskLevelByScore(predictedScore);

  const predictedVehicleCount = Math.max(
    0,
    Math.round(
      trafficState === "NORMAL"
        ? vehicleCount
        : trafficState === "CONGESTION"
        ? vehicleCount * 1.05
        : trafficState === "JAM"
        ? vehicleCount * 1.12
        : vehicleCount * 1.15
    )
  );

  const predictedSpeed = Math.max(
    0,
    trafficState === "NORMAL"
      ? avgSpeed * 0.98
      : trafficState === "CONGESTION"
      ? avgSpeed * 0.92
      : trafficState === "JAM"
      ? avgSpeed * 0.82
      : avgSpeed * 0.75
  );

  const predictedDensity = clamp(
    trafficState === "NORMAL"
      ? density + 0.02
      : trafficState === "CONGESTION"
      ? density + 0.05
      : trafficState === "JAM"
      ? density + 0.08
      : density + 0.10,
    0,
    1.5
  );

  const predictedDwell = Math.max(
    0,
    trafficState === "NORMAL"
      ? dwell * 1.05
      : trafficState === "CONGESTION"
      ? dwell * 1.12
      : trafficState === "JAM"
      ? dwell * 1.22
      : dwell * 1.30
  );

  return {
    risk_score_final: predictedScore,
    risk_level: predictedLevel,
    message: getAirMessage(predictedLevel),
    vehicle_count_roi: predictedVehicleCount,
    avg_speed_roi: predictedSpeed,
    traffic_density: predictedDensity,
    avg_dwell_time_roi: predictedDwell,
  };
}

/* =========================================================
 * 공기질 위험 게이지 공통 컴포넌트
 * ========================================================= */
function AirQualityGauge({ level, score }) {
  const safeScore = clamp01(score);
  const left = `${safeScore * 100}%`;

  return (
    <div className="air-gauge-wrap modern">
      <div className="air-gauge-label-row">
        <span className={level === "NORMAL" ? "active" : ""}>정상</span>
        <span className={level === "CAUTION" ? "active" : ""}>주의</span>
        <span className={level === "WARNING" ? "active" : ""}>경고</span>
        <span className={level === "DANGER" ? "active" : ""}>위험</span>
      </div>

      <div className="air-gauge-track">
        <div className="air-gauge-segment seg-normal" />
        <div className="air-gauge-segment seg-caution" />
        <div className="air-gauge-segment seg-warning" />
        <div className="air-gauge-segment seg-danger" />
        <div className="air-gauge-thumb" style={{ left }} />
      </div>
    </div>
  );
}

function AirMetricCard({ label, value, unit = "" }) {
  return (
    <div className="air-metric-card modern">
      <div className="air-metric-label">{label}</div>
      <div className="air-metric-value">
        {value}
        <span className="air-metric-unit">{unit}</span>
      </div>
    </div>
  );
}

function AirPanel({ title, subtitle, ventData, showForecastBadge = false }) {
  const riskInfo = getVentRiskInfo(ventData?.risk_level);
  const safeScore = clamp01(ventData?.risk_score_final ?? 0);

  return (
    <div className="air-half-panel modern">
      <div className="air-half-header modern">
        <div>
          <div className="air-half-title">{title}</div>
          <div className="air-half-subtitle">{subtitle}</div>
        </div>

        <div className={`air-status-chip modern ${riskInfo.className}`}>
          <span className="air-status-dot" />
          <span className="air-status-text">{riskInfo.label}</span>
          {showForecastBadge && <span className="air-status-badge">예측</span>}
        </div>
      </div>

      <div className="air-score-row">
        <div className="air-score-label">위험 점수</div>
        <div className="air-score-value">{safeScore.toFixed(2)}</div>
      </div>

      <AirQualityGauge level={ventData?.risk_level} score={safeScore} />

      <div className="air-metrics-grid modern">
        <AirMetricCard
          label="차량수"
          value={Number(ventData?.vehicle_count_roi ?? 0)}
          unit=" 대"
        />
        <AirMetricCard
          label="평균속도"
          value={Number(ventData?.avg_speed_roi ?? 0).toFixed(2)}
          unit=" px/s"
        />
        <AirMetricCard
          label="교통밀도"
          value={`${(Number(ventData?.traffic_density ?? 0) * 100).toFixed(0)}`}
          unit=" %"
        />
        <AirMetricCard
          label="평균체류시간"
          value={Number(ventData?.avg_dwell_time_roi ?? 0).toFixed(2)}
          unit=" 초"
        />
      </div>

      <div className="air-message-line modern">
        {ventData?.message || "공기질 상태 정상"}
      </div>
    </div>
  );
}

function TunnelModule({ host }) {
  const [status, setStatus] = useState({
    state: "READY",
    traffic_state: "NORMAL",
    accident_status: "NONE",
    pending_accident_event: null,
    avg_speed: 0,
    avg_speed_roi: 0,
    vehicle_count: 0,
    accident: false,
    lane_count: 0,
    target_lane_count: null,
    lane_count_stable: false,
    template_confirmed: false,
    events: [],
    event_logs: [],
    frame_id: 0,
    cctv_name: "-",
    cctv_url: "",
    dwell_times: {},
    vehicles: [],
    vehicles_in_roi: [],
    lane_reestimate_status: "idle",
    lane_reestimate_frame_count: 0,
    lane_reestimate_window: 50,
    minute_vehicle_count: 0,
    ventilation: {
      risk_score_base: 0,
      risk_score_final: 0,
      risk_level: "NORMAL",
      alarm: false,
      message: "공기질 상태 정상",
      vehicle_count_roi: 0,
      weighted_vehicle_count: 0,
      traffic_density: 0,
      avg_dwell_time_roi: 0,
      avg_speed_roi: 0,
    },
  });

  const [cctvSource, setCctvSource] = useState("");
  const [videoFeedUrl, setVideoFeedUrl] = useState("");
  const [videoLoading, setVideoLoading] = useState(true);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [reestimateLoading, setReestimateLoading] = useState(false);
  const [laneEditMode, setLaneEditMode] = useState(false);
  const [laneTargetInput, setLaneTargetInput] = useState("");
  const [laneTargetLoading, setLaneTargetLoading] = useState(false);
  const [accidentModal, setAccidentModal] = useState(null);
  const [resolvedAccidentIds, setResolvedAccidentIds] = useState(() => new Set());
  const [eventTab, setEventTab] = useState("logs");
  const [eventStats, setEventStats] = useState(null);
  const [statsLoading, setStatsLoading] = useState(false);
  const [lastSelectedCctv, setLastSelectedCctv] = useState(() => {
    try {
      const saved = sessionStorage.getItem(`tunnel_last_cctv_${host}`);
      return saved ? JSON.parse(saved) : null;
    } catch {
      return null;
    }
  });

  const BACKEND_URL = `http://${host}:5000`;

  const prevVentLevelRef = useRef("NORMAL");
  const initOnceRef = useRef(false);
  const videoRestartTimerRef = useRef(null);
  const videoErrorTimerRef = useRef(null);
  const lastRestartAtRef = useRef(0);
  const wasHiddenRef = useRef(false);
  const stoppedOnExitRef = useRef(false);

  const speedHintText = useMemo(() => getSpeedHintText(status.avg_speed), [status.avg_speed]);
  const laneReestimateText = useMemo(() => getLaneReestimateText(status), [
    status.lane_reestimate_status,
    status.lane_reestimate_frame_count,
    status.lane_reestimate_window,
  ]);

  const currentVent = useMemo(() => {
    const raw = status.ventilation || {};
    return {
      risk_score_base: Number(raw.risk_score_base ?? 0),
      risk_score_final: clamp01(raw.risk_score_final ?? 0),
      risk_level: raw.risk_level || "NORMAL",
      alarm: Boolean(raw.alarm),
      message: raw.message || getAirMessage(raw.risk_level || "NORMAL"),
      vehicle_count_roi: Number(raw.vehicle_count_roi ?? 0),
      weighted_vehicle_count: Number(raw.weighted_vehicle_count ?? 0),
      traffic_density: Number(raw.traffic_density ?? 0),
      avg_dwell_time_roi: Number(raw.avg_dwell_time_roi ?? 0),
      avg_speed_roi: Number(raw.avg_speed_roi ?? status.avg_speed_roi ?? 0),
    };
  }, [status.ventilation, status.avg_speed_roi]);

  const futureVent = useMemo(() => {
    return buildFutureVentPreview(currentVent, status);
  }, [currentVent, status]);

  const cctvSourceText = useMemo(() => {
    if (cctvSource === "priority") return "목록 구성: 우선 후보 포함";
    if (cctvSource === "its_random") return "목록 구성: ITS 랜덤";
    if (cctvSource === "cache") return "목록 구성: 캐시";
    if (cctvSource === "fallback") return "목록 구성: 테스트 대체";
    return "";
  }, [cctvSource]);

  const rememberCctv = (data) => {
    const name = data?.cctv_name;
    const url = data?.cctv_url;

    if (!name || name === "-") return;

    const next = { name, url: url || "" };
    setLastSelectedCctv(next);

    try {
      sessionStorage.setItem(`tunnel_last_cctv_${host}`, JSON.stringify(next));
    } catch {
      // sessionStorage를 사용할 수 없는 환경에서는 메모리 상태만 유지한다.
    }
  };

  const stopBackendStream = async () => {
    try {
      await stopTunnelStream(BACKEND_URL);
    } catch (err) {
      console.error("stop tunnel stream error:", err);
    }
  };

  const applyStatusData = (data) => {
    rememberCctv(data);

    setStatus({
      state: data?.state ?? "READY",
      traffic_state: data?.traffic_state ?? data?.state ?? "NORMAL",
      accident_status: data?.accident_status ?? "NONE",
      pending_accident_event: data?.pending_accident_event ?? null,
      avg_speed: Number(data?.avg_speed ?? 0),
      avg_speed_roi: Number(data?.avg_speed_roi ?? 0),
      vehicle_count: Number(data?.vehicle_count ?? 0),
      accident: Boolean(data?.accident ?? false),
      lane_count: Number(data?.lane_count ?? 0),
      target_lane_count:
        data?.target_lane_count === null || data?.target_lane_count === undefined
          ? null
          : Number(data.target_lane_count),
      lane_count_stable: Boolean(data?.lane_count_stable ?? false),
      template_confirmed: Boolean(data?.template_confirmed ?? false),
      events: Array.isArray(data?.events) ? data.events : [],
      event_logs: Array.isArray(data?.event_logs) ? data.event_logs : [],
      frame_id: Number(data?.frame_id ?? 0),
      cctv_name: data?.cctv_name ?? "-",
      cctv_url: data?.cctv_url ?? "",
      dwell_times: data?.dwell_times ?? {},
      vehicles: Array.isArray(data?.vehicles) ? data.vehicles : [],
      vehicles_in_roi: Array.isArray(data?.vehicles_in_roi) ? data.vehicles_in_roi : [],
      lane_reestimate_status: data?.lane_reestimate_status ?? "idle",
      lane_reestimate_frame_count: Number(data?.lane_reestimate_frame_count ?? 0),
      lane_reestimate_window: Number(data?.lane_reestimate_window ?? 50),
      minute_vehicle_count: Number(data?.minute_vehicle_count ?? 0),
      ventilation: data?.ventilation ?? {
        risk_score_base: 0,
        risk_score_final: 0,
        risk_level: "NORMAL",
        alarm: false,
        message: "공기질 상태 정상",
        vehicle_count_roi: 0,
        weighted_vehicle_count: 0,
        traffic_density: 0,
        avg_dwell_time_roi: 0,
        avg_speed_roi: 0,
      },
    });
  };

  const stopVideo = () => {
    if (videoRestartTimerRef.current) {
      clearTimeout(videoRestartTimerRef.current);
      videoRestartTimerRef.current = null;
    }
    if (videoErrorTimerRef.current) {
      clearTimeout(videoErrorTimerRef.current);
      videoErrorTimerRef.current = null;
    }
    setVideoFeedUrl("");
    setVideoLoading(false);
  };

  const restartVideo = (delay = 900, force = false) => {
    const now = Date.now();
    if (!force && now - lastRestartAtRef.current < 1200) return;
    lastRestartAtRef.current = now;

    setVideoLoading(true);

    if (videoRestartTimerRef.current) {
      clearTimeout(videoRestartTimerRef.current);
    }

    setVideoFeedUrl("");

    videoRestartTimerRef.current = setTimeout(() => {
      stoppedOnExitRef.current = false;
      setVideoFeedUrl(`${BACKEND_URL}/api/tunnel/video-feed?t=${Date.now()}`);
    }, delay);
  };

  const retryVideoAfterError = () => {
    if (videoErrorTimerRef.current) return;

    videoErrorTimerRef.current = setTimeout(() => {
      videoErrorTimerRef.current = null;
      restartVideo(0);
    }, 1800);
  };

  useEffect(() => {
    let mounted = true;

    const initialize = async () => {
      try {
        if (!initOnceRef.current) {
          const sessionKey = `tunnel_cctv_initialized_${host}`;
          const alreadyInitialized = sessionStorage.getItem(sessionKey) === "1";

          // 백엔드 실제 캐시 상태 확인
          const currentListRes = await fetch(`${BACKEND_URL}/api/tunnel/cctv-list`);
          const currentListData = await currentListRes.json();

          const backendHasList =
            currentListData?.ok &&
            Array.isArray(currentListData?.items) &&
            currentListData.items.length > 0;

          // 세션상 초기화 안 되었거나, 백엔드 캐시가 비어 있으면 다시 초기화
          if (!alreadyInitialized || !backendHasList) {
            const cctvRes = await fetchTunnelCctvUrl(host);

            if (
              !cctvRes?.ok ||
              !Array.isArray(cctvRes?.items) ||
              cctvRes.items.length === 0
            ) {
              throw new Error("CCTV 리스트를 불러오지 못했습니다.");
            }

            if (!mounted) return;
            setCctvSource(cctvRes?.source || "");
            sessionStorage.setItem(sessionKey, "1");
          }

          initOnceRef.current = true;
        }

        const data = await fetchTunnelStatus(BACKEND_URL);
        if (!mounted) return;

        applyStatusData(data);
        restartVideo(900);
      } catch (err) {
        console.error("initialize error:", err);
        setVideoLoading(false);
        setError("초기 CCTV 리스트 설정 실패");
      }
    };

    const loadStatus = async () => {
      try {
        const data = await fetchTunnelStatus(BACKEND_URL);
        if (!mounted) return;
        applyStatusData(data);
      } catch (err) {
        console.error("status fetch error:", err);
      }
    };

    initialize();
    const timer = setInterval(loadStatus, 1000);

    return () => {
      mounted = false;
      clearInterval(timer);
      stopVideo();
      stoppedOnExitRef.current = true;
      stopBackendStream();
    };
  }, [BACKEND_URL, host]);

  useEffect(() => {
    const currentLevel = currentVent?.risk_level || "NORMAL";
    const prevLevel = prevVentLevelRef.current;

    if (currentLevel === "DANGER" && prevLevel !== "DANGER") {
      alert("🔴 공기질 위험 단계입니다.\n환기 대응 조치가 필요합니다.");
    }

    prevVentLevelRef.current = currentLevel;
  }, [currentVent?.risk_level]);

  useEffect(() => {
    const event = status.pending_accident_event;
    if (!event?.event_id) return;
    if (resolvedAccidentIds.has(event.event_id)) return;
    if (accidentModal?.event_id === event.event_id) return;

    setAccidentModal(event);
  }, [status.pending_accident_event, resolvedAccidentIds, accidentModal?.event_id]);

  const loadEventStats = async () => {
    try {
      setStatsLoading(true);
      const res = await fetch(`${BACKEND_URL}/api/tunnel/event/stats`);
      const data = await res.json();
      setEventStats(data);
    } catch (err) {
      console.error("event stats fetch error:", err);
    } finally {
      setStatsLoading(false);
    }
  };

  useEffect(() => {
    if (eventTab === "stats") {
      loadEventStats();
    }
  }, [eventTab, BACKEND_URL]);

  /* =========================================================
   * 탭 이동 시: 종료
   * 다시 돌아오면: 새로 시작
   * ========================================================= */
  useEffect(() => {
  const handleVisibilityChange = async () => {
    try {
      if (document.visibilityState === "hidden") {
        wasHiddenRef.current = true;
        stopVideo();
        if (!stoppedOnExitRef.current) {
          stoppedOnExitRef.current = true;
          await stopBackendStream();
        }
        return;
      }

      if (document.visibilityState === "visible" && wasHiddenRef.current) {
        wasHiddenRef.current = false;
        // 마지막 CCTV는 백엔드 service.current_cctv가 보존하므로
        // select-random 없이 video-feed만 새 timestamp로 재연결한다.
        restartVideo(300, true);
      }
    } catch (err) {
      console.error("tab visibility stream control error:", err);
    }
  };

  document.addEventListener("visibilitychange", handleVisibilityChange);

  return () => {
    document.removeEventListener("visibilitychange", handleVisibilityChange);
  };
}, [BACKEND_URL]);

  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  const handleRandom = async () => {
  try {
    setLoading(true);
    setError("");

    await stopTunnelStream(BACKEND_URL);
    stopVideo();

    await selectRandomCctv(BACKEND_URL);
    await sleep(1200);
    restartVideo(900, true);
  } catch (err) {
    console.error(err);
    setVideoLoading(false);
    setError("랜덤 CCTV 선택 실패");
  } finally {
    setLoading(false);
  }
};

  const handleRefreshVideo = async () => {
    try {
      setLoading(true);
      setError("");

      // 현재 CCTV는 유지하고 img 연결만 끊은 뒤 새 timestamp로 재연결
      stopVideo();
      restartVideo(300, true);
    } catch (err) {
      console.error("refresh video error:", err);
      setVideoLoading(false);
      setError("영상 새로고침 실패");
    } finally {
      setLoading(false);
    }
  };

  const handleLaneReestimate = async () => {
    try {
      setReestimateLoading(true);
      setError("");

      const res = await fetch(`${BACKEND_URL}/api/tunnel/lane/reestimate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });

      const data = await res.json();

      if (!data?.ok) {
        setError(data?.message || "차선 재추정 요청 실패");
        return;
      }

      const refreshed = await fetchTunnelStatus(BACKEND_URL);
      applyStatusData(refreshed);
    } catch (err) {
      console.error("lane reestimate error:", err);
      setError("차선 재추정 요청 실패");
    } finally {
      setReestimateLoading(false);
    }
  };

  const handleLaneSave = async () => {
    try {
      setError("");

      const res = await fetch(`${BACKEND_URL}/api/tunnel/lane/save`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });

      const data = await res.json();

      if (!data?.ok) {
        setError(data?.message || "차선 저장 실패");
        return;
      }

      alert("현재 차선을 저장했습니다.");
    } catch (err) {
      console.error("lane save error:", err);
      setError("차선 저장 요청 실패");
    }
  };

  const handleLaneTargetApply = async () => {
    const nextLaneCount = Number(laneTargetInput);

    if (![2, 3, 4].includes(nextLaneCount)) {
      setError("목표 차선 수는 2, 3, 4만 입력할 수 있습니다.");
      return;
    }

    try {
      setLaneTargetLoading(true);
      setError("");

      const data = await setTunnelTargetLaneCount(BACKEND_URL, nextLaneCount);

      if (!data?.ok) {
        setError(data?.message || "목표 차선 수 설정 실패");
        return;
      }

      setLaneEditMode(false);
      const refreshed = await fetchTunnelStatus(BACKEND_URL);
      applyStatusData(refreshed);
    } catch (err) {
      console.error("lane target count error:", err);
      setError("목표 차선 수 설정 실패");
    } finally {
      setLaneTargetLoading(false);
    }
  };

  const handleResolveAccident = async (action) => {
    if (!accidentModal?.event_id) return;

    try {
      setLoading(true);
      const res = await fetch(`${BACKEND_URL}/api/tunnel/event/resolve`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          event_id: accidentModal.event_id,
          action,
        }),
      });
      const data = await res.json();

      if (!data?.ok) {
        setError(data?.message || "사고 이벤트 처리 실패");
        return;
      }

      setResolvedAccidentIds((prev) => {
        const next = new Set(prev);
        next.add(accidentModal.event_id);
        return next;
      });
      setAccidentModal(null);

      const refreshed = await fetchTunnelStatus(BACKEND_URL);
      applyStatusData(refreshed);

      if (eventTab === "stats") {
        loadEventStats();
      }
    } catch (err) {
      console.error("resolve accident error:", err);
      setError("사고 이벤트 처리 실패");
    } finally {
      setLoading(false);
    }
  };

  const displayState = useMemo(() => {
    if (status.accident_status === "CONFIRMED") return "🔴 사고 확정";
    if (status.accident_status === "SUSPECT") return "🚨 사고 의심";
    return status.traffic_state || status.state;
  }, [status.accident_status, status.traffic_state, status.state]);

  const stateClass = getStateClass(
    status.accident_status === "CONFIRMED"
      ? "CONFIRMED"
      : status.accident_status === "SUSPECT"
      ? "SUSPECT"
      : status.traffic_state || status.state
  );

  const laneDisplayCount = status.target_lane_count || status.lane_count;
  const hasTargetLaneCount = status.target_lane_count !== null && status.target_lane_count !== undefined;
  const laneTargetPending =
    hasTargetLaneCount &&
    (!status.template_confirmed || Number(status.lane_count) !== Number(status.target_lane_count));
  const laneCountColor = laneTargetPending ? "#facc15" : "#ffffff";
  const laneHelperText = laneTargetPending
    ? status.lane_reestimate_status === "reestimating"
      ? "차선 안정화 중"
      : "재추정 대기중"
    : "";

  return (
    <div className="smart-page">
      <main className="smart-main">
        <section className="panel panel-header">
          <div className="panel-header-left">
            <div className="panel-title">🚨 스마트 터널 시스템</div>
            <div className="panel-subtitle">{status.cctv_name || lastSelectedCctv?.name || "-"}</div>
          </div>

          <div className="panel-header-right">
            <button className="action-btn" onClick={handleRandom} disabled={loading}>
              랜덤선택
            </button>
            <button className="action-btn" onClick={handleRefreshVideo} disabled={loading}>
              영상새로고침
            </button>
          </div>
        </section>

        {loading && <div className="top-notice">처리 중...</div>}
        {error && <div className="top-error">{error}</div>}

        <section className="top-grid">
          <div className="panel panel-video">
            <div className="section-title">📹 CCTV</div>

            <div className="video-wrap">
              {videoLoading && (
                <div className="video-overlay-message">
                  <div>영상 연결 중입니다</div>
                  <div className="video-overlay-sub">
                      화면이 계속 나오지 않으면 영상새로고침을 눌러주세요
                  </div>
                </div>
              )}

              {videoFeedUrl && (
                <img
                  key={videoFeedUrl}
                  src={videoFeedUrl}
                  alt="cctv"
                  className="video-image"
                  onLoad={() => {
                    setVideoLoading(false);
                    setError("");
                  }}
                  onError={() => {
                    setVideoLoading(false);
                    setError("영상 스트리밍 연결 실패");
                    retryVideoAfterError();
                  }}
                />
              )}
            </div>

            <div className="video-caption">{status.cctv_name || lastSelectedCctv?.name || "-"}</div>
            {cctvSourceText && <div className="video-debug-source">{cctvSourceText}</div>}
          </div>

          <div className="panel panel-status">
            <div className="status-title-row">
              <div className="section-title">🚦 교통흐름 상태</div>
              <div className="traffic-legend">
                <span className="traffic-legend-item normal">정상</span>
                <span className="traffic-legend-item congestion">혼잡</span>
                <span className="traffic-legend-item jam">정체</span>
                <span className="traffic-legend-item accident">사고</span>
              </div>
            </div>

            <div className="status-subpanel traffic-state-card">
              <div className={`state-badge ${stateClass}`}>{displayState}</div>

              <div className="status-speed-line">
                <span className="status-speed-main">
                  평균속도 : {status.avg_speed.toFixed(2)} px/s
                </span>
                <span className="status-speed-sub">({speedHintText})</span>
              </div>
            </div>

            <div className="status-subpanel lane-management-card">
              <div className="status-subpanel-title">차선 관리</div>

              <div className="traffic-summary-grid">
                <div className="summary-card">
                  <div className="summary-card-label">차선수</div>
                  {laneEditMode ? (
                    <>
                      <div className="summary-card-value small">목표 차선 수</div>
                      <input
                        type="number"
                        min="2"
                        max="4"
                        value={laneTargetInput}
                        onChange={(event) => setLaneTargetInput(event.target.value)}
                        style={{
                          width: "64px",
                          marginTop: "6px",
                          padding: "4px 6px",
                          color: "#fff",
                          background: "#111827",
                          border: "1px solid #4b5563",
                          borderRadius: "6px",
                        }}
                      />
                      <div style={{ display: "flex", gap: "6px", marginTop: "8px" }}>
                        <button
                          className="summary-mini-btn"
                          onClick={handleLaneTargetApply}
                          disabled={laneTargetLoading}
                        >
                          {laneTargetLoading ? "적용중..." : "적용"}
                        </button>
                        <button
                          className="summary-mini-btn secondary"
                          onClick={() => {
                            setLaneEditMode(false);
                            setLaneTargetInput("");
                          }}
                          disabled={laneTargetLoading}
                        >
                          취소
                        </button>
                      </div>
                    </>
                  ) : (
                    <>
                      <div className="summary-card-value" style={{ color: laneCountColor }}>
                        {laneDisplayCount}차선
                      </div>
                      {laneHelperText && (
                        <div className="summary-card-value small" style={{ color: "#facc15" }}>
                          {laneHelperText}
                        </div>
                      )}
                      <button
                        className="summary-mini-btn"
                        onClick={() => {
                          setLaneTargetInput(String(status.target_lane_count || status.lane_count || 2));
                          setLaneEditMode(true);
                        }}
                      >
                        수정
                      </button>
                    </>
                  )}
                </div>

                <div className="summary-card">
                  <div className="summary-card-label">차선재추정</div>
                  <div className="summary-card-value small">{laneReestimateText}</div>
                  <button
                    className="summary-mini-btn"
                    onClick={handleLaneReestimate}
                    disabled={reestimateLoading}
                  >
                    {reestimateLoading ? "요청중..." : "재추정"}
                  </button>
                </div>

                <div className="summary-card">
                  <div className="summary-card-label">차선저장</div>
                  <div className="summary-card-value small">현재 차선 메모리</div>
                  <button className="summary-mini-btn secondary" onClick={handleLaneSave}>
                    저장
                  </button>
                </div>
              </div>
            </div>

            <div className="status-subpanel event-log-card">
              <div className="event-card-tabs">
                <button
                  className={`event-tab-btn ${eventTab === "logs" ? "active" : ""}`}
                  onClick={() => setEventTab("logs")}
                >
                  이벤트 로그
                </button>
                <button
                  className={`event-tab-btn ${eventTab === "stats" ? "active" : ""}`}
                  onClick={() => setEventTab("stats")}
                >
                  이벤트 통계관리
                </button>
              </div>

              {eventTab === "logs" ? (
                <div className="event-log scrollable-log">
                  {status.event_logs && status.event_logs.length > 0 ? (
                    status.event_logs
                      .slice()
                      .reverse()
                      .map((event, idx) => (
                        <div key={`${event}-${idx}`} className="event-item">
                          {event}
                        </div>
                      ))
                  ) : status.events.length > 0 ? (
                    status.events.map((event, idx) => (
                      <div key={`${event}-${idx}`} className="event-item">
                        {event}
                      </div>
                    ))
                  ) : (
                    <div className="event-empty">이상 징후 없음 (정상 흐름 유지)</div>
                  )}
                </div>
              ) : (
                <div className="event-stats-panel scrollable-log">
                  {statsLoading ? (
                    <div className="event-empty">통계 불러오는 중...</div>
                  ) : (
                    <>
                      <div className="stats-date-line">
                        기준일: {eventStats?.date || "-"}
                      </div>
                      <div className="stats-grid">
                        <div className="stats-row">
                          <span>사고 의심</span>
                          <strong>{eventStats?.total_suspect ?? 0}건</strong>
                        </div>
                        <div className="stats-row">
                          <span>사고 확정</span>
                          <strong>{eventStats?.confirmed ?? 0}건</strong>
                        </div>
                        <div className="stats-row">
                          <span>이상 없음</span>
                          <strong>{eventStats?.false_alarm ?? 0}건</strong>
                        </div>
                        <div className="stats-row">
                          <span>확정률</span>
                          <strong>{eventStats?.confirm_rate ?? 0}%</strong>
                        </div>
                        <div className="stats-row">
                          <span>오탐률</span>
                          <strong>{eventStats?.false_alarm_rate ?? 0}%</strong>
                        </div>
                      </div>
                      <div className="recent-title">최근 처리 기록</div>
                      <div className="recent-events-list">
                        {(eventStats?.recent_events || []).length > 0 ? (
                          eventStats.recent_events.map((event) => (
                            <div key={event.event_id || `${event.event_datetime}-${event.cctv_name}`} className="recent-event-item">
                              <span>{event.event_time}</span>
                              <strong>{event.operator_action || event.event_status}</strong>
                              <em>{event.cctv_name || "-"}</em>
                            </div>
                          ))
                        ) : (
                          <div className="event-empty compact">최근 처리 기록 없음</div>
                        )}
                      </div>
                    </>
                  )}
                </div>
              )}
            </div>
          </div>
        </section>

        <section className="panel panel-air">
          <div className="panel-air-grid">
            <AirPanel
              title="현재 터널 공기질"
              subtitle="(기준 : ROI 기반)"
              ventData={currentVent}
            />

            <AirPanel
              title="5분 후 터널 공기질 추정"
              subtitle="(기준 : 현재 상태로 지속 유지 시)"
              ventData={futureVent}
              showForecastBadge
            />
          </div>
        </section>
      </main>

      {accidentModal && (
        <div className="accident-modal-backdrop">
          <div className="accident-modal">
            <div className="accident-modal-title">🚨 사고 이벤트 감지</div>
            <div className="accident-modal-body">
              <div>
                <span>CCTV</span>
                <strong>{accidentModal.cctv_name || "-"}</strong>
              </div>
              <div>
                <span>날짜</span>
                <strong>{accidentModal.event_date || "-"}</strong>
              </div>
              <div>
                <span>시간</span>
                <strong>{accidentModal.event_time || "-"}</strong>
              </div>
              <div>
                <span>안내</span>
                <strong>
                  AI가 사고 상황으로 판단했습니다.
                  <br />
                  현재 상황을 확인해 주세요.
                </strong>
              </div>
            </div>
            <div className="accident-modal-actions">
              <button
                className="accident-action-btn confirm"
                onClick={() => handleResolveAccident("confirm")}
                disabled={loading}
              >
                사고 확정
              </button>
              <button
                className="accident-action-btn normal"
                onClick={() => handleResolveAccident("normal")}
                disabled={loading}
              >
                이상 없음
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default TunnelModule;
