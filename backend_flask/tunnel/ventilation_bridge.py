# ==========================================
# 파일명: ventilation_bridge.py
# 위치: backend_flask/modules/tunnel/ventilation_bridge.py
# 역할:
# - pipeline result -> ventilation input 변환
# - VentilationRiskManager 호출
# - service.py를 가볍게 유지하기 위한 브릿지 함수
# ==========================================


def build_ventilation_result(result, ventilation_manager):
    """
    pipeline 결과(result)에서 환기 대응 계산에 필요한 값을 추출해
    ventilation_result 생성

    기대 우선순위:
    1) result["vehicles_in_roi"] 제공
    2) 없으면 result["vehicles"]에서 roi_in=True 인 차량 사용
    3) 둘 다 없으면 빈 리스트 처리
    """

    frame_id = int(result.get("frame_id", 0))
    lane_count = int(result.get("lane_count", 0) or 0)
    avg_speed = float(result.get("avg_speed", 0.0) or 0.0)
    accident_status = str(result.get("accident_status", "NONE") or "NONE").upper()
    accident_confirmed = bool(result.get("accident_confirmed", False))
    accident_applied = accident_status == "CONFIRMED" or accident_confirmed
    state = str(result.get("state", "NORMAL")).upper()
    traffic_state = str(result.get("traffic_state", state) or state).upper()

    # 관제 확인 전 사고 의심은 환기 위험도 사고 가중치에 반영하지 않는다.
    traffic_state_for_vent = "ACCIDENT" if accident_applied else traffic_state

    # 허용 상태 외 값 방어
    allowed_states = {"NORMAL", "CONGESTION", "JAM", "ACCIDENT"}
    if traffic_state_for_vent not in allowed_states:
        traffic_state_for_vent = "NORMAL"

    # --------------------------------------------------
    # ROI 차량 리스트 추출
    # --------------------------------------------------
    vehicles_in_roi = []

    if isinstance(result.get("vehicles_in_roi"), list):
        vehicles_in_roi = result.get("vehicles_in_roi", [])

    elif isinstance(result.get("vehicles"), list):
        raw_vehicles = result.get("vehicles", [])
        for v in raw_vehicles:
            if not isinstance(v, dict):
                continue

            if v.get("roi_in") is True:
                vehicles_in_roi.append(v)

    # --------------------------------------------------
    # avg_speed_roi
    # --------------------------------------------------
    avg_speed_roi = float(result.get("avg_speed_roi", avg_speed) or 0.0)

    # --------------------------------------------------
    # roi_est_length (선택)
    # --------------------------------------------------
    roi_est_length = result.get("roi_est_length", None)

    # lane_count 방어
    if lane_count <= 0:
        lane_count = 1

    try:
        ventilation_result = ventilation_manager.update(
            frame_id=frame_id,
            lane_count=lane_count,
            traffic_state=traffic_state_for_vent,
            vehicles_in_roi=vehicles_in_roi,
            avg_speed_roi=avg_speed_roi,
            roi_est_length=roi_est_length,
            accident_status=accident_status,
            accident_applied=accident_applied
        )
    except Exception as e:
        print(f"⚠️ 환기 대응 계산 실패: {e}")
        ventilation_result = {
            "risk_score_base": 0.0,
            "risk_score_final": 0.0,
            "risk_level": "NORMAL",
            "alarm": False,
            "message": f"환기 계산 실패: {e}",
            "vehicle_count_roi": 0,
            "weighted_vehicle_count": 0.0,
            "traffic_density": 0.0,
            "avg_dwell_time_roi": 0.0,
            "accident_status": accident_status,
            "accident_applied": accident_applied,
        }

    ventilation_result["accident_status"] = accident_status
    ventilation_result["accident_applied"] = accident_applied
    if accident_status == "SUSPECT" and not accident_applied:
        ventilation_result["accident_note"] = "사고 의심은 관제 확인 전까지 공기질 가중치에 반영하지 않음"

    return ventilation_result
