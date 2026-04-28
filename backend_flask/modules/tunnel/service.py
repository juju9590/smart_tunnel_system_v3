# ==========================================
# 파일명: service.py
# 위치: backend_flask/modules/tunnel/service.py
# 역할:
# - 프론트에서 받은 CCTV 후보 리스트 캐시
# - 캐시된 리스트 기반 랜덤 선택 / 이름 선택
# - CCTV 변경 시 이전 스트림 종료 + 파이프라인 reset
# - 연결 가능한 CCTV health 관리
# - 영상 스트리밍
# - 최신 상태 저장
# - 단일 스트림 보호
# - lane_template 수동 재추정 요청 연결
# - 사고 이벤트 캡처/저장
# - 환기 대응 위험도 계산 / 상태 포함
# ==========================================

import os
import json
import random
import threading
import cv2
import time
from datetime import datetime
from pathlib import Path

from .pipeline_adapter import TunnelPipelineAdapter
from .ventilation_risk import VentilationRiskManager
from .ventilation_bridge import build_ventilation_result
from .event_logger import TunnelEventLogger


BAD_CCTV_COOLDOWN = 60
MAX_OPEN_RETRY = 8
ACCIDENT_POPUP_COOLDOWN_SEC = 30


class TunnelLiveService:
    def __init__(self):
        self.lock = threading.Lock()
        self.tunnel_dir = Path(__file__).resolve().parent
        self.runtime_root = self.tunnel_dir / "runtime_data"
        self.runtime_log_dir = self.runtime_root / "logs"
        self.runtime_capture_dir = self.runtime_root / "captures"
        self.runtime_lane_memory_dir = self.runtime_root / "lane_memory"
        self.good_cctv_cache_path = self.runtime_root / "good_cctv_cache.json"

        for path in (
            self.runtime_root,
            self.runtime_log_dir,
            self.runtime_capture_dir,
            self.runtime_lane_memory_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

        # 프론트가 넘겨준 CCTV 후보 리스트
        self.cached_cctv_list = []

        self.current_cctv = None
        self.pipeline = TunnelPipelineAdapter()
        self.pipeline.current_dir = str(self.runtime_root)
        self.active_cctv_name = None

        # CCTV 변경 시 이전 스트림 종료용 토큰
        self.stream_token = 0

        # 단일 스트림 보호용
        self.stream_lock = threading.Lock()
        self.stream_active = False
        self.active_capture = None
        self.latest_frame_bytes = None

        # CCTV 연결 상태 관리
        self.cctv_health = {}
        self.bad_cctv_cache = {}
        self.good_cctv_cache = {}
        self.bad_cctv_ttl_sec = BAD_CCTV_COOLDOWN
        self.pending_user_random = False
        self.user_random_candidates = []
        self.user_random_exclude_name = None

        # ==================================================
        # 환기 대응 매니저
        # ==================================================
        self.ventilation_manager = VentilationRiskManager(fps=10)

        # 설정 확정값
        self.ventilation_manager.capacity_per_lane = 10
        self.ventilation_manager.weight_clip_min = 0.8
        self.ventilation_manager.weight_clip_max = 1.5
        self.ventilation_manager.accident_state_bonus = 0.15
        self.ventilation_manager.free_flow_speed = 8.0
        self.ventilation_manager.max_dwell_time = 60.0

        # 차선 수별 기준 승용차 bbox
        self.ventilation_manager.set_bbox_ref(2, 3500.0)
        self.ventilation_manager.set_bbox_ref(3, 2600.0)
        self.ventilation_manager.set_bbox_ref(4, 2000.0)

        # 필요 시 500m 환산 활성화
        # self.ventilation_manager.enable_tunnel_scaling(True, default_roi_est_length=50.0)

        # --------------------------------------------------
        # 최신 상태 캐시
        # --------------------------------------------------
        self.latest_status = {
            "state": "READY",
            "traffic_state": "NORMAL",
            "accident_status": "NONE",
            "pending_accident_event": None,
            "avg_speed": 0.0,
            "vehicle_count": 0,
            "accident": False,
            "accident_locked": False,
            "lane_count": 0,
            "target_lane_count": None,
            "lane_count_stable": False,
            "template_confirmed": False,
            "events": [],
            "event_logs": [],
            "event_log_entries": [],
            "frame_id": 0,
            "accident_candidate_only": False,
            "frame_accident_prediction": False,
            "recent_prediction_count": 0,
            "weak_suspect": False,
            "strong_suspect": False,
            "confirm_candidate": False,
            "weak_confirmed": False,
            "has_real_accident_evidence": False,
            "has_final_accident_evidence": False,
            "final_accumulation_blocked": False,
            "accident_score": 0,
            "reasons": "",
            "cctv_name": "-",
            "cctv_url": "",

            "lane_reestimate_status": "idle",
            "lane_reestimate_frame_count": 0,
            "lane_reestimate_window": 50,

            "minute_vehicle_count": 0,

            "ventilation": {
                "risk_score_base": 0.0,
                "risk_score_final": 0.0,
                "risk_level": "NORMAL",
                "alarm": False,
                "message": "공기질 상태 정상",
                "vehicle_count_roi": 0,
                "weighted_vehicle_count": 0.0,
                "traffic_density": 0.0,
                "avg_dwell_time_roi": 0.0,
            }
        }

        # ==================================================
        # 사고 이벤트 저장 폴더
        # ==================================================
        self.event_root = os.path.join(os.path.dirname(__file__), "event_storage")
        self.event_snapshot_dir = str(self.runtime_capture_dir)
        self.event_log_dir = str(self.runtime_log_dir)

        self.event_logger = TunnelEventLogger(self.runtime_root)
        self.current_accident_event = None
        self.resolved_event_ids = set()
        self.false_alarm_suppress_until_frame = -1
        self.last_saved_accident_frame = -999999
        self.accident_save_cooldown = 180
        self.last_accident_popup_ts = 0.0
        self.event_log_seq = 0

    # =========================================================
    # 1) CCTV 목록 관리
    # =========================================================
    def get_cctv_list(self):
        return self.cached_cctv_list

    def refresh_cctv_list(self):
        return self.cached_cctv_list

    def set_cctv_list(self, cctv_list):
        if not isinstance(cctv_list, list):
            return False

        cleaned = []
        for item in cctv_list:
            if not isinstance(item, dict):
                continue

            name = str(item.get("name", "")).strip()
            url = str(item.get("url", "")).strip()

            if not name or not url:
                continue

            cleaned.append({
                "name": name,
                "url": url
            })

        if not cleaned:
            return False

        self.cached_cctv_list = cleaned

        self.cctv_health = {}
        self.bad_cctv_cache = {}
        self.good_cctv_cache = self._load_good_cctv_cache()
        for cctv in self.cached_cctv_list:
            self.cctv_health[cctv["url"]] = {
                "ok": None,
                "fail_count": 0,
            }

        print(f"📡 [최초 캐시 저장] {len(self.cached_cctv_list)}개 CCTV 고정 완료")
        return True

    # =========================================================
    # 2) 상태 업데이트
    # =========================================================
    def get_status(self):
        with self.lock:
            data = dict(self.latest_status)

        data.setdefault("lane_reestimate_status", "idle")
        data.setdefault("lane_reestimate_frame_count", 0)
        data.setdefault("lane_reestimate_window", 50)
        data.setdefault("target_lane_count", None)
        data.setdefault("template_confirmed", False)
        target_lane_count = data.get("target_lane_count")
        data["lane_count_stable"] = (
            target_lane_count is not None
            and int(data.get("lane_count", 0) or 0) == int(target_lane_count)
            and bool(data.get("template_confirmed", False))
        )
        data.setdefault("minute_vehicle_count", 0)
        data.setdefault("traffic_state", data.get("state", "NORMAL"))
        data.setdefault("accident_status", "NONE")
        data.setdefault("pending_accident_event", None)
        data.setdefault("ventilation", {
            "risk_score_base": 0.0,
            "risk_score_final": 0.0,
            "risk_level": "NORMAL",
            "alarm": False,
            "message": "공기질 상태 정상",
        })

        return data

    def _update_status(self, data):
        with self.lock:
            if "events" not in data:
                data["events"] = []

            status_accident_safety_corrected = bool(data.get("status_accident_safety_corrected", False))
            accident_locked_in = bool(data.get("accident_locked", self.latest_status.get("accident_locked", False)))
            if bool(data.get("accident", False)) and not accident_locked_in:
                defense_queue_only = bool(data.get(
                    "defense_queue_only_without_real_evidence",
                    self.latest_status.get("defense_queue_only_without_real_evidence", False),
                ))
                has_real_evidence = bool(data.get(
                    "has_real_accident_evidence",
                    self.latest_status.get("has_real_accident_evidence", False),
                ))
                has_final_evidence = bool(data.get(
                    "has_final_accident_evidence",
                    self.latest_status.get("has_final_accident_evidence", False),
                ))
                confirm_candidate = bool(data.get(
                    "confirm_candidate",
                    self.latest_status.get("confirm_candidate", False),
                ))
                recent_prediction_count = int(data.get(
                    "recent_prediction_count",
                    self.latest_status.get("recent_prediction_count", 0),
                ) or 0)

                if (
                    defense_queue_only
                    or (
                        not has_real_evidence
                        and not confirm_candidate
                        and recent_prediction_count == 0
                    )
                    or not (
                        has_final_evidence
                        or confirm_candidate
                    )
                ):
                    data["accident"] = False
                    status_accident_safety_corrected = True

            data["status_accident_safety_corrected"] = status_accident_safety_corrected

            accident_now = bool(data.get("accident", self.latest_status.get("accident", False)))
            accident_locked_now = bool(data.get("accident_locked", self.latest_status.get("accident_locked", False)))
            if not accident_now and not accident_locked_now and isinstance(data.get("events"), list):
                data["events"] = [
                    event for event in data["events"]
                    if "사고 감지" not in str(event)
                ]
            if not accident_now and not accident_locked_now and isinstance(data.get("event_logs"), list):
                data["event_logs"] = [
                    event for event in data["event_logs"]
                    if "사고 감지" not in str(event)
                ]
            if not accident_now and not accident_locked_now and isinstance(data.get("event_log_entries"), list):
                data["event_log_entries"] = [
                    event for event in data["event_log_entries"]
                    if "사고 감지" not in str(event.get("text", event.get("message", "")))
                ]

            self.latest_status.update(data)

            self.latest_status.setdefault("lane_reestimate_status", "idle")
            self.latest_status.setdefault("lane_reestimate_frame_count", 0)
            self.latest_status.setdefault("lane_reestimate_window", 50)
            self.latest_status.setdefault("target_lane_count", None)
            self.latest_status.setdefault("template_confirmed", False)
            target_lane_count = self.latest_status.get("target_lane_count")
            self.latest_status["lane_count_stable"] = (
                target_lane_count is not None
                and int(self.latest_status.get("lane_count", 0) or 0) == int(target_lane_count)
                and bool(self.latest_status.get("template_confirmed", False))
            )
            self.latest_status.setdefault("minute_vehicle_count", 0)
            self.latest_status.setdefault("traffic_state", self.latest_status.get("state", "NORMAL"))
            self.latest_status.setdefault("accident_locked", False)
            self.latest_status.setdefault("accident_status", "NONE")
            self.latest_status.setdefault("pending_accident_event", None)
            self.latest_status.setdefault("event_logs", [])
            self.latest_status.setdefault("event_log_entries", [])
            self.latest_status.setdefault("recent_prediction_count", 0)
            self.latest_status.setdefault("frame_accident_prediction", False)
            self.latest_status.setdefault("accident_candidate_only", False)
            self.latest_status.setdefault("ventilation", {
                "risk_score_base": 0.0,
                "risk_score_final": 0.0,
                "risk_level": "NORMAL",
                "alarm": False,
                "message": "공기질 상태 정상",
            })

    # =========================================================
    # 2-1) 차선 재추정 요청
    # =========================================================
    def request_lane_reestimate(self):
        frame_id = self.latest_status.get("frame_id", 0)

        lane_template = None

        if hasattr(self.pipeline, "get_lane_template"):
            lane_template = self.pipeline.get_lane_template()

        if lane_template is None:
            return {
                "ok": False,
                "message": "lane_template 객체를 찾지 못했습니다."
            }

        if not hasattr(lane_template, "request_reestimate"):
            return {
                "ok": False,
                "message": "lane_template에 request_reestimate() 메서드가 없습니다."
            }

        lane_template.request_reestimate(frame_id=frame_id)

        self._update_status({
            "lane_reestimate_status": "reestimating",
            "lane_reestimate_frame_count": 0,
            "lane_reestimate_window": 50,
            "events": [f"차선 재추정 요청 접수 (frame={frame_id})"]
        })

        print(f"🔄 service 차선 재추정 요청 완료: frame_id={frame_id}")

        return {
            "ok": True,
            "message": "차선 재추정 요청 접수",
            "frame_id": frame_id,
            "window": 50
        }

    def save_current_lane_memory(self):
        lane_template = None

        if hasattr(self.pipeline, "get_lane_template"):
            lane_template = self.pipeline.get_lane_template()

        if lane_template is None:
            return {
                "ok": False,
                "message": "lane_estimator 객체를 찾지 못했습니다."
            }

        self._redirect_lane_memory_to_runtime()

        if not hasattr(lane_template, "save_lane_memory"):
            return {
                "ok": False,
                "message": "save_lane_memory() 메서드가 없습니다."
            }

        cctv_name = self.latest_status.get("cctv_name", "")
        save_path = lane_template.save_lane_memory(cctv_name=cctv_name)

        if not save_path:
            return {
                "ok": False,
                "message": "차선 저장 실패"
            }

        self._update_status({
            "events": [f"차선 저장 완료: {cctv_name}"]
        })

        return {
            "ok": True,
            "message": "현재 차선 저장 완료",
            "path": save_path
        }

    def set_target_lane_count(self, lane_count):
        try:
            lane_count = int(lane_count)
        except Exception:
            return {
                "ok": False,
                "message": "lane_count는 정수여야 합니다."
            }

        if lane_count not in (2, 3, 4):
            return {
                "ok": False,
                "message": "목표 차선 수는 2, 3, 4 중 하나여야 합니다."
            }

        lane_template = None
        if hasattr(self.pipeline, "get_lane_template"):
            lane_template = self.pipeline.get_lane_template()

        if lane_template is None:
            return {
                "ok": False,
                "message": "lane_template 객체를 찾지 못했습니다."
            }

        if hasattr(lane_template, "set_target_lane_count"):
            ok = lane_template.set_target_lane_count(lane_count)
        else:
            lane_template.manual_lane_count = lane_count
            ok = True

        if not ok:
            return {
                "ok": False,
                "message": "목표 차선 수 설정 실패"
            }

        current_lane_count = int(self.latest_status.get("lane_count", 0) or 0)
        template_confirmed = bool(getattr(lane_template, "template_confirmed", False))
        lane_count_stable = (
            current_lane_count == lane_count
            and template_confirmed is True
        )

        self._update_status({
            "target_lane_count": lane_count,
            "lane_count_stable": lane_count_stable,
            "template_confirmed": template_confirmed,
            "events": [f"목표 차선 수 {lane_count}차선 설정"],
        })

        return {
            "ok": True,
            "target_lane_count": lane_count,
            "message": f"목표 차선 수가 {lane_count}차선으로 설정되었습니다."
        }

    # =========================================================
    # 2-2) 사고 이벤트 캡처/저장
    # =========================================================
    def _traffic_state_from_status(self, status_data):
        state = str(status_data.get("traffic_state") or status_data.get("state") or "NORMAL").upper()
        if state == "ACCIDENT":
            return str(self.latest_status.get("traffic_state") or "JAM").upper()
        if state in ("NORMAL", "CONGESTION", "JAM"):
            return state
        return "NORMAL"

    def _build_accident_reason(self, status_data):
        events = status_data.get("events") or status_data.get("event_logs") or []
        if isinstance(events, list):
            accident_events = [str(event) for event in events if "사고" in str(event)]
            if accident_events:
                return " / ".join(accident_events[-2:])
        return "속도 급감/거리 변화/정지 패턴"

    def _is_final_accident_for_popup(self, status_data):
        if not (
            bool(status_data.get("accident", False))
            or bool(status_data.get("accident_locked", False))
        ):
            return False

        if bool(status_data.get("defense_queue_only_without_real_evidence", False)):
            return False

        return True

    def _save_accident_event(self, frame, status_data):
        accident_flag = self._is_final_accident_for_popup(status_data)
        frame_id = int(status_data.get("frame_id", 0))

        # 팝업/이벤트 저장은 AI 사고 확정 플래그에서만 시작한다.
        # weak_suspect, strong_suspect, confirm_candidate, recent_prediction_count는
        # status/log 디버그용으로만 사용하고 여기서는 보지 않는다.
        if not accident_flag:
            if self.latest_status.get("accident_status") == "SUSPECT":
                return
            if self.latest_status.get("accident_status") != "CONFIRMED":
                self.current_accident_event = None
                self._update_status({
                    "accident": False,
                    "accident_status": "NONE",
                    "pending_accident_event": None,
                    "traffic_state": self._traffic_state_from_status(status_data),
                })
            return

        if frame_id <= self.false_alarm_suppress_until_frame:
            self._update_status({
                "accident": False,
                "accident_status": "FALSE_ALARM",
                "pending_accident_event": None,
                "traffic_state": self._traffic_state_from_status(status_data),
            })
            return

        if self.current_accident_event:
            self._update_status({
                "accident": False,
                "accident_status": "SUSPECT",
                "pending_accident_event": self.current_accident_event,
                "traffic_state": self.current_accident_event.get("traffic_state", self._traffic_state_from_status(status_data)),
            })
            return

        now_ts = time.time()
        if now_ts - self.last_accident_popup_ts < ACCIDENT_POPUP_COOLDOWN_SEC:
            return

        if frame_id - self.last_saved_accident_frame < self.accident_save_cooldown:
            return

        self.last_saved_accident_frame = frame_id
        self.last_accident_popup_ts = now_ts

        now = datetime.now()
        ts_compact = now.strftime("%Y%m%d_%H%M%S")
        ts_text = now.strftime("%Y-%m-%d %H:%M:%S")
        event_date = now.strftime("%Y-%m-%d")
        event_time = now.strftime("%H:%M:%S")

        event_id = f"EVT_{ts_compact}_{frame_id}"

        image_path = os.path.join(self.event_snapshot_dir, f"{event_id}.jpg")
        cv2.imwrite(image_path, frame)
        capture_path = os.path.relpath(image_path, self.tunnel_dir).replace("\\", "/")
        traffic_state = self._traffic_state_from_status(status_data)
        reason = self._build_accident_reason(status_data)

        payload = {
            "event_id": event_id,
            "timestamp": ts_text,
            "event_date": event_date,
            "event_time": event_time,
            "event_datetime": ts_text,
            "frame_id": frame_id,
            "type": "accident",
            "state": "SUSPECT",
            "event_status": "SUSPECT",
            "cctv_name": status_data.get("cctv_name", "-"),
            "cctv_url": status_data.get("cctv_url", ""),
            "avg_speed": float(status_data.get("avg_speed", 0.0)),
            "vehicle_count": int(status_data.get("vehicle_count", 0)),
            "lane_count": int(status_data.get("lane_count", 0)),
            "traffic_state": traffic_state,
            "reason": reason,
            "snapshot_path": image_path,
            "capture_path": capture_path,
            "events": status_data.get("events", []),
            "ventilation": status_data.get("ventilation", {}),
        }

        json_path = os.path.join(self.event_log_dir, f"{event_id}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        pending_event = {
            "event_id": event_id,
            "event_date": event_date,
            "event_time": event_time,
            "event_datetime": ts_text,
            "cctv_name": payload["cctv_name"],
            "reason": reason,
            "frame_id": frame_id,
            "traffic_state": traffic_state,
        }
        self.current_accident_event = pending_event
        self.event_logger.append_suspect_event({
            "event_id": event_id,
            "event_date": event_date,
            "event_time": event_time,
            "event_datetime": ts_text,
            "cctv_name": payload["cctv_name"],
            "event_type": "ACCIDENT",
            "event_status": "SUSPECT",
            "operator_action": "",
            "frame_id": frame_id,
            "traffic_state": traffic_state,
            "avg_speed": payload["avg_speed"],
            "vehicle_count": payload["vehicle_count"],
            "lane_count": payload["lane_count"],
            "reason": reason,
            "capture_path": capture_path,
        })

        self.event_log_seq += 1
        accident_event_log_entry = {
            "event_id": self.event_log_seq,
            "timestamp": event_time,
            "message": "사고 감지",
            "text": f"[{event_time}] 사고 감지",
        }
        event_logs = list(self.latest_status.get("event_logs") or [])
        event_log_entries = list(self.latest_status.get("event_log_entries") or [])
        event_logs.append(accident_event_log_entry["text"])
        event_logs.append(f"[{event_time}] AI 사고 확정 감지")
        event_log_entries.append(accident_event_log_entry)
        self._update_status({
            "accident": bool(status_data.get("accident", False)),
            "accident_locked": bool(status_data.get("accident_locked", False)),
            "accident_status": "SUSPECT",
            "pending_accident_event": pending_event,
            "traffic_state": traffic_state,
            "event_logs": event_logs[-10:],
            "event_log_entries": event_log_entries[-10:],
            "events": ["사고 감지", "AI 사고 확정 감지"],
        })

        print(f"📸 사고 이벤트 저장 완료: {event_id}")

    def _reset_false_alarm_accident_state(self, frame_id, traffic_state):
        cctv_name = self.latest_status.get("cctv_name", "-")

        self.false_alarm_suppress_until_frame = max(
            self.false_alarm_suppress_until_frame,
            frame_id + self.accident_save_cooldown,
        )

        if hasattr(self.pipeline, "clear_accident_state"):
            self.pipeline.clear_accident_state(
                cctv_name=cctv_name,
                frame_id=frame_id,
            )

        self.current_accident_event = None
        self.prev_accident_flag = False
        self.last_saved_accident_frame = max(
            self.last_saved_accident_frame,
            frame_id,
        )
        self.last_accident_popup_ts = time.time()

        self._update_status({
            "accident": False,
            "accident_locked": False,
            "accident_candidate_only": False,
            "frame_accident_prediction": False,
            "recent_prediction_count": 0,
            "weak_suspect": False,
            "strong_suspect": False,
            "confirm_candidate": False,
            "weak_confirmed": False,
            "has_real_accident_evidence": False,
            "has_final_accident_evidence": False,
            "final_accumulation_blocked": False,
            "accident_score": 0,
            "reasons": "",
            "traffic_state": traffic_state,
            "state": traffic_state,
            "pending_accident_event": None,
        })

        print(
            "[ACCIDENT FALSE ALARM RESET] "
            "histories cleared, accident_locked=False, recent_prediction_count=0"
        )

    def resolve_accident_event(self, event_id, action):
        action = str(action or "").strip().lower()
        if action not in ("confirm", "normal"):
            return {"ok": False, "message": "지원하지 않는 action입니다."}

        if action == "confirm":
            event_status = "CONFIRMED"
            operator_action = "사고 확정"
            accident_status = "CONFIRMED"
            accident_flag = True
            event_message = "사고 확정"
        else:
            event_status = "FALSE_ALARM"
            operator_action = "이상 없음"
            accident_status = "FALSE_ALARM"
            accident_flag = False
            event_message = "이상 없음"

        row = self.event_logger.resolve_event(event_id, event_status, operator_action)
        self.resolved_event_ids.add(event_id)

        if self.current_accident_event and self.current_accident_event.get("event_id") == event_id:
            frame_id = int(self.current_accident_event.get("frame_id") or self.latest_status.get("frame_id") or 0)
            traffic_state = self.current_accident_event.get("traffic_state") or self.latest_status.get("traffic_state") or "NORMAL"
            self.current_accident_event = None
        else:
            frame_id = int(self.latest_status.get("frame_id") or 0)
            traffic_state = self.latest_status.get("traffic_state") or "NORMAL"

        if action == "normal":
            self._reset_false_alarm_accident_state(
                frame_id=frame_id,
                traffic_state=traffic_state,
            )

        now_text = datetime.now().strftime("%H:%M:%S")
        event_logs = list(self.latest_status.get("event_logs") or [])
        event_logs.append(f"[{now_text}] {event_message}")
        self._update_status({
            "accident": accident_flag,
            "accident_locked": False if action == "normal" else self.latest_status.get("accident_locked", False),
            "accident_candidate_only": False if action == "normal" else self.latest_status.get("accident_candidate_only", False),
            "frame_accident_prediction": False if action == "normal" else self.latest_status.get("frame_accident_prediction", False),
            "recent_prediction_count": 0 if action == "normal" else self.latest_status.get("recent_prediction_count", 0),
            "weak_suspect": False if action == "normal" else self.latest_status.get("weak_suspect", False),
            "strong_suspect": False if action == "normal" else self.latest_status.get("strong_suspect", False),
            "confirm_candidate": False if action == "normal" else self.latest_status.get("confirm_candidate", False),
            "weak_confirmed": False if action == "normal" else self.latest_status.get("weak_confirmed", False),
            "has_real_accident_evidence": False if action == "normal" else self.latest_status.get("has_real_accident_evidence", False),
            "has_final_accident_evidence": False if action == "normal" else self.latest_status.get("has_final_accident_evidence", False),
            "final_accumulation_blocked": False if action == "normal" else self.latest_status.get("final_accumulation_blocked", False),
            "accident_score": 0 if action == "normal" else self.latest_status.get("accident_score", 0),
            "reasons": "" if action == "normal" else self.latest_status.get("reasons", ""),
            "accident_status": accident_status,
            "pending_accident_event": None,
            "traffic_state": traffic_state,
            "state": "ACCIDENT" if action == "confirm" else traffic_state,
            "event_logs": event_logs[-10:],
            "events": [event_message],
        })

        return {
            "ok": True,
            "event_id": event_id,
            "event_status": event_status,
            "operator_action": operator_action,
            "event": row,
        }

    def get_event_stats(self, date_text=None):
        return self.event_logger.get_stats(date_text)

    # =========================================================
    # 2-3) 저장된 이벤트 목록 조회
    # =========================================================
    def get_saved_event_list(self):
        items = []

        if not os.path.exists(self.event_log_dir):
            return items

        for name in os.listdir(self.event_log_dir):
            if not name.endswith(".json"):
                continue

            path = os.path.join(self.event_log_dir, name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    items.append(data)
            except Exception:
                continue

        items.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return items

    # =========================================================
    # 3) health 관리
    # =========================================================
    def _load_good_cctv_cache(self):
        if not self.good_cctv_cache_path.exists():
            return {}

        try:
            with open(self.good_cctv_cache_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception:
            return {}

        loaded = {}
        items = raw if isinstance(raw, list) else raw.get("items", []) if isinstance(raw, dict) else []
        for item in items:
            cctv = item.get("cctv", item) if isinstance(item, dict) else None
            if not isinstance(cctv, dict):
                continue

            name = str(cctv.get("name", "")).strip()
            url = str(cctv.get("url", "")).strip()
            if not name or not url:
                continue

            loaded[url] = {
                "cctv": {"name": name, "url": url},
                "last_success_at": float(item.get("last_success_at", 0) or 0),
            }

        return loaded

    def _save_good_cctv_cache(self):
        items = sorted(
            self.good_cctv_cache.values(),
            key=lambda item: item.get("last_success_at", 0),
            reverse=True,
        )
        payload = {"items": items[:20]}

        with open(self.good_cctv_cache_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        print("💾 good_cctv_cache 저장 완료")

    def _is_known_cctv_url(self, url):
        return any(cctv.get("url") == url for cctv in self.cached_cctv_list)

    def _is_bad_cached(self, cctv):
        url = cctv.get("url", "")
        expires_at = self.bad_cctv_cache.get(url)

        if not expires_at:
            return False

        if time.time() >= expires_at:
            self.bad_cctv_cache.pop(url, None)
            return False

        return True

    def _remember_bad_cctv(self, cctv):
        url = cctv.get("url", "")
        if not url:
            return

        self.bad_cctv_cache[url] = time.time() + self.bad_cctv_ttl_sec

    def _clear_bad_cctv(self, cctv):
        self.bad_cctv_cache.pop(cctv.get("url", ""), None)

    def _remember_good_cctv(self, cctv):
        url = cctv.get("url", "")
        if not url:
            return

        self.good_cctv_cache[url] = {
            "cctv": cctv,
            "last_success_at": time.time(),
        }
        self._save_good_cctv_cache()

    def _get_good_candidates(self, tried_urls=None):
        tried_urls = tried_urls or set()
        items = []

        for url, entry in list(self.good_cctv_cache.items()):
            if url in tried_urls or not self._is_known_cctv_url(url):
                continue

            cctv = entry.get("cctv")
            if not cctv or self._is_bad_cached(cctv):
                continue

            items.append((entry.get("last_success_at", 0), cctv))

        items.sort(key=lambda item: item[0], reverse=True)
        return [cctv for _, cctv in items]

    def _get_fixed_candidates(self, tried_urls=None, limit=5):
        tried_urls = tried_urls or set()
        candidates = []

        for cctv in self.cached_cctv_list:
            url = cctv["url"]
            if url in tried_urls or self._is_bad_cached(cctv):
                continue

            candidates.append(cctv)
            if len(candidates) >= limit:
                break

        return candidates

    def _mark_cctv_success(self, cctv):
        url = cctv["url"]
        self._clear_bad_cctv(cctv)
        self._remember_good_cctv(cctv)

        if url not in self.cctv_health:
            self.cctv_health[url] = {"ok": True, "fail_count": 0}
            return

        self.cctv_health[url]["ok"] = True
        self.cctv_health[url]["fail_count"] = 0

    def _mark_cctv_failure(self, cctv):
        url = cctv["url"]
        self._remember_bad_cctv(cctv)

        if url not in self.cctv_health:
            self.cctv_health[url] = {"ok": False, "fail_count": 1}
            return

        self.cctv_health[url]["ok"] = False
        self.cctv_health[url]["fail_count"] += 1

    def _get_recent_failure_candidates(self, tried_urls=None):
        tried_urls = tried_urls or set()
        items = []
        now = time.time()

        for cctv in self.cached_cctv_list:
            url = cctv["url"]
            if url in tried_urls:
                continue

            expires_at = self.bad_cctv_cache.get(url)
            if not expires_at:
                continue

            failed_at = expires_at - self.bad_cctv_ttl_sec
            remaining = max(0, expires_at - now)
            items.append((failed_at, remaining, cctv))

        items.sort(key=lambda item: item[0])
        return [cctv for _, _, cctv in items]

    def _get_healthy_candidates(self, tried_urls=None):
        tried_urls = tried_urls or set()
        healthy = []

        for cctv in self.cached_cctv_list:
            url = cctv["url"]
            if url in tried_urls:
                continue

            health = self.cctv_health.get(url, {})
            fail_count = health.get("fail_count", 0)

            if self._is_bad_cached(cctv):
                continue

            if fail_count < 10:  #ITS 스트림이 흔들리는 환경이면 나중에 10 → 15 정도로 완화
                healthy.append(cctv)

        return healthy

    def _get_open_candidates(self, tried_urls=None, prefer_good=True, allow_bad_relax=False):
        tried_urls = tried_urls or set()

        if not self.cached_cctv_list:
            print("❌ 캐시된 CCTV 목록 비어있음")
            return []

        good = self._get_good_candidates(tried_urls) if prefer_good else []
        fixed = self._get_fixed_candidates(tried_urls)
        healthy = self._get_healthy_candidates(tried_urls)

        seen = set()
        candidates = []
        for cctv in good + fixed + healthy:
            url = cctv["url"]
            if url in seen:
                continue
            seen.add(url)
            candidates.append(cctv)

        if candidates:
            if good:
                print("✅ good_cctv_cache 우선 후보 사용")
            elif fixed:
                print("🔄 고정 후보 CCTV 재시도")
            else:
                print("🔍 일반 캐시 후보 탐색")
            return candidates

        bad_candidates = self._get_recent_failure_candidates(tried_urls)
        if bad_candidates:
            print("⚠️ CCTV 후보는 있으나 최근 실패 목록에 모두 포함됨")

            if allow_bad_relax:
                print("🔄 가장 오래된 실패 후보부터 재시도")
                return bad_candidates

        return []

    def _get_user_random_candidates(self, tried_urls=None, exclude_name=None, allow_bad_relax=False):
        tried_urls = tried_urls or set()
        exclude_name = exclude_name or ""

        if not self.cached_cctv_list:
            print("❌ 캐시된 CCTV 목록 비어있음")
            return []

        candidates = [
            cctv for cctv in self.cached_cctv_list
            if cctv.get("url") not in tried_urls
            and cctv.get("name") != exclude_name
            and not self._is_bad_cached(cctv)
        ]

        if candidates:
            random.shuffle(candidates)
            print("🎲 [RANDOM] 현재 CCTV 제외 후 후보 선택")
            return candidates

        bad_candidates = [
            cctv for cctv in self._get_recent_failure_candidates(tried_urls)
            if cctv.get("name") != exclude_name
        ]

        if bad_candidates:
            print("⚠️ CCTV 후보는 있으나 최근 실패 목록에 모두 포함됨")
            if allow_bad_relax:
                print("🔄 가장 오래된 실패 후보부터 재시도")
                return bad_candidates

        return []

    def _open_user_random_stream_with_retry(self, tried_urls=None, max_retry=MAX_OPEN_RETRY):
        if tried_urls is None:
            tried_urls = set()

        ordered = []
        seen = set()

        for cctv in self.user_random_candidates:
            url = cctv.get("url")
            if not url or url in tried_urls or url in seen:
                continue
            if cctv.get("name") == self.user_random_exclude_name:
                continue
            if self._is_bad_cached(cctv):
                continue
            seen.add(url)
            ordered.append(cctv)

        for cctv in self._get_user_random_candidates(
            tried_urls=tried_urls | seen,
            exclude_name=self.user_random_exclude_name,
            allow_bad_relax=False,
        ):
            url = cctv.get("url")
            if url in seen:
                continue
            seen.add(url)
            ordered.append(cctv)

        if not ordered:
            ordered = self._get_user_random_candidates(
                tried_urls=tried_urls,
                exclude_name=self.user_random_exclude_name,
                allow_bad_relax=True,
            )

        retry_count = 0
        for cctv in ordered:
            if retry_count >= max_retry:
                break

            tried_urls.add(cctv["url"])
            cap = self._try_open_cctv(cctv)
            if cap is not None:
                self.current_cctv = cctv
                self.pending_user_random = False
                return cap, cctv

            retry_count += 1

        print("🎲 [RANDOM] 랜덤 후보 연결 실패 → good_cctv_cache fallback 허용")
        cap, cctv = self._open_stream_with_retry(
            tried_urls=tried_urls,
            max_retry=max(0, MAX_OPEN_RETRY - retry_count),
        )
        self.pending_user_random = False
        return cap, cctv

    # =========================================================
    # 3-1) 스트림 상태 제어
    # =========================================================
    def _request_stream_stop(self, wait_timeout=2.0, log_if_idle=True):
        if not self.stream_active and not self.stream_lock.locked():
            if log_if_idle:
                print("ℹ️ 이미 정지 상태 → stop 요청 무시")
            return False

        self.stream_active = False
        self.stream_token += 1

        started_at = time.time()
        while self.stream_lock.locked() and time.time() - started_at < wait_timeout:
            time.sleep(0.05)

        if self.stream_lock.locked() and self.active_capture is not None:
            try:
                self.active_capture.release()
            except Exception:
                pass

        return True

    def _reset_runtime_after_open(self, cctv):
        reset_frame_id = int(self.latest_status.get("frame_id", 0) or 0)
        if self.active_cctv_name == cctv["name"]:
            self.pipeline.clear_accident_state(
                cctv_name=cctv["name"],
                frame_id=reset_frame_id,
            )
            self._clear_runtime_event_state()
            self._redirect_lane_memory_to_runtime()
            return

        self.pipeline.reset_pipeline(
            cctv_name=cctv["name"],
            frame_id=reset_frame_id,
        )
        self.active_cctv_name = cctv["name"]
        self._clear_runtime_event_state()

        self.ventilation_manager.vehicle_entry_memory.clear()
        self.ventilation_manager.current_level = "NORMAL"
        self.ventilation_manager.pending_level = "NORMAL"
        self.ventilation_manager.pending_count = 0
        self.ventilation_manager.release_count = 0

        print(f"♻️ 스트림 시작 시 reset 완료: {self.active_cctv_name}")

    def _clear_runtime_event_state(self):
        """
        새 CCTV 스트림 시작 시 화면/status에 남아 있던 실시간 이벤트 잔상을 지운다.
        저장된 event json/csv 파일은 건드리지 않는다.
        """
        if hasattr(self.pipeline, "event_logs"):
            self.pipeline.event_logs.clear()

        if hasattr(self.pipeline, "event_log_entries"):
            self.pipeline.event_log_entries.clear()
        if hasattr(self.pipeline, "event_log_seq"):
            self.pipeline.event_log_seq = 0

        self.event_log_seq = 0

        if hasattr(self.pipeline, "prev_accident_flag"):
            self.pipeline.prev_accident_flag = False

        self.current_accident_event = None
        self.prev_accident_flag = False

        self._update_status({
            "accident": False,
            "accident_locked": False,
            "accident_candidate_only": False,
            "frame_accident_prediction": False,
            "recent_prediction_count": 0,
            "weak_suspect": False,
            "strong_suspect": False,
            "confirm_candidate": False,
            "weak_confirmed": False,
            "has_real_accident_evidence": False,
            "has_final_accident_evidence": False,
            "final_accumulation_blocked": False,
            "accident_score": 0,
            "reasons": "",
            "accident_status": "NONE",
            "pending_accident_event": None,
            "events": [],
            "event_logs": [],
            "event_log_entries": [],
        })

    def _redirect_lane_memory_to_runtime(self):
        if not hasattr(self.pipeline, "get_lane_template"):
            return

        lane_template = self.pipeline.get_lane_template()
        if lane_template is None:
            return

        runtime_lane_debug = self.runtime_root / "lane_debug"
        runtime_lane_debug.mkdir(parents=True, exist_ok=True)
        self.runtime_lane_memory_dir.mkdir(parents=True, exist_ok=True)

        if hasattr(lane_template, "output_dir"):
            lane_template.output_dir = str(runtime_lane_debug)
        if hasattr(lane_template, "memory_dir"):
            lane_template.memory_dir = str(self.runtime_lane_memory_dir)

    # =========================================================
    # 4) CCTV 선택
    # =========================================================
    def select_random_cctv(self, user_random=False):
        previous_name = (
            self.active_cctv_name
            or (self.current_cctv or {}).get("name")
            or self.latest_status.get("cctv_name")
            or ""
        )

        if self.stream_active or self.stream_lock.locked():
            print("🔄 [SWITCH] 기존 CCTV에서 새 CCTV로 전환")
            self._request_stream_stop(wait_timeout=2.0, log_if_idle=False)

        if user_random:
            print("🎲 [RANDOM] 사용자 랜덤 CCTV 선택 요청")
            candidates = self._get_user_random_candidates(
                exclude_name=previous_name,
                allow_bad_relax=True,
            )
        else:
            candidates = self._get_open_candidates(allow_bad_relax=True)

        if not candidates:
            return None

        self.current_cctv = candidates[0]
        self.pending_user_random = bool(user_random)
        self.user_random_exclude_name = previous_name if user_random else None
        self.user_random_candidates = candidates[1:] if user_random else []
        self.active_cctv_name = None

        self.stream_active = False
        self.stream_token += 1

        self._update_status({
            "state": "READY",
            "traffic_state": "NORMAL",
            "accident_status": "NONE",
            "pending_accident_event": None,
            "avg_speed": 0.0,
            "vehicle_count": 0,
            "accident": False,
            "lane_count": 0,
            "target_lane_count": None,
            "lane_count_stable": False,
            "template_confirmed": False,
            "events": ["CCTV 변경됨 - 분석 초기화 중"],
            "frame_id": 0,
            "cctv_name": self.current_cctv["name"],
            "cctv_url": self.current_cctv["url"],
            "lane_reestimate_status": "idle",
            "lane_reestimate_frame_count": 0,
            "lane_reestimate_window": 50,
            "minute_vehicle_count": 0,
            "ventilation": {
                "risk_score_base": 0.0,
                "risk_score_final": 0.0,
                "risk_level": "NORMAL",
                "alarm": False,
                "message": "공기질 상태 정상",
                "vehicle_count_roi": 0,
                "weighted_vehicle_count": 0.0,
                "traffic_density": 0.0,
                "avg_dwell_time_roi": 0.0,
            }
        })

        if user_random:
            print(f"🎲 [RANDOM] 선택된 CCTV: {self.current_cctv['name']}")
        else:
            print(f"✅ 랜덤 선택 CCTV(캐시): {self.current_cctv['name']}")
        return self.current_cctv

    def select_cctv_by_name(self, name):
        keyword = (name or "").strip()
        if not keyword:
            return None

        def _reset_selected(cctv):
            already_active = (
                (self.stream_active or self.stream_lock.locked())
                and self.active_cctv_name == cctv["name"]
            )

            if self.stream_active or self.stream_lock.locked():
                if already_active:
                    print(f"✅ 이미 실행 중인 CCTV 유지: {cctv['name']}")
                else:
                    print("🔄 [SWITCH] 기존 CCTV에서 새 CCTV로 전환")
                    self._request_stream_stop(wait_timeout=2.0, log_if_idle=False)

            self.current_cctv = cctv

            if already_active:
                return

            if not already_active:
                self.active_cctv_name = None
                self.stream_active = False
                self.stream_token += 1

            self.ventilation_manager.vehicle_entry_memory.clear()
            self.ventilation_manager.current_level = "NORMAL"
            self.ventilation_manager.pending_level = "NORMAL"
            self.ventilation_manager.pending_count = 0
            self.ventilation_manager.release_count = 0

            self._update_status({
                "state": "READY",
                "traffic_state": "NORMAL",
                "accident_status": "NONE",
                "pending_accident_event": None,
                "avg_speed": 0.0,
                "vehicle_count": 0,
                "accident": False,
                "lane_count": 0,
                "target_lane_count": None,
                "lane_count_stable": False,
                "template_confirmed": False,
                "events": ["CCTV 변경됨 - 분석 초기화 중"],
                "frame_id": 0,
                "cctv_name": self.current_cctv["name"],
                "cctv_url": self.current_cctv["url"],
                "lane_reestimate_status": "idle",
                "lane_reestimate_frame_count": 0,
                "lane_reestimate_window": 50,
                "minute_vehicle_count": 0,
                "ventilation": {
                    "risk_score_base": 0.0,
                    "risk_score_final": 0.0,
                    "risk_level": "NORMAL",
                    "alarm": False,
                    "message": "공기질 상태 정상",
                    "vehicle_count_roi": 0,
                    "weighted_vehicle_count": 0.0,
                    "traffic_density": 0.0,
                    "avg_dwell_time_roi": 0.0,
                }
            })

        for cctv in self.cached_cctv_list:
            if keyword == cctv["name"]:
                _reset_selected(cctv)
                print(f"✅ 정확 이름으로 선택 CCTV: {self.current_cctv['name']}")
                return self.current_cctv

        for cctv in self.cached_cctv_list:
            if keyword in cctv["name"]:
                _reset_selected(cctv)
                print(f"✅ 부분 이름으로 선택 CCTV: {self.current_cctv['name']}")
                return self.current_cctv

        normalized_keyword = keyword.replace(" ", "")
        for cctv in self.cached_cctv_list:
            normalized_name = cctv["name"].replace(" ", "")
            if normalized_keyword in normalized_name:
                _reset_selected(cctv)
                print(f"✅ 정규화 이름으로 선택 CCTV: {self.current_cctv['name']}")
                return self.current_cctv

        return None

    # =========================================================
    # 5) CCTV 열기 테스트
    # =========================================================
    def _try_open_cctv(self, cctv):
        name = cctv["name"]
        url = cctv["url"]

        print(f"🎥 [OPEN] CCTV 연결 시도: {name}")
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

    # --------------------------------------------------
    # 1) open 확인: 기존보다 더 길게 대기
    # --------------------------------------------------
        open_ok = False
        for _ in range(20):   # 기존 10 -> 20
            if cap.isOpened():
                open_ok = True
                break
            time.sleep(0.15)

        if not open_ok:
            print(f"❌ [OPEN] CCTV 연결 실패(open): {name}")
            self._mark_cctv_failure(cctv)
            cap.release()
            return None

    # --------------------------------------------------
    # 2) 첫 프레임 확인: 기존보다 더 길게 대기
    # --------------------------------------------------
        frame_ok = False
        for _ in range(40):   # 기존 20 -> 40
            ok, frame = cap.read()
            if ok and frame is not None:
                frame_ok = True
                break
            time.sleep(0.1)

        if not frame_ok:
            print(f"❌ [OPEN] CCTV 연결 실패(first frame): {name}")
            self._mark_cctv_failure(cctv)
            cap.release()
            return None

        print(f"✅ [OPEN] CCTV 연결 성공: {name}")
        self._mark_cctv_success(cctv)
        return cap

    # =========================================================
    # 6) 랜덤 열기
    # =========================================================
    def _open_stream_with_retry(self, tried_urls=None, max_retry=MAX_OPEN_RETRY):
        if tried_urls is None:
            tried_urls = set()

        candidates = self._get_open_candidates(
            tried_urls=tried_urls,
            prefer_good=True,
            allow_bad_relax=True,
        )

        if not candidates:
            return None, None

        retry_count = 0

        for cctv in candidates:
            if retry_count >= max_retry:
                break

            tried_urls.add(cctv["url"])

            cap = self._try_open_cctv(cctv)
            if cap is not None:
                self.current_cctv = cctv
                return cap, cctv

            retry_count += 1

        return None, None

    # =========================================================
    # 7) 스트리밍
    # =========================================================
    def _generate_cached_frames(self):
        empty_started_at = time.time()

        while self.stream_active:
            frame_bytes = self.latest_frame_bytes

            if frame_bytes:
                empty_started_at = time.time()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" +
                    frame_bytes +
                    b"\r\n"
                )
                time.sleep(0.08)
                continue

            if time.time() - empty_started_at > 3.0:
                return

            time.sleep(0.1)

    def generate_frames(self):
        if not self.stream_lock.acquire(blocking=False):
            if self.current_cctv and self.active_cctv_name == self.current_cctv["name"]:
                print(f"✅ 이미 실행 중인 CCTV 유지: {self.current_cctv['name']}")
                yield from self._generate_cached_frames()
                return
            else:
                print("🔄 [SWITCH] 기존 CCTV에서 새 CCTV로 전환")

                self._request_stream_stop(wait_timeout=2.0, log_if_idle=False)

                if not self.stream_lock.acquire(timeout=2.0):
                    self._update_status({
                        "state": "READY",
                        "events": ["기존 스트림 종료 대기 중"],
                    })
                    return

        print("🚦 [START] 요청 수신")
        self.stream_active = True
        tried_urls = set()
        cap = None
        selected_cctv = None
        my_token = self.stream_token

        try:
            if self.current_cctv is not None:
                selected_cctv = self.current_cctv
                if self._is_bad_cached(selected_cctv):
                    print(f"⚠️ 선택된 CCTV 최근 실패 상태 → 우선 사용 제외: {selected_cctv['name']}")
                    tried_urls.add(selected_cctv["url"])
                    self.current_cctv = None
                    selected_cctv = None
                else:
                    print(f"🎯 선택된 CCTV 우선 사용: {selected_cctv['name']}")
                    cap = self._try_open_cctv(selected_cctv)

                    if cap is None:
                        print(f"❌ 선택된 CCTV 열기 실패: {selected_cctv['name']}")
                        tried_urls.add(selected_cctv["url"])
                        self.current_cctv = None
                        cap = None
                        selected_cctv = None
            else:
                selected_cctv = None

            if cap is None or selected_cctv is None:
                if self.pending_user_random:
                    cap, selected_cctv = self._open_user_random_stream_with_retry(
                        tried_urls=tried_urls,
                        max_retry=max(0, MAX_OPEN_RETRY - len(tried_urls)),
                    )
                else:
                    cap, selected_cctv = self._open_stream_with_retry(
                        tried_urls=tried_urls,
                        max_retry=max(0, MAX_OPEN_RETRY - len(tried_urls))
                    )

            if cap is None or selected_cctv is None:
                self.pending_user_random = False
                self._update_status({
                    "state": "ERROR",
                    "events": ["실시간 CCTV 연결 실패"],
                    "cctv_name": "-",
                    "cctv_url": "",
                })
                return

            self.pending_user_random = False

            self.active_capture = cap
            self.latest_frame_bytes = None
            self._reset_runtime_after_open(selected_cctv)

            frame_id = 0
            fail_count = 0

            self._update_status({
                "cctv_name": selected_cctv["name"],
                "cctv_url": selected_cctv["url"],
                "events": [f"{selected_cctv['name']} 연결 완료"],
            })

            while True:
                if my_token != self.stream_token:
                    print("🛑 새 CCTV 선택됨 → 이전 스트림 종료")
                    break

                if not self.stream_active:
                    print("🛑 stream_active=False → 스트림 종료")
                    break

                ok, frame = cap.read()

                if not ok or frame is None:
                    fail_count += 1
                    print(f"❌ 프레임 읽기 실패 ({fail_count})")

                    if fail_count >= 20:
                        print("⚠️ 스트림 재연결 시도")
                        cap.release()
                        cap = None

                        if self.current_cctv is not None:
                            if self._is_bad_cached(self.current_cctv):
                                print(f"⚠️ 현재 CCTV 최근 실패 상태 → 재연결 우선 제외: {self.current_cctv['name']}")
                                tried_urls.add(self.current_cctv["url"])
                                self.current_cctv = None
                            else:
                                print(f"🔁 현재 CCTV 재연결 시도: {self.current_cctv['name']}")
                                cap = self._try_open_cctv(self.current_cctv)

                                if cap is not None:
                                    self.active_capture = cap
                                    selected_cctv = self.current_cctv
                                    fail_count = 0

                                    self._reset_runtime_after_open(selected_cctv)

                                    self._update_status({
                                        "cctv_name": selected_cctv["name"],
                                        "cctv_url": selected_cctv["url"],
                                        "events": [f"{selected_cctv['name']} 재연결 완료"],
                                    })
                                    continue

                                print(f"❌ 현재 CCTV 재연결 실패: {self.current_cctv['name']}")

                        cap, fallback_cctv = self._open_stream_with_retry(
                            tried_urls=tried_urls,
                            max_retry=MAX_OPEN_RETRY,
                        )

                        if cap is not None and fallback_cctv is not None:
                            self.active_capture = cap
                            selected_cctv = fallback_cctv
                            fail_count = 0
                            self._reset_runtime_after_open(selected_cctv)
                            self._update_status({
                                "cctv_name": selected_cctv["name"],
                                "cctv_url": selected_cctv["url"],
                                "events": [f"{selected_cctv['name']} 대체 연결 완료"],
                            })
                            continue

                        self._update_status({
                            "state": "ERROR",
                            "events": ["선택한 CCTV 재연결 실패"],
                            "cctv_name": self.current_cctv["name"] if self.current_cctv else "-",
                            "cctv_url": self.current_cctv["url"] if self.current_cctv else "",
                        })
                        break

                    continue

                fail_count = 0
                frame_id += 1

                self.pipeline.current_cctv_name = selected_cctv["name"]

                annotated, result = self.pipeline.process_frame(frame, frame_id)
                self._redirect_lane_memory_to_runtime()

                if result is None:
                    result = {}

                result["cctv_name"] = selected_cctv["name"]
                result["cctv_url"] = selected_cctv["url"]

                if "lane_reestimate_status" not in result:
                    result["lane_reestimate_status"] = self.latest_status.get("lane_reestimate_status", "idle")

                if "lane_reestimate_frame_count" not in result:
                    result["lane_reestimate_frame_count"] = self.latest_status.get("lane_reestimate_frame_count", 0)

                if "lane_reestimate_window" not in result:
                    result["lane_reestimate_window"] = self.latest_status.get("lane_reestimate_window", 50)

                if "minute_vehicle_count" not in result:
                    result["minute_vehicle_count"] = self.latest_status.get("minute_vehicle_count", 0)

                result["traffic_state"] = self._traffic_state_from_status(result)
                result["accident_status"] = self.latest_status.get("accident_status", "NONE")
                result["accident_confirmed"] = result["accident_status"] == "CONFIRMED"

                result["ventilation"] = build_ventilation_result(
                    result=result,
                    ventilation_manager=self.ventilation_manager
                )

                self._update_status(result)

                self._save_accident_event(annotated, result)

                ok, buffer = cv2.imencode(".jpg", annotated)
                if not ok:
                    continue

                frame_bytes = buffer.tobytes()
                self.latest_frame_bytes = frame_bytes

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" +
                    frame_bytes +
                    b"\r\n"
                )

        except GeneratorExit:
            print("🛑 스트리밍 종료 요청")
        except Exception as e:
            print(f"❌ 스트리밍 중 오류: {e}")
            self._update_status({
                "state": "ERROR",
                "events": [f"stream error: {e}"],
            })
        finally:
            self.stream_active = False

            if cap is not None:
                cap.release()

            self.active_capture = None

            if self.stream_lock.locked():
                self.stream_lock.release()

            print("🛑 [STOP] 스트림 종료")
        
            # =========================================================
    # 8) 스트림 명시 종료
    # =========================================================
    def stop_stream(self):
        stopped = self._request_stream_stop(wait_timeout=2.0)
        if stopped:
            print("🛑 [STOP] 스트림 종료")

        # 현재 스트림만 끊고, 상태는 READY로 갱신
        self._update_status({
            "state": "READY",
            "events": ["스트림 종료"],
            "lane_reestimate_status": "idle",
            "lane_reestimate_frame_count": 0,
            "lane_reestimate_window": 50,
        })

        return {
            "ok": True,
            "message": "스트림 종료 완료" if stopped else "이미 정지 상태"
        }

    # =========================================================
    # 9) 랜덤 CCTV로 새 시작 준비
    # - 기존 스트림 종료
    # - current_cctv 새로 선택
    # - 이후 프론트가 /video-feed 다시 호출하면 새 시작
    # =========================================================
    def restart_with_random_cctv(self):
        # 기존 스트림 완전 종료
        if self.stream_active or self.stream_lock.locked():
            print("🔄 [SWITCH] 기존 CCTV에서 새 CCTV로 전환")
            self._request_stream_stop(wait_timeout=2.0, log_if_idle=False)

        # 이전 선택을 끊고 새 랜덤 선택
        selected = self.select_random_cctv()

        if not selected:
            return {
                "ok": False,
                "message": "새로 선택할 CCTV가 없습니다."
            }

        print(f"🔄 [API] 새 랜덤 CCTV 재시작 준비 완료: {selected['name']}")
        return {
            "ok": True,
            "message": "새 랜덤 CCTV 재시작 준비 완료",
            "cctv": selected
        }
