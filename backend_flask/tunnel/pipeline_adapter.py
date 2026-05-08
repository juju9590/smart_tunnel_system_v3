# ==========================================
# 파일명: pipeline_adapter.py
# 위치: backend_flask/modules/tunnel/pipeline_adapter.py
# 역할:
# - YOLO + ByteTrack 수행
# - tunnel_main_V5_5 시각화 흐름을 웹용으로 반영
# - pipeline_V6_1/PipelineCore 연결
# - 프론트(status API)용 결과 정리
# - CCTV 변경 시 full reset 지원
# - [추가] service.py가 lane_template에 더 쉽게 접근할 수 있도록 getter 제공
# ==========================================

import os
import sys
import cv2
import traceback
import csv
from ultralytics import YOLO
import time
from collections import deque


DEBUG_GET_LANE_TEMPLATE = False


class TunnelPipelineAdapter:
    def __init__(self):
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.module_dir = self.current_dir

        # ------------------------------------------
        # 1) YOLO 모델 로드
        # ------------------------------------------
        self.model_path = os.path.join(self.current_dir, "models", "best.pt")
        self.model = YOLO(self.model_path)

        # ------------------------------------------
        # 2) 내부 파이프라인 경로 연결
        # ------------------------------------------
        pipeline_dir = os.path.join(self.current_dir, "pipeline_V6_1")
        if pipeline_dir not in sys.path:
            sys.path.append(pipeline_dir)

        # ------------------------------------------
        # 3) 실제 파이프라인 import
        # ------------------------------------------
        from pipeline_core_V6 import PipelineCore
        self.PipelineCore = PipelineCore

        self.pipeline = None
        self.frame_height = 720
        self.current_cctv_name = None
        self.current_source_type = "cctv"
        self.current_source_name = "-"
        self.current_demo_video_filename = None

        self.vehicle_seen_log = deque()

        self.last_result = {
            "state": "READY",
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
            "analysis": {},
            "avg_speed_roi": 0.0,
            "vehicles_in_roi": [],

            # [추가] 차선 재추정 상태
            "lane_reestimate_status": "idle",
            "lane_reestimate_frame_count": 0,
            "lane_reestimate_window": 50,
            "smoke_fire_candidate": False,
            "smoke_fire_area_ratio": 0.0,
            "smoke_fire_max_area": 0.0,
            "smoke_fire_bbox": [],
        }

        self.track_first_seen = {}
        self.event_logs = deque(maxlen=10)
        self.event_log_entries = deque(maxlen=10)
        self.event_log_seq = 0
        self.prev_state_text = None
        self.prev_accident_flag = False
        self.prev_roi_fallback = None
        self.prev_template_confirmed = None

        self.live_accident_debug_header = [
            "timestamp",
            "frame_id",
            "cctv_name",
            "vehicle_count",
            "lane_count",
            "target_lane_count",
            "lane_count_stable",
            "template_confirmed",
            "state",
            "avg_speed",
            "accident",
            "weak_suspect",
            "strong_suspect",
            "confirm_candidate",
            "has_real_accident_evidence",
            "accident_score",
            "recent_prediction_count",
            "vehicle_drop",
            "pair_evidence_raw",
            "pair_collision_valid",
            "pair_collision_repeat_count",
            "pair_collision_repeat_lock",
            "congestion_pair_only_false",
            "defense_small_far_pair",
            "defense_pair_only_without_direct_stop",
            "false_stop_by_bbox_split",
            "defense_bbox_split_false_stop",
            "defense_queue_only_without_real_evidence",
            "defense_large_vehicle_occlusion",
            "defense_jam_stationary_cell",
            "defense_bbox_occlusion_false_stop",
            "reasons",
        ]
        self.live_accident_debug_file = None
        self.live_accident_debug_writer = None
        self.live_accident_debug_header_written = False
        self.live_accident_debug_path = os.path.join(
            self.module_dir,
            "runtime_data",
            "csv_logs",
            f"live_accident_debug_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        )

        # =========================================================
        # [추가] 운영화면용 얇은 오버레이 스타일
        # =========================================================
        self.ROI_COLOR = (120, 140, 160)      # ROI 선: 중간 청회색
        self.LANE_COLOR = (190, 210, 230)     # 차선 중심선: 밝은 청회색
        self.LABEL_COLOR = (220, 235, 245)    # 차선 라벨: 거의 흰색에 가까운 밝은색
        self.BOX_COLOR = (0, 255, 0)          # bbox는 초록 유지

        self.ROI_THICKNESS = 1
        self.LANE_THICKNESS = 1
        self.BOX_THICKNESS = 1

        self.FONT = cv2.FONT_HERSHEY_SIMPLEX
        self.FONT_SCALE_SMALL = 0.40
        self.FONT_THICKNESS = 1

    # =========================================================
    # 1) CCTV 변경 시 full reset
    # =========================================================
    def _reset_accident_state(self, cctv_name=None, frame_id=None):
        reset_cctv_name = cctv_name or self.current_cctv_name or "-"
        reset_frame_id = int(frame_id if frame_id is not None else self.last_result.get("frame_id", 0) or 0)
        histories_cleared = True

        accident_model = getattr(self.pipeline, "accident_model", None) if self.pipeline is not None else None
        if accident_model is not None and hasattr(accident_model, "clear_accident"):
            histories_cleared = bool(accident_model.clear_accident())

        self.prev_accident_flag = False

        if isinstance(self.last_result, dict):
            self.last_result.update({
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
                "accident_score": 0,
                "reasons": "",
            })

        print(
            f"[ACCIDENT RESET] cctv_name={reset_cctv_name}, "
            f"frame_id={reset_frame_id}, "
            f"histories_cleared={histories_cleared}, "
            "accident_locked=False"
        )

        return histories_cleared

    def reset_pipeline(self, cctv_name=None, frame_id=None):
        print("♻️ TunnelPipelineAdapter full reset")
        self._reset_accident_state(cctv_name=cctv_name, frame_id=frame_id)

        # YOLO tracker 상태까지 완전히 초기화
        self.model = YOLO(self.model_path)

        # 내부 파이프라인도 완전 초기화
        self.pipeline = None
        self.frame_height = 720
        self.current_cctv_name = None

        self.last_result = {
            "state": "READY",
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
            "frame_accident_prediction": False,
            "recent_prediction_count": 0,
            "weak_suspect": False,
            "strong_suspect": False,
            "confirm_candidate": False,
            "has_real_accident_evidence": False,
            "accident_score": 0,
            "reasons": "",
            "analysis": {},
            "lane_reestimate_status": "idle",
            "lane_reestimate_frame_count": 0,
            "lane_reestimate_window": 50,
            "avg_speed_roi": 0.0,
            "vehicles_in_roi": [],
        }

        self.track_first_seen = {}
        self.event_logs = deque(maxlen=10)
        self.event_log_entries = deque(maxlen=10)
        self.event_log_seq = 0
        self.prev_state_text = None
        self.prev_accident_flag = False
        self.prev_roi_fallback = None
        self.prev_template_confirmed = None

    def clear_accident_state(self, cctv_name=None, frame_id=None):
        return self._reset_accident_state(cctv_name=cctv_name, frame_id=frame_id)

    # =========================================================
    # 2) 파이프라인 lazy init
    # =========================================================
    def _ensure_pipeline(self, frame):
        if self.pipeline is None:
            self.frame_height = frame.shape[0]

            lane_output_dir = os.path.join(self.current_dir, "outputs", "lane_debug")
            os.makedirs(lane_output_dir, exist_ok=True)

            self.pipeline = self.PipelineCore(
                frame_height=self.frame_height,
                lane_output_dir=lane_output_dir
            )

    # =========================================================
    # 2-1) [추가] service.py에서 lane_template 쉽게 접근하도록 getter 제공
    # =========================================================
    def get_lane_template(self):
        """
        service.py에서 self.pipeline.get_lane_template()로 바로 접근하기 위한 함수

        현재 구조:
        TunnelLiveService
          └─ self.pipeline (TunnelPipelineAdapter)
               └─ self.pipeline (PipelineCore)
                    └─ lane_template

        이 메서드를 두면 service.py에서 내부 구조를 덜 의존하게 된다.
        """
        if self.pipeline is None:
            if DEBUG_GET_LANE_TEMPLATE:
                print("❌ get_lane_template: self.pipeline is None")
            return None

        if DEBUG_GET_LANE_TEMPLATE:
            print("🧪 get_lane_template dir(self.pipeline):", dir(self.pipeline))

        if hasattr(self.pipeline, "lane_template"):
            if DEBUG_GET_LANE_TEMPLATE:
                print("✅ get_lane_template: self.pipeline.lane_template 찾음")
            return getattr(self.pipeline, "lane_template")
        
        if hasattr(self.pipeline, "lane_estimator"):
            if DEBUG_GET_LANE_TEMPLATE:
                print("✅ get_lane_template: self.pipeline.lane_estimator 찾음")
            return getattr(self.pipeline, "lane_estimator")

        if DEBUG_GET_LANE_TEMPLATE:
            print("❌ get_lane_template: lane template 관련 속성 못 찾음")
        return None

    # =========================================================
    # 3) YOLO + ByteTrack
    # =========================================================
    def _run_yolo_track(self, frame):
        yolo_results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            conf=0.25,
            verbose=False
        )

        tracks = []

        if yolo_results and len(yolo_results) > 0:
            r = yolo_results[0]

            if r.boxes is not None and r.boxes.id is not None:
                boxes_xyxy = r.boxes.xyxy.cpu().numpy()
                ids = r.boxes.id.cpu().numpy().astype(int)

                for box, tid in zip(boxes_xyxy, ids):
                    x1, y1, x2, y2 = box.tolist()
                    tracks.append({
                        "id": int(tid),
                        "bbox": (int(x1), int(y1), int(x2), int(y2))
                    })

        return tracks

    # =========================================================
    # 4) 시각화 함수들
    # =========================================================
    def _draw_centerlines(self, frame, centerlines, lane_y1, lane_y2):
        """
        차선 중심선을 얇은 회색으로 그린다.
        라벨은 크게 'LANE 0' 대신 시작지점 근처에 'L0', 'L1'처럼 작게 표시한다.
        """
        for lane in centerlines:
            lane_id = lane.get("lane_id", -1)
            model = lane.get("rep_model", {})

            if not model:
                continue

            pts = []

            for y in range(int(lane_y1), int(lane_y2), 20):
                if model.get("type") == "linear":
                    coef = model.get("coef", [0, 0])
                    if len(coef) < 2:
                        continue
                    a, b = coef[:2]
                    x = a * y + b
                else:
                    coef = model.get("coef", [0, 0, 0])
                    if len(coef) < 3:
                        continue
                    a, b, c = coef[:3]
                    x = a * (y ** 2) + b * y + c

                pts.append((int(x), int(y)))

            # 얇은 회색 선
            for i in range(1, len(pts)):
                cv2.line(
                    frame,
                    pts[i - 1],
                    pts[i],
                    self.LANE_COLOR,
                    self.LANE_THICKNESS
                )

            # 시작지점 근처에만 작게 L0, L1 표시
            if len(pts) > 0:
                x0, y0 = pts[0]
                cv2.putText(
                    frame,
                    f"L{lane_id}",
                    (x0 + 4, max(15, y0 - 4)),
                    self.FONT,
                    self.FONT_SCALE_SMALL,
                    self.LABEL_COLOR,
                    self.FONT_THICKNESS,
                    cv2.LINE_AA
                )

    def _draw_tracks(self, frame, tracks, merged_analysis):
        """
        차량 bbox는 초록색 유지하되 얇게,
        텍스트는 작고 간결하게:
        ID:12 V:2.8 L1
        """
        boxes = merged_analysis.get("boxes", {})
        speeds = merged_analysis.get("speeds", {})
        lane_map = merged_analysis.get("lane_map", {})

        for t in tracks:
            tid = t["id"]

            bbox = boxes.get(tid, t.get("bbox"))
            if not bbox:
                continue

            x1, y1, x2, y2 = bbox
            speed = float(speeds.get(tid, 0.0))
            lane = lane_map.get(tid, None)

            # bbox 얇게
            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                self.BOX_COLOR,
                self.BOX_THICKNESS
            )

            lane_text = f"L{lane}" if lane is not None else "L-"
            text = f"ID:{tid} V:{speed:.1f} {lane_text}"

            cv2.putText(
                frame,
                text,
                (int(x1), max(15, int(y1) - 5)),
                self.FONT,
                self.FONT_SCALE_SMALL,
                self.BOX_COLOR,
                self.FONT_THICKNESS,
                cv2.LINE_AA
            )

    def _draw_roi_lines(self, frame, merged_analysis):
        """
        ROI는 얇은 회색 선만 표시하고,
        텍스트(ROI 수치)는 표시하지 않는다.
        """
        h, w = frame.shape[:2]

        roi_y1 = merged_analysis.get("roi_y1")
        roi_y2 = merged_analysis.get("roi_y2")

        if roi_y1 is None:
            roi_y1 = merged_analysis.get("roi_raw_y1", int(h * 0.2))
        if roi_y2 is None:
            roi_y2 = merged_analysis.get("roi_raw_y2", int(h * 0.8))

        roi_y1 = int(roi_y1)
        roi_y2 = int(roi_y2)

        cv2.line(frame, (0, roi_y1), (w, roi_y1), self.ROI_COLOR, self.ROI_THICKNESS)
        cv2.line(frame, (0, roi_y2), (w, roi_y2), self.ROI_COLOR, self.ROI_THICKNESS)

    def _draw_summary(self, frame, result, frame_id):
        merged = result.get("analysis", {})
        state_result = result.get("state", {})
        accident_result = result.get("accident", {})

        display_frame_id = frame_id
        vehicle_count = merged.get("vehicle_count", 0)

        if isinstance(state_result, dict):
            state_debug = state_result.get("debug", {})
            state_text = state_result.get("state", "UNKNOWN")
        else:
            state_debug = {}
            state_text = str(state_result)

        avg_speed = state_debug.get("buffer_avg_speed", 0.0)

        if isinstance(accident_result, dict):
            accident_flag = accident_result.get("accident", False)
            accident_text = "True" if accident_flag else "False"
        else:
            accident_text = str(accident_result)

        if state_text == "NORMAL":
            state_color = (0, 255, 0)
        elif state_text == "CONGESTION":
            state_color = (0, 165, 255)
        elif state_text == "JAM":
            state_color = (0, 0, 255)
        else:
            state_color = (255, 255, 255)

        accident_color = (0, 0, 255) if accident_text == "True" else (255, 255, 255)

        panel_x = 12
        panel_y = 12
        panel_w = 300
        row_h = 34
        header_h = 38
        panel_h = header_h + row_h * 5 + 12

        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            (30, 30, 30),
            -1
        )
        alpha = 0.65
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        cv2.rectangle(
            frame,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            (180, 180, 180),
            1
        )

        cv2.putText(
            frame,
            "SMART TUNNEL",
            (panel_x + 12, panel_y + 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2
        )

        cv2.line(
            frame,
            (panel_x + 8, panel_y + header_h),
            (panel_x + panel_w - 8, panel_y + header_h),
            (120, 120, 120),
            1
        )

        rows = [
            ("Frame ID", str(display_frame_id), (255, 255, 255)),
            ("State", state_text, state_color),
            ("Vehicles", str(vehicle_count), (255, 255, 255)),
            ("Avg Speed", f"{float(avg_speed):.2f}", (255, 255, 255)),
            ("Accident", accident_text, accident_color),
        ]

        label_x = panel_x + 14
        value_x = panel_x + 155
        start_y = panel_y + header_h + 24

        for i, (label, value, value_color) in enumerate(rows):
            y = start_y + i * row_h

            if i > 0:
                line_y = y - 18
                cv2.line(
                    frame,
                    (panel_x + 10, line_y),
                    (panel_x + panel_w - 10, line_y),
                    (70, 70, 70),
                    1
                )

            cv2.putText(
                frame,
                label,
                (label_x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (210, 210, 210),
                2
            )

            cv2.putText(
                frame,
                value,
                (value_x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                value_color,
                2
            )

    def _append_event_log(self, message):
        if not message:
            return

        timestamp = time.strftime("%H:%M:%S")
        log_text = f"[{timestamp}] {message}"

        if len(self.event_logs) == 0 or self.event_logs[-1] != log_text:
            self.event_log_seq += 1
            self.event_logs.append(log_text)
            self.event_log_entries.append({
                "event_id": self.event_log_seq,
                "timestamp": timestamp,
                "message": str(message),
                "text": log_text,
            })

    def _ensure_live_accident_debug_writer(self):
        if self.live_accident_debug_writer is not None:
            return

        log_dir = os.path.dirname(self.live_accident_debug_path)
        os.makedirs(log_dir, exist_ok=True)

        self.live_accident_debug_file = open(
            self.live_accident_debug_path,
            "a",
            newline="",
            encoding="utf-8-sig"
        )
        self.live_accident_debug_writer = csv.writer(self.live_accident_debug_file)

        if not self.live_accident_debug_header_written:
            self.live_accident_debug_writer.writerow(self.live_accident_debug_header)
            self.live_accident_debug_header_written = True
            self.live_accident_debug_file.flush()

    def _write_live_accident_debug(self, front_status):
        try:
            self._ensure_live_accident_debug_writer()

            row = [
                time.strftime("%Y-%m-%d %H:%M:%S"),
                int(front_status.get("frame_id", 0)),
                str(front_status.get("cctv_name", self.current_cctv_name or "")),
                int(front_status.get("vehicle_count", 0)),
                int(front_status.get("lane_count", 0)),
                front_status.get("target_lane_count", ""),
                bool(front_status.get("lane_count_stable", False)),
                bool(front_status.get("template_confirmed", False)),
                str(front_status.get("state", "")),
                float(front_status.get("avg_speed", 0.0)),
                bool(front_status.get("accident", False)),
                bool(front_status.get("weak_suspect", False)),
                bool(front_status.get("strong_suspect", False)),
                bool(front_status.get("confirm_candidate", False)),
                bool(front_status.get("has_real_accident_evidence", False)),
                front_status.get("accident_score", 0),
                int(front_status.get("recent_prediction_count", 0)),
                int(front_status.get("vehicle_drop", 0)),
                bool(front_status.get("pair_evidence_raw", False)),
                bool(front_status.get("pair_collision_valid", False)),
                int(front_status.get("pair_collision_repeat_count", 0)),
                bool(front_status.get("pair_collision_repeat_lock", False)),
                bool(front_status.get("congestion_pair_only_false", False)),
                bool(front_status.get("defense_small_far_pair", False)),
                bool(front_status.get("defense_pair_only_without_direct_stop", False)),
                bool(front_status.get("false_stop_by_bbox_split", False)),
                bool(front_status.get("defense_bbox_split_false_stop", False)),
                bool(front_status.get("defense_queue_only_without_real_evidence", False)),
                bool(front_status.get("defense_large_vehicle_occlusion", False)),
                bool(front_status.get("defense_jam_stationary_cell", False)),
                bool(front_status.get("defense_bbox_occlusion_false_stop", False)),
                str(front_status.get("reasons", "")),
            ]

            self.live_accident_debug_writer.writerow(row)
            self.live_accident_debug_file.flush()

        except Exception as e:
            print(f"⚠️ live accident debug CSV 저장 실패: {e}")

    # =========================================================
    # 5) 프론트(status API)용 값 정리
    # =========================================================
    def _build_front_status(self, result, frame_id, tracks):
        merged = result.get("analysis", {})
        state_result = result.get("state", {})
        accident_result = result.get("accident", {})

        vehicle_count = merged.get("vehicle_count", 0)

        if isinstance(state_result, dict):
            state_text = state_result.get("state", "UNKNOWN")
            state_debug = state_result.get("debug", {})
            avg_speed = float(state_debug.get("buffer_avg_speed", 0.0))
        else:
            state_text = str(state_result)
            state_debug = {}
            avg_speed = 0.0

        if isinstance(accident_result, dict):
            accident_flag = bool(accident_result.get("accident", False))
            accident_locked = bool(accident_result.get("accident_locked", False))
            accident_debug = accident_result
        elif isinstance(accident_result, bool):
            accident_flag = accident_result
            accident_locked = accident_flag
            accident_debug = {}
        else:
            accident_flag = False
            accident_locked = False
            accident_debug = {}

        accident_reasons = accident_debug.get("reasons", [])
        if isinstance(accident_reasons, list):
            accident_reasons_text = "|".join(str(reason) for reason in accident_reasons)
        else:
            accident_reasons_text = str(accident_reasons) if accident_reasons is not None else ""

        defense_queue_only_without_real_evidence = bool(
            accident_debug.get(
                "defense_queue_only_without_real_evidence",
                accident_debug.get("queue_only_without_real_evidence", False)
            )
        )
        final_accident_event = bool(
            (accident_flag or accident_locked)
            and not defense_queue_only_without_real_evidence
        )

        lane_count = merged.get("lane_count", 0)
        target_lane_count_raw = merged.get("target_lane_count", None)
        target_lane_count = int(target_lane_count_raw) if target_lane_count_raw is not None else None

        speeds = merged.get("speeds", {})
        lane_map = merged.get("lane_map", {})
        raw_lane_map = merged.get("raw_lane_map", {})
        boxes = merged.get("boxes", {})

    # -----------------------------
    # [추가] 차선 재추정 상태값 읽기
    # -----------------------------
        lane_reestimate_status = merged.get("lane_reestimate_status", "idle")
        lane_reestimate_frame_count = int(merged.get("lane_reestimate_frame_count", 0))
        lane_reestimate_window = int(merged.get("lane_reestimate_window", 50))

    # -----------------------------
    # ROI 범위
    # -----------------------------
        roi_y1 = int(merged.get("roi_y1", merged.get("roi_raw_y1", self.frame_height * 0.20)))
        roi_y2 = int(merged.get("roi_y2", merged.get("roi_raw_y2", self.frame_height * 0.95)))

    # -----------------------------
    # 1) vehicles + dwell_times + vehicles_in_roi 생성
    # -----------------------------
        now_ts = time.time()
        current_ids = set()
        vehicles = []
        vehicles_in_roi = []
        dwell_times = {}

        roi_speed_sum = 0.0
        roi_speed_count = 0

        for t in tracks:
            tid = int(t["id"])
            current_ids.add(tid)

            if tid not in self.track_first_seen:
                self.track_first_seen[tid] = now_ts

            dwell_sec = round(now_ts - self.track_first_seen[tid], 2)
            dwell_times[str(tid)] = dwell_sec

            bbox = boxes.get(tid, t.get("bbox"))
            speed = float(speeds.get(tid, 0.0))
            lane = lane_map.get(tid, None)
            raw_lane = raw_lane_map.get(tid, None)

            if not bbox:
                continue

            x1, y1, x2, y2 = bbox
            bbox_w = max(int(x2) - int(x1), 1)
            bbox_h = max(int(y2) - int(y1), 1)
            bbox_area = bbox_w * bbox_h

            bottom_x = int((x1 + x2) / 2)
            bottom_y = int(y2)

            roi_in = (roi_y1 <= bottom_y <= roi_y2)

            vehicle_item = {
                "id": tid,
                "track_id": tid,          # [추가] ventilation용
                "speed": round(speed, 2),
                "ema_speed": round(speed, 2),   # [추가] ventilation용
                "lane": lane,
                "raw_lane": raw_lane,
                "dwell_time": dwell_sec,
                "bbox": bbox,
                "bbox_size": bbox_area,   # [추가] ventilation용
                "roi_in": roi_in,         # [추가] ventilation용
                "bottom_point": (bottom_x, bottom_y),
            }

            vehicles.append(vehicle_item)

            if roi_in:
                vehicles_in_roi.append({
                    "track_id": tid,
                    "bbox": bbox,
                    "bbox_size": bbox_area,
                    "ema_speed": round(speed, 2),
                    "roi_in": True,
                })
                roi_speed_sum += speed
                roi_speed_count += 1

            # 최근 1분 누적 차량수 계산용 로그에 기록
            self.vehicle_seen_log.append((now_ts, tid))

        stale_ids = [tid for tid in list(self.track_first_seen.keys()) if tid not in current_ids]
        for tid in stale_ids:
            del self.track_first_seen[tid]

    # --------------------------------------------------
    # 최근 60초 이내 로그만 유지
    # --------------------------------------------------
        while self.vehicle_seen_log and now_ts - self.vehicle_seen_log[0][0] > 60:
            self.vehicle_seen_log.popleft()

        # 최근 60초 동안 등장한 고유 차량 수
        minute_vehicle_count = len(set(tid for _, tid in self.vehicle_seen_log))

    # -----------------------------
    # [추가] ROI 기준 평균속도
    # -----------------------------
        avg_speed_roi = round(roi_speed_sum / roi_speed_count, 2) if roi_speed_count > 0 else 0.0

    # -----------------------------
    # 2) 이벤트 로그 생성
    # -----------------------------
        roi_fallback = bool(merged.get("roi_used_fallback", False))
        template_confirmed = bool(merged.get("template_confirmed", False))
        lane_count_stable = (
            target_lane_count is not None
            and int(lane_count) == int(target_lane_count)
            and template_confirmed is True
        )

        if self.prev_state_text is None:
            self.prev_state_text = state_text
        elif self.prev_state_text != state_text:
            self._append_event_log(f"상태 변경: {self.prev_state_text} → {state_text}")
            self.prev_state_text = state_text

        self.prev_accident_flag = final_accident_event

        if self.prev_roi_fallback is None:
            self.prev_roi_fallback = roi_fallback
        elif self.prev_roi_fallback != roi_fallback:
            if roi_fallback:
                self._append_event_log("ROI fallback 사용")
            else:
                self._append_event_log("ROI 자동설정 정상 복귀")
            self.prev_roi_fallback = roi_fallback

        if self.prev_template_confirmed is None:
            self.prev_template_confirmed = template_confirmed
        elif self.prev_template_confirmed != template_confirmed:
            if template_confirmed:
                self._append_event_log("차선 bootstrap 완료")
            else:
                self._append_event_log("차선 bootstrap 중")
            self.prev_template_confirmed = template_confirmed

    # -----------------------------
    # 3) 기존 events 유지
    # -----------------------------
        events = []
        if roi_fallback:
            events.append("ROI fallback 사용")
        if final_accident_event:
            events.append("사고 감지")
        if not template_confirmed:
            events.append("차선 bootstrap 중")

        if lane_reestimate_status == "reestimating":
            events.append(f"차선 재추정 중 ({lane_reestimate_frame_count}/{lane_reestimate_window})")
        elif lane_reestimate_status == "reestimated":
            events.append("차선 재추정 완료")

        event_logs = list(self.event_logs)
        event_log_entries = list(self.event_log_entries)
        if not final_accident_event:
            event_logs = [
                event for event in event_logs
                if "사고 감지" not in str(event)
            ]
            event_log_entries = [
                event for event in event_log_entries
                if "사고 감지" not in str(event.get("text", event.get("message", "")))
            ]

        confirm_candidate = bool(accident_debug.get("confirm_candidate", False))
        has_final_accident_evidence = bool(accident_debug.get("has_final_accident_evidence", False))
        has_real_accident_evidence = bool(accident_debug.get("has_real_accident_evidence", False))
        recent_prediction_count = int(accident_debug.get("recent_prediction_count", 0) or 0)

        status_accident_safety_corrected = False
        safe_accident_flag = bool(accident_flag)
        if (
            safe_accident_flag
            and not accident_locked
            and (
                defense_queue_only_without_real_evidence
                or (
                    not has_real_accident_evidence
                    and not confirm_candidate
                    and recent_prediction_count == 0
                )
                or not (
                    has_final_accident_evidence
                    or confirm_candidate
                )
            )
        ):
            safe_accident_flag = False
            status_accident_safety_corrected = True

        return {
            "state": state_text,
            "avg_speed": avg_speed,
            "avg_speed_roi": avg_speed_roi,          # [추가]
            "vehicle_count": vehicle_count,
            "accident": safe_accident_flag,
            "accident_locked": accident_locked,
            "lane_count": lane_count,
            "target_lane_count": target_lane_count,
            "lane_count_stable": bool(lane_count_stable),
            "template_confirmed": bool(template_confirmed),
            "events": events,
            "event_logs": event_logs,
            "event_log_entries": event_log_entries,
            "vehicles": vehicles,
            "vehicles_in_roi": vehicles_in_roi,      # [추가]
            "dwell_times": dwell_times,
            "frame_id": frame_id,
            "analysis": merged,

            "minute_vehicle_count": minute_vehicle_count,

            "lane_reestimate_status": lane_reestimate_status,
            "lane_reestimate_frame_count": lane_reestimate_frame_count,
            "lane_reestimate_window": lane_reestimate_window,
            "smoke_fire_candidate": bool(merged.get("smoke_fire_candidate", False)),
            "smoke_fire_area_ratio": float(merged.get("smoke_fire_area_ratio", 0.0) or 0.0),
            "smoke_fire_max_area": float(merged.get("smoke_fire_max_area", 0.0) or 0.0),
            "smoke_fire_bbox": merged.get("smoke_fire_bbox", []),
            "smoke_fire_map": merged.get("smoke_fire_map", {}),

            # V6 사고탐지 디버그 값.
            # 실시간 CCTV 평가용 CSV logger가 status payload만 보고도
            # 사고 후보/방어 조건을 기록할 수 있도록 그대로 노출한다.
            "weak_suspect": bool(accident_debug.get("weak_suspect", False)),
            "strong_suspect": bool(accident_debug.get("strong_suspect", False)),
            "confirm_candidate": bool(confirm_candidate),
            "has_real_accident_evidence": bool(has_real_accident_evidence),
            "has_final_accident_evidence": bool(has_final_accident_evidence),
            "status_accident_safety_corrected": bool(status_accident_safety_corrected),
            "accident_score": accident_debug.get("accident_score", 0),
            "recent_prediction_count": recent_prediction_count,
            "vehicle_drop": accident_debug.get("vehicle_drop", 0),
            "defense_large_vehicle_occlusion": bool(accident_debug.get("defense_large_vehicle_occlusion", False)),
            "defense_jam_stationary_cell": bool(accident_debug.get("defense_jam_stationary_cell", False)),
            "defense_bbox_occlusion_false_stop": (
                "defense_bbox_occlusion_false_stop" in accident_reasons
                if isinstance(accident_reasons, list)
                else "defense_bbox_occlusion_false_stop" in accident_reasons_text
            ),
            "defense_queue_only_without_real_evidence": bool(defense_queue_only_without_real_evidence),
            "defense_small_far_pair": bool(accident_debug.get("defense_small_far_pair", False)),
            "defense_pair_only_without_direct_stop": bool(
                accident_debug.get("defense_pair_only_without_direct_stop", False)
            ),
            "false_stop_by_bbox_split": bool(accident_debug.get("false_stop_by_bbox_split", False)),
            "defense_bbox_split_false_stop": (
                "defense_bbox_split_false_stop" in accident_reasons
                if isinstance(accident_reasons, list)
                else "defense_bbox_split_false_stop" in accident_reasons_text
            ),
            "pair_evidence_raw": bool(accident_debug.get("pair_evidence_raw", False)),
            "pair_collision_valid": bool(accident_debug.get("pair_collision_valid", False)),
            "pair_collision_repeat_count": int(
                accident_debug.get(
                    "pair_collision_repeat_count",
                    accident_debug.get("pair_collision_repeat_count_50f", 0)
                )
            ),
            "pair_collision_repeat_count_window": int(
                accident_debug.get(
                    "pair_collision_repeat_count_window",
                    accident_debug.get("pair_collision_repeat_count", 0)
                )
            ),
            "pair_collision_repeat_lock": bool(accident_debug.get("pair_collision_repeat_lock", False)),
            "congestion_pair_only_false": bool(accident_debug.get("congestion_pair_only_false", False)),
            "reasons": accident_reasons_text,
        }

    # =========================================================
    # 6) 메인 처리
    # =========================================================
    def process_frame(self, frame, frame_id):
        annotated = frame.copy()

        try:
            if self.pipeline is None:
                self._ensure_pipeline(frame)

            tracks = self._run_yolo_track(frame)

            if self.pipeline is None:
                self._ensure_pipeline(frame)

            if self.pipeline is None:
                ready_result = {
                    "state": "READY",
                    "avg_speed": 0.0,
                    "vehicle_count": 0,
                    "accident": False,
                    "accident_locked": False,
                    "lane_count": 0,
                    "target_lane_count": None,
                    "lane_count_stable": False,
                    "template_confirmed": False,
                    "events": ["pipeline initializing"],
                    "event_logs": [],
                    "event_log_entries": [],
                    "frame_id": frame_id,
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
                    "analysis": {},
                    "lane_reestimate_status": "idle",
                    "lane_reestimate_frame_count": 0,
                    "lane_reestimate_window": 50,
                    "avg_speed_roi": 0.0,
                    "vehicles_in_roi": [],
                    "smoke_fire_candidate": False,
                    "smoke_fire_area_ratio": 0.0,
                    "smoke_fire_max_area": 0.0,
                    "smoke_fire_bbox": [],
                }
                self.last_result = ready_result
                return annotated, ready_result

            result = self.pipeline.process(
                frame_id,
                tracks,
                frame.shape[1],
                cctv_name=self.current_cctv_name,
                frame=frame,
                source_type=getattr(self, "current_source_type", "cctv"),
                source_name=getattr(self, "current_source_name", self.current_cctv_name),
                demo_video_filename=getattr(self, "current_demo_video_filename", None),
            )
            merged = result.get("analysis", {})

            lane_y1 = int(
                merged.get("roi_y1", merged.get("roi_raw_y1", frame.shape[0] * 0.20))
            )
            lane_y2 = int(
                merged.get("roi_y2", merged.get("roi_raw_y2", frame.shape[0] * 0.95))
            )

            if merged.get("roi_sample_count", 0) > 0 or merged.get("roi_fixed", False):
                self._draw_roi_lines(annotated, merged)

            if merged.get("template_confirmed", False):
                self._draw_centerlines(
                    annotated,
                    merged.get("centerlines", []),
                    lane_y1,
                    lane_y2
                )

            self._draw_tracks(annotated, tracks, merged)

            front_status = self._build_front_status(result, frame_id, tracks)
            self._write_live_accident_debug(front_status)
            self.last_result = front_status

            return annotated, front_status

        except Exception as e:
            print(f"❌ pipeline_adapter 처리 실패: {e}")
            traceback.print_exc()

            previous = self.last_result if isinstance(self.last_result, dict) else {}
            error_result = {
                "state": previous.get("state", "READY"),
                "avg_speed": previous.get("avg_speed", 0.0),
                "vehicle_count": previous.get("vehicle_count", 0),
                "accident": False,
                "accident_locked": False,
                "lane_count": previous.get("lane_count", 0),
                "target_lane_count": previous.get("target_lane_count", None),
                "lane_count_stable": previous.get("lane_count_stable", False),
                "template_confirmed": previous.get("template_confirmed", False),
                "events": [f"pipeline error: {e}"],
                "event_logs": list(self.event_logs),
                "event_log_entries": list(self.event_log_entries),
                "frame_id": frame_id,
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
                "analysis": previous.get("analysis", {}),
                "lane_reestimate_status": previous.get("lane_reestimate_status", "idle"),
                "lane_reestimate_frame_count": previous.get("lane_reestimate_frame_count", 0),
                "lane_reestimate_window": previous.get("lane_reestimate_window", 50),
                "avg_speed_roi": previous.get("avg_speed_roi", 0.0),
                "vehicles": previous.get("vehicles", []),
                "vehicles_in_roi": previous.get("vehicles_in_roi", []),
                "dwell_times": previous.get("dwell_times", {}),
                "smoke_fire_candidate": previous.get("smoke_fire_candidate", False),
                "smoke_fire_area_ratio": previous.get("smoke_fire_area_ratio", 0.0),
                "smoke_fire_max_area": previous.get("smoke_fire_max_area", 0.0),
                "smoke_fire_bbox": previous.get("smoke_fire_bbox", []),
            }
            self.last_result = error_result

            return annotated, error_result
